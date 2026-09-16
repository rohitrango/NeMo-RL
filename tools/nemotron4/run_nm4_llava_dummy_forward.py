#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Load Pixtral dummy images, patchify them, and run NM4LlavaModel.

The processor emits CHW ``pixel_values``; this script uses the same
``PackedTensor(patchify)`` path as NeMo-RL, then calls the Bridge provider
model. Context parallelism requires packed THD, so sequences are packed and
aligned to ``2 * CP * TP`` when sequence parallel is on.

World size must be ``TP * PP * CP * DP``. ``EP * ETP`` must divide ``DP * TP``.
PP must be 1 for this smoke test (vision lives on the first pipeline stage).

Single-node launches use ``uv run torchrun --standalone``. World size is
``--nproc-per-node``. TP=4, CP=4, EP=4 needs 16 GPUs on that node. An 8-GPU
node can run TP=4, CP=2, EP=4 instead.

    # 16-GPU node: CP=4, 1 image, seq_len=4096
    uv run torchrun --standalone --nproc-per-node=16 \\
        scripts/run_nm4_llava_dummy_forward.py \\
        --checkpoint /path/to/iter_0030000 \\
        --processor nt4_processor \\
        --precision-recipe /path/to/precision_recipe.yaml \\
        --tp 4 --cp 4 --ep 4 --seq-len 4096 --num-images 1

    # 16-GPU node: CP=4, 3 images, seq_len=8192
    uv run torchrun --standalone --nproc-per-node=16 \\
        scripts/run_nm4_llava_dummy_forward.py \\
        --checkpoint /path/to/iter_0030000 \\
        --processor nt4_processor \\
        --precision-recipe /path/to/precision_recipe.yaml \\
        --tp 4 --cp 4 --ep 4 --seq-len 8192 --num-images 3

    # 8-GPU node: CP=2 (EP=4 still divides DP*TP)
    uv run torchrun --standalone --nproc-per-node=8 \\
        scripts/run_nm4_llava_dummy_forward.py \\
        --checkpoint /path/to/iter_0030000 --processor nt4_processor \\
        --precision-recipe /path/to/precision_recipe.yaml \\
        --tp 4 --cp 2 --ep 4 --seq-len 4096 --num-images 1

    # 4-GPU node: CP off, same TP/EP as the Nano recipe
    uv run torchrun --standalone --nproc-per-node=4 \\
        scripts/run_nm4_llava_dummy_forward.py \\
        --checkpoint /path/to/iter_0030000 --processor nt4_processor \\
        --precision-recipe /path/to/precision_recipe.yaml \\
        --tp 4 --cp 1 --ep 4 --seq-len 4096 --num-images 1

    # Text-only packed THD
    uv run torchrun --standalone --nproc-per-node=8 \\
        scripts/run_nm4_llava_dummy_forward.py \\
        --checkpoint /path/to/iter_0030000 --processor nt4_processor \\
        --precision-recipe /path/to/precision_recipe.yaml \\
        --tp 4 --cp 2 --ep 4 --seq-len 4096 --num-images 0

    # Mixed-resolution images (sizes must be divisible by patch_size 14)
    uv run torchrun --standalone --nproc-per-node=8 \\
        scripts/run_nm4_llava_dummy_forward.py \\
        --checkpoint /path/to/iter_0030000 --processor nt4_processor \\
        --precision-recipe /path/to/precision_recipe.yaml \\
        --tp 4 --cp 2 --ep 4 --seq-len 8192 --image-sizes 448x448,336x224
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE_ROOT = REPO_ROOT / "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
MCORE_ROOT = BRIDGE_ROOT / "3rdparty/Megatron-LM"
for source_root in (REPO_ROOT, BRIDGE_ROOT / "src", MCORE_ROOT):
    sys.path.insert(0, str(source_root))

import torch
import yaml
from PIL import Image
from transformers import AutoProcessor

from megatron.core.packed_seq_params import PackedSeqParams

from nemo_rl.data.multimodal_utils import (
    PackedTensor,
    extract_multimodal_model_inputs,
    image_patch_dim,
)

_DEFAULT_IMAGE_SIZES = ((448, 448), (336, 224), (560, 280))


def _print_rank0(rank: int, message: str) -> None:
    if rank == 0:
        print(message, flush=True)


def _parse_image_sizes(raw: str | None, num_images: int) -> list[tuple[int, int]]:
    if num_images < 0:
        raise ValueError("--num-images must be >= 0")
    if raw:
        sizes: list[tuple[int, int]] = []
        for item in raw.split(","):
            height_str, width_str = item.lower().split("x", 1)
            sizes.append((int(height_str), int(width_str)))
        if len(sizes) != num_images:
            raise ValueError(
                f"--image-sizes has {len(sizes)} entries but --num-images is {num_images}"
            )
        return sizes
    return [size for size, _ in zip(_DEFAULT_IMAGE_SIZES * num_images, range(num_images))]


def _aligned_seq_len(seq_len: int, *, tp: int, cp: int, sequence_parallel: bool) -> int:
    factor = 2 * max(cp, 1)
    if sequence_parallel:
        factor *= max(tp, 1)
    remainder = seq_len % factor
    if remainder == 0:
        return seq_len
    return seq_len + factor - remainder


def _validate_world_size(*, world_size: int, tp: int, pp: int, cp: int, ep: int, etp: int) -> int:
    model_parallel = tp * pp * cp
    if world_size % model_parallel:
        raise ValueError(
            f"world_size={world_size} is not divisible by TP*PP*CP={model_parallel}"
        )
    data_parallel = world_size // model_parallel
    expert_mesh = ep * etp
    if (data_parallel * tp) % expert_mesh:
        raise ValueError(
            f"EP*ETP={expert_mesh} must divide DP*TP={data_parallel * tp} "
            f"(world_size={world_size}, TP={tp}, PP={pp}, CP={cp}, EP={ep}, ETP={etp})"
        )
    return data_parallel


def _resolve_precision_recipe(checkpoint: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    sibling = checkpoint / "precision_recipe.yaml"
    if sibling.is_file():
        return sibling
    raise FileNotFoundError(
        "Pass --precision-recipe or put precision_recipe.yaml next to the checkpoint"
    )


def _checkpoint_from_config(config_path: Path) -> Path | None:
    with config_path.open() as handle:
        parsed = yaml.safe_load(handle)
    path = (
        parsed.get("checkpointing", {})
        .get("pretrained_checkpoint", {})
        .get("path")
    )
    if not path:
        return None
    return Path(path)


def _dummy_images(sizes: list[tuple[int, int]]) -> list[Image.Image]:
    images = []
    for index, (height, width) in enumerate(sizes):
        color = ((40 * (index + 1)) % 256, 90, 140)
        images.append(Image.new("RGB", (width, height), color=color))
    return images


def _as_nchw_tiles(pixel_values: Any) -> list[torch.Tensor]:
    if isinstance(pixel_values, list):
        tiles = [torch.as_tensor(item) for item in pixel_values]
    else:
        tensor = torch.as_tensor(pixel_values)
        if tensor.ndim == 3:
            tiles = [tensor]
        elif tensor.ndim == 4:
            tiles = [tensor[index] for index in range(tensor.shape[0])]
        else:
            raise ValueError(
                f"processor pixel_values must be CHW or NCHW, got {tuple(tensor.shape)}"
            )
    normalized = []
    for tile in tiles:
        if tile.ndim != 3:
            raise ValueError(f"Each image tile must be CHW, got {tuple(tile.shape)}")
        normalized.append(tile.unsqueeze(0))
    return normalized


def _processor_batch(
    processor: Any,
    *,
    images: list[Image.Image],
    seq_len: int,
    device: torch.device,
) -> dict[str, torch.Tensor | PackedSeqParams | None]:
    image_token = getattr(processor, "image_token", "<image>")
    if images:
        text = "Describe the images. " + " ".join(image_token for _ in images)
        processed = dict(
            processor(text=text, images=images, return_tensors=None)
        )
        tiles = _as_nchw_tiles(processed["pixel_values"])
        patch_dim = image_patch_dim(processor)
        processed["imgs_sizes"] = torch.tensor(
            [[int(tile.shape[-2]), int(tile.shape[-1])] for tile in tiles],
            dtype=torch.long,
        )
        processed["pixel_values"] = PackedTensor(
            tiles,
            dim_to_pack=0,
            preprocess_mode="patchify",
            preprocess_kwargs={"patch_dim": patch_dim},
        ).as_tensor()
        processed["input_ids"] = torch.as_tensor(processed["input_ids"])
        extracted = extract_multimodal_model_inputs(processor, processed)
        pixel_values = extracted["pixel_values"].as_tensor()
        imgs_sizes = extracted["imgs_sizes"].as_tensor()
        input_ids = processed["input_ids"]
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
    else:
        pad_id = processor.tokenizer.pad_token_id or 0
        input_ids = torch.full((1, seq_len), pad_id, dtype=torch.long)
        input_ids[0, :16] = torch.arange(16, dtype=torch.long) + 10
        pixel_values = None
        imgs_sizes = None
        patch_dim = image_patch_dim(processor)

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"Expected processor input_ids [1, S], got {tuple(input_ids.shape)}")
    token_len = int(input_ids.shape[1])
    if token_len > seq_len:
        raise ValueError(
            f"Processor sequence length {token_len} exceeds --seq-len {seq_len}. "
            "Use a longer seq-len or fewer/smaller images."
        )
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0
    padded = torch.full((1, seq_len), int(pad_id), dtype=torch.long)
    padded[:, :token_len] = input_ids.to(dtype=torch.long)
    padding_mask = torch.ones((1, seq_len), dtype=torch.bool)
    padding_mask[:, :token_len] = False
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    labels = padded.clone()
    loss_mask = (~padding_mask).to(dtype=torch.float32)
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32)
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens,
        cu_seqlens_kv_padded=cu_seqlens,
        max_seqlen_q=seq_len,
        max_seqlen_kv=seq_len,
    )
    if pixel_values is not None:
        pixel_values = pixel_values.to(device=device, dtype=torch.bfloat16)
        imgs_sizes = imgs_sizes.to(device=device)
    return {
        "input_ids": padded.to(device=device),
        "position_ids": position_ids.to(device=device),
        "labels": labels.to(device=device),
        "loss_mask": loss_mask.to(device=device),
        "padding_mask": padding_mask.to(device=device),
        "pixel_values": pixel_values,
        "imgs_sizes": imgs_sizes,
        "packed_seq_params": packed_seq_params,
        "token_len": torch.tensor(token_len),
        "patch_dim": torch.tensor(patch_dim),
    }


def _summarize_output(output: Any) -> str:
    if isinstance(output, tuple):
        pieces = [_summarize_output(item) for item in output]
        return "tuple(" + ", ".join(pieces) + ")"
    if not torch.is_tensor(output):
        return type(output).__name__
    finite = bool(torch.isfinite(output).all().item())
    return (
        f"Tensor(shape={tuple(output.shape)}, dtype={output.dtype}, "
        f"finite={finite}, absmax={output.detach().float().abs().max().item():.4g})"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--processor", type=Path, default=REPO_ROOT / "nt4_processor")
    parser.add_argument("--precision-recipe", type=Path, default=None)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional NeMo-RL YAML used only to fill --checkpoint from pretrained_checkpoint.path",
    )
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=4)
    parser.add_argument("--etp", type=int, default=1)
    parser.add_argument("--gtp", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument(
        "--image-sizes",
        type=str,
        default=None,
        help="Comma-separated HxW list, e.g. 448x448,336x224. Must match --num-images.",
    )
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="Build the model but skip dist-ckpt weights (random init smoke test)",
    )
    return parser


def main() -> None:
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    args = _parser().parse_args()
    if args.pp != 1:
        raise ValueError("This smoke test only supports --pp 1")

    checkpoint = args.checkpoint
    if checkpoint is None and args.config is not None:
        checkpoint = _checkpoint_from_config(args.config.resolve())
    if checkpoint is None:
        raise ValueError("Pass --checkpoint or --config with checkpointing.pretrained_checkpoint.path")
    checkpoint = checkpoint.resolve()
    processor_path = args.processor.resolve()
    precision_recipe = _resolve_precision_recipe(checkpoint, args.precision_recipe)
    image_sizes = _parse_image_sizes(args.image_sizes, args.num_images)
    sequence_parallel = args.tp > 1
    seq_len = _aligned_seq_len(
        args.seq_len, tp=args.tp, cp=args.cp, sequence_parallel=sequence_parallel
    )

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    data_parallel = _validate_world_size(
        world_size=world_size,
        tp=args.tp,
        pp=args.pp,
        cp=args.cp,
        ep=args.ep,
        etp=args.etp,
    )
    device = torch.device("cuda", local_rank)

    from megatron.bridge.models.experimental_nm4_llava_provider import (
        ExperimentalNM4LlavaProvider,
    )
    from megatron.bridge.training.checkpointing import _load_model_weights_from_checkpoint
    from megatron.bridge.training.model_load_save import load_model_config

    _print_rank0(
        rank,
        "NM4 dummy forward: "
        f"world={world_size} TP={args.tp} PP={args.pp} CP={args.cp} "
        f"EP={args.ep} ETP={args.etp} DP={data_parallel} GTP={args.gtp} "
        f"seq_len={seq_len} images={image_sizes}",
    )
    _print_rank0(rank, f"checkpoint={checkpoint}")
    _print_rank0(rank, f"processor={processor_path}")
    _print_rank0(rank, f"precision_recipe={precision_recipe}")

    processor = AutoProcessor.from_pretrained(str(processor_path), trust_remote_code=True)
    batch = _processor_batch(
        processor,
        images=_dummy_images(image_sizes),
        seq_len=seq_len,
        device=device,
    )
    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is None:
        image_token_id = processor.tokenizer.convert_tokens_to_ids(
            getattr(processor, "image_token", "<image>")
        )
    placeholder_count = int((batch["input_ids"] == int(image_token_id)).sum().item())
    pixel_shape = None if batch["pixel_values"] is None else tuple(batch["pixel_values"].shape)
    size_shape = None if batch["imgs_sizes"] is None else tuple(batch["imgs_sizes"].tolist())
    _print_rank0(
        rank,
        "processor batch: "
        f"token_len={int(batch['token_len'])} patch_dim={int(batch['patch_dim'])} "
        f"pixel_values={pixel_shape} imgs_sizes={size_shape} "
        f"image_placeholders={placeholder_count} image_token_id={image_token_id}",
    )

    parallel_overrides = {
        "tensor_model_parallel_size": args.tp,
        "pipeline_model_parallel_size": args.pp,
        "context_parallel_size": args.cp,
        "expert_model_parallel_size": args.ep,
        "expert_tensor_parallel_size": args.etp,
        "sequence_parallel": sequence_parallel,
        "gtp_weight_remat_size": args.gtp,
        "expert_gtp_weight_remat_size": args.gtp,
        "bf16": True,
        "params_dtype": torch.bfloat16,
    }
    if (checkpoint / "run_config.yaml").is_file():
        provider, mlm_args = load_model_config(str(checkpoint))
        if mlm_args is not None:
            raise TypeError(
                "Expected a Megatron-Bridge provider checkpoint, not Megatron-LM args"
            )
        provider.apply_overrides_and_finalize(
            dtype=torch.bfloat16,
            overrides=parallel_overrides,
        )
    else:
        provider = ExperimentalNM4LlavaProvider(
            checkpoint_path=str(checkpoint),
            precision_recipe_path=str(precision_recipe),
            **parallel_overrides,
        )
        provider.finalize()
    provider.initialize_model_parallel(seed=0)
    models = provider.provide_distributed_model(
        wrap_with_ddp=False,
        bf16=True,
        mixed_precision_wrapper=None,
    )
    model = models[0]
    if not args.skip_load:
        _load_model_weights_from_checkpoint(
            str(checkpoint),
            models,
            dist_ckpt_strictness="log_unexpected",
        )
    model.eval()

    forward_kwargs: dict[str, Any] = {
        "input_ids": batch["input_ids"],
        "position_ids": batch["position_ids"],
        "attention_mask": None,
        "labels": batch["labels"],
        "loss_mask": batch["loss_mask"],
        "padding_mask": batch["padding_mask"],
        "packed_seq_params": batch["packed_seq_params"],
        "media_token_validity_mask": ~batch["padding_mask"],
    }
    if batch["pixel_values"] is not None:
        forward_kwargs["pixel_values"] = batch["pixel_values"]
        forward_kwargs["imgs_sizes"] = batch["imgs_sizes"]

    with torch.no_grad():
        output = model(**forward_kwargs)

    torch.distributed.barrier()
    print(f"[rank {rank}] output {_summarize_output(output)}", flush=True)
    if rank == 0:
        print("PASS: NM4LlavaModel dummy forward completed", flush=True)


if __name__ == "__main__":
    main()
