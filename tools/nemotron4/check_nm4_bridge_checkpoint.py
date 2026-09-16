#!/usr/bin/env python3
"""Validate NM4 Bridge config round-tripping and NeMo-RL checkpoint loading.

Examples:

    python scripts/check_nm4_bridge_checkpoint.py prepare \
        --checkpoint /path/to/iter_0030000 \
        --output /shared/path/nm4_roundtrip_check

    torchrun --nproc-per-node=4 scripts/check_nm4_bridge_checkpoint.py load \
        --config examples/configs/sft_v2_tests/nm4_generalist_16n.yaml \
        --checkpoint /path/to/iter_0030000

    torchrun --nproc-per-node=4 scripts/check_nm4_bridge_checkpoint.py load \
        --config examples/configs/sft_v2_tests/nm4_generalist_16n.yaml \
        --checkpoint /shared/path/nm4_roundtrip_check \
        --resume

For multi-node loading, use the site's normal torchrun/Slurm launcher and make
``--output`` visible to every node. The prepare command symlinks weight shards;
it does not duplicate the checkpoint payload.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE_ROOT = REPO_ROOT / "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
MCORE_ROOT = BRIDGE_ROOT / "3rdparty/Megatron-LM"
for source_root in (REPO_ROOT, BRIDGE_ROOT / "src", MCORE_ROOT):
    sys.path.insert(0, str(source_root))


def _provider_signature(provider: Any) -> dict[str, Any]:
    """Return fields that must survive a provider serialization round-trip."""
    names = (
        "hidden_size",
        "num_layers",
        "seq_length",
        "vocab_size",
        "hybrid_layer_pattern",
        "mtp_num_layers",
        "mtp_use_repeated_layer",
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
        "expert_model_parallel_size",
        "expert_tensor_parallel_size",
        "gtp_weight_remat_size",
        "expert_gtp_weight_remat_size",
    )
    return {name: getattr(provider, name) for name in names}


def _require_empty_output(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to use non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)


def _link_checkpoint_payload(checkpoint: Path, output: Path) -> None:
    for source in checkpoint.iterdir():
        if source.name in {"run_config.yaml", "precision_recipe.yaml"}:
            continue
        destination = output / source.name
        if not destination.exists():
            destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def prepare_roundtrip_checkpoint(checkpoint: Path, output: Path) -> None:
    """Serialize, reconstruct, and create a loadable checkpoint view."""
    from megatron.bridge.training.model_load_save import load_model_config
    from megatron.bridge.training.utils.checkpoint_utils import read_run_config
    from megatron.bridge.training.utils.config_utils import _ConfigContainerBase

    @dataclass
    class RunConfig(_ConfigContainerBase):
        model: Any
        checkpoint: dict[str, Any] = field(
            default_factory=lambda: {
                "save_optim": False,
                "save_rng": False,
                "fully_parallel_save": True,
            }
        )

    checkpoint = checkpoint.resolve()
    if not (checkpoint / ".metadata").is_file():
        raise FileNotFoundError(f"Missing torch_dist metadata: {checkpoint / '.metadata'}")
    _require_empty_output(output)

    provider, mlm_args = load_model_config(str(checkpoint))
    if mlm_args is not None:
        raise TypeError("Expected a Megatron-Bridge provider, not a legacy Megatron-LM config")

    original_signature = _provider_signature(provider)
    run_config_path = output / "run_config.yaml"
    RunConfig(model=provider).to_yaml(str(run_config_path))
    save_assets = getattr(provider, "save_precision_recipe_assets", None)
    if not callable(save_assets):
        raise TypeError("Provider does not expose save_precision_recipe_assets()")
    save_assets(str(output))

    # Prove the serialized architecture is self-contained: reconstruction must
    # succeed even when the original source checkpoint path is unusable.
    serialized = yaml.safe_load(run_config_path.read_text())
    model_config = serialized["model"]
    if not model_config.get("saved_args"):
        raise AssertionError("Serialized provider is missing saved_args")
    if not model_config.get("precision_recipe_yaml"):
        raise AssertionError("Serialized provider is missing precision_recipe_yaml")
    model_config["checkpoint_path"] = "/__nm4_source_checkpoint_must_not_be_read__"
    run_config_path.write_text(yaml.safe_dump(serialized, sort_keys=False))

    restored, restored_mlm_args = load_model_config(str(output))
    if restored_mlm_args is not None:
        raise AssertionError("Round-tripped provider was treated as a legacy checkpoint")
    restored_signature = _provider_signature(restored)
    if restored_signature != original_signature:
        raise AssertionError(
            "Provider round-trip changed configuration:\n"
            f"before={original_signature}\nafter={restored_signature}"
        )

    # Exercise the same TP/PP metadata lookup used by Bridge checkpoint resume.
    from megatron.bridge.training.checkpointing import _get_run_config_tp_pp

    roundtrip_run_config = read_run_config(str(run_config_path))
    _get_run_config_tp_pp(roundtrip_run_config["model"])
    _link_checkpoint_payload(checkpoint, output)
    print(f"PASS: self-contained provider/config round-trip: {output}")
    print("PASS: serialized run_config contains Bridge resume TP/PP metadata")


def load_with_nemo_rl(config_path: Path, checkpoint: Path, *, resume: bool) -> None:
    """Load weights through NeMo-RL's normal Megatron setup path."""
    import torch
    from omegaconf import OmegaConf

    from megatron.bridge.training.checkpointing import _get_run_config_tp_pp
    from megatron.bridge.training.utils.checkpoint_utils import read_run_config
    from nemo_rl.models.megatron.setup import (
        destroy_parallel_state,
        handle_model_import,
        setup_distributed,
        setup_model_and_optimizer,
        validate_and_set_config,
        validate_megatron_config,
        validate_model_paths,
    )
    from nemo_rl.models.policy.workers.patches import apply_transformer_engine_patch
    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

    checkpoint = checkpoint.resolve()
    run_config_path = checkpoint / "run_config.yaml"
    if not run_config_path.is_file():
        raise FileNotFoundError(f"Missing Bridge run config: {run_config_path}")
    run_config = read_run_config(str(run_config_path))
    _get_run_config_tp_pp(run_config["model"])

    register_omegaconf_resolvers()
    resolved = OmegaConf.to_container(load_config(config_path.resolve()), resolve=True)
    policy = resolved["policy"]
    policy["pretrained_checkpoint"] = {
        "path": str(checkpoint),
        "format": "megatron_bridge",
    }

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    apply_transformer_engine_patch()
    setup_distributed(policy)
    try:
        rank = torch.distributed.get_rank()
        hf_model_name, pretrained_path, checkpoint_exists = validate_model_paths(policy)
        handle_model_import(
            policy,
            hf_model_name,
            pretrained_path,
            checkpoint_exists,
        )
        weights_path = str(checkpoint) if resume else None
        runtime = validate_and_set_config(
            policy,
            rank,
            hf_model_name,
            pretrained_path,
            weights_path,
            None,
        )
        validate_megatron_config(runtime.megatron_cfg, policy)
        state = setup_model_and_optimizer(
            policy,
            runtime.megatron_cfg,
            load_optimizer=False,
            load_weights=True,
        )
        if not state.model:
            raise AssertionError("NeMo-RL setup returned no model")
        torch.distributed.barrier()
        if rank == 0:
            mode = "resume" if resume else "initial pretrained"
            print(f"PASS: NeMo-RL standard Bridge {mode} load: {checkpoint}")
    finally:
        destroy_parallel_state()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="check config round-trip and create a checkpoint view")
    prepare.add_argument("--checkpoint", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)

    load = subparsers.add_parser("load", help="load a checkpoint through NeMo-RL's standard Bridge path")
    load.add_argument("--config", type=Path, required=True)
    load.add_argument("--checkpoint", type=Path, required=True)
    load.add_argument(
        "--resume",
        action="store_true",
        help="Treat --checkpoint as both the serialized provider source and resumed weights",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare":
        prepare_roundtrip_checkpoint(args.checkpoint, args.output.resolve())
    else:
        load_with_nemo_rl(args.config, args.checkpoint, resume=args.resume)


if __name__ == "__main__":
    main()
