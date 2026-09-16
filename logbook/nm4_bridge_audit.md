# NM4 Megatron-Bridge checkpoint audit

- Date: 2026-09-15
- Checkpoint: `/data/models/nm4_nano_bridge`
- Provider: `megatron.bridge.models.experimental_nm4_llava_provider.ExperimentalNM4LlavaProvider`
- Node: one GB300 node, four GPUs, Slurm job `3774615`
- Runtime layout: TP=4, PP=1, EP=4, ETP=1
- Model size reported on each TP rank: 25,187,030,784 parameters

## Result

- The checkpoint payload is internally consistent and all 1,419 tensors load when the checkpoint's quantile-balanced router configuration is preserved.
- The exact audit had:
  - 0 missing checkpoint keys
  - 0 unused checkpoint keys
  - 0 global-shape mismatches
  - 0 dtype mismatches
  - 0 loaded shards containing NaN or Inf
- The current `nm4_generalist_16n.yaml` training configuration does not load 58 FP32 router-state tensors. It changes the router load-balancing method from `quantile_balancing` to `seq_aux_loss`.
- The 58 tensors are valid and nonzero. They are 29 learned `expert_bias` buffers and 29 `qb_bin_bounds` buffers. They load correctly when quantile-balanced routing is preserved.
- Training with the current recipe is possible, but it starts with a different router algorithm and does not continue the checkpoint's router state.
- No checkpoint `_extra_state` data is discarded. The `_extra_state` warnings come from empty state expected by the newly constructed Transformer Engine modules.

## Checkpoint inventory

- Logical tensors: 1,419
- Logical elements: 100,145,309,114
- Dtypes:
  - 1,361 `torch.bfloat16` tensors
  - 58 `torch.float32` tensors
- Stored FP8 tensor names: 0
- Stored scale tensor names: 0
- Stored amax tensor names: 0
- Stored `_extra_state` names: 0
- MXFP8 is therefore a runtime Transformer Engine recipe for this checkpoint. The persistent model tensors are BF16, plus FP32 router buffers.

## Exact-router audit

- Audit override:
  - `moe_router_load_balancing_type: quantile_balancing`
  - `moe_aux_loss_coeff: 0.0`
- Expected model tensors after factory expansion: 1,419
- Checkpoint tensors: 1,419
- Loaded local shards across four ranks: 35,956
- Loaded local elements across four ranks: 100,748,182,760
- Missing keys: 0
- Unused keys: 0
- Shape mismatches: 0
- Dtype mismatches: 0
- Non-finite loaded shards: 0
- Loaded-statistics digest: `960e567e69313725ec342f12bf69820e3afe045ec22e20ec2077d4a26a5dbbf2`
- Raw result: [`nm4_bridge_audit_preserved_router_results.json`](nm4_bridge_audit_preserved_router_results.json)
- Run log: [`nm4_bridge_audit_preserved_router_run.log`](nm4_bridge_audit_preserved_router_run.log)

## Current SFT recipe audit

- Recipe: `examples/configs/sft_v2_tests/nm4_generalist_16n.yaml`
- Its inherited base recipe sets:
  - `moe_router_load_balancing_type: seq_aux_loss`
  - `moe_aux_loss_coeff: 0.0001`
- The checkpoint configuration sets:
  - `moe_router_load_balancing_type: quantile_balancing`
  - `moe_router_quantile_balancing_estimation_scope: global_batch`
  - `moe_router_qb_num_bins: 1000`
- Expected runtime tensors after the recipe override: 1,361
- Loaded BF16 checkpoint tensors: 1,361
- Missing runtime tensors: 0
- Shape mismatches: 0
- Dtype mismatches: 0
- Non-finite loaded shards: 0
- Unused checkpoint tensors: 58
- Loaded-statistics digest: `c5960316623dcece5122c7567973767752bdd4a8c63946d2a636fe1a615eef4f`
- Raw result: [`nm4_bridge_audit_results.json`](nm4_bridge_audit_results.json)

### The 58 unused router tensors

- Each affected router contributes:
  - `router.expert_bias`: shape `[512]`, FP32, learned and nonzero
  - `router.qb_bin_bounds`: shape `[2]`, FP32, learned and non-default
- Affected decoder layers:
  - 5, 7, 9, 12, 14, 16, 18, 21, 23, 25, 27, 30, 32, 34
  - 36, 39, 41, 43, 45, 48, 50, 52, 54, 57, 59, 61, 64, 66
- The repeated MTP layer is also affected:
  - `language_model.mtp.layers.0.mtp_model_layer.layers.0.moe_layer.mlp.router`
- The exact key suffixes are `.expert_bias` and `.qb_bin_bounds` for every router listed above.
- Example checkpoint values from decoder layer 12:
  - `expert_bias`: min `-0.6791793704`, max `0.0415765941`, sum approximately `-1.64e-7`
  - `qb_bin_bounds`: `[-1.6791794300, 1.0415766239]`
- These tensors affect quantile-balanced expert routing. They are not used by the recipe's `seq_aux_loss` router.
- `nm4_clevr_4n.yaml` inherits the same `seq_aux_loss` setting and does not override it, so it also omits these 58 buffers.

## `_extra_state` audit

- Checkpoint `_extra_state` entries: 0
- Model-side `_extra_state` entries after construction: 520
- Values:
  - 519 empty `torch.uint8` tensors with shape `[0]` and zero elements
  - 1 `None` value at `language_model.output_layer._extra_state`
- Module counts:
  - `TERowParallelLinear`: 135
  - `TELayerNormColumnParallelLinear`: 105
  - `RMSNorm`: 93
  - `TELinear`: 58
  - `TEDotProductAttention`: 39
  - `TEColumnParallelLinear`: 31
  - `TEColumnParallelGroupedLinear`: 29
  - `TERowParallelGroupedLinear`: 29
  - `ColumnParallelLinear`: 1
- All 520 entries are present on every TP rank.
- These entries contain no weights, scales, amax history, or quantization payload.
- `NM4LlavaModel.sharded_state_dict()` removes them from the requested distributed state before checkpoint loading.
- The module load post-hooks removed 480 missing `_extra_state` names during this load: 356 from the language model, 122 from the vision model, and 2 from the projector. The other 40 model-side entries did not appear in the incompatible-key lists returned to these hooks.
- The old diagnostic said `Ignoring Transformer Engine checkpoint key ...`. This was inaccurate because the keys are missing model-side placeholders and do not exist in the checkpoint.
- The provider diagnostic now reports `missing_keys` and `unexpected_keys` separately and aggregates each category. Loading behavior is unchanged.
- The full list of 520 names and their module classes is in both raw result JSON files.

## Physical checkpoint files

- `__0_0.distcp`: 60,181,546,578 bytes
  - SHA-256: `a2a326e6f4be10169b576b9f29880e1d7cab68a647d4bfc6a598b27ce724b863`
- `__1_0.distcp`: 46,719,476,992 bytes
  - SHA-256: `dbf7a66c944617efaab0f86136c0c6fb82f7d3856e6818813097e1cf2288fac1`
- `__2_0.distcp`: 46,719,476,992 bytes
  - SHA-256: `159a902fa2832760fd4ab9c090a98a90603c2dd974c865b4a5059d340b115498`
- `__3_0.distcp`: 46,719,476,992 bytes
  - SHA-256: `656a728c992b2434ea54de7168da926eecc82d68e6a17d866d697b363e2f4d02`
- `.metadata`: 8,839,855 bytes
  - SHA-256: `34bd0455df72421e91e945d945f5ef0718b823b81b31552a11f73a851034fbed`
- The local checkpoint directory contains symlinks. Sizes and SHA-256 values above follow the symlinks to the source payload.

## Audit method

- Loaded the checkpoint through NeMo-RL's normal Megatron-Bridge path with `load_weights=True` and `load_optimizer=False`.
- Expanded every model `ShardedTensorFactory` to the exact distributed checkpoint keys.
- Compared the union of expected keys across four ranks with all checkpoint tensor metadata.
- Compared global shapes and dtypes for each key.
- Scanned every loaded local shard in bounded chunks and counted finite and non-finite values.
- Collected the constructed model's full state dictionary and classified every `_extra_state` value by key, module class, value type, shape, dtype, and element count.
- Loaded the 58 router buffers separately to inspect their values.
- Read every physical checkpoint file to calculate SHA-256 fingerprints.
- Repeated the complete load audit with the recipe router settings and with the checkpoint router settings.

## Commands

```bash
UV_PROJECT_ENVIRONMENT=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker \
  uv run --no-sync --extra mcore torchrun --standalone --nproc-per-node=4 \
  scripts/audit_nm4_bridge_checkpoint.py \
  --config examples/configs/sft_v2_tests/nm4_generalist_16n.yaml \
  --checkpoint /data/models/nm4_nano_bridge \
  --output logbook/nm4_bridge_audit_results.json

UV_PROJECT_ENVIRONMENT=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker \
  uv run --no-sync --extra mcore torchrun --standalone --nproc-per-node=4 \
  scripts/audit_nm4_bridge_checkpoint.py \
  --config examples/configs/sft_v2_tests/nm4_generalist_16n.yaml \
  --checkpoint /data/models/nm4_nano_bridge \
  --preserve-checkpoint-router \
  --output logbook/nm4_bridge_audit_preserved_router_results.json
```

## Limits

- The audit proves complete tensor mapping, successful loading, metadata agreement, physical readability, and finite loaded values.
- It does not compare the model with an external reference checkpoint because no HF checkpoint exists.
- The loaded-statistics digest is a repeatability aid based on per-shard statistics. It is not an exact byte hash of the in-memory tensors.
- The physical SHA-256 values fingerprint the source files, but no publisher checksum was available for comparison.
- The audit does not run a forward pass, backward pass, optimizer step, or save-and-resume training cycle.

## Recommendation

- Decide whether SFT should continue the checkpoint's quantile-balanced routing or intentionally change to `seq_aux_loss`.
- For exact continuation, preserve `quantile_balancing`, set the auxiliary-loss coefficient to zero, and keep the 58 router buffers in the model state.
- If `seq_aux_loss` is intentional, document that the 58 learned quantile-router buffers are unused and that router behavior changes at initialization.
- Run one training-step smoke test after this choice. The tensor audit alone does not test training execution.
