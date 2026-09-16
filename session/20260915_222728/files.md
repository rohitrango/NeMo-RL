# Files

## Inspected

- `3775506-logs/ray-driver.log` - matched NT4 resolved config and results.
- `3775507-logs/ray-driver.log` - matched Qwen resolved config and results.
- `3775508-logs/ray-driver.log` - matched Omni resolved config and results.
- `logbook/nt4_sft_clevr_comparison_w_omni_qwen.md` - prior findings and
  benchmark metrics.

## Changed

- `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src/megatron/bridge/models/experimental_nm4_llava_provider.py`
  - isolate every packed image as its own vision-attention sequence, forward
  aligned padding masks, and summarize TE checkpoint warnings.
- `nemo_rl/models/megatron/{data.py,train.py}` - construct and hand off packed
  MoE padding masks in the final model layout.
- `nemo_rl/models/{megatron/setup.py,policy/__init__.py}` - propagate and type
  the optional `moe_router_fusion` override.
- `nemo_rl/data/energon/{config.py,sft_dataloader.py}` - expose the packer's
  `max_sequences_per_bin` control.
- `nemo_rl/data/energon/multimodal/task_encoders/generic_sft.py` - add the
  same-size black-image diagnostic.
- `tests/unit/models/megatron/{test_megatron_data.py,test_megatron_setup.py}` -
  cover the config and mask plumbing (not run per user request).
- `logbook/nt4_sft_clevr_comparison_w_omni_qwen.md` - consolidate benchmark
  results and the confirmed root cause.
- `reports/auto_research/nt4_nano_clevr/experiments.tsv` - experiment ledger.

## Generated

- `session/20260915_222728/` - durable research-session state.
- `reports/auto_research/nt4_nano_clevr/` - untracked experiment artifacts.
