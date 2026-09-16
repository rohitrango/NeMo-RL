# Session State

- Session: 20260915_222728
- Repo: /lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/rohitkumarj/code/RL/nemo-rl-sft_v2_up_n4
- Branch: rohit/sft_v2_up_n4
- Started: 2026-09-15 22:27:28 PDT
- Updated: 2026-09-16 03:20 PDT

## Goal

Identify NT4 Nano training quirks that explain its worse CLEVR SFT performance
versus Qwen VL and Nemotron Omni, validate the leading hypotheses, and improve
the recipe when evidence supports a change.

## Current Subtask

Research campaign complete; hand off the confirmed fix, benchmark evidence,
and residual checkpoint-stage hypothesis.

## Loaded Skills

- `rem` - inspect and submit reproducible Slurm experiments without changing
  the rem project configuration.
- `nemo-rl-auto-research` - hypothesis-driven experiment campaign rules.
- `nemo-rl-session-memory` - durable session and experiment state.

## Current Status

- The primary NT4 packed-training regression is confirmed: all images in a
  physical language pack were one frozen-ViT attention sequence because the
  NT4 wrapper supplied no per-image vision `PackedSeqParams`.
- The fix builds one vision THD segment per `imgs_sizes` entry. Step-100 answer
  CE/EM improves from 1.197114/0.454380 to 0.903046/0.580292, matching unpacked
  0.924470/0.580000.
- One-source packing matches unpacked, and packed/unpacked black-image controls
  match each other, independently supporting the cross-image interaction.
- Router state, MTP weight, LR, Adam settings, projector freezing, prompt
  format, MoE padding, media count, processor geometry, and insufficient source
  exposure are ruled out as primary causes.
- Unpacked 500-step training plateaus at answer CE/EM 0.762353/0.652. The
  residual gap to Qwen/Omni is most consistent with checkpoint stage: NT4 is a
  base pretraining VLM, versus Instruct/Reasoning comparison checkpoints.
- Job 3784435 completed the fixed packed path through step 500 at answer
  CE/EM 0.667250/0.702359.
- Preserve all unrelated untracked files and the nested untracked Megatron-LM
  checkout inside the Bridge submodule.

## Plan

- [x] Diff resolved configs and training curves across the three runs.
- [x] Verify NT4 checkpoint loading, frozen/trainable modules, optimizer and
  scheduler behavior, token/label construction, image preprocessing, and
  packed-sequence semantics.
- [x] Rank concrete hypotheses with evidence and minimal A/B experiments.
- [x] Obtain explicit permission to run experiments until a leading hypothesis
  is ruled out or narrowed down.
- [x] Launch the three targeted experiments.
- [x] Monitor the router/MTP factorial and extract matched metrics.
- [x] Rule out checkpoint-native router state, MTP weight, and a 2x learning
  rate as primary causes.
- [x] Complete the unpacked matched-source follow-up.
- [x] Complete the reference-Adam follow-up.
- [x] Test whether prepacked padding tokens pollute MoE routing statistics.
- [x] Test whether freezing the pretrained vision projector preserves alignment.
- [x] Measure the old packed and unpacked 500-step adaptation curves.
- [x] Measure visual reliance with black-image controls.
- [x] Identify and confirm the cross-image ViT-attention bug.
- [x] Complete the fixed packed 500-step convergence curve.
- [x] Finalize documentation and commits.

## Assumptions

- CLEVR answer CE and teacher-forced exact match are the primary comparison
  metrics; full validation can use 500 examples for iteration speed, followed
  by a 5,000-example confirmation for a winning recipe.

## Latest Controlled Results

Step-100 validation uses about 500 source examples:

- Old multi-source packed: answer CE/EM 1.197114/0.454380.
- Unpacked GBS16: 0.924470/0.580000.
- One source per physical pack: 0.930239/0.604000.
- Fixed multi-source packed with per-image ViT boundaries:
  0.903046/0.580292 in the dedicated 100-step run and
  0.878369/0.614964 at step 100 of the 500-step run.
- Old packed/unpacked black-image controls: 1.356617/0.416058 and
  1.410979/0.424000, respectively.
- Packed MoE-padding mask: 1.378526/0.430657; not an improvement.
- Exact checkpoint prompt unpacked: 0.979755/0.588000; neutral after training.
- Zero MTP unpacked: 0.875884/0.604000; a small/noisy change rather than the
  packed-path recovery.
- Old unpacked step 500: 0.762353/0.652000; old packed step 500:
  1.063758/0.509982.

The fixed packed result is the only controlled intervention that recovers the
entire packed/unpacked gap while keeping the original source exposure.

## Fixed Packed 500-Step Curve

- Step 0: answer CE/EM 12.043454/0.000000.
- Step 100: 0.878369/0.614964.
- Step 200: 0.843255/0.621818.
- Step 300: 0.751602/0.685083.
- Step 400: 0.786209/0.639344.
- Step 500: 0.667250/0.702359.
- Final training batch: loss 0.232847, answer CE 0.465688, grad norm 4.540074.

## Conclusion So Far

The concrete training quirk is cross-image vision attention introduced by
language packing. Fixing vision boundaries restores the expected short-horizon
quality without disabling packing. The remaining cross-model gap survives
unpacked/fixed execution and all tested recipe controls; checkpoint stage is
the leading residual explanation.
