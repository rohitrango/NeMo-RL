# Timeline

## 2026-09-15 22:27:28 PDT

- User started a goal to identify NT4 Nano training quirks and improve CLEVR
  SFT performance.
- Loaded the rem, auto-research, and session-memory procedures.
- Initial resolved-config inspection found major differences in packing,
  sequence length, topology, and model-specific paths that require controlled
  comparison before attributing the gap to model quality.

## 2026-09-15 23:05 PDT

- Read the existing exact checkpoint audit: preserving quantile-balanced
  routing loads 1,419/1,419 tensors; the CLEVR recipe's seq-aux router loads
  1,361 and drops 58 trained router-state tensors.
- Read native checkpoint args directly from distributed common state. Native
  values include MTP scale 0.1, quantile-balanced routing, FP32 router logits,
  zero router auxiliary coefficient, MTP HSM enabled, and global batch 72.
- Confirmed the current recipe uses MTP scale 0.3, seq-aux routing coefficient
  1e-4, unset router dtype, and an effective 18-20 sources per optimizer step.
- Confirmed the Pixtral tokenizer and processor are internally aligned and the
  media token/projected feature counts match. Gross tokenizer and media-count
  mismatches are no longer leading hypotheses.
- Formed a three-run factorial screen and paused for the required campaign
  confirmation before submitting jobs.

## 2026-09-15 23:45 PDT

- User explicitly authorized running experiments until the leading NT4
  performance hypotheses are ruled out or narrowed down.
- Re-read the checkpoint common state. The native router settings are
  quantile balancing, FP32 router logits, global-batch estimation, 1,000 bins,
  sigmoid scores, top-k 10, scaling 3.16, router fusion, bias update rate
  0.001, and zero auxiliary coefficient.
- Validated each experiment's Hydra overrides from a freshly loaded config and
  passed `MasterConfig` validation. The first check exposed that the override
  helper mutates its input; reloading per arm confirmed there is no actual
  experiment contamination.
- Next: register, dry-run, and submit the native-router, native-MTP, and
  combined 100-step jobs with 500-example validation at steps 0 and 100.
- Registered and dry-ran three rem experiments without changing project
  metadata, then submitted them on `nemotron_n4_post`, 4 nodes x 4 GPUs:
  native router job 3778348, native MTP job 3778355, combined job 3778359.

## 2026-09-15 22:53 PDT

- Resumed after context compaction and reloaded the auto-research,
  session-memory, and rem experiment procedures.
- Verified jobs 3778348, 3778355, and 3778359 are still pending with reason
  `Priority`; no run metrics exist yet.
- Continue source-level packing/batch audit while waiting, then use the
  factorial result to decide whether an unpacked follow-up is necessary.

## 2026-09-15 23:13 PDT

- Native-router job 3778348 completed step-0 validation, then failed on its
  first training forward because Transformer Engine 2.15's fused router does
  not expose the histogram outputs required by quantile balancing.
- The failure gives a required compatibility setting for checkpoint-native
  routing: `policy.megatron_cfg.moe_router_fusion=false`.
- The nominal `data.validation.limit=500` run still evaluated all 5,000 valA
  sources. The distributed packed Energon loader does not make that setting a
  global cap in this topology. The full validation is 56 batches, so the
  replacement experiments use `sft.val_batches=6` (about 500 global sources).
- Cancelled superseded jobs and submitted the corrected three-arm screen:
  native MTP 3779992, native router/unfused 3780002, and combined/unfused
  3780003.
- Those replacements exposed a command-rendering error before model setup:
  the optional answer-diagnostic env value lost its quotes and Hydra parsed it
  as an integer. Removed that redundant override and submitted clean jobs:
  native MTP 3780226, native router/unfused 3780231, combined/unfused 3780238.

## 2026-09-15 23:27 PDT

- The six-batch cap was confirmed to cover 550 sources, but answer metrics were
  zero because answer-mask construction is gated by
  `NRL_SFT_ANSWER_DIAGNOSTICS=1`; the override is required, not redundant.
- The router jobs 3780231 and 3780238 also showed that the newly introduced
  `moe_router_fusion` key must use Hydra's `+` append syntax.
- Cancelled only the defective attempts and submitted corrected jobs with the
  diagnostic value explicitly quoted as a string: native MTP 3780694, native
  router/unfused 3780695, and combined/unfused 3780696.

## 2026-09-16 00:02 PDT

- Found that the Hydra config accepted `moe_router_fusion=false`, but
  `_apply_moe_config` did not propagate it to the model provider. Added the
  generic propagation and test coverage. Corrected router jobs then trained.
- Completed the matched screen. Baseline answer CE/EM was 1.197114/0.454380;
  native router 1.250975/0.456204; MTP 0.1 1.236772/0.441606; combined
  1.290178/0.427007; LR 1e-5 1.283191/0.425182.
- Concluded router state, MTP weight, and an undersized LR are not the main
  cause of the NT4 gap.
- Submitted two independent follow-ups: unpacked GBS16 job 3781358 and packed
  reference-Adam job 3781516.

## 2026-09-16 00:37 PDT

- Unpacked GBS16 job 3781358 completed. On 500 validation sources, answer CE
  is 0.924470 and exact match is 0.580000, versus 1.197114 and 0.454380 for
  the matched packed control. Final train answer CE is 0.561022 with grad norm
  4.184837.
- Audited the prepacked path and found that it never constructs the MoE
  padding mask, so physical padding tokens participate in expert routing and
  router auxiliary/bias statistics. Added an opt-in mask that preserves
  physical attention boundaries and submitted job 3782122.
- Submitted job 3782018 to test whether freezing the pretrained vision
  projector preserves cross-modal alignment during short SFT.

## 2026-09-16 01:55 PDT

- Resumed after compaction and reloaded the auto-research, session-memory,
  rem experiment, and repository workflow instructions.
- Verified packed 500-step validation plateaus at answer CE 1.063758 and EM
  0.509982, while unpacked reaches CE 0.762565 and EM 0.640 at step 300.
- Verified exact checkpoint-native role prompting reduces unpacked step-0
  answer CE from 12.066993 to 4.094320 and produces 0.082 exact match before
  any SFT update. This is the strongest current hypothesis.
- Active controlled jobs: long unpacked 3782209, exact-prompt unpacked
  3783261, true one-source packed 3783473, exact-prompt packed 3783552, and
  clean MoE padding-mask resubmission 3783617.

## 2026-09-16 02:02 PDT

- Exact checkpoint-native prompting finished neutral at step 100: unpacked
  answer CE/EM 0.979755/0.588 versus plain 0.924470/0.580; packed
  1.263203/0.463504 versus plain 1.197114/0.454380. It fixes initialization,
  not trained quality.
- Padding-mask job 3783617 exposed that the original 512-wide source mask was
  still forwarded in generic multimodal kwargs after a 2048-wide aligned mask
  was built. Removed the duplicate transport and submitted job 3783753.
- Verified the active Pixtral processor matches the saved processor config and
  checkpoint vision geometry. Submitted unpacked zero-MTP-loss job 3783835 to
  test auxiliary interference on one-token answers.

## 2026-09-16 02:50 PDT

- Completed the remaining controls. Zero MTP, exact checkpoint prompting,
  frozen projector, reference Adam, and excluding padding from MoE routing are
  all neutral or worse. Unpacked training reaches CE/EM 0.762/0.652 at step
  500, still far below Qwen and Omni.
- One source per physical pack matches unpacked training, while multi-source
  packed and unpacked runs also match when all images are replaced by identical
  black images. This isolated an image-content interaction within a pack.
- Traced the NT4 wrapper and found that concatenated image patches were sent to
  the frozen ViT without per-image vision boundaries. Added one vision THD
  segment per `imgs_sizes` entry, matching the Nemotron Omni implementation.
- The fixed multi-source packed arm recovers the full short-horizon gap:
  step-100 answer CE/EM is 0.903046/0.580292 versus old packed
  1.197114/0.454380 and unpacked 0.924470/0.580000.
- Committed the vision fix, packed MoE mask plumbing, router-fusion config
  propagation, packing-cap control, black-image diagnostic, and Bridge warning
  cleanup as separate feature commits. Job 3784435 is running the fixed path
  to 500 steps before the research documentation is finalized.

## 2026-09-16 03:17 PDT

- Fixed packed job 3784435 completed all 500 steps. Validation answer CE/EM is
  0.878369/0.614964 at step 100, 0.751602/0.685083 at step 300, and
  0.667250/0.702359 at step 500.
- This beats old packed 1.063758/0.509982 and unpacked 0.762353/0.652000 at
  step 500, confirming that per-image ViT boundaries are the actionable fix.
- Final training-batch loss is 0.232847, answer CE 0.465688, and grad norm
  4.540074. Updated the experiment ledger and comparison logbook.
