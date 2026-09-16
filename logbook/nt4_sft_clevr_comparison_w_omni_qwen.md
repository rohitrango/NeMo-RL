# NT4 CLEVR SFT comparison with Nemotron Omni and Qwen VL

Date: 2026-09-15

## Scope

This comparison runs 100 CLEVR SFT optimizer steps for:

- Nemotron 4 Nano (NT4 Nano)
- Qwen2.5-VL-7B-Instruct (Qwen VL)
- Nemotron Omni Nano (Omni Nano)

All runs use the model-native processor/tokenizer path appropriate to the
checkpoint. The diagnostics separate answer-token cross entropy and answer
exact-match accuracy from the ordinary SFT loss, which also supervises chat
template tokens.

## Initial diagnostic step-100 training results

| Model | Slurm job | Loss | Grad norm | Answer-token CE | Answer exact match |
| --- | ---: | ---: | ---: | ---: | ---: |
| NT4 Nano | 3773196 | 0.573692 | 6.844917 | 1.147176 | 0.50 |
| Qwen VL | 3773896 | 0.009745 | 12.359440 | 0.055438 | 1.00 |
| Omni Nano | 3774107 | 0.100625 | 11.498828 | 0.402493 | 0.75 |

These are final training-batch values, not full-validation scores. The three
jobs completed all 100 steps successfully.

## Assistant-token supervision

The decoded assistant spans differ substantially by chat template:

| Model | Decoded assistant span | Answer tokens | Template tokens | Total tokens in dumped example |
| --- | --- | ---: | ---: | ---: |
| NT4 Nano | `0\n\n` | 1 | 1 | 2 |
| Qwen VL | `<|im_start|>assistant\nno<|im_end|>\n` | 1 | 5 | 6 |
| Omni Nano | `<think></think>0<|im_end|>` | 1 | 3 | 4 |

The corresponding step-100 aggregate token counts were:

| Model | Answer tokens | All supervised tokens | Answer fraction |
| --- | ---: | ---: | ---: |
| NT4 Nano | 20 | 40 | 0.500 |
| Qwen VL | 17 | 97 | 0.175 |
| Omni Nano | 16 | 64 | 0.250 |

Qwen and Omni therefore average their ordinary SFT loss over more easy chat
template tokens than NT4. This explains a substantial part of their lower
ordinary loss. The answer-only metrics remove that denominator difference.
Qwen still has the best answer-token CE and exact match on the final training
batch, followed by Omni and NT4, but a single training batch is too small for
a model-quality conclusion.

Assistant-token dumps:

- NT4 Nano: `/lustre/fsw/portfolios/coreai/users/rohitkumarj/data/nemorl/rohit-sft_v2_up_n4/answerdiag-nt4-clevr/assistant_tokens.json`
- Qwen VL: `/lustre/fsw/portfolios/coreai/users/rohitkumarj/data/nemorl/rohit-sft_v2_up_n4/answerdiag-qwen-clevr/assistant_tokens.json`
- Omni Nano: `/lustre/fsw/portfolios/coreai/users/rohitkumarj/data/nemorl/rohit-sft_v2_up_n4/answerdiag-omni-clevr/assistant_tokens.json`

## Data and processor findings

- The NT4 Pixtral processor expands the CLEVR `<image>` placeholder into the
  model's image-token sequence.
- Qwen and Omni use their built-in chat templates. The prior generic template
  expected string message contents and was incompatible with multimodal
  OpenAI-style content lists.
- The generic CLEVR Energon shards omit image width and height. The generic
  cooker now derives missing dimensions from the image payload so Omni can
  encode the examples.
- Omni Nano used TP=4, EP=4, CP=1, ETP=1 on one four-GPU node.

## Full CLEVR valA validation

The matched rerun evaluates all 5,000 CLEVR valA source examples before
training and after 100 optimizer steps. These are teacher-forced metrics:

- `Loss` is CE over every supervised assistant/template token.
- `Answer-token CE` includes only the raw answer tokens.
- `Answer exact match` requires the teacher-forced argmax prediction to match
  every raw answer token. It is not autoregressive generation accuracy.

| Model | Slurm job | Step | Loss | Answer-token CE | Answer exact match | Sources |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NT4 Nano | 3775506 | 0 | 8.048137 | 10.986673 | 0.0000 | 5,000 |
| NT4 Nano | 3775506 | 100 | 0.622899 | 1.245606 | 0.4596 | 5,000 |
| Qwen VL | 3775507 | 0 | 9.353258 | 13.540986 | 0.0000 | 5,000 |
| Qwen VL | 3775507 | 100 | 0.010812 | 0.063142 | 0.9790 | 5,000 |
| Omni Nano | 3775508 | 0 | 2.202190 | 8.601472 | 0.0010 | 5,000 |
| Omni Nano | 3775508 | 100 | 0.041173 | 0.160820 | 0.9582 | 5,000 |

The matching final training-batch values from these reruns are:

| Model | Step-100 train loss | Step-100 grad norm | Step-100 train answer-token CE |
| --- | ---: | ---: | ---: |
| NT4 Nano | 0.538354 | 6.771293 | 1.076549 |
| Qwen VL | 0.000406 | 0.754948 | 0.002279 |
| Omni Nano | 0.017680 | 4.301732 | 0.070708 |

All three jobs completed successfully. The NT4 forward path also exercised
the NM4 LLaVA media-alignment guard, which compares valid image placeholders
with projected feature rows; no mismatch was detected.

## NT4 training-quirk ablations

The NT4 checkpoint and the CLEVR recipe differ in two conspicuous auxiliary
settings: the checkpoint was trained with quantile-balanced FP32 routing and
MTP weight 0.1, while the recipe uses sequence-aux routing and MTP weight 0.3.
The base NT4 recipe also specifies LR 1e-5, while the CLEVR override uses
5e-6. Each setting was tested for 100 steps against a fresh matched control.
Validation was capped at six packed batches: 550 sources at step 0 and 548 at
step 100.

| Arm | Slurm job | Val loss | Answer-token CE | Answer exact match | Final train loss | Final grad norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Matched packed control | 3780834 | 0.598654 | 1.197114 | 0.454380 | 0.534454 | 6.501661 |
| Native quantile/FP32 router | 3781007 | 0.625544 | 1.250975 | 0.456204 | 0.580437 | 6.343593 |
| Native MTP weight 0.1 | 3780694 | 0.618454 | 1.236772 | 0.441606 | 0.584752 | 5.889461 |
| Native router + MTP 0.1 | 3781009 | 0.645125 | 1.290178 | 0.427007 | 0.595470 | 5.514694 |
| LR 1e-5 | 3781084 | 0.641641 | 1.283191 | 0.425182 | 0.633642 | 8.555198 |
| Adam beta2 0.98, epsilon 1e-5 | 3781516 | 0.688307 | 1.375886 | 0.399635 | 0.678837 | 7.909110 |
| Frozen vision projector | 3782018 | 0.623045 | 1.245824 | 0.441606 | 0.586761 | 7.196460 |

These experiments rule out the router mismatch, MTP loss weight, a too-small
learning rate, and the Adam moment settings as primary causes of NT4's gap.
Matching Qwen and Omni's beta2 and epsilon makes every validation metric worse.
Freezing the vision projector is also neutral to slightly worse, ruling out
rapid projector drift as the explanation for NT4's short-horizon gap.
The native router
does reduce the very large early gradient norms, but it does not improve
step-100 answer quality. The observed metric differences are small and mostly
favor the current packed control. With 548 validation examples and accuracy
near 45%, the unpaired 95% binomial interval is roughly plus or minus four
percentage points, so the exact-match differences among these arms are not
meaningful improvements.

The router experiment also exposed a configuration bug: NeMo-RL accepted
`policy.megatron_cfg.moe_router_fusion=false` but did not propagate it into
the Megatron model provider. Quantile routing therefore still selected the
unsupported Transformer Engine fused path. The generic config propagation is
fixed in the working tree; after that fix the native-router arms train normally.

The matched-source unpacked arm is materially better after 100 steps:

| Arm | Slurm job | Val loss | Answer-token CE | Answer exact match | Sources | Final train answer CE | Final grad norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Packed control | 3780834 | 0.598654 | 1.197114 | 0.454380 | 548 | 1.068757 | 6.501661 |
| Unpacked, GBS 16 | 3781358 | 0.462334 | 0.924470 | 0.580000 | 500 | 0.561022 | 4.184837 |

The unpacked arm sees 16 source answers per optimizer step versus 18.3 on
average for the packed control, so it is closely matched in examples but not
bitwise matched in order. The 12.6-point validation exact-match gain and 23%
lower answer CE are nevertheless much larger than the noise among the prior
arms. This narrows a real part of the short-horizon regression to the packed
training path rather than checkpoint router state, MTP weight, or LR.

### Packed-image boundary bug

The primary short-horizon NT4 regression was caused by the vision path, not by
language-attention packing or MoE padding. Energon concatenates the pixel
patches for all images in a physical language pack into one THD tensor. The NT4
wrapper passed that tensor to `ViTModel` without vision `PackedSeqParams`, so
the frozen ViT treated every image in the language pack as one attention
sequence. An image's projected features therefore depended on unrelated
images that happened to share its pack.

The single-source and black-image controls isolated the mechanism:

| Arm | Slurm job | Step-100 val answer CE | Step-100 val exact match |
| --- | ---: | ---: | ---: |
| Multi-source packed, old vision path | 3780834 | 1.197114 | 0.454380 |
| Unpacked, real images | 3781358 | 0.924470 | 0.580000 |
| One source per physical pack | 3783473 | 0.930239 | 0.604000 |
| Unpacked, same-size black images | 3782170 | 1.410979 | 0.424000 |
| Packed, same-size black images | 3784032 | 1.356617 | 0.416058 |
| Multi-source packed, per-image ViT boundaries | 3784299 | 0.903046 | 0.580292 |

One-source packing matches unpacked execution, proving that the fused/prepacked
language path itself is sound. Packed and unpacked black-image results also
match: making all images identical hides the cross-image content interaction.
Most decisively, deriving one vision THD segment per `imgs_sizes` entry recovers
the full gap while retaining multi-source language packing. The fixed packed
answer CE is slightly lower than unpacked and exact match is the same to three
decimal places.

The step-0 answer CE also moves from 10.764785 on the old multi-image path to
12.043454 with per-image vision boundaries, matching unpacked/one-source step 0
(about 12.06). This is expected: the old forward pass had already mixed image
content before any optimizer update.

The fix mirrors the established Nemotron Omni wrapper: compute each image's
raw ViT patch count as `(height // patch_dim) * (width // patch_dim)`, build
CUDA int32 cumulative lengths, and pass them as `PackedSeqParams` to the vision
encoder. The existing media-alignment guard still independently verifies that
the projected feature count equals the number of valid expanded image tokens.

### Ruled-out packed-path alternatives

Physical language-pack padding was allowed into MoE router statistics in the
original path. The opt-in padding-mask implementation preserves physical THD
attention boundaries while excluding padding only from routing. It does not
help: job 3784091 reaches answer CE 1.378526 and exact match 0.430657. This
rules out MoE padding as the primary packed regression.

Packing-cap screens on the old vision path reached CE/EM 0.933106/0.578 with
at most two sources per bin and 0.736500/0.6805 with at most four. They are not
occupancy-controlled comparisons: keeping 16 physical bins raises exposure to
as many as 32 and 64 source answers per optimizer step, respectively. They
show that additional examples can partly compensate for the corrupted vision
context, but do not identify a better packing topology. The fixed-boundary arm
is the controlled intervention because it retains the original data exposure.

Language attention must continue using physical padded boundaries. Replacing
them with logical source lengths makes the hybrid stack consume tensors whose
non-hidden dimensions no longer match and can violate tensor-parallel
divisibility. Logical lengths are appropriate for source-local position IDs
and supervision bookkeeping, but not as direct offsets into a physically
padded THD tensor.

### Prompt and visual-grounding controls

The black-image control degrades unpacked exact match by 15.6 points and raises
answer CE by 53%, proving that NT4 uses the image rather than learning CLEVR
question/answer priors alone. Its zero-shot logits differ under black images,
so only the matched post-training comparison is interpreted as grounding
evidence.

Neither generic ChatML nor the checkpoint's exact `nemotron-h-aligned` role
contract fixes the gap:

| Unpacked prompt | Slurm job | Step-100 val answer CE | Step-100 val exact match | Final train grad norm |
| --- | ---: | ---: | ---: | ---: |
| Plain current template | 3781358 | 0.924470 | 0.580000 | 4.184837 |
| ChatML role boundaries | 3782699 | 0.974068 | 0.576000 | 21.461853 |
| Exact checkpoint prompt | 3783261 | 0.979755 | 0.588000 | -- |

The exact checkpoint prompt improves step-0 answer CE to 4.094 and exact match
to 8.2%, but is neutral after 100 steps. Its packed counterpart is also neutral
(job 3783552: CE 1.263203, exact match 0.463504). Prompt mismatch therefore
changes initialization but does not explain the learned-quality gap.

### Longer-horizon behavior and residual model gap

The old packed, unpacked, and fixed packed 500-step curves are:

| Step | Old packed CE / EM | Unpacked CE / EM | Fixed packed CE / EM |
| ---: | ---: | ---: | ---: |
| 0 | 10.764785 / 0.000000 | 12.066993 / 0.000000 | 12.043454 / 0.000000 |
| 100 | 1.211253 / 0.463504 | 0.882675 / 0.602000 | 0.878369 / 0.614964 |
| 200 | 1.236881 / 0.458182 | 0.917877 / 0.578000 | 0.843255 / 0.621818 |
| 300 | 1.093557 / 0.532228 | 0.762565 / 0.640000 | 0.751602 / 0.685083 |
| 400 | 1.138918 / 0.489982 | 0.742420 / 0.654000 | 0.786209 / 0.639344 |
| 500 | 1.063758 / 0.509982 | 0.762353 / 0.652000 | 0.667250 / 0.702359 |

The fixed packed curve tracks or improves on unpacked training throughout and
finishes 19.8 points above the old packed exact match. This confirms the
per-image vision boundary change as the actionable fix, not merely a favorable
step-100 fluctuation. The final fixed training batch has loss 0.232847,
answer-token CE 0.465688, and grad norm 4.540074.

NT4 still remains far below Qwen's 97.9% and Omni's 95.8% exact match at 100
steps. The NT4 checkpoint is a base pretraining VLM (`model_provider:
pretrain-vlm`, `finetune: false`, `sft: false`, iteration 30,000), whereas Qwen
is an Instruct checkpoint and Omni is a Reasoning checkpoint. After fixing the
concrete packing bug and ruling out the tested recipe controls, this
checkpoint/instruction-tuning difference is the leading explanation for the
remaining cross-model gap.

## Recommendation

- Keep multi-source language packing enabled, but always construct per-image
  vision `PackedSeqParams` before calling the NT4 ViT.
- Retain the media placeholder/projected-feature count guard; it catches count
  mismatches but cannot detect cross-image attention, so both checks are
  needed.
- Do not change router state, MTP weight, LR, Adam settings, prompt format, or
  projector freezing based on this benchmark; none produced a controlled gain.
- Compare against an instruction-tuned NT4 checkpoint, or add an instruction
  tuning stage, before treating the remaining Qwen/Omni gap as an optimizer or
  data-path problem.
