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
