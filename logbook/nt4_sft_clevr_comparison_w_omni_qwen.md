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

## Step-100 training results

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

## Validation follow-up

The original comparison had `checkpointing.enabled: false`, so it did not
preserve step-100 weights. A matched rerun will evaluate the complete CLEVR
validation split before training (step 0) and after 100 steps (step 100). The
full-validation results will be appended here.
