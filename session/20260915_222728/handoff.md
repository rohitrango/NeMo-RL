# Handoff

## Resume From Here

The primary NT4 multi-source packing regression is confirmed and fixed.
Energon concatenated every image's raw patches into one THD tensor, while the
NT4 wrapper passed no vision `PackedSeqParams`; the frozen ViT therefore let
unrelated images in the same language pack attend to each other. Building one
vision segment per `imgs_sizes` entry improves step-100 validation from answer
CE/EM 1.197/0.454 to 0.903/0.580, matching unpacked 0.924/0.580.

## Next Actions

- Monitor job 3784435 through step 500 and record validation at steps
  100/200/300/400/500.
- Update experiment row 45 and the logbook's fixed 500-step curve.
- Update session state and commit the research documentation.
- Run `git diff --check`; tests were intentionally not run per user request.
- Mark the explicit goal complete only after the job and documentation finish.

## Watch Outs

- Use explicit `/lustre/fsw/...` paths.
- Do not edit `~/.config/rem/projects/sft_v2_up_n4/project.json`.
- Preserve all pre-existing dirty and untracked files.
- The residual Qwen/Omni gap remains after unpacked/fixed packing. NT4 is a
  base pretraining VLM checkpoint, while Qwen is Instruct and Omni is Reasoning.
