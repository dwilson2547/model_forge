"""Teacher-dataset workflow: build a prompt mix once, answer it with any teacher, distill.

  python -m distill.prompts fetch                      # pinned sources -> $FORGE_DATA/distill/src
  python -m distill.prompts build [--mix default]      # -> $FORGE_DATA/distill/<mix>.prompts.jsonl
  python -m distill.datagen <label> <teacher.gguf> [--mix default] [--slots 4]
                                                       # -> $FORGE_DATA/distill/<label>.<mix>.jsonl
  python -m distill.datagen <label> queue:<model> ...  # same, as background jobs on an HTTP GPU
                                                       # job queue (distill/queue.py)
  surgery/recover_gpu.py <student hf> --sft-dataset <that jsonl> --distill <teacher hf> ...

The prompt mix is model-agnostic and fixed by its seed, so every teacher (Qwen, Gemma, ...)
answers the same prompts and their datasets are comparable. The teacher runs through
llama-server with the GGUF's own chat template, thinking off by default (--think keeps
reasoning_content separately from the answer; a teacher without a thinking mode leaves it
empty).

Every prompt that shares a 13-word run with a board v2 item (bench/suites.py) is dropped
before it reaches a teacher: training on the board would make the board meaningless.

Records are the trainer's --sft-dataset format: {id, kind, user, reasoning, answer} plus
provenance (teacher, stop, tokens, check). With --distill the trainer scores KL against the
teacher's full distribution on the answer positions, so the text is a vehicle: what the
student learns is the teacher's whole next-token distribution in the deployed chat format.
"""
