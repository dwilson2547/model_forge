"""Pipeline: load base -> apply recipe -> build -> evaluate. CLI: python -m surgery.pipeline <recipe.yaml>"""

import argparse
import json
import sys
import time
from pathlib import Path

import transformers


def load_model(base_dir):
    """Load the base checkpoint in bf16 on CPU, whatever wrapper class it needs."""
    errors = {}
    for loader in (transformers.AutoModelForCausalLM, transformers.AutoModelForImageTextToText,
                   transformers.AutoModel):
        try:
            model = loader.from_pretrained(base_dir, dtype="bfloat16", device_map=None)
            model.eval()
            return model
        except Exception as e:  # noqa: BLE001 - try the next loader class
            errors[loader.__name__] = str(e).splitlines()[-1][:300]
    raise RuntimeError(f"could not load {base_dir} with any Auto class: {errors}")


def run_recipe(recipe_path, eval_prompt=None, eval_ppl_file=None, ppl_chunks=24,
               build_model=True):
    from .recipe import load_recipe, Quantize
    from .ops import apply_recipe
    from .build import build, save_hf
    from . import evaluate

    recipe = load_recipe(recipe_path)
    Path(recipe.out_dir).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    tok = transformers.AutoTokenizer.from_pretrained(recipe.base_dir)
    model = load_model(recipe.base_dir)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[pipeline] base loaded: {n_params/1e9:.2f}B params, "
          f"{len(list(model.parameters()))} tensors, {time.time()-t0:.0f}s", flush=True)

    report = apply_recipe(model, recipe, tok)
    report["base_params"] = n_params
    n_params_after = sum(p.numel() for p in model.parameters())
    report["edited_params"] = n_params_after
    report["param_delta"] = n_params_after / n_params
    print(f"[pipeline] surgery done: {n_params_after/1e9:.2f}B params "
          f"({report['param_delta']:.3f}x of base)", flush=True)

    if not build_model:
        # surgery only: save the edited HF dir and stop here — convert/quantize/eval
        # happen on the build host (e.g. x86 desktop slices, box builds). Note
        # apply_drop/explicit is pure tensor bookkeeping, so the edited checkpoint
        # is byte-identical to running the surgery on any other architecture.
        hf_dir = save_hf(model, tok, Path(recipe.out_dir) / "hf-edited")
        report["build"] = {"hf_edited": str(hf_dir), "surgery_only": True}
    else:
        # The GGUF convert must run under this interpreter — the venv has torch; system python doesn't.
        report = build(recipe, report, model, tok, python=sys.executable)
    del model

    qop = next((o for o in recipe.ops if isinstance(o, Quantize)), None)
    if qop and eval_ppl_file:
        f16 = report["build"]["f16"]["gguf"]
        report["eval"] = {"f16": {**evaluate.ppl(f16, eval_ppl_file, recipe.llamacpp_dir,
                                                 chunks=ppl_chunks),
                                  **evaluate.speed(f16, eval_prompt or eval_ppl_file,
                                                  recipe.llamacpp_dir)}}
        for q, info in report["build"]["quants"].items():
            report["eval"][q] = {**evaluate.ppl(info["gguf"], eval_ppl_file, recipe.llamacpp_dir,
                                               chunks=ppl_chunks),
                                 **evaluate.speed(info["gguf"], eval_prompt or eval_ppl_file,
                                                  recipe.llamacpp_dir)}

    (Path(recipe.out_dir) / "results.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in ("base_params", "edited_params", "param_delta")
                      if k in report}, indent=2), flush=True)
    print(json.dumps(report.get("eval", {}), indent=2), flush=True)
    print(f"[pipeline] results: {recipe.out_dir}/results.json", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recipe", help="recipe YAML")
    ap.add_argument("--ppl-file", help="fixed text corpus for perplexity")
    ap.add_argument("--prompt-file", help="prompt file for speed runs (default: ppl file)")
    ap.add_argument("--chunks", type=int, default=24)
    ap.add_argument("--no-build", action="store_true",
                    help="surgery only: save the edited HF dir, skip GGUF convert/quantize/eval")
    args = ap.parse_args()

    run_recipe(args.recipe, eval_prompt=args.prompt_file, eval_ppl_file=args.ppl_file,
               ppl_chunks=args.chunks, build_model=not args.no_build)


if __name__ == "__main__":
    main()
