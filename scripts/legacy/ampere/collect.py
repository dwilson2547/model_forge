import json, glob
rows = []
runs = sorted(glob.glob("/home/daniel/m0/runs/qwen35-4b-drop*"))
for run in runs:
    try:
        d = json.load(open(run + "/results.json"))
    except Exception as e:
        rows.append((run, "no results.json: " + str(e)))
        continue
    o = d["ops"][0]
    e = d.get("eval", {})
    rows.append((
        d["recipe"]["name"],
        f"delta={d['param_delta']:.3f}",
        f"dropped={o['dropped']}",
        f"fused={o.get('fused_pairs')}",
        f"f16_ppl={e.get('f16',{}).get('ppl')}",
        f"q4_ppl={e.get('Q4_K_M',{}).get('ppl')}",
        f"q4_gen={e.get('Q4_K_M',{}).get('gen_tps')}",
        f"q4_pp={e.get('Q4_K_M',{}).get('prompt_tps')}",
    ))
base = json.load(open("/home/daniel/m0/runs/qwen35-4b-base/results.json"))
print("base", f"ppl16={base['eval']['f16']['ppl']}", f"q4_ppl={base['eval']['Q4_K_M']['ppl']}",
      f"q4_gen={base['eval']['Q4_K_M']['gen_tps']}", f"q4_pp={base['eval']['Q4_K_M']['prompt_tps']}")
for r in rows:
    print(*r)
