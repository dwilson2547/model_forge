# qwen35-4b arms: what each one is

One line per trained arm. The card beside it has the recipe, floor and verdict; the same card sits as
`README.md` in the arm's archive folder on the cold share (`docs/storage.md`).

Unrepaired wounds, f16 (wikitext / fineweb / code): base 9.33 / 10.44 / 3.17; drop2 [24,28] 12.82 / 13.55 / 3.57;
drop2 [25,29] 12.823 / 13.954 / 3.610. The best floors are 2529-fresh1e (QLoRA) and full-repair-2529 lr3e-6 / lr1e-6.

| experiment | arm | what | floor (wiki / fw / code) or board | weights |
|---|---|---|---|---|
| drop2-repair | [wikirep](wikirep.md) | The first repair of the drop2 wound, on wikitext only | 8.163 / 19.446 / 3.898 | archive |
| drop2-repair | [fw](fw.md) | drop2 repaired on fineweb-edu only | 10.22 / 12.466 / 3.578 | archive |
| drop2-repair | [codefw](codefw.md) | drop2 repaired on the head of the mixed file | 9.97 / 12.341 / 3.282 | archive |
| drop2-repair | [mixfull](mixfull.md) | drop2 repaired on the whole mixed file, one epoch | 8.423 / 12.387 / 3.363 | archive |
| drop2-repair | [r8](r8.md) | The codefw recipe with LoRA rank 8 instead of 16 (alpha/r held at 2) | 10.015 / 12.41 / 3.377 | archive |
| drop2-repair | [r32](r32.md) | The codefw recipe with LoRA rank 32 instead of 16 (alpha/r held at 2) | 10.624 / 12.992 / 3.568 | archive |
| drop2-repair | [r64](r64.md) | The codefw recipe with LoRA rank 64 instead of 16 (alpha/r held at 2) | 11.483 / 13.733 / 3.67 | archive |
| drop2-repair | [lr1e4](lr1e4.md) | The codefw recipe at r16 with lr 1e-4 (half the default then) | 9.873 / 12.217 / 3.333 | archive |
| drop2-repair | [lr4e4](lr4e4.md) | The codefw recipe at r16 with lr 4e-4 | 11.637 / 13.735 / 3.722 | archive |
| drop2-repair | [r2fix-lr1e4](r2fix-lr1e4.md) | First run after review round 2's trainer fixes | 8.66 / 12.715 / 3.626 | archive |
| drop2-repair | [iso-classic](iso-classic.md) | Isolation pair, classic targets | 8.365 / 12.104 / 3.506 | archive |
| drop2-repair | [iso-all](iso-all.md) | Isolation pair, all targets (mixers included) | 8.359 / 12.221 / 3.509 | archive |
| drop2-repair | [dose4e](dose4e.md) | iso-all repeated for 4 epochs over the same 12M-char draw | 10.284 / 15.386 / 4.079 | deleted |
| drop2-repair | [2529-pass1](2529-pass1.md) | The iso-all protocol on the [25,29] wound instead of [24,28] | 8.031 / 11.894 / 3.485 | archive |
| drop2-repair | [2529-fresh1e](2529-fresh1e.md) | The [25,29] slice repaired on the WHOLE mixed file for one epoch | 7.785 / 11.556 / 3.186 | archive |
| drop2-repair | [sftrep1](sftrep1.md) | First SFT-format repair | chat board 0.0 (coding 0, instruct 0, needle 0) | archive |
| drop2-repair | [sftrep2](sftrep2.md) | SFT v2 | chat board 0.333 (coding 0, instruct 0, needle 1.0) | archive |
| drop2-repair | [sftrep3](sftrep3.md) | SFT v3 | chat board 0.733 (coding 1.0, instruct 0.2, needle 1.0) | archive |
| full-repair-2529 | [lr1e-5](lr1e-5.md) | Full-weight repair (no LoRA) of the [25,29] slice at lr 1e-5, on a rented A100 | 8.085 / 12.082 / 3.542 | archive |
| full-repair-2529 | [lr3e-6](lr3e-6.md) | Full-weight repair (no LoRA) of the [25,29] slice at lr 3e-6, on a rented A100 | 7.631 / 11.362 / 3.259 | archive |
| full-repair-2529 | [lr1e-6](lr1e-6.md) | Full-weight repair (no LoRA) of the [25,29] slice at lr 1e-6, on a rented A100 | 7.685 / 11.39 / 3.182 | archive |

Not arms: `drop2-repair/calib`, `calib-r2fixes` are `--calibrate` smoke tests of the trainer;
`cloud-session1/run0-parity` is the cloud trainer-parity adapter (§0j.9 run 0b).
