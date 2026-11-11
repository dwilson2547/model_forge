"""Task registry v1: coding (docker-graded unit tests), instruct checklists,
needle probes, and the loop-threshold temperature sweep.

Every grader is a pure function of the response text (plus task data): no
human judgment, no LLM-judge, fully reproducible. Coding tasks execute the
model's code ONLY inside docker (network disabled, memory-capped, non-root),
the local-model-builder pattern.
"""

import json
import re
import subprocess
import tempfile
from pathlib import Path

CODE_TEST_MAIN = '''
import json, sys
results = []
for name, fn in TESTS:
    try:
        fn()
        results.append({"name": name, "ok": True})
    except Exception as e:
        results.append({"name": name, "ok": False, "err": f"{type(e).__name__}: {e}"})
n_ok = sum(r["ok"] for r in results)
print(json.dumps(results))
sys.exit(0 if n_ok == len(results) else 1)
'''


def _extract_code(text):
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    return text  # no fence: grade raw (tests will fail loudly on chatter)


def _grade_code(task, response):
    code = _extract_code(response)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "solution.py").write_text(code)
        tests = (td / "run_tests.py")
        body = [
            "import json",
            f"from solution import {task['entry']}",
            "TESTS = []",
        ]
        for i, t in enumerate(task["tests"]):
            body.append("def _t%d():" % i)
            for line in t.splitlines():
                body.append("    " + line)
            body.append("TESTS.append(('t%d', _t%d))" % (i, i))
        body.append("TESTS.sort(key=lambda x: x[0])")
        tests.write_text("\n".join(body) + "\n" + CODE_TEST_MAIN)
        try:
            proc = subprocess.run(
                ["docker", "run", "--rm", "--network", "none", "--memory", "512m",
                 "--user", "1000:1000", "-e", "PYTHONDONTWRITEBYTECODE=1",
                 "-v", f"{td}:/t", "-w", "/t", "python:3.12-slim", "python", "run_tests.py"],
                capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            return 0.0, "timeout in sandbox"
        try:
            rows = json.loads(proc.stdout.strip().splitlines()[-1])
            ok = sum(1 for r in rows if r["ok"])
        except Exception:
            return 0.0, f"sandbox rc={proc.returncode}: {(proc.stdout or proc.stderr)[:220]!r}"
        return ok / len(rows), f"{ok}/{len(rows)} unit tests"


CODING = [
    {
        "id": "code-two-sum", "group": "coding", "entry": "two_sum",
        "prompt": "Write a Python function two_sum(nums, target) that returns the list of two "
                  "indices whose values sum to target, or an empty list if none. "
                  "Respond with ONLY a python code block.",
        "tests": [
            "assert two_sum([2,7,11,15], 9) == [0,1] or two_sum([2,7,11,15], 9) == [1,0]",
            "assert two_sum([3,2,4], 6) in ([1,2],[2,1])",
            "assert two_sum([1,2], 10) == []",
            "assert two_sum([5,5], 10) in ([0,1],[1,0])",
        ],
    },
    {
        "id": "code-prime", "group": "coding", "entry": "is_prime",
        "prompt": "Write a Python function is_prime(n) returning True if n is a prime number "
                  "(n >= 2), False otherwise. Respond with ONLY a python code block.",
        "tests": [
            "assert is_prime(2) is True",
            "assert is_prime(97) is True",
            "assert is_prime(1) is False",
            "assert is_prime(100) is False and is_prime(7919) is True",
        ],
    },
    {
        "id": "code-wordfreq", "group": "coding", "entry": "top_words",
        "prompt": "Write a Python function top_words(text, k) returning the k most frequent "
                  "whitespace-separated words (lowercased), ordered by count descending then "
                  "alphabetically. Respond with ONLY a python code block.",
        "tests": [
            "assert top_words('a b a c a b', 2) == ['a','b']",
            "assert top_words('x y x z', 3) == ['x','y','z'] or top_words('x y x z', 3) == ['x','z','y']",
            "assert top_words('', 3) == []",
        ],
    },
    {
        "id": "code-csv", "group": "coding", "entry": "parse_rows",
        "prompt": "Write a Python function parse_rows(csv_text) that takes lines of "
                  "'name,age' CSV (no header) and returns a list of (name, age:int) tuples. "
                  "Respond with ONLY a python code block.",
        "tests": [
            "assert parse_rows('ann,3') == [('ann', 3)]",
            "assert parse_rows('a,1\\nb,2') == [('a',1),('b',2)]",
        ],
    },
]

def _answer_part(t):
    """Qwen3.5 emits thinking traces even on raw completions, usually AFTER the
    answer but sometimes before. Instruct graders score the pre-thinking answer
    block — adherence modulo the thinking-tail artifact of raw mode."""
    s = t.strip()
    for marker in ("\nThinking", "Thinking Process:", "Think through"):
        if marker in s:
            s = s.split(marker)[0]
    lines = s.splitlines()
    while lines and (not lines[-1].strip()
                     or lines[-1].strip() in {"-", "- ", "—", "```", "***"}):
        lines.pop()
    return "\n".join(lines).rstrip()


INSTRUCT = [
    {
        "id": "inst-bullets", "group": "instruct",
        "prompt": "Follow this instruction EXACTLY: reply with exactly three bullet points, "
                  "each starting with '- ', about rivers. No other text.",
        "grade": lambda t: _g_bullets(t),
    },
    {
        "id": "inst-wordcount", "group": "instruct",
        "prompt": "Follow this instruction EXACTLY: answer in exactly 10 words. "
                  "The question: why is the sky blue?",
        "grade": lambda t: _g_words(t),
    },
    {
        "id": "inst-constraints", "group": "instruct",
        "prompt": "Follow this instruction EXACTLY: write one paragraph about coffee that "
                  "includes the word 'arabica' at least twice and contains no capital letters "
                  "at all.",
        "grade": lambda t: _g_arabica(t),
    },
    {
        "id": "inst-json", "group": "instruct",
        "prompt": "Respond with ONLY a JSON object (no other text) with keys 'ok' (boolean) "
                  "and 'n' (integer). Any values you like.",
        "grade": lambda t: _json_keys_ok(_answer_part(t), {"ok", "n"}),
    },
    {
        "id": "inst-format", "group": "instruct",
        "prompt": "Follow this instruction EXACTLY: write exactly two sentences about the "
                  "ocean, each ending with a period. Nothing else.",
        "grade": lambda t: _two_sentences(_answer_part(t)),
    },
]




def _g_bullets(t):
    a = _answer_part(t)
    lines = [l for l in a.splitlines() if l.strip()]
    ok = len(lines) == 3 and all(l.strip().startswith("- ") for l in lines)
    return (1.0 if ok else 0.0, "3 bullets only")


def _g_words(t):
    a = _answer_part(t)
    return (1.0 if len(a.split()) == 10 else 0.0, "== 10 words")


def _g_arabica(t):
    a = _answer_part(t)
    return (1.0 if a.lower().count("arabica") >= 2 and a == a.lower()
            and len(a.strip()) > 0 else 0.0, "arabica x2, lowercase")


def _json_keys_ok(t, keys):
    s = t.strip()
    try:
        obj = json.loads(s[re.search(r"\{", s).start():] if not s.startswith("{") else s)
    except Exception:
        return 0.0, "not valid json"
    return (1.0 if keys <= set(obj) else 0.0), f"keys {keys}"


def _two_sentences(t):
    a = " ".join(x.strip() for x in _answer_part(t).splitlines() if x.strip())
    return (1.0 if a.count(".") == 2 and a.endswith(".") and len(a) > 10 else 0.0), \
        "exactly 2 periods"


_NEEDLE_FILLER = (
    "The regional infrastructure review board publishes quarterly findings on the "
    "operational health of distributed systems. Their reports cover maintenance windows, "
    "hardware replacement cycles, and coordination between teams. Each report is archived "
    "before the next quarter begins. "
) * 120  # ~ 30k chars ≈ 7.5k tokens

NEEDLE = []
for depth, needle_at in [(0.15, "15%"), (0.55, "55%"), (0.85, "85%")]:
    doc = _NEEDLE_FILLER[: int(len(_NEEDLE_FILLER) * depth)]
    doc += " The magic number for this quarter is 64701, as signed by the coordinator. "
    doc += _NEEDLE_FILLER[int(len(_NEEDLE_FILLER) * depth):]
    NEEDLE.append({
        "id": f"needle-{needle_at}", "group": "needle",
        "prompt": doc + "\n\nQuestion: What is the magic number for this quarter? "
                        "Answer with just the number.",
        "grade": (lambda t: (1.0 if "64701" in t else 0.0, "contains 64701")),
        "max_tokens": 256,
    })


LOOP_PROMPT = ("Explain the difference between TCP and UDP, then recommend which to use "
               "for a video call and why. Take your time and reason carefully.")
LOOP_TEMPS = [0.0, 0.15, 0.3, 0.5, 0.7]


def _detect_loop(text, n=10, threshold=3):
    """A word-level n-gram appearing >= threshold times at different positions = a loop."""
    words = text.split()
    if len(words) < n * threshold:
        return False
    grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    from collections import Counter
    common, count = Counter(grams).most_common(1)[0]
    return count >= threshold


def loop_threshold(server, max_tokens=192, chat=False):
    """Highest temperature at which the model does NOT loop, plus per-temp results."""
    rows = []
    for temp in LOOP_TEMPS:
        if chat:
            out = server.chat(LOOP_PROMPT, max_tokens=max_tokens,
                              temperature=temp, top_k=40)
        else:
            out = server.completion(LOOP_PROMPT, max_tokens=max_tokens,
                                    temperature=temp, top_k=40)
        looped = _detect_loop(out["text"])
        rows.append({"temperature": temp, "looped": looped})
    thresh = max((r["temperature"] for r in rows if not r["looped"]), default=None)
    return thresh, rows


ALL_TASKS = CODING + INSTRUCT + NEEDLE


def run_task(server, task, chat=False):
    # Coding needs 2048: raw completions think hundreds of tokens before/after the
    # code; 512 truncated blocks mid-statement (SyntaxError) across every arm.
    budget = 2048 if task["group"] == "coding" else task.get("max_tokens", 256)
    if chat:
        resp = server.chat(task["prompt"], max_tokens=budget)
        if task["group"] == "instruct":
            # Chat-mode thinking is elaborate planning (hundreds of tokens) and
            # PRECEDES the answer; at 256 the answer never arrives (stop=limit).
            resp = server.chat(task["prompt"], max_tokens=1024)
        if task["group"] != "coding":
            # The server already separated thinking from answer: grade the answer
            # part when it exists (raw _answer_part() expects thinking-TAILS).
            c = (resp.get("content") or "").strip()
            if c:
                resp = dict(resp, text=resp["content"])
    else:
        resp = server.completion(task["prompt"], max_tokens=budget)
    if task["group"] == "coding":
        score, detail = _grade_code(task, resp["text"])
        if resp.get("stop_type") in ("length", "limit"):  # llama.cpp calls it "limit"
            detail = f"BUDGET-EXHAUSTED; {detail}"
        return {"id": task["id"], "group": task["group"], "score": score,
                "detail": detail, "stop_type": resp.get("stop_type"),
                "text_preview": resp["text"][-400:]}
    score, detail = task["grade"](resp["text"])
    return {"id": task["id"], "group": task["group"], "score": score, "detail": detail}
