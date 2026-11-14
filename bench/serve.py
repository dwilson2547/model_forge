"""llama-server lifecycle for suite runs: start, wait for health, completions, stop.

Suite methodology (M0-derived): server owns the box with -t 32 / -tb 96; raw
/completion (no chat template) so grading is comparable across candidate
families regardless of template availability; temperature 0 + fixed seed unless
a task explicitly sweeps temperature.

Server PID is surfaced so the runner can read peak RSS from /proc.
"""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from surgery import paths  # noqa: E402


class LlamaServer:
    def __init__(self, model, port=None, ctx=32768, threads=32, threads_batch=96, parallel=None):
        import random
        self.model = str(model)
        self.port = port or random.randint(8100, 8990)
        self.ctx = ctx
        self.threads = threads
        self.threads_batch = threads_batch
        self.parallel = parallel  # None = llama-server's default; board v2 runs N slots, ctx = N x per-slot
        self.proc = None

    def _wait_ready(self, timeout=600):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc and self.proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited rc={self.proc.returncode} "
                    "(port collision? another server holding it?)")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2).read()
                status = open(f"/proc/{self.proc.pid}/status").read()
                if "VmPeak" not in status:
                    raise RuntimeError("/health answered but our process looks dead "
                                       "(stale server on this port?)")
                return True
            except RuntimeError:
                raise
            except Exception:
                time.sleep(2)
        raise TimeoutError(f"llama-server not healthy after {timeout}s")

    def __enter__(self):
        # FORGE_LLAMA picks the build (paths.LLAMA); FORGE_NGL=99 offloads to a GPU, as in
        # surgery/evaluate.py. A GPU board is its own surface: compare rows from one machine.
        binpath = paths.LLAMA / "build" / "bin" / "llama-server"
        ngl = os.environ.get("FORGE_NGL", "")
        self.proc = subprocess.Popen(
            [str(binpath), "-m", self.model,
             "--port", str(self.port), "-c", str(self.ctx),
             "-t", str(self.threads), "-tb", str(self.threads_batch),
             "--metrics"] + (["-ngl", ngl] if ngl else [])
            + (["-np", str(self.parallel), "--no-kv-unified"] if self.parallel else []),
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
        )
        self._wait_ready()
        return self

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        return False

    def peak_rss_gib(self):
        try:
            for line in open(f"/proc/{self.proc.pid}/status"):
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 2**20
        except Exception:
            pass
        return None

    def chat(self, user, max_tokens=256, temperature=0.0, top_k=1, top_p=1.0, seed=42):
        """One /v1/chat/completions turn (server applies the GGUF's chat template).
        stop_type normalized to the raw endpoint's vocabulary: eos / limit."""
        body = {"messages": [{"role": "user", "content": user}],
                "n_predict": max_tokens, "temperature": temperature,
                "top_k": top_k, "top_p": top_p, "seed": seed, "cache_prompt": False}
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            out = json.loads(r.read())
        choice = out["choices"][0]
        msg = choice["message"]
        # llama.cpp splits "thinking" into reasoning_content and can leave content
        # EMPTY for models that think on every turn (this family does).
        text = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
        return {"text": text, "content": msg.get("content") or "",
                "reasoning": msg.get("reasoning_content") or "",
                "stop_type": {"stop": "eos", "length": "limit"}.get(
                    choice.get("finish_reason"), choice.get("finish_reason")),
                "tokens": out.get("usage", {}).get("completion_tokens", 0)}

    def completion(self, prompt, max_tokens=256, temperature=0.0, top_k=1,
                   stop=None, stream_first_token=False):
        """One completion. stream_first_token=True returns TTFT seconds alongside text
        (consumes the stream until content arrives)."""
        body = {"prompt": prompt, "n_predict": max_tokens,
                "temperature": temperature, "top_k": top_k, "seed": 42,
                "cache_prompt": False}
        if stop:
            body["stop"] = stop
        if stream_first_token:
            body["stream"] = True
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/completion",
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        if not stream_first_token:
            with urllib.request.urlopen(req, timeout=600) as r:
                out = json.loads(r.read())
            return {"text": out.get("content", ""), "tokens": out.get("tokens_predicted", 0),
                    "stop_type": out.get("stop_type", "stop")}
        t0 = time.time()
        ttft = None
        text = []
        with urllib.request.urlopen(req, timeout=600) as r:
            for raw in r:
                line = raw.decode(errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if ttft is None and chunk.get("content"):
                    ttft = time.time() - t0
                if chunk.get("content"):
                    text.append(chunk["content"])
                if chunk.get("stop"):
                    break
        return {"text": "".join(text), "tokens": 0, "ttft": ttft}
