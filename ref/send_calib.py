#!/usr/bin/env python3
"""Pushes a corpus (`{"prompt", ...}` per line) through the prefill of a calibrating radiance.

Each document goes to /v1/completions with max_tokens=1: the GPTQ tap (RADIANCE_CALIB_DIR) adds up
the Grams of what goes through the prefill and writes when it reaches RADIANCE_CALIB_TOKENS. It
stops when it has sent `--tokens` of prompt (above the tap target, because of step rounding) or
the corpus runs out. With `--log` it also watches the server log and stops as soon as the tap says
it has written.

Usage: send_calib.py --corpus calib.jsonl --tokens N [--url http://127.0.0.1:18093] [--parallel 1]
     [--start 0] [--log server.log]
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.error
import urllib.request


def send(url, model, prompt):
    """Prompt tokens that went through the prefill; 0 if the server rejects the document (longer
    than its max-model-len: the calibrating server runs at 8K to leave VRAM for the Gram)."""
    b = {"model": model, "prompt": prompt, "max_tokens": 1, "temperature": 0}
    try:
        r = urllib.request.urlopen(urllib.request.Request(url + "/v1/completions", data=json.dumps(b).encode(),
                                                          headers={"Content-Type": "application/json"}),
                                   timeout=7200)
    except urllib.error.HTTPError as e:
        if e.code == 400:
            return 0
        raise
    return json.loads(r.read())["usage"]["prompt_tokens"]


def written(log):
    if not log or not os.path.exists(log):
        return False
    with open(log, errors="replace") as f:
        return any("calibration wrote" in l for l in f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--url", default="http://127.0.0.1:18093")
    ap.add_argument("--model")
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--start", type=int, default=0, help="document to start from")
    ap.add_argument("--log")
    a = ap.parse_args()
    model = a.model or json.loads(urllib.request.urlopen(a.url + "/v1/models").read())["data"][0]["id"]
    docs = [json.loads(l)["prompt"] for l in open(a.corpus)][a.start:]
    t0, n, done_docs = time.time(), 0, 0
    with cf.ThreadPoolExecutor(a.parallel) as ex:
        pend = set()
        it = iter(docs)
        while True:
            while len(pend) < a.parallel and n < a.tokens and not written(a.log):
                p = next(it, None)
                if p is None:
                    break
                pend.add(ex.submit(send, a.url, model, p))
            if not pend:
                break
            done, pend = cf.wait(pend, return_when=cf.FIRST_COMPLETED)
            for f in done:
                n += f.result()
                done_docs += 1
            dt = time.time() - t0
            print(f"{done_docs} documents, {n} tokens, {n / dt:.0f} tok/s, {dt / 60:.1f} min", flush=True)
            if written(a.log):
                print("the tap has written", flush=True)
    print(f"end: {done_docs} documents (up to {a.start + done_docs}), {n} tokens in "
          f"{(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
