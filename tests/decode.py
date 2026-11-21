#!/usr/bin/env python3
"""Decode against prefill on a radiance server with the Hy4 model (plan steps 4 and 6b).

Generates N tokens greedily and, for each k, sends prompt + generated[:k] as a single prefill: the
first token must be generated[k], with the same logprob. This tests the decode steps (indexer
groups that close during decode, the selection with T=1, the window and the Engram state) against
the prefill path, which is already compared with the reference.

Usage: decode.py PROMPT N [--drop K] [--url URL] [--model M]
  PROMPT  a key of ref/prompts.json; --drop K drops its last K tokens (largo with K=1 leaves an
          open group tail that closes during decode, with the selection active).
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("prompt")
ap.add_argument("n", type=int)
ap.add_argument("--drop", type=int, default=0)
ap.add_argument("--url", default="http://127.0.0.1:18093/v1/completions")
ap.add_argument("--model", default="recorte3")
ap.add_argument("--prompts", default=str(Path(__file__).resolve().parents[1] / "ref/prompts.json"))
a = ap.parse_args()
P = json.load(open(a.prompts))["prompts"][a.prompt]
if a.drop:
    P = P[:-a.drop]


def fetch(prompt, mt):
    body = {"model": a.model, "prompt": prompt, "max_tokens": mt, "temperature": 0, "logprobs": 5,
            "ignore_eos": True}
    r = urllib.request.urlopen(urllib.request.Request(a.url, data=json.dumps(body).encode(),
                                                      headers={"Content-Type": "application/json"}),
                               timeout=7200)
    return json.loads(r.read())["choices"][0]["logprobs"]["content"]


print(f"prompt {a.prompt}, {len(P)} tokens ({len(P) % 4} in the tail of the open group)")
gen = fetch(P, a.n)
ids = [c["id"] for c in gen]
print("generated", ids)
worst, bad = 0.0, 0
for k in range(1, a.n):
    pre = fetch(P + ids[:k], 1)[0]
    lp = [x["logprob"] for x in pre["top_logprobs"] if x["id"] == ids[k]]
    d = abs(gen[k]["logprob"] - lp[0]) if lp else 99.0
    worst, bad = max(worst, d), bad + (pre["id"] != ids[k])
    print(k, ids[k], pre["id"], "same" if pre["id"] == ids[k] else "DIFFERENT", f"dlogprob {d:.4f}")
print(f"different {bad}, worst dlogprob {worst:.4f}")
sys.exit(1 if bad else 0)
