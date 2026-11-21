#!/usr/bin/env python3
"""Compares the last position of a prompt with an image between radiance (API, top-20 logprobs)
and HF (ref/vision_logits.py): KL over the renormalized HF top-20, max |Δ logprob| over the HF
top-5, whether the top-1 matches, and the 8 greedy tokens.

Usage: compare_vision_logits.py --ref DIR --url http://127.0.0.1:18095 --model M img1.png ...
"""
import argparse
import base64
import json
import math
import sys
import urllib.request
from pathlib import Path

import torch


def fetch(url, model, img, extra):
    b64 = base64.b64encode(img.read_bytes()).decode()
    b = {"model": model, "temperature": 0, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
        {"type": "text", "text": "Describe la imagen."}]}], **extra}
    import urllib.error
    try:
        r = urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(b).encode(),
                                                          headers={"Content-Type": "application/json"}), timeout=3600)
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code}: {e.read()[:400]!r}")
    return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--url", default="http://127.0.0.1:18095")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tok", type=Path, required=True, help="tokenizer.json, to turn text into ids")
    ap.add_argument("images", nargs="+", type=Path)
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(a.tok))
    # resumen.json: the name vision_logits.py wrote before the English rename
    summary = json.loads(next(p for p in (a.ref / "summary.json", a.ref / "resumen.json") if p.exists()).read_text())
    for img in a.images:
        ref = torch.load(a.ref / f"{img.stem}.pt")
        lp = ref["lp"].double()
        d = fetch(a.url, a.model, img, {"max_tokens": 1, "logprobs": True, "top_logprobs": 20})
        top = d["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        rad = {}
        for t in top:
            rad[int(t["id"])] = t["logprob"]
        hf20 = lp.topk(20)
        # KL(HF || radiance) over the HF top-20 that radiance also returns, renormalized
        com = [int(i) for i in hf20.indices if int(i) in rad]
        p = torch.tensor([lp[i] for i in com]).exp()
        q = torch.tensor([rad[i] for i in com], dtype=torch.double).exp()
        p, q = p / p.sum(), q / q.sum()
        kl = float((p * (p.log() - q.log())).sum())
        d5 = max(abs(float(lp[i]) - rad[i]) for i in [int(x) for x in hf20.indices[:5]] if i in rad)
        top1 = int(hf20.indices[0]) == max(rad, key=rad.get)
        g = fetch(a.url, a.model, img, {"max_tokens": 8, "logprobs": True})
        print(f"{img.stem}: tokens {d['usage']['prompt_tokens']} (HF {summary[img.stem]['tokens']}), top-20 in common "
              f"{len(com)}, KL {kl:.2e}, |Δlp| top-5 {d5:.3f}, top-1 equal {top1}; greedy equal "
              f"{[int(t['id']) for t in g['choices'][0]['logprobs']['content']] == summary[img.stem]['greedy']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
