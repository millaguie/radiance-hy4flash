#!/usr/bin/env python3
"""The `reasoning_effort` map Hy4 is served with (deploy/example-tp2.yaml), against a live server.

The template only accepts `no_think` and `high` (Tencent dropped `max` from it on 2026-11-19);
`reasoning-effort-map` translates the OpenAI levels, and `max`, before rendering. rad-chatparse does not apply the map (it belongs to the server), so this
test sends each level in the body of /v1/chat/completions and compares the prompt tokens with those
of the HF template at the target level, and checks that all levels that go to the same target give
the same logprobs of the first token, and `high` and `no_think` different ones (if they matched, the
test could not tell them apart, and it says so). It also sends an explicit `"reasoning_effort": null`, which the
template rejects if it gets it.

Usage: test_reasoning_map.py --snap <checkpoint> --url http://127.0.0.1:18093
       (server started with override-chat-template and the reasoning-effort-map from deploy/)
"""
import argparse
import json
import sys
import urllib.error
import urllib.request

ABSENT = object()
TARGET = {ABSENT: "high", None: "high", "low": "high", "medium": "high", "high": "high", "xhigh": "high", "max": "high",
           "minimal": "no_think", "none": "no_think", "no_think": "no_think"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--url", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.snap)
    model = json.loads(urllib.request.urlopen(a.url + "/v1/models").read())["data"][0]["id"]
    msgs = [{"role": "user", "content": "Hola"}]
    failures = 0
    by_target = {}
    for level, dest in TARGET.items():
        hf = len(tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                             reasoning_effort=dest), add_special_tokens=False)["input_ids"])
        b = {"model": model, "messages": msgs, "max_tokens": 1, "temperature": 0, "logprobs": True,
             "top_logprobs": 5}
        if level is not ABSENT:
            b["reasoning_effort"] = level
        lp = None
        try:
            r = urllib.request.urlopen(urllib.request.Request(a.url + "/v1/chat/completions", data=json.dumps(b).encode(),
                                                              headers={"Content-Type": "application/json"}), timeout=600)
            j = json.loads(r.read())
            rad = j["usage"]["prompt_tokens"]
            lp = tuple((x["token"], round(x["logprob"], 4))
                       for x in j["choices"][0]["logprobs"]["content"][0]["top_logprobs"])
        except urllib.error.HTTPError as e:
            rad = f"HTTP {e.code} {e.read()[:200]!r}"
        ok = rad == hf
        if lp is not None:
            ok = ok and by_target.setdefault(dest, lp) == lp
        failures += not ok
        name = "(no field)" if level is ABSENT else str(level)
        print(f"{'ok   ' if ok else 'FAIL '} {name:11s} -> {dest:9s} radiance {rad} HF {hf}")
    if by_target.get("high") == by_target.get("no_think"):
        print("FAIL high and no_think give the same logprobs: the test cannot tell those targets apart")
        failures += 1
    print("all good" if not failures else f"{failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
