#!/usr/bin/env python3
"""GPTQ calibration corpus (plan step 9) and its held-out chunk, with the Hy4 chat template.

Sources:
- `--pairs`, a private corpus (a JSONL of `{session, profile, prompt, completion}`): real
  conversations of tool-using agents, in Spanish, which is the traffic the model would serve. It is
  personal data and is not published. Each pair carries the history in its
  own format (`[user]`, `[assistant]`, `[assistant->tools] name({json})`, `[tool:name]`) and the
  answer; it is turned into messages and rendered with the HF template. One pair per session (the
  one with the longest history), so the same prefix is not repeated.
- `bigcodebench-hard` (`--bcb`, its directory in a Hugging Face cache): statement and canonical
  solution of each problem, as a user turn and an assistant turn: code, which the agent traffic
  hardly has.

Sessions are shuffled with a fixed seed and split before rendering: the held-out chunk
(`--heldout-tokens`) shares no session with the calibration one, so GPTQ is scored without the
text it was fitted to.

Usage: corpus_calib.py --snap <checkpoint> --pairs pairs.jsonl --out-dir DIR [--bcb DIR]
       [--tokens 2400000] [--heldout-tokens 200000]
Writes DIR/calib.jsonl and DIR/heldout.jsonl (`{"prompt", "source"}` per line, the radiance corpus
format) and DIR/summary.json.
"""
import argparse
import glob
import json
import os
import random
import re
import sys

BCB = os.path.expanduser("~/.cache/huggingface/hub/datasets--bigcode--bigcodebench-hard")
MARK = re.compile(r"^\[(user|assistant->tools|assistant|tool:[^\]\n]*)\] ?", re.M)
CALL = re.compile(r"^([A-Za-z_][\w.\-]*)\((.*)\)\s*$", re.S)


def to_messages(prompt, completion):
    """The history of a pair as chat messages. None if it cannot be parsed."""
    # the history comes truncated at the front (it starts in the middle of a tool output): that
    # loose chunk is dropped
    chunks = MARK.split(prompt)
    msgs, n_call = [], 0
    for i in range(1, len(chunks) - 1, 2):
        role, txt = chunks[i], chunks[i + 1].strip("\n")
        if role == "user":
            msgs.append({"role": "user", "content": txt})
        elif role == "assistant":
            msgs.append({"role": "assistant", "content": txt})
        elif role == "assistant->tools":
            calls = []
            for line in txt.split("\n"):
                line = line.strip()
                if not line:
                    continue
                m = CALL.match(line)
                if not m:
                    return None
                try:
                    args = json.loads(m.group(2)) if m.group(2).strip() else {}
                except json.JSONDecodeError:
                    args = None
                if not isinstance(args, dict):
                    # arguments truncated in the dump: they go as text
                    args = {"input": m.group(2)}
                n_call += 1
                calls.append({"id": f"call_{n_call}", "type": "function",
                              "function": {"name": m.group(1), "arguments": args}})
            if msgs and msgs[-1]["role"] == "assistant" and not msgs[-1].get("tool_calls"):
                msgs[-1]["tool_calls"] = calls
            else:
                msgs.append({"role": "assistant", "content": "", "tool_calls": calls})
        else:
            msgs.append({"role": "tool", "tool_call_id": f"call_{n_call}", "content": txt})
    if not msgs or msgs[0]["role"] not in ("user", "system"):
        # history truncated at the front: the template wants to start with a user turn
        msgs.insert(0, {"role": "user", "content": "(continúa)"})
    msgs.append({"role": "assistant", "content": completion})
    return msgs


def agent_sessions(pairs):
    best = {}
    with open(pairs) as f:
        for l in f:
            d = json.loads(l)
            s = d["session"]
            if s not in best or len(d["prompt"]) > len(best[s]["prompt"]):
                best[s] = d
    return [("agents:" + d["profile"], d) for d in best.values()]


def bcb_problems(bcb):
    # v0.1.0_hf and v0.1.1 are the same problems: only the latest
    fs = sorted(glob.glob(os.path.join(bcb, "snapshots", "*", "data", "v*.parquet")))[-1:]
    if not fs:
        return []
    import pyarrow.parquet as pq
    out = []
    for f in fs:
        t = pq.read_table(f).to_pylist()
        for r in t:
            if r.get("instruct_prompt") and r.get("canonical_solution"):
                out.append(("bigcodebench", r))
    return out


def render(tok, source, d):
    if source == "bigcodebench":
        code = d["code_prompt"] + d["canonical_solution"] if d.get("code_prompt") else d["canonical_solution"]
        msgs = [{"role": "user", "content": d["instruct_prompt"]},
                {"role": "assistant", "content": "```python\n" + code.strip("\n") + "\n```"}]
    else:
        msgs = to_messages(d["prompt"], d["completion"])
        if msgs is None:
            return None
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--pairs", required=True, help="the agent conversations, one pair per line")
    ap.add_argument("--bcb", default=BCB, help="bigcode/bigcodebench-hard in a Hugging Face cache")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tokens", type=int, default=2_400_000)
    ap.add_argument("--heldout-tokens", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=20261119)
    ap.add_argument("--max-doc-tokens", type=int, default=15000,
                    help="truncates each document (the calibrating server has 16K of context)")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.snap)
    os.makedirs(a.out_dir, exist_ok=True)
    if os.listdir(a.out_dir):
        sys.exit(f"{a.out_dir} is not empty")

    sessions = agent_sessions(a.pairs)
    code = bcb_problems(a.bcb)
    rnd = random.Random(a.seed)
    rnd.shuffle(sessions)
    rnd.shuffle(code)
    # ~1 in 8 tokens is code
    summary = {"seed": a.seed, "sources": {}, "dropped": 0}
    outputs = {"heldout": (a.heldout_tokens, []), "calib": (a.tokens, [])}
    it_s, it_c = iter(sessions), iter(code)
    for name in ("heldout", "calib"):
        target, rows = outputs[name]
        n = n_code = 0
        while n < target:
            use_code = n_code < n / 8
            source, d = next(it_c, (None, None)) if use_code else (None, None)
            if source is None:
                source, d = next(it_s, (None, None))
            if source is None:
                print(f"{name}: ran out of sources at {n} tokens", file=sys.stderr)
                break
            txt = render(tok, source, d)
            if txt is None:
                summary["dropped"] += 1
                continue
            ids = tok(txt, add_special_tokens=False)["input_ids"]
            if len(ids) > a.max_doc_tokens:
                ids = ids[:a.max_doc_tokens]
                txt = tok.decode(ids)
            k = len(ids)
            rows.append({"prompt": txt, "source": source, "tokens": k})
            n += k
            if source == "bigcodebench":
                n_code += k
            r = summary["sources"].setdefault(name, {})
            r[source] = r.get(source, 0) + k
        with open(os.path.join(a.out_dir, f"{name}.jsonl"), "w") as f:
            for x in rows:
                f.write(json.dumps({"prompt": x["prompt"], "source": x["source"]}, ensure_ascii=False) + "\n")
        summary[name] = {"docs": len(rows), "tokens": n,
                           "max_tokens_doc": max(x["tokens"] for x in rows)}
        print(f"{name}: {len(rows)} documents, {n} tokens", file=sys.stderr)
    json.dump(summary, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
