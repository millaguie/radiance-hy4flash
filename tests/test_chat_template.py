#!/usr/bin/env python3
"""data/chat_template.jinja (the Hy4 template without `.format()`, which radiance's Jinja lacks)
against the original rendered by HF, conversation by conversation, and the parsing of replies.

Renders each conversation with transformers' `apply_chat_template` (original template) and with
`rad-chatparse` (the template from data/ over a Hy4 container), and requires the same prompt. Then
it feeds model replies through rad-chatparse and requires the expected reasoning, text and tool
calls.

Usage: test_chat_template.py --snap <HF checkpoint> --rad <Hy4 .rad container>
       [--chatparse .../rad-chatparse] [--home DIRS]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "..", "data", "chat_template.jinja")

TOOLS = [{"type": "function", "function": {
    "name": "tiempo", "description": "Tiempo actual en una ciudad (año, mañana, «comillas», 天气)",
    "parameters": {"type": "object", "properties": {"ciudad": {"type": "string"},
                                                    "dias": {"type": "integer"}},
                   "required": ["ciudad"]}}}]

CALL = {"id": "call_1", "type": "function",
           "function": {"name": "tiempo", "arguments": json.dumps({"ciudad": "Bilbao", "dias": 2})}}
CALL2 = {"id": "call_2", "type": "function",
            "function": {"name": "tiempo", "arguments": json.dumps({"ciudad": "A Coruña"}, ensure_ascii=False)}}

CASES = {
    "user_only": {"messages": [{"role": "user", "content": "Hola"}]},
    "system": {"messages": [{"role": "system", "content": "Responde en una frase."},
                             {"role": "user", "content": "¿Qué es un átomo?"}]},
    "multi_turn": {"messages": [
        {"role": "user", "content": "Hola"},
        {"role": "assistant", "content": "¡Hola! ¿En qué te ayudo?", "reasoning_content": "Saludo."},
        {"role": "user", "content": "Dime un número primo."}]},
    "text_parts": {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "Primera parte. "}, {"type": "text", "text": "Segunda parte."}]}],
        "join_parts": True},
    "tools": {"messages": [{"role": "user", "content": "¿Qué tiempo hace en Bilbao?"}],
                     "tools": TOOLS},
    "tool_result": {"messages": [
        {"role": "user", "content": "¿Qué tiempo hace en Bilbao?"},
        {"role": "assistant", "content": "", "reasoning_content": "Llamo a tiempo.", "tool_calls": [CALL]},
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": 14, \"cielo\": \"lluvia\"}"}],
        "tools": TOOLS},
    "no_think": {"messages": [{"role": "user", "content": "Hola"}],
                   "kwargs": {"reasoning_effort": "no_think"}},
    # the level in the request body, not through kwargs (the OpenAI ones are translated by the
    # server's map, which rad-chatparse does not apply: tests/test_reasoning_map.py)
    "high_level_in_request": {"messages": [{"role": "user", "content": "Hola"}], "effort": "high"},
    "no_think_level_in_request_with_tools": {
        "messages": [{"role": "user", "content": "¿Qué tiempo hace en Bilbao?"}], "tools": TOOLS,
        "effort": "no_think"},
    "system_and_tools": {"messages": [
        {"role": "system", "content": "Eres un agente. Usa herramientas."},
        {"role": "user", "content": "¿Qué tiempo hace en Bilbao?"},
        {"role": "assistant", "content": "", "reasoning_content": "Llamo.", "tool_calls": [CALL]},
        {"role": "tool", "tool_call_id": "call_1", "content": "14 grados"}], "tools": TOOLS},
    "empty_system_and_tools": {"messages": [
        {"role": "system", "content": ""}, {"role": "user", "content": "Hola"}], "tools": TOOLS},
    # two parallel calls, results in reverse order, and the conversation continues after them
    "parallel_calls": {"messages": [
        {"role": "user", "content": "Tiempo en Bilbao y en A Coruña"},
        {"role": "assistant", "content": "", "reasoning_content": "Dos llamadas.", "tool_calls": [CALL, CALL2]},
        {"role": "tool", "tool_call_id": "call_2", "content": "lluvia, 12 °C"},
        {"role": "tool", "tool_call_id": "call_1", "content": "sol, 14 °C"},
        {"role": "assistant", "content": "En Bilbao sol y en A Coruña lluvia."},
        {"role": "user", "content": "¿Y mañana?"}], "tools": TOOLS},
    "mismatched_ids": {"messages": [
        {"role": "user", "content": "Tiempo en Bilbao y en A Coruña"},
        {"role": "assistant", "content": "", "tool_calls": [CALL, CALL2]},
        {"role": "tool", "tool_call_id": "otro", "content": "a"},
        {"role": "tool", "tool_call_id": "call_1", "content": "b"}], "tools": TOOLS},
}

# model reply -> what the server must return
REPLIES = [
    ("Bien.</think:opensource>Hola, ¿qué tal?", {"reasoning": "Bien.", "content": "Hola, ¿qué tal?"}),
    ("Llamo a la herramienta.</think:opensource><tool_calls:opensource><tool_call:opensource>tiempo"
     "<arg_key:opensource>ciudad</arg_key:opensource><arg_value:opensource>Bilbao</arg_value:opensource>"
     "<arg_key:opensource>dias</arg_key:opensource><arg_value:opensource>2</arg_value:opensource>"
     "</tool_call:opensource></tool_calls:opensource>",
     {"reasoning": "Llamo a la herramienta.", "tool": [("tiempo", {"ciudad": "Bilbao", "dias": 2})]}),
    ("Dos.</think:opensource><tool_calls:opensource><tool_call:opensource>tiempo"
     "<arg_key:opensource>ciudad</arg_key:opensource><arg_value:opensource>Bilbao</arg_value:opensource>"
     "</tool_call:opensource><tool_call:opensource>tiempo"
     "<arg_key:opensource>ciudad</arg_key:opensource><arg_value:opensource>A Coruña</arg_value:opensource>"
     "</tool_call:opensource></tool_calls:opensource>",
     {"reasoning": "Dos.", "tool": [("tiempo", {"ciudad": "Bilbao"}), ("tiempo", {"ciudad": "A Coruña"})]}),
]
# with no_think the prompt already closes the reasoning: the reply arrives without tags
REPLIES_NO_THINK = [
    ("Hola, ¿qué tal?", {"reasoning": None, "content": "Hola, ¿qué tal?"}),
]


def hf_render(tok, case):
    """The HF template expects a call's arguments as an object (it does `.items()`); the OpenAI API
    sends them as JSON text and radiance converts them before rendering."""
    msgs = json.loads(json.dumps(case["messages"]))
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
    if case.get("join_parts"):
        # radiance (llama.cpp's chat.cpp) passes multi-part content as one text, joined with
        # "\n", to any template that does not index the list; HF joins it with no separator
        for m in msgs:
            if isinstance(m["content"], list):
                m["content"] = "\n".join(x["text"] for x in m["content"])
    kw = dict(case.get("kwargs", {}))
    if case.get("effort"):
        kw["reasoning_effort"] = case["effort"]
    return tok.apply_chat_template(msgs, tools=case.get("tools"), tokenize=False,
                                   add_generation_prompt=True, **kw)


def chatparse(a, case, replies=()):
    req = {"model": "m", "messages": case["messages"]}
    if case.get("tools"):
        req["tools"] = case["tools"]
    if case.get("effort"):
        req["reasoning_effort"] = case["effort"]
    with tempfile.TemporaryDirectory() as d:
        rq = os.path.join(d, "req.json")
        json.dump(req, open(rq, "w"))
        cmd = [a.chatparse, "--home", a.home, "--request", rq, "--rad", a.rad]
        if not a.no_override:
            cmd += ["--override-chat-template", TEMPLATE]
        if case.get("kwargs"):
            cmd += ["--chat-template-kwargs", json.dumps(case["kwargs"])]
        for i, r in enumerate(replies):
            f = os.path.join(d, f"r{i}.txt")
            open(f, "w").write(r)
            cmd += ["--reply", f]
        p = subprocess.run(cmd, capture_output=True, text=True)
        lin = [l for l in p.stdout.splitlines() if l.startswith("{")]
        if not lin:
            raise RuntimeError(f"rad-chatparse gave no JSON (rc {p.returncode}): {p.stderr[-2000:]}")
        return json.loads(lin[-1]), p.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--rad", required=True)
    ap.add_argument("--chatparse", default=os.environ.get("RAD_CHATPARSE") or shutil.which("rad-chatparse") or "rad-chatparse",
                    help="radiance's rad-chatparse (default: $RAD_CHATPARSE, or the one on PATH)")
    ap.add_argument("--home", default=":".join([os.path.join(HERE, "..", "build", "radiance_home"),
                                                os.path.join(HERE, "..", "prefix", "share", "radiance")]))
    ap.add_argument("--no-override", action="store_true",
                    help="use the template inside the container (checks that convert-hip.sh put it there)")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.snap)
    failures = 0
    # data/ is the original with the 22 .format() calls rewritten, and nothing else
    import re
    orig = open(os.path.join(a.snap, "chat_template.jinja"), encoding="utf-8").read()
    deriv, n = re.subn(r"'([^'{}]*)\{\}([^'{}]*)'\.format\(HYTK\)", r"'\1' ~ HYTK ~ '\2'", orig)
    ok = n == 22 and deriv == open(TEMPLATE, encoding="utf-8").read()
    failures += not ok
    print(f"{'ok   ' if ok else 'FAIL '} data/chat_template.jinja = original with {n} .format() rewritten")
    # the example config pins the hash of the copy it serves (deploy/run-hy4.sh checks it there)
    import hashlib
    h = hashlib.sha256(open(TEMPLATE, "rb").read()).hexdigest()
    cfg = open(os.path.join(HERE, "..", "deploy", "example-tp2.yaml")).read()
    pinned = re.search(r"^# template-sha256: *([0-9a-f]+)", cfg, re.M)
    ok = bool(pinned) and pinned.group(1) == h
    failures += not ok
    print(f"{'ok   ' if ok else 'FAIL '} deploy/example-tp2.yaml requires the sha256 of data/chat_template.jinja ({h[:12]})")
    for name, case in CASES.items():
        hf = hf_render(tok, case)
        rad, _ = chatparse(a, case)
        ok = rad["prompt"] == hf
        failures += not ok
        print(f"{'ok   ' if ok else 'FAIL '} render {name}")
        if not ok:
            i = next((k for k in range(min(len(hf), len(rad["prompt"]))) if hf[k] != rad["prompt"][k]),
                     min(len(hf), len(rad["prompt"])))
            print(f"      HF : {hf[max(0, i - 60):i + 80]!r}\n      rad: {rad['prompt'][max(0, i - 60):i + 80]!r}")
    for case, resp in (("tools", REPLIES), ("no_think_level_in_request_with_tools", REPLIES_NO_THINK)):
        failures += read_replies(a, CASES[case], resp)
    print("all good" if not failures else f"{failures} failures")
    return 1 if failures else 0


def read_replies(a, case, replies):
    failures = 0
    rad, rc = chatparse(a, case, [r for r, _ in replies])
    if rc != 0:
        print(f"FAIL rad-chatparse exited with {rc}: the stream and the full parse disagree, or the grammar "
              "rejects a token")
        failures += 1
    if len(rad["replies"]) != len(replies):
        print(f"FAIL rad-chatparse returned {len(rad['replies'])} replies out of {len(replies)}")
        failures += 1
    for (reply_text, expect), got in zip(replies, rad["replies"]):
        m = got["message"]
        ok = (m.get("reasoning_content") or None) == expect["reasoning"] and got.get("stream_agrees", False)
        if "content" in expect:
            ok = ok and m.get("content") == expect["content"] and not m.get("tool_calls")
        if "tool" in expect:
            tc = m.get("tool_calls") or []
            ok = ok and [(t["function"]["name"], json.loads(t["function"]["arguments"])) for t in tc] == expect["tool"] \
                and not m.get("content")
        failures += not ok
        print(f"{'ok   ' if ok else 'FAIL '} reply {reply_text[:40]!r}")
        if not ok:
            print(f"      {json.dumps(m, ensure_ascii=False)[:600]}")
    return failures


if __name__ == "__main__":
    sys.exit(main())
