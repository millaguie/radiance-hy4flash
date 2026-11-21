#!/usr/bin/env python3
"""Generates the 4 fixed reference prompts (plan step 2) as token ids.

corto  < 512 tokens, natural Spanish text: its top-1 accuracy proves the reference works.
medio  1,500-2,000 tokens: goes past the 512 window and does not trigger the indexer selection
       (2,048).
largo  >= 8,448 tokens: triggers the indexer; the final queries discard more than 70 % of the
       groups.
tools  a chat turn with a tool, rendered with the model's template.

Usage: prompts.py <snapshot> <radiance-docs-dir> <output.json>
"""
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

SHORT = """El río que cruza el pueblo nace en una sierra de granito y baja muy rápido durante los primeros
kilómetros. En primavera, cuando se derrite la nieve, el caudal se multiplica por diez en pocos días y el
agua arrastra arena, ramas y alguna piedra grande. Por eso el puente viejo tiene los arcos tan altos:
quien lo construyó sabía que el agua subía hasta casi tocar la clave. Más abajo, el valle se abre y el
río se calma. Allí había tres molinos harineros; hoy solo queda uno en pie, convertido en un pequeño
museo donde se ve cómo el agua movía la piedra. En verano el caudal baja tanto que en algunos tramos se
puede cruzar andando, y los vecinos aprovechan las pozas más profundas para bañarse. El ayuntamiento
mide el nivel cada mañana en una regla pintada en el estribo del puente, y apunta el dato en un cuaderno
que ya lleva más de cuarenta años. Con esas notas se puede saber qué inviernos fueron secos y cuáles
trajeron crecidas, aunque nadie las haya estudiado todavía con calma."""

TOOLS = [{
    "type": "function",
    "function": {
        "name": "consultar_temperatura",
        "description": "Devuelve la temperatura actual de un sensor de la casa.",
        "parameters": {
            "type": "object",
            "properties": {
                "sensor": {"type": "string", "description": "Nombre del sensor, por ejemplo 'salon'."},
                "unidad": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["sensor"],
        },
    },
}]

MESSAGES = [
    {"role": "user", "content": "¿Qué temperatura hace ahora en el salón? Dímela en grados Celsius."},
    {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
        "name": "consultar_temperatura", "arguments": {"sensor": "salon", "unidad": "celsius"}}}]},
    {"role": "tool", "content": "{\"sensor\": \"salon\", \"temperatura\": 21.4, \"unidad\": \"celsius\"}"},
    {"role": "assistant", "content": "En el salón hay ahora 21,4 °C."},
]


def trim(tok, text: str, n: int, bos: int) -> list[int]:
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) < n - 1:
        raise SystemExit(f"text too short: {len(ids)} tokens, {n - 1} needed")
    return [bos] + ids[: n - 1]


def main() -> None:
    snapshot, docs, output = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    tok = AutoTokenizer.from_pretrained(snapshot)
    bos = tok.bos_token_id
    short = [bos] + tok(SHORT, add_special_tokens=False)["input_ids"]
    medium = trim(tok, (docs / "ARCHITECTURES.md").read_text(), 1800, bos)
    long_ = trim(tok, (docs / "GUIDE.md").read_text(), 8448, bos)
    text = tok.apply_chat_template(MESSAGES, tools=TOOLS, tokenize=False, add_generation_prompt=False)
    tools = tok(text, add_special_tokens=False)["input_ids"]
    prompts = {"corto": short, "medio": medium, "largo": long_, "tools": tools}
    assert len(tools) > 50, len(tools)
    assert len(short) < 512, len(short)
    assert 1500 <= len(medium) <= 2000, len(medium)
    assert len(long_) >= 8448, len(long_)
    Path(output).write_text(json.dumps({"bos": bos, "prompts": prompts}))
    for k, v in prompts.items():
        print(k, len(v))


if __name__ == "__main__":
    main()
