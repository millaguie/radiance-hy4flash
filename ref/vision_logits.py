#!/usr/bin/env python3
"""HF reference with an image on a trimmed model with the tower (conv/trim.py --vision): the
distribution of the last prompt position (log-softmax in FP32) and 8 greedy tokens, per image.
ref/compare_vision_logits.py compares it with what radiance returns through the API.

Usage: vision_logits.py --ckpt ~/hy4/recorte1-vis --snapshot S --out DIR img1.png img2.png ...
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor
from transformers.models.hy_v4_flash import modeling_hy_v4_flash as M


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("images", nargs="+", type=Path)
    a = ap.parse_args()
    torch.set_num_threads(16)
    a.out.mkdir(parents=True, exist_ok=True)
    proc = AutoProcessor.from_pretrained(a.snapshot)
    # The trimmed model's config does not pass HF validation with a single layer type: start from
    # the snapshot's config and trim it the same way as conv/trim.py.
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(a.snapshot)
    trimmed_cfg = json.loads((a.ckpt / "config.json").read_text())["text_config"]
    t = cfg.text_config
    N = trimmed_cfg["num_hidden_layers"]
    t.num_hidden_layers = N
    for k in ("layer_types", "mlp_layer_types", "swa_pattern", "indexer_types"):
        if getattr(t, k, None) is not None:
            setattr(t, k, list(getattr(t, k))[:N])
    t.engram_layer_ids = trimmed_cfg["engram_layer_ids"]
    t.engram_layer_head_vocab_sizes = trimmed_cfg["engram_layer_head_vocab_sizes"]
    model = M.HYV4FlashForConditionalGeneration.from_pretrained(a.ckpt, config=cfg, torch_dtype=torch.float32,
                                                                 attn_implementation="eager").eval()
    res = {}
    for img in a.images:
        im = Image.open(img).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe la imagen."}]}]
        text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        enc = proc(text=[text], images=[im], return_tensors="pt")
        with torch.no_grad():
            out = model(**enc)
            lp = torch.log_softmax(out.logits[0, -1].double(), -1)
            gen = model.generate(**enc, max_new_tokens=8, do_sample=False)
        torch.save({"lp": lp.float(), "ids": enc["input_ids"][0]}, a.out / f"{img.stem}.pt")
        res[img.stem] = {"tokens": int(enc["input_ids"].shape[1]), "top5": lp.topk(5).indices.tolist(),
                         "greedy": gen[0, enc["input_ids"].shape[1]:].tolist()}
        print(img.stem, res[img.stem], flush=True)
    (a.out / "summary.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
