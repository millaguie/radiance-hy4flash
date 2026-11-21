#!/usr/bin/env python3
"""Reference for the Hy4-flash vision tower (plan step 11), with the code from Transformers PR
#50903 (commit c2ec850), the same as plan step 2.

For each image it saves, in `<out>/<name>.safetensors`:
- `pixel_values` [P, 768] and `grid_thw` [1, 3]: the output of the HF processor (default backend,
  torchvision), which the radiance processor has to reproduce;
- `h.LNN` [P, 1152]: the residual after each tower block (27);
- `emb` [P, 1152]: the input to the first block (patches + positions);
- `out` [rows, 3072]: the adapter output (`perceive`), with the line-break column:
  `(h/2) * (w/2 + 1)` rows, the ones that go to the image markers.
- `ids`: the tokens of a user turn with the image and a question, through the snapshot's chat
  template, for the text model pass.

With `--dtype fp32` (the reference) and `--dtype bf16` (the noise floor) on the same pixels: the
tower is ~0.4 B parameters, it fits whole in RAM.

Usage: vision_ref.py --snapshot DIR --out DIR --dtype fp32|bf16 [--images name=path ...]
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoProcessor
from transformers.models.hy_v4_flash import modeling_hy_v4_flash as M

COMMIT = "c2ec850510a37978ea83df226589623d5f2be776"


def default_images(snap: Path):
    a = snap / "assets"
    # landscape, very wide, huge (falls to the pixel cap) and small (rises to the floor)
    out = {"apaisada": a / "blind-evaluation-en.png", "ancha": a / "logo-en.png",
           "enorme": a / "dfly2.png"}
    return out


def load_tower(snap: Path, cfg, dtype):
    tower = M.HYV4FlashVisionTransformer(cfg.vision_config).to(dtype)
    wmap = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    # checkpoint names -> HF names (conversion_mapping.py of the same commit)
    def hf(n):
        n = n[len("vit."):]
        n = n.replace(".input_layernorm.", ".layer_norm1.").replace(".post_attention_layernorm.", ".layer_norm2.")
        n = n.replace(".mlp.dense_h_to_4h.", ".mlp.fc1.").replace(".mlp.dense_4h_to_h.", ".mlp.fc2.")
        n = n.replace("perceive.norm.", "patch_merger.before_rms.")
        n = n.replace("perceive.proj.0.", "patch_merger.proj_conv.").replace("perceive.proj.2.", "patch_merger.proj_out.")
        n = n.replace("perceive.", "patch_merger.")
        return n
    sd, open_files = {}, {}
    for k, f in wmap.items():
        if not k.startswith("vit."):
            continue
        if f not in open_files:
            open_files[f] = safe_open(str(snap / f), framework="pt")
        sd[hf(k)] = open_files[f].get_tensor(k).to(dtype)
    missing, unexpected = tower.load_state_dict(sd, strict=False)
    if missing or unexpected:
        sys.exit(f"tower weights: missing {missing}, unexpected {unexpected}")
    return tower.eval()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], required=True)
    ap.add_argument("--images", nargs="*", default=[])
    ap.add_argument("--small", action="store_true", default=True,
                    help="adds a 96x64 crop of the landscape image (rises to the pixel floor)")
    a = ap.parse_args()
    dtype = torch.float32 if a.dtype == "fp32" else torch.bfloat16
    torch.set_num_threads(16)
    a.out.mkdir(parents=True, exist_ok=True)

    cfg = AutoConfig.from_pretrained(a.snapshot)
    cfg.vision_config._attn_implementation = "eager"
    proc = AutoProcessor.from_pretrained(a.snapshot)
    tower = load_tower(a.snapshot, cfg, dtype)

    imgs = default_images(a.snapshot)
    for x in a.images:
        n, r = x.split("=", 1)
        imgs[n] = Path(r)
    loaded = {n: Image.open(r).convert("RGB") for n, r in imgs.items()}
    if a.small:
        loaded["pequena"] = loaded["apaisada"].crop((100, 100, 196, 164))

    meta = {"commit_transformers": COMMIT, "dtype": a.dtype, "images": {}}
    for n, im in loaded.items():
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe la imagen."}]}]
        text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        enc = proc(text=[text], images=[im], return_tensors="pt")
        pv, grid = enc["pixel_values"], enc["image_grid_thw"]
        block_out = {}
        hooks = [blk.register_forward_hook(lambda m, i, o, k=k: block_out.__setitem__(f"h.L{k:02d}", o.detach()))
                   for k, blk in enumerate(tower.layers)]
        emb = {}
        g2 = tower.embeddings.register_forward_hook(lambda m, i, o: emb.__setitem__("emb", o.detach()))
        with torch.no_grad():
            r = tower(pv.to(dtype), grid_thw=grid)
        for g in hooks + [g2]:
            g.remove()
        out = r.pooler_output.squeeze(0)
        t = {"pixel_values": pv.float().contiguous(), "grid_thw": grid.to(torch.int64),
             "emb": emb["emb"].squeeze(0).float().contiguous(), "out": out.float().contiguous(),
             "ids": enc["input_ids"][0].to(torch.int64)}
        for k, v in block_out.items():
            t[k] = v.squeeze(0).float().contiguous()
        save_file(t, str(a.out / f"{n}.safetensors"))
        _, gh, gw = grid[0].tolist()
        rows = (gh // 2) * (gw // 2 + 1)
        meta["images"][n] = {"source_size": list(im.size), "grid_thw": grid[0].tolist(), "patches": int(pv.shape[0]),
                               "rows": int(out.shape[0]), "expected_rows": rows, "tokens": int(enc["input_ids"].shape[1])}
        print(n, meta["images"][n], flush=True)
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
