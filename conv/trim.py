#!/usr/bin/env python3
"""Hy4-flash checkpoint trimmed to its first N layers (plan step 4), and the Engram constants.

Writes to <output>:
- config.json with a trimmed `text_config`: num_hidden_layers, layer_types, mlp_layer_types,
  swa_pattern, indexer_types and engram_layer_ids (only those < N), with their
  engram_layer_head_vocab_sizes;
- the tensors of layers 0..N-1, the embedding, the final norm, the stream reduction and the
  head (no drafter; the vision tower only with --vision), one safetensors per layer, with its index;
- engram-consts.safetensors: what the Engram hash needs and the checkpoint does not ship, and
  `hy4.ceros_rope` [rope, kv_lora] bf16, the zero rows of the absorbed MLA's kv_b;
  (`model.layers.L.engram_layer.hash_multipliers` [3] i64, `...head_vocab_sizes` [16] i64 and
  `...head_offsets` [16] i64), computed with the code of the Transformers PR (multipliers with
  numpy default_rng(engram_seed + 10007·layer_id) and compressed vocabulary max(lookup_table)+1).
- tokenizer and template, copied.

With N = 42 it only writes engram-consts.safetensors (for the whole model, with `rad-convert --add`).

With --vision the trimmed model also carries the vision tower (`vit.*`) and processor_config.json, and
the constants always carry `hy4.vit_merge_w` (the adapter's 2x2 conv permuted for the GEMM).

Usage: trim.py <snapshot> <output> [N=3] [--vision]
"""
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ref"))
from transformers.models.hy_v4_flash import modeling_hy_v4_flash as M  # noqa: E402


def consts(snap: Path, cfg: dict, idx: dict, layers: list[int]) -> dict:
    t = cfg["text_config"]
    out = {}
    for li in layers:
        layer_id = li + 1
        key = f"model.layers.{li}.engram_layer.lookup_table"
        with safe_open(str(snap / idx[key]), framework="pt") as f:
            lt = f.get_tensor(key)
        eng = M.HYV4FlashEngramLayer.__new__(M.HYV4FlashEngramLayer)
        eng.seed, eng.layer_id, eng.max_ngram_size = t["engram_seed"], layer_id, t["engram_max_ngram_size"]
        mult = eng.draw_hash_multipliers(int(lt.max()) + 1)
        sizes = [x for o in t["engram_layer_head_vocab_sizes"][str(layer_id)] for x in o]
        pre = f"model.layers.{li}.engram_layer."
        out[pre + "hash_multipliers"] = mult.to(torch.int64)
        out[pre + "head_vocab_sizes"] = torch.tensor(sizes, dtype=torch.int64)
        out[pre + "head_offsets"] = torch.tensor([0] + sizes[:-1], dtype=torch.int64).cumsum(0)
    return out


def main() -> None:
    snap, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    pos = [x for x in sys.argv[3:] if not x.startswith("--")]
    N = int(pos[0]) if pos else 3
    vision = "--vision" in sys.argv[3:]
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((snap / "config.json").read_text())
    idx = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    t = cfg["text_config"]
    engram_layers = [l - 1 for l in t["engram_layer_ids"] if l - 1 < N]
    cts = consts(snap, cfg, idx, engram_layers)
    # zeros [rope, kv_lora] for the absorbed kv_b (hyv4flash, decl_kvb_abs_map)
    cts["hy4.ceros_rope"] = torch.zeros(t["qk_rope_head_dim"], t["kv_lora_rank"], dtype=torch.bfloat16)
    # the vision adapter's 2x2 conv as a GEMM over [p00 | p01 | p10 | p11] (arch/hyv4flash/vision.h):
    # columns in (ky, kx, c) order
    k = "vit.perceive.proj.0.weight"
    with safe_open(str(snap / idx[k]), framework="pt") as f:
        w = f.get_tensor(k)
    cts["hy4.vit_merge_w"] = w.permute(0, 2, 3, 1).reshape(w.shape[0], -1).contiguous()
    save_file(cts, str(out_dir / "engram-consts.safetensors"))
    if N >= t["num_hidden_layers"]:
        print("Engram constants only:", out_dir / "engram-consts.safetensors")
        return

    t["num_hidden_layers"] = N
    for k in ("layer_types", "mlp_layer_types", "swa_pattern", "indexer_types"):
        if k in t:
            t[k] = t[k][:N]
    t["engram_layer_ids"] = [l for l in t["engram_layer_ids"] if l - 1 < N]
    # The trimmed model does not carry the drafter, nor its config, so that nothing announces speculation.
    for k in ("speculative_config", "num_nextn_predict_layers", "mtp_num_layers"):
        cfg.pop(k, None)
        t.pop(k, None)
    t["engram_layer_head_vocab_sizes"] = {str(l): t["engram_layer_head_vocab_sizes"][str(l)]
                                         for l in t["engram_layer_ids"]}
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=1))
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
              "processor_config.json"):
        if (snap / f).exists():
            shutil.copy(snap / f, out_dir / f)

    def expect(k: str) -> int | None:
        if k in ("lm_head.weight", "model.embed_tokens.weight", "model.norm.weight") or k.startswith("model.hc_head."):
            return -1
        if vision and k.startswith("vit."):
            return -2
        if k.startswith("model.layers."):
            l = int(k.split(".")[2])
            return l if l < N else None
        return None

    groups: dict[int, list[str]] = {}
    for k in idx:
        g = expect(k)
        if g is not None:
            groups.setdefault(g, []).append(k)
    wmap = {}
    for g, keys in sorted(groups.items()):
        name = ("model-vit.safetensors" if g == -2 else "model-common.safetensors" if g < 0
                  else f"model-layer{g:02d}.safetensors")
        tens = {}
        for k in keys:
            with safe_open(str(snap / idx[k]), framework="pt") as f:
                tens[k] = f.get_tensor(k)
        save_file(tens, str(out_dir / name))
        for k in keys:
            wmap[k] = name
        print(name, len(keys), "tensors", flush=True)
        del tens
    for k in cts:
        wmap[k] = "engram-consts.safetensors"
    (out_dir / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wmap}, indent=1))


if __name__ == "__main__":
    main()
