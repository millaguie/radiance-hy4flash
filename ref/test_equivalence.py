#!/usr/bin/env python3
"""Checks that `reference.py` reproduces the HF forward (finding F5 of a review).

Builds a tiny Hy4-flash with random weights from the real config, saves it as a checkpoint (with
the names of the published checkpoint, via `save_pretrained`) and compares:

  HF:     HYV4FlashForConditionalGeneration.from_pretrained(...)(input_ids).logits, unpatched
  script: reference.reference_pass(...) on the same checkpoint

In FP32 with single-block attention they must be bit-identical. With small blocks (which trim the
window keys) `allclose` is required. In BF16, HF is loaded with its dtype plan and the Engram
norms are set to the same eps the script uses.

Usage: test_equivalence.py <real snapshot>
"""
import copy
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.models.hy_v4_flash import modeling_hy_v4_flash as M

import reference as R

# Copies of the original functions before `reference` applies its patches.
ORIG = {
    "eager": M.eager_attention_forward,
    "idx": M.HYV4FlashCompressedIndexer.forward,
    "pool": M.HYV4FlashCompressedIndexer.pool_groups,
    "router": M.HYV4FlashTopkRouter.forward,
    "engram": M.HYV4FlashEngramLayer.forward,
}


def remove_patches() -> None:
    M.eager_attention_forward = ORIG["eager"]
    M.HYV4FlashCompressedIndexer.forward = ORIG["idx"]
    M.HYV4FlashCompressedIndexer.pool_groups = ORIG["pool"]
    M.HYV4FlashTopkRouter.forward = ORIG["router"]
    M.HYV4FlashEngramLayer.forward = ORIG["engram"]


def _set_fields(obj, **kw) -> None:
    """Sets only real config fields; derived ones (properties without a setter) are skipped."""
    for k, v in kw.items():
        prop = getattr(type(obj), k, None)
        if isinstance(prop, property) and prop.fset is None:
            continue
        setattr(obj, k, v)


def tiny_config(snapshot: str):
    cfg = AutoConfig.from_pretrained(snapshot)
    t = cfg.text_config
    t.hidden_size = 64
    t.intermediate_size = 96
    t.moe_intermediate_size = 32
    t.routed_expert_hidden_size = 32
    _set_fields(t, n_routed_experts=8, num_local_experts=8)
    t.num_experts_per_tok = 2
    t.num_attention_heads = 2
    t.num_key_value_heads = 2
    t.swa_num_attention_heads = 4
    t.q_lora_rank = 32
    t.kv_lora_rank = 16
    t.qk_nope_head_dim = t.swa_qk_nope_head_dim = 8
    t.qk_rope_head_dim = t.swa_qk_rope_head_dim = 8
    _set_fields(t, qk_head_dim=16, swa_qk_head_dim=16, v_head_dim=16, swa_v_head_dim=16, head_dim=16)
    # sparse with dense FFN, sparse with Engram and MoE, window with MoE, sparse with MoE
    t.num_hidden_layers = 4
    t.layer_types = ["indexed_attention", "indexed_attention", "sliding_attention", "indexed_attention"]
    t.mlp_layer_types = ["dense", "sparse", "sparse", "sparse"]
    t.first_k_dense_replace = 1
    t.sliding_window = 8
    t.compressed_indexer_candidate_topk = 16
    t.compressed_indexer_group_topk = 4
    t.compressed_indexer_num_attention_heads = 2
    t.compressed_indexer_head_dim = 16
    for lt in t.rope_parameters:
        t.rope_parameters[lt]["mrope_section"] = [2, 1, 1]
    t.engram_layer_ids = [2]
    t.engram_n_embed_per_ngram = 32
    t.engram_n_head_per_ngram = 2
    t.engram_layer_head_vocab_sizes = {"2": [[101, 103], [107, 109]]}
    if hasattr(t, "swa_pattern"):
        t.swa_pattern = [0, 0, 1, 0]
    if hasattr(t, "indexer_types"):
        t.indexer_types = t.indexer_types[:4]
    v = cfg.vision_config
    for k, val in (("depth", 1), ("num_hidden_layers", 1), ("hidden_size", 32), ("intermediate_size", 64),
                   ("num_heads", 2), ("num_attention_heads", 2), ("out_hidden_size", 64)):
        if hasattr(v, k):
            setattr(v, k, val)
    return cfg


def randomize_degenerate(model) -> None:
    """`_init_weights` leaves sinks, biases and bases at 0 and the Engram map as identity: that way
    neither the sinks nor the multiplier re-sorting (`max()+1` of the map) are tested. They are
    randomized."""
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if any(k in name for k in ("sinks", ".base", ".scale", "hc_base", "hc_scale")):
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)
            elif name.endswith("weight") and p.ndim == 1:  # norm weights
                p.copy_(1.0 + torch.randn(p.shape, generator=g) * 0.2)
        for name, b in model.named_buffers():
            if name.endswith("e_score_correction_bias"):
                b.copy_(torch.randn(b.shape, generator=g) * 0.1)
            elif name.endswith("lookup_table"):
                # non-identity map with max()+1 (< size): changes the multipliers
                b.copy_(torch.randint(0, b.shape[0] // 3, b.shape, generator=g, dtype=b.dtype))


def compare(name: str, a: torch.Tensor, b: torch.Tensor, exact: bool) -> bool:
    a, b = a.float(), b.float()
    if exact:
        ok = torch.equal(a, b)
    else:
        ok = None
    err = ((a - b).norm() / a.norm().clamp_min(1e-30)).item()
    if ok is None:
        ok = err < 1e-5
    print(f"  {name}: {'OK' if ok else 'FAIL'} (relative L2 {err:.3e})")
    return ok


def main() -> int:
    snapshot = sys.argv[1]
    # Same threads and deterministic mode for HF and the script (otherwise FP32 differs in rounding).
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(1234)
    cfg = tiny_config(snapshot)
    prompts = {"a": torch.randint(10, 1000, (40,)).tolist(), "b": torch.randint(10, 1000, (70,)).tolist()}
    all_ok = True
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        model = M.HYV4FlashForConditionalGeneration(cfg)
        randomize_degenerate(model)
        model.save_pretrained(tmp)
        (tmp / "model.safetensors.index.json").exists() or _single_index(tmp)
        del model

        for dtype_txt, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
            print(f"{dtype_txt}:")
            remove_patches()
            hf = M.HYV4FlashForConditionalGeneration.from_pretrained(tmp, dtype=dtype)
            hf.config.text_config._attn_implementation = "eager"
            for m in hf.modules():
                if isinstance(m, M.HYV4FlashEngramLayer):
                    for n in m.modules():
                        if isinstance(n, torch.nn.RMSNorm):
                            n.eps = R.ENGRAM_EPS
            ref = {}
            with torch.no_grad():
                for k, ids in prompts.items():
                    ref[k] = hf(input_ids=torch.tensor([ids])).logits[0].float()
            del hf
            # Block 8 in FP32 trims the window keys and changes the summation order: L2 < 1e-5.
            # In BF16 both blocks give exact equality.
            for block, exact in ((1024, True), (8, dtype_txt == "bf16")):
                R.QUERY_BLOCK = block
                cfg_txt = copy.deepcopy(AutoConfig.from_pretrained(tmp).text_config)
                ours = R.reference_pass(tmp, prompts, dtype_txt, None, threads=4, cfg=cfg_txt)
                for k in prompts:
                    all_ok &= compare(f"{k} block {block}", ref[k], ours[k], exact)
            remove_patches()
        all_ok &= check_dumps(tmp, prompts)
    print("ALL OK" if all_ok else "THERE ARE DIFFERENCES")
    return 0 if all_ok else 1


# Dumps that plan step 2 promises, per layer type (layers of the tiny model: 0 sparse dense,
# 1 sparse with Engram and MoE, 2 window with MoE, 3 sparse with MoE).
ALWAYS = {"layer.out"}
ENGRAM = {"engram.ids", "engram.hash_ids", "engram.out"}
ROUTER = {"router.logits", "router.weights", "router.experts"}
INDEX = {"index.selection", "index.band_scores", "index.band_groups", "index.cutoff_score"}
SUBBLOCK = {"layer.in", "ihc_attn.post", "ihc_attn.mix", "attn.in", "attn.out", "layer.after_attn",
             "ihc_ffn.post", "ihc_ffn.mix", "ffn.in", "ffn.out"}
SUB_ENGRAM = {"engram.in", "engram.rows_fp8", "engram.rows_norm", "engram.key_norm",
              "engram.query_norm", "engram.gate", "engram.value", "engram.signal"}
SUB_INDEX = {"index.token_keys", "index.group_keys", "index.query", "index.head_weights",
              "index.logits"}


def check_dumps(tmp: Path, prompts: dict) -> bool:
    """Tiny pass with a dump to disk: names, shapes, dtypes and consistency of each file."""
    import json
    from safetensors.torch import load_file
    R.QUERY_BLOCK = 1024
    out = tmp / "dump"
    cfg_txt = copy.deepcopy(AutoConfig.from_pretrained(tmp).text_config)
    logits = R.reference_pass(tmp, prompts, "fp32", out, threads=4, cfg=cfg_txt)
    ok = True
    meta = json.loads((out / "meta.json").read_text())
    ok &= meta["invalid_selection"] == -1 and meta["layers"] == 4
    for k, ids in prompts.items():
        t = len(ids)
        ok &= torch.equal(load_file(str(out / f"{k}.ids.safetensors"))["ids"], torch.tensor(ids))
        ok &= torch.equal(load_file(str(out / f"{k}.logits.safetensors"))["logits"], logits[k])
        sel_active = t > cfg_txt.compressed_indexer_candidate_topk
        for i in range(4):
            v = load_file(str(out / f"{k}.L{i:02d}.safetensors"))
            wanted = set(ALWAYS) | (SUBBLOCK if i in R.SUBBLOCK_LAYERS else set())
            if i == 1:
                wanted |= ENGRAM | (SUB_ENGRAM if i in R.SUBBLOCK_LAYERS else set())
            if i >= 1:
                wanted |= ROUTER
            if cfg_txt.layer_types[i] == "indexed_attention" and sel_active:
                wanted |= INDEX
                if i in R.SUBBLOCK_LAYERS:
                    wanted |= SUB_INDEX
            if i == 2:
                wanted |= {"logits_trim"}
            missing = wanted - set(v)
            if missing:
                print(f"  dump {k} L{i}: missing {sorted(missing)}")
                ok = False
            ok &= v["layer.out"].shape == (t, cfg_txt.hc_mult, cfg_txt.hidden_size)
            if "engram.ids" in v:
                sizes = [x for o in cfg_txt.engram_layer_head_vocab_sizes["2"] for x in o]
                offsets0 = torch.tensor([0] + sizes[:-1]).cumsum(0)
                ids_v = v["engram.ids"]
                ok &= ids_v.shape == (t, len(sizes))
                ok &= torch.equal(ids_v, v["engram.hash_ids"] + offsets0)
                ok &= bool(((ids_v >= offsets0) & (ids_v < offsets0 + torch.tensor(sizes))).all())
            if "router.experts" in v:
                ok &= v["router.experts"].shape == (t, cfg_txt.num_experts_per_tok)
            if "index.selection" in v:
                ok &= v["index.selection"].shape[1] == cfg_txt.compressed_indexer_group_topk
                ok &= v["index.selection"].dtype == torch.int32
    print(f"  dumps: {'OK' if ok else 'FAIL'}")
    return ok


def _single_index(tmp: Path) -> None:
    """`save_pretrained` of a small model writes a single file without an index: one is created."""
    import json
    from safetensors import safe_open
    file_name = "model.safetensors"
    with safe_open(str(tmp / file_name), framework="pt") as f:
        wmap = {k: file_name for k in f.keys()}
    (tmp / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wmap}))


if __name__ == "__main__":
    sys.exit(main())
