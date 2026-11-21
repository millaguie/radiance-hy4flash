#!/usr/bin/env python3
"""Layer-by-layer Hy4-flash reference, with forced input (plan step 2).

Uses the code from Transformers PR #50903 (commit c2ec850) as is, but loads the weights of one
layer at a time: the FP32 model is ~622 GB and the host had 124 GB of RAM. Each layer is read from
the checkpoint, run on the 4 prompts and released.

Decisions that change numbers (they also go in meta.json):
- The Engram norms (`nn.RMSNorm` without eps) use eps = 1.1920929e-07 in both passes. Without
  setting it, PyTorch uses the eps of the input dtype, which in BF16 is 0.0078: the BF16 pass
  would measure a different eps instead of rounding noise. llama.cpp (PR #32517) also uses
  1.1920929e-07.
- The Engram table stays in BF16 (as in the checkpoint) and goes to the compute dtype after the
  gather, which is what `HYV4FlashEngramLayer.forward` already does.
- Attention is computed by query blocks (mathematically the same as `eager`).

Usage: reference.py --snapshot DIR --prompts prompts.json --out DIR --dtype fp32|bf16
       [--layers N] [--only corto,medio]
"""
import argparse
import gc
import json
import re
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.hy_v4_flash import modeling_hy_v4_flash as M

COMMIT_TRANSFORMERS = "c2ec850510a37978ea83df226589623d5f2be776"
ENGRAM_EPS = 1.1920929e-07
QUERY_BLOCK = 1024
SUBBLOCK_LAYERS = {0, 1, 2}  # the trimmed model ones: the inside of each sub-block is dumped
BAND = (448, 576)  # indexer order ranks around the top-512 cutoff

# Renames from `conversion_mapping.py` of the same commit (only the text tower ones).
RENAMES = [
    (r"\.self_attn\.learnable_sink_param", ".self_attn.sinks"),
    (r"\.self_attn\.linear_gate\.weight", ".self_attn.gate_proj.weight"),
    (r"\.hc_attn_layer\.hc_pre\.hc_fn", ".attn_hc.fn"),
    (r"\.hc_attn_layer\.hc_pre\.hc_base", ".attn_hc.base"),
    (r"\.hc_attn_layer\.hc_pre\.hc_scale", ".attn_hc.scale"),
    (r"\.hc_mlp_layer\.hc_pre\.hc_fn", ".ffn_hc.fn"),
    (r"\.hc_mlp_layer\.hc_pre\.hc_base", ".ffn_hc.base"),
    (r"\.hc_mlp_layer\.hc_pre\.hc_scale", ".ffn_hc.scale"),
    (r"\.hc_head\.hc_head_fn", ".hc_head.hc_fn"),
    (r"\.hc_head\.hc_head_base", ".hc_head.hc_base"),
    (r"\.hc_head\.hc_head_scale", ".hc_head.hc_scale"),
]

# Dumps of the current pass: {name: tensor}, per prompt. Emptied after each layer.
DUMP: dict[str, torch.Tensor] = {}
CURRENT_LAYER = -1


def save_dump(name: str, t: torch.Tensor, always: bool = False) -> None:
    """Saves a tensor of the current prompt and layer. Without `always`, only in the trimmed model
    layers."""
    if always or CURRENT_LAYER in SUBBLOCK_LAYERS:
        DUMP[name] = t.detach().to("cpu").contiguous().clone()


# ---------------------------------------------------------------- weights

class Weights:
    """Reads checkpoint tensors by HF text model name (already renamed)."""

    def __init__(self, snapshot: Path):
        self.snapshot = snapshot
        wmap = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
        self.source: dict[str, tuple[str, str]] = {}
        for k, file_name in wmap.items():
            if not k.startswith("model.") or k.startswith(("model.dfly", "model.vit")):
                if k == "lm_head.weight":
                    self.source["lm_head.weight"] = (file_name, k)
                continue
            n = "." + k[len("model."):]
            for a, b in RENAMES:
                n = re.sub(a, b, n)
            self.source[n[1:]] = (file_name, k)
        self.open_files: dict[str, object] = {}

    def read(self, name: str) -> torch.Tensor:
        file_name, key = self.source[name]
        if file_name not in self.open_files:
            self.open_files[file_name] = safe_open(str(self.snapshot / file_name), framework="pt")
        return self.open_files[file_name].get_tensor(key)

    def release(self) -> None:
        self.open_files.clear()


def materialize(module: torch.nn.Module, prefix: str, weights: Weights, dtype: torch.dtype, cfg) -> None:
    """Puts the real weights into a module built on `meta`, with the HF dtype rules."""
    strict = M.HYV4FlashPreTrainedModel._keep_in_fp32_modules_strict
    persistent = set(module.state_dict().keys())
    shapes = {n: tuple(t.shape) for n, t in module.state_dict().items()}
    module.to_empty(device="cpu")
    for name, p in list(module.named_parameters()) + list(module.named_buffers()):
        full_name = prefix + name
        if full_name not in weights.source:
            if name in persistent:
                raise SystemExit(f"{full_name}: not in the checkpoint (broken rename?)")
            continue  # non-persistent buffer: fix_non_persistent_buffers recomputes it
        t = weights.read(full_name)
        if tuple(t.shape) != shapes[name] and not name.endswith("lookup_table"):
            raise SystemExit(f"{full_name}: shape {tuple(t.shape)}, expected {shapes[name]}")
        if full_name.endswith("multi_head_embedding.embedding.weight"):
            pass  # the Engram table stays in BF16
        elif t.is_floating_point():
            # Exactly the HF rule: only `_keep_in_fp32_modules_strict` stays in FP32.
            fp32 = any(e in full_name for e in strict)
            t = t.to(torch.float32 if fp32 else dtype)
        with torch.no_grad():
            p.data = t
    fix_non_persistent_buffers(module, cfg)


def fix_non_persistent_buffers(module: torch.nn.Module, cfg) -> None:
    """`to_empty` leaves garbage in the buffers that are not in the checkpoint: they are
    recomputed."""
    for name, sub in module.named_modules():
        if isinstance(sub, M.HYV4FlashRotaryEmbedding):
            fresh = M.HYV4FlashRotaryEmbedding(cfg)
            for b, v in fresh.named_buffers():
                setattr(sub, b, v)
        elif isinstance(sub, M.HYV4FlashEngramEmbedding):
            sub.offsets = sub.head_offsets()
        elif isinstance(sub, M.HYV4FlashEngramLayer):
            sub.head_vocab_sizes = torch.tensor(sub.config_head_vocab_sizes)
            vocab = int(sub.lookup_table.max()) + 1  # like `_init_weights`
            sub.hash_multipliers = sub.draw_hash_multipliers(vocab)
            for m in sub.modules():
                if isinstance(m, torch.nn.RMSNorm):
                    m.eps = ENGRAM_EPS


# ---------------------------------------------------------------- capture patches

def blockwise_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """`eager_attention_forward` with sinks, by query blocks. Same result: the keys outside the key
    block are masked (exp(min) = exactly 0)."""
    q_len, k_len = query.shape[-2], key.shape[-2]
    offset = k_len - q_len
    output = []
    window = module.config.sliding_window if module.attention_type == "sliding_attention" else None
    for q0 in range(0, q_len, QUERY_BLOCK):
        q1 = min(q_len, q0 + QUERY_BLOCK)
        k1 = offset + q1
        k0 = max(0, offset + q0 - window + 1) if window else 0
        w = torch.matmul(query[:, :, q0:q1], key[:, :, k0:k1].transpose(2, 3)) * scaling
        if attention_mask is not None:
            w = w + attention_mask[:, :, q0:q1, k0:k1]
        sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, q1 - q0, -1)
        # Like HF: the sinks are FP32, so the `cat` promotes and the softmax runs in FP32.
        w = torch.cat([w, sinks], dim=-1)
        w = w - w.max(dim=-1, keepdim=True).values
        probs = F.softmax(w, dim=-1, dtype=w.dtype)[..., :-1].to(value.dtype)
        output.append(torch.matmul(probs, value[:, :, k0:k1]))
    out = torch.cat(output, dim=2).transpose(1, 2).contiguous()
    return out, None


def indexer_with_capture(self, hidden_states, q_resid, group_keys, group_position_ids, position_ids,
                          query_group_index):
    """Copy of `HYV4FlashCompressedIndexer.forward` that also saves scores and selection."""
    batch_size, seq_length, _ = hidden_states.shape
    num_groups = group_keys.shape[1]
    query = self.query_projection(q_resid).view(batch_size, seq_length, self.num_heads, self.head_dim)
    query_cos, query_sin = self.rotary_emb(query, position_ids, layer_type="indexed_attention")
    query = M._apply_partial_rotary_pos_emb(query, query_cos, query_sin, self.rope_dim, unsqueeze_dim=2)
    group_cos, group_sin = self.rotary_emb(group_keys, group_position_ids, layer_type="indexed_attention")
    keys = M._apply_partial_rotary_pos_emb(
        group_keys.unsqueeze(2), group_cos, group_sin, self.rope_dim, unsqueeze_dim=2).squeeze(2)
    head_weights = self.head_weight_projection(hidden_states).float() * (
        self.num_heads ** -0.5 * self.head_dim ** -0.5)
    scores = F.relu(torch.matmul(query.float(), keys.float().unsqueeze(1).transpose(-1, -2)))
    logits = torch.matmul(head_weights.unsqueeze(-2), scores).squeeze(-2)
    group_ids = torch.arange(num_groups, device=logits.device)
    logits = logits.masked_fill(group_ids >= query_group_index.unsqueeze(-1), float("-inf"))
    complete_topk = min(self.group_topk - 1, num_groups)
    recalled = logits.topk(complete_topk, dim=-1).indices
    selection = torch.cat((recalled, query_group_index.unsqueeze(-1)), dim=-1)

    # Groups with score -inf (not closed yet) are picked by the topk tie-break and hidden by the
    # causal mask: in the dump they go as -1 so the comparer does not take them for a selection.
    invalid = torch.isneginf(logits.gather(-1, recalled))
    sel_dump = torch.cat((recalled.masked_fill(invalid, -1), query_group_index.unsqueeze(-1)), dim=-1)
    save_dump("index.selection", sel_dump[0].to(torch.int32), always=True)
    upto = min(BAND[1], num_groups)
    order = logits.topk(upto, dim=-1)
    band_g = order.indices[0, :, BAND[0]:upto].masked_fill(torch.isneginf(order.values[0, :, BAND[0]:upto]), -1)
    save_dump("index.band_scores", order.values[0, :, BAND[0]:upto].float(), always=True)
    save_dump("index.band_groups", band_g.to(torch.int32), always=True)
    save_dump("index.cutoff_score", order.values[0, :, complete_topk - 1].float(), always=True)
    save_dump("index.group_keys", group_keys[0])
    save_dump("index.query", query[0])
    save_dump("index.head_weights", head_weights[0])
    save_dump("index.logits", logits[0].float())
    return selection


_pool_original = M.HYV4FlashCompressedIndexer.pool_groups


def pool_with_capture(self, token_keys):
    save_dump("index.token_keys", token_keys[0])
    return _pool_original(self, token_keys)


_router_original = M.HYV4FlashTopkRouter.forward


def router_with_capture(self, hidden_states):
    logits, weights, indices = _router_original(self, hidden_states)
    save_dump("router.logits", logits.float(), always=True)
    save_dump("router.weights", weights.float(), always=True)
    save_dump("router.experts", indices.to(torch.int32), always=True)
    return logits, weights, indices


def engram_with_capture(self, hidden_states, ngram_tokens, engram_mask):
    """Copy of `HYV4FlashEngramLayer.forward` that saves ids, gate and output."""
    save_dump("engram.in", hidden_states[0])
    hash_ids = self.hash_ngrams(ngram_tokens)
    # `engram.ids`: flat table ids (local hash + head offset), the ones that go into the gather
    # and the ones `hy4_ngram_ids` emits. `engram.hash_ids`: the local hash per head.
    flat = (hash_ids.clamp_min(0) + self.multi_head_embedding.offsets).masked_fill(hash_ids < 0, -1)
    save_dump("engram.ids", flat[0].to(torch.int64), always=True)
    save_dump("engram.hash_ids", hash_ids[0].to(torch.int64), always=True)
    embeddings = self.multi_head_embedding(hash_ids).to(hidden_states.dtype)
    if self.fp8_embedding:
        embeddings = M._fp8_round_trip(embeddings)
    save_dump("engram.rows_fp8", embeddings[0])
    embeddings = torch.stack(
        [norm(embeddings[:, :, h]) for h, norm in enumerate(self.lookup_per_head_norms)], dim=2)
    embeddings = embeddings.flatten(start_dim=-2)
    save_dump("engram.rows_norm", embeddings[0])
    gates, keys, queries = [], [], []
    for hc in range(self.hc_mult):
        key = self.norm_key[hc](self.key_projs[hc](embeddings))
        query = self.norm_query[hc](hidden_states[:, :, hc])
        keys.append(key)
        queries.append(query)
        gate = (key * query).sum(dim=-1) * self.hidden_size ** -0.5
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gates.append(gate.sigmoid())
    gates = torch.stack(gates, dim=2).unsqueeze(-1)
    save_dump("engram.key_norm", torch.stack(keys, dim=2)[0])
    save_dump("engram.query_norm", torch.stack(queries, dim=2)[0])
    save_dump("engram.gate", gates[0, :, :, 0])
    value = self.value_proj(embeddings)
    save_dump("engram.value", value[0])
    main_signal = self.unified_pre_gamma * self.unified_rmsnorm(value)
    save_dump("engram.signal", main_signal[0])
    branch = torch.stack([scale * main_signal for scale in self.unified_per_hc_scales], dim=2)
    output = (gates * main_signal.unsqueeze(2) + branch).masked_fill(~engram_mask[:, :, None, None], 0.0)
    save_dump("engram.out", output[0], always=True)
    return output


def layer_with_capture(layer, hidden_states, **kw):
    """Copy of `HYV4FlashDecoderLayer.forward` with a dump of each sub-block."""
    dtype = hidden_states.dtype
    if layer.engram_layer is not None:
        hidden_states = hidden_states + layer.engram_layer(hidden_states, kw["ngram_tokens"], kw["engram_mask"])
    save_dump("layer.in", hidden_states[0])
    residual = hidden_states
    post, hidden_states = layer.attn_hc(hidden_states)
    save_dump("ihc_attn.post", post[0])
    save_dump("ihc_attn.mix", hidden_states[0])
    hidden_states = layer.input_layernorm(hidden_states)
    save_dump("attn.in", hidden_states[0])
    hidden_states, _ = layer.self_attn(
        hidden_states=hidden_states, attention_mask=kw["attention_mask"], position_ids=kw["position_ids"],
        cache=None, use_cache=False, position_embeddings=kw["position_embeddings"],
        indexer_position_ids=kw["indexer_position_ids"])
    save_dump("attn.out", hidden_states[0])
    hidden_states = (post.float().unsqueeze(-1) * hidden_states.float().unsqueeze(-2) + residual.float()).to(dtype)
    save_dump("layer.after_attn", hidden_states[0])
    residual = hidden_states
    post, hidden_states = layer.ffn_hc(hidden_states)
    save_dump("ihc_ffn.post", post[0])
    save_dump("ihc_ffn.mix", hidden_states[0])
    hidden_states = layer.post_attention_layernorm(hidden_states)
    save_dump("ffn.in", hidden_states[0])
    hidden_states = layer.mlp(hidden_states)
    save_dump("ffn.out", hidden_states[0])
    hidden_states = (post.float().unsqueeze(-1) * hidden_states.float().unsqueeze(-2) + residual.float()).to(dtype)
    save_dump("layer.out", hidden_states[0], always=True)
    return hidden_states


def apply_patches() -> None:
    M.eager_attention_forward = blockwise_attention
    M.HYV4FlashCompressedIndexer.forward = indexer_with_capture
    M.HYV4FlashCompressedIndexer.pool_groups = pool_with_capture
    M.HYV4FlashTopkRouter.forward = router_with_capture
    M.HYV4FlashEngramLayer.forward = engram_with_capture


# ---------------------------------------------------------------- pass

def final_head(head, x: torch.Tensor) -> torch.Tensor:
    """Reduction of the 4 streams, final norm and head (FP32), like the whole model."""
    hc_head, norm, lm_head = head
    h = norm(hc_head(x))
    return F.linear(h.float(), lm_head)


def reference_pass(snapshot: Path, prompts: dict, dtype_txt: str, out: Path | None, layers: int | None = None,
           threads: int = 24, cfg=None) -> dict:
    """Runs the reference and returns the logits per prompt. With `out`, dumps to disk.
    `cfg` lets you pass an already loaded text config (the equivalence test uses it)."""
    global CURRENT_LAYER
    torch.manual_seed(0)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    dtype = torch.float32 if dtype_txt == "fp32" else torch.bfloat16
    apply_patches()

    if cfg is None:
        cfg = AutoConfig.from_pretrained(snapshot).text_config
    cfg._attn_implementation = "eager"
    n_layers = layers or cfg.num_hidden_layers
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    weights = Weights(snapshot)
    with init_empty_weights():
        tm = M.HYV4FlashTextModel(cfg)

    # Embedding and closing head, which stay loaded for the whole pass.
    materialize(tm.embed_tokens, "embed_tokens.", weights, dtype, cfg)
    materialize(tm.hc_head, "hc_head.", weights, dtype, cfg)
    materialize(tm.norm, "norm.", weights, dtype, cfg)
    lm_head = weights.read("lm_head.weight").float()
    head = (tm.hc_head, tm.norm, lm_head)
    rotary = M.HYV4FlashRotaryEmbedding(cfg)

    state = {}
    for name, ids in prompts.items():
        ids_t = torch.tensor([ids])
        x = tm.embed_tokens(ids_t).to(dtype)
        t = ids_t.shape[1]
        pos = torch.arange(t).view(1, 1, -1).expand(3, 1, -1)
        mk = {"config": cfg, "inputs_embeds": x, "attention_mask": None, "cache": None,
              "position_ids": pos[0], "allow_is_causal_skip": False}
        state[name] = {
            "x": x.unsqueeze(2).expand(-1, -1, cfg.hc_mult, -1).contiguous(),
            "masks": {"indexed_attention": create_causal_mask(**mk),
                      "sliding_attention": create_sliding_window_causal_mask(**mk)},
            "rope": {lt: rotary(x, position_ids=pos, layer_type=lt) for lt in rotary.layer_types},
            "pos": pos,
            "ngram": M.HYV4FlashTextModel.build_ngram_tokens(types.SimpleNamespace(config=cfg), ids_t, pos),
            "engram_mask": torch.ones(1, t, dtype=torch.bool),
        }
        if out is not None:
            save_file({"ids": ids_t[0].to(torch.int64)}, str(out / f"{name}.ids.safetensors"))

    meta = {"commit_transformers": COMMIT_TRANSFORMERS, "dtype": dtype_txt, "engram_eps": ENGRAM_EPS,
            "layers": n_layers, "threads": threads, "query_block": QUERY_BLOCK,
            "subblock_layers": sorted(SUBBLOCK_LAYERS), "indexer_band": BAND,
            "invalid_selection": -1,
            "prompts": {k: len(v) for k, v in prompts.items()}, "times": {}}

    for i in range(n_layers):
        t0 = time.time()
        CURRENT_LAYER = i
        layer = tm.layers[i]
        materialize(layer, f"layers.{i}.", weights, dtype, cfg)
        t_load = time.time() - t0
        lt = layer.attention_type
        for name, e in state.items():
            DUMP.clear()
            with torch.no_grad():
                e["x"] = layer_with_capture(
                    layer, e["x"], attention_mask=e["masks"][lt], position_ids=e["pos"],
                    position_embeddings=e["rope"][lt], indexer_position_ids=e["pos"],
                    ngram_tokens=e["ngram"], engram_mask=e["engram_mask"])
                if i == 2:
                    DUMP["logits_trim"] = final_head(head, e["x"])[0]
            if out is not None:
                save_file(dict(DUMP), str(out / f"{name}.L{i:02d}.safetensors"))
        tm.layers[i] = torch.nn.Identity()  # releases the layer weights
        del layer
        weights.release()
        gc.collect()
        meta["times"][i] = {"load_s": round(t_load, 1), "total_s": round(time.time() - t0, 1)}
        print(f"layer {i:2d} {lt:18s} load {t_load:6.1f} s  total {time.time() - t0:6.1f} s", flush=True)

    output = {}
    for name, e in state.items():
        with torch.no_grad():
            logits = final_head(head, e["x"])[0]
        output[name] = logits
        ids = torch.tensor(prompts[name])
        accuracy = (logits[:-1].argmax(-1) == ids[1:]).float().mean().item()
        meta.setdefault("top1_accuracy", {})[name] = round(accuracy, 4)
        print(f"{name}: top-1 accuracy {accuracy:.3f}", flush=True)
        if out is not None:
            save_file({"logits": logits}, str(out / f"{name}.logits.safetensors"))
    if out is not None:
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True, type=Path)
    ap.add_argument("--prompts", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], required=True)
    ap.add_argument("--layers", type=int, default=None, help="only the first N layers (test)")
    ap.add_argument("--only", default=None, help="only these prompts, comma-separated")
    ap.add_argument("--threads", type=int, default=24)
    a = ap.parse_args()
    data = json.loads(a.prompts.read_text())
    prompts = {k: v for k, v in data["prompts"].items() if not a.only or k in a.only.split(",")}
    reference_pass(a.snapshot, prompts, a.dtype, a.out, a.layers, a.threads)


if __name__ == "__main__":
    sys.exit(main())
