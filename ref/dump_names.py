"""Dump names before and after the English rename (2026-11-21).

ref/reference.py and the plugin's debug tap (HY4_DUMP_DIR) write the English names. The dumps written
before the rename (the FP32 and BF16 reference passes, the plugin dumps in ~/hy4) carry the
Spanish ones: `canonical` maps an old name to the new one, so the readers take both.
"""

LEGACY = {
    "capa.entrada": "layer.in",
    "capa.tras_atn": "layer.after_attn",
    "capa.salida": "layer.out",
    "ihc_atn.post": "ihc_attn.post",
    "ihc_atn.mezcla": "ihc_attn.mix",
    "ihc_ffn.mezcla": "ihc_ffn.mix",
    "atn.entrada": "attn.in",
    "atn.salida": "attn.out",
    "ffn.entrada": "ffn.in",
    "ffn.salida": "ffn.out",
    "engram.entrada": "engram.in",
    "engram.filas_fp8": "engram.rows_fp8",
    "engram.filas_norm": "engram.rows_norm",
    "engram.clave_norm": "engram.key_norm",
    "engram.consulta_norm": "engram.query_norm",
    "engram.puerta": "engram.gate",
    "engram.valor": "engram.value",
    "engram.senal": "engram.signal",
    "engram.salida": "engram.out",
    "router.pesos": "router.weights",
    "router.expertos": "router.experts",
    "indice.seleccion": "index.selection",
    "indice.banda_scores": "index.band_scores",
    "indice.banda_grupos": "index.band_groups",
    "indice.score_corte": "index.cutoff_score",
    "indice.claves_agrupadas": "index.group_keys",
    "indice.consulta": "index.query",
    "indice.pesos_cabeza": "index.head_weights",
    "indice.logits": "index.logits",
    "indice.claves_token": "index.token_keys",
    "indice.sel_paginas": "index.sel_pages",
    "indice.bt": "index.bt",
    "logits_recorte": "logits_trim",
}


def canonical(name: str) -> str:
    """The current name of a dump, whether it was written with the old or the new one."""
    return LEGACY.get(name, name)


def canonical_keys(d: dict) -> dict:
    """A loaded dump ({name: tensor}) with its keys under the current names."""
    return {canonical(k): v for k, v in d.items()}
