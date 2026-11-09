"""Header-derived admission for ordinary Gemma inference without an expert store."""

from __future__ import annotations

import struct
from collections.abc import Mapping, Sequence
from typing import Optional, cast

from inferssd.source import header

ARCHITECTURE = "gemma4"
WORKSPACE_BYTES = 1 << 30
HEADROOM_BYTES = 2 << 30


def is_model(path: str) -> bool:
    """Whether PATH is an ordinary RAM profile, without running SSD calibration."""
    try:
        return bool(header.read(path).kv.get("general.architecture") == ARCHITECTURE)
    except (OSError, ValueError, struct.error):
        return False  # Normal preparation reports invalid/missing model inputs.


def facts(headers: Sequence[header.Header]) -> dict[str, object]:
    """The mapped weights and unique F16 KV layers described by these headers."""
    kv = headers[0].kv
    if kv.get("general.architecture") != ARCHITECTURE:
        raise ValueError("the RAM profile requires a Gemma4 GGUF")

    def count(name: str, default: Optional[int] = None) -> int:
        value = kv.get(ARCHITECTURE + "." + name, default)
        if type(value) is not int or value < 0:
            raise ValueError("invalid Gemma4 RAM metadata: " + name)
        return value

    layers = count("block_count")
    unique = layers - count("attention.shared_kv_layers", 0)
    heads = count("attention.head_count_kv")
    key, value = count("attention.key_length"), count("attention.value_length")
    window = count("attention.sliding_window", 0)
    pattern = kv.get(ARCHITECTURE + ".attention.sliding_window_pattern", [False] * layers)
    if not layers or not 0 < unique <= layers or not heads or not key or not value:
        raise ValueError("invalid Gemma4 RAM dimensions")
    if (
        not isinstance(pattern, list)
        or len(pattern) != layers
        or any(type(x) is not bool for x in pattern)
    ):
        raise ValueError("invalid Gemma4 sliding-window pattern")
    swa_key = count("attention.key_length_swa", key)
    swa_value = count("attention.value_length_swa", value)
    tensors = [tensor for item in headers for tensor in item.tensors]
    return {
        "architecture": ARCHITECTURE,
        # Include the header and alignment: --mmap can keep those pages resident too.
        "model_bytes": sum(
            max((t.offset + t.nbytes for t in item.tensors), default=item.data_offset)
            for item in headers
        ),
        "tensor_bytes": sum(t.nbytes for t in tensors),
        "embedding_bytes": sum(t.nbytes for t in tensors if t.name == "token_embd.weight"),
        # Shared tail layers reuse earlier KV storage. Sliding layers retain a window
        # plus the batch being processed, as llama_memory_iswa does.
        "kv_layers": [
            [
                window if sliding else 0,
                heads * ((swa_key + swa_value) if sliding else (key + value)) * 2,
            ]
            for sliding in pattern[:unique]
        ],
    }


def validate(raw: object) -> dict[str, object]:
    """Validate recorded header facts without inventing missing RAM dimensions."""
    if not isinstance(raw, Mapping) or raw.get("architecture") != ARCHITECTURE:
        raise ValueError("the RAM profile requires Gemma4 header facts")
    for key in ("model_bytes", "tensor_bytes", "embedding_bytes"):
        if type(raw.get(key)) is not int or raw[key] <= 0:
            raise ValueError("invalid RAM model fact: " + key)
    layers = raw.get("kv_layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("invalid RAM model fact: kv_layers")
    for layer in layers:
        if (
            not isinstance(layer, list)
            or len(layer) != 2
            or type(layer[0]) is not int
            or layer[0] < 0
            or type(layer[1]) is not int
            or layer[1] <= 0
        ):
            raise ValueError("invalid RAM KV layer dimensions")
    return dict(raw)


def admission(
    raw: object,
    *,
    context: int,
    batch: int,
    gpu: bool,
    available: Optional[int],
    budget: int,
    headroom: int,
) -> dict[str, object]:
    """Fail closed unless the whole model and working buffers fit memory now."""
    model = validate(raw)
    if available is None:
        raise ValueError("cannot verify RAM-only model fit: available physical memory is unknown")
    # A CPU file mapping avoids the tied embedding copy. Ordinary GPU loading
    # reads directly into buffers, avoiding a duplicate source mapping.
    weights = (
        cast(int, model["tensor_bytes"]) + cast(int, model["embedding_bytes"])
        if gpu
        else cast(int, model["model_bytes"])
    )
    kv = sum(
        ((min(context, window + batch) if window else context) + 255) // 256 * 256 * width
        for window, width in cast(list[list[int]], model["kv_layers"])
    )
    required = weights + kv + WORKSPACE_BYTES
    reserve = max(HEADROOM_BYTES, headroom)
    if required > budget or required + reserve > available:
        raise ValueError(
            f"RAM-only model needs {required / 2**30:.2f} GiB "
            f"plus {reserve / 2**30:.2f} GiB headroom; "
            f"available {available / 2**30:.2f} GiB, budget {budget / 2**30:.2f} GiB. "
            "Free memory or reduce context; SSD offload is not enabled for this model."
        )
    return {
        "weights_bytes": weights,
        "kv_bytes": kv,
        "workspace_bytes": WORKSPACE_BYTES,
        "required_bytes": required,
        "headroom_bytes": reserve,
        "available_bytes": available,
        "budget_bytes": budget,
    }
