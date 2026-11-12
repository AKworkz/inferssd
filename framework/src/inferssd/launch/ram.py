"""Header-derived admission for ordinary Gemma inference without an expert store."""

from __future__ import annotations

import struct
from collections.abc import Mapping, Sequence
from typing import Optional, cast

from inferssd.source import header

ARCHITECTURE = "gemma4"
WORKSPACE_BYTES = 1 << 30
# How the engine holds the weights (see admission).
FILE, COPIED, LOOKUPS_IN_FILE = "file", "copied", "lookups-in-file"


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
        # The per-layer embedding table, a row lookup of one row a token.
        "ple_bytes": sum(t.nbytes for t in tensors if t.name == "per_layer_token_embd.weight"),
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
    ple = raw.get("ple_bytes")
    if type(ple) is not int or not 0 <= ple < cast(int, raw["tensor_bytes"]):
        raise ValueError("invalid RAM model fact: ple_bytes")
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
    weights: str,
    budget: int,
    available: Optional[int] = None,
    reserve: int = 0,
) -> dict[str, object]:
    """Fail closed unless the whole model and working buffers fit memory.

    This memory is all required, so it is refused only past the budget (the machine's physical
    memory or a smaller configured one), as the engine refuses a required allocation only at a
    real limit: the system makes room by compressing or paging other memory out. A runtime
    policy that keeps RESERVE bytes free also needs that much available beside the model now.
    """
    model = validate(raw)
    tensors = cast(int, model["tensor_bytes"])
    # FILE: the whole file mapping, locked. COPIED: the tensors read into a GPU backend's
    # buffers, plus the tied output's own copy of the token embedding. LOOKUPS_IN_FILE (Apple
    # silicon): every tensor but the per-layer embedding table, which stays in the file, unlocked,
    # its rows read as tokens use them; the GPU's tensors, the tied output among them, are copied
    # into a Metal buffer while the token embedding's CPU rows also stay in the file.
    resident = {
        FILE: cast(int, model["model_bytes"]),
        COPIED: tensors + cast(int, model["embedding_bytes"]),
        LOOKUPS_IN_FILE: tensors - cast(int, model["ple_bytes"]),
    }[weights]
    kv = sum(
        ((min(context, window + batch) if window else context) + 255) // 256 * 256 * width
        for window, width in cast(list[list[int]], model["kv_layers"])
    )
    required = resident + kv + WORKSPACE_BYTES
    if required > budget:
        raise ValueError(
            f"RAM-only model needs {required / 2**30:.2f} GiB at context {context}, "
            f"more than the memory budget of {budget / 2**30:.2f} GiB. "
            "Reduce context; SSD offload is not enabled for this model."
        )
    if reserve and (available is None or required + reserve > available):
        shown = "unknown" if available is None else f"{available / 2**30:.2f} GiB"
        raise ValueError(
            f"RAM-only model needs {required / 2**30:.2f} GiB at context {context} and the "
            f"runtime policy keeps {reserve / 2**30:.2f} GiB free; available {shown}. "
            "Free memory or reduce context; SSD offload is not enabled for this model."
        )
    return {
        "weights": weights,
        "weights_bytes": resident,
        "kv_bytes": kv,
        "workspace_bytes": WORKSPACE_BYTES,
        "required_bytes": required,
        "reserve_bytes": reserve,
        "available_bytes": available,
        "budget_bytes": budget,
    }
