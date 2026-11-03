# SPDX-License-Identifier: Apache-2.0
"""The models InferSSD supports, with the header facts of their tested quantization.

Each model's facts are ``model_facts_json`` of its shards' GGUF headers at the
revision named, so a speed estimate needs no download. ``lookup_bytes`` counts
the dense tensors a token reads one row of rather than whole: the token
embeddings and DeepSeek-V4.1's engram tables.
"""

from __future__ import annotations

from dataclasses import dataclass

from inferssd.strategy.inputs import ModelFacts, model_facts_from


@dataclass(frozen=True)
class Model:
    repo: str
    quantization: str
    revision: str  # the Hugging Face commit the facts were read at
    facts: ModelFacts
    lookup_bytes: int

    @property
    def dense_read_bytes(self) -> int:
        """The dense weights one token reads whole."""
        return self.facts.dense_bytes - self.lookup_bytes


def _shards(prefix: str, count: int) -> list[str]:
    return [f"{prefix}-{index:05d}-of-{count:05d}.gguf" for index in range(1, count + 1)]


MODELS: tuple[Model, ...] = (
    Model(
        "mrfakename/DeepSeek-V4.1-Flash-1017-GGUF",
        "Q8_0",
        "1df54e7d1b1e442051ad393ce871870a929f243e",
        model_facts_from(
            {
                "paths": _shards("Q8_0/DeepSeek-V4.1-Flash-1017-Q8_0", 10),
                "architecture": "deepseek41",
                "layers": 40,
                "experts": 384,
                "experts_used": 6,
                "roles": {
                    "gate": {"formats": [["MXFP4", 40]], "rows": 2304, "cols": 5120},
                    "up": {"formats": [["MXFP4", 40]], "rows": 2304, "cols": 5120},
                    "down": {"formats": [["MXFP4", 40]], "rows": 5120, "cols": 2304},
                },
                "expert_bytes": 18800640,
                "expert_read_bytes": 18812928,
                "dense_bytes": 219170559968,
                "format_bytes": [["MXFP4", 288777830400]],
            }
        ),
        210226042400,
    ),
    Model(
        "unsloth/Muse-Spark-Open-816B-A42B-GGUF",
        "UD-Q4_K_XL",
        "c00670568b6ce5e56a247f67091247984df2a01e",
        model_facts_from(
            {
                "paths": _shards("UD-Q4_K_XL/Muse-Spark-Open-816B-A42B-UD-Q4_K_XL", 12),
                "architecture": "muse-spark",
                "layers": 62,
                "experts": 256,
                "experts_used": 8,
                "roles": {
                    "gate": {"formats": [["Q4_K", 62]], "rows": 4096, "cols": 4096},
                    "up": {"formats": [["Q4_K", 62]], "rows": 4096, "cols": 4096},
                    "down": {"formats": [["Q5_K", 62]], "rows": 4096, "cols": 4096},
                },
                "expert_bytes": 30408704,
                "expert_read_bytes": 30420992,
                "dense_bytes": 22842110976,
                "format_bytes": [["Q4_K", 299573968896], ["Q5_K", 183072980992]],
            }
        ),
        1758625792,
    ),
)
