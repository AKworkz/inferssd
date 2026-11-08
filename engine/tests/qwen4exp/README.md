# Qwen4Exp CPU integration fixture

The fixture uses deterministic small weights, Q8_0 routed experts (12 experts,
top-10), BF16 QSA indexer projections, four HC streams, three GDN layers and one
QSA layer. PLE has 16 hash heads, a quantized mmap table and nonzero dilated
convolution weights. This checks implementation behavior; it is not a model
quality or speed benchmark.

Generate it with Python and NumPy:

```sh
python3 tests/qwen4exp/make_fixture.py /tmp/qwen4exp-fixture.gguf
```

To exercise the optional boolean recurrent-layer metadata array, add
`--recurrent-array` after the output path. The weights and layer pattern are
identical.

Build the CPU library, then compile the probe against that build:

```sh
c++ -std=c++17 -O2 tests/qwen4exp/probe.cpp -Iinclude -Iggml/include \
  -Lbuild-cpu/bin -Wl,-rpath,"$PWD/build-cpu/bin" -lllama -lggml-base -lggml \
  -o /tmp/qwen4exp-probe
/tmp/qwen4exp-probe /tmp/qwen4exp-fixture.gguf /path/to/text.txt /tmp/logits.bin 8 0
```

The arguments are the fixture, a text file, the logits output, the prefill
chunk size and flash attention (0 or 1). The probe uses the first 48 bytes of
the text as IDs in its fixture byte vocabulary and writes all 256 logits at
every position. It then checks whole-context restore, sequence restore with
other occupied cells, and sequence copy. Chunk sizes 48, 8 and 1 cover prefill
and decode.

The probe also compiles against llama.cpp eaf9376 with `-DQWEN_UPSTREAM`, which
adapts it to that revision's load-mode API. Compare the logits files from the
same fixture and text. That revision's indexer update context omits a pending
separate-stream sequence copy; later upstream revisions, whose graph and
indexer memory this engine follows, apply the indexer update alongside
attention.

Environment variables:

| Variable | Effect |
|---|---|
| `QWEN_TOKEN_COUNT` | Number of text bytes used as tokens (default 48, at most 96). |
| `QWEN_UNIFIED=0` | Uses separate KV streams. |
| `QWEN_PROBE_ONLY=1` | Writes the logits and skips the state checks. |
| `QWEN_STREAMED=1` | Loads with deferred experts and one sequence, for runs with SSD expert sources. |
| `QWEN_PREFILL_TOKENS` | Prefill length; the remaining tokens decode one at a time. |
| `QWEN_WARMUP=1` | Decodes 24 tokens, then clears the model memory before the measured run. |
| `QWEN_GPU_LAYERS` | Layers to offload (default 0). |
| `QWEN_UBATCH` | Microbatch size (default 64); the input batch capacity stays 64. |
| `QWEN_LAST_LOGIT_ONLY=1` | Requests logits only for the last token of each input chunk. With chunk 48 and `QWEN_UBATCH=8`, intermediate prompt ubatches request no logits, as in a server prefill. |
| `QWEN_ROUTE_AUDIT` | Path of a JSONL file that records each layer's selected experts and normalized weights. |

Compile with `-DQWEN_INTERNAL -Isrc -lggml-base` to check model hparams, print
the mapped PLE address, and confirm with `QWEN_STREAMED=1` that deferred expert
tensors stay in mapped CPU buffers, never repacked or uploaded copies. Adding
`QWEN_LOAD_ONLY=1` stops after that load check.
