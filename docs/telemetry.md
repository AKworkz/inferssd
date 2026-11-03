# Runtime JSONL v1

`ssd::telemetry::writer` owns one exclusively created UTF-8 JSONL file per
runtime/context. The context owner joins all users before destroying it. The
writer serializes each complete record under a mutex, flushes it, and assigns a
contiguous sequence beginning at zero. Invalid JSON, duplicate/reserved fields,
invalid event names, nonfinite numbers and malformed UTF-8 fail before
publication. I/O errors return status::io; after an I/O failure the stream stays
failed and refuses further publication. Callers must check every returned result.
No destructor exception is used as an error channel.

Each line has `v`, `ev` and `seq`, plus the event payload:

```json
{"v":1,"ev":"ssd.io.read","seq":1,"read_bytes":12345}
```

The first effective-settings publication uses `ev=ssd.settings`. Its `values`
object contains all canonical typed settings. Its matching `sources`
object contains `origin` (`cli|file|default`), `location` and the exact trailing
file `receipt`. The settings parser schema is the authority for value types,
units, bounds, producers and scopes. JSONL telemetry is distinct from the
human-readable launch log and is not parsed by searching old log markers.

`inferssd.telemetry.jsonl.read_events` rejects unsupported versions, malformed
JSON, duplicate keys, nonfinite values and sequence gaps/duplicates. Unknown
well-formed event kinds remain inspectable. `effective_settings` requires one
settings event, the complete registry-defined value/receipt set and typed values.
It consumes actual C++ writer output in the cross-language contract test.

The SSD core reads only two environment variables, both in
`telemetry/debug.cpp`: `INFERSSD_DEBUG=trace,profile,mrc,ledger` and
`INFERSSD_TRACE_DIR`. They select output channels and their destination only,
never placement, numerical kernels or policy. There is no process-global mutable
settings or telemetry owner.

The server option `--ssd-telemetry FILE` creates the context-owned writer; all
publishers are joined before it is destroyed. The writer publishes the effective
settings first, then `ssd.controller.begin` (see
[native observations](architecture/native-observations.md)).
`inferssd serve` and `cli` write this stream to `native-events.jsonl` in the
launch's run directory. Numerical equality with channels enabled and disabled is an engine
integration check, not a conclusion of the JSON protocol unit tests.
