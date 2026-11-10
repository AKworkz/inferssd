<!-- SPDX-License-Identifier: Apache-2.0 -->
# Usage events

[Back to documentation](README.md)

The server can record how many inference requests it served and how many tokens they used, for the host that
runs it. It writes nothing unless `INFERSSD_USAGE_DIR` names a directory. `inferssd serve` and `cli` pass the
variable through to the server unchanged.

## File

The server appends to `usage-events.jsonl` in that directory, creating both if needed (the file with mode 0600
where the OS has modes). Each record is one JSON object on one line that ends in `\n`, written with a single
append; a reader consumes complete lines only. One server process is one run; several runs may append to the
same file, and their event IDs never collide.

This version neither rotates nor prunes the file. After a write error the server stops recording for the rest
of the run and says so once in its log, so a partial line is never followed by another record.

## Records

```json
{"event_id":"Xc2kq0RmB7aTn4Lw:1","type":"request_received","request_id":"req-8b1KfQ2s9mZ0aLx3cV7nT4yR","occurred_at":"2026-11-10T04:59:58Z"}
{"event_id":"Xc2kq0RmB7aTn4Lw:2","type":"usage_finalized","request_id":"req-8b1KfQ2s9mZ0aLx3cV7nT4yR","occurred_at":"2026-11-10T05:00:03Z","token_count":840,"input_tokens":812,"output_tokens":28,"outcome":"completed"}
```

| Field | Meaning |
| --- | --- |
| `event_id` | `<run>:<seq>`. The run is random for each server process; `seq` starts at 1 and follows the file order. Unique across runs, so a reader that reads a line twice can drop it by this ID. |
| `type` | `request_received` or `usage_finalized`. |
| `request_id` | Random for each HTTP request; pairs its two records. |
| `occurred_at` | UTC, to the second, when the request reached its route or when its usage was final. |
| `input_tokens` | Prompt tokens, as the OpenAI-style `prompt_tokens` reports them: prompt-cache hits included. (Anthropic's `input_tokens` leaves the hits out.) |
| `output_tokens` | Generated tokens, reasoning included. |
| `token_count` | `input_tokens + output_tokens`. |
| `outcome` | `completed`, `failed` or `cancelled`. |

## What is counted

A request to one of these routes writes `request_received` as soon as its handler starts, before its body is
read, so a request that fails to parse is still counted:

- `/completion`, `/completions`, `/v1/completions`
- `/chat/completions`, `/v1/chat/completions`
- `/responses`, `/v1/responses`
- `/v1/messages`

Other routes are not counted, among them `/health`, `/models`, `/props`, `/metrics`, `/tokenize`,
`/apply-template` and `/v1/messages/count_tokens`. Neither are requests refused before they reach a route:
`503 Loading model` while the model loads, or a missing API key.

Each request gets at most one `usage_finalized`, written when all of its work has ended and its response is done:

- Several prompts in one request add up. Several candidates (`n`) share one prompt, so it counts once, and
  their outputs add up. `inferssd serve` runs one slot, so it refuses `n` above 1, which counts as a failed
  request with no tokens.
- A request that fails before it runs, such as one that does not parse, counts 0 and 0.
- A cancelled request keeps what it used: the prompt once a slot has taken it, and the tokens generated until the
  cancellation. One still waiting for a slot counts 0.
- When the server stops on SIGINT or SIGTERM, each open request is written as it stands, as `cancelled`. If the
  process is killed, a request may have only its `request_received` line; its usage is then unknown.
