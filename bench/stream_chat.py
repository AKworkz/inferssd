#!/usr/bin/env python3
"""Measure one short greedy chat through the native server token stream."""
import argparse
import hashlib
import json
import math
import _thread
import signal
import threading
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path


def decode_json(data):
    def invalid(value):
        raise ValueError(f"non-finite JSON value: {value}")
    return json.loads(data, parse_constant=invalid)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def deadline(seconds):
    """Raise TimeoutError in the calling (main) thread once SECONDS have passed.

    SIGALRM, where the OS has it, also interrupts a blocked read. Windows has
    no alarm signal: a timer interrupts the main thread between reads, each of
    which the socket timeout bounds.
    """
    return alarm_deadline(seconds) if hasattr(signal, "SIGALRM") else timer_deadline(seconds)


@contextmanager
def alarm_deadline(seconds):
    def expired(_signal, _frame):
        raise TimeoutError("whole request deadline expired")
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@contextmanager
def timer_deadline(seconds):
    state = {"expired": False, "done": False}
    lock = threading.Lock()

    def interrupted(_signal, _frame):
        if state["expired"]:
            raise TimeoutError("whole request deadline expired")
        raise KeyboardInterrupt

    def expire():
        with lock:
            if not state["done"]:
                state["expired"] = True
                _thread.interrupt_main()  # as a SIGINT

    previous = signal.signal(signal.SIGINT, interrupted)
    timer = threading.Timer(seconds, expire)
    timer.daemon = True
    timer.start()
    try:
        yield
    finally:
        with lock:
            state["done"] = True
        timer.cancel()
        signal.signal(signal.SIGINT, previous)


def request(url, body):
    return urllib.request.Request(url, json.dumps(body).encode(),
                                  {"Content-Type": "application/json"})


def post(base, endpoint, body, timeout):
    with urllib.request.urlopen(request(base + endpoint, body), timeout=timeout) as response:
        return decode_json(response.read())


def get(base, endpoint, timeout):
    with urllib.request.urlopen(base + endpoint, timeout=timeout) as response:
        return decode_json(response.read())


def chat_request(content, props):
    """The benchmark's chat request. Thinking is off where the chat template can turn it off; a template that
    always thinks, as Muse's does, takes its first reasoning_effort, its least. /props reports both."""
    chat = {"messages": [{"role": "user", "content": content}]}
    reasoning = props.get("reasoning") if isinstance(props, dict) else None
    if isinstance(reasoning, dict) and reasoning.get("can_turn_off") is False:
        if reasoning.get("efforts"):
            chat["reasoning_effort"] = reasoning["efforts"][0]
        return chat
    chat.update({"chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "none"})
    return chat


def events(response, wire=None):
    """Read SSE data records, including multiline data and comment heartbeats."""
    data = []
    size = 0
    done = False
    for line in response:
        if wire is not None:
            wire.write(line)
        if len(line) > 1024 * 1024:
            raise ValueError("SSE line exceeds 1 MiB")
        line = line.rstrip(b"\r\n")
        if not line:
            if data:
                payload = b"\n".join(data)
                data = []
                size = 0
                if done:
                    raise ValueError("data follows SSE DONE marker")
                if payload == b"[DONE]":
                    done = True
                    continue
                yield decode_json(payload)
        elif line.startswith(b"data:"):
            value = line[5:]
            value = value[1:] if value.startswith(b" ") else value
            size += len(value)
            if size > 1024 * 1024:
                raise ValueError("SSE event exceeds 1 MiB")
            data.append(value)
    if data:
        raise ValueError("unterminated SSE record")


def token_list(value):
    return isinstance(value, list) and all(type(v) is int and v >= 0 for v in value)


def validate_probability(value, token):
    """Validate the exact pre-sampling top-five record, without rounding it."""
    def entry(item):
        return (isinstance(item, dict) and type(item.get("id")) is int and item["id"] >= 0
                and type(item.get("logprob")) in (int, float)
                and math.isfinite(item["logprob"]) and item["logprob"] <= 0)

    if not entry(value) or value["id"] != token:
        raise ValueError("missing or mismatched token probabilities")
    top = value.get("top_logprobs")
    if not isinstance(top, list) or len(top) != 5 or not all(entry(item) for item in top):
        raise ValueError("expected five valid pre-sampling top probabilities")
    if len({item["id"] for item in top}) != 5:
        raise ValueError("duplicate top probability token IDs")
    if any(a["logprob"] < b["logprob"] for a, b in zip(top, top[1:])):
        raise ValueError("top probabilities are not sorted")
    for item in top:
        if item["id"] == token and item["logprob"] != value["logprob"]:
            raise ValueError("sampled token probability differs from its top entry")


def summarize(records, start_ns, end_ns, prompt_ids, output_tokens):
    tokens, probabilities, milestones = [], [], {}
    final = None
    for row in records:
        event, timestamp = row["event"], row["monotonic_ns"]
        if not isinstance(event, dict) or "error" in event:
            raise ValueError("server returned an invalid event or error")
        if final is not None:
            raise ValueError("data follows the final response")
        if event.get("stop") is True:
            final = event
            continue
        if event.get("prompt_progress"):
            continue
        values = event.get("tokens")
        # This server sends one native partial per sampled token, even when its
        # text is empty. Do not count text chunks or guess missing milestones.
        if not token_list(values) or len(values) != 1:
            raise ValueError("expected one token ID per native partial")
        count = len(tokens) + 1
        if (count > output_tokens or type(event.get("tokens_predicted")) is not int
                or event["tokens_predicted"] != count):
            raise ValueError("token indices are missing, repeated or out of order")
        if timestamp <= (milestones[count - 1] if count > 1 else start_ns):
            raise ValueError("token timestamps are not strictly increasing")
        probs = event.get("completion_probabilities")
        if not isinstance(probs, list) or len(probs) != 1:
            raise ValueError("missing or mismatched token probabilities")
        validate_probability(probs[0], values[0])
        tokens.extend(values)
        probabilities.extend(probs)
        milestones[count] = timestamp
    if not final or len(tokens) != output_tokens:
        raise ValueError("stream ended before the requested completion")
    timings = final.get("timings", {})
    if not isinstance(timings, dict):
        raise ValueError("final timings must be an object")
    if (final.get("stop_type") != "limit" or final.get("truncated") is not False
            or final.get("tokens_predicted") != output_tokens
            or final.get("tokens_evaluated") != len(prompt_ids)
            or timings.get("prompt_n") != len(prompt_ids)
            or timings.get("predicted_n") != output_tokens):
        raise ValueError("final counts, stop reason or context differ from the protocol")
    if end_ns < milestones[output_tokens]:
        raise ValueError("invalid stream end time")
    metrics = {"ttft_s": (milestones[1] - start_ns) / 1e9,
               "request_wall_s": (end_ns - start_ns) / 1e9}
    prompt_ms = timings.get("prompt_ms")
    if type(prompt_ms) not in (int, float) or not math.isfinite(prompt_ms) or prompt_ms <= 0:
        raise ValueError("invalid prompt timing")
    metrics["server_prefill_tps"] = len(prompt_ids) * 1000 / prompt_ms
    if output_tokens > 1:
        metrics["decode_tps"] = (output_tokens - 1) * 1e9 / (milestones[output_tokens] - milestones[1])
    if output_tokens >= 128:
        metrics["decode_first128_tps"] = 127 * 1e9 / (milestones[128] - milestones[1])
    if output_tokens >= 256:
        metrics["decode_second128_tps"] = 128 * 1e9 / (milestones[256] - milestones[128])
    return {"input_tokens": len(prompt_ids), "output_tokens": len(tokens), "metrics": metrics,
            "tokens": tokens, "probabilities": probabilities, "timings": timings,
            "milestones_ns": milestones, "start_ns": start_ns, "end_ns": end_ns}


def run(args):
    if not 1 <= args.input_tokens <= 1024 or not 1 <= args.output_tokens <= 512:
        raise ValueError("short protocol requires 1..1024 input and 1..512 output tokens")
    def validate_input(prompt_ids, _base):
        if not token_list(prompt_ids) or len(prompt_ids) != args.input_tokens:
            raise ValueError("rendered prompt token count differs from the fixed protocol")
        return {}
    return record_chat(args, validate_input)


def record_chat(args, validate_input, protocol_extra=None, validate_result=None):
    """Shared recorder; each entry point supplies its own input-size contract."""
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    raw = args.prompt.read_bytes()
    prompt = raw.decode("utf-8")
    out = args.output
    # The arm directory may exist, but never overwrite another measurement.
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = out.with_suffix("")
    paths = {kind: stem.with_name(stem.name + suffix) for kind, suffix in
             (("protocol", ".protocol.json"), ("events", ".events.jsonl"), ("sse", ".events.sse"))}
    if out.exists() or any(path.exists() for path in paths.values()):
        raise ValueError("result or evidence file already exists")
    protocol = {"prompt_sha256": digest(raw), "prompt_bytes": len(raw),
                "input_tokens": args.input_tokens, "output_tokens": args.output_tokens,
                "base_url": args.url.rstrip("/"), "cache_prompt": False,
                "clock": "time.perf_counter_ns", "mode": "native-completion-of-rendered-chat",
                "timeout_s": args.timeout}
    protocol.update(protocol_extra or {})
    result = {"status": "failed", "prompt_sha256": digest(raw)}
    try:
        with deadline(args.timeout):
            base = protocol["base_url"]
            chat = chat_request(prompt, get(base, "/props", args.timeout))
            protocol["chat_request"] = chat
            templated = post(base, "/apply-template", chat, args.timeout)
            rendered, prompt_ids = templated["prompt"], templated["tokens"]
            # The server retains message provenance when tokenizing a template.
            # Re-tokenizing flattened text could interpret user text as control tokens.
            if not isinstance(rendered, str):
                raise ValueError("template response must contain rendered text")
            protocol.update(validate_input(prompt_ids, base))
            protocol["input_tokens"] = len(prompt_ids)
            body = {"prompt": prompt_ids, "n_predict": args.output_tokens, "stream": True,
                    "temperature": 0, "top_k": 1, "samplers": ["top_k"],
                    "seed": 0, "n_probs": 5, "post_sampling_probs": False,
                    "ignore_eos": True, "cache_prompt": False, "return_tokens": True,
                    "timings_per_token": True}
            protocol.update({"rendered_prompt_sha256": digest(rendered.encode()),
                             "prompt_ids": prompt_ids, "request": body})
            save(paths["protocol"], protocol)
            records = []
            start_ns = time.perf_counter_ns()
            with (paths["events"].open("x", encoding="utf-8") as raw_events,
                  paths["sse"].open("xb") as wire,
                  urllib.request.urlopen(request(base + "/completion", body), timeout=args.timeout) as response):
                for event in events(response, wire):
                    row = {"monotonic_ns": time.perf_counter_ns(), "event": event}
                    raw_events.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    raw_events.flush()
                    records.append(row)
            end_ns = time.perf_counter_ns()
            summary = summarize(records, start_ns, end_ns, prompt_ids, args.output_tokens)
            if validate_result is not None:
                validate_result(summary)
            result.update(summary)
            result.update({"status": "passed", "rendered_prompt_sha256": protocol["rendered_prompt_sha256"],
                           "prompt_ids": prompt_ids})
    except Exception as error:  # noqa: BLE001 - retain failed measurement evidence for all request errors
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        if not paths["protocol"].exists():
            save(paths["protocol"], protocol)
        save(out, result)
    return 0 if result["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--input-tokens", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    try:
        return run(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
