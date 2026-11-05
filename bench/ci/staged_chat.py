"""Fixed prompt recorder outside candidate stages, including Linux net namespaces.

Only the independently owned HTTP client supplies receive timestamps. Replaying
its captured SSE never replaces those timestamps with the coordinator's clock.
The production final benchmark entry point and protocol remain unchanged.
"""

import base64
import http.client
import json
import socket
from pathlib import Path
import time

import final_chat
import stream_chat


MAX_WIRE_BYTES = 512 * 1024


def local_request(port, method, path, body, timeout):
    """Mac's trusted coordinator uses the broker-verified fixed loopback port."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    encoded = None if body is None else json.dumps(body, allow_nan=False).encode()
    started = time.monotonic_ns()
    chunks, size = [], 0
    try:
        connection.request(method, path, encoded,
                           {"Content-Type": "application/json", "Connection": "close"})
        response = connection.getresponse()
        while True:
            line = response.readline(65536)
            if not line:
                break
            elapsed = time.monotonic_ns() - started
            size += len(line)
            if size > MAX_WIRE_BYTES or elapsed > timeout * 10**9:
                raise ValueError("trusted HTTP response exceeded byte/deadline bound")
            chunks.append(dict(elapsed_ns=elapsed,
                               data_b64=base64.b64encode(line).decode()))
        return dict(status=response.status, headers=response.getheaders(),
                    started_monotonic_ns=started, ended_monotonic_ns=time.monotonic_ns(),
                    chunks=chunks)
    except (TimeoutError, socket.timeout) as error:
        # Match the isolated Linux client's transport receipt. Only the explicit
        # startup-health path may tolerate it; generation is never retried.
        return dict(status=0, headers=[], transport_error=f"TimeoutError: {error}",
                    started_monotonic_ns=started, ended_monotonic_ns=time.monotonic_ns(),
                    chunks=chunks)
    finally:
        connection.close()


def validate_response(response, allow_statuses=(200,)):
    if response.get("status") not in allow_statuses:
        raise ValueError("native HTTP request did not succeed")
    start, end = response["started_monotonic_ns"], response["ended_monotonic_ns"]
    if type(start) is not int or type(end) is not int or not 0 < start < end:
        raise ValueError("trusted HTTP clock bounds are invalid")
    previous, size, chunks = 0, 0, []
    for chunk in response["chunks"]:
        elapsed = chunk["elapsed_ns"]
        if type(elapsed) is not int or not previous <= elapsed <= end - start:
            raise ValueError("trusted HTTP receive times are not ordered")
        data = base64.b64decode(chunk["data_b64"], validate=True)
        size += len(data)
        if size > MAX_WIRE_BYTES:
            raise ValueError("trusted HTTP wire bound exceeded")
        chunks.append((start + elapsed, data))
        previous = elapsed
    return chunks


class NativeHTTP:
    def __init__(self, session, stage, port):
        self.session, self.stage, self.port = session, stage, port
        self.completions = 0

    def request(self, method, path, body, timeout, *, allow_health_timeout=False):
        if allow_health_timeout and (method, path, body) != ("GET", "/health", None):
            raise ValueError("only startup health checks may tolerate a transport timeout")
        if path not in ("/health", "/apply-template", "/props", "/completion"):
            raise ValueError("unexpected native benchmark endpoint")
        if path == "/completion":
            if self.completions:
                raise ValueError("one completion per fresh server; no retries")
            self.completions += 1
        value = self.session.call(self.stage, "stage.client",
                                  request=dict(port=self.port, method=method, path=path,
                                               body=body, timeout_s=timeout),
                                  timeout=timeout + 10)
        if "url" in value:
            if value["url"] != "http://127.0.0.1:" + str(self.port):
                raise ValueError("broker returned an unexpected loopback endpoint")
            if set(value.get("listener_pids", ())) != {self.stage.worker_pid} or any(
                value.get(key) != getattr(self.stage, key) for key in
                ("worker_pid", "worker_uid", "birth_identity", "scope_identity")
            ):
                raise ValueError("loopback listener differs from the owned native server")
            value = local_request(self.port, method, path, body, timeout)
        error = value.get("transport_error", "")
        if (allow_health_timeout and value.get("status") == 0 and isinstance(error, str)
                and (error.startswith("TimeoutError:")
                     or error == "trusted HTTP client exceeded request deadline")):
            # This is an unsuccessful readiness observation, never a completion
            # result. The caller retains it and checks the live stage/deadline.
            validate_response(value, (0,))
        else:
            validate_response(value, (200, 503) if path == "/health" else (200,))
        return value

    def json(self, method, path, body, timeout):
        response = self.request(method, path, body, timeout)
        return stream_chat.decode_json(b"".join(data for _, data in validate_response(response)))


class TimedLines:
    def __init__(self, response):
        self.chunks = validate_response(response)
        self.timestamp = response["started_monotonic_ns"]

    def __iter__(self):
        pending = b""
        for timestamp, data in self.chunks:
            pending += data
            while b"\n" in pending:
                line, _, pending = pending.partition(b"\n")
                self.timestamp = timestamp
                yield line + b"\n"
            if len(pending) > 1024 * 1024:
                raise ValueError("SSE line exceeds protocol bound")
        if pending:
            self.timestamp = self.chunks[-1][0]
            yield pending


def record(http, tier, output, timeout):
    """Write the same raw result/protocol/SSE ledger accepted by run_final."""
    final_chat.bounded_timeout(timeout)
    raw = final_chat.prompt_path(tier).read_bytes()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stem = output.with_suffix("")
    paths = {name: stem.with_name(stem.name + suffix) for name, suffix in (
        ("protocol", ".protocol.json"), ("events", ".events.jsonl"), ("sse", ".events.sse"))}
    if output.exists() or any(path.exists() for path in paths.values()):
        raise ValueError("result or raw evidence already exists")
    protocol = dict(prompt_sha256=stream_chat.digest(raw), prompt_bytes=len(raw),
                    input_tokens=None, output_tokens=256, base_url=f"http://127.0.0.1:{http.port}",
                    cache_prompt=False, clock="time.monotonic_ns",
                    mode="native-completion-of-rendered-chat", timeout_s=timeout,
                    protocol=final_chat.PROTOCOL, tier=tier, samples_per_tier=1,
                    cache_condition="fresh-server-first-completion; OS page cache recorded by wrapper",
                    decode_intervals={"first128": 127, "second128": 128, "overall256": 255})
    result = dict(status="failed", prompt_sha256=protocol["prompt_sha256"])
    try:
        props = http.json("GET", "/props", None, timeout)
        chat = stream_chat.chat_request(raw.decode("utf-8"), props)
        protocol["chat_request"] = chat
        templated = http.json("POST", "/apply-template", chat, timeout)
        rendered, prompt_ids = templated["prompt"], templated["tokens"]
        if not isinstance(rendered, str):
            raise ValueError("template response lacks rendered text")
        context = final_chat.validate_context(prompt_ids, props)
        request = dict(prompt=prompt_ids, n_predict=256, stream=True, temperature=0,
                       top_k=1, samplers=["top_k"], seed=0, n_probs=5,
                       post_sampling_probs=False, ignore_eos=True, cache_prompt=False,
                       return_tokens=True, timings_per_token=True)
        protocol.update(server_props=props, server_context_tokens=context,
                        input_tokens=len(prompt_ids), prompt_ids=prompt_ids, request=request,
                        rendered_prompt_sha256=stream_chat.digest(rendered.encode()))
        stream_chat.save(paths["protocol"], protocol)
        response = http.request("POST", "/completion", request, timeout)
        lines, records = TimedLines(response), []
        with paths["events"].open("x") as events, paths["sse"].open("xb") as wire:
            for event in stream_chat.events(lines, wire):
                row = dict(monotonic_ns=lines.timestamp, event=event)
                events.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                records.append(row)
        summary = stream_chat.summarize(records, response["started_monotonic_ns"],
                                        response["ended_monotonic_ns"], prompt_ids, 256)
        final_chat.validate_timings(summary)
        result.update(summary, status="passed", prompt_ids=prompt_ids,
                      rendered_prompt_sha256=protocol["rendered_prompt_sha256"])
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        if not paths["protocol"].exists():
            stream_chat.save(paths["protocol"], protocol)
        stream_chat.save(output, result)
    return result
