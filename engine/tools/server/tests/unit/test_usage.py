# SPDX-License-Identifier: Apache-2.0
# Usage events written to INFERSSD_USAGE_DIR; see docs/usage-events.md.
import json
import threading
import time

import pytest
import requests

from utils import *

server: ServerProcess


@pytest.fixture(autouse=True)
def create_server(tmp_path, monkeypatch):
    global server
    monkeypatch.setenv("INFERSSD_USAGE_DIR", str(tmp_path / "usage"))
    server = ServerPreset.tinyllama2()


def usage_path(tmp_path):
    return tmp_path / "usage" / "usage-events.jsonl"


def read_events(tmp_path) -> list[dict]:
    path = usage_path(tmp_path)
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    assert text == "" or text.endswith("\n")
    return [json.loads(line) for line in text.splitlines()]


def wait_finalized(tmp_path, n: int, timeout: float = 10) -> list[dict]:
    """The events once n requests have their usage_finalized line."""
    deadline = time.time() + timeout
    while True:
        events = read_events(tmp_path)
        if sum(e["type"] == "usage_finalized" for e in events) >= n or time.time() > deadline:
            return events
        time.sleep(0.05)


def single_request(events: list[dict]) -> tuple[dict, dict]:
    assert [e["type"] for e in events] == ["request_received", "usage_finalized"], events
    received, finalized = events
    assert received["request_id"] == finalized["request_id"]
    assert set(received) == {"event_id", "type", "request_id", "occurred_at"}
    assert set(finalized) == {
        "event_id", "type", "request_id", "occurred_at", "token_count", "input_tokens", "output_tokens", "outcome"
    }
    run, seq = received["event_id"].rsplit(":", 1)
    assert finalized["event_id"] == f"{run}:{int(seq) + 1}"
    for event in events:
        assert event["occurred_at"].endswith("Z")
        time.strptime(event["occurred_at"], "%Y-%m-%dT%H:%M:%SZ")
    assert finalized["token_count"] == finalized["input_tokens"] + finalized["output_tokens"]
    return received, finalized


def test_completion_counts_its_usage(tmp_path):
    server.start()
    res = server.make_request("POST", "/completion", data={"prompt": "I believe the meaning of life is", "n_predict": 8})
    assert res.status_code == 200
    _, finalized = single_request(wait_finalized(tmp_path, 1))
    assert finalized["outcome"] == "completed"
    assert finalized["input_tokens"] == res.body["tokens_evaluated"]
    assert finalized["output_tokens"] == res.body["tokens_predicted"] == 8


def test_streamed_chat_counts_its_usage(tmp_path):
    server.start()
    usage = None
    for chunk in server.make_stream_request("POST", "/v1/chat/completions", data={
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "What is the best book"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }):
        usage = chunk.get("usage") or usage
    assert usage is not None
    _, finalized = single_request(wait_finalized(tmp_path, 1))
    assert finalized["outcome"] == "completed"
    assert finalized["input_tokens"] == usage["prompt_tokens"]
    assert finalized["output_tokens"] == usage["completion_tokens"] == 8


def test_messages_count_cached_prompt_tokens_as_input(tmp_path):
    server.start()
    body = {"model": "x", "max_tokens": 8, "messages": [{"role": "user", "content": "What is the best book"}]}
    for _ in range(2):
        res = server.make_request("POST", "/v1/messages", data=body)
        assert res.status_code == 200
    events = wait_finalized(tmp_path, 2)
    finalized = [e for e in events if e["type"] == "usage_finalized"]
    usage = res.body["usage"]
    # Anthropic's input_tokens leaves out the cache hits; the event counts the whole prompt
    assert finalized[-1]["input_tokens"] == usage["input_tokens"] + usage.get("cache_read_input_tokens", 0)
    assert finalized[-1]["output_tokens"] == usage["output_tokens"]


def test_responses_are_counted(tmp_path):
    server.start()
    res = server.make_request("POST", "/v1/responses", data={"input": "What is the best book", "max_output_tokens": 8})
    assert res.status_code == 200
    _, finalized = single_request(wait_finalized(tmp_path, 1))
    assert finalized["outcome"] == "completed"
    assert finalized["input_tokens"] == res.body["usage"]["input_tokens"]
    assert finalized["output_tokens"] == res.body["usage"]["output_tokens"]


@pytest.mark.parametrize("path", ["/completion", "/completions", "/v1/completions"])
def test_every_completion_route_is_counted(tmp_path, path):
    server.start()
    res = server.make_request("POST", path, data={"prompt": "I believe", "max_tokens": 4, "n_predict": 4})
    assert res.status_code == 200
    single_request(wait_finalized(tmp_path, 1))


def test_candidates_add_up_and_share_the_prompt(tmp_path):
    server.start()
    res = server.make_request("POST", "/v1/chat/completions", data={
        "max_tokens": 8,
        "n": 2,
        "messages": [{"role": "user", "content": "What is the best book"}],
    })
    assert res.status_code == 200
    assert len(res.body["choices"]) == 2
    _, finalized = single_request(wait_finalized(tmp_path, 1))
    assert finalized["input_tokens"] == res.body["usage"]["prompt_tokens"]
    assert finalized["output_tokens"] == 16


def test_prompts_of_one_request_add_up(tmp_path):
    server.start()
    res = server.make_request("POST", "/completion", data={"prompt": ["I believe", "The sky is"], "n_predict": 4})
    assert res.status_code == 200
    _, finalized = single_request(wait_finalized(tmp_path, 1))
    assert finalized["input_tokens"] == sum(r["tokens_evaluated"] for r in res.body)
    assert finalized["output_tokens"] == 8


def test_unreadable_request_counts_with_no_tokens(tmp_path):
    server.start()
    res = requests.post(f"http://{server.server_host}:{server.server_port}/v1/chat/completions", data="{")
    assert res.status_code >= 400
    _, finalized = single_request(wait_finalized(tmp_path, 1))
    assert finalized["outcome"] == "failed"
    assert finalized["input_tokens"] == finalized["output_tokens"] == 0


def test_other_routes_are_not_counted(tmp_path):
    server.start()
    base = f"http://{server.server_host}:{server.server_port}"
    requests.get(base + "/health")
    requests.get(base + "/v1/models")
    requests.get(base + "/props")
    requests.post(base + "/tokenize", json={"content": "hello"})
    requests.post(base + "/apply-template", json={"messages": [{"role": "user", "content": "hi"}]})
    requests.post(base + "/v1/messages/count_tokens", json={"model": "x", "messages": [{"role": "user", "content": "hi"}]})
    time.sleep(0.5)
    assert read_events(tmp_path) == []


def endless_stream(chunks: int, ready: threading.Event | None = None) -> int:
    """Stream a completion that would not end by itself and leave after the given number of chunks."""
    response = requests.post(f"http://{server.server_host}:{server.server_port}/completion", json={
        "prompt": "Once upon a time",
        "n_predict": -1,
        "ignore_eos": True,
        "stream": True,
    }, stream=True)
    seen = 0
    for line in response.iter_lines():
        if line.startswith(b"data: "):
            seen += 1
            if seen == chunks:
                if ready is not None:
                    ready.set()
                    continue
                break
    response.close()
    return seen


def test_cancelled_stream_keeps_its_counts(tmp_path):
    server.enable_ctx_shift = True
    server.n_predict = -1
    server.start()
    seen = endless_stream(5)
    _, finalized = single_request(wait_finalized(tmp_path, 1))
    assert finalized["outcome"] == "cancelled"
    assert finalized["input_tokens"] > 0
    assert finalized["output_tokens"] >= seen


def test_stopping_the_server_finishes_open_requests(tmp_path):
    server.enable_ctx_shift = True
    server.n_predict = -1
    server.start()
    ready = threading.Event()

    def stream():
        try:
            endless_stream(5, ready)
        except requests.RequestException:
            pass  # the server goes away mid-stream

    client = threading.Thread(target=stream, daemon=True)
    client.start()
    assert ready.wait(30)
    server.stop()
    _, finalized = single_request(read_events(tmp_path))
    assert finalized["outcome"] == "cancelled"
    assert finalized["output_tokens"] >= 5


def test_nothing_is_written_without_the_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("INFERSSD_USAGE_DIR")
    server.start()
    res = server.make_request("POST", "/completion", data={"prompt": "I believe", "n_predict": 4})
    assert res.status_code == 200
    assert not (tmp_path / "usage").exists()
