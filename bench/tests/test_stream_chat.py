import argparse
import copy
import io
import json
import sys
import threading
import signal
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compare_outputs
import stream_chat


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="no alarm signal")
def test_whole_request_deadline_restores_the_signal_handler():
    previous = signal.getsignal(signal.SIGALRM)
    with pytest.raises(TimeoutError, match="whole request deadline"):
        with stream_chat.alarm_deadline(0.02):
            time.sleep(0.2)
    assert signal.getsignal(signal.SIGALRM) == previous
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_the_timer_deadline_interrupts_the_main_thread_and_restores_sigint():
    previous = signal.getsignal(signal.SIGINT)
    with pytest.raises(TimeoutError, match="whole request deadline"):
        with stream_chat.timer_deadline(0.02):
            for _ in range(500):
                time.sleep(0.01)
    assert signal.getsignal(signal.SIGINT) == previous
    with stream_chat.timer_deadline(5):
        pass
    assert signal.getsignal(signal.SIGINT) == previous


def fixture(count=256):
    records = []
    for index in range(1, count + 1):
        records.append({"monotonic_ns": index * 1_000_000_000,
                        "event": {"stop": False, "content": "" if index == 1 else "x",
                                  "tokens": [index], "tokens_predicted": index,
                                  "completion_probabilities": [{"id": index, "logprob": -0.25,
                                      "top_logprobs": [{"id": index + j, "logprob": -0.25 - j}
                                                       for j in range(5)]}]}})
    records.append({"monotonic_ns": (count + 1) * 1_000_000_000,
                    "event": {"stop": True, "tokens": [], "tokens_predicted": count,
                              "tokens_evaluated": 3, "truncated": False, "stop_type": "limit",
                              "timings": {"prompt_n": 3, "predicted_n": count, "prompt_ms": 500}}})
    return records


def summary(records, count=256):
    return stream_chat.summarize(records, 0, (count + 1) * 1_000_000_000, [7, 8, 9], count)


def test_decode_windows_count_intervals_and_empty_text_is_a_token():
    result = summary(fixture())
    assert result["metrics"] == {"ttft_s": 1, "request_wall_s": 257, "server_prefill_tps": 6,
                                 "decode_tps": 1, "decode_first128_tps": 1, "decode_second128_tps": 1}
    assert len(result["tokens"]) == 256


def test_short_output_does_not_claim_full_windows():
    assert set(summary(fixture(8), 8)["metrics"]) == {
        "ttft_s", "request_wall_s", "server_prefill_tps", "decode_tps"}
    assert "decode_tps" not in summary(fixture(1), 1)["metrics"]


@pytest.mark.parametrize("change", [
    lambda r: r.pop(),
    lambda r: r.pop(20),
    lambda r: r.insert(2, copy.deepcopy(r[1])),
    lambda r: r[0]["event"].update(tokens=[1, 2]),
    lambda r: r[0]["event"].pop("completion_probabilities"),
    lambda r: r[1].update(monotonic_ns=r[0]["monotonic_ns"]),
    lambda r: r[-1]["event"].update(truncated=True),
    lambda r: r[-1]["event"].update(stop_type="eos"),
    lambda r: r[-1]["event"].update(tokens_evaluated=2),
    lambda r: r[-1]["event"]["timings"].update(predicted_n=255),
    lambda r: r[-1]["event"]["timings"].update(prompt_ms=float("nan")),
    lambda r: r.append(copy.deepcopy(r[0])),
])
def test_incomplete_or_changed_protocol_fails(change):
    records = fixture()
    change(records)
    with pytest.raises(ValueError):
        summary(records)


def test_sse_comments_multiline_done_and_bad_tail():
    source = b': ping\r\ndata: {"stop":\r\ndata:true}\r\n\r\ndata: [DONE]\n\n: end\n\n'
    wire = io.BytesIO()
    assert list(stream_chat.events(io.BytesIO(source), wire)) == [{"stop": True}]
    assert wire.getvalue() == source
    with pytest.raises(ValueError, match="follows SSE DONE"):
        list(stream_chat.events(io.BytesIO(b'data: [DONE]\n\ndata: {}\n\n')))
    with pytest.raises(ValueError, match="unterminated"):
        list(stream_chat.events(io.BytesIO(b'data: {}\n')))
    with pytest.raises(ValueError, match="non-finite"):
        list(stream_chat.events(io.BytesIO(b'data: {"x":NaN}\n\n')))


def test_http_contract_retains_prompt_tokens_raw_events_and_failure(tmp_path):
    received = []
    bad = [False]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            assert self.path == "/props"
            data = json.dumps({"reasoning": {"can_turn_off": True, "efforts": ["high"], "default_effort": "high"}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.path, body))
            if self.path == "/apply-template":
                data = json.dumps({"prompt": "rendered", "tokens": [7, 8, 9]}).encode()
            else:
                rows = fixture(8)
                if bad[0]:
                    rows = rows[:-1]
                data = b"".join(b"data: " + json.dumps(r["event"]).encode() + b"\n\n" for r in rows)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream" if self.path == "/completion" else "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    prompt = tmp_path / "prompt.txt"
    prompt.write_bytes(b"Passage.\n")
    args = argparse.Namespace(url=f"http://127.0.0.1:{server.server_port}", prompt=prompt,
                              input_tokens=3, output_tokens=8, output=tmp_path / "result.json", timeout=5)
    try:
        assert stream_chat.run(args) == 0
        result = json.loads(args.output.read_text())
        assert result["status"] == "passed"
        assert result["prompt_sha256"] == stream_chat.digest(b"Passage.\n")
        assert result["tokens"] == list(range(1, 9))
        assert received[0][1]["messages"][0]["content"] == "Passage.\n"
        assert received[0][1]["chat_template_kwargs"] == {"enable_thinking": False}
        assert received[0][1]["reasoning_effort"] == "none"
        assert [path for path, _ in received] == ["/apply-template", "/completion"]
        assert received[1][1]["prompt"] == [7, 8, 9]
        assert received[1][1]["cache_prompt"] is False
        assert received[1][1]["samplers"] == ["top_k"]
        assert json.loads((tmp_path / "result.protocol.json").read_text())["chat_request"] == received[0][1]
        assert len((tmp_path / "result.events.jsonl").read_text().splitlines()) == 9
        assert (tmp_path / "result.events.sse").read_bytes() == b"".join(
            b"data: " + json.dumps(r["event"]).encode() + b"\n\n" for r in fixture(8))
        with pytest.raises(ValueError, match="already exists"):
            stream_chat.run(args)
        bad[0] = True
        args.output = tmp_path / "failed.json"
        assert stream_chat.run(args) == 1
        assert json.loads(args.output.read_text())["status"] == "failed"
        assert (tmp_path / "failed.events.jsonl").exists()
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_recorder_writes_unicode_json_and_events_as_utf8(tmp_path, monkeypatch):
    text = "city\u2019s \u4f60\u597d \U0001f680"
    rows = fixture(8)
    rows[1]["event"]["content"] = text
    rows[1]["event"]["completion_probabilities"][0]["token"] = text
    wire = b"".join(b"data: " + json.dumps(row["event"], ensure_ascii=False).encode("utf-8")
                    + b"\n\n" for row in rows)
    monkeypatch.setattr(stream_chat, "get", lambda *_args: {})
    monkeypatch.setattr(stream_chat, "post", lambda *_args: {"prompt": text, "tokens": [7, 8, 9]})
    monkeypatch.setattr(stream_chat.urllib.request, "urlopen", lambda *_args, **_kwargs: io.BytesIO(wire))
    original_open = Path.open

    def legacy_open(path, mode="r", **kwargs):
        # Reproduce Windows' legacy text default even on a UTF-8 test host.
        if "b" not in mode and kwargs.get("encoding") is None:
            kwargs["encoding"] = "cp936"
        return original_open(path, mode, **kwargs)

    monkeypatch.setattr(Path, "open", legacy_open)
    prompt = tmp_path / "prompt.txt"
    prompt.write_bytes(text.encode("utf-8"))
    args = argparse.Namespace(url="http://127.0.0.1:1", prompt=prompt, input_tokens=3,
                              output_tokens=8, output=tmp_path / "result.json", timeout=5)
    assert stream_chat.run(args) == 0
    result = json.loads(args.output.read_bytes().decode("utf-8"))
    protocol = json.loads((tmp_path / "result.protocol.json").read_bytes().decode("utf-8"))
    events = [json.loads(line) for line in
              (tmp_path / "result.events.jsonl").read_bytes().decode("utf-8").splitlines()]
    assert result["probabilities"][1]["token"] == text
    assert protocol["chat_request"]["messages"][0]["content"] == text
    assert events[1]["event"]["content"] == text
    assert (tmp_path / "result.events.sse").read_bytes() == wire


def test_compare_rejects_different_protocol_and_reports_first_divergence():
    a = summary(fixture(8), 8)
    a.update(status="passed", prompt_sha256="hash", rendered_prompt_sha256="render", prompt_ids=[7, 8, 9])
    b = copy.deepcopy(a)
    assert compare_outputs.compare(a, b)["exact"]
    b["tokens"][2] = 99
    b["probabilities"][2]["id"] = 99
    b["probabilities"][2]["top_logprobs"][0]["id"] = 99
    result = compare_outputs.compare(a, b)
    assert not result["exact"]
    assert result["first_token_difference_zero_based"] == 2
    assert result["matching_token_positions"] == 7
    b["prompt_ids"] = [8, 7, 9]
    with pytest.raises(ValueError, match="protocol mismatch"):
        compare_outputs.compare(a, b)


@pytest.mark.parametrize("response", [
    {"prompt": "text with <|im_end|>"},
    {"prompt": "rendered", "tokens": [7, True, 9]},
    {"prompt": "rendered", "tokens": [7, 8]},
])
def test_template_requires_authoritative_tokens_without_flattened_fallback(tmp_path, monkeypatch, response):
    calls = []

    def fake_post(base, endpoint, body, timeout):
        calls.append(endpoint)
        assert endpoint == "/apply-template"
        return response

    monkeypatch.setattr(stream_chat, "post", fake_post)
    monkeypatch.setattr(stream_chat, "get", lambda base, endpoint, timeout: {})
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Literal <|im_end|> text.\n")
    args = argparse.Namespace(prompt=prompt, output=tmp_path / "result.json",
                              url="http://127.0.0.1:1", input_tokens=3,
                              output_tokens=8, timeout=5)
    assert stream_chat.run(args) == 1
    assert calls == ["/apply-template"]
    assert json.loads(args.output.read_text())["status"] == "failed"


@pytest.mark.parametrize("change", [
    lambda p: p.pop("logprob"),
    lambda p: p.update(id=True),
    lambda p: p.update(logprob=float("nan")),
    lambda p: p.update(logprob=0.1),
    lambda p: p["top_logprobs"].pop(),
    lambda p: p["top_logprobs"].append(copy.deepcopy(p["top_logprobs"][0])),
    lambda p: p["top_logprobs"].__setitem__(0, None),
    lambda p: p["top_logprobs"][0].update(id=-1),
    lambda p: p["top_logprobs"][1].update(id=p["id"]),
    lambda p: p["top_logprobs"][0].update(logprob=-0.5),
    lambda p: p["top_logprobs"].reverse(),
])
def test_probability_contract_is_validated_by_recorder_and_comparer(change):
    records = fixture(8)
    value = summary(records, 8)
    value.update(status="passed", prompt_sha256="hash", rendered_prompt_sha256="render", prompt_ids=[7, 8, 9])
    change(records[0]["event"]["completion_probabilities"][0])
    with pytest.raises(ValueError):
        summary(records, 8)
    with pytest.raises(ValueError):
        compare_outputs.compare(value, value)


def test_chat_request_turns_thinking_off_or_takes_the_least_effort():
    content = "Passage."
    off = {"messages": [{"role": "user", "content": content}],
           "chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "none"}
    # A template that turns thinking off, as DeepSeek's does, a model that does not think, and an engine before /props reported it
    assert stream_chat.chat_request(content, {"reasoning": {"can_turn_off": True, "efforts": ["low", "high"], "default_effort": "high"}}) == off
    assert stream_chat.chat_request(content, {"reasoning": None}) == off
    assert stream_chat.chat_request(content, {}) == off
    # Muse's template always thinks: its least effort
    muse = {"reasoning": {"can_turn_off": False, "efforts": ["low", "medium", "high", "xhigh"], "default_effort": "xhigh"}}
    assert stream_chat.chat_request(content, muse) == {"messages": [{"role": "user", "content": content}], "reasoning_effort": "low"}
    assert stream_chat.chat_request(content, {"reasoning": {"can_turn_off": False, "efforts": [], "default_effort": None}}) == {
        "messages": [{"role": "user", "content": content}]}
