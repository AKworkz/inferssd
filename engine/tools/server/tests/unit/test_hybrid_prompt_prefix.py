# SPDX-License-Identifier: Apache-2.0
# Prompt reuse for a model whose memory cannot be rolled back (hybrid or recurrent, such as Qwen3.6), checked against
# a running server of one with a prompt cache, as `inferssd serve` starts it:
#   DEBUG_EXTERNAL=1 PORT=<port> pytest unit/test_hybrid_prompt_prefix.py
# Without DEBUG_EXTERNAL the tests are skipped: there is no hybrid model among the test presets.

import uuid

import pytest

from utils import *

server: ServerProcess

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a text file from the working folder.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "list_folder", "description": "List the files in a folder of the working folder.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
]


@pytest.fixture(autouse=True)
def create_server():
    global server
    if "DEBUG_EXTERNAL" not in os.environ:
        pytest.skip("needs a running server of a hybrid model with a prompt cache (DEBUG_EXTERNAL, PORT)")
    server = ServerProcess()


def system_prompt() -> str:
    # New to the server on every run, so the first conversation computes the prefix itself.
    rules = "\n".join(f"Rule {i}: answer the user's question {i} in plain words and keep each answer short."
                      for i in range(1, 61))
    return f"Session {uuid.uuid4()}. You are a careful assistant.\n{rules}"


def ask(messages: list) -> dict:
    body = {"messages": messages, "tools": TOOLS, "max_tokens": 8, "cache_prompt": True}
    reasoning = server.make_request("GET", "/props").body.get("reasoning") or {}
    if not reasoning or reasoning.get("can_turn_off"):
        body["reasoning_effort"] = "none"
    res = server.make_request("POST", "/v1/chat/completions", data=body, timeout=600)
    assert res.status_code == 200, res.body
    return res.body


def test_new_conversation_reuses_the_shared_prefix():
    system = {"role": "system", "content": system_prompt()}
    first = ask([system, {"role": "user", "content": "What is two plus two?"}])
    assert first["timings"]["prompt_n"] > 600
    # Another conversation behind the same system prompt and tools, after the first one left the slot.
    second = ask([system, {"role": "user", "content": "Name one primary color."}])
    timings = second["timings"]
    assert timings["prompt_n"] < 64, timings
    assert timings["cache_n"] > first["timings"]["prompt_n"] - 64, timings


def test_next_turn_prefills_only_the_new_turn():
    messages = [{"role": "system", "content": system_prompt()}, {"role": "user", "content": "What is two plus two?"}]
    first = ask(messages)
    messages += [{"role": "assistant", "content": first["choices"][0]["message"].get("content") or ""},
                 {"role": "user", "content": "And three plus three?"}]
    timings = ask(messages)["timings"]
    assert timings["prompt_n"] < 64, timings
    # The checkpoint 4 tokens before the end of the first prompt, or the one after the shared prefix when the first
    # question is shorter than the 64 tokens checkpoints keep between them.
    assert timings["cache_n"] > first["timings"]["prompt_n"] - 64, timings
