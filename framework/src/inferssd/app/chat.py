# SPDX-License-Identifier: Apache-2.0
"""Small streaming terminal client for the server owned by this invocation."""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Optional

THINKING_OFF: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": False}}


def _opener() -> urllib.request.OpenerDirector:
    # Ignore proxy environment variables for the private loopback server.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def least_thinking(port: int) -> dict[str, Any]:
    """The request fields for the least thinking the model's chat template offers, as /props
    reports it: thinking off where the template can turn it off, or its first reasoning_effort
    where it always thinks, as Muse's does. The terminal shows only the answer."""
    try:
        with _opener().open(f"http://127.0.0.1:{port}/props", timeout=30) as response:
            reasoning = json.loads(response.read()).get("reasoning")
    except (OSError, ValueError, AttributeError):
        reasoning = None
    if isinstance(reasoning, dict) and reasoning.get("can_turn_off") is False:
        efforts = reasoning.get("efforts") or []
        return {"reasoning_effort": efforts[0]} if efforts else {}
    return THINKING_OFF


def completion(
    port: int,
    messages: list[dict[str, str]],
    max_tokens: int,
    thinking: Optional[dict[str, Any]] = None,
) -> str:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json.dumps(
            {
                "messages": messages,
                "stream": True,
                "max_tokens": max_tokens,
                **(THINKING_OFF if thinking is None else thinking),
            }
        ).encode(),
        {"Content-Type": "application/json"},
    )
    result: list[str] = []
    with _opener().open(request, timeout=7200) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                print(flush=True)
                return "".join(result)
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(str(event["error"]))
            choices = event.get("choices", [])
            if choices:
                content = choices[0].get("delta", {}).get("content") or ""
                result.append(content)
                print(content, end="", flush=True)
    raise RuntimeError("server closed the response before completing the answer")


def chat(port: int, prompt: Optional[str], max_tokens: int) -> int:
    messages: list[dict[str, str]] = []
    thinking = least_thinking(port)
    if prompt is not None:
        completion(port, [{"role": "user", "content": prompt}], max_tokens, thinking)
        return 0
    print("InferSSD chat. /exit to quit, /clear to start a new conversation.")
    while True:
        try:
            text = input("\n> ").strip()
        except EOFError:
            print()
            return 0
        if text == "/exit":
            return 0
        if text == "/clear":
            messages.clear()
            continue
        if not text:
            continue
        messages.append({"role": "user", "content": text})
        answer = completion(port, messages, max_tokens, thinking)
        messages.append({"role": "assistant", "content": answer})
