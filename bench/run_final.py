#!/usr/bin/env python3
"""Run the two fixed final-benchmark tiers once under the existing device lock."""
import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import time
import uuid
from pathlib import Path

import final_chat
import run_abba
import stream_chat

ORDER = ("512", "4096")


def text(value):
    return isinstance(value, str) and bool(value.strip())


def sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def argv(value):
    return isinstance(value, list) and bool(value) and all(text(word) for word in value)


def object_field(value, field):
    result = value.get(field) if isinstance(value, dict) else None
    if not isinstance(result, dict):
        raise ValueError(f"receipt needs a {field} object")
    return result


def load_config(path):
    config = stream_chat.decode_json(path.read_bytes())
    if not isinstance(config, dict) or any(not text(config.get(k)) for k in ("device", "model")):
        raise ValueError("config requires device and model labels")
    if not isinstance(config.get("revision"), str) or not re.fullmatch(r"[0-9a-f]{40}", config["revision"]):
        raise ValueError("config requires the full final source revision")
    if not argv(config.get("argv")):
        raise ValueError("config requires one explicit wrapper argv shared by both tiers")
    config.setdefault("request_timeout_s", final_chat.DEFAULT_TIMEOUT)
    config.setdefault("arm_timeout_s", 3600)
    for name in ("request_timeout_s", "arm_timeout_s"):
        final_chat.bounded_timeout(config[name])
    if config["arm_timeout_s"] <= config["request_timeout_s"]:
        raise ValueError("arm deadline must exceed request deadline for loading and cleanup")
    return config


def validate_receipt(path, config, tier):
    """Validate the wrapper contract, not the truth of host-side assertions."""
    receipt = stream_chat.decode_json(path.read_bytes())
    if not isinstance(receipt, dict):
        raise ValueError("wrapper receipt must be an object")
    expected = {"schema_version": 1, "status": "passed", "protocol": final_chat.PROTOCOL,
                "tier": tier, "device": config["device"], "model": config["model"],
                "revision": config["revision"]}
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError("wrapper receipt identity/status differs from this arm")
    server = object_field(receipt, "server")
    if (type(server.get("pid")) is not int or server["pid"] <= 0
            or not text(server.get("start_identity"))
            or not argv(server.get("argv")) or server.get("fresh_process") is not True
            or type(server.get("completion_requests")) is not int or server["completion_requests"] != 1
            or type(server.get("returncode")) is not int or server["returncode"] != 0
            or server.get("owned_processes_gone") is not True
            or server.get("resource_scope_gone") is not True):
        raise ValueError("wrapper must confirm one fresh-server completion and clean shutdown")
    build = object_field(receipt, "build")
    if (build.get("revision") != config["revision"] or not sha(build.get("binary_sha256"))
            or build.get("actual_loaded_libraries_verified") is not True):
        raise ValueError("wrapper build and actual loaded-library identity are required")
    libraries = build.get("loaded_libraries")
    if (not isinstance(libraries, list) or not libraries
            or any(not isinstance(x, dict) or not text(x.get("path")) or not sha(x.get("sha256"))
                   for x in libraries)):
        raise ValueError("wrapper needs actual loaded library paths and hashes")
    model = object_field(receipt, "model_identity")
    shards = model.get("shards")
    if (model.get("verified") is not True or not isinstance(shards, list) or not shards
            or any(not isinstance(x, dict) or not text(x.get("path")) or not sha(x.get("sha256"))
                   or type(x.get("bytes")) is not int or x["bytes"] <= 0 for x in shards)):
        raise ValueError("wrapper needs verified identities for every model shard")
    host = object_field(receipt, "host")
    if (not text(host.get("hardware")) or not text(host.get("os"))
            or type(host.get("memory_budget_bytes")) is not int or host["memory_budget_bytes"] <= 0
            or host.get("resource_checks_passed") is not True
            or not isinstance(host.get("effective_settings"), dict) or not host["effective_settings"]
            or not text(host.get("os_page_cache_treatment"))):
        raise ValueError("wrapper needs hardware, effective budgets/settings and page-cache treatment")
    settings = host["effective_settings"]
    if settings.get("source_mode") != "direct-gguf":
        raise ValueError("final benchmark requires direct GGUF sources, not legacy packs")
    if ("h2d_limit_bytes_per_second" not in settings
            or settings["h2d_limit_bytes_per_second"] is not None):
        raise ValueError("final benchmark requires unrestricted H2D transfers")
    for key in ("expert_cache_budget_bytes", "vram_budget_bytes", "unified_memory_budget_bytes", "gpu_reserve_bytes"):
        if type(settings.get(key)) is not int or settings[key] < 0:
            raise ValueError(f"effective {key} must be a nonnegative integer (zero when inapplicable)")
    evidence = receipt.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("wrapper must retain host-side raw evidence")
    for entry in evidence:
        if not isinstance(entry, dict) or not text(entry.get("path")) or not sha(entry.get("sha256")):
            raise ValueError("invalid wrapper evidence reference")
        artifact = (path.parent / entry["path"]).resolve()
        if not artifact.is_relative_to(path.parent.resolve()) or artifact == path.resolve():
            raise ValueError("wrapper evidence must be a separate file inside its arm directory")
        if stream_chat.digest(artifact.read_bytes()) != entry["sha256"]:
            raise ValueError("wrapper evidence SHA-256 mismatch")
    return receipt


def read_result(path, tier, timeout):
    result = stream_chat.decode_json(path.read_bytes())
    stem = path.with_suffix("")
    protocol = stream_chat.decode_json(stem.with_suffix(".protocol.json").read_bytes())
    if not isinstance(result, dict) or not isinstance(protocol, dict):
        raise ValueError("result and protocol must be objects")
    expected_hash = final_chat.PROMPTS[tier][1]
    if (result.get("status") != "passed" or result.get("output_tokens") != 256
            or result.get("prompt_sha256") != expected_hash
            or protocol.get("protocol") != final_chat.PROTOCOL or protocol.get("tier") != tier
            or protocol.get("prompt_sha256") != expected_hash or protocol.get("output_tokens") != 256
            or protocol.get("timeout_s") != timeout):
        raise ValueError("result does not match the fixed final benchmark protocol")
    ids = protocol.get("prompt_ids")
    context = final_chat.validate_context(ids, protocol.get("server_props", {}))
    if (protocol.get("server_context_tokens") != context or result.get("prompt_ids") != ids
            or result.get("input_tokens") != len(ids) or protocol.get("input_tokens") != len(ids)):
        raise ValueError("template IDs or actual input/context counts differ")
    expected_chat = stream_chat.chat_request(final_chat.prompt_path(tier).read_text(), protocol.get("server_props", {}))
    expected_body = {"prompt": ids, "n_predict": 256, "stream": True,
                     "temperature": 0, "top_k": 1, "samplers": ["top_k"], "seed": 0,
                     "n_probs": 5, "post_sampling_probs": False, "ignore_eos": True,
                     "cache_prompt": False, "return_tokens": True, "timings_per_token": True}
    if protocol.get("chat_request") != expected_chat or protocol.get("request") != expected_body:
        raise ValueError("recorded request differs from the fixed chat/sampling protocol")
    records = [stream_chat.decode_json(line) for line in stem.with_suffix(".events.jsonl").read_bytes().splitlines()]
    if any(type(value) is not int or value < 0 for value in
           [result["start_ns"], result["end_ns"], *(row["monotonic_ns"] for row in records)]):
        raise ValueError("stream timestamps must be nonnegative integer nanoseconds")
    with stem.with_suffix(".events.sse").open("rb") as wire:
        wire_events = list(stream_chat.events(wire))
    if wire_events != [row["event"] for row in records]:
        raise ValueError("raw SSE differs from timestamped records")
    rebuilt = stream_chat.summarize(records, result["start_ns"], result["end_ns"], ids, 256)
    final_chat.validate_timings(rebuilt)
    # JSON object keys convert milestone integers to strings.
    rebuilt = json.loads(json.dumps(rebuilt))
    if any(result.get(k) != v for k, v in rebuilt.items()):
        raise ValueError("result does not match the complete raw token stream")
    return result


def owned_group_exists(process):
    process.poll()
    try:
        return run_abba.owned_group_exists(process)
    except PermissionError:
        # Darwin may return EPERM for a just-exited, unreaped group leader.
        # Reap our child and probe again; a persistent permission error is not
        # interpreted as a vanished group.
        if process.poll() is None:
            raise
        return run_abba.owned_group_exists(process)


def cleanup(process):
    if owned_group_exists(process):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    until = time.monotonic() + 5
    while owned_group_exists(process) and time.monotonic() < until:
        time.sleep(0.05)
    if owned_group_exists(process):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=5)
    until = time.monotonic() + 5
    while owned_group_exists(process) and time.monotonic() < until:
        time.sleep(0.05)
    if owned_group_exists(process):
        raise RuntimeError("owned group still exists after cleanup; keep device quarantined")


def confirm_wrapper_cleanup(path):
    receipt = stream_chat.decode_json(path.read_bytes())
    server = object_field(receipt, "server")
    if server.get("owned_processes_gone") is not True or server.get("resource_scope_gone") is not True:
        raise ValueError("wrapper did not confirm owned server and external scope cleanup")


def fixed_identity(receipt):
    host = receipt["host"]
    return {"build": receipt["build"], "model_identity": receipt["model_identity"],
            "host": {k: host[k] for k in ("hardware", "os", "memory_budget_bytes", "effective_settings",
                                          "os_page_cache_treatment")}}


def run(config, output, lock_path, stop_path, *, order=ORDER, purpose="formal", inherited_lock_fd=None, environment=None):
    if purpose not in ("formal", "ci") or (purpose == "formal" and tuple(order) != ORDER):
        raise ValueError("formal benchmark always runs both fixed tiers once")
    if not order or len(set(order)) != len(order) or any(t not in ORDER for t in order):
        raise ValueError("invalid or duplicate fixed prompt tier")
    # Use the established shared inode, including when it contains owner metadata.
    if inherited_lock_fd is not None:
        actual, expected = os.fstat(inherited_lock_fd), lock_path.stat()
        if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
            raise ValueError("inherited CI lease differs from the device lock")
    with (os.fdopen(os.dup(inherited_lock_fd), "a+") if inherited_lock_fd is not None
          else lock_path.open("a+")) as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if stop_path.exists():
            raise RuntimeError("STOP exists; no arm was started")
        for tier in order:
            final_chat.prompt_path(tier)
        output.mkdir(parents=True, exist_ok=False)
        run_id = str(uuid.uuid4())
        stream_chat.save(output / "manifest.json", {
            "schema_version": 1, "protocol": final_chat.PROTOCOL, "purpose": purpose, "run_id": run_id,
            "order": order, "samples_per_tier": 1, "config": config,
            "owner_pid": os.getpid(), "started_unix_s": time.time(),
            "driver_sha256": {name: stream_chat.digest(Path(__file__).with_name(name).read_bytes())
                              for name in ("run_final.py", "final_chat.py", "stream_chat.py", "run_abba.py")},
        })
        interrupted = []
        handlers = {sig: signal.signal(sig, lambda signum, _frame: interrupted.append(signum))
                    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
        rows, abort_reason, identity = [], None, None
        try:
            with (output / "attempts.jsonl").open("x") as ledger:
                for tier in order:
                    row = {"tier": tier, "device": config["device"], "model": config["model"],
                           "status": "not_run", "samples": 1}
                    if stop_path.exists() or interrupted:
                        abort_reason = "STOP or termination requested"
                    if abort_reason:
                        row["error"] = abort_reason
                    else:
                        arm_dir = output / tier
                        arm_dir.mkdir()
                        result_path, receipt_path = arm_dir / "result.json", arm_dir / "wrapper.json"
                        env = dict(os.environ if environment is None else environment, INFERSSD_RUN_ID=run_id, INFERSSD_RUN_DIR=str(arm_dir),
                                   INFERSSD_RESULT_PATH=str(result_path), INFERSSD_WRAPPER_RECEIPT=str(receipt_path),
                                   INFERSSD_TIER=tier, INFERSSD_PROMPT_PATH=str(final_chat.prompt_path(tier)),
                                   INFERSSD_REQUEST_TIMEOUT=str(config["request_timeout_s"]),
                                   INFERSSD_OUTPUT_TOKENS="256", INFERSSD_DEVICE=config["device"],
                                   INFERSSD_MODEL=config["model"], INFERSSD_REVISION=config["revision"],
                                   INFERSSD_LOCK_FD=str(lock.fileno()))
                        process, start = None, time.monotonic()
                        try:
                            with (arm_dir / "stdout.log").open("xb") as log:
                                process = subprocess.Popen(config["argv"], env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
                                while process.poll() is None:
                                    if interrupted or stop_path.exists():
                                        raise InterruptedError("STOP or termination requested")
                                    if time.monotonic() - start > config["arm_timeout_s"]:
                                        raise TimeoutError("arm deadline expired")
                                    time.sleep(0.05)
                            if interrupted or stop_path.exists():
                                raise InterruptedError("STOP or termination requested")
                            if process.returncode != 0:
                                raise RuntimeError(f"wrapper exited {process.returncode}")
                            if owned_group_exists(process):
                                raise RuntimeError("wrapper exited with owned descendants still running")
                            result = read_result(result_path, tier, config["request_timeout_s"])
                            receipt = validate_receipt(receipt_path, config, tier)
                            if identity is not None and fixed_identity(receipt) != identity:
                                raise ValueError("model, build, settings or resource budgets changed between tiers")
                            identity = fixed_identity(receipt)
                            server_identity = [receipt["server"]["pid"], receipt["server"]["start_identity"]]
                            if rows and rows[0].get("server_identity") == server_identity:
                                raise ValueError("tiers reused the same server process identity")
                            row.update(status="passed", result=result, server_identity=server_identity,
                                       wrapper_receipt_sha256=stream_chat.digest(receipt_path.read_bytes()))
                        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
                            row.update(status="failed", error=f"{type(error).__name__}: {error}")
                            abort_reason = row["error"]
                        finally:
                            if process is not None:
                                try:
                                    cleanup(process)
                                    row["owned_group_cleanup_confirmed"] = True
                                    confirm_wrapper_cleanup(receipt_path)
                                    row["wrapper_cleanup_confirmed"] = True
                                except (OSError, ValueError, TypeError, RuntimeError, subprocess.TimeoutExpired) as error:
                                    row.update(status="failed", cleanup_error=str(error),
                                               wrapper_cleanup_confirmed=False)
                                    abort_reason = "cleanup unconfirmed; device quarantined"
                                    # Do not silently release an uncertain device to another run.
                                    stop_path.touch(exist_ok=True)
                                row["returncode"] = process.returncode
                            row["elapsed_s"] = time.monotonic() - start
                    rows.append(row)
                    ledger.write(json.dumps(row, allow_nan=False) + "\n")
                    ledger.flush()
            complete = all(row["status"] == "passed" for row in rows) and not interrupted and not stop_path.exists()
            stream_chat.save(output / "summary.json", {
                "schema_version": 1, "protocol": final_chat.PROTOCOL, "purpose": purpose, "run_id": run_id,
                "complete": complete, "order": order, "samples_per_tier": 1,
                "interrupted_signals": interrupted, "rows": rows,
            })
            return 0 if complete else 1
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--stop", type=Path, required=True)
    args = parser.parse_args()
    try:
        return run(load_config(args.config), args.output.resolve(), args.lock, args.stop)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
