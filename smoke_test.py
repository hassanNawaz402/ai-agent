#!/usr/bin/env python3
"""End-to-end smoke test: hits every public route and prints a pass/fail summary.

Assumes `python3 run_server.py` is running in another terminal. Default URL is
http://127.0.0.1:8000; override with --url.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def _post(url: str, payload: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")
            return resp.getcode(), json.loads(body)
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode("utf-8", errors="replace")}


def _get(url: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            body = resp.read().decode("utf-8")
            return resp.getcode(), json.loads(body)
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode("utf-8", errors="replace")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    base = args.url.rstrip("/")

    results: list[tuple[str, bool, str]] = []

    code, body = _get(f"{base}/health")
    results.append(("GET /health", code == 200 and body.get("status") in ("ok", "degraded"), str(body)))

    code, body = _post(f"{base}/documents/reindex", {})
    results.append(("POST /documents/reindex", code == 200 and "scanned" in body, str(body)))

    code, body = _post(f"{base}/chat", {
        "text": "What is your booking process?", "session_id": "smoke-1",
    })
    results.append(("POST /chat (in-topic)", code == 200 and "reply" in body, str(body)[:200]))

    code, body = _post(f"{base}/chat", {
        "text": "What is the weather on Mars today?", "session_id": "smoke-2",
    })
    results.append(("POST /chat (off-topic refusal)", code == 200 and body.get("refused") is True, str(body)[:200]))

    code, body = _post(f"{base}/email", {
        "text": (
            "Hello, I would like to book a removal from Manchester to London next "
            "Thursday for a 3-bedroom house. Could you confirm availability and "
            "payment options? Also, where is your company based? Thanks."
        ),
    })
    results.append(("POST /email (short)", code == 200 and ("reply" in body or body.get("refused") is True), str(body)[:200]))

    long_body = "Please help with the following questions. " + ("booking details and payment information. " * 2000)
    code, body = _post(f"{base}/email", {"text": long_body})
    results.append(("POST /email (long, map-reduce)", code == 200 and ("reply" in body or body.get("refused") is True), str(body)[:200]))

    print("\n=== smoke test results ===")
    passed = 0
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}")
        if not ok:
            print(f"       {detail}")
        else:
            passed += 1
    print(f"\n{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
