from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from uuid import uuid4


def request_json(
    method: str,
    url: str,
    payload: dict[str, object] | None = None,
    api_key: str | None = None,
    timeout: float = 120.0,
) -> tuple[int, dict[str, object]]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-Api-Key"] = api_key
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
            return response.status, body
    except urllib.error.HTTPError as error:
        body = json.loads(error.read().decode("utf-8"))
        return error.code, body


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local AML-compatible Add/Search smoke test")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")
    run_id = f"local-{int(time.time())}-{uuid4().hex[:8]}"
    user_a = f"eval:{run_id}:user-a"
    user_b = f"eval:{run_id}:user-b"

    status, health = request_json("GET", f"{base_url}/health", timeout=args.timeout)
    check(status == 200 and health.get("status") == "ok", "health endpoint failed")

    add_a = {
        "request_id": f"{run_id}:chunk-0",
        "messages": [
            {
                "role": "user",
                "timestamp": 1704067200000,
                "content": "Ada Lovelace wrote notes about Charles Babbage's Analytical Engine.",
            },
            {
                "role": "assistant",
                "timestamp": 1704067201000,
                "content": "I will remember that historical fact.",
            },
        ],
        "user_id": user_a,
        "session_id": f"{run_id}:session-a",
    }
    status, response = request_json("POST", f"{base_url}/add", add_a, args.api_key, args.timeout)
    check(status == 200 and response.get("success") is True, "add failed")
    for field in ("request_id", "user_id", "session_id"):
        check(response.get(field) == add_a[field], f"add response did not echo {field}")

    status, repeated = request_json("POST", f"{base_url}/add", add_a, args.api_key, args.timeout)
    check(status == 200 and repeated == response, "idempotent add failed")

    conflicting = {**add_a, "messages": [{"role": "user", "content": "different payload"}]}
    status, _ = request_json("POST", f"{base_url}/add", conflicting, args.api_key, args.timeout)
    check(status == 409, "conflicting request_id must return HTTP 409")

    search_a = {
        "query": "Who wrote notes about the Analytical Engine?",
        "options": ["A. Ada Lovelace", "B. Grace Hopper"],
        "user_id": user_a,
        "top_k": 100,
    }
    status, result = request_json("POST", f"{base_url}/search", search_a, args.api_key, args.timeout)
    check(status == 200 and isinstance(result.get("data"), list), "search failed")
    data = result["data"]
    check(0 < len(data) <= 100, "search returned an invalid result count")
    check(all(item.get("id") and item.get("content") for item in data), "search items are incomplete")
    check(any("Ada Lovelace" in item["content"] for item in data), "expected evidence was not retrieved")

    add_b = {
        "request_id": f"{run_id}:private-chunk",
        "messages": [{"role": "user", "content": "PRIVATE_MARKER belongs only to user B."}],
        "user_id": user_b,
        "session_id": f"{run_id}:session-b",
    }
    status, _ = request_json("POST", f"{base_url}/add", add_b, args.api_key, args.timeout)
    check(status == 200, "second user add failed")
    isolation_query = {"query": "PRIVATE_MARKER", "user_id": user_a, "top_k": 100}
    status, isolated = request_json("POST", f"{base_url}/search", isolation_query, args.api_key, args.timeout)
    check(status == 200, "isolation search failed")
    check(all("PRIVATE_MARKER" not in item["content"] for item in isolated["data"]), "cross-user memory leak detected")

    empty_query = {"query": "anything", "user_id": f"{run_id}:empty", "top_k": 100}
    status, empty = request_json("POST", f"{base_url}/search", empty_query, args.api_key, args.timeout)
    check(status == 200 and empty == {"data": []}, "empty user search must return an empty data array")

    print(json.dumps({"status": "passed", "run_id": run_id, "checks": 8}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False, indent=2), file=sys.stderr)
        raise

