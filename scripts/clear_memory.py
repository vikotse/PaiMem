"""Clear the running memory service before a fresh evaluation run."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.error
import urllib.request
from urllib.parse import urlsplit


class ClearError(Exception):
    """The service did not confirm that its memory is empty."""


def request_json(method: str, url: str, api_key: str, timeout: float) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=b"" if method == "POST" else None,
        headers={"X-Api-Key": api_key},
        method=method,
    )
    endpoint = urlsplit(url).path
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise ClearError(f"{method} {endpoint} returned HTTP {response.status}")
            body = response.read()
    except urllib.error.HTTPError as error:
        raise ClearError(f"{method} {endpoint} returned HTTP {error.code}") from None
    except (urllib.error.URLError, TimeoutError) as error:
        raise ClearError(f"{method} {endpoint} failed: {error.reason if isinstance(error, urllib.error.URLError) else 'timed out'}") from None

    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ClearError(f"{method} {endpoint} returned invalid JSON") from None
    if not isinstance(payload, dict):
        raise ClearError(f"{method} {endpoint} returned a non-object response")
    return payload


def require_zero_counts(payload: dict[str, object], endpoint: str) -> None:
    for field in ("requests", "memories", "users"):
        if type(payload.get(field)) is not int or payload[field] != 0:
            raise ClearError(f"{endpoint} did not confirm {field}=0")


def main() -> int:
    parser = argparse.ArgumentParser(description="Clear the running memory service before a fresh evaluation")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default=os.environ.get("MEMORY_API_KEY"))
    parser.add_argument("--timeout", type=float, default=120.0, help="seconds allowed per request")
    args = parser.parse_args()

    if not args.api_key:
        parser.error("set MEMORY_API_KEY or pass --api-key")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be a finite positive number")
    base_url = args.base_url.rstrip("/")
    parsed = urlsplit(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
        parser.error("--base-url must be an HTTP(S) service URL without credentials, query, or fragment")

    try:
        cleared = request_json("POST", f"{base_url}/clear", args.api_key, args.timeout)
        if cleared.get("success") is not True:
            raise ClearError("POST /clear did not confirm success=true")
        for field in ("deleted_requests", "deleted_memories"):
            if type(cleared.get(field)) is not int or cleared[field] < 0:
                raise ClearError(f"POST /clear returned invalid {field}")
        require_zero_counts(cleared, "POST /clear")

        health = request_json("GET", f"{base_url}/health", args.api_key, args.timeout)
        if health.get("status") != "ok":
            raise ClearError("GET /health did not confirm status=ok")
        require_zero_counts(health, "GET /health")
    except ClearError as error:
        print(f"clear_memory: {error}", file=sys.stderr)
        return 1

    print(
        "Memory cleared: "
        f"{cleared['deleted_requests']} requests and {cleared['deleted_memories']} memories removed; "
        "/health confirms 0 requests, 0 memories, 0 users."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
