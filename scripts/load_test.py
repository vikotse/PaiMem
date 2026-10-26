from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from uuid import uuid4

try:
    from .local_smoke import request_json
except ImportError:
    from local_smoke import request_json


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(len(ordered) * fraction))
    return ordered[min(rank - 1, len(ordered) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a bounded concurrent load test against Add/Search")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--users", type=int, default=4)
    parser.add_argument("--searches-per-user", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")
    run_id = f"load-{uuid4().hex[:12]}"
    users: list[tuple[str, str]] = []
    add_latencies: list[float] = []

    for index in range(args.users):
        user_id = f"local-load:{run_id}:user-{index}"
        marker = f"MARKER_{run_id}_{index}"
        payload = {
            "request_id": f"{run_id}:add-{index}",
            "messages": [{"role": "user", "content": f"{marker} is the private marker for user {index}."}],
            "user_id": user_id,
            "session_id": f"{run_id}:session-{index}",
        }
        started = time.perf_counter()
        status, body = request_json("POST", f"{base_url}/add", payload, args.api_key, args.timeout)
        add_latencies.append(time.perf_counter() - started)
        if status != 200 or body.get("success") is not True:
            raise RuntimeError(f"Add failed for {user_id}: HTTP {status} {body}")
        users.append((user_id, marker))

    def search(user_id: str, marker: str) -> tuple[float, bool, str | None]:
        started = time.perf_counter()
        try:
            status, body = request_json(
                "POST",
                f"{base_url}/search",
                {"query": marker, "user_id": user_id, "top_k": 100},
                args.api_key,
                args.timeout,
            )
            valid = status == 200 and any(marker in item["content"] for item in body.get("data", []))
            return time.perf_counter() - started, valid, None if valid else f"HTTP {status} or marker missing"
        except Exception as error:
            return time.perf_counter() - started, False, str(error)

    search_latencies: list[float] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(search, user_id, marker)
            for user_id, marker in users
            for _ in range(args.searches_per_user)
        ]
        for future in as_completed(futures):
            latency, valid, error = future.result()
            search_latencies.append(latency)
            if not valid:
                failures.append(error or "unknown failure")

    report = {
        "status": "passed" if not failures else "failed",
        "run_id": run_id,
        "users": args.users,
        "searches": len(search_latencies),
        "concurrency": args.concurrency,
        "failures": failures,
        "add_latency_seconds": {
            "mean": statistics.fmean(add_latencies),
            "p95": percentile(add_latencies, 0.95),
        },
        "search_latency_seconds": {
            "mean": statistics.fmean(search_latencies) if search_latencies else None,
            "p95": percentile(search_latencies, 0.95),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
