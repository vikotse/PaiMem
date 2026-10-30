from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import hashlib
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit

import httpx
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


def legacy_main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Legacy: serial Add then concurrent Search. Use --mode mixed for simultaneous load.",
        epilog="Mixed help: python scripts/load_test.py --mode mixed --help")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--users", type=int, default=4)
    parser.add_argument("--searches-per-user", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)
    args.api_key = args.api_key or os.environ.get("MEMORY_API_KEY")
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
        try:
            status, body = request_json("POST", f"{base_url}/add", payload, args.api_key, args.timeout)
        except Exception as error:
            raise RuntimeError(f"Add failed ({type(error).__name__})") from None
        add_latencies.append(time.perf_counter() - started)
        if status != 200 or body.get("success") is not True:
            raise RuntimeError(f"Add failed: HTTP {status}")
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
            return time.perf_counter() - started, False, type(error).__name__

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


# Mixed mode is independent of the legacy sequential-add / concurrent-search smoke test.
# Example (run on the client host; URL is PaiMem, NOT the vLLM server):
# MEMORY_API_KEY=... python scripts/load_test.py --mode mixed --base-url http://HOST:8000 \
#   --add-concurrency 16 --search-concurrency 16 --duration 120 --input data/load-input.jsonl
# JSONL: {"query": "...", "documents": ["...", ...]}; >= seed-documents per row.
# Each user receives one row's documents before measurement. Synthetic data is the
# default when --input is absent. Distinct test users remain in the database;
# no clear/delete/config endpoints are called. Mixed mode grows the corpus, so
# this is not a fixed-corpus reranker benchmark. A timeout may still commit server
# writes; never count it as a confirmed success and never retry it automatically.
# Reports contain identifiers/counters, not input text, response bodies or keys.


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def positive_seconds(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('must be finite and positive')
    return number


def mixed_parser():
    parser = argparse.ArgumentParser(description='Seed isolated users, then run concurrent Add/Search. Data is retained.')
    parser.add_argument('--base-url', default='http://127.0.0.1:8000', help='PaiMem service URL, not vLLM')
    parser.add_argument('--api-key-env', default='MEMORY_API_KEY')
    parser.add_argument('--users', type=positive_int, default=16)
    parser.add_argument('--seed-documents', type=positive_int, default=400, help='source documents per user; actual stored chunk count can differ')
    parser.add_argument('--seed-batch-size', type=positive_int, default=20)
    parser.add_argument('--seed-concurrency', type=positive_int, default=4)
    parser.add_argument('--messages-per-add', type=positive_int, default=10)
    parser.add_argument('--add-concurrency', type=positive_int, default=16)
    parser.add_argument('--search-concurrency', type=positive_int, default=16)
    parser.add_argument('--top-k', type=positive_int, default=20)
    parser.add_argument('--duration', type=positive_seconds, default=120)
    parser.add_argument('--timeout', type=positive_seconds, default=180)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path, help='optional JSON report, never overwrite')
    parser.add_argument('--dry-run', action='store_true', help='validate only, no requests or files')
    return parser


def load_workload(args):
    if not args.input:
        return [dict(query='项目进度、交付计划和验收要求是什么？', documents=[
            f'项目记录 {i}：团队讨论了第 {i + 1} 阶段的交付计划。负责人将在周五提交报告，'
            '客户要求先完成接口验收，再安排下一阶段部署。研发、测试和业务团队需要同步进展。'
            for i in range(args.seed_documents)])]
    records = []
    for line_number, line in enumerate(args.input.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            raise ValueError(f'invalid JSON at line {line_number}') from None
        if (not isinstance(row, dict) or not isinstance(row.get('query'), str) or not row['query'].strip()
                or not isinstance(row.get('documents'), list) or len(row['documents']) < args.seed_documents
                or any(not isinstance(d, str) or not d.strip() for d in row['documents'])):
            raise ValueError(f'line {line_number} requires query and >= {args.seed_documents} nonempty documents')
        records.append(dict(query=row['query'], documents=row['documents'][:args.seed_documents]))
    if not records:
        raise ValueError('input has no records')
    return records


async def call_api(client, path, payload, timeout):
    started = time.perf_counter()
    count = None
    error = None
    try:
        async with asyncio.timeout(timeout):
            response = await client.post(path, json=payload)
            if response.status_code != 200:
                error = f'HTTP_{response.status_code}'
            else:
                body = response.json()
                if not isinstance(body, dict):
                    raise ValueError('invalid response')
                if path == '/add':
                    if body.get('success') is not True or any(body.get(k) != payload[k] for k in ('request_id', 'user_id', 'session_id')):
                        raise ValueError('invalid add response')
                else:
                    data = body.get('data')
                    if not isinstance(data, list) or len(data) > payload['top_k']:
                        raise ValueError('invalid search response')
                    for item in data:
                        if (not isinstance(item, dict) or not isinstance(item.get('id'), str)
                                or not isinstance(item.get('content'), str)
                                or type(item.get('score')) not in (int, float)
                                or not math.isfinite(item['score'])):
                            raise ValueError('invalid search item')
                    count = len(data)
    except (httpx.HTTPError, TimeoutError, ValueError, OverflowError) as exc:
        error = type(exc).__name__
    return dict(seconds=time.perf_counter() - started, error=error, result_count=count)


def add_payload(user, request_id, texts):
    return dict(request_id=request_id, user_id=user, session_id=user + ':session',
                messages=[dict(role='user', content=text) for text in texts])


def summarize(samples, wall):
    good = [sample['seconds'] for sample in samples if sample['error'] is None]
    failed = [sample['seconds'] for sample in samples if sample['error'] is not None]
    return dict(attempted=len(samples), successful=len(good), failed=len(failed),
                successful_qps=len(good) / wall if wall else 0,
                error_rate=len(failed) / len(samples) if samples else None,
                mean_ms=statistics.fmean(good) * 1000 if good else None,
                p50_ms=percentile(good, .5) * 1000 if good else None,
                p95_ms=percentile(good, .95) * 1000 if good else None,
                p99_ms=percentile(good, .99) * 1000 if good else None,
                failed_mean_ms=statistics.fmean(failed) * 1000 if failed else None,
                empty_searches=sum(s['error'] is None and s['result_count'] == 0 for s in samples),
                errors=dict(Counter(s['error'] for s in samples if s['error'])))


async def run_mixed(args, records, transport=None):
    run_id = 'load-' + uuid4().hex
    users = [f'local-load:{run_id}:user-{i}' for i in range(args.users)]
    print(f'Run ID: {run_id}; user prefix: local-load:{run_id}:user-', file=sys.stderr, flush=True)
    report = dict(schema_version=1, run_id=run_id, mode=args.mode, status='seeding',
                  user_ids=users, input_kind='jsonl' if args.input else 'synthetic',
                  input_sha256=hashlib.sha256(json.dumps(records, ensure_ascii=False).encode()).hexdigest(),
                  settings={key: getattr(args, key) for key in (
                      'users', 'seed_documents', 'seed_batch_size', 'seed_concurrency', 'messages_per_add',
                      'add_concurrency', 'search_concurrency', 'top_k', 'duration', 'timeout')},
                  measurement='Closed loop; success-only latency; wall time includes drain; no retries. Seeding and validation searches excluded.',
                  seed=dict(successful_adds=0, confirmed_source_documents=0, ready_users=0),
                  add={}, search={})
    key = os.environ.get(args.api_key_env, '')
    headers = {'X-Api-Key': key} if key else {}
    capacity = max(args.seed_concurrency, args.add_concurrency + args.search_concurrency)
    async with httpx.AsyncClient(base_url=args.base_url.rstrip('/'), headers=headers,
                                timeout=args.timeout, trust_env=False, transport=transport,
                                limits=httpx.Limits(max_connections=capacity, max_keepalive_connections=capacity)) as client:
        # Fixed seed workers: do not enqueue an unbounded number of HTTP tasks.
        pending_users = iter(range(args.users))
        seed_errors = []
        async def seed_worker():
            for index in pending_users:
                if seed_errors:
                    return
                row = records[index % len(records)]
                for start in range(0, args.seed_documents, args.seed_batch_size):
                    if seed_errors:
                        return
                    texts = row['documents'][start:start + args.seed_batch_size]
                    result = await call_api(client, '/add', add_payload(users[index], f'{run_id}:seed:{index}:{start}', texts), args.timeout)
                    if result['error']:
                        seed_errors.append(result['error'])
                        return
                    report['seed']['successful_adds'] += 1
                    report['seed']['confirmed_source_documents'] += len(texts)
                # Verify users are searchable; empty users would silently bypass reranker.
                result = await call_api(client, '/search', dict(user_id=users[index], query=row['query'], top_k=args.top_k), args.timeout)
                if result['error'] or result['result_count'] == 0:
                    seed_errors.append(result['error'] or 'EmptySeedSearch')
                    return
                report['seed']['ready_users'] += 1
                print(f"Seeded {report['seed']['ready_users']}/{args.users} users", file=sys.stderr, flush=True)
        await asyncio.gather(*(seed_worker() for _ in range(min(args.seed_concurrency, args.users))))
        if seed_errors:
            report.update(status='seed_failed', seed_errors=dict(Counter(seed_errors)))
            return report

        samples = {'add': [], 'search': []}
        cancelled = {'add': 0, 'search': 0}
        started = time.perf_counter()
        deadline = started + args.duration
        report['status'] = 'running'
        print(f'Measuring {args.mode} for {args.duration}s; add={args.add_concurrency if args.mode != "search" else 0}, search={args.search_concurrency if args.mode != "add" else 0}', file=sys.stderr, flush=True)
        async def worker(kind, number):
            sequence = 0
            while time.perf_counter() < deadline:
                index = (number + sequence) % args.users
                row = records[index % len(records)]
                if kind == 'add':
                    texts = [row['documents'][(sequence * args.messages_per_add + j) % len(row['documents'])]
                             for j in range(args.messages_per_add)]
                    payload = add_payload(users[index], f'{run_id}:timed:{number}:{sequence}', texts)
                else:
                    payload = dict(user_id=users[index], query=row['query'], top_k=args.top_k)
                try:
                    samples[kind].append(await call_api(client, '/' + kind, payload, args.timeout))
                except asyncio.CancelledError:
                    cancelled[kind] += 1
                    raise
                sequence += 1
                if samples[kind][-1]['error']:
                    await asyncio.sleep(.1)  # Bound hammering of a failed service.
        tasks = []
        if args.mode in ('mixed', 'add'):
            tasks.extend(asyncio.create_task(worker('add', i)) for i in range(args.add_concurrency))
        if args.mode in ('mixed', 'search'):
            tasks.extend(asyncio.create_task(worker('search', i)) for i in range(args.search_concurrency))
        async def progress():
            while True:
                await asyncio.sleep(10)
                print(f"elapsed={time.perf_counter() - started:.0f}s completed_add={len(samples['add'])} completed_search={len(samples['search'])}", file=sys.stderr, flush=True)
        progress_task = asyncio.create_task(progress())
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            report['status'] = 'interrupted'
        finally:
            progress_task.cancel()
            await asyncio.gather(progress_task, return_exceptions=True)
            wall = time.perf_counter() - started
            report['wall_seconds'] = wall
            report['add'] = summarize(samples['add'], wall)
            report['search'] = summarize(samples['search'], wall)
            report['cancelled_inflight'] = cancelled
            if report['status'] != 'interrupted':
                report['status'] = 'failed' if any(s['error'] for group in samples.values() for s in group) else 'passed'
    return report


def main(argv=None):
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument('--mode', choices=['sequential', 'mixed', 'add', 'search'], default='sequential')
    selected, rest = selector.parse_known_args(argv)
    if selected.mode == 'sequential':
        return legacy_main(rest)
    parser = mixed_parser()
    args = parser.parse_args(rest)
    args.mode = selected.mode
    try:
        url = urlsplit(args.base_url)
        url.port
        if (url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password
                or url.query or url.fragment or url.path not in ('', '/')):
            raise ValueError('--base-url must be a server root without credentials or a path')
        if args.top_k > 100:
            raise ValueError('--top-k must be <= 100')
        records = load_workload(args)
        if args.output and args.output.exists():
            raise ValueError('--output already exists; choose a new file')
        if args.output and not args.output.parent.is_dir():
            raise ValueError('--output parent directory must already exist')
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f'Seed {args.users * args.seed_documents} source documents into {args.users} isolated users; data will remain in the service.', file=sys.stderr)
    if args.dry_run:
        print(json.dumps(dict(mode=args.mode, users=args.users, input_kind='jsonl' if args.input else 'synthetic',
                              add_concurrency=args.add_concurrency if args.mode != 'search' else 0,
                              search_concurrency=args.search_concurrency if args.mode != 'add' else 0,
                              duration=args.duration), indent=2))
        return
    try:
        report = asyncio.run(run_mixed(args, records))
    except KeyboardInterrupt:
        print('Interrupted during seeding; partial test data may remain. No cleanup was performed.', file=sys.stderr)
        raise SystemExit(130)
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(output + '\n')
    if report['status'] != 'passed':
        raise SystemExit(130 if report['status'] == 'interrupted' else 1)


if __name__ == '__main__':
    main()
