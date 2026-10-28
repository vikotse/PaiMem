"""Closed-loop /score capacity benchmark; see docs/reranker-benchmark.zh-CN.md."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'paimem'))
from configuration import prepare  # standalone, no torch/model imports


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def seconds(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('must be finite and positive')
    return value


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def validate_scores(payload, count):
    data = payload.get('data') if isinstance(payload, dict) else None
    if not isinstance(data, list) or len(data) != count:
        raise ValueError('invalid score count')
    seen = set()
    for item in data:
        if not isinstance(item, dict):
            raise ValueError('invalid score item')
        index, score = item.get('index'), item.get('score')
        if (type(index) is not int or not 0 <= index < count or index in seen
                or type(score) not in (int, float) or not math.isfinite(score)
                or abs(score) > 3.4028234663852886e38):
            raise ValueError('invalid score index/value')
        seen.add(index)


def load_records(args):
    if args.input:
        records = []
        for line, raw in enumerate(args.input.read_text(encoding='utf-8').splitlines(), 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except ValueError:
                raise ValueError(f'invalid input JSON at line {line}') from None
            if (not isinstance(row, dict) or not isinstance(row.get('query'), str)
                    or not row['query'].strip() or not isinstance(row.get('documents'), list)
                    or len(row['documents']) < args.candidates
                    or any(not isinstance(d, str) or not d.strip() for d in row['documents'])):
                raise ValueError(f'line {line}: query and at least {args.candidates} nonempty documents required')
            records.append(dict(query=row['query'], documents=row['documents'][:args.candidates]))
        if not records:
            raise ValueError('input has no records')
        return records
    # Explicit synthetic mode: smoke/load testing only, not real token-length matching.
    return [dict(query=f'Which memory concerns project {q}?', documents=[
        f'Record {q}-{d}: ' + ' '.join(('project discussion schedule budget customer meeting '.split() * args.synthetic_words)[:args.synthetic_words])
        for d in range(args.candidates)]) for q in range(16)]


async def measure(client, args, records, batch_size, concurrency, batch_concurrency, duration):
    """Each worker owns one logical query; drain all admitted work before returning."""
    query_latencies, http_latencies = [], []
    failures = Counter()
    successes = attempted = valid_pairs = http_attempted = http_ok = 0
    started = time.perf_counter()
    deadline = started + duration
    cursor = 0

    async def query(row):
        nonlocal valid_pairs, http_attempted, http_ok
        batches = iter(range(0, args.candidates, batch_size))
        errors = []

        async def batch_worker():
            nonlocal valid_pairs, http_attempted, http_ok
            for start in batches:
                docs = row['documents'][start:start + batch_size]
                t0 = time.perf_counter()
                http_attempted += 1
                error = None
                try:
                    # Total request deadline, including connection pool wait and response body.
                    async with asyncio.timeout(args.timeout):
                        response = await client.post('/score', json={
                            'model': args.model, 'queries': [row['query']] * len(docs),
                            'documents': docs, 'truncate_prompt_tokens': args.max_length,
                        })
                        if response.status_code != 200:
                            error = f'HTTP_{response.status_code}'
                        else:
                            validate_scores(response.json(), len(docs))
                except (httpx.HTTPError, TimeoutError, ValueError) as exc:
                    error = type(exc).__name__  # Never save response bodies, keys or input texts.
                if error:
                    errors.append(error)
                else:
                    valid_pairs += len(docs)
                    http_ok += 1
                    http_latencies.append((time.perf_counter() - t0) * 1000)

        await asyncio.gather(*(batch_worker() for _ in range(
            min(batch_concurrency, math.ceil(args.candidates / batch_size)))))
        return errors

    async def worker():
        nonlocal cursor, attempted, successes
        while time.perf_counter() < deadline:
            row = records[cursor % len(records)]
            cursor += 1
            attempted += 1
            t0 = time.perf_counter()
            errors = await query(row)
            if errors:
                failures.update(errors)
                # Avoid hammering a failed/unavailable endpoint at unbounded request rates.
                await asyncio.sleep(0.1)
            else:
                successes += 1
                query_latencies.append((time.perf_counter() - t0) * 1000)

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    elapsed = time.perf_counter() - started
    return dict(
        batch_size=batch_size, concurrency=concurrency, batch_concurrency=batch_concurrency,
        requires_client_change=batch_concurrency > 1,
        wall_seconds=elapsed, admission_seconds=duration,
        attempted_queries=attempted, successful_queries=successes,
        failed_queries=attempted - successes,
        error_rate=(attempted - successes) / attempted if attempted else 1.0,
        queries_per_second=successes / elapsed,
        # Only fully successful queries count toward useful throughput.
        useful_pairs_per_second=successes * args.candidates / elapsed,
        valid_pairs_per_second=valid_pairs / elapsed,
        http_attempted=http_attempted, http_successful=http_ok,
        query_p50_ms=percentile(query_latencies, .5),
        query_p95_ms=percentile(query_latencies, .95),
        query_p99_ms=percentile(query_latencies, .99),
        http_p95_ms=percentile(http_latencies, .95), errors=dict(failures),
    )


async def check_model(client, model):
    response = await client.get('/v1/models')
    response.raise_for_status()
    if model not in [item.get('id') for item in response.json().get('data', [])]:
        raise ValueError('requested model is not advertised by /v1/models')


def port_busy(port):
    with socket.socket() as sock:
        return sock.connect_ex(('127.0.0.1', port)) == 0


def stop_owned(process):
    # Only signal the new process group started by this benchmark, never discover/kill by port.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def launch_command(args, profile):
    return ['bash', str(ROOT / 'scripts/start_vllm.sh'), '--config', str(args.launch_config), '--',
            '--data-parallel-size', str(profile['dp']),
            '--max-num-batched-tokens', str(profile['tokens']),
            '--max-num-seqs', str(profile['seqs'])]


async def wait_ready(client, args, process):
    deadline = time.monotonic() + args.startup_timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError('test server exited; inspect its startup log')
        try:
            await check_model(client, args.model)
            return
        except (httpx.HTTPError, ValueError):
            await asyncio.sleep(2)
    raise TimeoutError('test server startup timed out')


def recommendations(rows, args):
    # Aggregate repeat runs; require EVERY repeat to meet reliability/latency/sample constraints.
    groups = {}
    for row in rows:
        key = (row['server'], row['batch_size'], row['concurrency'], row['batch_concurrency'])
        groups.setdefault(key, []).append(row)
    choices = []
    for key, trials in groups.items():
        if len(trials) != args.repeats or any(
            r['error_rate'] > args.max_error_rate or r['successful_queries'] < args.min_success
            or (args.p95_ms is not None and (r['query_p95_ms'] is None or r['query_p95_ms'] > args.p95_ms))
            for r in trials
        ):
            continue
        total_time = sum(r['wall_seconds'] for r in trials)
        choices.append(dict(server=key[0], batch_size=key[1], concurrency=key[2], batch_concurrency=key[3],
                            useful_pairs_per_second=sum(r['successful_queries'] for r in trials) * args.candidates / total_time,
                            worst_repeat_p95_ms=max(r['query_p95_ms'] for r in trials),
                            requires_client_change=key[3] > 1))
    choices.sort(key=lambda r: r['useful_pairs_per_second'], reverse=True)
    current = [r for r in choices if not r['requires_client_change']]
    return dict(best=choices[0] if choices else None,
                best_current_client=current[0] if current else None, ranked=choices,
                note='Best measured point only; no qualifying result means increase samples or inspect errors/SLO.')


def save_report(report, args):
    report['recommendations'] = recommendations(report['results'], args)
    target = Path(str(args.output) + '.json')
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(target)
    rows = report['results']
    if rows:
        target = Path(str(args.output) + '.csv')
        temporary = target.with_suffix('.csv.tmp')
        with temporary.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows({**r, 'errors': json.dumps(r['errors'])} for r in rows)
        temporary.replace(target)


async def run(args, records, profiles, launch_env):
    report = dict(schema_version=1, model=args.model, max_length=args.max_length,
                  candidates=args.candidates, input_kind='real' if args.input else 'synthetic',
                  input_sha256=hashlib.sha256(json.dumps(records, ensure_ascii=False).encode()).hexdigest(),
                  input_records=len(records), seed=args.seed, repeats=args.repeats,
                  benchmark_config=dict(duration=args.duration, warmup=args.warmup, cooldown=args.cooldown,
                                        timeout=args.timeout, batch_sizes=args.batch_sizes,
                                        concurrency=args.concurrency, batch_concurrency=args.batch_concurrency),
                  slo=dict(p95_ms=args.p95_ms, max_error_rate=args.max_error_rate, min_success=args.min_success),
                  measurement='Closed loop; wall time includes drain; latencies are successful queries only; no retries.',
                  profiles=profiles, results=[], server_errors=[],
                  server_config=({key: launch_env.get(key) for key in (
                      'VLLM_VERSION', 'VLLM_DTYPE', 'VLLM_MAX_MODEL_LEN',
                      'VLLM_GPU_MEMORY_UTILIZATION', 'RERANKER_REVISION')} if launch_env else None),
                  configured_gpus=args.gpus if launch_env else None)
    save_report(report, args)
    headers = {'Authorization': f'Bearer {args.api_key}'} if args.api_key else {}
    max_connections = max(args.concurrency) * min(max(args.batch_concurrency), math.ceil(args.candidates / min(args.batch_sizes)))
    for profile in profiles:
        process = log = None
        try:
            if args.launch_config:
                if port_busy(args.port):
                    raise RuntimeError('benchmark port is occupied; choose a dedicated free port')
                log_path = Path(str(args.output) + f".{profile['name']}.server.log")
                log = log_path.open('x', encoding='utf-8')
                env = dict(launch_env, CUDA_VISIBLE_DEVICES=args.gpus, VLLM_HOST='127.0.0.1',
                           VLLM_PORT=str(args.port), VLLM_TENSOR_PARALLEL_SIZE=str(profile['tp']))
                process = subprocess.Popen(launch_command(args, profile), cwd=ROOT, env=env,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                print(f"Starting {profile['name']}; log: {log_path}", flush=True)
            async with httpx.AsyncClient(base_url=args.base_url, headers=headers,
                    timeout=min(args.timeout, 10), trust_env=False,
                    limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections)) as client:
                if process:
                    await wait_ready(client, args, process)
                else:
                    await check_model(client, args.model)
                # Measurement timeout is the requested timeout, not the short startup probe timeout.
                client.timeout = httpx.Timeout(args.timeout)
                matrix = list(itertools.product(args.batch_sizes, args.concurrency, args.batch_concurrency))
                for repeat in range(args.repeats):
                    random.Random(args.seed + repeat).shuffle(matrix)
                    for batch, concurrency, batch_concurrency in matrix:
                        print(f"{profile['name']} repeat={repeat + 1} batch={batch} queries={concurrency} batch_parallel={batch_concurrency}: warmup", flush=True)
                        warmup = await measure(client, args, records, batch, concurrency, batch_concurrency, args.warmup)
                        if warmup['failed_queries']:
                            print('  warmup had failures; timed run will record errors', flush=True)
                        row = await measure(client, args, records, batch, concurrency, batch_concurrency, args.duration)
                        row = dict(server=profile['name'], repeat=repeat + 1, **row)
                        report['results'].append(row)
                        save_report(report, args)
                        p95 = row['query_p95_ms']
                        print(f"  {row['useful_pairs_per_second']:.1f} pairs/s; p95={p95 if p95 is not None else 'N/A'} ms; errors={row['error_rate']:.1%}", flush=True)
                        await asyncio.sleep(args.cooldown)
        except (OSError, RuntimeError, TimeoutError, httpx.HTTPError, ValueError) as exc:
            # Exception bodies/URLs may contain private server data; save only a category.
            report['server_errors'].append(dict(server=profile['name'], error=type(exc).__name__))
            print(f"Server profile failed: {profile['name']} ({type(exc).__name__}); inspect startup log/connectivity/model", flush=True)
        finally:
            if process:
                stop_owned(process)
            if log:
                log.close()
            save_report(report, args)
    print(json.dumps(report['recommendations'], ensure_ascii=False, indent=2))
    return 0 if report['recommendations']['best'] and not report['server_errors'] else 1


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', type=Path, help='JSONL: query and documents; real candidate text recommended')
    source.add_argument('--synthetic', action='store_true', help='explicit synthetic smoke/load test')
    p.add_argument('--synthetic-words', type=positive, default=128)
    p.add_argument('--base-url', default='http://127.0.0.1:8089')
    p.add_argument('--model', default=None)
    p.add_argument('--api-key-env', default='MEMORY_RERANKER_API_KEY')
    p.add_argument('--server-label', default='existing-server')
    p.add_argument('--launch-config', type=Path, help='Linux only: start/stop owned test servers using this TOML')
    p.add_argument('--gpus', default='0')
    p.add_argument('--port', type=positive, default=18089)
    p.add_argument('--parallel', nargs='+', default=['1:1'], help='TP:DP pairs, e.g. 1:4 2:2 4:1')
    p.add_argument('--server-tokens', nargs='+', type=positive, default=[8192, 16384])
    p.add_argument('--server-seqs', nargs='+', type=positive, default=[128])
    p.add_argument('--batch-sizes', nargs='+', type=positive, default=[64, 128])
    p.add_argument('--concurrency', nargs='+', type=positive, default=[4, 8, 16])
    p.add_argument('--batch-concurrency', nargs='+', type=positive, default=[1])
    p.add_argument('--candidates', type=positive, default=400)
    p.add_argument('--max-length', type=positive, default=1024)
    p.add_argument('--duration', type=seconds, default=30)
    p.add_argument('--warmup', type=seconds, default=5)
    p.add_argument('--cooldown', type=seconds, default=1)
    p.add_argument('--repeats', type=positive, default=2)
    p.add_argument('--timeout', type=seconds, default=120)
    p.add_argument('--startup-timeout', type=seconds, default=900)
    p.add_argument('--p95-ms', type=seconds, help='query latency SLO; omit for unconstrained throughput')
    p.add_argument('--max-error-rate', type=float, default=0)
    p.add_argument('--min-success', type=positive, default=100, help='minimum successful queries per repeat for recommendations')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=Path, default=Path('artifacts/vllm-capacity'))
    p.add_argument('--dry-run', action='store_true')
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        if not 0 <= args.max_error_rate <= 1 or args.port > 65535:
            raise ValueError('invalid error-rate or port')
        for field in ('batch_sizes', 'concurrency', 'batch_concurrency', 'server_tokens', 'server_seqs', 'parallel'):
            setattr(args, field, list(dict.fromkeys(getattr(args, field))))
        launch_env = None
        if args.launch_config:
            args.launch_config = args.launch_config.resolve()
            _, launch_env = prepare(args.launch_config, scope='vllm')
            args.model = args.model or launch_env['MEMORY_RERANKER_MODEL']
            if args.model != launch_env['MEMORY_RERANKER_MODEL']:
                raise ValueError('--model must match the launch config served model')
            if args.max_length > int(launch_env['VLLM_MAX_MODEL_LEN']):
                raise ValueError('--max-length exceeds server max_model_len')
            args.api_key = os.environ.get(args.api_key_env, launch_env.get('MEMORY_RERANKER_API_KEY', ''))
            launch_env['MEMORY_RERANKER_API_KEY'] = args.api_key
            args.base_url = f'http://127.0.0.1:{args.port}'
            gpu_ids = args.gpus.split(',')
            if any(not x.strip() or x != x.strip() for x in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
                raise ValueError('--gpus must contain distinct GPU identifiers')
            profiles = []
            for parallel, tokens, seqs in itertools.product(args.parallel, args.server_tokens, args.server_seqs):
                parts = parallel.split(':')
                if len(parts) != 2:
                    raise ValueError('--parallel expects TP:DP')
                tp, dp = map(int, parts)
                if min(tp, dp) < 1 or tp * dp > len(gpu_ids):
                    raise ValueError('TP*DP must fit the selected GPUs')
                if tokens < int(launch_env['VLLM_MAX_MODEL_LEN']) or seqs > tokens:
                    raise ValueError('require server-tokens >= max_model_len and server-seqs <= server-tokens')
                profiles.append(dict(name=f'tp{tp}-dp{dp}-tok{tokens}-seq{seqs}', tp=tp, dp=dp, tokens=tokens, seqs=seqs))
            if not args.dry_run and sys.platform != 'linux':
                raise ValueError('--launch-config requires Linux; use --dry-run here or test an existing URL')
        else:
            args.model = args.model or os.environ.get('MEMORY_RERANKER_MODEL', 'MemReranker-4B')
            args.api_key = os.environ.get(args.api_key_env, '')
            profiles = [dict(name=args.server_label, settings='externally managed; label is not verified server metadata')]
        parsed = urlsplit(args.base_url)
        parsed.port
        if (parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment or parsed.path not in ('', '/')):
            raise ValueError('--base-url must be a server root without credentials/query/path')
        records = load_records(args)
        count = len(profiles) * len(args.batch_sizes) * len(args.concurrency) * len(args.batch_concurrency) * args.repeats
        print(f'{len(profiles)} server profiles, {count} trials; at least {count * (args.warmup + args.duration + args.cooldown) / 60:.1f} min + startup/drain')
        if args.dry_run:
            print(json.dumps(dict(profiles=profiles, batch_sizes=args.batch_sizes, concurrency=args.concurrency,
                                  batch_concurrency=args.batch_concurrency), indent=2))
            return 0
        outputs = [Path(str(args.output) + suffix) for suffix in ('.json', '.csv')]
        if any(path.exists() for path in outputs):
            raise ValueError('output exists; choose a new --output prefix')
        if args.launch_config and port_busy(args.port):
            raise ValueError('benchmark port is occupied; choose a dedicated free --port')
        args.output.parent.mkdir(parents=True, exist_ok=True)
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    try:
        return asyncio.run(run(args, records, profiles, launch_env))
    except KeyboardInterrupt:
        print('Interrupted; completed trials saved and owned test server stopped.', file=sys.stderr)
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
