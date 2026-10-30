# Add/Search load testing

[English](load-testing.md) | [简体中文](load-testing.zh-CN.md) | [Configuration](configuration.md)

`scripts/load_test.py` measures PaiMem's `/add` and `/search` endpoints. It supports simultaneous or single-endpoint load, and a sequential smoke-test mode. These are end-to-end API measurements; use [reranker capacity testing](reranker-benchmark.md) to measure vLLM separately.

## Run from a client host

Run from a PaiMem checkout with the project Python dependencies installed. The client and server can be on different machines. Set `PAIMEM_URL` to the reachable PaiMem address; `127.0.0.1` refers to the client itself. The server must listen on a reachable interface and its port must be accessible from the client. This script does not start or configure either server.

```bash
export PAIMEM_URL='http://paimem.example:8000' # Replace with your service address
# Set MEMORY_API_KEY in this shell to the service's API key.
.venv/bin/python scripts/load_test.py \
  --mode mixed \
  --base-url "$PAIMEM_URL" \
  --add-concurrency 16 \
  --search-concurrency 16 \
  --users 16 \
  --seed-documents 400 \
  --messages-per-add 10 \
  --duration 120 \
  --timeout 180 \
  --output /tmp/paimem-mixed-16.json
```

The URL is the PaiMem service root, without a path, credentials, query string or fragment. Mixed/add/search modes read the key from `MEMORY_API_KEY`; `--api-key-env NAME` selects another variable. These modes use persistent HTTP connections and do not read proxy settings from the environment (`trust_env=False`), so they require a direct route to the service. An output file must not already exist and its parent directory must exist. Add `--dry-run` to validate arguments and input without sending requests or creating files.

`--mode add` or `--mode search` measures only that endpoint, but **both modes still seed data and run validation searches**. Without `--mode`, the existing sequential smoke test runs: serial adds followed by concurrent searches. Its output schema and latency definitions differ from the timed modes. Inspect the applicable options with `--mode mixed --help` or `--help`.

## Workload and retained data

Each run generates unique user and request identifiers. With the example settings, it seeds 400 source documents for each of 16 users (6,400 in total), using batches of 20 and up to 4 seed workers. A nonempty validation search is required for each user before measurement. Seeding and validation requests are excluded from reported measurements, but still consume server resources and provider quota. They are a readiness check, not a separately controlled warm-up period. Report source-document counts do not imply the same number of stored chunks or reranker candidates.

The default workload is synthetic Chinese project text. To provide your own workload, use `--input data/load-input.jsonl`; each nonempty line must contain a nonempty `query` and at least `--seed-documents` nonempty document strings:

```json
{"query":"When is the release review?","documents":["The release review is on Friday.","Testing finishes on Thursday."]}
```

For this two-document example, pass `--seed-documents 2`. Users cycle through input rows, and only the first configured number of documents per row is used. Add workers cycle through those documents with fresh request IDs. Reports label file input as `jsonl`, synthetic input as `synthetic`, and include a hash of the effective input. A file input is not necessarily representative of production traffic; record its provenance and document/token length distribution when comparing runs.

Test data remains in the database. The script does not call clear/delete/config endpoints. Mixed and add modes grow the corpus during measurement, while every run adds a fresh seeded corpus. Use a dedicated test database and keep corpus size, input, strategies, model revisions, quotas and concurrency comparable between runs. A nonempty search alone does not establish retrieval quality.

## Measurement definitions

- Each worker uses a closed loop: send one request, wait for completion, then send another until the admission window ends. Concurrency 16 is not 16 requests per second; mixed 16/16 permits up to 32 measured requests in flight.
- `--duration` is the window for starting measured requests. Requests already in flight drain afterward, subject to `--timeout` per request. `wall_seconds` includes this drain, so a 120-second run can take longer; seeding time is additional.
- `successful_qps` divides successful requests by the full measured wall time, including drain. For add, this counts API calls, not documents. `messages_per_add` controls messages in each measured add request.
- `mean_ms`, `p50_ms`, `p95_ms` and `p99_ms` include successful requests only. Percentiles use the nearest-rank method. Read them alongside `error_rate`, error counts and `failed_mean_ms`; a timeout-heavy run can have deceptively low success-only latency.
- Search responses are structurally validated. Empty measured searches are counted in `empty_searches` but are still successful API calls. Any failed measured request makes the report status `failed`; seeding failures produce `seed_failed` without measured load.
- The client performs no request retries. A timeout does not prove an add was rolled back: work may continue on the server. Ctrl+C during measurement cancels client requests and reports `cancelled_inflight`; it does not cancel server work or remove data. Interruption during seeding may leave partial data without a full JSON report.

Reports contain aggregate metrics and generated identifiers, not input text, response bodies or API keys. They are not a general export of server diagnostics.

## Compare client and server timings

Enable `MEMORY_REQUEST_TIMING_ENABLED=1` on the server for one elapsed-time record per add/search request. Use `MEMORY_TIMING_ENABLED=1` temporarily for stage-level investigation; details and timing boundaries are in [configuration](configuration.md#request-and-stage-timing). Server timing logs also include seeding and validation traffic, so isolate the measured interval before comparing aggregates.

Keep the workload fixed when changing one server setting. Compare throughput, latency percentiles, errors and provider quota waits. Then vary client concurrency to find the range the service sustains. Remote embedding worker concurrency and RPM/TPM are shared per embedder instance, while the script's add/search concurrency controls client requests; these are different limits. Changes that alter retrieval strategies or candidate counts also require separate quality evaluation.
