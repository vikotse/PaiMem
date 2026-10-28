# vLLM environment and startup

[English](vllm.md) | [简体中文](vllm.zh-CN.md) | [README](../README.md)

This workflow serves MemReranker/Qwen3 on a compatible Linux NVIDIA GPU host. Confirm the [vLLM GPU requirements](https://docs.vllm.ai/en/v0.32.0/getting_started/installation/gpu.html); drivers are not installed by the preparation script.

## Deployment modes and parameters

| Field | Local deployment (`deployment="local"`) | Existing server (`deployment="remote"`) |
|---|---|---|
| `model` | Served name, example `MemReranker-4B` | Must match the server's served name |
| `base_url`, `api_key` | Client connection; key also configures server auth | Existing server address and credential |
| `model_path`, `repo_id`, `download`, `revision` | Model download on GPU host | Ignored |
| `env`, `python`, `version`, `chat_template`, `chat_template_url` | Separate vLLM environment and template preparation | Ignored |
| GPU/listener parameters | Configure this vLLM process | Ignored |

`"remote"` means connecting to an already running server, on this machine or another host. [Reranker performance evaluation](reranker-benchmark.md) describes how to compare backends with the same inputs.

## 1. Configure the GPU host

Copy `configs/default.toml` to `configs/config.local.toml`. Set `[reranker] enabled=true, backend="vllm"`. In `[reranker.vllm]`, the template already provides model repository/directory, served name, separate environment, Python/vLLM versions, template URL, GPU and listener parameters. Set `deployment="local"` and replace `api_key`, or explicitly clear it to disable API-key authentication. Keep `base_url` consistent with `host`/`port`.

The supplied conversion and versioned template are specific to [Qwen3 scoring](https://docs.vllm.ai/en/v0.32.0/examples/pooling/score/) and [MemReranker](https://huggingface.co/IAAR-Shanghai/MemReranker-4B). Review them when changing model families or vLLM versions. `download=false` uses an existing model directory; an existing nonempty template is reused. The default template is bundled at `third_party/vllm/qwen3_reranker.jinja` and copied to the configured `chat_template` path without contacting GitHub. Existing configurations need no changes. Only a different `chat_template_url` triggers a download. Include `third_party/vllm` when syncing the repository. See [third-party notices](../THIRD_PARTY_NOTICES.md) for provenance and licensing.

```bash
bash scripts/setup_env.sh --only vllm
bash scripts/start_vllm.sh
```

The GPU-only setup does not require memory-service or embedding credentials. It installs uv and the small Python bootstrap as needed, creates the isolated vLLM environment, downloads the reranker, and prepares the template. Leave the server terminal running. To prepare both services on one machine, complete all active credentials and run `setup_env.sh` without `--only`.

## 2. Configure the memory host

On the memory host, set `reranker.backend="vllm"`. In `[reranker.vllm]`, set `deployment="remote"`, the actual `base_url`, matching served `model` and `api_key`. No local reranker directory, template or vLLM installation is required. The server base must not end in `/v1` or `/score`. Complete the embedding and service settings in the same file.

For co-located services, keep `deployment="local"` and the loopback URL instead. For separate hosts, the GPU server must listen on a reachable address (for example `host="0.0.0.0"` with access restricted appropriately).

```bash
bash scripts/setup_env.sh --only service
```

## 3. Check the model, then start PaiMem

From the repository root, this readiness request reads the selected TOML and exported overrides directly:

```bash
.venv/bin/python - <<'CHECK'
import httpx
from paimem.configuration import default_path, prepare
_, env = prepare(default_path())
base = env['MEMORY_RERANKER_BASE_URL'].rstrip('/')
key = env.get('MEMORY_RERANKER_API_KEY')
response = httpx.post(base + '/score',
    headers={'Authorization': 'Bearer ' + key} if key else {},
    json={'model': env['MEMORY_RERANKER_MODEL'],
          'text_1': 'What do I like to drink?',
          'text_2': ['I like green tea.', 'The meeting starts at nine.'],
          'truncate_prompt_tokens': int(env['MEMORY_RERANKER_MAX_LENGTH'])},
    timeout=float(env['MEMORY_RERANKER_TIMEOUT']))
response.raise_for_status()
print(response.json())
CHECK
bash start_svr.sh --check
bash start_svr.sh
```

A successful `/score` call confirms model readiness; `--check` validates configuration and checks existing database compatibility read-only, without loading models or calling endpoints. All three scripts accept `--config PATH`. The client token budget `reranker.max_length` must not exceed `reranker.vllm.max_model_len`. Remote failures do not silently fall back to local inference.

`reranker.vllm.batch_size` controls documents per `/score` request (128 in `configs/default.toml`). It is independent of `reranker.local.batch_size` and of vLLM tensor parallelism.
