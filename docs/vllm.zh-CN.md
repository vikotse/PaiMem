# vLLM 环境与启动

[English](vllm.md) | [简体中文](vllm.zh-CN.md) | [返回 README](../README.zh-CN.md)

本流程在兼容的 Linux NVIDIA GPU 机器部署 MemReranker/Qwen3。先确认 [vLLM GPU 要求](https://docs.vllm.ai/en/v0.32.0/getting_started/installation/gpu.html)，环境准备脚本不安装驱动。

## 部署方式与参数

| 字段 | 本机部署（`deployment="local"`） | 已有服务（`deployment="remote"`） |
|---|---|---|
| `model` | 服务模型名，示例 `MemReranker-4B` | 必须匹配已有服务的模型名 |
| `base_url`、`api_key` | 客户端连接地址，密钥也用于服务端鉴权 | 已有服务地址与密钥 |
| `model_path`、`repo_id`、`download`、`revision` | GPU 机器上的模型下载配置 | 忽略 |
| `env`、`python`、`version`、`chat_template`、`chat_template_url` | 独立 vLLM 环境与模板准备 | 忽略 |
| GPU、监听参数 | 配置本机 vLLM 进程 | 忽略 |

`"remote"` 表示连接已启动的服务，可以位于本机或另一台机器。[Reranker 性能评测](reranker-benchmark.zh-CN.md) 提供使用相同输入对比后端的方法。

## 1. 配置 GPU 机器

复制 `configs/default.toml` 为 `configs/config.local.toml`，设置 `[reranker]` 的 `enabled=true`、`backend="vllm"`。`[reranker.vllm]` 已给出模型仓库/目录、服务名、独立环境、Python/vLLM 版本、模板 URL、GPU 与监听参数。设置 `deployment="local"`，替换 `api_key`；或明确清空以关闭 API key 鉴权。保持 `base_url` 与 `host`/`port` 一致。

模板和分类转换专用于 [Qwen3 评分](https://docs.vllm.ai/en/v0.32.0/examples/pooling/score/) 及 [MemReranker](https://huggingface.co/IAAR-Shanghai/MemReranker-4B)。更换模型系列或 vLLM 版本时需重新核对。`download=false` 使用已有模型目录，非空的已有模板会直接复用。

```bash
bash scripts/setup_env.sh --only vllm
bash scripts/start_vllm.sh
```

仅准备 GPU 服务时无需填写记忆服务或 embedding 凭据。脚本在需要时安装 uv 和小型 Python 引导环境，创建独立 vLLM 环境、下载重排模型并准备模板。保持服务端终端运行。同机准备两个服务时，补齐启用组件凭据并运行不带 `--only` 的 `setup_env.sh`。

## 2. 配置记忆服务机器

设置 `reranker.backend="vllm"`，在 `[reranker.vllm]` 中填写 `deployment="remote"`、实际 `base_url`、匹配的服务模型名 `model` 和 `api_key`。无需本地重排目录、模板或 vLLM 安装。服务地址不加 `/v1` 或 `/score`。在同一文件补齐 embedding 与 service 配置。

同机部署时保持 `deployment="local"` 与回环地址即可。分机部署时，GPU 服务需监听可访问地址，例如 `host="0.0.0.0"`，并适当限制访问范围。

```bash
bash scripts/setup_env.sh --only service
```

## 3. 验证模型，再启动 PaiMem

在仓库根目录运行，以下请求直接读取所选 TOML 及 export 覆盖项：

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

`/score` 请求成功表示模型就绪，`--check` 校验配置，并只读检查已有数据库的兼容性，不加载模型或请求接口。三个脚本均接受 `--config PATH`。客户端 token 预算 `reranker.max_length` 不得超过 `reranker.vllm.max_model_len`。远程失败不会静默切回本地推理。

`reranker.vllm.batch_size` 控制每次 `/score` 请求的文档条数，`configs/default.toml` 中为 128，与 `reranker.local.batch_size` 和 vLLM 张量并行数独立。
