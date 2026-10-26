# Reranker 后端与上下文评分

[English](RERANKING.md) | [简体中文](RERANKING.zh-CN.md) | [返回 README](../../README.zh-CN.md)

PaiMem 支持进程内 Sentence Transformers CrossEncoder 和 vLLM pooling `/score` 服务。[默认配置](../../configs/default.toml) 通过 vLLM 启用 MemReranker-4B，并开启上下文重排；baseline 配置关闭重排。

请编辑 `configs/config.local.toml` 中已有分组；环境变量覆盖和配置文件选择规则见[配置说明](../configuration.zh-CN.md)。切换后端或模型后需重启服务。仅更换 reranker 无需重建记忆数据库。

## vLLM 部署

`[reranker.vllm]` 同时配置服务端和 PaiMem 客户端。以下字段与默认配置一致；需替换密钥占位符，或明确设置 `api_key = ""` 关闭鉴权：

```toml
[reranker]
enabled = true
backend = "vllm"
candidates = 400
max_length = 1024
small2big_enabled = true
score_threshold = ""

[reranker.vllm]
deployment = "local"
model = "MemReranker-4B"
base_url = "http://127.0.0.1:8089"
api_key = "REPLACE_WITH_RERANKER_API_KEY"
timeout = 120
batch_size = 128
```

运行前需完成服务与 embedding 的其他必填配置。本机部署 vLLM 需要 Linux 和兼容的 NVIDIA GPU：

```bash
bash scripts/setup_env.sh
# 终端 1：保持 vLLM 运行，并等待服务就绪。
bash scripts/start_vllm.sh
```

```bash
# 终端 2：vLLM 就绪后启动 PaiMem。
bash start_svr.sh --check
bash start_svr.sh
```

连接已启动的服务时，将 `deployment` 设为 `"remote"`，并填写其地址、服务模型名和密钥。该服务可位于本机或其他机器。此模式下环境准备跳过本机 vLLM 安装与重排模型下载；执行 `setup_env.sh` 和 `start_svr.sh`，无需执行 `start_vllm.sh`。

vLLM 启动脚本使用 MemReranker 所需的 Qwen3 重排模板与分类头配置：

```text
--runner pooling
--hf_overrides '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}'
```

环境、模型下载和就绪检查见[vLLM 部署](../vllm.zh-CN.md)，模型说明见 [MemReranker 模型卡](https://huggingface.co/IAAR-Shanghai/MemReranker-4B)。

## 本地 CrossEncoder

设置 `backend="cross-encoder"`，由 PaiMem 进程加载重排模型。后端值为 `cross-encoder`；`local` 是配置分组名称，不是后端选项。

```toml
[reranker]
enabled = true
backend = "cross-encoder"
score_threshold = ""

[reranker.local]
model_path = "models/bge-reranker-v2-m3"
repo_id = "BAAI/bge-reranker-v2-m3"
download = true
revision = "main"
device = "cuda:0"
batch_size = 16
```

CPU 推理可设置 `device="cpu"`。执行 `setup_env.sh` 准备依赖和所选模型，再执行 `start_svr.sh` 加载模型。已有模型目录可设置 `download=false`。此后端不使用 vLLM 连接配置。

## 评分与上下文

| 字段 | 默认配置值 | 行为 |
|---|---:|---|
| `reranker.candidates` | `400` | 候选数，至少为内部 `top_k`，不超过可用记忆数 |
| `reranker.max_length` | `1024` | 评分输入的最大长度 |
| `reranker.small2big_enabled` | `true` | 评分时包含同会话相邻片段 |
| `reranker.score_threshold` | `""` | 空值关闭阈值过滤，否则填写数值字符串 |
| `reranker.vllm.batch_size` | `128` | 每次 HTTP `/score` 请求的文档条数 |
| `reranker.vllm.timeout` | `120` | 单次请求超时，单位为秒 |

开启 small-to-big 后，评分文本包含检索序列中前后各一个位置、且属于同一会话的片段。此评分窗口与最终输出扩展分别控制。重排分数用于排序，并可在 MemPursuit 和上下文打包前过滤种子片段。

vLLM 客户端顺序发送各批请求。200 个候选、`batch_size=128` 时，发送两次请求，分别包含 128 和 72 条文档。该参数不指定 vLLM 内部的 GPU 批量。请求失败会直接返回错误，不自动重试、丢弃候选或回退至本地后端。

客户端以 `truncate_prompt_tokens` 传递 `max_length`。该值不能超过 `reranker.vllm.max_model_len`，长度包含 query、document 和模板 token；截断可能移除有用上下文，服务端需支持此请求参数。GPU 分配、精度和服务端显存通过 `[reranker.vllm]` 配置；本地 CrossEncoder 的 device 设置不作用于 HTTP 后端。

客户端保留服务端分数，不额外应用 sigmoid 或 logit 变换。切换模型或后端后应重新校准阈值。关闭过滤时设置 `reranker.score_threshold=""`；如果已导出 `MEMORY_RERANK_SCORE_THRESHOLD`，可将其设为空字符串，或取消导出以恢复 TOML 值。仅执行 `unset` 不会清除 TOML 中的非空阈值。

`base_url` 应为服务根地址，不附加 `/v1` 或 `/score`；`model` 必须与服务模型名一致。PaiMem 启动时校验配置，不主动探测重排端点；连通性、鉴权及服务模型可用性由评分请求验证。

## 运行时与 Python API

`POST /config` 支持 `reranker_enabled`、`rerank_candidates` 和 `rerank_small2big_enabled`。重新启用要求 reranker 已在启动时配置。运行时修改只影响当前进程，不写回 TOML。管理接口访问方式见[配置说明](../configuration.zh-CN.md)。

直接使用 Python API 时，在首次创建 store 前调用 `configure_store()`：

```python
from paimem.api import configure_store

configure_store(
    reranker_backend="vllm",
    reranker_model="MemReranker-4B",
    reranker_base_url="http://127.0.0.1:8089",
    reranker_api_key="",  # 服务端启用鉴权时填写对应密钥。
    reranker_batch_size=128,
    reranker_timeout=120,
    reranker_max_length=1024,
    rerank_score_threshold=None,
)
```

显式参数优先于环境变量。直接 API 的默认值与启动预设不同：后端默认为 `cross-encoder`，评分长度为 512；未指定模型时不构建主 reranker。在其他应用中集成 PaiMem 时应显式设置所需参数。

角色名抽取复用主 reranker；若指定不同的 `role_name.reranker_model`，则在本地加载独立的角色名重排模型。

## 性能评测

使用 `scripts/benchmark_rerankers.py` 对比本地和 vLLM 推理，步骤见[重排性能评测](../reranker-benchmark.zh-CN.md)。对比时使用相同输入，并记录模型、模板、截断、批量及硬件配置。吞吐与显存占用取决于实际部署配置。
