# PaiMem

[English](README.md) | [简体中文](README.zh-CN.md)

**融合混合检索与上下文感知证据选择的长期记忆框架**

*Long-Term Memory with Hybrid Retrieval and Context-Aware Evidence Selection*

PaiMem 保存原始对话片段，通过语义与词法混合召回、上下文重排、证据选择和上下文重建，为下游问答提供可追溯的历史证据。提供 SQLite 持久化、用户隔离的 HTTP API，以及 Recall 和端到端评测工具。

## 核心思路

PaiMem 将长期记忆读取建模为**查询条件下的上下文证据选择**，遵循三个原则：

- **保留原始信息，延后证据组织**：保留对话细节与来源，在查询时确定证据的用途，降低预先压缩造成的信息损失。
- **解耦检索粒度与判断粒度**：以细粒度片段定位信息，以上下文扩展后的表示进行相关性判断和证据选择。
- **联合考虑相关性与互补性**：通过重排评估候选相关性，并以 MemPursuit 的查询重构残差作为未覆盖信息的代理，选择补充性证据。

```text
/add: 消息 → 分块 / embedding → SQLite
/search: 查询 → Dense + BM25 → RRF → 上下文重排 → MemPursuit
     → 可选排名融合 → 同会话上下文扩展 / 打包 → 证据
```

SIRA 查询扩展、NEAR 残差检索、角色过滤、冲突证据核验和长问题处理均为可选策略。Search 返回证据，回答生成与裁判属于独立评测流程。各策略细节见 [策略文档](docs/configuration.zh-CN.md)。

## 1. 快速开始

脚本支持 macOS/Linux 的 Bash。PaiMem 默认使用 Python 3.11；vLLM 需要兼容的 Linux NVIDIA GPU 机器。三个脚本统一读取**一份 TOML**，部署信息与策略参数集中填写。

```bash
cd /path/to/PaiMem
cp -n configs/default.toml configs/config.local.toml
# 编辑 configs/config.local.toml，再准备环境。
```

### 默认配置的必填参数

默认方案为 **remote embedding + 在本机通过 vLLM 部署 MemReranker**。在兼容的 Linux NVIDIA GPU 机器上，先修改 `configs/config.local.toml` 中以下 **4 个字段**：

| 字段 | 填写内容 |
|---|---|
| `service.api_key` | 自行设置的记忆服务访问密钥，调用记忆 API 时使用。 |
| `embedding.remote.base_url` | embedding 供应商的实际 API 地址，不附加 `/embeddings`。 |
| `embedding.remote.api_key` | embedding 供应商的 API 密钥。 |
| `reranker.vllm.api_key` | 将占位符替换为自行设置的密钥，或明确填写 `api_key = ""` 关闭 API key 鉴权，不能保留占位符。 |

**启动 vLLM 与 PaiMem 调用重排服务都读取 `reranker.vllm.api_key`**，只需配置一次，无需分别填写服务端、客户端密钥。

如果供应商使用其他模型名，再修改 `embedding.remote.model`（模板为 `text-embedding-v4`），RPM/TPM 按供应商配额调整。其余策略参数和关闭策略的示例可先保留。remote embedding 不下载模型，环境准备仅下载所选 reranker 及模板。

> **更换 embedding 模型时**，请同时将 `service.db_path` 设置为新路径，例如 `artifacts/memory-bge-m3.db`，并重新写入原始数据。已有向量不会自动转换；新数据库初始为空，原数据库保留不变。仅调整 API key、批量大小或 RPM/TPM 无需换库。
>
> 兼容性检查还记录 embedding 后端、远程 `base_url`、本地模型路径、query prefix 及入库上下文配置（`context.messages`）；修改这些字段可能阻止复用非空数据库。检查无法识别模型名或路径不变时的权重变化。

```bash
# 安装/检查 uv、安装依赖，并下载所选重排模型及模板。
bash scripts/setup_env.sh
# 终端 1：启动 vLLM，保持运行并等待模型就绪。
bash scripts/start_vllm.sh
```

```bash
# 终端 2，在仓库根目录执行：
bash start_svr.sh --check
bash start_svr.sh
```

**连接已启动的 vLLM 重排服务时**，将 `reranker.vllm.deployment` 设置为 `"remote"`，并填写该服务的地址 `base_url`、服务模型名 `model` 和密钥 `api_key`（未启用 API key 鉴权则填 `""`）。这里的 `"remote"` 表示“使用已有服务”，该服务可以在本机，也可以在另一台机器。随后执行环境准备脚本 `setup_env.sh` 和 PaiMem 启动脚本 `start_svr.sh` 即可，无需下载重排模型、安装 vLLM 或运行 `start_vllm.sh`。

无需手动激活环境。PaiMem 安装到 `.venv`，vLLM 使用独立环境；模型准备完成且 vLLM 就绪后再启动 PaiMem。`--check` 校验配置，并只读检查已有数据库的兼容性，不加载模型或请求远端；正常启动也会在加载模型前执行相同检查。

本地 embedding、CrossEncoder、配置覆盖与策略参数见[配置说明](docs/configuration.zh-CN.md)；vLLM 参数、分机器部署和就绪验证见[vLLM 部署](docs/vllm.zh-CN.md)。

## 2. 接口与请求示例

在另一个终端设置访问地址，并将 `MEMORY_API_KEY` 填为 `service.api_key` 的相同值；后续请求与评测均使用这两个变量：

```bash
export MEMORY_BASE_URL='http://127.0.0.1:8000'
export MEMORY_API_KEY='YOUR_MEMORY_SERVICE_KEY'
```

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 服务健康及记忆数量，无须鉴权 |
| POST | `/add` | 同步写入，成功后立即可检索 |
| POST | `/search` | 按 user_id 检索，top_k 为 1–100 |
| GET | `/config` | 查看生效配置，不返回密钥 |
| POST | `/config` | 更新受支持的运行时开关和参数 |
| POST | `/clear` | 清空整个专用记忆数据库，需鉴权 |
| GET | `/docs` | 自动生成的 OpenAPI 接口文档 |

受保护接口使用 `X-Api-Key`，也支持 `Authorization: Bearer <key>` 或 `Token <key>`。管理接口 `/config` 与 `/clear` 必须提供服务密钥，即使设置 `ALLOW_UNAUTHENTICATED=true` 也不豁免。参数列表见 `/docs` 及配置说明。

```bash
curl --fail "$MEMORY_BASE_URL/health"

curl --fail "$MEMORY_BASE_URL/add" \
  -H "X-Api-Key: $MEMORY_API_KEY" -H 'Content-Type: application/json' \
  -d '{"request_id":"demo:1","user_id":"demo","session_id":"session:1","messages":[{"role":"user","timestamp":1704067200000,"content":"My favorite drink is green tea."}]}'

curl --fail "$MEMORY_BASE_URL/search" \
  -H "X-Api-Key: $MEMORY_API_KEY" -H 'Content-Type: application/json' \
  -d '{"query":"What is my favorite drink?","user_id":"demo","top_k":10}'

# 清空整个记忆数据库，包括所有用户的数据。
curl --fail -X POST "$MEMORY_BASE_URL/clear" \
  -H "X-Api-Key: $MEMORY_API_KEY"
```

Add 返回 `success/request_id/user_id/session_id`。相同 request_id 和相同内容重复写入幂等；内容不同返回 409。`timestamp` 为可选 Unix 毫秒。

Search 返回 `{"data":[{"id":"...","content":"...","score":0.8,"created_at":"..."}]}`；可传 `options` 支持选择题。不同 user_id 的数据隔离。使用 MemPursuit 或排名融合时应保留返回顺序，不能再按 score 排序。`include_recall_queue: true` 可返回诊断队列来源；文件诊断仅允许本机鉴权请求，并限定在 `[service].diagnostics_dir` 内写入；详见[诊断目录配置](docs/configuration.zh-CN.md#诊断输出目录)。

## 3. 复现 Recall 与端到端结果

完整流程和数据集对应关系见 [评测复现](docs/evaluation.zh-CN.md)。顺序是：获取固定版本 AML pipeline → 下载公开数据 → 转为 normalized JSONL → 启动指定配置服务 → 执行评测。

Recall 示例（不执行 answer/judge，启用的服务策略仍可能调用模型）：

```bash
uv run python scripts/local_benchmark.py \
  --input data/normalized/locomo_refined.fixed.jsonl \
  --output artifacts/recall/locomo.jsonl \
  --base-url "$MEMORY_BASE_URL" --run-label locomo-run-001 --top-k 100
```

端到端示例（先 export `ANSWER_API_BASE/ANSWER_MODEL/ANSWER_API_KEY` 和对应 `JUDGE_*`）：

```bash
PYTHONHASHSEED=0 uv run python scripts/run_aml_e2e.py \
  --input data/normalized/locomo_refined.fixed.jsonl \
  --output artifacts/aml-e2e/run-001/locomo_refined \
  --pipeline agent-memory-leaderboard/data/locomo-refined/pipeline.py \
  --base-url "$MEMORY_BASE_URL" --top-k 100 --concurrency 2 \
  --save-search-results
```

相同配置和输出目录支持续跑；更换模型、数据或策略必须使用新的输出目录。不要在续跑中清空记忆。评测结果需同时记录代码版本、模型、数据和配置，详见评测复现文档。

## 4. 效率对比

在 **LoCoMo-refined** 数据集上评测 PaiMem 的写入与检索延迟。采用默认参数，将 embedding 后端切换为 **local**，避免远程 embedding 网络请求的影响；reranker 通过 **vLLM** 部署，两种 GPU 均通过 `gpu_memory_utilization` 将 vLLM 显存预算控制在约 **20 GB**。

本次评测包含 **1 条 record、21 次 Add 请求和 138 个问题**。

| GPU | 平均 Add 延迟（秒/请求） | 平均 Search 延迟（秒/请求） |
|---|---:|---:|
| NVIDIA A800 (80GB) | 0.1581 | 3.5244 |
| NVIDIA L4（24GB） | 0.0380 | 12.3927 |

以上为本次配置与样本下的平均请求延迟。Search 延迟衡量记忆检索过程，不包含下游回答生成与评判。

## 5. 贡献者

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/vikotse">
        <img src="docs/assets/contributors/vikotse.svg" width="80" height="80" alt="vikotse"/><br/>
        vikotse
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Vicky-shi">
        <img src="docs/assets/contributors/Vicky-shi.svg" width="80" height="80" alt="Vicky-shi"/><br/>
        Vicky-shi
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/nicole0220">
        <img src="docs/assets/contributors/nicole0220.svg" width="80" height="80" alt="nicole0220"/><br/>
        nicole0220
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/hiudawn">
        <img src="docs/assets/contributors/hiudawn.svg" width="80" height="80" alt="hiudawn"/><br/>
        hiudawn
      </a>
    </td>
  </tr>
</table>

## 6. 参考工作

| 工作 | 本项目使用方式 |
|---|---|
| [InvMem / vanilla-rag-memory](https://github.com/wenxiaof345-ctrl/vanilla-rag-memory) | 基础 RAG 与服务实现的来源 |
| [MemPursuit — Recall What is Missing: Composing Evidence from Long-Term Agent Memory](https://github.com/FengQi-HITSZ/MemPursuit) | 改编固定预算 NumPy 证据选择算法 |
| [SIRA](https://github.com/facebookresearch/sira) · [论文](https://arxiv.org/abs/2605.06647) | 查询扩展、DF 过滤及词法增强适配 |
| [NEAR: Neighborhood-Aware Residuals for Memory Retrieval](https://arxiv.org/abs/2610.13314) | 可选邻域残差评分实现 |
| [MemReranker](https://huggingface.co/IAAR-Shanghai/MemReranker-4B) · [论文](https://arxiv.org/abs/2605.06132) | 可选的记忆重排模型 |
| [Agent Memory Leaderboard](https://github.com/AML-memory/agent-memory-leaderboard) | 评测协议与外部 answer/judge pipeline |

来源与许可见 [第三方说明](THIRD_PARTY_NOTICES.zh-CN.md)，各策略在 PaiMem 中的实现见 [策略文档](docs/configuration.zh-CN.md#算法说明)。

## 7. License

PaiMem 采用 [MIT License](LICENSE)。第三方组件许可见[第三方说明](THIRD_PARTY_NOTICES.zh-CN.md)。

## 8. Citation

```bibtex
@software{paimem2026,
  author = {Xie, Weijian and Shi, Lihui and Zhang, Xinyue and Gao, Xiaodong},
  title = {PaiMem: Long-Term Memory with Hybrid Retrieval and Context-Aware Evidence Selection},
  year = {2026},
  url = {https://github.com/vikotse/PaiMem}
}
```

机器可读引用见 [CITATION.cff](CITATION.cff)。引用实验结果时请另记代码 commit、模型 revision、配置、数据划分和评测版本。
