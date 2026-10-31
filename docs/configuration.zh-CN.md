# 配置与策略

[English](configuration.md) | [简体中文](configuration.zh-CN.md) | [返回 README](../README.zh-CN.md)

三个脚本读取同一份 TOML。将 [default.toml](../configs/default.toml)（或 [baseline.toml](../configs/baseline.toml)）复制为 `configs/config.local.toml` 后修改，模型凭据、后端部署和算法参数按功能集中配置。

## 文件选择与参数覆盖

优先级为 `--config PATH` → `PAIMEM_CONFIG` → 已存在的 `configs/config.local.toml` → `configs/default.toml`。所选文件是一份完整配置，不与其他预设叠加。使用 Shell 入口时，相对路径基于仓库根目录。

已有 export 覆盖所选 TOML；每个字段旁的注释标明对应环境变量名。空 export 也优先，必填项为空时会报错；执行 `unset 变量名` 可恢复文件值。例如：

```bash
export EMBEDDING_BATCH_SIZE=20
bash start_svr.sh --config configs/config.local.toml --check
```

仅映射所选后端：`embedding.local` 与 `embedding.remote` 二选一，`reranker.local` 与 `reranker.vllm` 二选一。通过 `embedding.backend` 和 `reranker.backend` 选择后端。本地和远程示例分别具有自己的批量和模型设置。`reranker.enabled=false` 强制关闭主重排，即使之前导出过模型变量。

`REPLACE_WITH_*` 与 `*.example` 是占位值，启用组件的必需字段必须填写，关闭的策略可保留示例。空 `reranker.vllm.api_key` 明确表示不使用服务端 API key 鉴权；空 `reranker.score_threshold` 关闭分数过滤。本地 `query_prefix="auto"` 保留模型默认，`""` 明确移除前缀。`download=false` 表示使用已有模型目录；`revision="main"` 跟随模型主分支，提交 SHA 则固定版本。

`--check` 校验启用组件的必填凭据字段、类型、部分数值范围和策略依赖，并只读检查已有数据库的兼容性。该过程不加载模型、不调用远程服务，也不验证凭据是否被供应商接受；其余数值约束在 API 启动时检查。`setup_env.sh --only vllm` 仅校验 vLLM 配置。校验输出不包含密钥。

每个 LLM 策略（`sira`、`role_name`、`conflict`、`long_question`）的模型、地址、密钥、超时和算法参数都在同一分组。只有连接字段明确为空时，才使用共享 `ANSWER_*`/`OPENAI_*` 回退；非空占位值必须替换。建议直接在策略分组填写实际值，减少歧义。

记忆服务启动器自动读取 TOML。其他仅消费环境变量的 Python 工具需要先显式加载配置，在仓库根目录的 Bash 中运行：

```bash
config_exports=$(.venv/bin/python -m paimem.configuration --config configs/config.local.toml --shell)
eval "$config_exports"
unset config_exports
```

`--shell` 输出经过转义的 export，**包含凭据**，仅按上述方式捕获使用，不要打印或记录。TOML 本身是数据，不作为 Shell 代码执行。

## 配置分组

| 分组 | 集中配置的内容 |
|---|---|
| `[service]`、`[environment]` | API 密钥、数据库、监听地址；Python、依赖和模型下载地址 |
| `[embedding]`、`.remote`、`.local` | 后端、模型/API 或下载路径、batch size、并发数、RPM/TPM |
| `[reranker]`、`.local`、`.vllm` | 开关、候选数、上下文、本地模型或 vLLM 客户端与服务端 |
| `[retrieval]`、`[context]`、`[time]` | 混合检索、分块、上下文重建、时间处理 |
| `[mempursuit]`、`[rank_fusion]`、`[near]` | 各策略开关和数值参数 |
| `[sira]`、`[role_name]`、`[conflict]`、`[long_question]` | 各策略开关、LLM 模型/地址/密钥及算法参数 |

## Embedding：remote 或本地

### Remote embedding：采用第三方 embedding 模型 API，不需下载模型

在复制的文件中修改已有分组，不要追加同名 TOML 表：

```toml
[embedding]
backend = "remote"

[embedding.remote]
model = "text-embedding-v4" # 示例，使用供应商接受的模型名
base_url = "https://your-provider.example/v1" # 替换为实际地址
api_key = "REPLACE_WITH_EMBEDDING_API_KEY"
batch_size = 10
max_concurrency = 4
rpm = 1500    # 示例配额，按实际值设置；0 关闭
tpm = 1800000 # 示例 token 配额；0 关闭
query_prefix = "" # 明确表示不加前缀
```

环境准备会跳过全部 embedding 下载，无需本地模型目录，也无需 embedding vLLM 进程。API 地址不要追加 `/embeddings`。RPM/TPM 按滚动 60 秒计算，TPM 先按 UTF-8 字节长度预留，再按供应商返回的有效 token 用量修正。上例配额仅供参考，请按供应商实际限制设置。详见[限流行为](#embedding-批量与远程限流)。

### 本地 embedding：先准备模型，再启动 PaiMem

将 `embedding.backend` 改为 `"local"`，本地分组已预填示例：

```toml
[embedding.local]
model_path = "models/bge-small-en-v1.5"
repo_id = "BAAI/bge-small-en-v1.5"
download = true
revision = "main" # 固定复现时替换为模型提交 SHA
device = "cuda:0" # 或 cpu、mps
batch_size = 64
query_prefix = "auto" # 保留模型默认；空字符串明确关闭前缀
```

环境准备脚本下载到 `model_path`；设置 `download=false` 可复用已有目录。模型由 PaiMem 启动时加载，远程密钥和远程批量配置会被忽略。更换 embedding 模型、后端或上下文编码方式时，使用新的 `service.db_path` 并重新写入数据。

## Reranker 后端

推荐通过 [vLLM](vllm.zh-CN.md) 部署 MemReranker/Qwen3；也可使用以下本地后端。

### 本地 CrossEncoder：由 PaiMem 加载

设置 `reranker.backend="cross-encoder"`。`[reranker.local]` 已预填 `BAAI/bge-reranker-v2-m3`、下载目录、default 配置的 `device="cuda:0"`（baseline 为 `cpu`）与 `batch_size=16`，按需修改即可。`download=false` 复用已有目录。无需启动 vLLM，也无需填写 vLLM 密钥。CUDA 推理需兼容驱动和 PyTorch，脚本不安装驱动。

重排候选数、输入 token 预算、得分阈值和上下文开关都在 `[reranker]` 中，与后端选择相邻。换模型后可能需重新校准阈值。[基线模板](../configs/baseline.toml) 关闭重排、MemPursuit 和排序融合，并将打包参数设为 1。

## 环境准备与其他预设

无需重复 export 或激活环境。PaiMem 按 `uv.lock` 安装到 `.venv`；vLLM 使用配置中指定的独立环境。环境准备可重复执行；启动脚本不安装依赖。模型准备需先于记忆服务启动。

三个脚本均支持 `--config configs/experiment.local.toml`。分机器部署见 [vLLM 部署文档](vllm.zh-CN.md)。

可选镜像位于 `[environment]`。已导出的 `HF_ENDPOINT` 优先于 `environment.hf_endpoint`；显式配置后不探测、不自动切换。两者都未设置时，模型准备会探测 `https://huggingface.co`（连接超时 5 秒，总超时 10 秒），成功则使用官方端点，失败则选择 `https://hf-mirror.com`。同一次准备进程中的后续下载复用该选择；复用已有模型目录时不探测。模板将 `hf_endpoint` 保留为注释示例，取消注释即可显式指定端点。

缺少 uv 时使用[官方安装器](https://docs.astral.sh/uv/getting-started/installation/)。使用基线时，将 `configs/baseline.toml` 复制为**新的** `*.local.toml`，填写启用组件的凭据，并在环境准备和启动时传入同一路径。

## Embedding 批量与远程限流

两份配置均按功能分组，并附中英文参数注释。已有 export 优先于 TOML；模型连接和算法参数位于同一功能分组。

| 变量 | default 预设值 | 含义 |
|---|---:|---|
| `EMBEDDING_BATCH_SIZE` | 本地 64；远程 10 | 正整数，每批最多输入条数，须遵守供应商单次请求限制。 |
| `EMBEDDING_MAX_CONCURRENCY` | 4 | 正整数，同一远程 embedder 实例中所有调用共用的批次工作线程数。 |
| `EMBEDDING_MAX_INPUT_TOKENS` | 8192 | 正整数，远程单条输入上限，按 UTF-8 字节数 / 4 估算，包含 query prefix。TOML：`[embedding.remote].max_input_tokens`。发送任何批次前检查全部输入，超限时本地报错、不截断。该估算不替代供应商 tokenizer 的限制。 |
| `EMBEDDING_RPM` | 1500 | 非负整数，每滚动 60 秒允许的远程 HTTP 请求尝试次数；0 关闭该限制。 |
| `EMBEDDING_TPM` | 1800000 | 非负整数，每滚动 60 秒的远程 token 预算，发送前估算、响应后按用量修正；0 关闭该限制。 |

表中数值来自 `configs/default.toml`；RPM/TPM 需按供应商配额调整。`baseline` 预设及未显式设置配额的直接 API 启动使用 0，表示关闭对应限流。

```bash
# 以下配额仅为示例，请替换为供应商实际限制。
export EMBEDDING_BATCH_SIZE=10
export EMBEDDING_MAX_CONCURRENCY=4
export EMBEDDING_RPM=1500
export EMBEDDING_TPM=1800000
bash start_svr.sh --config configs/config.local.toml
```

远程请求会等待至两项已启用配额均允许发送。每次重试都会重新占用配额，失败请求也计数。批次同时满足 batch size 与 TPM，必要时自动拆分；若单条输入超过 TPM，该次调用会在发送任何输入前报错，不截断文本。查询前缀计入估算。本地 embedding 仅使用 batch size。

TPM 在发送前按 UTF-8 字节长度预留，包含查询前缀。成功响应中若存在正整数 `usage.total_tokens`、`usage.prompt_tokens` 或 `usage.input_tokens`，会按此优先级选择用量，修正仍处于滚动窗口内的预留；缺少有效 usage 时保留估算。修正不退还 RPM，失败尝试保留预留额度。这是客户端计数，不保证与供应商的 tokenizer、限流窗口或计费规则完全一致，仍应预留配额余量。重试会遵守有效的 `Retry-After` 响应头（秒数或 HTTP 日期），同时应用指数退避和本地配额检查，每批最多尝试三次。

远程批次并发执行，返回结果按输入顺序恢复。query、document 和 MemPursuit 调用共用工作线程池，不再持有 store 的 embedding 锁；local/自定义后端保留该锁。工作线程限制包含等待配额或重试退避的批次，因此只是 HTTP 并发上限，并非目标请求速率。两份模板与直接构造实例的默认值均为 4。

限流器线程安全，同一 embedder 实例内的文档编码和查询调用共用额度；不同进程、不同实例或共用账号的其他客户端之间不共享计数，需自行分配配额。参数在实例创建时读取，修改后需重启服务。`--check` 会校验所选 embedding 后端的批量、并发与配额数值。


## 远程 embedding 失败诊断

失败采集默认关闭。在 `[embedding.remote]` 中设置 `failure_log_path` 为 JSONL 文件路径，或设置环境变量 `EMBEDDING_FAILURE_LOG_PATH` 后启用。空字符串或纯空白值关闭采集；显式导出的空值也会覆盖配置文件中的路径。该设置在 embedder 创建时读取，修改后需重启服务。相对路径基于服务的工作目录解析。

例如，在已有的 `[embedding.remote]` 配置节中添加 `failure_log_path = "artifacts/embedding-failures.jsonl"`。记录失败时会创建父目录，新建文件使用仅所有者可读写的权限（`0600`）；已有文件的权限不会改变。

每条记录包含 UTC 时间、批次序号、尝试次数、失败原因、可用时的 HTTP 状态码、完整 embedding 请求（包含查询前缀），以及可用时的上游响应正文。远程请求发生不可重试错误或耗尽重试次数时追加记录。本地输入长度校验失败、HTTP 成功但向量无效的情况不会生成失败记录。诊断文件写入失败会单独记日志，不覆盖原始请求错误。

采集仅将文本值中出现的当前 embedding API key 替换为 `[REDACTED]`，不进行通用脱敏；其他凭据、个人信息和文档原文仍可能保留。应将这些文件作为私有运维数据保管，限制访问，在服务外设置保留期限或轮转，并避免提交到仓库或附在公开 issue 中。测试均使用合成输入，运行测试不需要真实请求记录。

## 请求与阶段计时

以下环境变量开关相互独立，默认关闭，设置为 `1` 或 `true` 时启用；它们不是 TOML 字段。

| 变量 | 输出 |
|---|---|
| `MEMORY_REQUEST_TIMING_ENABLED` | 每个 POST `/add`、`/search` 输出一条 `paimem_request_timing` JSON 日志，包含失败请求。 |
| `MEMORY_TIMING_ENABLED` | 输出 `paimem_timing` 明细，覆盖 embedding 排队、配额等待、HTTP、重试，以及检索和重排阶段。 |

```bash
MEMORY_TIMING_ENABLED=0 MEMORY_REQUEST_TIMING_ENABLED=1 \
  bash start_svr.sh --config configs/config.local.toml
```

请求总耗时从进入应用中间件开始，计至最终响应体发送给 ASGI 服务层，包含应用线程池排队和锁等待；不含进入中间件前的代理/服务器排队及客户端网络耗时。未处理异常在离开中间件时记录，尚未开始响应时状态码按 500 记录。日志仅含生成的 trace ID、路由、HTTP 状态和毫秒耗时，不含请求正文、凭据或异常消息。

阶段耗时可能嵌套，不能直接将父子阶段相加。阶段 trace ID 标识 store 操作，与请求日志的 trace ID 独立。远程 embedding 保留 `.encode` 耗时，不输出 store 的 `.lock_wait`；工作线程池排队和配额等待分别看 `embedding.queue_wait`、`embedding.rate_limit_wait`。两类日志均使用 WARNING，以便在默认日志配置下可见；该级别本身不表示请求失败。

客户端命令及指标口径见 [Add/Search 压测](load-testing.zh-CN.md)。


## 两套预设

| 参数 | baseline | default |
|---|---:|---:|
| Dense + BM25 / RRF | 开 | 开 |
| BM25 权重 / RRF k | 0.5 / 60 | 0.5 / 60 |
| Reranker | 关 | 开，400 候选 |
| 上下文重排 | 关 | 开 |
| MemPursuit | 关 | 开，150 候选参数 |
| MemPursuit 上下文向量 | 关 | 开 |
| 五路排名融合 | 关 | 关；预设权重为 Dense 0.35 + reranker 0.65 |
| 融合候选前缀 MP / remaining | — | 100 / 300 |
| 邻居窗口 / seed 数 | 1 / 20 | 1 / 20 |
| pack_span | 1 | 2 |
| SIRA / NEAR / 角色 / 冲突 / 长问题 | 关 | 关 |

`default` 是 PaiMem 的组合策略预设，模型与设备在对应 TOML 分组配置，也可通过 export 覆盖。评测结果取决于所用模型、数据与运行环境。MemPursuit 实际候选池还受 top_k 与上游合法候选数量影响，详见策略文档。

编辑配置中的布尔值可切换策略。也可复制为 `configs/experiment.local.toml`；该后缀默认不提交。环境变量覆盖示例：

```bash
export MEMORY_MEMPURSUIT_ENABLED=false
export MEMORY_RANK_FUSION_ENABLED=false # 依赖 MP，需同步关闭
bash start_svr.sh --config configs/config.local.toml
```

baseline 强制不加载主 reranker，即使当前终端有残留的模型配置。保留 `[reranker] enabled=false` 时不能启用排名融合。

## 可选 LLM 策略

开启 SIRA、角色抽取或冲突核验时，在各自分组填写 model、base_url、api_key 并启用 enabled；仍支持以下环境变量覆盖：

```bash
export MEMORY_SIRA_MODEL='your-model'
export MEMORY_SIRA_API_BASE='https://your-provider.example/v1'
export MEMORY_SIRA_API_KEY='your-key'
```

角色抽取使用 `MEMORY_ROLE_NAME_MODEL/API_BASE/API_KEY`，冲突策略使用 `MEMORY_CONFLICT_MODEL/API_BASE/API_KEY`。连接字段明确为空时，支持共享 `ANSWER_MODEL/ANSWER_API_BASE/ANSWER_API_KEY`，其次回退到 `OPENAI_MODEL/OPENAI_BASE_URL/OPENAI_API_KEY`；非空占位值必须替换。角色抽取还需要 reranker；默认复用主 reranker，独立 `MEMORY_ROLE_NAME_RERANKER_MODEL` 会走本地加载。

## 长问题处理

`[long_question]` 默认关闭。启用后，查询超过 `threshold_chars` 时，LLM 仅接收开头 `head_chars` 和结尾 `tail_chars` 个字符，提取检索查询；完整原问题按常规写入准备流程分块，作为本次请求的临时证据与历史记忆共同参与检索，不写入 SQLite。提取或准备失败时回退到原查询，临时证据也不会进入共享 NEAR 缓存。

全部 9 个参数集中在一个分组：

```toml
[long_question]
enabled = false
model = "gpt-4o-mini"
base_url = "https://your-llm-provider.example/v1"
api_key = "REPLACE_WITH_LONG_QUESTION_API_KEY"
extra_body = '{}'
threshold_chars = 8000
head_chars = 2000
tail_chars = 2000
timeout = 30
```

`extra_body` 是 JSON **对象字符串**，不是 TOML 子表或 JSON 数组。`{}` 或空字符串表示不传额外参数。支持的供应商可使用 `'{"enable_thinking":false}'` 或 `'{"thinking":{"type":"disabled"}}'` 等形式，具体以所用接口为准。长度单位为字符，不是 token；头尾字符数之和不得超过阈值。启用时 `--check` 会校验长度关系、正数超时及 JSON 对象格式。

模型连接和提取参数在启动时生效。只有启动时已配置提取器，才能通过 `/config` 修改运行时开关 `long_question_enabled`；否则需在 TOML 中启用后重启。检索诊断包含生效/回退状态与临时证据数量。

## 运行时配置与管理接口

GET `/config` 查看当前值；POST `/config` 只支持 `ConfigPatch` 声明的字段（可在 `/docs` 查看）。运行时修改不保存回 TOML，重启后重新读取配置。启动配置和运行时开关不能互相替代。

GET/POST `/config` 与 `/clear` 必须提供服务 API key，即使设置了 `ALLOW_UNAUTHENTICATED=true` 也不豁免。使用 `X-Api-Key` 或 `Authorization: Bearer` 请求头传入 `[service].api_key`。

```bash
curl --fail "$MEMORY_BASE_URL/config" -H "X-Api-Key: $MEMORY_API_KEY"
curl --fail "$MEMORY_BASE_URL/config" -H "X-Api-Key: $MEMORY_API_KEY" -H 'Content-Type: application/json' \
  -d '{"rank_fusion_enabled":false,"mempursuit_enabled":false}'
```

## 算法说明

- [MemPursuit](strategies/MEMPURSUIT.zh-CN.md)：残差驱动的固定预算选择。
- [Rank fusion](strategies/RANK_FUSION.zh-CN.md)：候选排名分加权。
- [SIRA](strategies/SIRA.zh-CN.md)：查询侧扩展与 DF 过滤。
- [NEAR](strategies/NEAR.zh-CN.md)：邻域残差评分。
- [Reranker](strategies/RERANKING.zh-CN.md)：本地及 vLLM 接口。

算法文档说明各策略在 PaiMem 中的实现与适配边界；回归测试见 `tests/`。默认配置的效果需在指定模型和数据集上评测。

## 诊断输出目录

`[service].diagnostics_dir`（环境变量 `MEMORY_DIAGNOSTICS_DIR`，默认 `artifacts/diagnostics`）限定 `/search` 的 `debug_output_path` 写入范围。诊断请求必须来自回环地址并提供有效服务密钥；目标必须是该目录内的绝对 `.json` 路径，父目录需事先存在，不能经过符号链接或使用 `..`。同名普通文件会被原子替换，以支持评测重试。目录配置在启动时加载。
