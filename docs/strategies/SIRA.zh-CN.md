# SIRA 查询侧扩展

[English](SIRA.md) | [简体中文](SIRA.zh-CN.md) | [返回 README](../../README.zh-CN.md)

SIRA 通过 LLM 生成查询扩展短语，并使用当前记忆库的文档频率筛选词汇，增强 BM25 的词法匹配。PaiMem 集成其查询侧扩展与过滤机制；默认配置关闭，无需重新入库或重建 embedding。来源版本与许可见[第三方说明](../../THIRD_PARTY_NOTICES.zh-CN.md)。

## 配置

在 `configs/config.local.toml` 中编辑已有 `[sira]` 分组。以下示例开启策略；模型、地址和密钥需填写为实际可用值：

```toml
[sira]
enabled = true
model = "qwen-plus"
base_url = "https://your-llm-provider.example/v1"
api_key = "REPLACE_WITH_SIRA_API_KEY"
expansion_weight = 0.5
max_df_ratio = 0.05
max_n = 4
timeout = 30.0
max_tokens = 1024
```

```bash
bash start_svr.sh --check
bash start_svr.sh
```

| `[sira]` 字段 | 默认配置值 | 说明 |
|---|---:|---|
| `enabled` | `false` | 启用扩展；支持运行时切换 |
| `expansion_weight` | `0.5` | 扩展 BM25 分数权重；0 时不调用 LLM |
| `max_df_ratio` | `0.05` | DF 上限比例；0 时仅检查存在性 |
| `max_n` | `4` | DF 滑动窗口最大长度，1–4 |
| `timeout` | `30.0` | 模型请求超时（秒），不重试 |
| `max_tokens` | `1024` | 单次扩展最大输出 token 数 |

通过启动脚本加载时，已导出的环境变量优先于 TOML。启用策略后，只有明确留空的连接字段才会尝试 `ANSWER_*`、`OPENAI_*` 回退；非空占位符必须替换。建议直接填写完整的 `[sira]` 连接配置，详见[配置说明](../configuration.zh-CN.md)。

### 运行时切换

```bash
curl -X POST http://127.0.0.1:8000/config -H "X-Api-Key: $MEMORY_API_KEY" \
  -H 'Content-Type: application/json' -d '{"sira_enabled":false}'
```

`POST /config` 仅支持修改 `sira_enabled`，其余 SIRA 参数需重启生效。重新启用要求模型客户端已在启动时配置；建议先以有效配置启动，再按需暂停或恢复。启动时 `enabled=false` 会跳过该分组其他 TOML 字段，仅填写其中的连接信息并不能保证支持运行时开启。

`GET /config` 返回生效参数，不返回密钥。运行时开关只影响当前进程，不写回 TOML。管理接口访问方式见[配置说明](../configuration.zh-CN.md)。

### 直接使用 Python API

绕过 TOML 启动器、直接使用 `paimem.api` 时，客户端按以下顺序选择首个非空环境变量：

| 字段 | 优先级 |
|---|---|
| model | `MEMORY_SIRA_MODEL` → `ANSWER_MODEL` → `LLM_MODEL` → `OPENAI_MODEL` |
| key | `MEMORY_SIRA_API_KEY` → `ANSWER_API_KEY` → `API_KEY` → `OPENAI_API_KEY` |
| base | `MEMORY_SIRA_API_BASE` → `ANSWER_API_BASE` → `API_BASE` → `OPENAI_BASE_URL` |

连接信息完整时会创建客户端，以便运行时启用；创建客户端本身不发送请求。上述回退顺序仅描述直接 API 初始化行为。

## 检索行为

```text
原 query → dense ────────────────────────────────────┐
原 query → BM25 + weight × BM25（过滤后的扩展短语）─────┤
                        → RRF → rerank → MemPursuit → 可选排名融合 → 上下文打包
```

- 扩展只读取原始问题。LLM 使用配置的模型、temperature=0.4、seed=42、enable_thinking=false，最多保留10个短语。
- DF 在当前用户、当前角色过滤后的可检索记忆上计算，不跨用户共享。
- 短语只要有一个 1..max_n 的子 n-gram 满足存在且 DF 不超过阈值即可保留。整数阈值 `int(N * ratio)` 为0时不设 DF 上限。
- 采用项目分词器和 unigram BM25 评分，不引入文档扩展、词干化、Rust n-gram 评分或新 reranker；属于查询侧适配。
- 不更改 embedding/rerank/MemPursuit 的查询输入及 options 处理；Add 不调用扩展模型。
- 每次搜索采用不可变 SIRA 配置快照。无全局或跨用户扩展缓存。
- 冲突检索分支复用原问题的一次扩展，避免每个重写 query 各调用一次 LLM；各分支仍使用各自的 query 做 dense、原始 BM25 和 rerank。
- dense 模式、lexical_weight=0、SIRA关闭、扩展权重为0或空语料时不调用扩展模型。
- 网络/超时/解析失败、空扩展、全部过滤时使用原词法分数，不重试。

开启 diagnostics 时 `sira` 包含 enabled、active、status、expansion_ms、proposed_count、kept_count，异常仅记录类型，不记录 API 响应或密钥。状态可为 disabled、not_run（未进入词法分支）、zero_weight、empty_memories、unconfigured、error、no_phrases、all_filtered、ok。

## 参考与评测

来源及固定版本见 [第三方说明](../../THIRD_PARTY_NOTICES.zh-CN.md)，本仓库评测步骤见 [评测复现](../evaluation.zh-CN.md)。
