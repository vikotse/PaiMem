# PaiMem 中的 MemPursuit

[English](MEMPURSUIT.md) | [简体中文](MEMPURSUIT.zh-CN.md) | [返回 README](../../README.zh-CN.md)

MemPursuit 通过查询重构残差选择互补证据。PaiMem 采用其固定预算 NumPy 算法，并支持以同会话上下文构造候选表示。默认配置启用此策略及上下文扩展；baseline 配置关闭两者。

选择算法本身不调用生成式 LLM、不训练模型，也不要求重新入库。关闭 `mempursuit.small2big_enabled` 时复用已存储的文档向量；开启时会在查询阶段重新编码扩展后的候选文本，产生额外 embedding 计算或请求。数值求解在 CPU 上执行。

## 来源与算法

参考 [FengQi-HITSZ/MemPursuit](https://github.com/FengQi-HITSZ/MemPursuit)，固定版本 [`5138dd0e6f6dbf843692526a44b696a4b9f4bcfb`](https://github.com/FengQi-HITSZ/MemPursuit/tree/5138dd0e6f6dbf843692526a44b696a4b9f4bcfb)。`mempursuit.py` 改编自其 standalone `mempursuit/core.py` 与 `mempursuit/simplex.py` 的默认 NumPy、固定预算路径，保留 Apache-2.0 许可，全文见仓库 `third_party/licenses/MemPursuit-Apache-2.0.txt`。

1. 将查询和候选向量转成 float64 并做 L2 归一化。
2. 每轮选取与当前残差内积最大的未选候选。同分时遵循传入候选顺序。
3. 对全部已选候选重新求解非负、和为 1 的组合权重：最小化 `0.5 * a.T @ (G + ridge * I) @ a - b.T @ a`。求解器采用官方 active-set KKT、warm start 和奇异矩阵 `lstsq` 回退。
4. 更新残差，直到达到固定预算。输出保留入选顺序；即使后续权重变为零，已选片段也保留。

本实现采用固定预算选择，不包含自适应停止、Numba 加速及 Mem0 存储集成。求解器使用项目已有的 NumPy 依赖。输入向量为零、含非有限值、维度不匹配或数值求解未收敛时，返回明确错误。

## 配置

在 `configs/config.local.toml` 中编辑已有的 `[mempursuit]` 分组；下列值与 [默认配置](../../configs/default.toml) 一致：

```toml
[mempursuit]
enabled = true
candidates = 150
ridge = 1e-10
small2big_enabled = true
```

| 字段 | 默认配置值 | 说明 | 运行时字段 |
|---|---:|---|---|
| `enabled` | `true` | 启用证据选择 | `mempursuit_enabled` |
| `candidates` | `150` | 优化候选池大小，正整数 | 仅启动时 |
| `ridge` | `1e-10` | 有限、非负的正则项 | 仅启动时 |
| `small2big_enabled` | `true` | 将同会话相邻片段用于候选编码 | `mempursuit_small2big_enabled` |

上下文扩展使用候选在检索序列中前后各一个位置、且属于同一会话的片段；此窗口与最终输出阶段的 `context.result_window` 分别控制。它改变查询时使用的候选向量，不修改已存储的向量。

环境变量可覆盖 TOML，映射见配置文件注释。完成配置后运行：

```bash
bash start_svr.sh --check
bash start_svr.sh
```

运行时可向 `POST /config` 提交以下内容，无需重启或重新入库：

```json
{"mempursuit_enabled": true, "mempursuit_small2big_enabled": true}
```

运行时修改只作用于当前服务进程，不写回 TOML；重启后恢复启动配置。`GET /config` 返回生效参数。每次搜索使用配置快照，冲突检索分支共享该快照。管理接口访问方式见[配置说明](../configuration.zh-CN.md)。

直接使用 Python API 时，`MemPursuitConfig` 的默认值为 `enabled=false`、`candidates=200`、`ridge=1e-10`；与上述启动预设不同。`configure_store()` 的显式参数优先于环境变量。

## 接入顺序与候选预算

```text
Dense / NEAR + BM25 → RRF → 可选 reranker → rerank 阈值
  → MemPursuit 入选前缀 + 合法候选尾部 → 可选排名融合 → 窗口扩展 / 打包
```

优化候选池取阈值过滤后上游顺序的前 `max(top_k, mempursuit_candidates)` 项，数量不超过现存合法候选。若启用了 reranker，也受其已有候选范围约束。这里的 `top_k` 指该次内部检索请求的预算，冲突策略可能为配额分配多召回一些候选。

固定预算为 `min(top_k, pool_count)`。算法只求解所需入选前缀；每轮对当前已选集合重新拟合权重，因此诊断系数对应当前预算下的解。

为配合 PaiMem 的打包去重，在入选前缀之后附上所有未选合法候选的原始顺序，包含优化池以外的候选。它们仅供后处理回填；不会增加 Gram 矩阵或 QP 的规模。这是相对官方 adapter 的适配差异：上游不补未选尾部。被 rerank 阈值拒绝的片段不会重新成为 seed，但仍可能作为合法 seed 的相邻上下文出现，遵循配置的窗口规则。

窗口扩展可能重新引入相似邻居，因此该策略优化的是 seed 选择，不能将最终扩展集合视为论文中的原始选择结果。关闭开关时跳过归一化、Gram 构建和 QP，沿用原检索顺序、分数和模型调用。

## 策略组合

- **NEAR**：可独立同时开启。NEAR 影响候选池与上游顺序，MemPursuit 使用查询向量与候选向量；候选向量由 `small2big_enabled` 决定是复用存储向量还是重新编码上下文，不使用 NEAR 残差向量。
- **Reranker**：可有可无。先过滤其分数阈值，再执行 MemPursuit；原始评分不直接进入 QP。
- **冲突策略**：各路使用自己的查询向量独立选择，随后按配置分配配额、合并队列并附加冲突说明。
- **Options**：不增加查询向量。`enhanced` 使用已有的第一路原始 query 向量；其他模式使用已有的追加 options 后的 query 向量。选项视图仍可影响上游召回池。

## 分数与诊断

响应 `MemoryHit.score` 保留原 dense / fusion / reranker 分数，可能不随新的返回顺序递减。不能按这个原分数重新排序，否则会抹掉 MemPursuit 入选顺序。

开启检索诊断时，顶层 `mempursuit` 记录配置、是否执行、池大小、固定预算、入选数、尾部候选数、停止原因和查询视图。每条候选的 `channels.mempursuit` 记录：

- `rank`：入选顺序，未选为 `null`。
- `score`：按官方 adapter 方式计算的顺序分 `(selected_count - rank + 1) / selected_count`，仅供解释顺序，不是相关性概率。
- `coefficient`：当前入选前缀最终拟合系数。
- `selection_score`：被选中时的残差内积；同时记录当步残差与误差下降。
- `pool_rank`：优化候选池中的位置，池外为 `null`。
- `backfill_candidate`：是否为合法未选尾部候选。是否实际输出仍以`returned_rank` / `returned_ranks` 为准。

原始 dense、fusion、NEAR 和 reranker 诊断分数保留。

## 参考与评测

来源及固定版本见 [第三方说明](../../THIRD_PARTY_NOTICES.zh-CN.md)，本仓库评测步骤见 [评测复现](../evaluation.zh-CN.md)。
