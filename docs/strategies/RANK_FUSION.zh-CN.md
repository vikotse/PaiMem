# 多通道排名融合

[English](RANK_FUSION.md) | [简体中文](RANK_FUSION.zh-CN.md) | [返回 README](../../README.zh-CN.md)

默认配置关闭此策略。开启后在 MemPursuit 选择结束、邻居扩展与打包之前，对两类候选的指定前缀合并排序：原文向量、BM25、RRF、上下文重排、MP选择顺序的五路排名分分别乘对应权重后求和。无需重新入库，不新增模型调用。

| `[rank_fusion]` 字段 | 默认配置值 | 说明 |
|---|---:|---|
| `enabled` | `false` | 是否启用 |
| `dense_weight` | `0.35` | 原文向量排名分权重 |
| `rerank_weight` | `0.65` | 重排模型排名分权重 |
| `bm25_weight` | `0` | BM25正分候选的排名分权重 |
| `rrf_weight` | `0` | 初次RRF完整排名分权重 |
| `mp_weight` | `0` | MP完整选择顺序排名分权重 |
| `mp_limit` | `100` | 按MP选择顺序取前多少块参与加权 |
| `remaining_limit` | `300` | 从未被MP选中的合法候选中，按重排顺序取前多少块参与加权 |

数量均支持 `-1`（全部）、`0`（不取该类）；超出现存数量时取实际数量。两项均为0时保持原队列并诊断为 `zero_limits`。这两个参数不改变MP本身的候选池或选择预算，也不是最终返回配额。

五路权重统一归一化。BM25、RRF、MP权重默认均为0，默认使用35%向量＋65%重排。BM25只有正分候选有排名，缺失得0分；RRF使用截取重排候选前的完整排名；MP使用数量截取前的完整选择顺序，未选中块得0分。若当前检索模式未执行BM25/RRF，这两路视为缺失、得0分，缺失通道的权重不重新分配。

权重必须有限、非负且总和大于0，自动归一化，因此35/65与0.35/0.65等价。每阶段第r名的排名分为 `1 - (r-1)/max(N-1,1)`，N使用该阶段完整排名长度，不使用截取后的长度。原文向量排名取原始查询视图的文档向量相似度，不取NEAR或多选项混合后的排名。重排排名在阈值过滤前保存，但被阈值拒绝的候选不会重新进入队列。

参与加权的候选按分数降序置前；同分稳定保留“MP选择顺序＋其余候选的重排顺序”。未参与者保持原队列相对顺序接在后面，用于扩展去重后的回填。最终输出的窗口、种子数、打包方式与 `top_k` 可独立配置。

## 配置

在 `configs/config.local.toml` 中编辑已有 `[rank_fusion]` 分组。以下示例开启融合，其余字段与 [默认配置](../../configs/default.toml) 一致：

```toml
[rank_fusion]
enabled = true
dense_weight = 0.35
rerank_weight = 0.65
bm25_weight = 0
rrf_weight = 0
mp_weight = 0
mp_limit = 100
remaining_limit = 300
```

同时确保 `mempursuit.enabled=true`、`reranker.enabled=true`。环境变量可覆盖 TOML，变量映射见配置注释。启动器会拒绝缺少上述依赖的融合配置。

```bash
bash start_svr.sh --check
bash start_svr.sh
```

### 运行时修改

运行时字段名采用 `rank_fusion_` 前缀。修改只作用于当前进程，不写回 TOML；重启后恢复启动配置。

运行时向 `POST /config` 提交以下 JSON，管理端点访问方式见[配置说明](../configuration.zh-CN.md)：

```json
{
  "rank_fusion_enabled": true,
  "rank_fusion_dense_weight": 0.35,
  "rank_fusion_rerank_weight": 0.65,
  "rank_fusion_bm25_weight": 0,
  "rank_fusion_rrf_weight": 0,
  "rank_fusion_mp_weight": 0,
  "rank_fusion_mp_limit": 100,
  "rank_fusion_remaining_limit": 300
}
```

例如切换为重排60%＋RRF20%＋MP20%，提交 `{"rank_fusion_dense_weight":0,"rank_fusion_rerank_weight":0.6,"rank_fusion_bm25_weight":0,"rank_fusion_rrf_weight":0.2,"rank_fusion_mp_weight":0.2}`；五路权重总和必须大于0。

或通过CLI：

```bash
.venv/bin/python -m paimem.cli config --base-url http://127.0.0.1:8000 --set rank_fusion_enabled=true
.venv/bin/python -m paimem.cli config --base-url http://127.0.0.1:8000 --set rank_fusion_dense_weight=0.35 --set rank_fusion_rerank_weight=0.65
.venv/bin/python -m paimem.cli config --base-url http://127.0.0.1:8000 --set rank_fusion_mp_limit=50 --set rank_fusion_remaining_limit=150
```

`GET /config` 返回全部配置。非法更新整体拒绝，不会部分生效。同一Search及其冲突检索分支使用同一份不可变加权配置快照。关闭仅需 `{"rank_fusion_enabled":false}`。

## 与其他阶段的关系

[默认配置](../../configs/default.toml) 启用 hybrid 检索、上下文重排和上下文 MemPursuit，但关闭排名融合。重排候选数为 400，`reranker.max_length=1024`，`mempursuit.candidates=150`。启用融合后，`mp_limit` 和 `remaining_limit` 只决定参与融合的候选前缀，不改变上述配置。

将两个 limit 都设为 `-1` 可让全部合法候选参与融合。最终输出仍由 `top_k`、窗口扩展和打包规则决定。若运行时关闭 MemPursuit 或 reranker，融合阶段将跳过执行，诊断分别为 `mempursuit_disabled` 或 `reranker_disabled`。

直接使用 Python API 时，可通过 `configure_store(rank_fusion_enabled=True, ...)` 设置参数，显式参数优先于环境变量。

## 诊断与评测

检索诊断 `rank_fusion` 记录配置、active、reason、实际参与MP数、剩余数、加权总数、回填数。候选 `channels.rank_fusion` 记录是否参与、五路排名分和融合分。公开 `MemoryHit.score` 保留模型评分；调用方应保留返回顺序，不能再按该字段排序。

评测方法见[评测复现](../evaluation.zh-CN.md)。对比不同融合设置时，应固定候选池、模型、数据和其他检索参数。
