# NEAR 邻域残差检索

[English](NEAR.md) | [简体中文](NEAR.zh-CN.md) | [返回 README](../../README.zh-CN.md)

本实现参考 [NEAR, arXiv:2610.13314v1](https://arxiv.org/html/2610.13314v1) 的公式 1–4 和附录 C。
NEAR 利用邻域参考方向构造残差表示，增强相似记忆之间的区分能力，并将增强分数用于 Dense 检索通道。默认配置关闭此策略。

## 配置

在 `configs/config.local.toml` 中编辑已有分组。以下示例开启 NEAR，其余数值保持 [默认配置](../../configs/default.toml)：

```toml
[retrieval]
mode = "hybrid"

[near]
enabled = true
layers = 1
weight = 1.0
normalize = false
neighbors = 50
anchors = 16
routing = 4
beta = 10.0
cache_max_users = 4
cache_max_mb = 256.0
```

```bash
bash start_svr.sh --check
bash start_svr.sh
```

无需重新入库或重建 embedding。NEAR 参数仅在启动时读取；修改后需重启服务。`GET /config` 展示实际配置，`POST /config` 不支持修改 NEAR 参数。环境变量覆盖方式见[配置说明](../configuration.zh-CN.md)。

| `[near]` 字段 | 默认配置值 | 说明 |
|---|---:|---|
| `enabled` | `false` | 是否增强 Dense 分数 |
| `layers` | `1` | 残差层数，正整数 |
| `weight` | `1.0` | 总残差权重 α，有限且非负；0 时绕过 NEAR |
| `normalize` | `false` | 是否归一化用于匹配的残差 |
| `neighbors` | `50` | 每条记忆的语义近邻数，不包含自身 |
| `anchors` | `16` | 查询参考表最大条数 |
| `routing` | `4` | 每个查询使用的参考表条数 |
| `beta` | `10.0` | 路由 softmax 的逆温度，有限且为正 |
| `cache_max_users` | `4` | 派生缓存的最大用户数 |
| `cache_max_mb` | `256.0` | 缓存数组预算（MiB），不含构建临时内存 |

直接使用 Python API 时，可通过 `configure_store(near_enabled=True, ...)` 设置参数；显式参数优先于环境变量。

## 算法与边界

每层从当前归一化记忆状态找近邻，取其平均单位方向为参考。去除该参考方向的投影
形成残差，下一层使用归一化残差；匹配视图由 `normalize` 决定。球面 K-means 将输入
状态聚类，保存 anchor 及簇内记忆参考方向的平均单位向量。查询通过最相似 anchor
的 softmax 加权参考形成自己的残差。

分数为 `原始内积/τ0 + α/L × Σ(各层残差内积/τ层)`。τ 按附录 C 计算：
`sqrt(mean_i Var_j(z_i · z_j))`，包括对角线，最小值 1e-6。通过二阶矩与协方差
计算，不创建全量相似度矩阵。增强分数替代 Dense 通道，随后沿用选项视图融合、
BM25/RRF、可选 reranker、阈值及上下文打包。

与论文的实现差异：

- 使用 NumPy 精确评分，无 HNSW、量化和拼接索引；分数等价于论文的拼接内积。
- 球面 K-means 使用固定 seed=42 的 farthest-first 初始化，最多 50 轮；空簇被移除。
  上述初始化及收敛设置为本实现的具体选择。
- 范数不超过 1e-7 的残差视为零，不放大数值噪声。空库返回空；单条记忆沿用基础匹配。
- 近邻来自当前用户实际参与检索的记忆集合；启用角色过滤时只在过滤后的集合构建。
  来自同一来源的重叠片段也可成为彼此的近邻。
- 不改写原文、不重新生成文档 embedding、不调用生成式 LLM，也不改数据库 schema。

用户缓存键校验该快照所有 ID 与向量的哈希。Add 成功后失效该用户缓存，Clear
清空缓存；其他进程写入或角色过滤变化也会因快照不同而重建。查询持有已读取的
不可变快照，不会在多路冲突检索中混用新旧记忆。LRU 按用户数和总数组字节数淘汰；
超过单缓存预算的 bank 仍可评分，但不驻留。构建在缓存锁内串行。

首次 Search / 记忆更新后 Search 承担构建费用。近邻构建计算量为 O(L·N²·d)，
相似度按块生成，避免 N² 常驻内存；缓存上限不等于进程峰值内存上限。该实现采用全量重建方式；
大规模记忆库或高频写入场景应评估重建延迟与峰值内存。

离线诊断保留 `channels.dense` 的原始分数，新增 `channels.near`；
`dense_blended` 和 `fusion` 反映实际后续评分。`near` 元数据记录参数、是否命中缓存、
bank 数组大小、构建耗时和各层尺度。正式 `/search` 响应字段不变。

## 参考与评测

来源及固定版本见 [第三方说明](../../THIRD_PARTY_NOTICES.zh-CN.md)，本仓库评测步骤见 [评测复现](../evaluation.zh-CN.md)。
