# Reranker 性能评测

[English](reranker-benchmark.md) | [简体中文](reranker-benchmark.zh-CN.md) | [返回 README](../README.zh-CN.md)

GPU 部署 MemReranker 时，vLLM 支持独立管理模型服务和请求批处理。本文介绍本地 CrossEncoder 与 vLLM `/score` 部署的可复现对比方法。评估后端加速效果时，应保持模型、硬件、输入及评分配置一致。

## 可复现的对照方法

准备同一 `rerank-input.jsonl`，每行包含 `query` 和至少 200 条 `documents` 字符串。设置各后端实际模型身份，先运行 `--dry-run` 校验参数。

```bash
export LOCAL_RERANKER_MODEL='/absolute/models/compatible-cross-encoder'
export MEMORY_RERANKER_MODEL='served-reranker-name'
export MEMORY_RERANKER_BASE_URL='http://127.0.0.1:8089'
uv run python scripts/benchmark_rerankers.py \
  --backends cross-encoder vllm \
  --models "$LOCAL_RERANKER_MODEL" "$MEMORY_RERANKER_MODEL" \
  --base-urls - "$MEMORY_RERANKER_BASE_URL" \
  --device cuda:0 --batch-sizes 16 --max-lengths 512 \
  --input rerank-input.jsonl --queries 3 --candidates 200 \
  --warmup 2 --repeats 10 --pause-before-model \
  --api-key-env MEMORY_RERANKER_API_KEY \
  --output artifacts/reranker-comparison
```

无鉴权时省略 `--api-key-env`。本地和 vLLM 使用同一 GPU 时，按提示停掉另一端以避免共享显存/计算干扰；脚本本身不启动或停止服务。

输出 `.json` 和 `.csv`，包含每配置耗时与可用的内存指标。该脚本测量串行整批调用；`候选数 / 平均耗时` 仅为该负载的处理率，不等于并发服务极限。HTTP 计时包含网络、排队和序列化，本地计时包含分词、批处理和分数回传，均排除模型加载与预热。

比较时记录 GPU、驱动、软件版本、模型 revision、dtype、模板、候选数、token 长度、批大小。该评测脚本的 `--max-lengths` 只控制本地路径；远端必须自行配置并确认输入不超限。若两端模型或评分模板不同，只能称为“部署方案耗时对比”，不能将差异归因于 vLLM。BGE 与 MemReranker 的跨模型对比无法单独衡量后端加速效果。

## vLLM 并发容量测试

`scripts/benchmark_vllm_capacity.py` 用于测试 `/score` 服务容量，支持已有服务和通过 `--launch-config` 启动的本地 Linux 服务。先运行 `--help` 查看参数，并用 `--dry-run` 检查测试矩阵。

脚本输出 JSON/CSV 报告，按错误率、延迟和样本量约束筛选配置。测试采用闭环请求并计入排空时间，结果代表重排服务容量，不等同于 PaiMem `/search` 端到端性能；查询内批次并发大于 1 的配置需要调整现有客户端才能采用。
