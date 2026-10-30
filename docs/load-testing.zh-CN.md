# Add/Search 压测

[English](load-testing.md) | [简体中文](load-testing.zh-CN.md) | [配置说明](configuration.zh-CN.md)

`scripts/load_test.py` 测量 PaiMem 的 `/add` 与 `/search` 接口，支持同时压测、单接口压测和顺序冒烟测试。这些是完整 API 链路的指标；单独测量 vLLM 请使用[重排容量压测](reranker-benchmark.zh-CN.md)。

## 在请求侧运行

在已有项目 Python 依赖的 PaiMem 仓库目录中运行。请求侧与服务侧可以位于不同机器。将 `PAIMEM_URL` 设置为可访问的 PaiMem 地址；`127.0.0.1` 指向请求侧自己。服务需监听可访问的网卡，端口须对请求侧可达。脚本不启动或配置服务。

```bash
export PAIMEM_URL='http://paimem.example:8000' # 替换为实际服务地址
# 在当前 Shell 中将 MEMORY_API_KEY 设置为服务的 API key。
.venv/bin/python scripts/load_test.py \
  --mode mixed \
  --base-url "$PAIMEM_URL" \
  --add-concurrency 16 \
  --search-concurrency 16 \
  --users 16 \
  --seed-documents 400 \
  --messages-per-add 10 \
  --duration 120 \
  --timeout 180 \
  --output /tmp/paimem-mixed-16.json
```

URL 填写 PaiMem 服务根地址，不带路径、凭据、查询参数或片段。mixed/add/search 模式从 `MEMORY_API_KEY` 读取密钥，也可用 `--api-key-env NAME` 指定其他变量。这些模式复用 HTTP 连接，不读取环境中的代理配置（`trust_env=False`），需要直接可达的服务地址。输出文件不得已存在，其父目录必须存在。添加 `--dry-run` 可只校验参数与输入，不发送请求、不创建文件。

`--mode add` 或 `--mode search` 只对指定接口进行正式测量，但**两种模式仍会预置数据并执行验证搜索**。不指定 `--mode` 时保留原顺序冒烟测试：串行 add 后并发 search，其输出结构及耗时口径与定时模式不同。参数可分别通过 `--mode mixed --help`、`--help` 查看。

## 输入与数据保留

每轮生成独立的用户和请求标识。上述示例为 16 个用户各预置 400 条源文档，共 6,400 条；默认每批 20 条，最多 4 个预置工作协程。每个用户须通过非空搜索验证后才进入测量。预置和验证请求不计入正式指标，但仍消耗服务器资源和供应商额度；它们用于就绪检查，并非单独控制的预热阶段。源文档数量不等于实际存储分块数，也不等于重排候选数。

默认输入为合成的中文项目文本。自定义输入使用 `--input data/load-input.jsonl`，每个非空行应包含非空 `query`，以及至少 `--seed-documents` 条非空文档字符串：

```json
{"query":"何时进行发布评审？","documents":["发布评审安排在周五。","测试将在周四完成。"]}
```

使用这个两条文档的示例时，传入 `--seed-documents 2`。用户轮流使用输入行，每行只取配置数量的前若干文档；add 工作协程循环使用这些文档，每次生成新的请求 ID。报告将文件输入标记为 `jsonl`，默认合成输入标记为 `synthetic`，并包含实际使用输入的哈希。文件输入不一定代表生产流量；比较测试时应记录数据来源、文档和 token 长度分布。

测试数据会保留在数据库中，脚本不调用 clear/delete/config 接口。mixed 和 add 模式在测量期间持续增大语料，每次运行也会新增预置数据。建议使用专用测试数据库，在不同轮次间保持语料规模、输入、策略、模型版本、配额和并发数可比。搜索非空本身不代表检索质量达标。

## 指标口径

- 每个工作协程采用闭环：发送一个请求，等待完成，再在时间窗口内发送下一个。并发 16 不代表每秒 16 个请求；mixed 16/16 最多同时存在 32 个正式测量请求。
- `--duration` 控制正式请求的发起窗口；窗口结束后等待在途请求完成，每个请求受 `--timeout` 限制。`wall_seconds` 包含这段等待，因此 120 秒压测可能运行更久，预置数据另需时间。
- `successful_qps` 为成功数除以包含在途等待的正式测量总时长。add 统计的是 API 调用数，不是文档数；每次正式 add 的消息数由 `messages_per_add` 控制。
- `mean_ms`、`p50_ms`、`p95_ms`、`p99_ms` 只统计成功请求，分位数采用最近秩方法。应结合 `error_rate`、错误计数和 `failed_mean_ms` 阅读；大量超时可能让成功请求延迟看起来偏低。
- 搜索响应会校验结构。正式测量中的空搜索计入 `empty_searches`，但仍视为成功 API 调用。任一正式请求失败都会使报告状态为 `failed`；预置失败为 `seed_failed`，不启动正式压测。
- 客户端不重试请求。超时不代表 add 已回滚，服务端可能继续执行。测量阶段按 Ctrl+C 会取消客户端请求并报告 `cancelled_inflight`，不会取消服务端工作或删除数据。预置阶段中断可能留下部分数据，且不生成完整 JSON 报告。

报告包含聚合指标和生成的标识，不包含输入正文、响应正文或 API key，也不代替服务端诊断日志。

## 对照请求侧与服务侧

服务端启用 `MEMORY_REQUEST_TIMING_ENABLED=1`，可为每次 add/search 输出一条总耗时。阶段定位时可临时启用 `MEMORY_TIMING_ENABLED=1`；计时范围见[配置说明](configuration.zh-CN.md#请求与阶段计时)。服务端日志也会包含预置和验证流量，对比聚合指标前应先选取正式测量区间。

修改单个服务参数时保持输入不变，比较吞吐、延迟分位数、错误和供应商额度等待，再调整请求侧并发，观察服务可持续承载的范围。远程 embedding 工作线程数和 RPM/TPM 在实例内共享；压测脚本的 add/search 并发控制客户端请求，两者不是同一种限制。改变检索策略或候选数量时，还需单独评估质量。
