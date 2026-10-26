# 评测复现

[English](evaluation.md) | [简体中文](evaluation.zh-CN.md) | [返回 README](../README.zh-CN.md)

所有命令从 PaiMem 根目录运行。按 README 完成模型配置并启动独立的评测用记忆服务。数据、模型和生成结果不随本仓库分发。

## 1. 获取外部评测器与数据

```bash
git clone https://github.com/AML-memory/agent-memory-leaderboard.git
# 固定评测器版本
git -C agent-memory-leaderboard checkout 1b8142bfe0f20f1c5218d6b554aa0012de34e504
```

按该版本 [上游说明](https://github.com/AML-memory/agent-memory-leaderboard) 下载数据到 `agent-memory-leaderboard/downloads/benchmarks/`。下载脚本与数据访问条件由上游维护，数据版本必须另外记录。

## 2. 预处理

LoCoMo：

```bash
uv run python scripts/prepare_public_data.py locomo \
  --input agent-memory-leaderboard/downloads/benchmarks/locomo-refined/locomo_refined.json \
  --output data/normalized/locomo_refined.jsonl
uv run python scripts/fix_normalized_answers.py \
  --input data/normalized/locomo_refined.jsonl \
  --output data/normalized/locomo_refined.fixed.jsonl
```

LongMemEval-S（oracle 版换为对应文件名）：

```bash
uv run python scripts/prepare_public_data.py longmemeval \
  --input agent-memory-leaderboard/downloads/benchmarks/longmemeval/longmemeval_s_cleaned.json \
  --output data/normalized/longmemeval_s.jsonl
uv run python scripts/fix_normalized_answers.py \
  --input data/normalized/longmemeval_s.jsonl \
  --output data/normalized/longmemeval_s.fixed.jsonl
```

其他公开数据：

```bash
uv run python scripts/prepare_aml_benchmarks.py personamem
uv run python scripts/prepare_aml_benchmarks.py beam
uv run python scripts/prepare_aml_benchmarks.py clbench
uv run python scripts/fix_normalized_answers.py \
  --input data/normalized/clbench.jsonl --output data/normalized/clbench.fixed.jsonl
```

PersonaMem 的 parquet 读取需要 pyarrow，由 datasets 依赖安装。ScriptMem 若缺少原始脚本正文，不能执行有效的记忆检索评测。

统一格式是一行一个记忆单元：

```json
{"id":"record-1","session_id":"session-1","messages":[{"role":"user","content":"I like tea."}],"questions":[{"id":"q1","question":"What do I like?","answer":"tea","evidence_texts":["I like tea."]}]}
```

小样本应按完整 record 抽样，保留其全部 messages，并保存所选 ID、输入文件哈希与抽样种子。复现实验时使用相同的数据划分，并记录缓存配置。

## 3. Recall

```bash
export MEMORY_BASE_URL='http://127.0.0.1:8000'
export MEMORY_API_KEY='same-key-as-server'
uv run python scripts/local_benchmark.py \
  --input data/normalized/locomo_refined.fixed.jsonl \
  --output artifacts/recall/locomo-k100.jsonl \
  --base-url "$MEMORY_BASE_URL" --run-label locomo-k100-001 --top-k 100
```

逐题 JSONL 写入 output；汇总 JSON 输出到终端，可重定向保存。换成 `--top-k 20` 等可得到不同预算结果，使用独立 run-label 避免混入旧记忆。脚本不会清空全库。

- `evidence_recall_at_k`：有 evidence_texts 的问题中，返回内容覆盖的证据子串比例，再按问题平均。
- `answer_in_context_rate`：答案字符串是否出现的代理指标，不等价于正确作答。
- 无证据标签的 Recall 为 null，不作为零分。片段拼接、标点和说话者前缀可能影响子串指标。
- Recall 不执行回答生成和裁判，但远端 embedding、重排或已启用 LLM 策略仍可能产生调用费用。

CLBench 原始 rubric 不能当作证据。可选 `scripts/prepare_clbench_evidence.py` 仅转换源中可核验的直接标注，输出 audit；该转换仅为符合条件的样本生成证据标签。

## 4. 端到端

先设置回答及裁判服务，二者可使用不同模型：

```bash
export ANSWER_API_BASE='https://your-answer-provider.example/v1'
export ANSWER_MODEL='your-answer-model'
export ANSWER_API_KEY='your-answer-key'
export JUDGE_API_BASE='https://your-judge-provider.example/v1'
export JUDGE_MODEL='your-judge-model'
export JUDGE_API_KEY='your-judge-key'

PYTHONHASHSEED=0 uv run python scripts/run_aml_e2e.py \
  --input data/normalized/locomo_refined.fixed.jsonl \
  --output artifacts/aml-e2e/run-001/locomo_refined \
  --base-url "$MEMORY_BASE_URL" \
  --pipeline agent-memory-leaderboard/data/locomo-refined/pipeline.py \
  --limit 3 --top-k 100 --concurrency 2 \
  --call-timeout 180 --item-timeout 600 --retries 1 \
  --save-search-results
```

使用 `--limit 3` 验证流程，完整评测时移除该参数并使用新的输出目录。顺序是 Add → Search → 上游 answer → 上游 judge/规则评分。`--dry-run` 检查输入与配置但仍需要可用的 pipeline 文件。

| normalized 数据 | 上游 pipeline（相对 agent-memory-leaderboard） |
|---|---|
| locomo_refined | `data/locomo-refined/pipeline.py` |
| longmemeval_s / longmemeval_oracle | `data/longmemeval-s/pipeline.py` |
| personamem_v2 | `data/personamem/pipeline_v2.py` |
| beam_100k / beam_1m | `data/beam/pipeline.py` |
| clbench | `data/clbench/pipeline.py` |

`--concurrency` 同时作为 replay 与模型阶段默认并发，可用 `--replay-concurrency`、`--model-concurrency` 分别指定。保存的 report/answers/judge 等文件用于追溯。

同配置、同输出目录可以断点续跑。改变数据、模型、策略、top_k 或 limit 使用新目录。**续跑不可先清库**，否则已完成的 Add checkpoint 会跳过写入。确需清理时，仅对独立测试实例显式执行 `uv run python scripts/clear_memory.py --base-url "$MEMORY_BASE_URL"`，随后开始新的评测运行。

通过 `--answer-price-in/out` 和 `--judge-price-in/out` 提供供应商实际价格（每百万 token）。缺失用量或价格时，无法计算完整费用。

## 5. 汇总和诊断

```bash
uv run python scripts/summarize_aml_e2e.py summarize artifacts/aml-e2e/run-001
uv run python scripts/summarize_aml_e2e.py serve artifacts/aml-e2e --port 8765
```

浏览器打开 `http://127.0.0.1:8765`。若需全检索通道诊断，可在本机同文件系统服务上增加 `--save-retrieval-diagnostics`；普通远端部署只使用 `--save-search-results`。

使用 `--save-retrieval-diagnostics` 时，先创建 `<output>/search_diagnostics`，再将记忆服务的 `[service].diagnostics_dir` 设置为该目录的绝对路径，然后启动 PaiMem。评测脚本与服务必须共享此目录；脚本会在请求中发送服务密钥。重试时会替换同名诊断 JSON 文件。

### 可选的问题分类

报告页面支持中英文切换。刷新报告时，只要对应版本仍可用，就会保留已选版本。首次对比默认使用列表首项和末项，不要求特定版本命名。

问题类型统计为可选功能。将标签文件放在 PaiMem 仓库的 `data/splits/question_categories.<suffix>.json`。例如，`question_categories.demo.json` 适用于名为 `baseline_demo`、`experiment_demo` 的报告目录。版本名必须以 `_<suffix>` 结尾；多个文件匹配时，优先使用最长后缀。分类文件位置与 NEAR 的 `--data-dir` 参数相互独立。

```json
{
  "questions": [
    {
      "source_dataset": "locomo_refined",
      "item_id": "record-1:q1",
      "type": "temporal",
      "subtype": "relative-time"
    }
  ]
}
```

`source_dataset` 必须与数据集输出目录名一致；`item_id` 必须匹配保存的评测条目 ID（使用 normalized 输入时，通常为 `<record_id>:<question_id>`）。`type` 必填，`subtype` 可选；每个 `(source_dataset, item_id)` 组合必须唯一。示例中的 `temporal` 是自定义标签，并非必须采用的分类体系。

没有匹配文件时，不显示分类统计与分类筛选，整体、数据集和案例视图仍可使用。匹配文件格式错误时会显示警告。未映射的条目归入“未分类”，对比版本标签不一致时显示“分类不一致”。分类题数包含已有标签但尚无结果的条目，平均分仅使用有效评分。修改标签文件后刷新页面即可。

## 6. 结果可复现记录

每轮保存：PaiMem commit、配置文件和外部非敏感覆盖项、模型 revision、硬件、Python/依赖版本、数据哈希、完整命令、上游 pipeline commit、运行输出。不要记录 API key。LLM 输出和裁判可能变化，相同配置不保证逐次相同分数。

PaiMem 提供从数据预处理到结果汇总的完整评测流程。报告分数、排名或性能提升时，应附上可复现的模型配置、样本划分和运行记录。

## 7. 本地测试

```bash
uv sync --group dev
uv run python -m pytest tests -q
```

测试使用临时数据库、固定向量和模拟 HTTP；不需下载模型或填写付费服务密钥。

AML 集成测试需先获取上面的外部评测仓库；未提供时相应测试会跳过。也可设置环境变量 `AML_PIPELINE_ROOT=/absolute/agent-memory-leaderboard/data` 复用已下载的同版本 pipeline。

## 8. NEAR 消融评测

`scripts/run_near_ablation.py` 使用自身固定的 `BASE_SETTINGS` 比较 Hybrid 与 Hybrid + NEAR，独立于记忆服务的 TOML 预设。脚本直接加载本地 embedding 模型和 SQLite 存储，无需启动记忆服务。先按本地 embedding 流程准备模型；以下命令假设 `models/bge-small-en-v1.5` 已可用，使用兼容 GPU 时可将设备改为 `--device cuda:0`。

### 准备开发集和验证集

使用第 2 节生成并修正答案的 LoCoMo-refined、LongMemEval-S 统一格式文件。每行 JSONL 是包含 `id`、`messages`、`questions` 的完整 record；问题包含 `id`、`question`、`answer` 和 `evidence_texts` 证据标签。开发集选择要求两个数据集都有带有效证据标签的问题。按完整 record 划分互不重叠的开发集与验证集；如果同一会话跨多个 record，应按会话分组，避免两侧重叠。

输入目录应包含：

```text
locomo_refined.dev.jsonl
longmemeval_s.dev.jsonl
locomo_refined.validation.jsonl
longmemeval_s.validation.jsonl
```

对于每行代表一个独立记忆单元的数据，以下示例将约 20% 的 record 分配到开发集，其余用于验证。该划分仅为示例，并非官方基准划分；请保存所选 ID 和文件哈希，对比实验使用相同文件。

```bash
uv run python - <<'PYTHON'
import hashlib
import json
from pathlib import Path

source = Path("data/normalized")
target = Path("data/near-splits")
target.mkdir(parents=True, exist_ok=True)
for dataset in ("locomo_refined", "longmemeval_s"):
    rows = [json.loads(line) for line in
            (source / f"{dataset}.fixed.jsonl").read_text().split("\n") if line.strip()]
    ids = [str(row["id"]) for row in rows]
    assert len(rows) >= 2 and len(set(ids)) == len(ids), "Need at least two unique records"
    rows.sort(key=lambda row: hashlib.sha256(
        ("near-20261015:" + str(row["id"])).encode()).hexdigest())
    boundary = min(len(rows) - 1, max(1, len(rows) // 5))
    for phase, subset in (("dev", rows[:boundary]), ("validation", rows[boundary:])):
        (target / f"{dataset}.{phase}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in subset))
PYTHON
```

### 执行检索与端到端评测

```bash
uv run python scripts/run_near_ablation.py --phase dev \
  --data-dir data/near-splits --output artifacts/near/dev \
  --embedding-model models/bge-small-en-v1.5 --device cpu

uv run python scripts/run_near_ablation.py --phase validation \
  --data-dir data/near-splits --output artifacts/near/validation \
  --selection artifacts/near/dev/selection.json \
  --embedding-model models/bge-small-en-v1.5 --device cpu
```

`--data-dir` 默认指向仓库的 `data/splits`；显式传入相对路径时，基于当前工作目录解析。每个检索阶段会在创建输出和加载模型前，检查两个数据集文件是否齐全。数据读取和 manifest 哈希计算使用同一组解析后的输入路径。

开发阶段使用确定性抽样（默认最多 3 个 LoCoMo record、每个 record 8 个问题，以及 16 个 LongMemEval record）比较六组单层 NEAR 配置，生成 `selection.json`。验证阶段固定该选择，评测验证文件中的全部 record 和问题。两个阶段必须使用相同 embedding 模型。两组检索耗时均排除查询编码；token 预算召回率使用本地 embedding tokenizer。这些耗时不代表端到端服务延迟。

输出包含 `manifest.json`、`retrieval-summary.json`，以及各数据集选中的 record 和检索结果。验证阶段还会为每组生成 `official-input.jsonl`。按第 4 节配置 `ANSWER_*`、`JUDGE_*`，并将指定版本的 AML 仓库放在 `agent-memory-leaderboard` 后，运行：

```bash
uv run python scripts/run_near_ablation.py --phase e2e \
  --output artifacts/near/validation
```

`e2e` 阶段读取已保存的验证输入，不读取 `--data-dir`；该阶段会调用回答与裁判服务，可能产生费用。更换输入、模型或实现后使用新的输出目录。固定抽样种子和算法配置保持不变。
