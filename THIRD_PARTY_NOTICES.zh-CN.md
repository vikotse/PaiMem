# 第三方来源与许可说明

[English](THIRD_PARTY_NOTICES.md) | [简体中文](THIRD_PARTY_NOTICES.zh-CN.md) | [返回 README](README.zh-CN.md)

第三方组件适用各自的许可；PaiMem 的 MIT 许可不替代这些许可。

- **[InvMem / vanilla-rag-memory](https://github.com/wenxiaof345-ctrl/vanilla-rag-memory)**：PaiMem 基于该 RAG 基线构建。
- **[MemPursuit](https://github.com/FengQi-HITSZ/MemPursuit/tree/5138dd0e6f6dbf843692526a44b696a4b9f4bcfb)**：版本 `5138dd0e6f6dbf843692526a44b696a4b9f4bcfb`。`src/paimem/mempursuit.py` 中的选择器与 simplex 求解器改编自该工作，适用 [Apache-2.0 许可](third_party/licenses/MemPursuit-Apache-2.0.txt)。文件头保留来源信息；详见[实现说明](docs/strategies/MEMPURSUIT.zh-CN.md)。
- **[SIRA](https://github.com/facebookresearch/sira/tree/57f9cbbe5d0a00c91a60f14f3532c8268edd6332)**：版本 `57f9cbbe5d0a00c91a60f14f3532c8268edd6332`，适用 [MIT 许可](third_party/licenses/SIRA-MIT.txt)。`src/paimem/sira.py` 适配查询扩展、解析、DF 过滤及加权词法增强。详见[论文](https://arxiv.org/abs/2605.06647)与[实现范围](docs/strategies/SIRA.zh-CN.md)。
- **[NEAR](https://arxiv.org/abs/2610.13314)**：基于公开方法的 NumPy 实现，聚类与评分选择见[实现文档](docs/strategies/NEAR.zh-CN.md)。
- **[MemReranker](https://huggingface.co/IAAR-Shanghai/MemReranker-4B)**（[论文](https://arxiv.org/abs/2605.06132)）：可选外部模型，仓库不包含权重。
- **[AML](https://github.com/AML-memory/agent-memory-leaderboard)**：评测版本 `1b8142bfe0f20f1c5218d6b554aa0012de34e504`。评测适配器加载该外部仓库；本仓库不包含上游 pipeline 和数据集，它们适用各自许可。

模型文件、数据集和可选 vLLM 环境单独获取，适用各自的许可。
