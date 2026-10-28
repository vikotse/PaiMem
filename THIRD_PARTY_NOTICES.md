# Third-party sources and notices

[English](THIRD_PARTY_NOTICES.md) | [简体中文](THIRD_PARTY_NOTICES.zh-CN.md) | [README](README.md)

Third-party components retain their applicable licenses; PaiMem's MIT license does not replace them.

- **[InvMem / vanilla-rag-memory](https://github.com/wenxiaof345-ctrl/vanilla-rag-memory)**: RAG baseline on which PaiMem is built.
- **[MemPursuit](https://github.com/FengQi-HITSZ/MemPursuit/tree/5138dd0e6f6dbf843692526a44b696a4b9f4bcfb)**: revision `5138dd0e6f6dbf843692526a44b696a4b9f4bcfb`. The selector and simplex solver in `src/paimem/mempursuit.py` are adapted from this work under [Apache-2.0](third_party/licenses/MemPursuit-Apache-2.0.txt). File headers retain provenance; see [implementation details](docs/strategies/MEMPURSUIT.md).
- **[SIRA](https://github.com/facebookresearch/sira/tree/57f9cbbe5d0a00c91a60f14f3532c8268edd6332)**: revision `57f9cbbe5d0a00c91a60f14f3532c8268edd6332`, [MIT license](third_party/licenses/SIRA-MIT.txt). Query expansion, parsing, DF filtering, and weighted lexical augmentation are adapted in `src/paimem/sira.py`. See the [paper](https://arxiv.org/abs/2605.06647) and [implementation scope](docs/strategies/SIRA.md).
- **[NEAR](https://arxiv.org/abs/2610.13314)**: NumPy implementation based on the published method. Clustering and scoring choices are described in the [implementation documentation](docs/strategies/NEAR.md).
- **[vLLM](https://github.com/vllm-project/vllm/tree/v0.32.0)**: the unmodified Qwen3 reranker template at `third_party/vllm/qwen3_reranker.jinja` is bundled from v0.32.0 under [Apache-2.0](third_party/licenses/vLLM-Apache-2.0.txt). Source URL and checksum are recorded in `third_party/vllm/SOURCE.txt`. The default setup copies this file locally.
- **[MemReranker](https://huggingface.co/IAAR-Shanghai/MemReranker-4B)** ([paper](https://arxiv.org/abs/2605.06132)): optional external model; weights are not bundled.
- **[AML](https://github.com/AML-memory/agent-memory-leaderboard)**: evaluation revision `1b8142bfe0f20f1c5218d6b554aa0012de34e504`. Evaluation adapters load this external repository. Upstream pipelines and datasets are not bundled and retain their own licenses.

Model files, datasets, and optional vLLM installations are obtained separately and retain their applicable licenses.
