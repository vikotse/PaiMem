#!/usr/bin/env bash
# Model, service and strategy settings all come from ONE TOML.
# 模型、服务与策略参数统一读取一份 TOML；--config 优先于 PAIMEM_CONFIG。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
[[ -x .venv/bin/python ]] || { echo 'Run bash scripts/setup_env.sh first / 请先准备环境' >&2; exit 1; }
exec .venv/bin/python -m paimem.launcher "$@"
