#!/usr/bin/env bash
# Shared script helpers; configuration is TOML data, never sourced shell code.
# 公共脚本函数；配置统一使用 TOML，不执行配置内容。
paimem_fail() { printf 'PaiMem: %s\n' "$*" >&2; exit 1; }

paimem_select_config() {
  if [[ -z "${PAIMEM_CONFIG+x}" ]]; then
    if [[ -f configs/config.local.toml ]]; then PAIMEM_CONFIG=configs/config.local.toml
    else PAIMEM_CONFIG=configs/default.toml; fi
  fi
  [[ -f "$PAIMEM_CONFIG" ]] || paimem_fail 'Selected TOML does not exist / 指定 TOML 配置不存在'
  export PAIMEM_CONFIG
}

paimem_load_config() {
  local config_python config_exports
  if [[ -x .venv/bin/python ]]; then config_python=.venv/bin/python
  elif command -v python3 >/dev/null 2>&1 && python3 -c 'import tomllib' 2>/dev/null; then config_python=$(command -v python3)
  else
    if ! command -v uv >/dev/null 2>&1 && [[ -x "${HOME}/.local/bin/uv" ]]; then export PATH="${HOME}/.local/bin:$PATH"; fi
    command -v uv >/dev/null 2>&1 || paimem_fail 'Run setup_env.sh first / 请先准备环境'
    config_python=$(uv python find 3.11 --no-python-downloads) || paimem_fail 'Run setup_env.sh first / 请先准备环境'
  fi
  # Capture only trusted, shlex-quoted exports. Never print credentials.
  # 仅执行加载器生成并转义的 export，不输出密钥。
  config_exports=$("$config_python" src/paimem/configuration.py --config "$PAIMEM_CONFIG" --scope "$1" --shell) || return
  eval "$config_exports"
}

paimem_require() {
  local name
  for name in "$@"; do
    [[ -n "${!name-}" && "${!name-}" =~ [^[:space:]] ]] ||
      paimem_fail "Set $name in the selected TOML (see its annotated export name), or export it / 请填写或导出该变量"
  done
}

paimem_uv() {
  if command -v uv >/dev/null 2>&1; then return; fi
  # Also find a prior standalone install without changing shell startup files.
  if [[ -x "${HOME}/.local/bin/uv" ]]; then
    export PATH="${HOME}/.local/bin:$PATH"
    return
  fi
  command -v curl >/dev/null 2>&1 || paimem_fail 'Install curl first / 请先安装 curl'
  printf 'Installing uv / 安装 uv\n'
  curl --fail --silent --show-error --location https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
  export PATH="${HOME}/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || paimem_fail 'uv was not found after installation / 安装后未找到 uv'
}

paimem_gpu_host() {
  [[ "$(uname -s)" == Linux ]] || paimem_fail 'Local vLLM requires Linux with a compatible NVIDIA GPU; use VLLM_DEPLOYMENT=remote on a client / 本机 vLLM 需 Linux NVIDIA GPU，客户端请选 remote'
  command -v nvidia-smi >/dev/null 2>&1 || paimem_fail 'Install a compatible NVIDIA driver first / 请先安装兼容的 NVIDIA 驱动'
  nvidia-smi -L >/dev/null || paimem_fail 'NVIDIA GPU is unavailable / NVIDIA GPU 不可用'
}

paimem_download_model() {
  local repo="$1" destination="$2" revision="$3"
  if [[ -z "$repo" ]]; then
    [[ -f "$destination/config.json" ]] || paimem_fail 'Existing model directory has no config.json; set the corresponding REPO_ID to download / 已有模型目录缺少 config.json，请填写对应 REPO_ID 下载'
    return
  fi
  local -a args=(tool run --from huggingface_hub hf download "$repo" --local-dir "$destination")
  [[ -z "$revision" ]] || args+=(--revision "$revision")
  uv "${args[@]}"
}
