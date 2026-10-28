#!/usr/bin/env bash
# Install the service dependencies and prepare the selected model backends.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/config.sh

# Shared workflow state, set by argument parsing and backend selection.
only=all
prepare_vllm=false

parse_arguments() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --only)
        [[ $# -ge 2 ]] || paimem_fail '--only needs all, service or vllm'
        only="$2"
        shift 2
        ;;
      --config)
        [[ $# -ge 2 ]] || paimem_fail '--config needs a TOML path'
        PAIMEM_CONFIG="$2"
        shift 2
        ;;
      -h|--help)
        echo 'Usage: bash scripts/setup_env.sh [--only all|service|vllm] [--config PATH]'
        exit 0
        ;;
      *)
        paimem_fail "Unknown option: $1"
        ;;
    esac
  done

  case "$only" in
    all|service|vllm) ;;
    *) paimem_fail '--only must be all, service or vllm' ;;
  esac
}

load_configuration() {
  paimem_select_config
  paimem_uv

  # Bootstrap TOML parsing before project dependencies are available.
  uv python install 3.11
  paimem_load_config bootstrap

  # Validate only the components needed by the requested setup scope.
  case "$only" in
    vllm) paimem_load_config vllm ;;
    service) paimem_load_config service ;;
    all) paimem_load_config setup ;;
  esac
}

install_service_environment() {
  [[ -f "$PAIMEM_CONFIG" ]] || paimem_fail 'Strategy TOML does not exist / 策略配置不存在'

  uv python install "${PAIMEM_PYTHON:-3.11}"
  UV_PROJECT_ENVIRONMENT="$PWD/.venv" uv sync --python "${PAIMEM_PYTHON:-3.11}" --frozen
  uv pip check --python "$PWD/.venv/bin/python"

  # Use the launcher's validation without loading models or calling APIs.
  .venv/bin/python -m paimem.launcher --config "$PAIMEM_CONFIG" --check
}

prepare_embedding() {
  if [[ "$EMBEDDING_BACKEND" != local ]]; then
    echo 'Remote embedding: no model download / 远程 embedding：跳过模型下载'
    return
  fi

  paimem_require EMBEDDING_MODEL
  local embedding_repo="${EMBEDDING_REPO_ID:-}"

  # An empty repository tells the helper to validate an existing model directory.
  if [[ "${EMBEDDING_DOWNLOAD:-true}" != true ]]; then
    embedding_repo=""
  fi
  paimem_download_model "$embedding_repo" "$EMBEDDING_MODEL" "${EMBEDDING_REVISION:-}"
}

prepare_reranker() {
  if [[ "$PAIMEM_RERANKER_ENABLED" != true ]]; then
    echo 'Profile disables reranker: skipping its download and installation / 配置已关闭重排，跳过下载和安装'
    return
  fi

  if [[ "$MEMORY_RERANKER_BACKEND" == cross-encoder ]]; then
    paimem_require MEMORY_RERANKER_MODEL
    local reranker_repo="${RERANKER_REPO_ID:-}"
    if [[ "${RERANKER_DOWNLOAD:-true}" != true ]]; then
      reranker_repo=""
    fi
    paimem_download_model "$reranker_repo" "$MEMORY_RERANKER_MODEL" "${RERANKER_REVISION:-}"
  elif [[ "$only" == all && "$VLLM_DEPLOYMENT" == local ]]; then
    # Install the local vLLM backend after the service environment is ready.
    prepare_vllm=true
  else
    echo 'Using separately prepared vLLM; skipping local reranker download and vLLM install / 使用独立 vLLM，跳过本机下载和安装'
  fi
}

install_vllm_environment() {
  local vllm_env="${VLLM_ENV:-.venv-vllm}"
  local service_env="$PWD/.venv"

  # Resolve symlinks so vLLM cannot overwrite the service's dependencies.
  mkdir -p "$vllm_env"
  vllm_env=$(cd "$vllm_env" && pwd -P)
  if [[ -d "$service_env" ]]; then
    service_env=$(cd "$service_env" && pwd -P)
  fi
  [[ "$vllm_env" != "$service_env" ]] || paimem_fail 'VLLM_ENV must be separate from .venv / vLLM 必须使用独立环境'

  uv python install "${VLLM_PYTHON:-3.12}"
  if [[ ! -x "$vllm_env/bin/python" ]]; then
    uv venv --python "${VLLM_PYTHON:-3.12}" "$vllm_env"
  fi

  uv pip install --python "$vllm_env/bin/python" "vllm==${VLLM_VERSION:-0.32.0}"
  uv pip check --python "$vllm_env/bin/python"
}

prepare_vllm_model() {
  local reranker_repo="${RERANKER_REPO_ID:-}"
  if [[ "${RERANKER_DOWNLOAD:-true}" != true ]]; then
    reranker_repo=""
  fi
  paimem_download_model "$reranker_repo" "$RERANKER_MODEL_PATH" "${RERANKER_REVISION:-}"
}

prepare_chat_template() (
  # A subshell keeps the temporary-file cleanup trap scoped to this download.
  if [[ -s "$RERANKER_CHAT_TEMPLATE" ]]; then
    return
  fi

  paimem_require RERANKER_CHAT_TEMPLATE_URL
  mkdir -p "$(dirname "$RERANKER_CHAT_TEMPLATE")"

  # Match the exact pinned source, so custom URLs never receive a different template.
  bundled_template='third_party/vllm/qwen3_reranker.jinja'
  bundled_url='https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/examples/pooling/score/template/qwen3_reranker.jinja'
  # Publish only a complete, nonempty copy/download.
  template_tmp=$(mktemp "${RERANKER_CHAT_TEMPLATE}.XXXXXX")
  trap 'rm -f "$template_tmp"' EXIT
  if [[ "$RERANKER_CHAT_TEMPLATE_URL" == "$bundled_url" ]]; then
    [[ -s "$bundled_template" ]] || paimem_fail 'Bundled template is missing; sync third_party/vllm from the repository / 内置模板缺失，请同步仓库的 third_party/vllm 目录'
    printf 'Using bundled reranker chat template / 使用仓库内置重排评分模板\n'
    cp "$bundled_template" "$template_tmp"
  else
    command -v curl >/dev/null 2>&1 || paimem_fail 'Install curl first / 请先安装 curl'
    printf 'Downloading custom reranker chat template / 正在下载自定义重排评分模板\n'
    # Include TLS/reset failures in retries; keep each attempt bounded.
    if ! curl --fail --silent --show-error --location \
      --retry 3 --retry-all-errors --connect-timeout 15 --max-time 120 \
      "$RERANKER_CHAT_TEMPLATE_URL" -o "$template_tmp"; then
      printf 'Template destination / 模板目标路径: %s\n' "$RERANKER_CHAT_TEMPLATE" >&2
      paimem_fail 'Chat template download failed. Dependencies and model preparation already completed. Retry setup, or download the configured chat_template_url on another machine and copy it to the path above, then rerun setup. / 评分模板下载失败，依赖和模型准备已完成。可重试，或在其他机器下载配置中的 chat_template_url 并上传至上述路径，再重新运行。'
    fi
  fi
  [[ -s "$template_tmp" ]] || paimem_fail 'Prepared template is empty / 准备的模板为空'
  mv "$template_tmp" "$RERANKER_CHAT_TEMPLATE"
  trap - EXIT
)

prepare_local_vllm() {
  paimem_gpu_host
  paimem_require RERANKER_MODEL_PATH MEMORY_RERANKER_MODEL RERANKER_CHAT_TEMPLATE

  install_vllm_environment
  prepare_vllm_model
  prepare_chat_template
}

main() {
  # 1. Resolve arguments and load the effective TOML/environment settings.
  parse_arguments "$@"
  load_configuration

  # 2. Prepare the memory service and choose which model backends to install.
  if [[ "$only" == vllm ]]; then
    [[ "$MEMORY_RERANKER_BACKEND" == vllm && "$VLLM_DEPLOYMENT" == local ]] ||
      paimem_fail '--only vllm requires MEMORY_RERANKER_BACKEND=vllm and VLLM_DEPLOYMENT=local'
    prepare_vllm=true
  else
    install_service_environment
    prepare_embedding
    prepare_reranker
  fi

  # 3. Prepare a separate vLLM environment only for a local deployment.
  if [[ "$prepare_vllm" == true ]]; then
    prepare_local_vllm
  fi

  printf 'Environment ready. Start vLLM first if deployed here, then bash start_svr.sh.\n环境准备完成。如在本机部署 vLLM，请先启动它，再运行 bash start_svr.sh。\n'
}

main "$@"
