#!/usr/bin/env bash
# Start the reranker using [reranker] and [reranker.vllm] in the selected TOML.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/config.sh

# Arrays preserve argument boundaries, including values containing spaces.
extra_args=()
args=()
vllm_env=""

parse_arguments() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --config)
        [[ $# -ge 2 ]] || paimem_fail '--config requires a TOML path'
        PAIMEM_CONFIG="$2"
        shift 2
        ;;
      --)
        shift
        extra_args+=("$@")
        break
        ;;
      *)
        extra_args+=("$1")
        shift
        ;;
    esac
  done
}

validate_startup() {
  # Existing remote deployments must be started on their own server.
  [[ "${VLLM_DEPLOYMENT:-local}" == local ]] || paimem_fail 'VLLM_DEPLOYMENT=remote: use the existing server; do not start vLLM here / 已选远程服务，无须在本机启动'
  [[ "${MEMORY_RERANKER_BACKEND:-vllm}" == vllm ]] || paimem_fail 'MEMORY_RERANKER_BACKEND must be vllm'
  paimem_require RERANKER_MODEL_PATH MEMORY_RERANKER_MODEL RERANKER_CHAT_TEMPLATE

  # Check the prepared environment and model assets before probing the GPU.
  vllm_env="${VLLM_ENV:-.venv-vllm}"
  [[ -x "$vllm_env/bin/vllm" ]] || paimem_fail 'vLLM is not installed; run bash scripts/setup_env.sh --only vllm / 请先安装 vLLM'
  [[ -f "$RERANKER_MODEL_PATH/config.json" ]] || paimem_fail 'Model is missing; run setup_env.sh first / 模型不存在，请先准备环境'
  [[ -s "$RERANKER_CHAT_TEMPLATE" ]] || paimem_fail 'Chat template is missing or empty; run setup_env.sh first / 模板不存在或为空'

  paimem_gpu_host
}

build_launch_arguments() {
  args=(
    # Model identity and scoring mode.
    serve "$RERANKER_MODEL_PATH"
    --served-model-name "$MEMORY_RERANKER_MODEL"
    --runner pooling

    # GPU execution and input limits.
    --dtype "${VLLM_DTYPE:-half}"
    --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE:-1}"
    --max-model-len "${VLLM_MAX_MODEL_LEN:-1024}"
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.8}"

    # HTTP listener.
    --host "${VLLM_HOST:-127.0.0.1}"
    --port "${VLLM_PORT:-8089}"

    # Qwen3 reranker template and yes/no classification head.
    --chat-template "$RERANKER_CHAT_TEMPLATE"
    --hf_overrides '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}'
  )

  # The server and PaiMem client share this key; an empty value disables auth.
  if [[ -n "${MEMORY_RERANKER_API_KEY:-}" ]]; then
    args+=(--api-key "$MEMORY_RERANKER_API_KEY")
  fi
}

start_server() {
  # Replace the shell so vLLM receives signals directly. Keep CLI overrides last.
  # The guarded expansion also supports empty arrays on macOS Bash 3.2.
  exec "$vllm_env/bin/vllm" "${args[@]}" ${extra_args[@]+"${extra_args[@]}"}
}

main() {
  # 1. Load deployment settings and any additional vLLM CLI arguments.
  parse_arguments "$@"
  paimem_select_config
  paimem_load_config vllm

  # 2. Validate prerequisites and construct the scoring server command.
  validate_startup
  build_launch_arguments

  # 3. Start vLLM in the foreground.
  start_server
}

main "$@"
