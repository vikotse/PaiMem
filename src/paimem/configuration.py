"""Single-file deployment/strategy configuration; standard library only."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import shlex
import tomllib

# TOML field -> environment variable and default value/type.
SCHEMA = {'service': {'host': ('PAIMEM_HOST', '127.0.0.1'),
             'port': ('PAIMEM_PORT', 8000),
             'api_key': ('MEMORY_API_KEY', 'REPLACE_WITH_MEMORY_API_KEY'),
             'db_path': ('MEMORY_DB_PATH', 'artifacts/memory.db'),
             'diagnostics_dir': ('MEMORY_DIAGNOSTICS_DIR', 'artifacts/diagnostics')},
 'environment': {'python': ('PAIMEM_PYTHON', '3.11'),
                 'package_index': ('UV_DEFAULT_INDEX', 'https://pypi.org/simple'),
                 'hf_endpoint': ('HF_ENDPOINT', 'https://huggingface.co')},
 'embedding': {'backend': ('EMBEDDING_BACKEND', 'remote')},
 'embedding.remote': {'model': ('EMBEDDING_MODEL', 'text-embedding-v4'),
                      'base_url': ('EMBEDDING_BASE_URL', 'https://your-provider.example/v1'),
                      'api_key': ('EMBEDDING_API_KEY', 'REPLACE_WITH_EMBEDDING_API_KEY'),
                      'batch_size': ('EMBEDDING_BATCH_SIZE', 10),
                      'rpm': ('EMBEDDING_RPM', 0),
                      'tpm': ('EMBEDDING_TPM', 0),
                      'query_prefix': ('EMBEDDING_QUERY_PREFIX', '')},
 'embedding.local': {'model_path': ('EMBEDDING_MODEL', 'models/bge-small-en-v1.5'),
                     'repo_id': ('EMBEDDING_REPO_ID', 'BAAI/bge-small-en-v1.5'),
                     'download': ('EMBEDDING_DOWNLOAD', True),
                     'revision': ('EMBEDDING_REVISION', 'main'),
                     'device': ('EMBEDDING_DEVICE', 'cpu'),
                     'batch_size': ('EMBEDDING_BATCH_SIZE', 64),
                     'query_prefix': ('EMBEDDING_QUERY_PREFIX', 'auto')},
 'reranker': {'enabled': ('PAIMEM_RERANKER_ENABLED', True),
              'backend': ('MEMORY_RERANKER_BACKEND', 'vllm'),
              'candidates': ('MEMORY_RERANK_CANDIDATES', 400),
              'max_length': ('MEMORY_RERANKER_MAX_LENGTH', 1024),
              'small2big_enabled': ('MEMORY_RERANK_SMALL2BIG_ENABLED', True),
              'score_threshold': ('MEMORY_RERANK_SCORE_THRESHOLD', '')},
 'reranker.local': {'model_path': ('MEMORY_RERANKER_MODEL', 'models/bge-reranker-v2-m3'),
                    'repo_id': ('RERANKER_REPO_ID', 'BAAI/bge-reranker-v2-m3'),
                    'download': ('RERANKER_DOWNLOAD', True),
                    'revision': ('RERANKER_REVISION', 'main'),
                    'device': ('MEMORY_RERANKER_DEVICE', 'cuda:0'),
                    'batch_size': ('MEMORY_RERANKER_BATCH_SIZE', 16)},
 'reranker.vllm': {'deployment': ('VLLM_DEPLOYMENT', 'local'),
                   'model': ('MEMORY_RERANKER_MODEL', 'MemReranker-4B'),
                   'base_url': ('MEMORY_RERANKER_BASE_URL', 'http://127.0.0.1:8089'),
                   'api_key': ('MEMORY_RERANKER_API_KEY', 'REPLACE_WITH_RERANKER_API_KEY'),
                   'timeout': ('MEMORY_RERANKER_TIMEOUT', 120),
                   'batch_size': ('MEMORY_RERANKER_BATCH_SIZE', 16),
                   'env': ('VLLM_ENV', '.venv-vllm'),
                   'python': ('VLLM_PYTHON', '3.12'),
                   'version': ('VLLM_VERSION', '0.32.0'),
                   'model_path': ('RERANKER_MODEL_PATH', 'models/MemReranker-4B'),
                   'repo_id': ('RERANKER_REPO_ID', 'IAAR-Shanghai/MemReranker-4B'),
                   'download': ('RERANKER_DOWNLOAD', True),
                   'revision': ('RERANKER_REVISION', 'main'),
                   'chat_template': ('RERANKER_CHAT_TEMPLATE', 'models/templates/qwen3_reranker.jinja'),
                   'chat_template_url': ('RERANKER_CHAT_TEMPLATE_URL',
                                         'https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/examples/pooling/score/template/qwen3_reranker.jinja'),
                   'cuda_visible_devices': ('CUDA_VISIBLE_DEVICES', '0'),
                   'host': ('VLLM_HOST', '127.0.0.1'),
                   'port': ('VLLM_PORT', 8089),
                   'dtype': ('VLLM_DTYPE', 'half'),
                   'tensor_parallel_size': ('VLLM_TENSOR_PARALLEL_SIZE', 1),
                   'max_model_len': ('VLLM_MAX_MODEL_LEN', 1024),
                   'gpu_memory_utilization': ('VLLM_GPU_MEMORY_UTILIZATION', 0.8)},
 'retrieval': {'mode': ('MEMORY_RETRIEVAL_MODE', 'hybrid'),
               'lexical_weight': ('MEMORY_LEXICAL_WEIGHT', 0.5),
               'option_view_weight': ('MEMORY_OPTION_VIEW_WEIGHT', 0.0),
               'rrf_k': ('MEMORY_RRF_K', 60)},
 'context': {'chunk_tokens': ('MEMORY_CHUNK_TOKENS', 320),
             'chunk_overlap': ('MEMORY_CHUNK_OVERLAP', 40),
             'messages': ('MEMORY_CONTEXT_MESSAGES', 0),
             'result_window': ('MEMORY_RESULT_WINDOW', 1),
             'result_window_seed_k': ('MEMORY_RESULT_WINDOW_SEED_K', 20),
             'pack_span': ('MEMORY_PACK_SPAN', 2)},
 'time': {'inject_add_timestamp': ('MEMORY_INJECT_ADD_TIMESTAMP', False),
          'resolve_relative_time_on_add': ('MEMORY_RESOLVE_RELATIVE_TIME_ON_ADD', False),
          'temporal_enrichment': ('MEMORY_TEMPORAL_ENRICHMENT', False)},
 'mempursuit': {'enabled': ('MEMORY_MEMPURSUIT_ENABLED', True),
                'candidates': ('MEMORY_MEMPURSUIT_CANDIDATES', 150),
                'ridge': ('MEMORY_MEMPURSUIT_RIDGE', 1e-10),
                'small2big_enabled': ('MEMORY_MEMPURSUIT_SMALL2BIG_ENABLED', True)},
 'rank_fusion': {'enabled': ('MEMORY_RANK_FUSION_ENABLED', False),
                 'dense_weight': ('MEMORY_RANK_FUSION_DENSE_WEIGHT', 0.35),
                 'rerank_weight': ('MEMORY_RANK_FUSION_RERANK_WEIGHT', 0.65),
                 'bm25_weight': ('MEMORY_RANK_FUSION_BM25_WEIGHT', 0),
                 'rrf_weight': ('MEMORY_RANK_FUSION_RRF_WEIGHT', 0),
                 'mp_weight': ('MEMORY_RANK_FUSION_MP_WEIGHT', 0),
                 'mp_limit': ('MEMORY_RANK_FUSION_MP_LIMIT', 100),
                 'remaining_limit': ('MEMORY_RANK_FUSION_REMAINING_LIMIT', 300)},
 'sira': {'enabled': ('MEMORY_SIRA_ENABLED', False),
          'expansion_weight': ('MEMORY_SIRA_EXPANSION_WEIGHT', 0.5),
          'max_df_ratio': ('MEMORY_SIRA_MAX_DF_RATIO', 0.05),
          'max_n': ('MEMORY_SIRA_MAX_N', 4),
          'timeout': ('MEMORY_SIRA_TIMEOUT', 30.0),
          'max_tokens': ('MEMORY_SIRA_MAX_TOKENS', 1024),
          'model': ('MEMORY_SIRA_MODEL', 'qwen-plus'),
          'base_url': ('MEMORY_SIRA_API_BASE', 'https://your-llm-provider.example/v1'),
          'api_key': ('MEMORY_SIRA_API_KEY', 'REPLACE_WITH_SIRA_API_KEY')},
 'long_question': {'enabled': ('MEMORY_LONG_QUESTION_ENABLED', False),
                   'model': ('MEMORY_LONG_QUESTION_MODEL', 'qwen3.7-flash'),
                   'base_url': ('MEMORY_LONG_QUESTION_API_BASE', 'https://your-llm-provider.example/v1'),
                   'api_key': ('MEMORY_LONG_QUESTION_API_KEY', 'REPLACE_WITH_LONG_QUESTION_API_KEY'),
                   'extra_body': ('MEMORY_LONG_QUESTION_EXTRA_BODY', '{}'),
                   'threshold_chars': ('MEMORY_LONG_QUESTION_THRESHOLD_CHARS', 8000),
                   'head_chars': ('MEMORY_LONG_QUESTION_HEAD_CHARS', 2000),
                   'tail_chars': ('MEMORY_LONG_QUESTION_TAIL_CHARS', 2000),
                   'timeout': ('MEMORY_LONG_QUESTION_TIMEOUT', 30.0)},
 'near': {'enabled': ('MEMORY_NEAR_ENABLED', False),
          'layers': ('MEMORY_NEAR_LAYERS', 1),
          'weight': ('MEMORY_NEAR_WEIGHT', 1.0),
          'normalize': ('MEMORY_NEAR_NORMALIZE', False),
          'neighbors': ('MEMORY_NEAR_NEIGHBORS', 50),
          'anchors': ('MEMORY_NEAR_ANCHORS', 16),
          'routing': ('MEMORY_NEAR_ROUTING', 4),
          'beta': ('MEMORY_NEAR_BETA', 10.0),
          'cache_max_users': ('MEMORY_NEAR_CACHE_MAX_USERS', 4),
          'cache_max_mb': ('MEMORY_NEAR_CACHE_MAX_MB', 256.0)},
 'role_name': {'enabled': ('MEMORY_ROLE_NAME_ENABLED', False),
               'model': ('MEMORY_ROLE_NAME_MODEL', 'qwen-plus'),
               'base_url': ('MEMORY_ROLE_NAME_API_BASE', 'https://your-llm-provider.example/v1'),
               'api_key': ('MEMORY_ROLE_NAME_API_KEY', 'REPLACE_WITH_ROLE_NAME_API_KEY'),
               'reranker_model': ('MEMORY_ROLE_NAME_RERANKER_MODEL', ''),
               'timeout': ('MEMORY_ROLE_NAME_TIMEOUT', 60),
               'top_k': ('MEMORY_ROLE_NAME_TOP_K', 8),
               'window': ('MEMORY_ROLE_NAME_WINDOW', 1),
               'context_chars': ('MEMORY_ROLE_NAME_CONTEXT_CHARS', 800),
               'max_tokens': ('MEMORY_ROLE_NAME_MAX_TOKENS', 2500),
               'disable_thinking': ('MEMORY_ROLE_NAME_DISABLE_THINKING', False)},
 'conflict': {'enabled': ('MEMORY_CONFLICT_ENABLED', False),
              'model': ('MEMORY_CONFLICT_MODEL', 'qwen-plus'),
              'base_url': ('MEMORY_CONFLICT_API_BASE', 'https://your-llm-provider.example/v1'),
              'api_key': ('MEMORY_CONFLICT_API_KEY', 'REPLACE_WITH_CONFLICT_API_KEY'),
              'positive_ratio': ('MEMORY_CONFLICT_POSITIVE_RATIO', 0.2),
              'negative_ratio': ('MEMORY_CONFLICT_NEGATIVE_RATIO', 0.2),
              'scope_ratio': ('MEMORY_CONFLICT_SCOPE_RATIO', 0.2),
              'max_tokens': ('MEMORY_CONFLICT_MAX_TOKENS', 8000),
              'timeout': ('MEMORY_CONFLICT_TIMEOUT', 120),
              'disable_thinking': ('MEMORY_CONFLICT_DISABLE_THINKING', False)}}


def default_path() -> Path:
    if 'PAIMEM_CONFIG' in os.environ:
        return Path(os.environ['PAIMEM_CONFIG'])
    local = Path('configs/config.local.toml')
    return local if local.is_file() else Path('configs/default.toml')


def boolean(value):
    if str(value).lower() not in {'true', 'false', '1', '0'}:
        raise ValueError('Boolean settings must be true/false or 1/0')
    return str(value).lower() in {'true', '1'}


def _placeholder(value):
    return 'REPLACE_WITH_' in value or '.example' in value


def prepare(path: Path, environ=None, *, scope='service'):
    """Return effective service settings and env; validate only active components."""
    env = dict(os.environ if environ is None else environ)
    origins = {}
    try:
        config = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError:
        # Parser diagnostics can contain the offending source (possibly a secret).
        raise ValueError('Invalid TOML syntax; check the selected configuration file') from None

    def check_keys(table, prefix=''):
        if not isinstance(table, dict):
            raise ValueError(f'{prefix} must be a TOML table')
        for key, value in table.items():
            name = f'{prefix}.{key}' if prefix else key
            if name in SCHEMA or any(s.startswith(name + '.') for s in SCHEMA):
                check_keys(value, name)
            elif prefix not in SCHEMA or key not in SCHEMA[prefix]:
                raise ValueError(f'Unknown configuration field: {name}')
    check_keys(config)

    def fill(section):
        table = config
        for part in section.split('.'):
            table = table.get(part, {})
        for field, (key, default) in SCHEMA[section].items():
            if section == 'reranker.vllm' and key == 'CUDA_VISIBLE_DEVICES' and scope not in ('setup', 'vllm'):
                continue
            value = table.get(field, default)
            origins[key] = f'[{section}].{field} ({key})'
            if key not in env:
                if (type(value) is not type(default)
                        and not (type(default) is float and type(value) is int)):
                    raise ValueError(f'{origins[key]} has the wrong TOML type')
                env[key] = str(value).lower() if isinstance(value, bool) else str(value)
            raw = env[key]
            try:
                if type(default) is bool:
                    env[key] = str(boolean(raw)).lower()
                elif type(default) is int:
                    int(raw)
                elif type(default) is float and not math.isfinite(float(raw)):
                    raise ValueError()
            except (ValueError, TypeError, OverflowError):
                raise ValueError(f'{origins[key]} has an invalid value') from None

    def require(*keys, optional=False):
        for key in keys:
            value = env.get(key, '')
            if (not optional and not value.strip()) or _placeholder(value):
                raise ValueError(f'Set {origins.get(key, key)} in {path}, or export {key}; replace example values')

    def choice(key, allowed):
        if env[key] not in allowed:
            raise ValueError(f'{origins.get(key, key)} must be one of: {", ".join(allowed)}')
        return env[key]

    def integer(key, minimum=1, maximum=None):
        value = int(env[key])
        if value < minimum or (maximum is not None and value > maximum):
            raise ValueError(f'{origins.get(key, key)} is outside its valid range')
        return value

    fill('environment')
    if scope == 'bootstrap':
        return {}, env
    fill('reranker')
    reranker = boolean(env['PAIMEM_RERANKER_ENABLED'])
    if scope == 'vllm' and not reranker:
        raise ValueError('Enable [reranker].enabled before starting/preparing vLLM')
    if scope != 'vllm':
        fill('service')
        require('MEMORY_API_KEY', 'MEMORY_DB_PATH', 'PAIMEM_HOST', 'MEMORY_DIAGNOSTICS_DIR')
        fill('embedding')
        backend = choice('EMBEDDING_BACKEND', ('local', 'remote'))
        fill('embedding.' + backend)
        require('EMBEDDING_MODEL')
        integer('EMBEDDING_BATCH_SIZE')
        if backend == 'remote':
            require('EMBEDDING_BASE_URL', 'EMBEDDING_API_KEY')
            integer('EMBEDDING_RPM', 0)
            integer('EMBEDDING_TPM', 0)
        elif env.get('EMBEDDING_QUERY_PREFIX') == 'auto':
            env.pop('EMBEDDING_QUERY_PREFIX')
        for section in ('retrieval', 'context', 'time', 'mempursuit', 'rank_fusion', 'sira', 'near', 'role_name', 'conflict', 'long_question'):
            # Disabled LLM strategies do not introduce placeholder credentials into the environment.
            if section in ('sira', 'role_name', 'conflict', 'long_question'):
                key = 'MEMORY_' + section.upper() + '_ENABLED'
                value = env.get(key, config.get(section, {}).get('enabled', False))
                env[key] = str(boolean(value)).lower()
                if not boolean(value):
                    continue
            fill(section)
        for section in ('sira', 'role_name', 'conflict', 'long_question'):
            if boolean(env['MEMORY_' + section.upper() + '_ENABLED']):
                for suffix in ('MODEL', 'API_BASE', 'API_KEY'):
                    key = f'MEMORY_{section.upper()}_{suffix}'
                    # The section itself is complete; shared fallback applies only to an explicit empty field.
                    if not env.get(key):
                        common = 'BASE_URL' if suffix == 'API_BASE' else suffix
                        env[key] = env.get('ANSWER_' + suffix) or env.get('OPENAI_' + common, '')
                    require(key)
        if boolean(env.get('MEMORY_LONG_QUESTION_ENABLED', 'false')):
            # Standard library only: usable before the project dependencies are installed.
            if __package__:
                from .long_question import LongQuestionConfig
            else:
                from long_question import LongQuestionConfig
            try:
                LongQuestionConfig(
                    enabled=True,
                    threshold_chars=int(env['MEMORY_LONG_QUESTION_THRESHOLD_CHARS']),
                    head_chars=int(env['MEMORY_LONG_QUESTION_HEAD_CHARS']),
                    tail_chars=int(env['MEMORY_LONG_QUESTION_TAIL_CHARS']),
                    timeout=float(env['MEMORY_LONG_QUESTION_TIMEOUT']),
                    extra_body=env['MEMORY_LONG_QUESTION_EXTRA_BODY'],
                )
            except (ValueError, TypeError, OverflowError) as error:
                raise ValueError(f'[long_question]: {error}') from None
        if boolean(env.get('MEMORY_ROLE_NAME_ENABLED', 'false')) and not reranker:
            require('MEMORY_ROLE_NAME_RERANKER_MODEL')
        if boolean(env['MEMORY_RANK_FUSION_ENABLED']) and (not reranker or not boolean(env['MEMORY_MEMPURSUIT_ENABLED'])):
            raise ValueError('Rank fusion requires reranker and MemPursuit')
    if reranker:
        kind = choice('MEMORY_RERANKER_BACKEND', ('cross-encoder', 'vllm'))
        if scope == 'vllm' and kind != 'vllm':
            raise ValueError('[reranker].backend must be vllm')
        fill('reranker.local' if kind == 'cross-encoder' else 'reranker.vllm')
        require('MEMORY_RERANKER_MODEL')
        integer('MEMORY_RERANKER_MAX_LENGTH')
        integer('MEMORY_RERANKER_BATCH_SIZE')
        if kind == 'vllm':
            deployment = choice('VLLM_DEPLOYMENT', ('local', 'remote'))
            require('MEMORY_RERANKER_API_KEY', optional=True)
            if scope != 'vllm':
                require('MEMORY_RERANKER_BASE_URL')
            if scope == 'vllm' or (scope == 'setup' and deployment == 'local'):
                if deployment != 'local':
                    raise ValueError('VLLM_DEPLOYMENT=remote: use the existing server; do not start vLLM here')
                require('VLLM_ENV', 'VLLM_VERSION', 'VLLM_PYTHON', 'RERANKER_MODEL_PATH', 'RERANKER_CHAT_TEMPLATE')
                integer('VLLM_PORT', 1, 65535)
                integer('VLLM_TENSOR_PARALLEL_SIZE')
                if integer('VLLM_MAX_MODEL_LEN') < int(env['MEMORY_RERANKER_MAX_LENGTH']):
                    raise ValueError('reranker.max_length must not exceed reranker.vllm.max_model_len')
    else:
        env.pop('MEMORY_RERANKER_MODEL', None)
        env['MEMORY_RERANKER_BACKEND'] = 'cross-encoder'
    service = {} if scope == 'vllm' else {
        'host': env['PAIMEM_HOST'], 'port': integer('PAIMEM_PORT', 1, 65535), 'reranker': reranker,
    }
    return service, env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=default_path())
    parser.add_argument('--scope', choices=('bootstrap', 'service', 'setup', 'vllm'), default='service')
    parser.add_argument('--shell', action='store_true', help='Shell-quoted exports for scripts; includes secrets, never log this output')
    args = parser.parse_args(argv)
    try:
        service, env = prepare(args.config, scope=args.scope)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.shell:
        keys = {key for fields in SCHEMA.values() for key, _ in fields.values()}
        for key in sorted(keys & env.keys()):
            print(f'export {key}={shlex.quote(env[key])}')
        if args.scope != 'bootstrap' and env.get('EMBEDDING_QUERY_PREFIX') is None:
            print('unset EMBEDDING_QUERY_PREFIX')
        if not boolean(env.get('PAIMEM_RERANKER_ENABLED', 'true')):
            print('unset MEMORY_RERANKER_MODEL')
    else:
        print(json.dumps({'valid': True, 'config': str(args.config), 'scope': args.scope,
                          'reranker': service.get('reranker')}, indent=2))


if __name__ == '__main__':
    main()
