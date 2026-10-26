#!/usr/bin/env python3
"""Aggregate AML reports or serve a live, local version comparison."""
import argparse
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import tempfile
from urllib.parse import parse_qs, urlsplit


METRICS = ('questions', 'answered', 'judged', 'wall_seconds_this_run',
           'request_seconds', 'input_tokens', 'output_tokens', 'requests',
           'unknown_usage', 'estimated_usage')
USAGE_METRICS = set(METRICS[4:])


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def read_json(path):
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object')
    return value


def aggregate(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f'Not a directory: {root}')
    datasets, warnings = [], []
    # One immediate child per dataset avoids accidentally combining versions.
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or folder.is_symlink():
            continue
        path = folder / 'report.json'
        if not path.exists():
            warnings.append(f'{folder.name}: report.json missing')
            continue
        try:
            report = read_json(path)
            usage = report.get('usage') or {}
            if not isinstance(usage, dict):
                raise ValueError('usage must be an object')
            row = dict(dataset=folder.name, source=f'{folder.name}/report.json',
                       contract=report.get('contract'), complete=report.get('complete') is True)
            for key in METRICS:
                value = (usage if key in USAGE_METRICS else report).get(key)
                row[key] = value if number(value) else None
            score = report.get('score_mean')
            row['score_mean'] = score if number(score) and score <= 1 else None
            row['cost_currency'] = report.get('cost_currency') or 'UNSPECIFIED'
            prices = report.get('prices_configured')
            priced = isinstance(prices, dict) and bool(prices) and all(v is True for v in prices.values())
            cost = usage.get('estimated_cost')
            row['estimated_cost'] = cost if priced and number(cost) else None
            row['cost_complete'] = row['estimated_cost'] is not None
            row['cost_note'] = report.get('cost_note')
            missing = [key for key in METRICS if row[key] is None]
            if missing:
                warnings.append(f'{folder.name}: missing/invalid metrics: {", ".join(missing)}')
            if row['score_mean'] is None:
                warnings.append(f'{folder.name}: score unavailable')
            if not row['cost_complete']:
                warnings.append(f'{folder.name}: cost unavailable or incomplete')
            datasets.append(row)
        except (OSError, ValueError, TypeError) as exc:
            warnings.append(f'{folder.name}: {exc}')
    if not datasets:
        raise ValueError(f'No readable dataset reports in {root}')
    totals = {key: sum(row[key] for row in datasets) if all(row[key] is not None for row in datasets) else None
              for key in METRICS}
    scored = [r for r in datasets if r['score_mean'] is not None and r['judged'] is not None and r['judged'] > 0]
    weight = sum(r['judged'] for r in scored)
    totals.update(dataset_count=len(datasets), complete=all(r['complete'] for r in datasets) and not warnings,
                  scored_questions=weight,
                  score_mean=sum(r['score_mean'] * r['judged'] for r in scored) / weight if weight else None)
    totals['costs'] = {}
    for currency in sorted({r['cost_currency'] for r in datasets}):
        rows = [r for r in datasets if r['cost_currency'] == currency]
        costs = [r['estimated_cost'] for r in rows if r['estimated_cost'] is not None]
        totals['costs'][currency] = dict(estimated_cost=sum(costs) if costs else None,
                                       complete=all(r['cost_complete'] for r in rows))
    return dict(schema_version=1, version=root.name, generated_at=datetime.now(timezone.utc).isoformat(),
                datasets=datasets, totals=totals, warnings=warnings)


def fmt(value, percent=False):
    if value is None:
        return '—'
    return f'{value:.2%}' if percent else f'{value:,.4f}'.rstrip('0').rstrip('.')


def markdown(report):
    lines = [f'# {report["version"]} 评测汇总', '', f'生成时间：{report["generated_at"]}', '',
             '| 数据集 | 评测口径 | 题数 | 已回答 | 已评测 | 得分 | 本次耗时(s) | 请求累计(s) | 估算花销 | 完成 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---|---|']
    for r in report['datasets']:
        cost = f'{fmt(r["estimated_cost"])} {r["cost_currency"]}' + ('' if r['cost_complete'] else '（不完整）')
        values = [r['dataset'], r['contract'] or '—', *(fmt(r[k]) for k in ('questions','answered','judged')),
                  fmt(r['score_mean'], True), fmt(r['wall_seconds_this_run']), fmt(r['request_seconds']), cost,
                  '是' if r['complete'] else '否']
        lines.append('| ' + ' | '.join(str(v).replace('|', '\\|').replace('\n', ' ') for v in values) + ' |')
    t = report['totals']
    lines += ['', f'整体加权得分：{fmt(t["score_mean"], True)}；参与加权题数：{t["scored_questions"]}。', '',
              '| 整体指标 | 数值 |', '|---|---:|']
    lines += [f'| {key} | {fmt(t[key])} |' for key in METRICS]
    lines += [f'| 估算花销 ({currency}) | {fmt(cost["estimated_cost"])}' + ('' if cost['complete'] else '（不完整）') + ' |'
              for currency, cost in t['costs'].items()]
    lines += ['', '得分按已评测且有有效分数的题数加权；不同评测口径混合后的得分仅供参考。',
              '本次耗时为各数据集 wall_seconds_this_run 之和，并非并发运行的实际总时长，也不包含历史续跑耗时。',
              '请求累计耗时和用量沿用原 report 的累计口径；花销仅统计成功返回且有可计价用量的请求，不含未知用量请求的费用。', '']
    lines += [f'- {w}' for w in report['warnings']]
    return '\n'.join(lines) + '\n'


def atomic_write(path, content):
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


SPLITS_DIR = Path(__file__).resolve().parents[1] / 'data' / 'splits'


def category_mapping(version):
    """Match the longest split suffix, including suffixes containing underscores."""
    candidates = sorted(SPLITS_DIR.glob('question_categories.*.json'),
                        key=lambda p: len(p.name), reverse=True)
    for path in candidates:
        suffix = path.name[len('question_categories.'):-len('.json')]
        if not suffix or not version.endswith('_' + suffix):
            continue
        try:
            data = read_json(path)
            entries = data.get('questions')
            if not isinstance(entries, list):
                raise ValueError('questions must be a list')
            mapping = {}
            for entry in entries:
                if not isinstance(entry, dict) or any(not isinstance(entry.get(k), str) or not entry[k].strip()
                        for k in ('source_dataset', 'item_id', 'type')):
                    raise ValueError('invalid question category entry')
                key = (entry['source_dataset'], entry['item_id'])
                if key in mapping:
                    raise ValueError('duplicate dataset/item_id')
                mapping[key] = dict(type=entry['type'], subtype=str(entry.get('subtype', '')))
            return mapping, []
        except (OSError, ValueError, TypeError) as exc:
            return None, [f'{path.name}: {exc}']
    return None, []


def categorize_cases(rows, versions, dataset, category):
    mappings, warnings = [], []
    for version in versions:
        mapping, notes = category_mapping(version)
        mappings.append(mapping)
        warnings.extend(notes)
    available = any(m is not None for m in mappings)
    if not available:
        return rows, [], warnings
    for row in rows:
        labels = [m.get((dataset, row['id'])) if m is not None else None for m in mappings]
        known = [label for label in labels if label is not None]
        label = known[0] if known else dict(type='未分类', subtype='')
        if any(other != label for other in known):
            label = dict(type='分类不一致', subtype='')
        row.update(category=label['type'], subtype=label['subtype'])
    categories = sorted({row['category'] for row in rows})
    return [r for r in rows if not category or r['category'] == category], categories, warnings


def category_summary(root, report):
    mapping, warnings = category_mapping(report['version'])
    if mapping is None:
        return None, warnings
    groups = {}
    for dataset in report['datasets']:
        name = dataset['dataset']
        run = load_case_run(root, report['version'], name)
        warnings.extend(f'{name}: {w}' for w in run['warnings'])
        actual = set().union(*(set(rows) for rows in run['rows'].values()))
        ids = actual | {ident for ds, ident in mapping if ds == name}
        for ident in ids:
            label = mapping.get((name, ident), {}).get('type', '未分类')
            group = groups.setdefault(label, dict(category=label, questions=0, scored_questions=0,
                                                  score_sum=0, contracts=set()))
            group['questions'] += 1
            if run['contract']:
                group['contracts'].add(run['contract'])
            score = case_score(run['rows']['judge'].get(ident), run['contract'])
            if score is not None:
                group['scored_questions'] += 1
                group['score_sum'] += score
    for group in groups.values():
        group['score_mean'] = group.pop('score_sum') / group['scored_questions'] if group['scored_questions'] else None
        group['contracts'] = sorted(group['contracts'])
    return sorted(groups.values(), key=lambda row: row['category']), warnings


def load_versions(root):
    reports, errors = [], []
    for folder in sorted(Path(root).iterdir()):
        path = folder / 'summary.json'
        if folder.is_symlink() or not folder.is_dir() or not path.is_file() or path.is_symlink():
            continue
        try:
            report = read_json(path)
            if report.get('schema_version') != 1 or not isinstance(report.get('datasets'), list) or not isinstance(report.get('totals'), dict):
                raise ValueError('Invalid summary schema')
            report['version'] = folder.name
            report['categories'], category_warnings = category_summary(root, report)
            report['warnings'] = list(report.get('warnings', [])) + category_warnings
            reports.append(report)
        except (OSError, ValueError) as exc:
            errors.append(f'{folder.name}: {exc}')
    return dict(reports=reports, errors=errors)


def safe_path(root, *parts):
    """Only named descendants; never follow artifact or directory symlinks."""
    path = Path(root).resolve()
    for part in parts:
        if not isinstance(part, str) or not part or part in ('.', '..') or any(c in part for c in ('/', '\\', '\0')):
            raise ValueError('Invalid path component')
        path = path / part
        if path.is_symlink():
            raise ValueError('Symlink artifacts are not accessible')
    return path


def case_id(row, contract):
    # Same ID precedence as aml_pipeline_adapters.row_id_of, without importing
    # the runner's third-party dependencies into this standard-library server.
    keys = ('idx', 'id') if contract == 'clbench_rubric' else ('id', 'idx', 'qa_id')
    return next((str(row[k]) for k in keys if row.get(k) is not None), '')


def case_score(row, contract):
    if not row or not isinstance(contract, str):
        return None
    if contract in ('binary_judge', 'personamem_v1', 'personamem_v2'):
        value = row.get('is_correct')
        return float(value) if isinstance(value, bool) else None
    key = {'beam_rubric': 'llm_judge_score', 'clbench_rubric': 'rubric_clbench_score',
           'scriptmem_rule': 'score'}.get(contract)
    value = row.get(key) if key else None
    try:
        value = float(value) if isinstance(value, str) else value
    except ValueError:
        return None
    return value if number(value) and value <= 1 else None


def artifact_json(folder, name, warnings):
    path = safe_path(folder, name)
    if not path.exists():
        return None
    try:
        return read_json(path)
    except (OSError, ValueError) as exc:
        warnings.append(f'{name}: unreadable ({type(exc).__name__})')
        return None


def stage_rows(folder, name, checkpoint, contract, warnings):
    """Read available rows, including checkpoints newer than the stage JSONL."""
    rows, sources, duplicate = {}, {}, set()
    path = safe_path(folder, name)
    if path.exists():
        try:
            with path.open(encoding='utf-8') as stream:
                for index, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        if not isinstance(row, dict) or not (ident := case_id(row, contract)):
                            raise ValueError('Missing ID')
                        if ident in rows or ident in duplicate:
                            duplicate.add(ident)
                            rows.pop(ident, None)
                            sources.pop(ident, None)
                            warnings.append(f'{name}:{index}: duplicate case ID {ident}')
                            continue
                        rows[ident], sources[ident] = row, f'{name}:{index}'
                    except (ValueError, TypeError):
                        warnings.append(f'{name}:{index}: invalid row')
        except (OSError, UnicodeError) as exc:
            warnings.append(f'{name}: unreadable ({type(exc).__name__})')
    directory = safe_path(folder, checkpoint)
    if directory.is_dir():
        for path in sorted(directory.glob('*.json')):
            # Skip symlinks even if their targets are inside the results root.
            path = safe_path(folder, checkpoint, path.name)
            try:
                row = read_json(path)
                ident = case_id(row, contract)
                if not ident or hashlib.sha256(ident.encode()).hexdigest() + '.json' != path.name:
                    raise ValueError('Checkpoint ID mismatch')
                if ident not in duplicate:
                    rows[ident], sources[ident] = row, f'{checkpoint}/{path.name}'
            except (OSError, ValueError, TypeError) as exc:
                warnings.append(f'{checkpoint}/{path.name}: unreadable ({type(exc).__name__})')
    if not rows:
        warnings.append(f'{name} / {checkpoint}: no usable rows')
    return rows, sources


def load_case_run(root, version, dataset):
    folder = safe_path(root, version, dataset)
    warnings = []
    report = artifact_json(folder, 'report.json', warnings) or {}
    manifest = artifact_json(folder, 'manifest.json', warnings) or {}
    contract = report.get('contract')
    if not isinstance(contract, str):
        warnings.append('report.json: missing/invalid contract')
        contract = None
    rows, sources = {}, {}
    for stage, filename, checkpoint in (('input', 'input.jsonl', 'replay'),
                                        ('answer', 'answers.jsonl', 'answer'),
                                        ('judge', 'judged.jsonl', 'judge')):
        rows[stage], sources[stage] = stage_rows(folder, filename, checkpoint, contract, warnings)
    return dict(version=version, dataset=dataset, folder=folder, contract=contract,
                rows=rows, sources=sources, warnings=warnings, manifest=manifest,
                metadata={key: manifest.get(key) for key in ('input_sha256', 'pipeline_sha256', 'mode',
                          'top_k', 'answer_model', 'judge_model', 'personamem_mode')})


def sample_signature(row):
    if not row:
        return None
    # Deliberately exclude retrieved context, option-letter shuffling and
    # memory diagnostics: these are expected to differ between versions.
    sample = {key: row.get(key) for key in ('question', 'user_query', 'gold_answer', 'correct_answer',
              'options', 'incorrect_answers', 'rubric', 'rubrics', 'qa_type', 'question_type')}
    return json.dumps(sample, sort_keys=True, ensure_ascii=True)


def comparison_row(ident, baseline, comparison):
    left = baseline['rows']['input'].get(ident)
    right = comparison['rows']['input'].get(ident)
    scores = [case_score(run['rows']['judge'].get(ident), run['contract']) for run in (baseline, comparison)]
    sample_matches = bool(left and right and sample_signature(left) == sample_signature(right))
    contract_matches = bool(baseline['contract'] and baseline['contract'] == comparison['contract'])
    comparable = sample_matches and contract_matches and all(s is not None for s in scores)
    delta = scores[1] - scores[0] if comparable else None
    if not left or not right or any(s is None for s in scores):
        status, reason = 'missing', '缺少输入或有效评测结果'
    elif not sample_matches or not contract_matches:
        status, reason = 'incomparable', '题目／参考答案／rubric 或评测口径不同'
    elif math.isclose(delta, 0, abs_tol=1e-9):
        status, reason, delta = 'same', '', 0.0
    else:
        status, reason = ('improved' if delta > 0 else 'regressed'), ''
    item = left or right or {}
    return dict(id=ident, question=item.get('question', item.get('user_query', '')),
                baseline_score=scores[0], comparison_score=scores[1], delta=delta,
                status=status, reason=reason, sample_matches=sample_matches,
                baseline_answered=ident in baseline['rows']['answer'],
                comparison_answered=ident in comparison['rows']['answer'])


def comparison_warnings(baseline, comparison):
    notes = [f'{run["version"]}: {note}' for run in (baseline, comparison) for note in run['warnings']]
    if baseline['contract'] != comparison['contract']:
        notes.append('评测口径不同，不计算分差。')
    for key, label in (('input_sha256', '源数据'), ('pipeline_sha256', '官方 pipeline')):
        values = [run['metadata'].get(key) for run in (baseline, comparison)]
        if all(values) and values[0] != values[1]:
            notes.append(f'{label}指纹不同，请确认比较范围与评分流程。')
        elif not all(values):
            notes.append(f'{label}指纹缺失，无法验证完全一致。')
    if set(baseline['rows']['input']) != set(comparison['rows']['input']):
        notes.append('两个版本的题目 ID 集合不同；缺失题目单独列出，不当作零分。')
    return notes


def case_datasets(root, versions, dataset):
    if dataset != '*':
        return [dataset]
    names = set()
    for version in versions:
        folder = safe_path(root, version)
        if not folder.is_dir():
            continue
        for child in folder.iterdir():
            if child.is_dir() and not child.is_symlink() and any(
                    (child / name).exists() for name in ('report.json', 'input.jsonl', 'replay',
                                                        'answers.jsonl', 'answer', 'judged.jsonl', 'judge')):
                names.add(child.name)
    return sorted(names)


def filter_case_scores(rows, score_filter, key):
    predicates = {
        'all': lambda score: True,
        'full': lambda score: score == 1,
        'partial': lambda score: score is not None and 0 < score < 1,
        'zero': lambda score: score == 0,
        'not_full': lambda score: score is not None and score < 1,
        'unscored': lambda score: score is None,
    }
    if score_filter not in predicates:
        raise ValueError('Invalid score filter')
    return [row for row in rows if predicates[score_filter](row[key])]


def compare_cases(root, baseline, comparison, dataset, filter='different', query='', page=1, page_size=50, category='',
                  score_filter='all', score_version='comparison'):
    if score_version not in ('baseline', 'comparison'):
        raise ValueError('Invalid score version')
    if filter not in ('different', 'regressed', 'improved', 'missing', 'incomparable', 'all'):
        raise ValueError('Invalid case filter')
    if page < 1 or not 1 <= page_size <= 100:
        raise ValueError('Invalid pagination')
    rows, categories, warnings, contracts, total = [], set(), [], [], 0
    for name in case_datasets(root, (baseline, comparison), dataset):
        left, right = (load_case_run(root, version, name) for version in (baseline, comparison))
        ids = set().union(*(set(stage) for run in (left, right) for stage in run['rows'].values()))
        total += len(ids)
        batch = [dict(comparison_row(ident, left, right), dataset=name) for ident in ids]
        batch, labels, notes = categorize_cases(batch, (baseline, comparison), name, category)
        rows.extend(batch)
        categories.update(labels)
        warnings.extend(f'{name}: {w}' for w in comparison_warnings(left, right) + notes)
        contracts.extend([left['contract'], right['contract']])
    rows = filter_case_scores(rows, score_filter, score_version + '_score')
    counts = {status: sum(row['status'] == status for row in rows)
              for status in ('improved', 'regressed', 'same', 'missing', 'incomparable')}
    query = query.casefold().strip()
    rows = [r for r in rows if (filter == 'all' or (filter == 'different' and r['status'] in ('improved', 'regressed'))
            or r['status'] == filter) and (not query or query in (r['id'] + ' ' + str(r['question'])).casefold())]
    rows.sort(key=lambda r: (-(abs(r['delta']) if r['delta'] is not None else -1), r['dataset'], r['id']))
    filtered = len(rows)
    page = min(page, max(1, math.ceil(filtered / page_size)))
    return dict(baseline=baseline, comparison=comparison, dataset=dataset,
                contracts=contracts, counts=counts, total=total,
                filtered_count=filtered, page=page, page_size=page_size,
                cases=rows[(page-1)*page_size:page*page_size], categories=sorted(categories),
                warnings=warnings)


def single_case_row(ident, run):
    item = next((rows[ident] for rows in run['rows'].values() if ident in rows), {})
    return dict(id=ident, question=item.get('question', item.get('user_query', '')),
                score=case_score(run['rows']['judge'].get(ident), run['contract']),
                stages={stage: ident in rows for stage, rows in run['rows'].items()})


def single_cases(root, version, dataset, query='', page=1, page_size=50, category='', score_filter='all'):
    if page < 1 or not 1 <= page_size <= 100:
        raise ValueError('Invalid pagination')
    rows, categories, warnings, total = [], set(), [], 0
    query = query.casefold().strip()
    for name in case_datasets(root, (version,), dataset):
        run = load_case_run(root, version, name)
        ids = set().union(*(set(stage) for stage in run['rows'].values()))
        total += len(ids)
        batch = [dict(single_case_row(ident, run), dataset=name) for ident in sorted(ids)]
        batch, labels, notes = categorize_cases(batch, (version,), name, category)
        rows.extend(batch)
        categories.update(labels)
        warnings.extend(f'{name}: {w}' for w in run['warnings'] + notes)
    rows = filter_case_scores(rows, score_filter, 'score')
    rows = [row for row in rows if not query or query in (row['id'] + ' ' + str(row['question'])).casefold()]
    page = min(page, max(1, math.ceil(len(rows) / page_size)))
    return dict(mode='single', version=version, dataset=dataset, total=total,
                filtered_count=len(rows), page=page, page_size=page_size,
                cases=rows[(page-1)*page_size:page*page_size], categories=sorted(categories),
                warnings=warnings)


def diagnostic_channels(candidates):
    return sorted({name for row in candidates if isinstance(row, dict)
                   and isinstance(row.get('channels'), dict) for name in row['channels']})


def display_search_results(response, diagnostics):
    """Normalize display-only metadata without changing saved/raw responses."""
    rows = response.get('data') if isinstance(response, dict) else None
    source = 'search'
    if not isinstance(rows, list):
        rows = diagnostics.get('final_results') if isinstance(diagnostics, dict) else None
        source = 'diagnostics'
    if not isinstance(rows, list):
        return None, None
    candidates = (diagnostics or {}).get('candidates', [])
    by_id = {row['id']: row for row in candidates
             if isinstance(row, dict) and isinstance(row.get('id'), str)} if isinstance(candidates, list) else {}
    results = []
    queues = {'semantic', 'positive', 'negative', 'scope', 'conflict_summary', 'unknown'}
    for index, row in enumerate(rows, 1):
        item = dict(row) if isinstance(row, dict) else {'content': row}
        queue = item.get('recall_queue')
        if not isinstance(queue, str) or queue not in queues:
            content = str(item.get('content') or '')
            if content.startswith('[Positive-related evidence] '):
                queue = 'positive'
            elif content.startswith('[Negative-related evidence] '):
                queue = 'negative'
            elif content.startswith('[Scope/correction evidence] '):
                queue = 'scope'
            elif content.startswith('[Conflicting evidence]') or str(item.get('id', '')).startswith('conflict:'):
                queue = 'conflict_summary'
            else:
                queue = 'unknown'
        item['recall_queue'] = queue
        candidate = by_id.get(str(item.get('id', '')), {})
        item['final_rank'] = index
        item['semantic_rank'] = candidate.get('semantic_returned_rank')
        item['semantic_rank_state'] = ('not_applicable' if queue == 'conflict_summary' else
                                       'unrecorded' if 'semantic_returned_rank' not in candidate else
                                       'not_in_top_k' if item['semantic_rank'] is None else 'recorded')
        routes = candidate.get('channels') or {}
        item['original_channels'] = {key: value for key, value in routes.items()
                                     if key in {'dense', 'fusion', 'reranker'}}
        results.append(item)
    return results, source


def conflict_display(diagnostics, results):
    """Separate classification from assessment, including legacy stage statuses."""
    info = (diagnostics or {}).get('conflict_strategy') or {}
    status = info.get('status')
    judgment = info.get('is_judgment')
    if type(judgment) is not bool:
        judgment = (True if status in {'judgment', 'conflict', 'no_conflict', 'assessment_failed',
                                      'insufficient_evidence', 'retrieval_failed'} else
                    False if status == 'not_judgment' else None)
    classification = ('judgment' if judgment is True else 'non_judgment' if judgment is False else
                      'failed' if status == 'classification_failed' else
                      'not_run' if status == 'bypassed' or info.get('enabled') is False else 'unknown')
    assessment = {'conflict': 'conflict', 'no_conflict': 'no_conflict',
                  'assessment_failed': 'failed'}.get(status,
                  'not_run' if info else 'unknown')
    counts = {queue: sum(row.get('recall_queue') == queue for row in (results or []))
              for queue in ('positive', 'negative', 'scope')}
    explanations = ([info['explanation']] if info.get('explanation') else []) + [
        row.get('content', '') for row in (results or []) if row.get('recall_queue') == 'conflict_summary']
    return dict(classification=classification, assessment=assessment, status=status,
                enabled=info.get('enabled'), strategy=info.get('strategy'),
                explanation_placement=info.get('explanation_placement'),
                skip_reason=info.get('skip_reason'),
                assessment_error_type=info.get('assessment_error_type'),
                assessment_error=info.get('assessment_error'),
                positive_query=info.get('positive_query'), negative_query=info.get('negative_query'),
                scope_query=info.get('scope_query'),
                positive_count=info.get('positive_count', counts['positive']),
                negative_count=info.get('negative_count', counts['negative']),
                scope_count=info.get('scope_count', counts['scope']),
                explanation=list(dict.fromkeys(explanations)))


def diagnostic_metadata(diagnostics):
    # Conflict queues contain their own full candidate sets. Keep those on the
    # saved diagnostics file rather than embedding them in case details.
    result = {key: value for key, value in diagnostics.items()
              if key not in {'candidates', 'final_results', 'semantic_results'}}
    conflict = result.get('conflict_strategy')
    if isinstance(conflict, dict) and isinstance(conflict.get('queues'), dict):
        result['conflict_strategy'] = {**conflict, 'queues': {
            name: {key: value for key, value in queue.items() if key != 'pipeline'}
            for name, queue in conflict['queues'].items() if isinstance(queue, dict)}}
    return result


def original_messages(run, item, ident):
    """Resolve original conversations only from the fingerprinted run input."""
    result = dict(messages=None, source=None, reason=None)
    manifest = run['manifest']
    source = manifest.get('input')
    if not isinstance(source, str) or not source:
        return dict(result, reason='manifest 未记录原始输入文件')
    path = Path(source)
    # Runner paths are relative to its working directory, normally the repo root.
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    try:
        if path.is_symlink():
            return dict(result, reason='原始输入路径包含符号链接，无法读取')
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != manifest.get('input_sha256'):
            return dict(result, reason='原始输入文件指纹不符，无法确认是本次运行的数据')
        rows = [(index, json.loads(line)) for index, line in enumerate(payload.decode('utf-8').split('\n'), 1)
                if line.strip()]
        record_id = (item or {}).get('x_record_id')
        if record_id is not None:
            matches = [(index, row) for index, row in rows if isinstance(row, dict)
                       and str(row.get('id')) == str(record_id)]
        elif manifest.get('input_format') == 'official':
            matches = [(index, row) for index, row in rows if isinstance(row, dict)
                       and case_id(row, run['contract']) == ident]
        else:
            return dict(result, reason='逐题输入未记录 x_record_id，无法关联原始对话')
        if len(matches) != 1:
            return dict(result, reason='原始记录未找到或 ID 重复，无法唯一关联')
        index, row = matches[0]
        result['source'] = f'{path}:{index}'
        if not isinstance(row.get('messages'), list):
            return dict(result, reason='原始记录未保存 messages 列表')
        return dict(result, messages=row['messages'])
    except (OSError, UnicodeError, ValueError) as exc:
        return dict(result, reason=f'原始输入文件不存在或无法读取（{type(exc).__name__}）')


def case_detail(root, baseline, comparison, dataset, ident):
    runs = [load_case_run(root, version, dataset) for version in
            ((baseline, comparison) if comparison is not None else (baseline,))]
    if not any(ident in rows for run in runs for rows in run['rows'].values()):
        raise FileNotFoundError('Case not found')
    versions = []
    for run in runs:
        filename = hashlib.sha256(ident.encode()).hexdigest() + '.json'
        search_dir = safe_path(run['folder'], 'search')
        diagnostic_dir = safe_path(run['folder'], 'search_diagnostics')
        warnings = []
        response = artifact_json(search_dir, filename, warnings)
        diagnostics = artifact_json(diagnostic_dir, filename, warnings)
        diagnostic_summary = None
        if diagnostics is not None:
            candidates = diagnostics.get('candidates')
            if isinstance(candidates, list):
                diagnostic_summary = diagnostic_metadata(diagnostics)
                diagnostic_summary['candidate_count'] = len(candidates)
                diagnostic_summary['channels'] = diagnostic_channels(candidates)
            else:
                warnings.append(f'search_diagnostics/{filename}: missing candidates')
        results, result_source = display_search_results(response, diagnostics)
        item = run['rows']['input'].get(ident)
        versions.append(dict(version=run['version'], contract=run['contract'], metadata=run['metadata'],
                             input=item, original_messages=original_messages(run, item, ident),
                             answer=run['rows']['answer'].get(ident),
                             judge=run['rows']['judge'].get(ident), search_response=response,
                             search_diagnostics=diagnostic_summary,
                             search_results=results, search_results_source=result_source,
                             conflict_analysis=conflict_display(diagnostics, results),
                             search_source=f'search/{filename}' if response is not None else None,
                             diagnostic_source=f'search_diagnostics/{filename}' if diagnostics is not None else None,
                             sources={stage: paths.get(ident) for stage, paths in run['sources'].items()},
                             warnings=warnings))
    return dict(case=comparison_row(ident, *runs) if comparison is not None else single_case_row(ident, runs[0]),
                mode='compare' if comparison is not None else 'single', dataset=dataset, versions=versions,
                warnings=comparison_warnings(*runs) if comparison is not None else runs[0]['warnings'])


def case_diagnostics_page(root, baseline, comparison, dataset, ident, version,
                          channel='', query='', page=1, page_size=50, order='asc'):
    if version not in (baseline, comparison):
        raise ValueError('Version must be one of the compared versions')
    if page < 1 or not 1 <= page_size <= 100:
        raise ValueError('Invalid pagination')
    if order not in ('asc', 'desc'):
        raise ValueError('Invalid sort order')
    filename = hashlib.sha256(ident.encode()).hexdigest() + '.json'
    path = safe_path(root, version, dataset, 'search_diagnostics', filename)
    if not path.is_file():
        raise FileNotFoundError('Search diagnostics not found')
    artifact = read_json(path)
    candidates = artifact.get('candidates')
    if not isinstance(candidates, list):
        raise ValueError('Invalid search diagnostics schema')
    channels = diagnostic_channels(candidates)
    channel = channel or 'returned_rank'
    if channel != 'returned_rank' and channel not in channels:
        raise ValueError('Invalid retrieval channel')
    query = query.strip().casefold()
    matches = [row for row in candidates if isinstance(row, dict)
               and (not query or query in (str(row.get('id', '')) + ' ' + str(row.get('content', ''))).casefold())]
    def rank(row):
        routes = row.get('channels') or {}
        entry = routes.get(channel) if isinstance(routes, dict) else None
        value = row.get('returned_rank') if channel == 'returned_rank' else (entry.get('rank') if isinstance(entry, dict) else None)
        valid = isinstance(value, int) and not isinstance(value, bool) and value > 0
        return (False, value if order == 'asc' else -value) if valid else (True, 0)
    matches.sort(key=lambda row: (rank(row), str(row.get('id', ''))))
    total = len(matches)
    page = min(page, max(1, math.ceil(total / page_size)))
    return dict(version=version, dataset=dataset, id=ident, channel=channel, channels=channels, order=order,
                total=len(candidates), filtered_count=total, page=page, page_size=page_size,
                candidates=matches[(page-1)*page_size:page*page_size])


def json_safe(value):
    """A malformed artifact must not break an otherwise usable HTTP response."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def make_handler(root):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlsplit(self.path)
            route = url.path
            status = 200
            if route in ('/api/reports', '/api/cases', '/api/case', '/api/case/diagnostics'):
                try:
                    if route == '/api/reports':
                        data = load_versions(root)
                    else:
                        params = parse_qs(url.query, keep_blank_values=True)
                        def param(key, default=None):
                            values = params.get(key, [default])
                            if len(values) != 1 or values[0] is None or (default is None and not values[0]):
                                raise ValueError(f'Missing/invalid parameter: {key}')
                            return values[0]
                        single = 'baseline' not in params and 'comparison' not in params
                        common = dict(root=root, baseline=param('version') if single else param('baseline'),
                                      comparison=None if single else param('comparison'), dataset=param('dataset'))
                        if route == '/api/cases':
                            paging = dict(query=param('q', ''), category=param('category', ''),
                                          score_filter=param('score_filter', 'all'), page=int(param('page', '1')),
                                          page_size=int(param('page_size', '50')))
                            data = (single_cases(root, common['baseline'], common['dataset'], **paging) if single else
                                    compare_cases(**common, filter=param('filter', 'different'),
                                                  score_version=param('score_version', 'comparison'), **paging))
                        elif route == '/api/case':
                            data = case_detail(**common, ident=param('id'))
                        else:
                            data = case_diagnostics_page(**common, ident=param('id'), version=param('version'),
                                                         channel=param('channel', ''), query=param('q', ''),
                                                         page=int(param('page', '1')),
                                                         page_size=int(param('page_size', '50')), order=param('order', 'asc'))
                except FileNotFoundError:
                    status, data = 404, dict(error='Case not found')
                except ValueError as exc:
                    status, data = 400, dict(error=str(exc))
                except OSError:
                    status, data = 500, dict(error='Unable to read case artifacts')
                content = json.dumps(json_safe(data), ensure_ascii=False, allow_nan=False).encode()
                content_type = 'application/json; charset=utf-8'
            elif route in ('/', '/compare_aml_e2e.html'):
                content = Path(__file__).with_name('compare_aml_e2e.html').read_bytes()
                content_type = 'text/html; charset=utf-8'
            else:
                self.send_error(404)
                return
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    summarize = sub.add_parser('summarize', help='Write summary.json and summary.md')
    summarize.add_argument('directory', type=Path)
    serve = sub.add_parser('serve', help='Serve dynamic version comparison on localhost')
    serve.add_argument('directory', type=Path)
    serve.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    try:
        if args.command == 'summarize':
            report = aggregate(args.directory)
            atomic_write(args.directory / 'summary.json', json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
            atomic_write(args.directory / 'summary.md', markdown(report))
            print(f'{args.directory}: {len(report["datasets"])} datasets, score={fmt(report["totals"]["score_mean"], True)}')
        else:
            if not args.directory.is_dir():
                parser.error(f'Not a directory: {args.directory}')
            server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(args.directory.resolve()))
            print(f'Comparison: http://127.0.0.1:{server.server_port}', flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
