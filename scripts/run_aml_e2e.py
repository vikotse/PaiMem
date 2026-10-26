"""Resumable AML replay -> official answer -> official judge, with live accounting."""
from __future__ import annotations

import argparse
import asyncio
import contextvars
import hashlib
import inspect
import json
import math
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import httpx
import portalocker

try:
    from . import aml_pipeline_adapters as a
    from .local_benchmark import chunk_messages
except ImportError:
    import aml_pipeline_adapters as a
    from local_benchmark import chunk_messages

# 主流程：回放消息并检索 → 调用官方回答流程 → 调用官方评测流程 → 汇总报告。
# 排查 badcase 时，用题目 id 关联 input.jsonl、answers.jsonl 和 judged.jsonl；
# 用 item_id 关联 requests.jsonl / errors.jsonl，查看请求状态和重试情况。
ROOT = Path(__file__).resolve().parents[1]
# 为并发任务分别保存阶段、题目 ID 和外层尝试次数，供 HTTP 请求记账使用。
CURRENT = contextvars.ContextVar('request_context', default=('memory', '', 1))


# 运行事件输出到 stderr；本函数不会自动生成日志文件。
def log(event, **fields):
    print(json.dumps({'event': event, **fields}, ensure_ascii=False), file=sys.stderr, flush=True)


# 先写临时文件并刷盘，再原子替换目标文件，避免留下半写入的 checkpoint。
def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=True, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


# 追加一行 JSON 并刷盘，用于请求账本和错误记录。
def append_json(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


# 优先使用本地 tokenizer；未配置时按 UTF-8 字节数粗估 token，估算并非实际用量。
class TokenCounter:
    def __init__(self, tokenizer=None):
        self.tokenizer = None
        if tokenizer:
            from transformers import AutoTokenizer
            # Local files only: token counting must never silently download or hang.
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, local_files_only=True)
        self.method = 'local_chat_template_estimate' if self.tokenizer else 'utf8_bytes_div_3_estimate'

    def text(self, text):
        if self.tokenizer:
            return len(self.tokenizer.encode(str(text), add_special_tokens=False))
        return math.ceil(len(str(text).encode('utf-8')) / 3)

    def messages(self, messages):
        # 有聊天模板时计入角色标记和生成前缀；否则用固定开销近似消息包装成本。
        if self.tokenizer and self.tokenizer.chat_template:
            return len(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
        return 3 + sum(4 + self.text(m.get('role', '')) + self.text(m.get('content', '')) for m in messages)


# requests.jsonl 只保存请求元数据、状态、用量和费用，不保存 prompt 或响应正文。
class Accounting:
    """Append-only request ledger survives retries and restarts, including unknown usage."""
    def __init__(self, path, args):
        self.path, self.args = Path(path), args
        self.totals = dict(input_tokens=0, output_tokens=0, request_seconds=0.0,
                           estimated_cost=0.0, requests=0, unknown_usage=0, estimated_usage=0)
        # 恢复历史累计用量；有 start 却没有 end 的请求记为中断且用量未知。
        starts, ends = {}, set()
        if self.path.exists():
            # Only repair a torn final append; corruption in complete lines is fatal.
            data = self.path.read_bytes()
            lines = data.splitlines(keepends=True)
            offset = 0
            for index, line in enumerate(lines):
                try:
                    row = json.loads(line)
                except ValueError:
                    if index != len(lines) - 1 or line.endswith(b'\n'):
                        raise
                    # 仅截掉末尾未写完的 JSON；完整行损坏时直接报错，避免静默丢失账本。
                    with self.path.open('r+b') as stream:
                        stream.truncate(offset)
                    break
                offset += len(line)
                # 按请求 ID 配对开始/结束记录，只将结束记录计入累计用量。
                if row['event'] == 'start':
                    starts[row['request_id']] = row
                else:
                    ends.add(row['request_id'])
                    self.add(row)
            if self.path.stat().st_size and not self.path.read_bytes().endswith(b'\n'):
                with self.path.open('ab') as stream:
                    stream.write(b'\n')
            for ident, row in starts.items():
                if ident not in ends:
                    self.finish(row, 0, None, None, 'interrupted_unknown_usage', 'unknown')

    # 未知用量不累加 token/费用数值，另计 unknown_usage，保留统计的不确定性。
    def add(self, row):
        self.totals['requests'] += 1
        for key in ('input_tokens', 'output_tokens', 'request_seconds', 'estimated_cost'):
            self.totals[key] += row.get(key) or 0
        self.totals['unknown_usage'] += row['usage_source'] == 'unknown'
        self.totals['estimated_usage'] += row['usage_source'] == 'estimated'

    # start/end 共用 request_id；同一题的多次 HTTP 调用各有独立的 request_id。
    def start(self, stage, ident, attempt, body, counter):
        estimated = counter.messages(body.get('messages', []))
        price = getattr(self.args, stage + '_price_in', None)
        row = dict(event='start', request_id=uuid.uuid4().hex, stage=stage, item_id=ident,
                   attempt=attempt, model=body.get('model'), input_tokens_estimate=estimated,
                   token_method=counter.method,
                   input_cost_estimate=None if price is None else estimated * price / 1e6)
        append_json(self.path, row)
        log('request_start', **{k: v for k, v in row.items() if k != 'event'}, totals=self.totals)
        return row

    def finish(self, start, elapsed, input_tokens, output_tokens, status, source):
        stage = start['stage']
        price_in = getattr(self.args, stage + '_price_in', None)
        price_out = getattr(self.args, stage + '_price_out', None)
        # 单价单位为每百万 token；缺少用量或任一单价时，费用保持未知。
        cost = None if input_tokens is None or price_in is None or price_out is None else (
            input_tokens * price_in + output_tokens * price_out) / 1e6
        row = dict(event='end', request_id=start['request_id'], stage=stage,
                   item_id=start['item_id'], attempt=start['attempt'], status=status,
                   input_tokens=input_tokens, output_tokens=output_tokens,
                   request_seconds=elapsed, estimated_cost=cost, usage_source=source)
        append_json(self.path, row)
        self.add(row)
        log('request_end', **{k: v for k, v in row.items() if k != 'event'}, totals=self.totals)


# 长请求等待期间定期输出心跳，帮助区分仍在等待与进程无响应。
async def heartbeat(stage, ident, started, interval, accounting=None):
    while True:
        await asyncio.sleep(interval)
        log('waiting', stage=stage, item_id=ident, elapsed_seconds=round(time.monotonic()-started, 2),
            totals=accounting.totals if accounting else None)


# 只替换官方 pipeline 使用的 HTTP 客户端，追踪 answer / judge 的模型请求。
# memory 的 /add、/search 请求不计入 requests.jsonl 和模型费用汇总。
def install_transport(pipeline, args, accounting, counters):
    class TrackedClient(httpx.AsyncClient):
        def __init__(self, *pos, **kw):
            kw['timeout'] = httpx.Timeout(args.call_timeout, connect=min(10, args.call_timeout))
            super().__init__(*pos, **kw)

        async def post(self, url, **kw):
            stage, ident, attempt = CURRENT.get()
            counter = counters[stage]
            body = dict(kw.get('json') or {})
            # 显式配置输出上限时覆盖官方请求；未配置则保留官方值。
            cap = getattr(args, stage + '_max_tokens')
            if cap is not None:
                body['max_tokens'] = cap
            kw['json'] = body
            start = accounting.start(stage, ident, attempt, body, counter)
            began = time.monotonic()
            task = asyncio.create_task(heartbeat(stage, ident, began, args.heartbeat, accounting))
            response = None
            try:
                response = await asyncio.wait_for(super().post(url, **kw), args.call_timeout)
                response.raise_for_status()
                payload = response.json()
                # 优先采用 API 返回用量；缺失时估算输入及 content / reasoning_content 的输出用量。
                # reasoning_content 仅参与计数，不会在此保存。
                usage = payload.get('usage') or {}
                if all(isinstance(usage.get(k), int) and usage[k] >= 0 for k in ('prompt_tokens', 'completion_tokens')):
                    inp, out, source = usage['prompt_tokens'], usage['completion_tokens'], 'api'
                else:
                    message = payload['choices'][0]['message']
                    inp = start['input_tokens_estimate']
                    out = counter.text(message.get('content') or '') + counter.text(message.get('reasoning_content') or '')
                    source = 'estimated'
                accounting.finish(start, time.monotonic()-began, inp, out, 'ok', source)
                return response
            # 包括超时和任务取消在内的失败均补写结束记录，再向上抛出异常。
            except BaseException as error:
                # A timeout does not mean the provider billed zero tokens.
                status = f'http_{response.status_code}' if response is not None and response.is_error else type(error).__name__
                accounting.finish(start, time.monotonic()-began, None, None, status, 'unknown')
                raise
            finally:
                # 无论成功或失败都停止心跳，并等待取消完成，避免遗留后台任务。
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    # 为当前 pipeline 创建独立的 httpx 命名空间，避免改动全局 httpx.AsyncClient。
    pipeline.httpx = SimpleNamespace(**{name: getattr(httpx, name) for name in dir(httpx) if not name.startswith('_')})
    pipeline.httpx.AsyncClient = TrackedClient


# 调用 AML memory 服务；这里只输出开始、结束和心跳，不落盘原始请求/响应。
async def memory_request(client, args, endpoint, payload):
    ident = payload.get('request_id', payload.get('user_id'))
    began = time.monotonic()
    log('memory_start', endpoint=endpoint, item_id=ident, input_bytes=len(json.dumps(payload).encode()))
    task = asyncio.create_task(heartbeat(endpoint, ident, began, args.heartbeat))
    try:
        headers = {'Authorization': f'Bearer {args.api_key}'} if args.api_key else {}
        response = await asyncio.wait_for(client.post(args.base_url.rstrip('/') + endpoint,
                                         json=payload, headers=headers), args.timeout)
        response.raise_for_status()
        return response.json()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        log('memory_end', endpoint=endpoint, item_id=ident, elapsed_seconds=time.monotonic()-began)


# input.jsonl 是完成 replay 后的逐题输入，包含问题、检索上下文和评测所需字段。
# 文件已存在时直接复用，不重新执行 /add 或 /search。
async def replay(args, contract, pipeline, work):
    path = work / 'input.jsonl'
    if path.exists():
        return a.read_rows(path)
    if args.skip_replay:
        raise ValueError('--skip-replay requires an existing output/input.jsonl')
    # official 输入直接交给官方流程；normalized 输入需要构建适配后的逐题数据。
    if args.input_format == 'official':
        items = a.read_rows(args.input)
        items = items[:args.limit] if args.limit else items
        for index, item in enumerate(items):
            item['id'] = pipeline.row_id(item, index) if hasattr(pipeline, 'row_id') else str(item['id'])
    else:
        # ScriptMem 的本地题号需映射为官方 qa_id，才能关联官方参考数据。
        gold = a.scriptmem_gold_map(pipeline, Path(args.scriptmem_data_dir)) if contract == a.CONTRACT_SCRIPTMEM else {}
        records = a.read_rows(args.input)
        # record ID 同时参与 memory user_id、Add checkpoint 和题目 ID；重复值会让两条记录
        # 共享状态。必须在发出任何 memory 请求前失败，不能通过并发串行化来掩盖数据错误。
        record_ids = [str(record['id']) for record in records]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError('duplicate normalized record IDs')
        plans = []
        planned_questions = 0
        for record in records:
            if args.limit and planned_questions >= args.limit:
                break
            # limit 限制题目总数；最后一条记录只取剩余额度内的问题。
            questions = (
                record['questions'][:args.limit-planned_questions]
                if args.limit
                else record['questions']
            )
            if questions:
                plans.append((record, questions))
                planned_questions += len(questions)

        async def replay_record(client, record, questions):
            record_items = []
            rid = str(record['id'])
            # run_label 与原始记录 ID 组成 memory 用户 ID，隔离不同运行的数据。
            uid = f'aml-e2e:{args.run_label}:{rid}'
            if args.mode == 'normal':
                # 同一用户内部保持 Add 顺序，并在全部 Add 完成后再执行 Search。
                for index, messages in enumerate(chunk_messages(record['messages'])):
                    # adds/<sha256(记录ID:批次序号)>.json 只标记成功，不保存 /add 响应。
                    marker = work / 'adds' / (hashlib.sha256(f'{rid}:{index}'.encode()).hexdigest() + '.json')
                    if marker.exists():
                        continue
                    result = await memory_request(client, args, '/add', dict(request_id=f'{uid}:{index}',
                        user_id=uid, session_id=str(record.get('session_id', rid)), messages=messages))
                    # HTTP 成功还需业务 success=true；确认后才标记该批次完成。
                    if result.get('success') is not True:
                        raise ValueError('memory Add did not return success=true')
                    atomic_json(marker, {'done': True})
            for question in questions:
                qid = f"{rid}:{question['id']}"
                if contract == a.CONTRACT_SCRIPTMEM:
                    qid = gold[(rid, int(str(question['id']).rsplit('-q', 1)[-1]))]
                # replay/<sha256(题目ID)>.json 保存单题适配输入，支持回放中断后继续。
                marker = work / 'replay' / (hashlib.sha256(qid.encode()).hexdigest() + '.json')
                if marker.exists():
                    record_items.append(json.loads(marker.read_text()))
                    continue
                # normal：检索 memory；oracle：使用题目提供的证据；empty：使用空上下文。
                retrieved = []
                if args.mode == 'normal':
                    # 用当前问题检索该记录对应用户的记忆，选择题同时传入选项。
                    payload = dict(query=question['question'], user_id=uid, top_k=args.top_k)
                    if question.get('options'):
                        payload['options'] = question['options']
                    if args.save_retrieval_diagnostics:
                        diagnostic_dir = work / 'search_diagnostics'
                        diagnostic_dir.mkdir(exist_ok=True)
                        diagnostic_path = diagnostic_dir / marker.name
                        previous_diagnostic = diagnostic_path.stat() if diagnostic_path.exists() else None
                        payload['debug_output_path'] = str(diagnostic_path)
                    result = await memory_request(client, args, '/search', payload)
                    if args.save_retrieval_diagnostics:
                        current_diagnostic = diagnostic_path.stat() if diagnostic_path.is_file() else None
                        if current_diagnostic is None or (previous_diagnostic is not None and
                                (current_diagnostic.st_ino, current_diagnostic.st_mtime_ns) ==
                                (previous_diagnostic.st_ino, previous_diagnostic.st_mtime_ns)):
                            raise ValueError('Memory service did not write search diagnostics; update the local service')
                    retrieved = result['data']
                    # 可选保存完整返回；题目及其他请求参数已在 replay/ 和 manifest 中。
                    if args.save_search_results:
                        search_dir = work / 'search'
                        search_dir.mkdir(exist_ok=True)
                        atomic_json(search_dir / marker.name, result)
                    # 拼接后的文本仍沿用原流程，完整返回不进入官方模型输入。
                    context = '\n'.join(row['content'] for row in retrieved)
                elif args.mode == 'oracle':
                    context = '\n'.join(str(v) for v in question.get('evidence_texts', []))
                    retrieved = [{'content': context}] if context else []
                else:
                    context = ''
                # 适配器补充 x_record_id、x_question_id、召回数及证据诊断等字段；
                # 各 benchmark 还可能保留额外检索字段，具体由适配器决定。
                item = a.build_item(question, rid, qid, context, retrieved, contract)
                atomic_json(marker, item)
                record_items.append(item)
            return record_items

        items = []
        async with httpx.AsyncClient(timeout=args.timeout) as client:
            # 固定数量的 worker 动态领取不同 record，避免慢 record 阻塞下一整批。
            ordered_results = [None] * len(plans)
            next_plan = 0

            async def worker():
                nonlocal next_plan
                while next_plan < len(plans):
                    index = next_plan
                    next_plan += 1
                    record, questions = plans[index]
                    ordered_results[index] = await replay_record(client, record, questions)

            await asyncio.gather(
                *(worker() for _ in range(min(args.replay_concurrency, len(plans))))
            )
            for record_items in ordered_results:
                items.extend(record_items)
    # ASCII escaping avoids official splitlines() splitting U+2028/U+0085 in content.
    # 汇总全部逐题输入后原子替换；避免下次续跑误把不完整输入当作 replay 已完成。
    a.write_rows(path.with_suffix('.tmp'), items)
    os.replace(path.with_suffix('.tmp'), path)
    return items


# 校验官方输出是否有效；答案有效或评测完成，不表示该题得分为满分。
def valid_result(row, stage, contract):
    if stage == 'answer':
        return bool(str(row.get('generated_answer', row.get('model_output', ''))).strip())
    if contract == a.CONTRACT_CLBENCH and not row.get('rubric_clbench_requirement_status'):
        return False  # official failure payload is a zero, not a valid judgement
    return a.row_score(row, contract) is not None


# answer/<hash>.json 和 judge/<hash>.json 保存单题成功结果，hash 为题目 ID 的 SHA-256。
# 已有 checkpoint 的题目直接跳过；阶段结束后合并为 answers.jsonl / judged.jsonl。
async def run_stage(stage, items, pipeline, contract, args, work):
    directory = work / stage
    # 信号量限制同时处理的单题尝试数，每个尝试可能包含官方内部多次调用。
    semaphore = asyncio.Semaphore(args.model_concurrency)
    failures = []

    async def one(item):
        ident = str(item['id'])
        key = hashlib.sha256(ident.encode()).hexdigest()
        checkpoint = directory / f'{key}.json'
        if checkpoint.exists():
            return
        answer_path = work / 'answer' / f'{key}.json'
        # 缺少成功答案时不执行该题评测，并将题目记录为 judge 阶段失败。
        if stage == 'judge' and not answer_path.exists():
            failures.append(ident)
            return
        async with semaphore:
            # retries 是额外的外层尝试次数；官方 pipeline 内部还可能有 HTTP 重试。
            for attempt in range(1, args.retries + 2):
                # 将本次题目上下文传递给请求记账层，确保并发请求归属正确。
                token = CURRENT.set((stage, ident, attempt))
                try:
                    # tmp/ 中的单题输入、答案和输出会在本次尝试退出时清理，不能用于历史追踪。
                    with tempfile.TemporaryDirectory(dir=work / 'tmp') as temporary:
                        temp = Path(temporary)
                        a.write_rows(temp/'input.jsonl', [item])
                        # 将单题文件路径及生成参数包装为官方入口需要的参数对象。
                        ns = a.official_ns(input=str(temp/'input.jsonl'), output=str(temp/'out.jsonl'),
                            answers=str(temp/'answers.jsonl'), timeout=args.call_timeout,
                            max_tokens=args.answer_max_tokens or (512 if contract == a.CONTRACT_BEAM else 256),
                            judge_max_tokens=args.judge_max_tokens or 1024, max_retries=1)
                        # 评测阶段读取此前的成功答案，作为官方评测入口的输入。
                        if stage == 'judge':
                            ns.max_tokens = args.judge_max_tokens or 256
                            a.write_rows(temp/'answers.jsonl', [json.loads(answer_path.read_text())])
                        # 根据阶段和评测契约选择入口：生成答案、选择题评测或通用评测。
                        if stage == 'answer':
                            entry = pipeline.answer
                        elif contract == a.CONTRACT_PERSONA_V2:
                            entry = pipeline.evaluate_mcq
                        else:
                            entry = pipeline.evaluate
                        # 兼容同步和异步官方入口，异步返回值需等待执行完成。
                        async def invoke():
                            result = entry(ns)
                            if inspect.isawaitable(result):
                                await result
                        # item_timeout 限制整个单题尝试；call_timeout 限制单次模型 HTTP 请求。
                        await asyncio.wait_for(invoke(), args.item_timeout)
                        rows = a.read_rows(temp/'out.jsonl')
                        # 必须恰好返回本题的一条有效结果，才允许写入成功 checkpoint。
                        if len(rows) != 1 or a.row_id_of(rows[0], contract) != ident or not valid_result(rows[0], stage, contract):
                            raise ValueError('official pipeline returned missing/invalid result')
                        atomic_json(checkpoint, rows[0])
                        log('item_done', stage=stage, item_id=ident, attempt=attempt)
                        return
                except Exception as error:
                    # Never log raw exception bodies: providers may include credentials/prompts.
                    log('item_failed', stage=stage, item_id=ident, attempt=attempt, error=type(error).__name__)
                    # errors.jsonl 仅在失败时产生，记录阶段、题目、尝试次数及异常类型。
                    append_json(work/'errors.jsonl', dict(stage=stage, item_id=ident, attempt=attempt, error=type(error).__name__))
                    # 尚有尝试次数时按 1、2、4、8 秒退避，降低连续失败时的请求压力。
                    if attempt <= args.retries:
                        await asyncio.sleep(min(2 ** (attempt-1), 8))
                finally:
                    # 恢复进入尝试前的上下文，防止后续请求沿用本题信息。
                    CURRENT.reset(token)
            # 所有外层尝试耗尽仍未成功，加入阶段失败列表。
            failures.append(ident)

    # Bounded task batches avoid creating tens of thousands of waiting coroutines.
    # 每批最多创建 model_concurrency 个任务；等本批结束后再启动下一批，控制任务数量。
    for offset in range(0, len(items), args.model_concurrency):
        await asyncio.gather(*(one(item) for item in items[offset:offset+args.model_concurrency]))
    # 按原输入顺序收集成功 checkpoint，包含本次完成及之前续跑已完成的结果。
    rows = []
    for item in items:
        path = directory / (hashlib.sha256(str(item['id']).encode()).hexdigest()+'.json')
        if path.exists():
            rows.append(json.loads(path.read_text()))
    a.write_rows(work / ('answers.jsonl' if stage == 'answer' else 'judged.jsonl'), rows)
    return rows, failures


# 阶段顺序仍为 replay、answer、judge；回放与模型阶段分别受各自并发参数控制。
async def run(args, pipeline, contract, work):
    began = time.monotonic()
    items = await replay(args, contract, pipeline, work)
    # 题目 ID 用于关联各阶段和命名 checkpoint，必须唯一；随后校验各契约必需字段。
    ids = [str(item['id']) for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate item IDs')
    a.validate_items(contract, items, args)
    if contract == a.CONTRACT_SCRIPTMEM:
        # 确认所有 ScriptMem 输入题目都能在官方 gold 集中找到。
        gold_ids = {row['qa_id'] for row in pipeline.load_gold_records(Path(args.scriptmem_data_dir))}
        if not set(ids) <= gold_ids:
            raise ValueError('ScriptMem input IDs do not match official gold data')
    # dry-run 仍会执行或复用 replay 并校验输入，但不进入回答和评测阶段。
    if args.dry_run:
        log('dry_run', questions=len(items), contract=contract, input=str(work/'input.jsonl'))
        return 0
    # 恢复账本、为两个模型阶段初始化计数器，然后安装请求追踪客户端。
    accounting = Accounting(work/'requests.jsonl', args)
    counters = {stage: TokenCounter(getattr(args, stage+'_tokenizer')) for stage in ('answer', 'judge')}
    install_transport(pipeline, args, accounting, counters)
    # 先完成全部回答，再进入评测；没有成功答案的题目不会调用评测入口。
    answers, failed_answers = await run_stage('answer', items, pipeline, contract, args, work)
    if contract == a.CONTRACT_SCRIPTMEM:
        # ScriptMem 使用官方本地评测：输出提交文件和完整 gold 集汇总，
        # 再筛选本次已回答题目写入 judged.jsonl，不生成逐题 judge checkpoint。
        submission = work/'submission.json'
        pipeline.convert_jsonl_answers(a.official_ns(answers=str(work/'answers.jsonl'), output=str(submission)))
        full_summary, all_rows = pipeline.evaluate_official(Path(args.scriptmem_data_dir), submission)
        answered = {a.row_id_of(row, contract) for row in answers}
        judged = [dict(row, id=row['qa_id']) for row in all_rows if row['qa_id'] in answered]
        failed_judges = list(set(ids) - {row['qa_id'] for row in judged})
        a.write_rows(work/'judged.jsonl', judged)
        atomic_json(work/'scriptmem_full_gold_summary.json', full_summary)
    else:
        judged, failed_judges = await run_stage('judge', items, pipeline, contract, args, work)
    # 将各 benchmark 不同的评分字段映射为统一数值，均分仅计算已有评测结果。
    scores = [a.row_score(row, contract) for row in judged]
    # complete 判断所有题目是否执行完成；score_mean 才反映答案质量。
    complete = len(answers) == len(items) and len(scores) == len(items) and not failed_judges
    summary = dict(complete=complete, contract=contract, questions=len(items), answered=len(answers), judged=len(judged),
        score_mean=sum(scores)/len(scores) if scores else None, failed_answers=failed_answers,
        failed_judges=failed_judges, usage=accounting.totals, wall_seconds_this_run=time.monotonic()-began,
        cost_currency=args.currency, prices_configured={s: getattr(args,s+'_price_in') is not None for s in ('answer','judge')},
        cost_note='Known/estimated returned usage only; unknown_usage requests may have additional provider charges.')
    # report.json 汇总本次结果和累计请求用量；wall_seconds_this_run 仅计本次进程耗时。
    atomic_json(work/'report.json', summary)
    log('report', **summary)
    return 0 if complete else 1


# 参数分为输入/回放、运行控制及 answer / judge 模型配置；模型配置可由环境变量提供。
def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='New run directory; same configuration resumes automatically')
    p.add_argument('--pipeline', type=Path, default=ROOT/'agent-memory-leaderboard/data/locomo-refined/pipeline.py')
    p.add_argument('--input-format', choices=['normalized', 'official'], default='normalized')
    p.add_argument('--mode', choices=['normal','oracle','empty'], default='normal')
    p.add_argument('--base-url', default='http://127.0.0.1:8000')
    p.add_argument('--api-key', default=os.environ.get('MEMORY_API_KEY'))
    p.add_argument('--run-label', default=None)
    p.add_argument('--top-k', type=int, default=100)
    p.add_argument('--limit', type=int)
    p.add_argument('--concurrency', type=int, default=2,
                   help='Backward-compatible default for replay and model concurrency')
    p.add_argument('--replay-concurrency', type=int,
                   help='Concurrent normalized replay records/users; defaults to --concurrency')
    p.add_argument('--model-concurrency', type=int,
                   help='Concurrent answer/judge items; defaults to --concurrency')
    p.add_argument('--retries', type=int, default=1, help='Extra attempts per item (official internal retries also bounded by item timeout)')
    p.add_argument('--timeout', type=float, default=120, help='Total deadline per memory request')
    p.add_argument('--call-timeout', type=float, default=180, help='Total deadline per LLM HTTP request')
    p.add_argument('--item-timeout', type=float, default=600, help='Deadline per item attempt including official retries')
    p.add_argument('--heartbeat', type=float, default=10)
    p.add_argument('--currency', default='CNY')
    p.add_argument('--skip-replay', action='store_true')
    p.add_argument('--save-search-results', action='store_true',
                   help='Save full /search responses to search/<sha256(item_id)>.json during new normal-mode replay; existing replay is reused')
    p.add_argument('--save-retrieval-diagnostics', action='store_true',
                   help='Ask a local memory service to write every candidate and route score to search_diagnostics/<sha256(item_id)>.json')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--personamem-mode', choices=['mcq'], default='mcq')
    p.add_argument('--scriptmem-data-dir', type=Path, default=ROOT/'agent-memory-leaderboard/downloads/benchmarks/scriptmem/data/raw')
    for stage in ('answer', 'judge'):
        for flag, env in [('base-url','API_BASE'),('model','MODEL'),('api-key','API_KEY')]:
            p.add_argument(f'--{stage}-{flag}', default=os.environ.get(f'{stage.upper()}_{env}', ''))
        p.add_argument(f'--{stage}-price-in', type=float, help='Currency units per 1M input tokens')
        p.add_argument(f'--{stage}-price-out', type=float, help='Currency units per 1M output tokens')
        p.add_argument(f'--{stage}-max-tokens', type=int, help='Explicit API output cap; omitted preserves official request')
        p.add_argument(f'--{stage}-tokenizer', help='Local Hugging Face tokenizer directory, no downloads')
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    # 新参数未提供时完全沿用原 --concurrency 行为，旧命令无需修改。
    if args.replay_concurrency is None:
        args.replay_concurrency = args.concurrency
    if args.model_concurrency is None:
        args.model_concurrency = args.concurrency
    # 先校验数量、超时等运行参数，避免无效配置进入回放或模型调用。
    for name in ('top_k','concurrency','replay_concurrency','model_concurrency',
                 'timeout','call_timeout','item_timeout','heartbeat'):
        if not math.isfinite(getattr(args,name)) or getattr(args,name) <= 0:
            p.error(f'{name} must be positive and finite')
    if args.retries < 0 or (args.limit is not None and args.limit <= 0):
        p.error('retries must be nonnegative and limit positive')
    for stage in ('answer','judge'):
        # 输入/输出单价必须同时设置且非负，输出 token 上限必须为正。
        prices = [getattr(args,stage+'_price_'+side) for side in ('in','out')]
        if (prices[0] is None) != (prices[1] is None) or any(v is not None and (v < 0 or not math.isfinite(v)) for v in prices):
            p.error(f'{stage}: specify both nonnegative finite input/output prices')
        cap = getattr(args,stage+'_max_tokens')
        if cap is not None and cap <= 0:
            p.error('max-tokens must be positive')
        for field, env in [('base_url','API_BASE'),('model','MODEL'),('api_key','API_KEY')]:
            os.environ[f'{stage.upper()}_{env}'] = getattr(args,stage+'_'+field)
    # 先设置模型环境变量再加载官方模块，使导入时读取的配置使用本次参数。
    pipeline = a.load_pipeline(args.pipeline.resolve())
    # api_config may already be cached by another imported pipeline.
    # 再覆盖模块内的配置常量，处理共享 api_config 已被其他模块导入缓存的情况。
    for stage in ('answer','judge'):
        for suffix in ('API_BASE','API_KEY','MODEL'):
            setattr(pipeline, f'{stage.upper()}_{suffix}', os.environ[f'{stage.upper()}_{suffix}'])
    # 安装官方脚本路径兼容层，并识别 benchmark 契约，决定输入字段和评测方式。
    a.install_path_shim(pipeline)
    contract = a.detect_contract(pipeline)
    # 正式运行必须提供所需模型配置；本地评测契约只要求 answer 模型。
    if not args.dry_run:
        for stage in (('answer','judge') if contract in a.LLM_JUDGE_CONTRACTS else ('answer',)):
            if not getattr(args,stage+'_base_url') or not getattr(args,stage+'_model'):
                p.error(f'{stage} model and base URL are required (flags or environment)')
    work = args.output.resolve()
    work.mkdir(parents=True, exist_ok=True)
    # run.lock 的文件内容无业务意义，依靠进程文件锁防止同目录同时运行。
    with (work/'run.lock').open('a') as lock:
        try:
            portalocker.lock(lock, portalocker.LOCK_EX | portalocker.LOCK_NB)
        except portalocker.exceptions.LockException:
            p.error('another process is using this output directory')
        for name in ('answer','judge','adds','replay','tmp'):
            (work/name).mkdir(exist_ok=True)
        # 默认由输出目录推导稳定标签，使同目录续跑复用相同的 memory 用户 ID。
        args.run_label = args.run_label or hashlib.sha256(str(work).encode()).hexdigest()[:16]
        # manifest.json 不记录 API key，也不记录可在续跑时调整的运行控制参数。
        # 其余配置及输入/官方 pipeline 哈希必须一致，避免混用不同实验的 checkpoint。
        excluded = {'api_key','answer_api_key','judge_api_key','skip_replay','save_search_results','save_retrieval_diagnostics','dry_run','concurrency','replay_concurrency','model_concurrency','retries','timeout','call_timeout','item_timeout','heartbeat'}
        config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in excluded}
        config['input_sha256'] = hashlib.sha256(args.input.read_bytes()).hexdigest()
        config['pipeline_sha256'] = hashlib.sha256(args.pipeline.read_bytes()).hexdigest()
        manifest = work/'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text()) != config:
            p.error('run configuration changed; choose a new --output directory to avoid mixing results')
        atomic_json(manifest, config)
        # 持有目录锁直到异步主流程结束，随后退出上下文自动释放锁。
        return asyncio.run(run(args, pipeline, contract, work))


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log('interrupted', message='Completed items saved; rerun the same command to resume.')
        raise SystemExit(130)
