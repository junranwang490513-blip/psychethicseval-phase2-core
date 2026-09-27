"""Frozen full-run pipeline: bounded dispatch, append-only facts, bounded transport retries."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections import deque, Counter
from dataclasses import asdict
from pathlib import Path
import json
import re
import errno
import urllib.error
import sqlite3
import threading
import time

from .contracts import canonical, digest, failure_fact, validate_answer_response, ClassifierCall
from .identity_classifier_v13 import classify_identity_v13
from .oeq_materialization_v3 import materialize_oeq_request_v3, validate_materialized_request_v3


def retryable_transport(error):
    cause = getattr(error, '__cause__', None)
    if cause is not None and cause is not error:
        return retryable_transport(cause)
    if isinstance(error, urllib.error.HTTPError):
        return error.code == 429 or 500 <= error.code <= 599
    if isinstance(error, urllib.error.URLError):
        if isinstance(error.reason, (urllib.error.HTTPError, urllib.error.URLError)):
            return retryable_transport(error.reason)
        return True
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    return isinstance(error, OSError) and error.errno in {
        errno.ECONNRESET, errno.ECONNREFUSED, errno.ECONNABORTED, errno.ETIMEDOUT,
        errno.EHOSTUNREACH, errno.ENETUNREACH, errno.EPIPE}


def transport_fact(error):
    root = error
    seen = set()
    while getattr(root, '__cause__', None) is not None and id(root) not in seen:
        seen.add(id(root)); root = root.__cause__
    reason = getattr(root, 'reason', root)
    # Do not serialize exception strings: URLs and proxy errors can contain secrets.
    fact = dict(failure_fact(error), cause_type=type(root).__name__, reason_type=type(reason).__name__,
        sanitized_reason='transport exception; original message omitted to protect credentials',
        uncertain_remote_processing=True)
    if isinstance(root, urllib.error.HTTPError): fact['http_status'] = root.code
    if isinstance(reason, OSError): fact['errno'] = reason.errno
    return fact


class FullExecutionStore:
    def __init__(self, path, manifest_id):
        self.path = Path(path).resolve()
        if self.path.name == 'runs.sqlite3':
            raise ValueError('production database forbidden')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.identity = {'schema': 'full-checkpoint-v1', 'manifest_id': manifest_id}
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        tables = {r[0] for r in self.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if tables and tables != {'full_identity', 'full_events'}:
            raise ValueError('dedicated full checkpoint required')
        self.db.execute('CREATE TABLE IF NOT EXISTS full_identity (data TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS full_events (sequence INTEGER PRIMARY KEY, kind TEXT NOT NULL, item TEXT NOT NULL, data TEXT NOT NULL, previous TEXT NOT NULL, hash TEXT NOT NULL)')
        self.db.execute('CREATE UNIQUE INDEX IF NOT EXISTS full_item_kind ON full_events(item,kind)')
        rows = self.db.execute('SELECT data FROM full_identity').fetchall()
        if not rows:
            self.db.execute('INSERT INTO full_identity VALUES (?)', (canonical(self.identity),))
        elif rows != [(canonical(self.identity),)]:
            raise ValueError('manifest identity drift')
        for table in ('full_identity', 'full_events'):
            for op in ('UPDATE', 'DELETE'):
                self.db.execute(f"CREATE TRIGGER IF NOT EXISTS {table}_{op} BEFORE {op} ON {table} BEGIN SELECT RAISE(ABORT,'append-only checkpoint'); END")
        self.db.commit()
        events = self.events()
        self.sequence = len(events)
        self.previous = events[-1]['hash'] if events else digest(self.identity)

    def events(self):
        with self.lock:
            if self.db.execute('SELECT data FROM full_identity').fetchall() != [(canonical(self.identity),)]:
                raise ValueError('manifest identity drift')
            rows = self.db.execute('SELECT sequence,kind,item,data,previous,hash FROM full_events ORDER BY sequence').fetchall()
        result, previous = [], digest(self.identity)
        for seq, kind, item, data, parent, sha in rows:
            event = dict(sequence=seq, kind=kind, item=item, data=json.loads(data), previous=parent)
            if seq != len(result)+1 or parent != previous or digest(event) != sha:
                raise ValueError('event chain drift')
            result.append(dict(event, hash=sha)); previous = sha
        return result

    def append(self, kind, item, data):
        with self.lock:
            event = dict(sequence=self.sequence+1, kind=kind, item=item, data=data, previous=self.previous)
            sha = digest(event)
            self.db.execute('INSERT INTO full_events VALUES (?,?,?,?,?,?)',
                (event['sequence'], kind, item, canonical(data), self.previous, sha))
            self.db.commit()
            self.sequence += 1; self.previous = sha

    def close(self):
        self.db.close()


def item_key(row):
    return row['kind'] + ':' + str(row['item_id'])


def prediction_for(row, response, payload):
    if not validate_answer_response(response, payload['model']):
        raise ValueError('answer response integrity')
    choice = response['response']['choices'][0]
    if choice['finish_reason'] != 'stop':
        raise ValueError('answer truncation')
    content = choice['message']['content']
    if row['kind'] == 'oeq':
        if len(content.strip()) < 40:
            raise ValueError('empty or short OEQ')
        return content
    from .mcq_prediction_v3 import parse_mcq_prediction_v3
    return parse_mcq_prediction_v3(content, option_count=len(row['source_item']['options']))


class ReplayIdentity:
    def __init__(self, request, call):
        self.request, self.call = request, call
    def complete(self, payload):
        if payload != self.request['payload']:
            raise ValueError('identity request drift')
        return self.call


def validate_identity_transport(call):
    try:
        raw = json.loads(call.usage['provider_raw_response'])
        choices = raw['choices']
        if (raw['model'] != call.response_model or len(choices) != 1
            or choices[0]['message']['content'] != call.output_text
            or choices[0]['finish_reason'] != call.finish_reason
            or raw.get('usage', {}) != {k:v for k,v in call.usage.items() if k != 'provider_raw_response'}):
            raise ValueError('identity raw envelope drift')
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise ValueError('identity raw envelope invalid') from error


def _identity_version(row):
    version = row['identity_request']['contract']['classifier_id'].removeprefix('identity-classifier-')
    if version != 'v13': raise ValueError('full identity version')
    return version

def _classify(row, call):
    _identity_version(row)
    return classify_identity_v13(question=row['source_item']['question'], provider=ReplayIdentity(row['identity_request'], call))

def _materialize(row, call, retriever):
    validate_identity_transport(call)
    request = materialize_oeq_request_v3(question=row['source_item']['question'],
        identity_provider=ReplayIdentity(row['identity_request'], call), retriever=retriever,
        arm='rag-v3', identity_version=_identity_version(row))
    if not validate_materialized_request_v3(request):
        raise ValueError('materialization invalid')
    return request


def _audit(manifest, store, retriever=None):
    events = store.events(); groups = {}; counts = Counter(e['kind'] for e in events)
    expected = {item_key(r): r for r in manifest['inputs']}
    if len(expected) != len(manifest['inputs']):
        raise ValueError('duplicate input')
    for e in events:
        if e['item'] not in expected and e['item'] != '__batch__':
            raise ValueError('unknown event item')
        groups.setdefault(e['item'], {})[e['kind']] = e['data']
    predictions = {}
    for key, row in expected.items():
        g = groups.get(key, {})
        if counts['execution_started']:
            for stage in ('identity', 'answer'):
                attempts = sorted(int(m.group(1)) for k in g
                    if (m := re.fullmatch(stage+r'_attempt_(\d+)_started', k)))
                if attempts != list(range(1, len(attempts)+1)) or len(attempts) > manifest['envelope']['retry_count']+1:
                    raise ValueError('attempt sequence or per-call retry limit drift')
                for number in attempts:
                    started = g[f'{stage}_attempt_{number}_started']
                    parent = g[stage+'_started']['request']
                    if started['payload'] != parent['payload'] or started['payload_sha256'] != digest(parent['payload']):
                        raise ValueError('attempt payload drift')
                    if number > 1 and f'{stage}_attempt_{number-1}_transport_error' not in g:
                        raise ValueError('retry without prior transport error')
                    returned = g.get(f'{stage}_attempt_{number}_returned')
                    if returned is not None and returned != g.get(stage+'_received'):
                        raise ValueError('raw returned attempt binding drift')
        if 'identity_started' in g and g['identity_started']['request'] != row['identity_request']:
            raise ValueError('identity receipt drift')
        if 'answer_started' not in g:
            continue
        request = g['answer_started']['request']
        if row['kind'] == 'mcq':
            if request != {'payload': row['payload']}:
                raise ValueError('MCQ request drift')
        else:
            if 'identity_received' not in g or 'identity_started' not in g:
                raise ValueError('missing identity provenance')
            if not validate_materialized_request_v3(request):
                raise ValueError('OEQ request invalid')
            call = ClassifierCall(**g['identity_received']['call'])
            validate_identity_transport(call)
            decision = _classify(row, call)
            if decision != request['identity'] or g.get('identity_accepted', {}).get('decision') != decision:
                raise ValueError('raw identity binding drift')
            if request['identity']['request'] != row['identity_request']:
                raise ValueError('OEQ identity parent drift')
            if retriever is not None and _materialize(row, ClassifierCall(**g['identity_received']['call']), retriever) != request:
                raise ValueError('OEQ replay drift')
        if 'answer_result' in g:
            if 'answer_received' not in g:
                raise ValueError('missing raw answer')
            prediction = prediction_for(row, g['answer_received']['response'], request['payload'])
            if g['answer_result']['prediction'] != prediction:
                raise ValueError('prediction drift')
            predictions[key] = prediction
    failed = any(e['kind'].endswith('failed') for e in events)
    result = dict(schema='full-run-audit-v1', manifest_id=manifest['manifest_id'],
        total=len(expected), completed=len(predictions), complete=len(predictions)==len(expected) and not failed,
        failed=failed, identity_calls=sum(n for k,n in counts.items() if k.startswith('identity_attempt_') and k.endswith('_started')) if counts['execution_started'] else counts['identity_started'],
        answer_calls=sum(n for k,n in counts.items() if k.startswith('answer_attempt_') and k.endswith('_started')) if counts['execution_started'] else counts['answer_started'],
        identity_logical_calls=counts['identity_started'], answer_logical_calls=counts['answer_started'],
        raw_identity_results=counts['identity_received'], raw_answer_results=counts['answer_received'],
        identity_abstentions=sum(g.get('identity_accepted',{}).get('decision',{}).get('interpretation_status') == 'automatic-abstention' for g in groups.values()),
        judge_calls=0, retry_calls=sum(n for k,n in counts.items() if '_attempt_' in k and k.endswith('_started') and not k.endswith('_attempt_1_started')), predictions=predictions,
        event_counts=dict(counts), event_chain_sha256=events[-1]['hash'] if events else digest(store.identity))
    if result['retry_calls'] > manifest['envelope'].get('retry_total_limit', 0):
        raise ValueError('batch retry limit drift')
    return dict(result, report_sha256=digest(result))


def audit_full(*, manifest, checkpoint, retriever=None, integrity_check=None):
    if integrity_check: integrity_check()
    if not Path(checkpoint).exists(): raise ValueError('checkpoint absent')
    if str(Path(checkpoint).resolve()) != manifest['envelope']['checkpoint_path']:
        raise ValueError('checkpoint path drift')
    store = FullExecutionStore(checkpoint, manifest['manifest_id'])
    try: return _audit(manifest, store, retriever)
    finally: store.close()


def execute_full(*, manifest, checkpoint, retriever, identity_provider, answer_provider, integrity_check):
    integrity_check()
    env = manifest['envelope']
    if env['retry_count'] not in (0, 1, 2) or not 0 <= env.get('retry_total_limit', 0) <= 100 or (env['retry_count'] and not env.get('retry_total_limit')) or str(Path(checkpoint).resolve()) != env['checkpoint_path']:
        raise ValueError('execution envelope drift')
    iw, aw = env['identity_workers'], env['answer_workers']
    if not 1 <= iw <= 16 or not 1 <= aw <= 64 or env['identity_rpm'] <= 0 or env['answer_rpm'] <= 0:
        raise ValueError('worker/rate envelope invalid')
    store = FullExecutionStore(checkpoint, manifest['manifest_id'])
    try:
        existing = store.events()
        if existing:
            report = _audit(manifest, store, retriever)
            if report['complete']: return report
            raise ValueError('partial/failed/uncertain checkpoint; automatic resume forbidden')
        store.append('execution_started', '__batch__', {'attempt_accounting': 'transport-retry-v1'})
        abort = threading.Event(); retrieval_lock = threading.Lock()
        reservations = deque(); actual_next = {'identity': 0.0, 'answer': 0.0}; retry_used = [0]
        dispatch_lock = threading.Lock()
        def stop_dispatch():
            with dispatch_lock: abort.set()
        token_cap = env.get('answer_tpm')
        if token_cap is not None and token_cap <= 0: raise ValueError('invalid token cap')
        ids = deque(r for r in manifest['inputs'] if r['kind']=='oeq')
        mcqs = deque(r for r in manifest['inputs'] if r['kind']=='mcq')
        ready = deque(); running = {}; next_id = next_answer = 0.0

        def provider_call(stage, row, payload, provider):
            key = item_key(row)
            reserve = (len(canonical(payload['messages']).encode('utf-8')) + 256
                       + payload.get('max_tokens', 4096)) if stage == 'answer' else 0
            if token_cap is not None and reserve > token_cap:
                raise ValueError('single request exceeds token envelope')
            for attempt in range(1, env['retry_count']+2):
                reservation = None
                while True:
                    with dispatch_lock:
                        if abort.is_set(): raise RuntimeError('dispatch cancelled after batch failure')
                        now = time.monotonic()
                        while reservations and reservations[0][0] <= now-60: reservations.popleft()
                        allowed = (stage != 'answer' or token_cap is None
                                   or sum(r[1] for r in reservations)+reserve <= token_cap)
                        if now >= actual_next[stage] and allowed:
                            integrity_check()
                            if attempt > 1:
                                if retry_used[0] >= env.get('retry_total_limit', 0):
                                    raise RuntimeError('batch retry budget exhausted')
                                retry_used[0] += 1
                            reservation = [now, reserve]
                            if stage == 'answer': reservations.append(reservation)
                            store.append(f'{stage}_attempt_{attempt}_started', key,
                                dict(payload=payload, payload_sha256=digest(payload),
                                     reserved_tokens=reserve, retry=attempt > 1))
                            actual_next[stage] = time.monotonic()+60/env[stage+'_rpm']
                            break
                    if abort.wait(.025): raise RuntimeError('dispatch cancelled after batch failure')
                try:
                    value = provider.complete(payload)
                except Exception as error:
                    if not retryable_transport(error): raise
                    store.append(f'{stage}_attempt_{attempt}_transport_error', key, transport_fact(error))
                    if attempt > env['retry_count']: raise
                    if abort.wait(.25 * (2 ** (attempt-1))):
                        raise RuntimeError('retry cancelled after batch failure')
                    continue
                raw = {'call': asdict(value)} if stage == 'identity' else {'response': value}
                store.append(f'{stage}_attempt_{attempt}_returned', key, raw)
                if stage == 'answer':
                    total = value.get('response', {}).get('usage', {}).get('total_tokens')
                    if type(total) is int and total > 0:
                        with dispatch_lock: reservation[1] = total
                return value

        def identity_work(row):
            key = item_key(row)
            try:
                call = provider_call('identity', row, row['identity_request']['payload'], identity_provider)
                store.append('identity_received', key, {'call': asdict(call)})
                with retrieval_lock: request = _materialize(row, call, retriever)
                store.append('identity_accepted', key, {'decision': request['identity']})
                return row, request
            except Exception as error:
                stop_dispatch(); store.append('identity_failed', key, failure_fact(error)); return None

        def answer_work(row, request):
            key = item_key(row)
            try:
                response = provider_call('answer', row, request['payload'], answer_provider)
                store.append('answer_received', key, {'response': response})
                prediction = prediction_for(row, response, request['payload'])
                store.append('answer_result', key, {'prediction': prediction})
            except Exception as error:
                stop_dispatch(); store.append('answer_failed', key, failure_fact(error))

        with ThreadPoolExecutor(max_workers=iw) as ipool, ThreadPoolExecutor(max_workers=aw) as apool:
            while ids or mcqs or ready or running:
                for future in [f for f in running if f.done()]:
                    stage = running.pop(future)
                    result = future.result()
                    if stage == 'identity' and result is not None: ready.append(result)
                if abort.is_set():
                    if not running: break
                    wait(running, timeout=.05, return_when=FIRST_COMPLETED); continue
                try:
                    now = time.monotonic()
                    active = Counter(running.values())
                    if ids and active['identity'] < iw and len(ready) < aw*2 and now >= next_id:
                        integrity_check()
                        with dispatch_lock:
                            if abort.is_set(): continue
                            row = ids.popleft()
                            store.append('identity_started', item_key(row), {'request': row['identity_request']})
                            running[ipool.submit(identity_work, row)] = 'identity'
                        next_id = time.monotonic()+60/env['identity_rpm']
                    if (ready or mcqs) and active['answer'] < aw and now >= next_answer:
                        integrity_check()
                        if abort.is_set(): continue
                        if ready: row, request = ready[0]
                        else:
                            row = mcqs[0]; request = {'payload': row['payload']}
                        with dispatch_lock:
                            if abort.is_set(): continue
                            if ready: ready.popleft()
                            else: mcqs.popleft()
                            store.append('answer_started', item_key(row), {'request': request})
                            running[apool.submit(answer_work, row, request)] = 'answer'
                        next_answer = time.monotonic()+60/env['answer_rpm']
                except Exception as error:
                    stop_dispatch(); store.append('batch_failed', '__batch__', failure_fact(error))
                if running: wait(running, timeout=.025, return_when=FIRST_COMPLETED)
                else: time.sleep(.025)
        if not abort.is_set():
            try: integrity_check()
            except Exception as error: store.append('batch_failed', '__batch__', failure_fact(error))
        return _audit(manifest, store)
    finally:
        store.close()
