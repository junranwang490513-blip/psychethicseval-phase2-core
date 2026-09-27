"""Explicit local abstention for unusable metadata-free model assertions.

The provider request and strict v12 binder are unchanged. A failed interpretation
is retained as an interpretation failure, never rewritten as a provider answer.
"""
import copy
from pathlib import Path
from . import identity_classifier_v12 as strict

canonical, sha = strict.canonical, strict.sha
PROMPT = strict.PROMPT
PROFILE = {**strict.PROFILE, 'profile_id': 'phase2-identity-qwen3.8-flash-v13'}
CONTRACT = dict(strict.CONTRACT, classifier_id='identity-classifier-v13', version='13.0',
    prompt_id='IDENTITY-CLASSIFIER-v13', binder_id='strict-v12-with-local-abstention-v13',
    cache_schema='identity-cache-v13', runtime_source_sha256=sha(Path(__file__).read_text()),
    strict_binder_source_sha256=sha(Path(strict.__file__).read_text()))
INTERPRETATION_ERRORS = frozenset({'invalid_identity_json', 'identity_keys', 'invalid_role',
    'invalid_candidate_ids', 'unknown_candidate_id', 'invalid_rationale',
    'abstention_requires_no_evidence', 'incompatible_author_evidence'})
ABSTENTION_RATIONALE = ('Automatic local abstention: the unchanged strict evidence binder '
    'could not validate the model assertion; no author role is granted by this policy.')


def build_identity_request_v13(*, question, declared_role=None):
    request = strict.build_identity_request_v12(question=question, declared_role=declared_role)
    request.update(schema='identity-candidate-request-v13', contract=copy.deepcopy(CONTRACT),
                   profile=copy.deepcopy(PROFILE))
    request['request_sha256'] = sha({k:v for k,v in request.items() if k != 'request_sha256'})
    return request


def materialize_identity_decision_v13(*, question, model_output, declared_role=None):
    # Invalid caller input is not a model interpretation failure.
    request = build_identity_request_v13(question=question, declared_role=declared_role)
    if not isinstance(model_output, str):
        raise ValueError('model_output_must_be_text')
    try:
        record = strict.materialize_identity_decision_v12(question=question,
            model_output=model_output, declared_role=declared_role)
    except ValueError as error:
        if declared_role is not None or str(error) not in INTERPRETATION_ERRORS:
            raise
        record = dict(response_output_text=model_output, response_sha256=sha(model_output),
            role=None, selected_candidate_ids=[], bound_candidates=[], permission_basis='abstained',
            rationale=ABSTENTION_RATIONALE, interpretation_status='automatic-abstention',
            interpretation_error=str(error))
    else:
        record.update(interpretation_status='bound', interpretation_error=None)
    record.update(classifier_id='identity-classifier-v13', request=request)
    record['record_sha256'] = sha({k:v for k,v in record.items() if k != 'record_sha256'})
    return record


def validate_v13_decision_record(record):
    try:
        if not isinstance(record, dict) or record.get('classifier_id') != 'identity-classifier-v13':
            return False, ['v13_required']
        rebuilt = materialize_identity_decision_v13(question=record['request']['question'],
            declared_role=record['request']['declared_role'], model_output=record['response_output_text'])
        return (True, []) if canonical(rebuilt) == canonical(record) else (False, ['identity_record_drift'])
    except (KeyError, TypeError, ValueError):
        return False, ['invalid_identity_record']


def classify_identity_v13(*, question, provider, declared_role=None, cache=None):
    request = build_identity_request_v13(question=question, declared_role=declared_role)
    key = request['request_sha256']
    if cache is not None and key in cache:
        entry = cache[key]
        if (not isinstance(entry, dict) or entry.get('request') != request
            or entry.get('cache_sha256') != sha({k:v for k,v in entry.items() if k != 'cache_sha256'})
            or not validate_v13_decision_record(entry.get('decision'))[0]
            or entry['decision']['request'] != request):
            raise ValueError('cache_drift')
        return copy.deepcopy(entry['decision'])
    call = provider.complete(copy.deepcopy(request['payload']))
    if call.response_model != PROFILE['model'] or call.revision != PROFILE['revision'] or call.finish_reason != 'stop':
        raise ValueError('identity_provider_drift_or_truncation')
    decision = materialize_identity_decision_v13(question=question, declared_role=declared_role,
                                               model_output=call.output_text)
    if cache is not None:
        entry = dict(request=request, decision=decision, response_model=call.response_model,
            revision=call.revision, usage=call.usage, latency_ms=call.latency_ms, finish_reason=call.finish_reason)
        entry['cache_sha256'] = sha(entry)
        cache[key] = copy.deepcopy(entry)
    return decision
