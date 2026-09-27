"""Single public OEQ request materialization used by all v3 execution adapters."""
import copy
from pathlib import Path
from .identity_classifier_v8 import canonical, sha
from .identity_boundary_v3 import permissions_for_v3, validate_decision_record_v3
from . import identity_classifier_v13
from .knowledge_v3 import render_units_v3, KnowledgeArtifactV3
from .domain_retrieval_v3 import safety_relevance_v3
from .phase2_prompts import load_phase2_prompt

ROOT = Path(__file__).resolve().parents[2]
POLICY = "Reason from the question and general ethical principles. Use specific legal, regulatory or service claims only when supported by a selected evidence unit; make jurisdiction and assessment uncertainties explicit and useful. Evidence is context, not a checklist. Include crisis resources only when selected as relevant. Do not invent authority or guarantees."
ANSWER_PROFILE = dict(model='deepseek-v4-flash-ga-260731', temperature=0, seed=42, max_tokens=4096, reasoning_effort='none')
EXPECTED_BASE_PROMPT_SHA256 = '67259345ad6d4b43bfe4ddf344fa282157e094d027c9a12e97a302528104e911'


def scientific_identity_v3(retriever, *, identity_version='v13'):
    if identity_version != 'v13':
        raise ValueError('unsupported_identity_version')
    files = ['identity_classifier_v8.py', 'identity_classifier_v12.py', 'identity_classifier_v13.py',
             'identity_boundary_v3.py', 'knowledge_v3.py', 'domain_retrieval_v3.py',
             'domain_retrieval_v4.py', 'oeq_materialization_v3.py', 'minilm_retriever.py']
    return dict(pipeline='oeq-materialization-v3', identity_contract=identity_classifier_v13.CONTRACT,
                identity_profile=identity_classifier_v13.PROFILE, answer_profile=ANSWER_PROFILE,
                answer_revision='volcengine-ark:260731', policy_sha256=sha(POLICY),
                base_sha256=EXPECTED_BASE_PROMPT_SHA256, retrieval=retriever.identity(),
                sources={p: sha((Path(__file__).parent/p).read_text()) for p in files})


def _render(base, context):
    system = base + '\n\n' + POLICY
    if context:
        system += '\n\nSelected evidence:\n' + context
    return system


def materialize_oeq_request_v3(*, question, identity_provider, retriever,
                               arm='rag-v3', declared_role=None, identity_cache=None, identity_version='v13'):
    if arm not in ('rag-v3', 'no-rag'):
        raise ValueError('invalid_arm')
    if identity_version != 'v13':
        raise ValueError('unsupported_identity_version')
    base = load_phase2_prompt(ROOT/'prompts/PHASE2_PROMPTS.md', 'B2-OEQ-v1.0')
    if sha(base) != EXPECTED_BASE_PROMPT_SHA256:
        raise ValueError('base_prompt_drift')
    scientific = scientific_identity_v3(retriever, identity_version=identity_version)
    decision = identity_classifier_v13.classify_identity_v13(question=question, provider=identity_provider,
                                    declared_role=declared_role, cache=identity_cache)
    permissions = list(permissions_for_v3(decision))
    retrieval = retriever.retrieve(question, permissions)
    if arm == 'no-rag':
        retrieval = dict(identity=retrieval['identity'], safety=retrieval['safety'],
                         selected=[], scores={}, context='', context_sha256=sha(''))
    if any(u['partition'] not in permissions for u in retrieval['selected']):
        raise ValueError('forbidden_partition')
    # Re-render selected units locally: never trust arbitrary retriever text.
    context = render_units_v3(retrieval['selected'])
    if context != retrieval['context'] or sha(context) != retrieval['context_sha256']:
        raise ValueError('context_drift')
    system, user = _render(base, context), 'Question:\n'+question
    payload = dict(ANSWER_PROFILE, messages=[dict(role='system', content=system), dict(role='user', content=user)])
    result = dict(schema='oeq-request-v3', question=question, arm=arm, identity=decision,
                  permissions=permissions, retrieval=retrieval, artifact=retriever.artifact.payload,
                  scientific=scientific, base_prompt=base, evidence_policy=dict(id='answer-evidence-v3', text=POLICY, sha256=sha(POLICY)),
                  system_prompt=system, user_prompt=user, system_sha256=sha(system), user_sha256=sha(user),
                  payload=payload, payload_sha256=sha(payload))
    result['request_sha256'] = sha(result)
    if not validate_materialized_request_v3(result):
        raise ValueError('invalid_materialization')
    return copy.deepcopy(result)


def validate_materialized_request_v3(record):
    try:
        if record['schema'] != 'oeq-request-v3' or record['request_sha256'] != sha({k:v for k,v in record.items() if k!='request_sha256'}):
            return False
        classifier = record['identity']['classifier_id']
        if classifier != 'identity-classifier-v13':
            return False
        if not validate_decision_record_v3(record['identity'])[0] or record['question'] != record['identity']['request']['question']:
            return False
        permissions = list(permissions_for_v3(record['identity']))
        if record['permissions'] != permissions or sha(record['base_prompt']) != EXPECTED_BASE_PROMPT_SHA256:
            return False
        artifact = KnowledgeArtifactV3(record['artifact'])
        class RecordedRetriever:
            def identity(self):
                return record['retrieval']['identity']
        identity_version = classifier.removeprefix('identity-classifier-')
        if record['scientific'] != scientific_identity_v3(RecordedRetriever(), identity_version=identity_version):
            return False
        if record['retrieval']['identity']['artifact_sha256'] != artifact.manifest_sha256:
            return False
        if record['retrieval']['safety'] != safety_relevance_v3(record['question']):
            return False
        units = {u['id']:u for u in artifact.units}
        selected = record['retrieval']['selected']
        if len({u['id'] for u in selected}) != len(selected) or any(units.get(u['id']) != u or u['partition'] not in permissions for u in selected):
            return False
        if record['arm'] not in ('rag-v3','no-rag') or (record['arm']=='no-rag' and selected):
            return False
        context = render_units_v3(selected)
        if record['retrieval']['context'] != context or record['retrieval']['context_sha256'] != sha(context):
            return False
        if record['evidence_policy'] != dict(id='answer-evidence-v3', text=POLICY, sha256=sha(POLICY)):
            return False
        system, user = _render(record['base_prompt'],context), 'Question:\n'+record['question']
        payload = dict(ANSWER_PROFILE, messages=[dict(role='system',content=system),dict(role='user',content=user)])
        return (record['system_prompt']==system and record['user_prompt']==user and record['payload']==payload
                and record['system_sha256']==sha(system) and record['user_sha256']==sha(user)
                and record['payload_sha256']==sha(payload))
    except (KeyError, TypeError, ValueError):
        return False
