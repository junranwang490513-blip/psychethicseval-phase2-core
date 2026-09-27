"""Source-bounded v3 artifact. No measurement or source fields are promptable."""
import copy
import json
import re
from pathlib import Path
from .identity_classifier_v8 import sha, ROLES

PARTITIONS = {'universal', 'domain_safe'} | {f'{p}:{r}' for p in ('framing', 'role_specific') for r in ROLES}


class KnowledgeArtifactV3:
    def __init__(self, payload):
        self._payload = copy.deepcopy(payload)
        self.validate()

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding='utf-8')))

    @property
    def payload(self):
        return copy.deepcopy(self._payload)

    @property
    def units(self):
        return copy.deepcopy(self._payload['units'])

    @property
    def manifest_sha256(self):
        return self._payload['manifest_sha256']

    def validate(self):
        p = self._payload
        if not isinstance(p, dict) or set(p) != {'schema', 'id', 'version', 'units', 'manifest_sha256'}:
            raise ValueError('artifact_schema_keys')
        if p['schema'] != 'rag-knowledge-v3' or p['version'] != '3.0' or not isinstance(p['id'], str):
            raise ValueError('artifact_version')
        if p['manifest_sha256'] != sha({k: v for k, v in p.items() if k != 'manifest_sha256'}):
            raise ValueError('artifact_hash')
        if not isinstance(p['units'], list) or not p['units']:
            raise ValueError('artifact_empty')
        seen = set()
        for u in p['units']:
            if not isinstance(u, dict) or set(u) != {'id', 'partition', 'title', 'content', 'tags', 'source', 'unit_sha256'}:
                raise ValueError('unit_keys')
            if not isinstance(u['id'], str) or not re.fullmatch(r'[a-zA-Z0-9_-]+', u['id']) or u['id'] in seen:
                raise ValueError('unit_id')
            seen.add(u['id'])
            if u['partition'] not in PARTITIONS:
                raise ValueError('unit_partition')
            if any(not isinstance(u[k], str) or not u[k].strip() for k in ('title', 'content')):
                raise ValueError('unit_text')
            if not isinstance(u['tags'], list) or any(not isinstance(t, str) for t in u['tags']):
                raise ValueError('unit_tags')
            if 'crisis_support' in u['tags'] and u['partition'] != 'universal':
                raise ValueError('crisis_partition')
            source = u['source']
            if not isinstance(source, dict) or set(source) != {'url','publisher','jurisdiction','retrieved_at','excerpt','limitation'}:
                raise ValueError('source_keys')
            if any(not isinstance(v, str) or not v.strip() for v in source.values()) or not source['url'].startswith('https://'):
                raise ValueError('source_provenance')
            if u['unit_sha256'] != sha({k: v for k, v in u.items() if k != 'unit_sha256'}):
                raise ValueError('unit_hash')


def render_units_v3(units):
    return '\n\n'.join(f"[unit:{u['id']}] {u['title']}\n{u['content']}" for u in units)
