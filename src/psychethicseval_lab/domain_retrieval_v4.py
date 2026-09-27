"""Bounded topical coverage additions over the unchanged v3 semantic baseline.

Privacy and document duties are commonly outranked by clinical symptom terms.
Add the corresponding neutral evidence when the question explicitly raises it;
never evict safety/domain evidence or admit a new permission partition.
"""
import re
from .domain_retrieval_v3 import DomainRetrieverV3
from .knowledge_v3 import render_units_v3
from .identity_classifier_v8 import sha

TOPICS={
    'privacy-disclosure':r'\b(?:confidential\w*|privacy|disclos\w*|consent|shar\w*|tell\w*|inform\w*|notif\w*|report\w*|call\w*|contact\w*|involv\w*)\b|\bbetween\s+(?:us|you and me)\b',
    'reports':r'\b(?:reports?|certificat\w*|letters?|records?|document\w*)\b',
}

class DomainRetrieverV4(DomainRetrieverV3):
    def identity(self):
        return {**super().identity(),'id':'domain-retriever-v4','coverage_policy_sha256':sha(TOPICS),
                'coverage_additions_max':2,'coverage_policy':'explicit-topic-neutral-additions-v1'}

    def retrieve(self,question,permissions):
        result=super().retrieve(question,permissions)
        units=list(result['selected']);selected={u['id'] for u in units}
        for unit_id,pattern in TOPICS.items():
            if unit_id in selected or not re.search(pattern,question,re.I): continue
            matches=[u for u in self.artifact.units if u['id']==unit_id and u['partition']=='domain_safe' and u['partition'] in permissions]
            if len(matches)==1: units.append(matches[0])
        context=render_units_v3(units)
        return {**result,'selected':units,'context':context,'context_sha256':sha(context),
                'coverage_additions':[u['id'] for u in units if u['id'] not in selected]}
