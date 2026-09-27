"""Domain-first retrieval with separate safety relevance and role additions."""
import math
import re
from .identity_classifier_v8 import sha
from .knowledge_v3 import render_units_v3


def safety_relevance_v3(question):
    """Bounded retrieval relevance heuristic; never a clinical risk assessment.

    Negation is local to a clause. Mixed or unresolved mentions retain context;
    hypothetical examples and clear denials alone do not reserve resources.
    """
    clauses = re.split(r'[.;!?\n]|\bbut\b|\bhowever\b|\band\b', question.casefold())
    matched = []
    for clause in clauses:
        if re.search(r'\b(?:not|never|no|denies|without)\b.{0,25}\b(?:suicid\w*|self.harm|safety concern\w*|immediate (?:risk|danger))', clause):
            continue
        if re.search(r'\b(?:hypothetical|training example|textbook)\b', clause):
            continue
        if re.search(r'suicid|self.harm|hurting (?:herself|himself|myself)|ending (?:it|my life)|better off dead|didn.t wake up|plan to jump|jump tonight|overdose|immediate danger|imminent risk|kill (?:myself|herself|himself)', clause):
            matched.append(clause.strip())
    return dict(policy_id='safety-relevance-v3', relevant=bool(matched),
                matched_clauses=matched, limitation='Retrieval relevance only; not clinical risk assessment.')


class PinnedEncoder:
    def __init__(self):
        from .minilm_retriever import _SentenceEncoder
        self.encoder = _SentenceEncoder()

    def encode(self, text):
        return list(self.encoder.encode(text))

    def identity(self):
        from .minilm_retriever import MODEL_NAME, MODEL_REVISION, MANIFEST_SHA256, EXPECTED_VERSIONS
        return dict(model=MODEL_NAME, revision=MODEL_REVISION, assets_sha256=MANIFEST_SHA256,
                    dependencies=EXPECTED_VERSIONS, pooling='attention-mask-mean-l2', dimensions=384)


class DomainRetrieverV3:
    def __init__(self, artifact, *, encoder=None, top_k=5):
        if type(top_k) is not int or not 1 <= top_k <= 12:
            raise ValueError('top_k')
        artifact.validate()
        self.artifact = artifact
        self.encoder = encoder if encoder is not None else PinnedEncoder()
        self.top_k = top_k
        self.vectors = {u['id']: self._vector(u['title']+'\n'+u['content']+'\n'+' '.join(u['tags'])) for u in artifact.units}
        self.index_sha256 = sha(self.vectors)

    def _vector(self, text):
        values = [float(v) for v in self.encoder.encode(text)]
        if not values or not all(math.isfinite(x) for x in values):
            raise ValueError('invalid_embedding')
        norm = math.sqrt(sum(x*x for x in values))
        if norm == 0:
            raise ValueError('zero_embedding')
        return [x/norm for x in values]

    def identity(self):
        self.artifact.validate()
        if sha(self.vectors) != self.index_sha256:
            raise ValueError('index_drift')
        return dict(id='domain-retriever-v3', artifact_sha256=self.artifact.manifest_sha256,
                    index_sha256=self.index_sha256, encoder=self.encoder.identity(), top_k=self.top_k,
                    ranking='cosine-plus-0.05-exact-tag-overlap', safety_policy='safety-relevance-v3',
                    framing_max=1, operations_max=1)

    def retrieve(self, question, permissions):
        identity = self.identity()
        safety = safety_relevance_v3(question)
        query = self._vector(question)
        words = set(re.findall(r'\w+', question.casefold()))
        ranked = []
        for u in self.artifact.units:
            vector = self.vectors[u['id']]
            if len(vector) != len(query):
                raise ValueError('embedding_dimension_drift')
            score = round(sum(a*b for a,b in zip(vector, query)) + 0.05*len(words.intersection(u['tags'])), 8)
            ranked.append((score, u))
        ranked.sort(key=lambda entry: (-entry[0], entry[1]['id']))
        domains = [(s,u) for s,u in ranked if u['partition'] in {'universal','domain_safe'} and 'crisis_support' not in u['tags']]
        selected = domains[:self.top_k]
        if safety['relevant']:
            crisis = [(s,u) for s,u in ranked if 'crisis_support' in u['tags']]
            if crisis:
                selected = domains[:max(0, self.top_k-1)] + crisis[:1]
        # Role framing/operations add to, and never evict, domain facts.
        for prefix in ('framing:', 'role_specific:'):
            eligible = [(s,u) for s,u in ranked if u['partition'].startswith(prefix) and u['partition'] in permissions]
            selected.extend(eligible[:1])
        units = [u for s,u in selected]
        context = render_units_v3(units)
        return dict(identity=identity, safety=safety, selected=units,
                    scores={u['id']: s for s,u in selected}, context=context, context_sha256=sha(context))
