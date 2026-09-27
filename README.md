# PsychEthicsEval Phase Two Core

PsychEthicsEval Phase Two Core is an auditable pipeline for psychology, psychiatry, and mental-health ethics questions. It reads the Phase Two datasets, constructs MCQ or OEQ requests, calls the pinned models, records an append-only execution history, audits the run, and exports the original records with a `prediction` field.

This repository intentionally contains only the complete input-to-output framework. Judge pipelines, historical experiments, web interfaces, intermediate reports, and superseded implementations are excluded.

## Final pipeline

```text
mcq_phase_2.json --> B1 prompt ----------------------------> Answer model --> MCQ parser --+
                                                                                              +--> audit --> submission.zip
oeq_phase_2.json --> identity v13 --> permission boundary --> RAG --> Answer model ----------+
```

- **MCQ:** uses the fixed B1 prompt and strictly parses an array of zero-based option indices.
- **OEQ:** uses identity classifier v13. If the claimed role cannot be bound safely to the source question, the classifier abstains locally and grants no role-specific knowledge access.
- **RAG:** retrieves from a versioned Australian-context knowledge artifact with an offline MiniLM encoder. Model output cannot expand the permitted knowledge partitions.
- **Execution:** uses bounded concurrency, RPM/TPM limits, bounded transport retries, and an append-only SQLite event chain.
- **Export:** produces the two original filenames and `submission.zip` only after a complete, failure-free audit.
- **Judge:** the final Phase Two path does not call a Judge. Every run manifest records `judge: 0`.

## 5B Freeze provenance

The final submission was not produced by one from-scratch run under a single Freeze:

- A222 performed the main run under `full-5b-v5` and returned 3,747 answers.
- The event history proved that 42 MCQ items had never reached a model call.
- A223 ran those 42 items under `full-5b-v6`, which included the revised MCQ wrapper parser and completion logic.
- The final export combined the A222 results with the A223 completions and was audited and exported by the v6 logic.

Therefore, v6 is the final completion, merge, audit, and export version, but it is not the sole execution version for all 3,789 predictions. The v5/v6 archives are release evidence, not part of the core source tree.

## Repository layout

```text
.
├── README.md
├── pyproject.toml
├── config/
│   ├── datasets.phase2.json
│   └── au_resource_pack/RAG-IDENTITY-KNOWLEDGE-v3.json
├── prompts/PHASE2_PROMPTS.md
├── scripts/run_full_phase2.py
├── src/psychethicseval_lab/
│   ├── api_client.py
│   ├── contracts.py
│   ├── identity_classifier_v8.py       # candidate extraction reused by v12
│   ├── identity_classifier_v12.py      # strict evidence binder reused by v13
│   ├── identity_classifier_v13.py      # final identity policy
│   ├── identity_boundary_v3.py
│   ├── domain_retrieval_v3.py
│   ├── domain_retrieval_v4.py
│   ├── knowledge_v3.py
│   ├── minilm_retriever.py
│   ├── oeq_materialization_v3.py
│   ├── mcq_prediction_v2.py            # strict base parser reused by v3
│   ├── mcq_prediction_v3.py
│   ├── full_run_manifest.py
│   └── full_execution.py
├── artifacts/semantic_embeddings/      # manifest, tokenizer metadata, and license
└── tests/test_phase2_core.py
```

The v8/v12 identity modules and the MCQ v2 parser remain because the final modules explicitly reuse their validated low-level contracts. They are dependencies, not alternative production paths.

## Dataset contract

By default, the repository expects the datasets in a sibling directory:

```text
../dataSET/mcq_phase_2.json
../dataSET/oeq_phase_2.json
```

File counts and SHA-256 values must match `config/datasets.phase2.json`. The loader rejects inputs that already contain `prediction`, an MCQ answer key, or an OEQ `inquirer` label.

## Runtime assets

The Git repository contains the semantic-asset manifest, tokenizer metadata, upstream revision, and license. The large ONNX model, pinned wheels, and expanded runtime are intentionally excluded from ordinary Git history and must be supplied from the private release bundle at:

```text
artifacts/semantic_embeddings/minilm-l6-v2/onnx/model.onnx
artifacts/semantic_embeddings/wheels/
artifacts/semantic_embeddings/runtime/
```

Every asset is checked against `artifacts/semantic_embeddings/manifest.json`. Missing or changed assets cause a hard failure; the pipeline never silently downloads or substitutes another model.

## Verification

The core test suite makes no remote or paid model calls:

```bash
PYTHONPATH=src python3 -m unittest -v tests.test_phase2_core
```

It covers v13 abstention and permission boundaries, RAG request construction, MCQ parsing, a small end-to-end input-to-output run, append-only replay, the full 3,789-item manifest, and exclusion of credential metadata from the manifest.

## Preparing and verifying a run

Create an immutable run manifest and bind it to a dedicated checkpoint:

```bash
PYTHONPATH=src python3 scripts/run_full_phase2.py prepare \
  --manifest var/phase2-manifest.json \
  --checkpoint var/phase2-checkpoint.sqlite3

PYTHONPATH=src python3 scripts/run_full_phase2.py verify \
  --manifest var/phase2-manifest.json
```

## Executing a run

Execution is an explicit paid operation. Credentials are read from the process environment only when `execute` starts:

```bash
PHASE2_IDENTITY_CREDENTIAL='...' \
PHASE2_ANSWER_CREDENTIAL='...' \
PYTHONPATH=src python3 scripts/run_full_phase2.py execute \
  --manifest var/phase2-manifest.json \
  --report var/phase2-audit.json
```

The repository does not store real keys, `.env` files, Keychain service names, or account names. Credentials are never serialized into manifests, SQLite events, errors, or exports.

An incomplete or uncertain checkpoint is not resumed automatically. This prevents results with ambiguous provenance from being mixed into a submission.

## Auditing and exporting

Audit and export do not require credentials:

```bash
PYTHONPATH=src python3 scripts/run_full_phase2.py audit \
  --manifest var/phase2-manifest.json \
  --report var/phase2-audit.json

PYTHONPATH=src python3 scripts/run_full_phase2.py export \
  --manifest var/phase2-manifest.json \
  --output output/phase2-submission
```

The export preserves every original source field and adds exactly the automatically persisted `prediction` value.

## Security properties

- Credentials remain process-local and are omitted from public profile identities.
- Provider error bodies and exception messages are not persisted as failure facts.
- Source files, prompts, datasets, model identities, knowledge artifacts, and run parameters are bound into the run manifest.
- Identity-model assertions do not directly grant operational permissions.
- Invalid, conflicting, or unbound identity output fails closed to the safe knowledge partitions.
- The remote `main` branch begins with a clean, single-commit history and does not contain deleted experimental configuration history.

## Limitations

- Python 3.11 or later is required. The currently pinned semantic runtime targets macOS on Apple Silicon with CPython 3.12.
- Phase Two inputs contain no local ground truth, so this project does not claim a Phase Two accuracy score.
- A successful audit establishes structural integrity, provenance, and coverage. It does not guarantee that a generated answer is clinically, legally, or ethically correct.
- Identity classification and RAG do not replace professional clinical, legal, or ethical review.

## Public-release checklist

Before changing the repository from private to public:

1. Confirm that the dataset license permits the documented use and distribution model.
2. Confirm redistribution rights for every release asset.
3. Choose and add a repository-level software license.
4. Run the core test suite and a fresh secret scan.
5. Review the private release bundle separately; large runtime assets are not part of Git history.

## License

No repository-level license has been selected yet. The MiniLM upstream license is retained at `artifacts/semantic_embeddings/minilm-l6-v2/LICENSE`.
