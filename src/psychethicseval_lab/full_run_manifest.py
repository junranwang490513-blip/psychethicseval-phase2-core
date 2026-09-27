"""Build and verify the final, v13-only Phase Two execution manifest."""
import hashlib
import json
from pathlib import Path
import platform

from .contracts import digest
from .identity_classifier_v13 import build_identity_request_v13
from .oeq_materialization_v3 import scientific_identity_v3
from .phase2_prompts import load_phase2_prompt

ROOT = Path(__file__).resolve().parents[2]
CORE_RUNTIME = (
    "scripts/run_full_phase2.py",
    "src/psychethicseval_lab/api_client.py",
    "src/psychethicseval_lab/contracts.py",
    "src/psychethicseval_lab/domain_retrieval_v3.py",
    "src/psychethicseval_lab/domain_retrieval_v4.py",
    "src/psychethicseval_lab/full_execution.py",
    "src/psychethicseval_lab/full_run_manifest.py",
    "src/psychethicseval_lab/identity_boundary_v3.py",
    "src/psychethicseval_lab/identity_classifier_v8.py",
    "src/psychethicseval_lab/identity_classifier_v12.py",
    "src/psychethicseval_lab/identity_classifier_v13.py",
    "src/psychethicseval_lab/knowledge_v3.py",
    "src/psychethicseval_lab/mcq_prediction_v2.py",
    "src/psychethicseval_lab/mcq_prediction_v3.py",
    "src/psychethicseval_lab/minilm_retriever.py",
    "src/psychethicseval_lab/oeq_materialization_v3.py",
    "src/psychethicseval_lab/phase2_prompts.py",
    "config/datasets.phase2.json",
    "config/au_resource_pack/RAG-IDENTITY-KNOWLEDGE-v3.json",
    "prompts/PHASE2_PROMPTS.md",
)


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def full_inventory():
    return {name: file_sha(ROOT / name) for name in CORE_RUNTIME}


def _mcq_user(item):
    options = "\n".join(f"{index}. {text}" for index, text in enumerate(item["options"]))
    return f"Question:\n{item['question']}\n\nOptions:\n{options}"


def build_full_manifest(*, retriever, profiles, checkpoint, answer_workers=64,
                        identity_workers=16, answer_rpm=200, identity_rpm=160,
                        answer_tpm=900000, transport_retries=0, retry_total_limit=0):
    if not 1 <= answer_workers <= 64 or not 1 <= identity_workers <= 16:
        raise ValueError("worker bounds")
    if not 0 < answer_rpm <= 500 or not 0 < identity_rpm <= 160 or not 0 < answer_tpm <= 1_000_000:
        raise ValueError("rate bounds")
    if transport_retries not in (0, 1, 2) or not 0 <= retry_total_limit <= 100:
        raise ValueError("transport retry bounds")
    if bool(transport_retries) != bool(retry_total_limit):
        raise ValueError("retry count and total limit must be enabled together")
    if len(profiles) != 2:
        raise ValueError("identity and answer profiles required")

    config = json.loads((ROOT / "config/datasets.phase2.json").read_text(encoding="utf-8"))
    mcq_system = load_phase2_prompt(ROOT / "prompts/PHASE2_PROMPTS.md", "B1-MCQ-v1.0")
    rows, datasets = [], {}
    for kind in ("mcq", "oeq"):
        spec = config["datasets"][kind]
        path = (ROOT / "config" / spec["path"]).resolve()
        if file_sha(path) != spec["sha256"]:
            raise ValueError("dataset drift")
        items = json.loads(path.read_text(encoding="utf-8"))
        expected = 1277 if kind == "mcq" else 2512
        if len(items) != expected or len({str(item["id"]) for item in items}) != expected:
            raise ValueError("full source coverage")
        if any("prediction" in item or "correct_answer" in item or "inquirer" in item for item in items):
            raise ValueError("unexpected Phase Two labels or predictions")
        datasets[kind] = {
            "path": spec["path"], "filename": path.name,
            "sha256": spec["sha256"], "count": expected,
        }
        for item in items:
            row = {"kind": kind, "item_id": str(item["id"]), "source_item": item}
            if kind == "oeq":
                row["identity_request"] = build_identity_request_v13(question=item["question"])
            else:
                row["payload"] = {
                    "model": profiles[1].model_id, "temperature": 0, "seed": 42,
                    "max_tokens": 96, "reasoning_effort": "none",
                    "messages": [
                        {"role": "system", "content": mcq_system},
                        {"role": "user", "content": _mcq_user(item)},
                    ],
                }
            rows.append(row)

    body = {
        "schema": "phase2-core-manifest-v1",
        "method": {
            "subsystem": scientific_identity_v3(retriever),
            "runtime_sources": full_inventory(),
            "profiles": [profile.public_identity() for profile in profiles],
            "runtime": {
                "python": platform.python_version(), "system": platform.system(),
                "machine": platform.machine(),
            },
            "identity_version": "v13",
            "mcq_prompt_id": "B1-MCQ-v1.0",
            "mcq_prompt_sha256": hashlib.sha256(mcq_system.encode("utf-8")).hexdigest(),
            "mcq_prediction_parser": "mcq-prediction-v3",
        },
        "datasets": datasets,
        "inputs": rows,
        "envelope": {
            "checkpoint_path": str(Path(checkpoint).resolve()),
            "answer_workers": answer_workers, "identity_workers": identity_workers,
            "answer_rpm": answer_rpm, "identity_rpm": identity_rpm,
            "answer_tpm": answer_tpm, "retry_count": transport_retries,
            "retry_total_limit": retry_total_limit, "timeout_seconds": 240,
            "dispatch_policy": "bounded-smooth-rate-v1",
            "counts": {"mcq": 1277, "oeq": 2512, "identity": 2512, "answer": 3789, "judge": 0},
            "stop_policy": "structural-integrity-abort-drain-inflight",
            "output_policy": "one-first-returned-response-per-item-no-manual-selection",
        },
    }
    return {**body, "manifest_id": digest(body)}


def verify_full_manifest(manifest, *, retriever, profiles):
    try:
        envelope = manifest["envelope"]
        current = build_full_manifest(
            retriever=retriever, profiles=profiles, checkpoint=envelope["checkpoint_path"],
            answer_workers=envelope["answer_workers"], identity_workers=envelope["identity_workers"],
            answer_rpm=envelope["answer_rpm"], identity_rpm=envelope["identity_rpm"],
            answer_tpm=envelope["answer_tpm"], transport_retries=envelope["retry_count"],
            retry_total_limit=envelope["retry_total_limit"],
        )
        if current != manifest:
            raise ValueError("full manifest drift")
    except (KeyError, TypeError) as error:
        raise ValueError("invalid full manifest") from error
    return manifest["manifest_id"]
