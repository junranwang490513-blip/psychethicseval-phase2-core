"""Prepare, verify, execute, audit, and export the final Phase Two pipeline."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from psychethicseval_lab.api_client import profiles, remote_factory
from psychethicseval_lab.contracts import canonical, failure_fact
from psychethicseval_lab.domain_retrieval_v4 import DomainRetrieverV4
from psychethicseval_lab.full_execution import audit_full, execute_full, item_key
from psychethicseval_lab.full_run_manifest import build_full_manifest, verify_full_manifest
from psychethicseval_lab.knowledge_v3 import KnowledgeArtifactV3


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def write_immutable(path, value):
    path = Path(path)
    text = canonical(value) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError("immutable output already differs: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def export_predictions(manifest, report, output):
    if (report.get("manifest_id") != manifest["manifest_id"] or not report.get("complete")
            or report.get("failed") or report.get("completed") != len(manifest["inputs"])):
        raise ValueError("incomplete or failed run cannot be exported")
    expected = {item_key(row) for row in manifest["inputs"]}
    if set(report["predictions"]) != expected or len(expected) != len(manifest["inputs"]):
        raise ValueError("export prediction coverage mismatch")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("immutable export directory exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".phase2-export-", dir=output.parent))
    try:
        names = []
        for kind in ("mcq", "oeq"):
            name = manifest["datasets"][kind]["filename"]
            if Path(name).name != name or name in names:
                raise ValueError("invalid original filename")
            rows = [row for row in manifest["inputs"] if row["kind"] == kind]
            if len(rows) != manifest["datasets"][kind]["count"]:
                raise ValueError("export track coverage mismatch")
            values = [
                {**row["source_item"], "prediction": report["predictions"][item_key(row)]}
                for row in rows
            ]
            (stage / name).write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            if read_json(stage / name) != values:
                raise ValueError("export roundtrip mismatch")
            names.append(name)
        with zipfile.ZipFile(stage / "submission.zip", "x", zipfile.ZIP_DEFLATED) as archive:
            for name in names:
                archive.write(stage / name, arcname=name)
        with zipfile.ZipFile(stage / "submission.zip") as archive:
            if archive.namelist() != names or archive.testzip() is not None:
                raise ValueError("ZIP verification failed")
        evidence = {
            "schema": "phase2-core-export-v1", "manifest_id": manifest["manifest_id"],
            "audit_report_sha256": report["report_sha256"],
            "counts": {kind: manifest["datasets"][kind]["count"] for kind in ("mcq", "oeq")},
            "files": {name: file_sha(stage / name) for name in names + ["submission.zip"]},
            "preserves_all_source_fields": True, "manual_prediction_changes": 0,
        }
        write_immutable(stage / "export-audit.json", evidence)
        stage.rename(output)
        return evidence
    except BaseException:
        shutil.rmtree(stage)
        raise


def main(argv=None, *, retriever=None, provider_factory=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify", "execute", "audit", "export"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--report")
    parser.add_argument("--output")
    parser.add_argument("--transport-retries", type=int, default=0)
    parser.add_argument("--retry-total-limit", type=int, default=0)
    parser.add_argument("--answer-workers", type=int, default=64)
    parser.add_argument("--identity-workers", type=int, default=16)
    parser.add_argument("--answer-rpm", type=int, default=200)
    parser.add_argument("--identity-rpm", type=int, default=160)
    args = parser.parse_args(argv)

    if retriever is None:
        artifact = KnowledgeArtifactV3.load(ROOT / "config/au_resource_pack/RAG-IDENTITY-KNOWLEDGE-v3.json")
        retriever = DomainRetrieverV4(artifact)
    transport = profiles()
    if args.action == "prepare":
        if not args.checkpoint:
            raise ValueError("checkpoint required")
        manifest = build_full_manifest(
            retriever=retriever, profiles=transport, checkpoint=args.checkpoint,
            answer_workers=args.answer_workers, identity_workers=args.identity_workers,
            answer_rpm=args.answer_rpm, identity_rpm=args.identity_rpm,
            transport_retries=args.transport_retries, retry_total_limit=args.retry_total_limit,
        )
        verify_full_manifest(manifest, retriever=retriever, profiles=transport)
        write_immutable(args.manifest, manifest)
        print(canonical({"manifest_id": manifest["manifest_id"], "envelope": manifest["envelope"]}))
        return 0

    manifest = read_json(args.manifest)
    verify_full_manifest(manifest, retriever=retriever, profiles=transport)
    if args.action == "verify":
        print(canonical({"manifest_id": manifest["manifest_id"], "verified": True}))
        return 0
    checkpoint = args.checkpoint or manifest["envelope"]["checkpoint_path"]
    if str(Path(checkpoint).resolve()) != manifest["envelope"]["checkpoint_path"]:
        raise ValueError("checkpoint path drift")
    integrity = lambda: verify_full_manifest(manifest, retriever=retriever, profiles=transport)

    if args.action == "execute":
        if not args.report:
            raise ValueError("execution report path required")
        if Path(checkpoint).exists():
            result = audit_full(manifest=manifest, checkpoint=checkpoint, retriever=retriever, integrity_check=integrity)
            if not result["complete"]:
                write_immutable(args.report, result)
                raise ValueError("existing partial checkpoint; resume forbidden")
        else:
            identity, answer = (provider_factory or remote_factory)(*transport)
            try:
                result = execute_full(
                    manifest=manifest, checkpoint=checkpoint, retriever=retriever,
                    identity_provider=identity, answer_provider=answer, integrity_check=integrity,
                )
            except Exception as error:
                evidence = {"schema": "phase2-core-execution-failure-v1",
                            "manifest_id": manifest["manifest_id"], "complete": False,
                            "error": failure_fact(error)}
                if Path(checkpoint).exists():
                    try:
                        evidence["audit"] = audit_full(manifest=manifest, checkpoint=checkpoint)
                    except Exception as audit_error:
                        evidence["audit_error"] = failure_fact(audit_error)
                write_immutable(args.report, evidence)
                raise
        write_immutable(args.report, result)
    else:
        result = audit_full(manifest=manifest, checkpoint=checkpoint, retriever=retriever, integrity_check=integrity)
        if args.report:
            write_immutable(args.report, result)
        if args.action == "export":
            if not args.output:
                raise ValueError("export output required")
            evidence = export_predictions(manifest, result, args.output)
            print(canonical(evidence))
            return 0
    print(canonical({key: value for key, value in result.items() if key != "predictions"}))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
