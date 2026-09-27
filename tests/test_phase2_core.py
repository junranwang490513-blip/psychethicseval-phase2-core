import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from psychethicseval_lab.api_client import profiles, remote_factory
from psychethicseval_lab.contracts import ClassifierCall
from psychethicseval_lab.domain_retrieval_v4 import DomainRetrieverV4
from psychethicseval_lab.full_execution import audit_full, execute_full
from psychethicseval_lab.full_run_manifest import build_full_manifest, verify_full_manifest
from psychethicseval_lab.identity_boundary_v3 import permissions_for_v3
from psychethicseval_lab.identity_classifier_v8 import sha
from psychethicseval_lab.identity_classifier_v13 import (
    build_identity_request_v13,
    materialize_identity_decision_v13,
)
from psychethicseval_lab.knowledge_v3 import KnowledgeArtifactV3
from psychethicseval_lab.mcq_prediction_v3 import parse_mcq_prediction_v3
from psychethicseval_lab.oeq_materialization_v3 import materialize_oeq_request_v3


def artifact():
    source = {
        "url": "https://example.org/source", "publisher": "Example",
        "jurisdiction": "AU", "retrieved_at": "2026-01-01",
        "excerpt": "Test provenance only.", "limitation": "Synthetic test unit.",
    }
    unit = {
        "id": "privacy-disclosure", "partition": "domain_safe",
        "title": "Privacy", "content": "Discuss privacy and proportionate disclosure.",
        "tags": ["privacy"], "source": source,
    }
    unit["unit_sha256"] = sha(unit)
    body = {"schema": "rag-knowledge-v3", "id": "test", "version": "3.0", "units": [unit]}
    body["manifest_sha256"] = sha(body)
    return KnowledgeArtifactV3(body)


class FakeEncoder:
    def encode(self, text):
        return [float((sum(text.encode("utf-8")) % 11) + 1), 1.0]

    def identity(self):
        return {"model": "test-encoder", "revision": "1"}


class IdentityProvider:
    calls = 0

    def complete(self, payload):
        self.calls += 1
        content = json.dumps({"role": None, "selected_candidate_ids": [], "rationale": "Role unclear."})
        raw = json.dumps({
            "model": "qwen3.8-flash",
            "choices": [{"finish_reason": "stop", "message": {"content": content}}],
        })
        return ClassifierCall(content, "qwen3.8-flash", "dashscope:qwen3.8-flash",
                              {"provider_raw_response": raw}, 0, "stop")


class AnswerProvider:
    calls = 0

    def complete(self, payload):
        self.calls += 1
        content = "[0]" if payload["max_tokens"] == 96 else (
            "Respect consent, privacy, safety, and the limits of the available facts."
        )
        value = {"model": payload["model"],
                 "choices": [{"finish_reason": "stop", "message": {"content": content}}]}
        return {"response": value, "raw_response": json.dumps(value)}


class PhaseTwoCoreTests(unittest.TestCase):
    def retriever(self):
        return DomainRetrieverV4(artifact(), encoder=FakeEncoder())

    def test_v13_abstention_and_rag_boundary(self):
        decision = materialize_identity_decision_v13(question="I need guidance.", model_output="not JSON")
        self.assertEqual(decision["interpretation_status"], "automatic-abstention")
        self.assertEqual(permissions_for_v3(decision), ("universal", "domain_safe"))
        request = materialize_oeq_request_v3(
            question="I need guidance about privacy.", identity_provider=IdentityProvider(),
            retriever=self.retriever(),
        )
        self.assertEqual(request["identity"]["classifier_id"], "identity-classifier-v13")
        self.assertEqual(request["permissions"], ["universal", "domain_safe"])

    def test_mcq_parser(self):
        self.assertEqual(parse_mcq_prediction_v3("[json][2,0][/json]", 3), [2, 0])
        with self.assertRaises(ValueError):
            parse_mcq_prediction_v3("Answer: [0]", 3)

    def test_small_input_to_output_and_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.sqlite3"
            question = "What should be considered about privacy?"
            rows = [
                {"kind": "mcq", "item_id": "1",
                 "source_item": {"id": 1, "question": question, "options": ["a", "b"]},
                 "payload": {"model": "deepseek-v4-flash-ga-260731", "max_tokens": 96,
                             "messages": []}},
                {"kind": "oeq", "item_id": "2", "source_item": {"id": 2, "question": question},
                 "identity_request": build_identity_request_v13(question=question)},
            ]
            manifest = {
                "manifest_id": "c" * 64, "inputs": rows,
                "envelope": {
                    "checkpoint_path": str(checkpoint.resolve()), "retry_count": 0,
                    "retry_total_limit": 0, "identity_workers": 1, "answer_workers": 1,
                    "identity_rpm": 100000, "answer_rpm": 100000, "answer_tpm": 900000,
                },
            }
            identity, answer, retriever = IdentityProvider(), AnswerProvider(), self.retriever()
            args = {"manifest": manifest, "checkpoint": checkpoint, "retriever": retriever,
                    "integrity_check": lambda: None}
            result = execute_full(**args, identity_provider=identity, answer_provider=answer)
            self.assertTrue(result["complete"])
            self.assertEqual(result["judge_calls"], 0)
            self.assertEqual(result["predictions"]["mcq:1"], [0])
            self.assertEqual(execute_full(**args, identity_provider=None, answer_provider=None), result)
            self.assertEqual(audit_full(**args), result)

    def test_full_manifest_has_all_inputs_and_no_private_credential_metadata(self):
        retriever = self.retriever()
        with tempfile.TemporaryDirectory() as directory:
            manifest = build_full_manifest(
                retriever=retriever, profiles=profiles(),
                checkpoint=Path(directory) / "checkpoint.sqlite3",
            )
            self.assertEqual(len(manifest["inputs"]), 3789)
            self.assertEqual(manifest["envelope"]["counts"]["judge"], 0)
            self.assertEqual(verify_full_manifest(manifest, retriever=retriever, profiles=profiles()),
                             manifest["manifest_id"])
            for profile in manifest["method"]["profiles"]:
                self.assertEqual(set(profile), {"profile_id", "model_id", "revision", "base_url"})
            changed = copy.deepcopy(manifest)
            changed["inputs"][0]["source_item"]["question"] = "changed"
            with self.assertRaises(ValueError):
                verify_full_manifest(changed, retriever=retriever, profiles=profiles())

    def test_credentials_are_environment_only_and_not_public_identity(self):
        identity, answer = profiles()
        self.assertNotIn("credential", json.dumps(identity.public_identity()))
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "credentials missing"):
                remote_factory(identity, answer)


if __name__ == "__main__":
    unittest.main()
