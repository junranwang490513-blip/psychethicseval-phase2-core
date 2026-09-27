"""Minimal OpenAI-compatible transports for the final Phase Two models.

Credentials are read only at execution time from generic environment variables.
They are never included in manifests, logs, checkpoints, errors, or exports.
"""
from dataclasses import dataclass
import json
import os
import time
import urllib.error
import urllib.request

from .contracts import ClassifierCall, ProviderResponseError, canonical


@dataclass(frozen=True, slots=True)
class TransportProfile:
    profile_id: str
    model_id: str
    revision: str
    base_url: str
    credential_env: str

    def public_identity(self):
        return {
            "profile_id": self.profile_id,
            "model_id": self.model_id,
            "revision": self.revision,
            "base_url": self.base_url,
        }


def profiles():
    return (
        TransportProfile(
            "identity-qwen-v13",
            "qwen3.8-flash",
            "dashscope:qwen3.8-flash",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "PHASE2_IDENTITY_CREDENTIAL",
        ),
        TransportProfile(
            "answer-deepseek-v4",
            "deepseek-v4-flash-ga-260731",
            "volcengine-ark:260731",
            "https://ark.cn-beijing.volces.com/api/v3",
            "PHASE2_ANSWER_CREDENTIAL",
        ),
    )


class ExactPayloadProvider:
    def __init__(self, profile, credential, *, identity):
        self.profile = profile
        self._credential = credential
        self.identity = identity

    def complete(self, payload):
        if payload.get("model") != self.profile.model_id:
            raise ValueError("materialized model/profile mismatch")
        request = urllib.request.Request(
            self.profile.base_url + "/chat/completions",
            data=canonical(payload).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + self._credential},
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            error.read()  # drain without persisting the provider body
            raise ProviderResponseError("provider HTTP failure") from error
        try:
            value = json.loads(raw)
        except ValueError as error:
            raise ProviderResponseError("provider response is not JSON") from error
        if not self.identity:
            return {
                "raw_response": raw,
                "response": value,
                "latency_ms": round((time.monotonic() - started) * 1000),
            }
        try:
            choice = value["choices"][0]
            if len(value["choices"]) != 1 or not isinstance(choice["message"]["content"], str):
                raise ValueError("invalid identity choices")
            return ClassifierCall(
                output_text=choice["message"]["content"],
                response_model=value.get("model", ""),
                revision=self.profile.revision,
                usage={**value.get("usage", {}), "provider_raw_response": raw},
                latency_ms=round((time.monotonic() - started) * 1000),
                finish_reason=choice.get("finish_reason"),
            )
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ProviderResponseError("invalid identity response envelope") from error


def remote_factory(identity_profile, answer_profile):
    values = [os.environ.get(p.credential_env, "").strip() for p in (identity_profile, answer_profile)]
    if not all(values):
        names = ", ".join(p.credential_env for p in (identity_profile, answer_profile))
        raise ValueError("execution credentials missing; set " + names)
    return (
        ExactPayloadProvider(identity_profile, values[0], identity=True),
        ExactPayloadProvider(answer_profile, values[1], identity=False),
    )
