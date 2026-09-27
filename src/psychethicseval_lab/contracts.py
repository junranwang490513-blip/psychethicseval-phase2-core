"""Small, credential-free contracts shared by the Phase Two pipeline."""
from dataclasses import dataclass
import hashlib
import json


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ClassifierCall:
    output_text: str
    response_model: str
    revision: str
    usage: dict
    latency_ms: int
    finish_reason: str | None


class ProviderResponseError(ValueError):
    """A provider response failed structural validation."""


def failure_fact(error):
    # Exception messages may contain request URLs or credentials. Persist only type.
    return {"error_type": type(error).__name__}


def validate_answer_response(response, expected_model):
    try:
        value = response["response"]
        choice = value["choices"][0]
        return (
            json.loads(response["raw_response"]) == value
            and value["model"] == expected_model
            and len(value["choices"]) == 1
            and choice["finish_reason"] in ("stop", "length")
            and isinstance(choice["message"]["content"], str)
            and bool(choice["message"]["content"].strip())
        )
    except (KeyError, TypeError, ValueError, IndexError):
        return False
