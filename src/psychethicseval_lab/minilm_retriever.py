"""Pinned, offline MiniLM sentence encoder used by the Phase Two retriever."""
import hashlib
import json
from pathlib import Path
import sys

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_LICENSE = "apache-2.0"
MAX_SEQUENCE_LENGTH = 512
MANIFEST_SHA256 = "cdc61b5f3f6d3a8b05bc6b76eeb6a1cd45606ccb1af923e63926452fc544f288"
EXPECTED_VERSIONS = {"onnxruntime": "1.29.0", "tokenizers": "0.23.1", "numpy": "2.4.5"}

ASSET_ROOT = Path(__file__).resolve().parents[2] / "artifacts/semantic_embeddings"
MODEL_DIR = ASSET_ROOT / "minilm-l6-v2"
RUNTIME_DIR = ASSET_ROOT / "runtime"
WHEELS_DIR = ASSET_ROOT / "wheels"
MANIFEST_PATH = ASSET_ROOT / "manifest.json"
MODEL_ONNX = MODEL_DIR / "onnx/model.onnx"
TOKENIZER_JSON = MODEL_DIR / "tokenizer.json"


class SemanticAssetUnavailableError(RuntimeError):
    pass


def _sha256_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def _verify_asset_manifest():
    if not MANIFEST_PATH.is_file() or _sha256_file(MANIFEST_PATH) != MANIFEST_SHA256:
        raise SemanticAssetUnavailableError("semantic asset manifest missing or changed")
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SemanticAssetUnavailableError("semantic asset manifest is invalid") from error
    if (manifest.get("schema") != "semantic-embedding-asset-manifest-v1"
            or manifest.get("asset_id") != "minilm-l6-v2-sentence-embedding-v1"
            or manifest.get("model", {}).get("license") != MODEL_LICENSE):
        raise SemanticAssetUnavailableError("semantic asset identity changed")
    for entry in manifest.get("model_files", []):
        path = MODEL_DIR / entry["path"]
        if (not path.is_file() or path.stat().st_size != entry["size"]
                or _sha256_file(path) != entry["sha256"]):
            raise SemanticAssetUnavailableError("semantic model file missing or changed: " + entry["path"])
    for entry in manifest.get("runtime_wheels", []):
        path = WHEELS_DIR / entry["wheel"]
        if (not path.is_file() or path.stat().st_size != entry["size"]
                or _sha256_file(path) != entry["sha256"]):
            raise SemanticAssetUnavailableError("runtime wheel missing or changed: " + entry["wheel"])


def _assert_runtime_module(module, name):
    origin = getattr(module, "__file__", None)
    if not origin or not Path(origin).resolve().is_relative_to(RUNTIME_DIR.resolve()):
        raise SemanticAssetUnavailableError(name + " was not loaded from the pinned runtime")
    if getattr(module, "__version__", None) != EXPECTED_VERSIONS[name]:
        raise SemanticAssetUnavailableError(name + " version changed")


class _SentenceEncoder:
    def __init__(self):
        runtime = str(RUNTIME_DIR)
        if runtime not in sys.path:
            sys.path.insert(0, runtime)
        _verify_asset_manifest()
        import numpy as np
        import onnxruntime as ort
        import tokenizers

        for module, name in ((ort, "onnxruntime"), (tokenizers, "tokenizers"), (np, "numpy")):
            _assert_runtime_module(module, name)
        self.np = np
        self.runtime_version = ort.__version__
        self._session = ort.InferenceSession(str(MODEL_ONNX), providers=["CPUExecutionProvider"])
        self._tokenizer = tokenizers.Tokenizer.from_file(str(TOKENIZER_JSON))
        self._tokenizer.enable_truncation(max_length=MAX_SEQUENCE_LENGTH)
        self._tokenizer.enable_padding(length=MAX_SEQUENCE_LENGTH)

    def encode(self, text):
        encoding = self._tokenizer.encode(text)
        input_ids = self.np.array([encoding.ids], dtype=self.np.int64)
        attention_mask = self.np.array([encoding.attention_mask], dtype=self.np.int64)
        token_type_ids = self.np.zeros_like(input_ids, dtype=self.np.int64)
        hidden = self._session.run(None, {
            "input_ids": input_ids, "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        })[0][0]
        mask = attention_mask[0][:, None].astype(self.np.float32)
        pooled = (hidden * mask).sum(axis=0) / mask.sum()
        pooled = pooled / (self.np.linalg.norm(pooled) + 1e-9)
        return tuple(float(value) for value in pooled)
