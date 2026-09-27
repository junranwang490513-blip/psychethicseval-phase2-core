"""Load the two final Phase Two prompts from the human-readable registry."""
import json
from pathlib import Path
import re


PROMPT_PATTERN = re.compile(
    r"<!--\s*PROMPT\s+(?P<meta>\{.*?\})\s*-->\s*```text\s*\n(?P<body>.*?)\n```",
    re.DOTALL,
)
EXPECTED = {
    "B1-MCQ-v1.0": "89d034833627f2b4a5971988579e71b261cebdb0a8221abf25d1a0dc8ceec3ba",
    "B2-OEQ-v1.0": "67259345ad6d4b43bfe4ddf344fa282157e094d027c9a12e97a302528104e911",
}


def load_phase2_prompt(path: str | Path, prompt_id: str) -> str:
    from .identity_classifier_v8 import sha

    if prompt_id not in EXPECTED:
        raise ValueError("unsupported Phase Two prompt")
    text = Path(path).read_text(encoding="utf-8")
    matches = []
    for match in PROMPT_PATTERN.finditer(text):
        metadata = json.loads(match.group("meta"))
        if metadata.get("id") == prompt_id:
            matches.append(match.group("body").strip())
    if len(matches) != 1 or sha(matches[0]) != EXPECTED[prompt_id]:
        raise ValueError("Phase Two prompt drift")
    return matches[0]
