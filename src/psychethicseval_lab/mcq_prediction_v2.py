"""Deterministic MCQ wire-format parsing without semantic answer selection."""
import json
import re

PARSER_ID = 'mcq-prediction-v2'
PARSER_VERSION = '2.0'
_FENCE = re.compile(r'\A```(?:json)?[ \t]*\r?\n(?P<body>.*?)\r?\n```\Z', re.DOTALL)
_DECIMAL = re.compile(r'\A(?:0|[1-9][0-9]*)\Z')


def parse_mcq_prediction_v2(content, option_count):
    """Parse one entire JSON array; preserve selection order and reject ambiguity.

    Canonical decimal strings are a wire-format alternative to integer indices.
    Empty selections are retained without judging whether the answer is correct.
    """
    if not isinstance(content, str) or type(option_count) is not int or option_count < 1:
        raise ValueError('invalid MCQ content or option count')
    text = content.strip()
    fence = _FENCE.fullmatch(text)
    if fence:
        text = fence.group('body')
    try:
        value = json.loads(text, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    except (TypeError, ValueError) as error:
        raise ValueError('MCQ requires one whole JSON array') from error
    if not isinstance(value, list):
        raise ValueError('MCQ requires an array')
    indices = []
    for item in value:
        if type(item) is int:
            index = item
        elif isinstance(item, str) and _DECIMAL.fullmatch(item):
            index = int(item)
        else:
            raise ValueError('MCQ index must be integer or canonical decimal string')
        if not 0 <= index < option_count:
            raise ValueError('MCQ index outside option range')
        indices.append(index)
    if len(set(indices)) != len(indices):
        raise ValueError('duplicate MCQ indices')
    return indices
