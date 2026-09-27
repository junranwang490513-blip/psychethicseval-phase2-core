"""One whole [json] wrapper over the unchanged v2 MCQ format decoder."""
import re
from .mcq_prediction_v2 import parse_mcq_prediction_v2

PARSER_ID = 'mcq-prediction-v3'
PARSER_VERSION = '3.0'
_WRAPPER = re.compile(r'\A\[json\](?P<body>.*?)\[/json\]\Z', re.DOTALL | re.IGNORECASE | re.ASCII)
_TAG = re.compile(r'\[/?json\]', re.IGNORECASE | re.ASCII)


def parse_mcq_prediction_v3(content, option_count):
    if isinstance(content, str):
        match = _WRAPPER.fullmatch(content.strip())
        if match:
            body = match.group('body').strip()
            if _TAG.search(body) or not body.startswith('['):
                raise ValueError('MCQ permits one outer json wrapper only')
            return parse_mcq_prediction_v2(body, option_count)
    return parse_mcq_prediction_v2(content, option_count)
