"""Input validation and output sanitisation for Iris."""

import re

MAX_INPUT_LENGTH = 1000

_INJECTION_PATTERNS = [
    r"\bignore (previous|all|above) instructions\b",
    r"\bsystem prompt\b",
    r"\byou are now\b",
    r"\bforget (everything|all|your)\b",
    r"\bact as\b(?! if)",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"\bjailbreak\b",
    r"\bpretend you\b",
]

_HARMFUL_OUTPUT_PATTERNS = [
    r"kill yourself",
    r"you will die",
    r"no hope",
    r"take (all|extra) medication",
    r"overdose",
]


def sanitize_input(text):
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def validate_user_input(text):
    """Returns (cleaned_text, error_message). error_message is None if valid."""
    if not text or not text.strip():
        return None, "Empty input."
    text = sanitize_input(text)
    if len(text) > MAX_INPUT_LENGTH:
        return None, f"Input too long (max {MAX_INPUT_LENGTH} characters)."
    lower = text.lower()
    if any(re.search(p, lower) for p in _INJECTION_PATTERNS):
        return None, "Invalid input detected. Please describe your symptoms."
    return text, None


def validate_output(text):
    """Scrub harmful LLM output patterns. Returns safe text."""
    lower = text.lower()
    for pattern in _HARMFUL_OUTPUT_PATTERNS:
        if re.search(pattern, lower):
            print(f"[security] blocked output matching pattern: {pattern!r}")
            return "Response could not be processed. Please consult a medical professional directly."
    return text
