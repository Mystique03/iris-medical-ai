import json
import os
import re
import threading
import time
import requests

from iris.security import validate_output

# Groq call

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_URL = ""
GROQ_MODEL = ""

SUFFICIENT_PROMPT = """You are a medical triage assistant. \
Decide whether a patient has described enough information for a preliminary assessment.
 
Rules:
- sufficient=true  : patient mentioned at least 2 specific symptoms, OR 1 symptom with duration or severity
- sufficient=false : input is vague, too short, or missing key details
 
Reply ONLY with valid JSON — no explanation, no markdown:
{"sufficient": true or false, "question": "short follow-up question if not sufficient, else null"}"""

def _call_groq(user_input, retries = 2):
    """ Call the Groq API with retry logic."""

    if not GROQ_API_KEY:
        raise RuntimeError("Groq API key not found.")
    
    headers = {
        "Authorization" : f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SUFFICIENT_PROMPT},
            {"role": "user", "content": user_input},
        ],
        "max_tokens": 150,
        "temperature": 0.2,
    }

    for attempt in range(retries):
        try: 
            response = requests.post(
                GROQ_URL,
                headers=headers,
                json=payload,
                timeout=30
            )
            if response.status_code == 429:
                print("Groq API rate limit hit. Retrying after delay...")
                time.sleep(5)
                continue
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        
        except Exception as e:
            print(f"Error during Groq API call (attempt {attempt + 1}): {e}")
            if attempt < retries - 1:
                time.sleep(3)
    return None

def check_sufficiency(user_input):
    """Check if the user's input is sufficient for a preliminary assessment."""
    
    raw = _call_groq(user_input)
    if not raw:
        return {"sufficient": True, "question": None}
 
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except (json.JSONDecodeError, AttributeError):
        pass
 
    return {"sufficient": True, "question": None}

# Fine Tuned LLM call

HF_SPACE_URL = (
    os.environ.get("HF_SPACE_URL", "http://localhost:7860")
    .split("/run/")[0]
    .split("/gradio_api")[0]
    .rstrip("/")
)

MAX_CALLS_PER_MINUTE = 10
call_times: list = []
_rate_lock = threading.Lock()

DIAGNOSIS_PROMPT = """You are Iris, a medical assistant providing preliminary symptom assessments.
 
Patient's description: {raw_input}
 
Provide:
1. Most likely condition(s) with brief reasoning
2. Recommended next steps and home care
3. Red flag symptoms that need immediate attention
 
Be specific and helpful."""

def check_rate_limits():
    """Check if we are within the allowed call rate limits."""
    now = time.time()
    with _rate_lock:
        call_times[:] = [t for t in call_times if now - t < 60]
        if len(call_times) >= MAX_CALLS_PER_MINUTE:
            raise RuntimeError(
                "Too many requests. Please wait a moment and try again."
            )
        call_times.append(now)

def _call_hf(prompt, retries = 2):
    """Call the Hugging Face Space API with retry logic."""
    
    submit_url = f"{HF_SPACE_URL}/gradio_api/call/respond"

    for attempt in range(retries):
        try:
            response  = requests.post(
                submit_url, json={"data": [prompt]}, timeout=60
            )
            response.raise_for_status()
            event_id = response.json().get("event_id")
            if not event_id:
                raise ValueError("No event_id in submit response.")

            poll_url = f"{HF_SPACE_URL}/gradio_api/call/respond/{event_id}"
            with requests.get(poll_url, stream=True, timeout=300) as stream:
                stream.raise_for_status()
                for line in stream.iter_lines():
                    if not line:
                        continue
                    decoded = line.decode("utf-8")
                    if decoded.startswith("data: "):
                        payload = json.loads(decoded[6:])
                        if isinstance(payload, list) and payload:
                            return payload[0]
        except Exception as e:
            print(f"Error during API call (attempt {attempt + 1}): {e}")
            if attempt < retries - 1:
                time.sleep(3)
    
    return "Medical assistance is currently unavailable. Please try again later."

def get_diagnosis(input):
    """Get a diagnosis from the LLM based on the patient's input."""
    check_rate_limits()
    prompt = DIAGNOSIS_PROMPT.format(raw_input=input)
    result = _call_hf(prompt)
    return validate_output(result)
