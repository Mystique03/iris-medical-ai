import os 
import time
import requests

HF_SPACE_URL = os.environ.get("HF_SPACE_URL")

DIAGNOSIS_PROMPT = """You are a medical assistant. A patient reports the following symptoms: {symptoms}

An ML Classifier predicts:
1. {d1} with a confidence of {c1:.2f}%
2. {d2} with a confidence of {c2:.2f}%
3. {d3} with a confidence of {c3:.2f}%

Reason over the symptoms and predictions. State the most likely diagnosis and recommend a treatment plan. Be concise and clear.

"""

DIET_PROMPT = """You are a nutritionist. 
For a patient with {condition}, list 5 foods that are beneficial for their condition and 5 foods that should be avoided. Be brief and specific.
"""

def _call(prompt, retries = 2):
    for attempt in range(retries):
        try:
            r = requests.post(HF_SPACE_URL, json={"data": [prompt]}, timeout=40)
            return r.json()["data"][0]
        except Exception as e:
            if attempt == retries - 1:
                return "Medical assistant is currently unavailable. Please try again."
            time.sleep(3)

def get_diagnosis(symptoms, top3):
    prompt = DIAGNOSIS_PROMPT.format(
        symptoms=symptoms,
        d1=top3[0]["disease"], c1=top3[0]["confidence"],
        d2=top3[1]["disease"], c2=top3[1]["confidence"],
        d3=top3[2]["disease"], c3=top3[2]["confidence"],
    )
    return _call(prompt)

def get_diet(condition):
    return _call(DIET_PROMPT.format(condition=condition))