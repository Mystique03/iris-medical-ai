# Entry point for Iris CLI

import os 
import threading
import warnings

from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

from iris.llm import check_sufficiency, get_diagnosis
from iris.security import validate_user_input
from iris.voice import listen

MAX_FOLLOWUP_QUESTIONS = 2
USE_VOICE =False

DISCLAIMER = (
    "\nDisclaimer: This is not a substitute for professional medical advice. "
    "Please consult a doctor."
)

def get_input(prompt):

    if prompt:
        print(f"IRIS: {prompt}")

    if USE_VOICE:
        return listen()
    
    text = input("YOU: ").strip()
    return text or None

def _warm_up() -> None:
    """Send a dummy request to wake the HF Space from cold start."""
    import requests
 
    base = (
        os.environ.get("HF_SPACE_URL", "")
        .split("/run/")[0]
        .split("/gradio_api")[0]
        .rstrip("/")
    )
    try:
        r = requests.post(
            f"{base}/gradio_api/call/respond",
            json={"data": ["Warm up"]},
            timeout=10,
        )
        if r.ok:
            print("LLM warmed up.")
    except Exception:
        pass

def collect_symptoms():
    """Collect symptoms from the user, asking follow-up questions if needed."""
    
    raw = get_input("Please describe your symptoms")
    cleaned, err = validate_user_input(raw)
    if err:
        print(f"\nIRIS: {err}")
        return None
    parts = [cleaned]

    for _ in range(MAX_FOLLOWUP_QUESTIONS):
        combined = " ".join(parts)
        result = check_sufficiency(combined)

        if result.get("sufficient"):
            break

        question = result.get("question")
        if not question:
            break

        follow_up = get_input(question)
        if not follow_up:
            break

        cleaned_followup, err = validate_user_input(follow_up)
        if err:
            print(f"\nIRIS: {err}")
            break

        parts.append(cleaned_followup)
    return " ".join(parts)

def handle_session():
    symptoms = collect_symptoms()
    if not symptoms:
        return
 
    print("\nIris: Analyzing, please wait...")
 
    try:
        diagnosis = get_diagnosis(symptoms)
    except RuntimeError as e:
        print(f"\nIris: {e}")
        return
 
    print(f"\nIris: {diagnosis}")
    print(DISCLAIMER)

def run_conversation():
    while True:
        handle_session()
 
        reply = get_input("Would you like to describe another concern? (yes / no)")
        if not reply or reply.lower() not in ("yes", "y"):
            print("\nIris: Goodbye! Take care.")
            return
        
def main():
    threading.Thread(target=_warm_up, daemon=True).start()
    run_conversation()

if __name__ == "__main__":
    main()

