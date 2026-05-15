import threading
import warnings
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

from iris.model import load_and_train, predict_top3 
from iris.voice import listen
from iris.llm import get_diagnosis, get_diet

SYMPTOM_KEYWORDS = ("symptom", "predict", "diagnose", "sick", "feeling", "pain", "unwell", "disease")
DISCLAIMER = "\n Disclaimer: This is not a substitute for professional medical advice. Please consult a doctor."
LOW_CONFIDENCE_THRESHOLD = 40.0

def warm_up():
    import requests, os, time
    try:
        requests.post(os.environ.get("HF_SPACE_URL"), json={"data": ["Warm up"]}, timeout=10)
        print("LLM warmed up.")
    except:
        pass

def handle_symptoms(model, cols, le):
    symptoms_text = listen("Please describe your symptoms.")
    if not symptoms_text:
        print("Iris: No symptoms detected. Please try again.")
        return
    
    top3 = predict_top3(model, cols, symptoms_text, le)
    top = top3[0]

    print(f"\nML Prediction:")
    for i, p in enumerate(top3, 1):
        print(f"{i}. {p['disease']} with confidence {p['confidence']:.2f}%")

    if top["confidence"] < LOW_CONFIDENCE_THRESHOLD:
        print("\nIris: The confidence in the predictions is low. Please consult a doctor directly.")
        return
    
    print(f"\nIris: Fetching diagnosis for {top['disease']}... please wait.")
    response = get_diagnosis(symptoms_text, top3)
    print(f"\nIris: {response}{DISCLAIMER}")

    reply = listen("Would you like dietary recommendations for managing this condition? (yes/no)")
    if reply and reply.lower() in ("yes", "y"):
        diet = get_diet(top["disease"])
        print(f"\nIris: {diet}{DISCLAIMER}")

def handle_diet():
    concern = listen("What is your dietary concern or condition?")
    if not concern:
        print("Iris: I didn't catch that. Please try again.")
        return
    print("Iris: Preparing your diet chart, please wait.")
    chart = get_diet(concern)
    print(f"\nIris: {chart}{DISCLAIMER}")

def conversation(model, cols, le, name):
    while True:
        intent = listen(f"Hello {name}. Say 'symptoms' to get a diagnosis or 'diet' for a diet chart.")
        if not intent:
            print("Iris: I didn't catch that. Please try again.")
            continue

        lower = intent.lower()

        if any(w in lower for w in ("stop", "exit", "quit", "goodbye")):
            print("Iris: Goodbye! Take care.")
            return
        
        if "diet" in lower:
            handle_diet()
        elif any(w in lower for w in SYMPTOM_KEYWORDS):
            handle_symptoms(model, cols, le)
        else:
            print("Iris: Sorry, I didn't understand that. Please say 'symptoms' or 'diet'.")
            continue

        reply = listen("Would you like to continue? (yes/no)")
        if not reply or reply.lower() not in ("yes", "y"):
            print("Iris: Goodbye! Take care.")
            return
        
def run():
    print("Loading and training model, please wait...")
    model, cols, le = load_and_train()
    print("Model is ready.")

    threading.Thread(target=warm_up, daemon=True).start()

    WAKE_WORDS = ("hello iris", "hi iris", "hey iris")
    EXIT_WORDS = ("stop", "exit", "quit", "goodbye")

    print("Say 'Hello Iris' to wake me up and start the conversation.")
    while True:
        text = listen()
        if not text:
            continue

        lower = text.lower()

        if any(w in lower for w in EXIT_WORDS):
            print("Iris: Goodbye! Take care.")
            break

        if any(w in lower for w in WAKE_WORDS):
            name = listen("Iris: Hello! What is your name?")
            name = name.strip().split()[0].capitalize() if name else "there"
            conversation(model, cols, le, name)
            break

        print("Iris: Please say 'Hello Iris' to start the conversation.")

if __name__ == "__main__":
    run()
