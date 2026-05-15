import speech_recognition as sr

_recognizer = sr.Recognizer()

def listen(prompt):
    if prompt:
        print(f"Iris: {prompt}")
    try:
        with sr.Microphone() as source:
            _recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print("Listening...")
            audio = _recognizer.listen(source, timeout=10, phrase_time_limit=15)
        text = _recognizer.recognize_goole(audio)
        print(f"You: {text}")
        return text
    except (sr.WaitTimeoutError, sr.UnknownValueError):
        print("Sorry, I didn't catch that. Please try again.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None
    
    