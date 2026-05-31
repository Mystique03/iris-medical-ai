"""Speech-to-text input for Iris using Google Speech Recognition."""

import speech_recognition as sr

_recognizer = sr.Recognizer()

def listen():
    """Listen to the user's voice and convert it to text."""
    
    try:
        with sr.Microphone() as source:
            _recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print("Listening...")
            audio = _recognizer.listen(source, timeout=10, phrase_time_limit=10)

        text = _recognizer.recognize_google(audio)
        print(f"You: {text}")
        return text
    
    except (sr.WaitTimeoutError, sr.UnknownValueError):
        print("Sorry, I didn't catch that. Please try again.")
        return None
    except sr.RequestError as e:
        print(f"Speech recognition service error: {e}")
        return None
