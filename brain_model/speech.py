import numpy as np
try:
    import speech_recognition as sr
except ImportError:  # pragma: no cover - optional
    sr = None

try:
    import pyttsx3
except ImportError:  # pragma: no cover - optional
    pyttsx3 = None

class SpeechRecognitionModule:
    """Simple wrapper around the speech_recognition library."""
    def __init__(self):
        if sr is not None:
            self.recognizer = sr.Recognizer()
        else:
            self.recognizer = None

    def transcribe(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> str:
        if self.recognizer is None:
            return ""
        audio_data = sr.AudioData(audio_chunk.tobytes(), sample_rate, 2)
        try:
            return self.recognizer.recognize_google(audio_data)
        except Exception:
            return ""

class TextToSpeechModule:
    """Wrapper around pyttsx3 for offline TTS."""
    def __init__(self):
        if pyttsx3 is not None:
            self.engine = pyttsx3.init()
        else:
            self.engine = None

    def speak(self, text: str):
        if self.engine is not None:
            self.engine.say(text)
            self.engine.runAndWait()
        else:
            print(text)
