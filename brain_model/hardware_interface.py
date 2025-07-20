import cv2
import numpy as np
import sounddevice as sd
import threading


def camera_capture():
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def mic_callback(indata, frames, time, status):
    peak = np.abs(indata).max()
    print('Mic peak', peak)

def audio_input_stream(callback, duration=None):
    with sd.InputStream(callback=callback, channels=1) as stream:
        if duration is None:
            threading.Event().wait()
        else:
            sd.sleep(int(duration * 1000))

def record_audio(duration=1.0, samplerate=16000):
    """Record a short audio clip and return a numpy array."""
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate,
                   channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()

def play_tone(frequency, duration, samplerate=44100):
    t = np.linspace(0, duration, int(samplerate * duration), False)
    tone = np.sin(frequency * t * 2 * np.pi)
    sd.play(tone, samplerate)
    sd.wait()
