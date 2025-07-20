import cv2
import numpy as np
try:
    import sounddevice as sd
except Exception:  # pragma: no cover - audio optional
    sd = None
import threading


def open_camera():
    """Open the default camera and return the capture object."""
    return cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)


def read_frame(cap):
    """Read a single frame from an open ``cv2.VideoCapture``."""
    if cap is None:
        return None
    ret, frame = cap.read()
    if not ret:
        return None
    return frame


def show_frame(frame, text=None):
    """Display a frame with optional overlay text."""
    if frame is None:
        return
    if text:
        cv2.putText(
            frame,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
    cv2.imshow("camera", frame)
    cv2.waitKey(1)


def close_camera(cap):
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

def mic_callback(indata, frames, time, status):
    peak = np.abs(indata).max()
    print('Mic peak', peak)

def audio_input_stream(callback, duration=None):
    if sd is None:
        raise RuntimeError("sounddevice not available")
    with sd.InputStream(callback=callback, channels=1) as stream:
        if duration is None:
            threading.Event().wait()
        else:
            sd.sleep(int(duration * 1000))

def record_audio(duration=1.0, samplerate=16000):
    """Record a short audio clip and return (audio, amplitude)."""
    if sd is None:
        raise RuntimeError("sounddevice not available")
    audio = sd.rec(
        int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32"
    )
    sd.wait()
    audio = audio.flatten()
    amplitude = float(np.abs(audio).max()) if audio.size else 0.0
    return audio, amplitude

def play_tone(frequency, duration, samplerate=44100):
    if sd is None:
        raise RuntimeError("sounddevice not available")
    t = np.linspace(0, duration, int(samplerate * duration), False)
    tone = np.sin(frequency * t * 2 * np.pi)
    sd.play(tone, samplerate)
    sd.wait()
