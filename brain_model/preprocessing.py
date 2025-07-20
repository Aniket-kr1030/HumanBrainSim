import numpy as np
import cv2


def text_to_vector(text: str, dim: int) -> np.ndarray:
    """Encode text as a simple letter-frequency vector."""
    counts = np.zeros(dim, dtype=float)
    text = text.lower()
    for ch in text:
        if 'a' <= ch <= 'z':
            idx = (ord(ch) - ord('a')) % dim
            counts[idx] += 1.0
    if counts.sum() > 0:
        counts /= counts.sum()
    return counts


def frame_to_vector(frame: np.ndarray, dim: int) -> np.ndarray:
    """Downsample an image frame into a feature vector."""
    if frame is None:
        return np.zeros(dim, dtype=float)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (dim, 1)).flatten()
    return resized.astype(float) / 255.0
