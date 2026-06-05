"""Emotion predictor — loads the trained CNN once and exposes a predict() method."""

from __future__ import annotations

import os
import sys
from typing import TypedDict

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_CNN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "CNN")
sys.path.insert(0, os.path.abspath(_CNN_DIR))
from cnn import EmotionCNN  # noqa: E402

EMOTIONS: tuple[str, ...] = (
    "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise",
)

# Per-class logit bias applied before softmax to counter training-set imbalance
# (disgust had ~1.9k samples, happy ~35k → model under-predicts disgust/fear).
# Values are in logit space. Defaults are calibrated to log(N_max / N_class):
#   disgust:  ln(35775/1880) ~= 2.9  (using 1.5 = noticeable but not dominant)
#   fear:     ln(35775/4097) ~= 2.2  (using 1.2)
# Override at runtime via env, e.g. EMOTION_BIAS_DISGUST=2.0
_DEFAULT_BIAS: dict[str, float] = {
    "angry": 0.0,
    "disgust": 1.5,
    "fear": 1.2,
    "happy": 0.0,
    "neutral": 0.0,
    "sad": 0.0,
    "surprise": 0.0,
}


def _load_bias() -> dict[str, float]:
    return {
        e: float(os.environ.get(f"EMOTION_BIAS_{e.upper()}", _DEFAULT_BIAS[e]))
        for e in EMOTIONS
    }


class Prediction(TypedDict):
    emotion: str
    probabilities: dict[str, float]
    face_box: tuple[int, int, int, int] | None


class EmotionPredictor:
    def __init__(self, model_path: str | None = None) -> None:
        cnn_dir = os.path.abspath(_CNN_DIR)
        self.model_path = model_path or os.path.join(cnn_dir, "emotionCNN.pth")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = EmotionCNN()
        state = torch.load(self.model_path, map_location=self.device, weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade at {cascade_path}")

        bias = _load_bias()
        self.bias_tensor = torch.tensor(
            [bias[e] for e in EMOTIONS], dtype=torch.float32, device=self.device
        )

    def _preprocess(self, gray_face: np.ndarray) -> torch.Tensor:
        face = cv2.resize(gray_face, (48, 48))
        tensor = torch.from_numpy(face / 255.0).float().unsqueeze(0).unsqueeze(0)
        return ((tensor - 0.5) / 0.5).to(self.device)

    def predict(self, bgr_frame: np.ndarray) -> Prediction:
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
        )

        if len(faces) == 0:
            uniform = {e: 1.0 / len(EMOTIONS) for e in EMOTIONS}
            return {"emotion": "neutral", "probabilities": uniform, "face_box": None}

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        tensor = self._preprocess(gray[y:y + h, x:x + w])

        with torch.no_grad():
            logits = self.model(tensor) + self.bias_tensor
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        prob_map = {e: float(p) for e, p in zip(EMOTIONS, probs)}
        return {
            "emotion": max(prob_map, key=prob_map.get),
            "probabilities": prob_map,
            "face_box": (int(x), int(y), int(w), int(h)),
        }
