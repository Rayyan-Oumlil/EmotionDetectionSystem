"""FastAPI app — serves the demo page and streams emotion predictions over WebSocket."""

from __future__ import annotations

import logging
import os

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from predictor import EmotionPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("emotion-web")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(title="Emotion Detection — Live Web Demo")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

predictor: EmotionPredictor | None = None


@app.on_event("startup")
def _load_model() -> None:
    global predictor
    log.info("Loading EmotionCNN...")
    predictor = EmotionPredictor()
    log.info("Model ready on %s", predictor.device)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/healthz")
def healthz() -> dict[str, object]:
    return {
        "ok": predictor is not None,
        "device": str(predictor.device) if predictor else None,
    }


@app.websocket("/ws/emotion")
async def ws_emotion(ws: WebSocket) -> None:
    await ws.accept()
    if predictor is None:
        await ws.send_json({"error": "model not loaded"})
        await ws.close()
        return

    try:
        while True:
            data = await ws.receive_bytes()
            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                await ws.send_json({"error": "decode failed"})
                continue
            await ws.send_json(predictor.predict(frame))
    except WebSocketDisconnect:
        log.info("client disconnected")
    except Exception:
        log.exception("ws error")
        await ws.close()
