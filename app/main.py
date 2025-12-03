# app/main.py

import io
import base64
import logging
import os
from typing import List, Dict, Any
from cv2 import data as cv2_data
CASCADE_PATH = os.path.join(cv2_data.haarcascades, "haarcascade_frontalface_default.xml")

import cv2
import numpy as np
from PIL import Image
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.custom_resnet import prediction_img  # your existing function

logging.basicConfig(level=logging.INFO)

# ---- Config ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_LABELS = ['Fear', 'Surprise', 'Angry', 'Sad', 'Happy']


EMOTION_DESCRIPTIONS = {
    'Angry': (
        "😠 Tension across the brow or jaw indicates heightened arousal.",
        "Offer decompression exercises or tag the clip for escalation review."
    ),
    'Fear': (
        "😨 Raised brows and widened eyes usually signal fear or anxiety.",
        "Provide reassurance cues or log the moment for sentiment tracking."
    ),
    'Happy': (
        "😊 Relaxed eyes and lifted cheeks suggest positive engagement.",
        "Celebrate the interaction or store as a positive training sample."
        ),
    'Sad': (
        "😔 Downward gaze or lip corners often align with sadness.",
        "Consider empathy workflows or proactive outreach."
    ),
    'Surprise': (
        "😮 Sudden eye and mouth widening typically indicate surprise.",
        "Verify the cause—surprise can precede both delight and concern."
    )
}

# ✅ Use OpenCV's built-in haarcascade path
CASCADE_PATH = os.path.join(cv2_data.haarcascades, "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

if face_cascade.empty():
    raise RuntimeError(
        f"Failed to load Haar cascade from: {CASCADE_PATH}"
    )

# ---- FastAPI app ----
app = FastAPI(
    title="Real-Time Face Emotion Detection API",
    description="Detect faces, draw bounding boxes, and classify emotions.",
    version="1.0.0",
)

# Allow CORS for local frontend / JS client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to specific domains in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----- Helpers ----- #

def preprocess_face_image(pil_img: Image.Image) -> torch.Tensor:
    """
    Resize and normalize a PIL face image to the same format
    your Streamlit app uses (384x384, ImageNet mean/std).
    """
    img_resized = pil_img.resize((384, 384))
    img_array = np.array(img_resized).astype(np.float32) / 255.0

    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)

    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # (C,H,W)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std

    return img_tensor.unsqueeze(0)  # add batch dim


def draw_boxes_on_frame(
    frame: np.ndarray,
    faces_data: List[Dict[str, Any]]
) -> np.ndarray:
    """
    Draw bounding boxes and labels on the BGR frame.
    faces_data entries: {bbox: {x,y,w,h}, label, confidence}
    """
    for face in faces_data:
        x = face["bbox"]["x"]
        y = face["bbox"]["y"]
        w = face["bbox"]["w"]
        h = face["bbox"]["h"]
        label = face["label"]
        conf = face["confidence"]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = f"{label} {conf*100:.1f}%"
        cv2.putText(
            frame,
            text,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return frame


def encode_image_to_base64(frame_bgr: np.ndarray) -> str:
    """Encode BGR image to base64 JPEG string."""
    success, buffer = cv2.imencode(".jpg", frame_bgr)
    if not success:
        raise RuntimeError("Failed to encode frame for response.")
    jpg_bytes = buffer.tobytes()
    return base64.b64encode(jpg_bytes).decode("utf-8")


# ----- Routes ----- #

@app.get("/")
def root():
    return {
        "message": "Face Emotion Detection API is running.",
        "endpoints": ["/predict"]
    }


@app.post("/predict")
async def predict_emotions(file: UploadFile = File(...)):
    """
    Upload a single image (jpg/png). Returns:
      - list of faces with bbox, label, confidence, probabilities
      - base64-encoded JPEG with bounding boxes and labels
    """
    contents = await file.read()

    # Decode bytes to OpenCV BGR image
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Could not decode image"}

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face detection
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(60, 60)
    )

    faces_data: List[Dict[str, Any]] = []

    if len(faces) == 0:
        # If no face found, optionally classify full frame
        logging.info("No faces detected; classifying entire image.")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        img_tensor = preprocess_face_image(pil_img)
        with torch.no_grad():
            pred_idx_tensor, prob_tensor = prediction_img(img_tensor)

        if isinstance(pred_idx_tensor, torch.Tensor):
            pred_idx = int(pred_idx_tensor.squeeze().item())
        else:
            pred_idx = int(pred_idx_tensor)

        probs = prob_tensor.squeeze().tolist()
        if isinstance(probs, float):
            probs = [probs]

        label = CLASS_LABELS[pred_idx]
        confidence = float(probs[pred_idx])

        summary, action = EMOTION_DESCRIPTIONS.get(label, ("", ""))

        faces_data.append({
            "bbox": {"x": 0, "y": 0, "w": frame.shape[1], "h": frame.shape[0]},
            "label": label,
            "confidence": confidence,
            "probabilities": {
                CLASS_LABELS[i]: float(probs[i]) if i < len(probs) else 0.0
                for i in range(len(CLASS_LABELS))
            },
            "summary": summary,
            "action": action,
        })
    else:
        # For each detected face, classify emotion
        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            pil_face = Image.fromarray(rgb)

            img_tensor = preprocess_face_image(pil_face)

            with torch.no_grad():
                pred_idx_tensor, prob_tensor = prediction_img(img_tensor)

            if isinstance(pred_idx_tensor, torch.Tensor):
                pred_idx = int(pred_idx_tensor.squeeze().item())
            else:
                pred_idx = int(pred_idx_tensor)

            probs = prob_tensor.squeeze().tolist()
            if isinstance(probs, float):
                probs = [probs]

            label = CLASS_LABELS[pred_idx]
            confidence = float(probs[pred_idx]) if pred_idx < len(probs) else 0.0

            summary, action = EMOTION_DESCRIPTIONS.get(label, ("", ""))

            faces_data.append({
                "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                "label": label,
                "confidence": confidence,
                "probabilities": {
                    CLASS_LABELS[i]: float(probs[i]) if i < len(probs) else 0.0
                    for i in range(len(CLASS_LABELS))
                },
                "summary": summary,
                "action": action,
            })

    # Draw bounding boxes on frame
    frame_boxed = draw_boxes_on_frame(frame.copy(), faces_data)
    annotated_base64 = encode_image_to_base64(frame_boxed)

    return {
        "num_faces": len(faces_data),
        "faces": faces_data,
        "annotated_image_base64": annotated_base64,
    }

# uvicorn app.main:app --reload