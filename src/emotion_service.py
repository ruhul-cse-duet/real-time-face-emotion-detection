
import logging
import os
from typing import List, Dict, Any

import cv2
from cv2 import data as cv2_data
import numpy as np
from PIL import Image
import torch

from src.custom_resnet import prediction_img

logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_LABELS = ["Fear", "Surprise", "Angry", "Sad", "Happy"]

EMOTION_DESCRIPTIONS = {
    "Angry": (
        "😠 Tension across the brow or jaw indicates heightened arousal.",
        "Offer decompression exercises or tag the clip for escalation review.",
    ),
    "Fear": (
        "😨 Raised brows and widened eyes usually signal fear or anxiety.",
        "Provide reassurance cues or log the moment for sentiment tracking.",
    ),
    "Happy": (
        "😊 Relaxed eyes and lifted cheeks suggest positive engagement.",
        "Celebrate the interaction or store as a positive training sample.",
    ),
    "Sad": (
        "😔 Downward gaze or lip corners often align with sadness.",
        "Consider empathy workflows or proactive outreach.",
    ),
    "Surprise": (
        "😮 Sudden eye and mouth widening typically indicate surprise.",
        "Verify the cause—surprise can precede both delight and concern.",
    ),
}

# Use OpenCV’s built-in Haar cascade
CASCADE_PATH = os.path.join(
    cv2_data.haarcascades, "haarcascade_frontalface_default.xml"
)
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    raise RuntimeError(f"Failed to load Haar cascade from: {CASCADE_PATH}")


def preprocess_face_image(pil_img: Image.Image) -> torch.Tensor:
    """Resize + normalize face image same as training (384x384, ImageNet mean/std)."""
    img_resized = pil_img.resize((384, 384))
    img_array = np.array(img_resized).astype(np.float32) / 255.0

    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)

    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # (C,H,W)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std

    return img_tensor.unsqueeze(0)  # (1,C,H,W)


# def detect_and_classify_faces(
#     frame_bgr: np.ndarray,
# ) -> (np.ndarray, List[Dict[str, Any]]):
#     """
#     Detect faces using Haar cascade, run prediction_img on each face,
#     draw bounding boxes + labels on the frame.
#
#     Returns:
#         annotated_frame_bgr, faces_data_list
#     """
#     # 1) Gray + equalize for low light (tomar screenshot-e room dark chhilo)
#     gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
#     gray = cv2.equalizeHist(gray)
#
#     # 2) Face detection (more permissive params)
#     faces = face_cascade.detectMultiScale(
#         gray,
#         scaleFactor=1.1,       # choto step, beshi sensitive
#         minNeighbors=4,        # slightly relaxed
#         minSize=(30, 30),      # choto face-o dhorbe
#         flags=cv2.CASCADE_SCALE_IMAGE,
#     )
#
#     faces_data: List[Dict[str, Any]] = []
#
#     # 👉 IMPORTANT: no fallback. If no faces -> no prediction.
#     if len(faces) == 0:
#         logging.info("No faces detected in this frame.")
#         annotated = frame_bgr.copy()
#         return annotated, faces_data
#
#     # 3) Ekta ekta face crop kore classify
#     for (x, y, w, h) in faces:
#         face_roi = frame_bgr[y : y + h, x : x + w]
#         rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
#         pil_face = Image.fromarray(rgb)
#
#         img_tensor = preprocess_face_image(pil_face)
#
#         with torch.no_grad():
#             pred_idx_tensor, prob_tensor = prediction_img(img_tensor)
#
#         if isinstance(pred_idx_tensor, torch.Tensor):
#             pred_idx = int(pred_idx_tensor.squeeze().item())
#         else:
#             pred_idx = int(pred_idx_tensor)
#
#         probs = prob_tensor.squeeze().tolist()
#         if isinstance(probs, float):
#             probs = [probs]
#
#         label = CLASS_LABELS[pred_idx]
#         confidence = float(probs[pred_idx]) if pred_idx < len(probs) else 0.0
#         summary, action = EMOTION_DESCRIPTIONS.get(label, ("", ""))
#
#         faces_data.append(
#             {
#                 "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
#                 "label": label,
#                 "confidence": confidence,
#                 "probabilities": {
#                     CLASS_LABELS[i]: float(probs[i]) if i < len(probs) else 0.0
#                     for i in range(len(CLASS_LABELS))
#                 },
#                 "summary": summary,
#                 "action": action,
#             }
#         )
#
#     # 4) Draw bounding boxes
#     annotated = frame_bgr.copy()
#     for face in faces_data:
#         x, y, w, h = (
#             face["bbox"]["x"],
#             face["bbox"]["y"],
#             face["bbox"]["w"],
#             face["bbox"]["h"],
#         )
#         label = face["label"]
#         conf = face["confidence"]
#
#         cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         text = f"{label} {conf*100:.1f}%"
#         cv2.putText(
#             annotated,
#             text,
#             (x, y - 5),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.6,
#             (0, 255, 0),
#             2,
#         )
#
#     return annotated, faces_data


import logging
import os
from typing import List, Dict, Any

import cv2
import numpy as np
from PIL import Image
import torch

from src.custom_resnet import prediction_img

logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_LABELS = ["Fear", "Surprise", "Angry", "Sad", "Happy"]

EMOTION_DESCRIPTIONS = {
    "Angry": (
        "😠 Tension across the brow or jaw indicates heightened arousal.",
        "Offer decompression exercises or tag the clip for escalation review.",
    ),
    "Fear": (
        "😨 Raised brows and widened eyes usually signal fear or anxiety.",
        "Provide reassurance cues or log the moment for sentiment tracking.",
    ),
    "Happy": (
        "😊 Relaxed eyes and lifted cheeks suggest positive engagement.",
        "Celebrate the interaction or store as a positive training sample.",
    ),
    "Sad": (
        "😔 Downward gaze or lip corners often align with sadness.",
        "Consider empathy workflows or proactive outreach.",
    ),
    "Surprise": (
        "😮 Sudden eye and mouth widening typically indicate surprise.",
        "Verify the cause—surprise can precede both delight and concern.",
    ),
}

# ---------------- DNN Face Detector ---------------- #

DNN_PROTO = os.path.join("assets", "deploy.prototxt")
DNN_MODEL = os.path.join("assets", "res10_300x300_ssd_iter_140000_fp16.caffemodel")

if not (os.path.exists(DNN_PROTO) and os.path.exists(DNN_MODEL)):
    raise FileNotFoundError(
        f"Could not find DNN face model. Expected:\n{DNN_PROTO}\n{DNN_MODEL}"
    )

face_net = cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL)
def detect_and_classify_faces(
    frame_bgr: np.ndarray,
) -> (np.ndarray, List[Dict[str, Any]]):
    """
    Detect faces using OpenCV DNN, run prediction_img on each face,
    draw bounding boxes + labels on the frame.

    Returns:
        annotated_frame_bgr, faces_data_list
    """
    (h, w) = frame_bgr.shape[:2]

    # Create blob for DNN: resize to 300x300, subtract mean, etc.
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame_bgr, (300, 300)),
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0),  # standard for this model
        swapRB=False,
        crop=False,
    )

    face_net.setInput(blob)
    detections = face_net.forward()

    faces_data: List[Dict[str, Any]] = []
    annotated = frame_bgr.copy()

    CONF_THRESH = 0.4  # adjust if needed (0.3–0.7)

    # Loop over detections
    for i in range(0, detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < CONF_THRESH:
            continue

        # Box coords are normalized [0,1] relative to original image
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")

        # Clip to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)

        # Defensive check
        if x2 <= x1 or y2 <= y1:
            continue

        face_roi = frame_bgr[y1:y2, x1:x2]
        if face_roi.size == 0:
            continue

        # BGR -> RGB -> PIL
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
        confidence_cls = float(probs[pred_idx]) if pred_idx < len(probs) else 0.0
        summary, action = EMOTION_DESCRIPTIONS.get(label, ("", ""))

        faces_data.append(
            {
                "bbox": {
                    "x": int(x1),
                    "y": int(y1),
                    "w": int(x2 - x1),
                    "h": int(y2 - y1),
                },
                "label": label,
                "confidence": confidence_cls,
                "probabilities": {
                    CLASS_LABELS[i]: float(probs[i]) if i < len(probs) else 0.0
                    for i in range(len(CLASS_LABELS))
                },
                "summary": summary,
                "action": action,
            }
        )

        # Draw rectangle + label on annotated image
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} {confidence_cls*100:.1f}%"
        cv2.putText(
            annotated,
            text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    # 👉 Important: no fallback. If no faces detected, faces_data = []
    if not faces_data:
        logging.info("No faces detected by DNN in this frame.")

    return annotated, faces_data
