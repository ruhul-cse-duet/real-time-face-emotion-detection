# app/app.py

import os
import sys
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image

import av
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    RTCConfiguration,
    VideoProcessorBase,
)

# ---------------- WebRTC config ---------------- #

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
        ]
    }
)

# ---------------- Project path & model import ---------------- #

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.emotion_service import detect_and_classify_faces, device   # noqa: E402


# ---------------- Page config & CSS ---------------- #

st.set_page_config(page_title="Face Emotion Detection", layout="wide")

css_path = os.path.join("assets", "style.css")
try:
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass  # App will still run without custom CSS


# ---------------- Simple page state ---------------- #

if "page" not in st.session_state:
    st.session_state["page"] = "home"


def set_page(p: str):
    st.session_state["page"] = p


# ---------------- Top nav buttons (no sidebar) ---------------- #

col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 1])
with col_nav1:
    if st.button("🏠 Home"):
        set_page("home")
with col_nav2:
    if st.button("📷 Image Upload"):
        set_page("upload")
with col_nav3:
    if st.button("🎥 Real-Time Camera"):
        set_page("camera")

st.markdown("<hr>", unsafe_allow_html=True)


# ---------------- Helper: pretty print face results ---------------- #

def render_face_results(faces_data):
    if not faces_data:
        st.warning("No face detected.")
        return

    for i, face in enumerate(faces_data, start=1):
        label = face["label"]
        conf = face["confidence"]
        summary = face["summary"]
        action = face["action"]
        probs = face["probabilities"]

        st.markdown(f"### Face {i}")
        st.write(f"**Detected Emotion:** {label} ({conf*100:.2f}%)")

        if summary:
            st.info(summary)
        if action:
            st.caption(action)

        st.write("**Probabilities:**")
        for emo, prob in probs.items():
            st.progress(int(prob * 100), text=f"{emo}: {prob*100:.1f}%")

        st.markdown("---")


# ---------------- WebRTC Video Processor ---------------- #

class EmotionProcessor(VideoProcessorBase):
    """
    Video processor used by streamlit-webrtc.
    It runs in a separate thread and updates attributes
    that we read from the Streamlit script.
    """

    def __init__(self):
        self.faces_data = []        # current-frame detections
        self.last_faces_data = []   # last non-empty detections

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # av.VideoFrame -> numpy BGR
        img_bgr = frame.to_ndarray(format="bgr24")

        annotated_bgr, faces_data = detect_and_classify_faces(img_bgr)

        # store current detections
        self.faces_data = faces_data

        # if we detected something, remember it
        if faces_data:
            self.last_faces_data = faces_data

        # back to VideoFrame for display
        return av.VideoFrame.from_ndarray(annotated_bgr, format="bgr24")


# ===================== PAGES ===================== #

# ---------- HOME ---------- #
if st.session_state["page"] == "home":
    st.markdown(
        """
        <div class="hero">
            <h1 class="title">Real-Time Face Emotion Detection</h1>
            <p class="subtitle">
                Detect <b>Happy</b>, <b>Sad</b>, <b>Angry</b>, <b>Fear</b>, and <b>Surprise</b>
                from face images or webcam in real time with bounding boxes.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(
        """
        Choose a mode above:

        - **Image Upload**: Upload a photo and get annotated bounding boxes + emotion probabilities.
        - **Real-Time Camera**: Use your webcam for continuous emotion detection.

        ⚠️ For educational/demo purposes only — not for medical or psychological diagnosis.
        """
    )
    st.info(f"Running on device: **{device}**")


# ---------- IMAGE UPLOAD ---------- #
elif st.session_state["page"] == "upload":
    st.header("📷 Image Upload Emotion Detection")

    uploaded = st.file_uploader(
        "Upload a face image (JPG/PNG)", type=["jpg", "jpeg", "png"]
    )

    if uploaded is not None:
        pil_image = Image.open(uploaded).convert("RGB")
        st.image(pil_image, caption="Uploaded Image", width=400)

        if st.button("Predict Emotion", type="primary"):
            with st.spinner("Running detection..."):
                frame_rgb = np.array(pil_image)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                annotated_bgr, faces_data = detect_and_classify_faces(frame_bgr)

                h, w = annotated_bgr.shape[:2]
                target_width = 480
                scale = target_width / w
                target_height = int(h * scale)
                annotated_small = cv2.resize(
                    annotated_bgr, (target_width, target_height)
                )

                annotated_rgb = cv2.cvtColor(annotated_small, cv2.COLOR_BGR2RGB)

                col1, col2 = st.columns([1.2, 1])
                with col1:
                    st.image(
                        annotated_rgb,
                        caption="Annotated Image with Bounding Boxes",
                        width="content",
                    )
                with col2:
                    render_face_results(faces_data)
    else:
        st.info("Please upload an image to begin.")


# ---------- REAL-TIME CAMERA (WebRTC) ---------- #
elif st.session_state["page"] == "camera":
    st.header("🎥 Real-Time Webcam Emotion Detection")

    st.write(
        "This mode uses your browser camera via WebRTC, "
        "so it works both locally and on Streamlit Community Cloud."
    )

    left, mid, right = st.columns([1, 2, 1])

    # ---- Live stream widget ---- #
    with mid:
        st.markdown("#### Live Camera Stream")
        webrtc_ctx = webrtc_streamer(
            key="emotion-stream",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=EmotionProcessor,
        )

    # ---- Probabilities for all detected faces ---- #
    with mid:
        st.markdown("#### Latest Detection (Probability Classifier)")

        if webrtc_ctx and webrtc_ctx.video_processor:
            processor = webrtc_ctx.video_processor

            # Prefer last non-empty detections (to avoid flicker)
            faces_data = getattr(processor, "last_faces_data", None)
            if not faces_data:
                faces_data = getattr(processor, "faces_data", [])

            if faces_data:
                for idx, face in enumerate(faces_data, start=1):
                    label = face["label"]
                    conf = face["confidence"]
                    probs = face["probabilities"]

                    with st.container():
                        st.markdown(
                            f"##### Face {idx}: `{label}` "
                            f"({conf * 100:.2f}% confidence)"
                        )

                        for emo, prob in probs.items():
                            st.progress(
                                int(prob * 100),
                                text=f"{emo}: {prob * 100:.1f}%",
                            )

                        st.markdown("---")
            # else:
            #     st.info("No face detected yet. Move closer to the camera 🙂")
        else:
            st.info(
                "Camera is not running. Click **Start** in the video widget "
                "above and allow camera permission."
            )

# streamlit run app/app.py
