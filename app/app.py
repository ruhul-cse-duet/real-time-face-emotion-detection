import time
import numpy as np
import cv2
import streamlit as st
from PIL import Image
import sys
import os

# Add ROOT directory to module search path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.emotion_service import detect_and_classify_faces, device

# ---------- ENV: local vs cloud ---------- #
RUN_ENV = os.getenv("RUN_ENV", "local")  # "local" or "cloud"



# ---------- Page config & CSS ---------- #

st.set_page_config(page_title="Face Emotion Detection", layout="wide")

css_path = "assets/style.css"
try:
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass  # app will still run without custom CSS

# ---------- Simple page state ---------- #

if "page" not in st.session_state:
    st.session_state["page"] = "home"

def set_page(p):
    st.session_state["page"] = p

# ---------- Top navigation buttons (no sidebar) ---------- #

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

# ---------- Helper: pretty print face results ---------- #

def render_face_results(faces_data):
    if not faces_data:
        st.warning("No face detected.")
        return

    for i, face in enumerate(faces_data):
        label = face["label"]
        conf = face["confidence"]
        summary = face["summary"]
        action = face["action"]
        probs = face["probabilities"]

        st.markdown(f"### Face {i+1}")
        st.write(f"**Detected Emotion:** {label} ({conf*100:.2f}%)")

        if summary:
            st.info(summary)
        if action:
            st.caption(action)

        st.write("**Probabilities:**")
        for emo, prob in probs.items():
            st.progress(int(prob * 100), text=f"{emo}: {prob*100:.1f}%")

        st.markdown("---")


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


# --------------- IMAGE UPLOAD ------------------------------------ #
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
                # PIL -> BGR
                frame_rgb = np.array(pil_image)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                annotated_bgr, faces_data = detect_and_classify_faces(frame_bgr)

                # 🔹 Reduce display size of image for nicer UI
                h, w = annotated_bgr.shape[:2]
                target_width = 480
                scale = target_width / w
                target_height = int(h * scale)
                annotated_small = cv2.resize(annotated_bgr, (target_width, target_height))

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


# ---------------------- REAL-TIME CAMERA ------------------------------------ #
elif st.session_state["page"] == "camera":
    st.header("🎥 Real-Time Webcam Emotion Detection")

    st.write(
        "Click **Start Camera** to begin live emotion detection. "
        "The preview size is reduced for smoother performance."
    )

    left, mid, right = st.columns([1, 2, 1])

    # ======================= CLOUD MODE (st.camera_input) ======================= #
    if RUN_ENV.lower() == "cloud":
        with mid:
            st.info(
                "Running in *cloud* mode (Streamlit Community Cloud). "
                "Using browser camera snapshots instead of OpenCV VideoCapture."
            )
            cam_img = st.camera_input("Take a photo", key="cam_input")

        if cam_img is not None:
            pil_image = Image.open(cam_img).convert("RGB")

            # Original image show
            with mid:
                st.image(pil_image, caption="Captured Frame", use_container_width=True)

            # Run detection
            with st.spinner("Detecting faces and emotions..."):
                frame_rgb = np.array(pil_image)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                annotated_bgr, faces_data = detect_and_classify_faces(frame_bgr)
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(
                    annotated_rgb,
                    caption="Annotated with Bounding Boxes",
                    use_container_width=True,
                )

            with mid:
                render_face_results(faces_data)

        else:
            with mid:
                st.info("Click **Take a photo** above to capture a frame.")

    # ======================= LOCAL MODE (cv2.VideoCapture) ====================== #
    else:
        with mid:
            start = st.button("Start Camera")
            frame_placeholder = st.empty()
            info_placeholder = st.empty()

        if start:
            cap = cv2.VideoCapture(0)  # local webcam

            if not cap.isOpened():
                with mid:
                    st.error("Could not access camera.")
            else:
                with mid:
                    stop_button = st.button("Stop")

                while True:
                    ret, frame_bgr = cap.read()
                    if not ret:
                        with mid:
                            st.error("Failed to grab frame from camera.")
                        break

                    annotated_bgr, faces_data = detect_and_classify_faces(frame_bgr)

                    # Resize preview for smooth performance
                    h, w = annotated_bgr.shape[:2]
                    target_width = 580
                    scale = target_width / w
                    target_height = int(h * scale)
                    annotated_small = cv2.resize(
                        annotated_bgr, (target_width, target_height)
                    )

                    annotated_rgb = cv2.cvtColor(annotated_small, cv2.COLOR_BGR2RGB)

                    with mid:
                        frame_placeholder.image(
                            annotated_rgb,
                            channels="RGB",
                            width=480,
                        )

                        if faces_data:
                            face = faces_data[0]
                            label = face["label"]
                            conf = face["confidence"]
                            probs = face["probabilities"]

                            text = f"**Detected:** {label} ({conf*100:.2f}%)\n\n"
                            for emo, prob in probs.items():
                                text += f"- {emo}: {prob*100:.1f}%\n"
                            info_placeholder.markdown(text)
                        else:
                            info_placeholder.info("No face detected.")

                        if stop_button:
                            break

                    time.sleep(0.03)

                cap.release()
                with mid:
                    frame_placeholder.empty()
                    info_placeholder.empty()
                    st.success("Camera stopped.")



# streamlit run app/app.py
