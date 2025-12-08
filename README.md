# 😃 Real-Time Face Emotion Detection

A comprehensive **face emotion recognition system** built with PyTorch that performs real-time emotion detection from webcam feeds and static images. The project includes multiple deployment options: Streamlit web app, FastAPI REST API, and a standalone web frontend.

---

## Live project link:
https://real-time-face-emotion-detection.streamlit.app/


## 🎯 Features

| Feature | Description |
|--------|-------------|
| 🧠 **Emotion Classification** | Detects 5 emotions: Happy, Sad, Angry, Fear, Surprise |
| 👁️ **Face Detection** | Uses OpenCV DNN (SSD) and Haar Cascade for robust face detection |
| 📷 **Real-time Webcam Mode** | Continuous emotion detection from live camera via WebRTC |
| 📁 **Image Upload Mode** | Upload JPG/PNG images for emotion prediction |
| 🎨 **Multiple UIs** | Streamlit app, FastAPI backend, and HTML/JS frontend |
| ⚡ **Custom CNN Model** | ResNet-inspired architecture optimized for emotion recognition |
| 🐳 **Docker Support** | Containerized deployment ready |
| 📊 **Detailed Results** | Bounding boxes, confidence scores, and probability distributions |

---

## 🏗️ Tech Stack

- **Python 3.12+**
- **PyTorch** - Deep learning framework
- **Streamlit** - Web app framework with WebRTC support
- **FastAPI** - REST API backend
- **OpenCV** - Face detection and image processing
- **OpenCV DNN** - Deep Neural Network face detector (SSD-based)
- **Haar Cascade** - Alternative face detection method
- **Docker** - Containerization

---

## 📂 Project Structure

```
Real time face emotion detection/
├── app/
│   ├── app.py              # Streamlit web application
│   └── main.py             # FastAPI REST API server
├── src/
│   ├── custom_resnet.py    # Custom ResNet-inspired CNN model
│   └── emotion_service.py  # Face detection and emotion classification service
├── web/
│   ├── index.html          # HTML frontend for FastAPI
│   └── script.js           # JavaScript client for API calls
├── model/
│   └── resnet_Model.pth    # Trained PyTorch model checkpoint
├── assets/
│   ├── style.css           # Custom CSS for Streamlit
│   ├── deploy.prototxt      # DNN face detector config
│   ├── res10_300x300_ssd_iter_140000_fp16.caffemodel  # DNN face detector weights
│   ├── haarcascade_frontalface_default.xml  # Haar Cascade classifier
│   └── *.png               # UI assets
├── test_img/               # Test images for validation
├── Codes/
│   └── face-emotion-classify-resnet-cnn-pytorch.ipynb  # Training notebook
├── Dockerfile              # Docker configuration
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## 🛠️ Installation

### Prerequisites

- Python 3.12 or higher
- pip or conda package manager
- Webcam (for real-time detection)
- Git (optional, for cloning)

### Step 1: Clone the Repository

```bash
git clone https://github.com/ruhul-cse-duet/real-time-face-emotion-detection.git
cd real-time-face-emotion-detection
```

### Step 2: Create Virtual Environment

**Using venv:**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

**Using conda:**
```bash
conda create -n face-emotion python=3.12 -y
conda activate face-emotion
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** For Docker deployment, OpenCV headless version is automatically installed.

---

## 🚀 Usage

### Option 1: Streamlit Web App (Recommended)

The Streamlit app provides an interactive UI with three modes: Home, Image Upload, and Real-Time Camera.

```bash
streamlit run app/app.py
```

Then open your browser to `http://localhost:8501`

**Features:**
- 🏠 **Home Page** - Project overview and navigation
- 📷 **Image Upload** - Upload and analyze static images
- 🎥 **Real-Time Camera** - Live webcam emotion detection using WebRTC

### Option 2: FastAPI REST API

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

**API Endpoints:**
- `GET /` - API information
- `POST /predict` - Upload an image file and get emotion predictions

**Example API Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_img/happy-1.png"
```

**Response Format:**
```json
{
  "num_faces": 1,
  "faces": [
    {
      "bbox": {"x": 100, "y": 150, "w": 200, "h": 200},
      "label": "Happy",
      "confidence": 0.95,
      "probabilities": {
        "Fear": 0.02,
        "Surprise": 0.01,
        "Angry": 0.01,
        "Sad": 0.01,
        "Happy": 0.95
      },
      "summary": "😊 Relaxed eyes and lifted cheeks suggest positive engagement.",
      "action": "Celebrate the interaction or store as a positive training sample."
    }
  ],
  "annotated_image_base64": "base64_encoded_image..."
}
```

### Option 3: Web Frontend (HTML/JS)

1. Start the FastAPI server (see Option 2)
2. Navigate to the `web` directory:
   ```bash
   cd web
   python -m http.server 5500
   ```
3. Open `http://localhost:5500` in your browser
4. Update `API_URL` in `web/script.js` if your FastAPI server runs on a different address

**Features:**
- Image upload with emotion prediction
- Real-time webcam emotion detection
- Visual bounding boxes and probability distributions

---

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t face-emotion-detection .
```

### Run Container

```bash
docker run -p 8501:8501 face-emotion-detection
```

The Streamlit app will be available at `http://localhost:8501`

**Note:** For FastAPI deployment, modify the Dockerfile `CMD` to:
```dockerfile
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 🧠 Model Architecture

The project uses a custom **ResNet-inspired CNN** architecture:

- **Input:** 384×384 RGB images (normalized with ImageNet mean/std)
- **Architecture:**
  - Convolutional blocks with BatchNorm and ReLU
  - Residual connections for improved gradient flow
  - Adaptive Average Pooling
  - Fully connected classifier
- **Output:** 5 emotion classes with softmax probabilities
- **Classes:** Fear, Surprise, Angry, Sad, Happy

### Model Details

- **Location:** `src/custom_resnet.py`
- **Checkpoint:** `model/resnet_Model.pth`
- **Device:** Automatically uses CUDA if available, otherwise CPU

---

## 🔍 Face Detection Methods

The project supports two face detection approaches:

1. **OpenCV DNN (Primary)** - Used in `emotion_service.py`
   - SSD-based face detector
   - More accurate and robust
   - Requires `deploy.prototxt` and `res10_300x300_ssd_iter_140000_fp16.caffemodel`

2. **Haar Cascade (Fallback)** - Used in `app/main.py`
   - Classic OpenCV face detector
   - Faster but less accurate
   - Uses built-in `haarcascade_frontalface_default.xml`

---

## 📊 Emotion Classes

| Emotion | Description | Visual Indicators |
|---------|-------------|-------------------|
| 😊 **Happy** | Positive engagement | Relaxed eyes, lifted cheeks |
| 😔 **Sad** | Negative emotion | Downward gaze, lowered lip corners |
| 😠 **Angry** | Heightened arousal | Tension in brow or jaw |
| 😨 **Fear** | Anxiety signal | Raised brows, widened eyes |
| 😮 **Surprise** | Sudden reaction | Eye and mouth widening |

Each prediction includes:
- **Label** - Detected emotion
- **Confidence** - Prediction confidence (0-1)
- **Probabilities** - Distribution across all 5 classes
- **Summary** - Description of facial features
- **Action** - Suggested interpretation

---

## 🧪 Testing

Test images are available in the `test_img/` directory:
- `happy-1.png`, `happy-2.png`
- `sad-1.png`, `sad-2.png`
- `angry-1.png`, `angry-2.png`
- `fear-1.png`, `fear-2.png`
- `surprise-1.png`, `surprise-2.png`, `surprise-3.png`

You can use these to test the emotion detection system.

---

## 📝 Training

The training notebook is located at `Codes/face-emotion-classify-resnet-cnn-pytorch.ipynb`. This Jupyter notebook contains:
- Data preprocessing
- Model architecture definition
- Training loop
- Validation and evaluation
- Model checkpoint saving

---

## ⚙️ Configuration

### Model Settings

- **Input Size:** 384×384 pixels
- **Normalization:** ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`
- **Device:** Auto-detected (CUDA/CPU)

### Face Detection Settings

- **DNN Confidence Threshold:** 0.4 (adjustable in `emotion_service.py`)
- **Haar Cascade Parameters:** `scaleFactor=1.3`, `minNeighbors=5`, `minSize=(60, 60)`

---

## 🐛 Troubleshooting

### Common Issues

1. **Model file not found:**
   - Ensure `model/resnet_Model.pth` exists
   - Check file path in `src/custom_resnet.py`

2. **Face detection files missing:**
   - Verify `assets/deploy.prototxt` and `assets/res10_300x300_ssd_iter_140000_fp16.caffemodel` exist
   - For Haar Cascade, OpenCV should include it by default

3. **WebRTC camera not working:**
   - Grant camera permissions in your browser
   - Check firewall settings
   - Try using HTTPS (required for some browsers)

4. **CUDA out of memory:**
   - The model automatically falls back to CPU
   - Reduce batch size if processing multiple images

---

## 📄 License

This project is for educational and demonstration purposes. Not intended for medical or psychological diagnosis.

---

## 👤 Author

[Md Ruhul Amin](https://www.linkedin.com/in/ruhul-duet-cse/);  
Email: ruhul.cse.duet@gmail.com

---

## 🙏 Acknowledgments

- OpenCV community for face detection models
- PyTorch team for the deep learning framework
- Streamlit for the web app framework

---

## 📞 Support

For issues, questions, or contributions, please open an issue on the GitHub repository.

---

**Happy Emotion Detecting! 😊**
