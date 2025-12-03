# Dockerfile

FROM python:3.12-slim

# Install system deps for OpenCV if needed
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency list & install
COPY requirements.txt ./requirements.txt

# Use headless opencv in container
RUN pip install --no-cache-dir -r requirements.txt opencv-python-headless

# Copy entire project
COPY . .

# Streamlit config (optional: reduce logging, enable headless)
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
