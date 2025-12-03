// ⭐⭐ CHANGE THIS to your FastAPI server address ⭐⭐
    const API_URL = "http://127.0.0.1:8000/predict";

    /* =========================================================
       SECTION 1: IMAGE UPLOAD EMOTION DETECTION
    ========================================================== */

    const fileInput = document.getElementById("fileInput");
    const uploadBtn = document.getElementById("uploadBtn");
    const uploadResultImg = document.getElementById("uploadResultImg");
    const uploadResults = document.getElementById("uploadResults");

    uploadBtn.addEventListener("click", async () => {
      if (!fileInput.files.length) {
        alert("Please select an image.");
        return;
      }

      const file = fileInput.files[0];
      const formData = new FormData();
      formData.append("file", file);

      uploadResults.textContent = "Analyzing...";

      try {
        const resp = await fetch(API_URL, { method: "POST", body: formData });
        const data = await resp.json();

        if (data.faces?.length > 0) {
          const face = data.faces[0];
          let output = `Emotion: ${face.label} (${(face.confidence * 100).toFixed(1)}%)\n\n`;

          output += "Class Probabilities:\n";
          for (const [emo, prob] of Object.entries(face.probabilities)) {
            output += `- ${emo}: ${(prob * 100).toFixed(1)}%\n`;
          }

          uploadResults.textContent = output;
        } else {
          uploadResults.textContent = "No face detected.";
        }

        if (data.annotated_image_base64) {
          uploadResultImg.src = "data:image/jpeg;base64," + data.annotated_image_base64;
        }

      } catch (err) {
        uploadResults.textContent = "Error: " + err;
      }
    });

    /* =========================================================
       SECTION 2: REALTIME CAMERA EMOTION DETECTION
    ========================================================== */

    const video = document.getElementById("video");
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");

    const realtimeResults = document.getElementById("realtimeResults");

    const startCamBtn = document.getElementById("startCamBtn");
    const stopCamBtn = document.getElementById("stopCamBtn");

    let streaming = false;
    let stream = null;

    // Open camera
    async function setupCamera() {
      stream = await navigator.mediaDevices.getUserMedia({ video: true });
      video.srcObject = stream;

      return new Promise((resolve) => {
        video.onloadedmetadata = () => {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          resolve();
        };
      });
    }

    async function sendFrameLoop() {
      if (!streaming) return;

      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      canvas.toBlob(async (blob) => {
        if (!blob || !streaming) return;

        const formData = new FormData();
        formData.append("file", blob, "frame.jpg");

        try {
          const resp = await fetch(API_URL, { method: "POST", body: formData });
          const data = await resp.json();

          if (data.faces?.length > 0) {
            const face = data.faces[0];
            let output = `Emotion: ${face.label} (${(face.confidence * 100).toFixed(1)}%)\n\n`;
            output += "Class Probabilities:\n";

            for (const [emo, prob] of Object.entries(face.probabilities)) {
              output += `- ${emo}: ${(prob * 100).toFixed(1)}%\n`;
            }

            realtimeResults.textContent = output;
          } else {
            realtimeResults.textContent = "No face detected.";
          }

          // Draw annotated bounding box returned by server
          if (data.annotated_image_base64) {
            const img = new Image();
            img.onload = () => {
              ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            };
            img.src = "data:image/jpeg;base64," + data.annotated_image_base64;
          }
        } catch (err) {
          realtimeResults.textContent = "Error: " + err;
        }

        requestAnimationFrame(sendFrameLoop);
      }, "image/jpeg", 0.8);
    }

    startCamBtn.onclick = async () => {
      if (streaming) return;
      await setupCamera();
      streaming = true;
      sendFrameLoop();
    };

    stopCamBtn.onclick = () => {
      streaming = false;
      if (stream) {
        stream.getTracks().forEach((t) => t.stop());
      }
      video.srcObject = null;
      realtimeResults.textContent = "";
    };