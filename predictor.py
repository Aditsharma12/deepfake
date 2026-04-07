import torch
import cv2
import numpy as np
import os
import urllib.request
from model.pred_func import load_cvit
from facenet_pytorch import MTCNN

# -------------------------
# 🔥 AUTO DOWNLOAD MODEL
# -------------------------
MODEL_URL = "https://huggingface.co/datasets/Deressa/cvit/resolve/main/cvit2_deepfake_detection_ep_50.pth"
WEIGHT_PATH = "weight/cvit2_deepfake_detection_ep_50.pth"

def download_weights():
    os.makedirs("weight", exist_ok=True)
    if not os.path.exists(WEIGHT_PATH):
        print("⬇️ Downloading model (~1GB)... please wait")
        urllib.request.urlretrieve(MODEL_URL, WEIGHT_PATH)
        print("✅ Download complete!")

download_weights()

# -------------------------
# MODEL LOAD
# -------------------------
MODEL_NAME = "cvit2"
device = torch.device("cpu")

model = load_cvit(WEIGHT_PATH, MODEL_NAME, fp16=False)
model.eval()

# Face detector
mtcnn = MTCNN(keep_all=True, device=device)


# -------------------------
# FRAME EXTRACTION
# -------------------------
def extract_frames(video_path, num_frames=15):
    cap = cv2.VideoCapture(video_path)
    frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // num_frames, 1)

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


# -------------------------
# FACE PREPROCESS
# -------------------------
def preprocess_faces(frames):
    faces = []

    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(rgb)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face = rgb[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                face = cv2.resize(face, (224, 224))
                faces.append(face)

    return faces


# -------------------------
# PREDICTION
# -------------------------
def predict_single(video_path):
    frames = extract_frames(video_path)
    faces = preprocess_faces(frames)

    if len(faces) == 0:
        return {
            "prediction": "NO_FACE",
            "confidence": 0.0
        }

    faces = np.array(faces) / 255.0
    faces = np.transpose(faces, (0, 3, 1, 2))
    faces = torch.tensor(faces, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(faces)
        prob = torch.sigmoid(outputs).mean().item()

    return {
        "prediction": "FAKE" if prob >= 0.5 else "REAL",
        "confidence": round(prob, 4)
    }
