from fastapi import FastAPI, UploadFile, File
import shutil
import os
from predictor import predict_single

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def home():
    return {"message": "Deepfake Detection API 🚀"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    filepath = os.path.join(UPLOAD_DIR, file.filename)

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_single(filepath)

    return result