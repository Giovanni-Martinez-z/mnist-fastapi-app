import os
import uvicorn
from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = FastAPI()
model = load_model('mnist_model.h5')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert('L')
    image = image.resize((28, 28))
    image_array = np.array(image) / 255.0
    prediction = model.predict(image_array[np.newaxis, ..., np.newaxis])
    return {"digit": int(np.argmax(prediction))}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
