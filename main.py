from fastapi import FastAPI, File, UploadFile, Request
from fastapi.staticfiles import StaticFiles  # ¡Esta importación faltaba!
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import uvicorn  # Asegúrate de importar uvicorn también

app = FastAPI()

# Configuración para servir archivos estáticos
app.mount("/templates", StaticFiles(directory="templates"), name="templates")
templates = Jinja2Templates(directory="templates")

# Cargar modelo MNIST
model = tf.keras.models.load_model('mnist_model.h5')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Leer y preprocesar imagen
    image = Image.open(io.BytesIO(await file.read())).convert('L')
    image = image.resize((28, 28))
    image_array = np.array(image) / 255.0
    input_data = image_array[np.newaxis, ..., np.newaxis].astype(np.float32)

    # Predecir
    prediction = model.predict(input_data)
    return {
        "digit": int(np.argmax(prediction)),
        "confidence": float(np.max(prediction))
    }

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("templates/index.html", {"request": request})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)