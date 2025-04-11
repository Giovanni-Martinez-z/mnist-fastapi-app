from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Cargar modelo
model = load_model('mnist_model.h5')

# Directorio de plantillas HTML
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('L')
    image = image.resize((28, 28))
    image_array = np.array(image)
    image_array = 255 - image_array  # invertir colores si es necesario
    image_array = image_array / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)

    prediction = model.predict(image_array)
    digit = int(np.argmax(prediction))
    return {"prediccion": digit}
