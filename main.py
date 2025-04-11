import os
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="MNIST Classifier API")

# Cargar modelo con manejo de errores
try:
    # Opción 1: Para modelos .keras (recomendado)
    model = tf.keras.models.load_model('mnist_model.keras')
    
    # Opción 2: Si usas TFLite (más ligero para producción)
    # interpreter = tf.lite.Interpreter(model_path="mnist_model.tflite")
    # interpreter.allocate_tensors()
    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_get_output_details()
    
    model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
except Exception as e:
    raise RuntimeError(f"Error al cargar el modelo: {str(e)}")

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>MNIST Classifier</title>
        </head>
        <body>
            <h1>API de Clasificación de Dígitos MNIST</h1>
            <p>Visita <a href="/docs">/docs</a> para probar la API</p>
            <p>Sube una imagen de un dígito a <code>/predict</code></p>
        </body>
    </html>
    """

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Validar tipo de archivo
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Solo se permiten imágenes")
        
        # Procesar imagen
        image = Image.open(io.BytesIO(await file.read())).convert('L')
        image = image.resize((28, 28))
        image_array = np.array(image) / 255.0
        input_data = image_array[np.newaxis, ..., np.newaxis].astype(np.float32)
        
        # Predicción
        prediction = model.predict(input_data)
        
        # Si usas TFLite:
        # interpreter.set_tensor(input_details[0]['index'], input_data)
        # interpreter.invoke()
        # prediction = interpreter.get_tensor(output_details[0]['index'])
        
        return {
            "digit": int(np.argmax(prediction)),
            "probabilities": prediction[0].tolist()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)