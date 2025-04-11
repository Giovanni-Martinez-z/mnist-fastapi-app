import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="MNIST Classifier API")

# Configuración de versiones compatibles con .h5
TF_VERSION = "2.12.0"  # Última versión con soporte completo para .h5

# Cargar modelo .h5 con manejo de compatibilidad
try:
    # Asegurar compatibilidad
    tf.keras.backend.set_floatx('float32')
    
    # Cargar modelo
    model = tf.keras.models.load_model(
        'mnist_model.h5',
        custom_objects=None,
        compile=False
    )
    
    # Compilar manualmente (opcional pero recomendado)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Modelo cargado correctamente")
except Exception as e:
    raise RuntimeError(f"Error al cargar el modelo .h5: {str(e)}")

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>MNIST Classifier</title>
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    max-width: 800px; 
                    margin: 0 auto; 
                    padding: 20px; 
                }
                code { 
                    background: #f4f4f4; 
                    padding: 2px 5px; 
                }
            </style>
        </head>
        <body>
            <h1>API de Clasificación MNIST</h1>
            <p><strong>Versión TensorFlow:</strong> {TF_VERSION}</p>
            <p>Endpoint disponible: <code>POST /predict</code></p>
            <p>Prueba la API en <a href="/docs">/docs</a></p>
        </body>
    </html>
    """.format(TF_VERSION="2.18.0")  # Usa el valor directamente o define TF_VERSION arriba

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Validar tipo de archivo
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Solo se permiten imágenes (JPEG, PNG)")
        
        # Procesar imagen
        image = Image.open(io.BytesIO(await file.read())).convert('L')  # Convertir a escala de grises
        image = image.resize((28, 28))
        
        # Normalización y preparación para el modelo
        image_array = np.array(image) / 255.0
        input_data = np.expand_dims(image_array, axis=(0, -1)).astype(np.float32)
        
        # Predicción
        prediction = model.predict(input_data)
        predicted_digit = int(np.argmax(prediction))
        probabilities = prediction[0].tolist()
        
        return {
            "digit": predicted_digit,
            "probabilities": probabilities,
            "model_type": "h5",
            "tensorflow_version": TF_VERSION
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error de procesamiento: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=False,  # Desactivar en producción
        timeout_keep_alive=120  # Útil para evitar timeouts en Render
    )