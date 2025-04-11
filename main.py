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
# Añade estas líneas ANTES de definir las rutas
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")

# Reemplaza tu ruta principal por esto:
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Clasificador MNIST</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }
            h1 { color: #333; }
            .container { max-width: 600px; margin: 0 auto; }
            .upload-box { border: 2px dashed #ccc; padding: 20px; margin: 20px 0; }
            button { background: #4CAF50; color: white; border: none; padding: 10px 20px; cursor: pointer; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Clasificador de Dígitos MNIST</h1>
            <div class="upload-box">
                <input type="file" id="imageInput" accept="image/*">
                <p>Sube una imagen de un dígito (0-9)</p>
                <img id="preview" style="max-width: 100%; display: none;">
            </div>
            <button onclick="predict()">Predecir</button>
            <div id="result" style="margin-top: 20px; font-size: 1.2em;"></div>
        </div>

        <script>
            document.getElementById('imageInput').addEventListener('change', function(e) {
                const file = e.target.files[0];
                const preview = document.getElementById('preview');
                preview.src = URL.createObjectURL(file);
                preview.style.display = 'block';
            });

            async function predict() {
                const fileInput = document.getElementById('imageInput');
                const resultDiv = document.getElementById('result');
                
                if (!fileInput.files.length) {
                    resultDiv.textContent = 'Por favor selecciona una imagen';
                    return;
                }

                const formData = new FormData();
                formData.append('file', fileInput.files[0]);

                try {
                    resultDiv.textContent = 'Procesando...';
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();
                    resultDiv.innerHTML = `Predicción: <strong>${data.digit}</strong> (Confianza: ${(data.confidence * 100).toFixed(2)}%)`;
                } catch (error) {
                    resultDiv.textContent = 'Error: ' + error.message;
                }
            }
        </script>
    </body>
    </html>
    """

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