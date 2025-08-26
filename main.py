from fastapi import FastAPI, UploadFile, Form
from PIL import Image
import requests
import os
from dotenv import load_dotenv

# ----------------------------
# Cargar variables de entorno
# ----------------------------
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
BLIP_MODEL = os.getenv("BLIP_MODEL")
HF_TOKEN = os.getenv("HF_TOKEN")  # Opcional si tu modelo no necesita autenticación

# ----------------------------
# Inicializar FastAPI
# ----------------------------
app = FastAPI(title="Backend BLIP + DeepSeek V3")

# ----------------------------
# Parámetros de optimización
# ----------------------------
MAX_IMAGE_SIZE = (512, 512)  # Redimensionar imágenes para ahorrar RAM

# ----------------------------
# Función para generar respuesta DeepSeek
# ----------------------------
def generar_respuesta_deepseek(message: str) -> str:
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    payload = {"prompt": message, "max_tokens": 150}
    try:
        response = requests.post("https://api.deepseek.ai/v3/text-generation", json=payload, headers=headers)
        if response.status_code == 150:
            data = response.json()
            return data.get("text", "No se pudo generar respuesta.")
        else:
            return f"Error: {response.status_code} {response.text}"
    except Exception as e:
        return f"Error al conectar con DeepSeek: {str(e)}"

# ----------------------------
# Endpoint principal
# ----------------------------
@app.post("/chat/")
async def chat_endpoint(message: str = Form(...), image: UploadFile = None):
    bot_message = ""

    if image:
        # Lazy loading de librerías pesadas
        from transformers import AutoProcessor, BlipForConditionalGeneration
        import torch

        # Configurar dispositivo
        device = torch.device("cpu")

        # Cargar modelo BLIP2 (small)
        processor = AutoProcessor.from_pretrained(BLIP_MODEL)
        blip2_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL).to(device)

        # Procesar imagen con redimensión
        img = Image.open(image.file).convert("RGB")
        img.thumbnail(MAX_IMAGE_SIZE)
        inputs = processor(images=img, return_tensors="pt").to(device)

        # Generar caption
        output_ids = blip_model.generate(**inputs, max_length=50)
        caption = processor.decode(output_ids[0], skip_special_tokens=True)
        bot_message += f"📸 Caption de la imagen: {caption}\n"

    # Generar respuesta DeepSeek
    deepseek_response = generar_respuesta_deepseek(message)
    bot_message += f"🤖 DeepSeek V3 dice: {deepseek_response}"

    return {"response": bot_message}

# ----------------------------
# Para correr local (opcional)
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=1)

