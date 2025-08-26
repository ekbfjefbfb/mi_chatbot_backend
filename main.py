from fastapi import FastAPI, UploadFile, Form
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch
import requests
import os
from dotenv import load_dotenv

# ----------------------------
# Cargar variables de entorno
# ----------------------------
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
KOSMOS2_MODEL = os.getenv("KOSMOS2_MODEL")
HF_TOKEN = os.getenv("HF_TOKEN")

# ----------------------------
# Inicializar FastAPI
# ----------------------------
app = FastAPI(title="Backend Kosmos-2 + DeepSeek V3")

# ----------------------------
# Configurar dispositivo
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Cargar modelo Kosmos-2
# ----------------------------
processor = AutoProcessor.from_pretrained(KOSMOS2_MODEL, use_auth_token=HF_TOKEN)
kosmos_model = AutoModelForImageTextToText.from_pretrained(KOSMOS2_MODEL, use_auth_token=HF_TOKEN).to(device)

# ----------------------------
# Función para generar respuesta DeepSeek
# ----------------------------
def generar_respuesta_deepseek(message: str) -> str:
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    payload = {"prompt": message, "max_tokens": 150}
    response = requests.post("https://api.deepseek.ai/v3/text-generation", json=payload, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return data.get("text", "No se pudo generar respuesta.")
    else:
        return f"Error: {response.status_code} {response.text}"

# ----------------------------
# Endpoint principal
# ----------------------------
@app.post("/chat/")
async def chat_endpoint(message: str = Form(...), image: UploadFile = None):
    bot_message = ""

    # Procesar imagen si existe
    if image:
        img = Image.open(image.file).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        output_ids = kosmos_model.generate(pixel_values=inputs.pixel_values, max_length=50)
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
