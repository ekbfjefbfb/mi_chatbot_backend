from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from openrouter import OpenRouter  # <-- Importamos OpenRouter
import os
from dotenv import load_dotenv

# ----------------------------
# Cargar variables de entorno
# ----------------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
BLIP_MODEL = os.getenv("BLIP_MODEL")

# ----------------------------
# Inicializar FastAPI
# ----------------------------
app = FastAPI(title="Backend BLIP + DeepSeek R1")

# ----------------------------
# Habilitar CORS
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # reemplaza por tu frontend si quieres
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Parámetros de optimización
# ----------------------------
MAX_IMAGE_SIZE = (512, 512)

# ----------------------------
# Inicializar cliente DeepSeek
# ----------------------------
client = OpenRouter(api_key=OPENROUTER_API_KEY)

# ----------------------------
# Función para generar respuesta con DeepSeek R1
# ----------------------------
def generar_respuesta_deepseek(message: str) -> str:
    try:
        response = client.chat.completions.create(
            model="deepseek-r1",
            messages=[
                {"role": "system", "content": "Eres un asistente educativo que explica conceptos de manera clara y sencilla."},
                {"role": "user", "content": message}
            ],
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error al conectar con DeepSeek R1: {str(e)}"

# ----------------------------
# Endpoint principal
# ----------------------------
@app.post("/chat/")
async def chat_endpoint(message: str = Form(...), image: UploadFile = None):
    bot_message = ""

    if image:
        from transformers import AutoProcessor, BlipForConditionalGeneration
        import torch

        device = torch.device("cpu")
        processor = AutoProcessor.from_pretrained(BLIP_MODEL)
        blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL).to(device)

        img = Image.open(image.file).convert("RGB")
        img.thumbnail(MAX_IMAGE_SIZE)
        inputs = processor(images=img, return_tensors="pt").to(device)

        output_ids = blip_model.generate(**inputs, max_length=50)
        caption = processor.decode(output_ids[0], skip_special_tokens=True)
        bot_message += f"📸 Caption de la imagen: {caption}\n"

    # Usamos DeepSeek R1
    deepseek_response = generar_respuesta_deepseek(message)
    bot_message += f"🤖 {deepseek_response}"

    return {"response": bot_message}

# ----------------------------
# Para correr local
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), workers=1)
