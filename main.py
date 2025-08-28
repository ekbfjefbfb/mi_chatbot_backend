from fastapi import FastAPI, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI
import os
import json
import requests
from dotenv import load_dotenv
from PIL import Image
import io
import re

# ----------------------------
# Cargar variables de entorno
# ----------------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
BLIP_MODEL = "Salesforce/blip2-flan-t5-xl"

# ----------------------------
# Inicializar FastAPI
# ----------------------------
app = FastAPI(title="Backend BLIP-2 + DeepSeek R1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_IMAGE_SIZE = (512, 512)

# ----------------------------
# Inicializar cliente DeepSeek vía OpenRouter
# ----------------------------
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# ----------------------------
# Función para limpiar texto
# ----------------------------
def limpiar_texto(texto: str) -> str:
    texto = re.sub(r"[*_`~]", "", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()

# ----------------------------
# Función para BLIP-2 vía Hugging Face API
# ----------------------------
def blip2_caption_hf(image_bytes: bytes) -> str:
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
    files = {"file": image_bytes}
    response = requests.post(f"https://api-inference.huggingface.co/models/{BLIP_MODEL}",
                             headers=headers, files=files)
    try:
        return response.json()[0]["generated_text"]
    except Exception:
        return "No se pudo generar caption de la imagen."

# ----------------------------
# Endpoint streaming
# ----------------------------
@app.post("/chat/stream/")
async def chat_stream(message: str = Form(...), image: UploadFile = None):

    async def event_generator():
        try:
            # ----------------------------
            # Procesar imagen si existe
            # ----------------------------
            if image:
                img_bytes = await image.read()
                caption = blip2_caption_hf(img_bytes)
                yield json.dumps({"delta": f"📸 Caption de la imagen: {caption}\n"}) + "\n"

            # ----------------------------
            # Stream DeepSeek R1 (síncrono)
            # ----------------------------
            stream = client.chat.completions.stream(
                model="deepseek/deepseek-r1",
                messages=[
                    {"role": "system", "content": "Eres un asistente educativo que explica conceptos de manera clara y sencilla."},
                    {"role": "user", "content": message},
                ],
            )

            # Usamos un bucle normal en lugar de async for
            for chunk in stream:
                delta = chunk.choices[0].delta
                if "content" in delta:
                    texto_limpio = limpiar_texto(delta["content"])
                    yield json.dumps({"delta": texto_limpio}) + "\n"

        except Exception as e:
            yield json.dumps({"error": str(e)})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ----------------------------
# Para correr local
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), workers=1)



