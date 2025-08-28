from fastapi import FastAPI, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI
import os
import json
from dotenv import load_dotenv
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration
import torch

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_IMAGE_SIZE = (512, 512)
device = torch.device("cpu")

# ----------------------------
# Inicializar cliente DeepSeek vía OpenRouter
# ----------------------------
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# ----------------------------
# Inicializar modelo BLIP (caption de imágenes)
# ----------------------------
processor = AutoProcessor.from_pretrained(BLIP_MODEL)
blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL).to(device)
# Función para limpiar texto
def limpiar_texto(texto: str) -> str:
    # Quita Markdown, asteriscos, guiones raros, pero mantiene emojis
    texto = re.sub(r"[*_`~]", "", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()

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
                img = Image.open(image.file).convert("RGB")
                img.thumbnail(MAX_IMAGE_SIZE)
                inputs = processor(images=img, return_tensors="pt").to(device)
                output_ids = blip_model.generate(**inputs, max_length=50)
                caption = processor.decode(output_ids[0], skip_special_tokens=True)
                yield json.dumps({"delta": f"📸 Caption de la imagen: {caption}\n"}) + "\n"

            # ----------------------------
            # Stream DeepSeek R1
            # ----------------------------
            stream = client.chat.completions.stream(
                model="deepseek/deepseek-r1",
                messages=[
                    {"role": "system", "content": "Eres un asistente educativo que explica conceptos de manera clara y sencilla."},
                    {"role": "user", "content": message},
                ],
            )

            async for chunk in stream:
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


