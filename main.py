from fastapi import FastAPI, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import os
import json
import requests
from dotenv import load_dotenv
import re
import asyncio
from groq import GroqClient



# ----------------------------
# Cargar variables de entorno
# ----------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")            # Para LLaMA 3.1
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")  # Para BLIP-2
BLIP_MODEL = "Salesforce/blip2-flan-t5-xl"

# ----------------------------
# Inicializar FastAPI
# ----------------------------
app = FastAPI(title="Backend BLIP-2 + LLaMA 3.1 (Groq)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Inicializar cliente Groq
# ----------------------------

groq_client = GroqClient(api_key=GROQ_API_KEY)


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
    files = {"file": ("image.jpg", image_bytes)}
    try:
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{BLIP_MODEL}",
            headers=headers,
            files=files
        )
        data = response.json()
        print("BLIP Response:", data)  # 🔹 Debug
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
        elif isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        else:
            return "No se pudo generar caption de la imagen."
    except Exception as e:
        return f"No se pudo generar caption. Error: {e}"

# ----------------------------
# Función para LLaMA 3.1 vía Groq
# ----------------------------
from groq import GroqClient



def llama3_response(prompt: str) -> str:
    response = groq_client.predict(
        model="meta-llama/Llama-3.1-8b-instruct",
        input=prompt
    )
    return response.output_text.strip()



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
                await image.close()
                yield json.dumps({"delta": f"📸 Caption de la imagen: {caption}\n"}) + "\n"

            # ----------------------------
            # Preparar prompt para LLaMA 3.1
            # ----------------------------
            prompt = f"Eres un asistente educativo que explica conceptos de manera clara y sencilla.\nUsuario: {message}\nRespuesta:"

            # ----------------------------
            # Obtener respuesta de LLaMA 3.1
            # ----------------------------
            texto = llama3_response(prompt)
            texto = limpiar_texto(texto)

            # ----------------------------
            # Streaming letra por letra
            # ----------------------------
            for char in texto:
                yield json.dumps({"delta": char}) + "\n"
                await asyncio.sleep(0.02)

        except Exception as e:
            yield json.dumps({"error": str(e)})

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# ----------------------------
# Para correr local
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000, 8000)),
        workers=1
    )
