from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI
import os
from dotenv import load_dotenv
import json

# ----------------------------
# Cargar variables de entorno
# ----------------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# ----------------------------
# Inicializar FastAPI
# ----------------------------
app = FastAPI(title="Backend DeepSeek R1 Streaming")

# ----------------------------
# Habilitar CORS
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # reemplaza por tu frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Inicializar cliente DeepSeek vía OpenRouter
# ----------------------------
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)
# Función para limpiar la respuesta (sin Markdown, solo emojis y texto)
# ----------------------------
def limpiar_respuesta(texto: str):
    texto_limpio = re.sub(r"[*_#`~]", "", texto)
    texto_limpio = re.sub(r"\n{2,}", "\n", texto_limpio)
    return texto_limpio
# ----------------------------
# Endpoint streaming
# ----------------------------
@app.post("/chat/stream/")
async def chat_stream(message: str = Form(...)):
    async def event_generator():
        try:
            # Stream de DeepSeek
            stream = client.chat.completions.stream(
                model="deepseek/deepseek-r1",
                messages=[
                    {"role": "system", "content": "Eres un asistente educativo que explica conceptos de manera clara y sencilla."},
                    {"role": "user", "content": message}
                ],
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if "content" in delta:
                    yield json.dumps({"delta": delta["content"]}) + "\n"
        except Exception as e:
            yield json.dumps({"error": str(e)})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ----------------------------
# Para correr local
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), workers=1)


