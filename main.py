from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
BLIP_MODEL = os.getenv("BLIP_MODEL")

# Cliente configurado a DeepSeek
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

app = FastAPI(title="Backend BLIP + DeepSeek")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_IMAGE_SIZE = (512, 512)

from openai import OpenAI

# Inicializar cliente DeepSeek
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

def generar_respuesta_deepseek(message: str) -> str:
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Eres un asistente educativo que explica conceptos de manera clara y sencilla."},
                {"role": "user", "content": message}
            ],
            stream=True
        )
        return response.choices[0].message.content
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

        device = torch.device("cpu")

        processor = AutoProcessor.from_pretrained(BLIP_MODEL)
        blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL).to(device)

        img = Image.open(image.file).convert("RGB")
        img.thumbnail(MAX_IMAGE_SIZE)
        inputs = processor(images=img, return_tensors="pt").to(device)

        output_ids = blip_model.generate(**inputs, max_length=50)
        caption = processor.decode(output_ids[0], skip_special_tokens=True)
        bot_message += f"📸 Caption de la imagen: {caption}\n"


    # Aquí reemplazamos DeepSeek por GPT-4.1 mini
    gpt_response = generar_respuesta_gpt(message)
    bot_message += f"🤖  {gpt_response}"

    return {"response": bot_message}
# ----------------------------
# Para correr local (opcional)
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    import os
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), workers=1)

