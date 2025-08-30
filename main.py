# backend_full_v2.py
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI
import os, io, re, requests, asyncio, textwrap
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from docx import Document
from dotenv import load_dotenv
from typing import Optional, List

# ----------------------------
# CARGAR VARIABLES DE ENTORNO
# ----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")  # Para noticias en tiempo real
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app = FastAPI(title="Asistente Educativo Definitivo 2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in ALLOWED_ORIGINS] if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------------
# UTILIDADES
# ----------------------------
def limpiar_texto(texto: str) -> str:
    texto = re.sub(r"[*_`~]{1,3}", "", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()

def json_bytes(obj: dict) -> bytes:
    import json
    return json.dumps(obj).encode("utf-8")

# ----------------------------
# ANALIZAR IMÁGENES CON BLIP + GPT
# ----------------------------
def process_image_caption(image_bytes: bytes) -> str:
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
    files = {"file": ("image.jpg", image_bytes)}
    url = "https://api-inference.huggingface.co/models/Salesforce/blip2-flan-t5-xl"
    r = requests.post(url, headers=headers, files=files, timeout=60)
    data = r.json()
    caption = data.get("generated_text") or str(data)
    # Mejorar con GPT
    prompt = f"Explica para un estudiante: {caption}"
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role":"user","content":prompt}],
        max_tokens=500
    )
    text = resp.choices[0].message.content if hasattr(resp.choices[0],"message") else resp.choices[0].text
    return limpiar_texto(text)

# ----------------------------
# GENERAR IMÁGENES CON DALL·E
# ----------------------------
def generate_image(prompt: str) -> str:
    resp = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024"
    )
    return resp.data[0].url

# ----------------------------
# ANALIZAR DOCUMENTOS (PDF / Word)
# ----------------------------
def analyze_document_bytes(file_bytes: bytes, filename: str) -> str:
    text = ""
    if filename.endswith(".pdf"):
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            text += page.extract_text() + "\n"
    elif filename.endswith(".docx"):
        doc = Document(io.BytesIO(file_bytes))
        for para in doc.paragraphs:
            text += para.text + "\n"
    else:
        raise HTTPException(status_code=400, detail="Formato no soportado")
    prompt = f"Resume el siguiente texto para un estudiante:\n{text}"
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role":"user","content":prompt}],
        max_tokens=800
    )
    summary = resp.choices[0].message.content if hasattr(resp.choices[0],"message") else resp.choices[0].text
    return limpiar_texto(summary)

# ----------------------------
# CREAR PDF FINAL
# ----------------------------
def create_pdf_report(title: str, body: str, image_urls: list) -> io.BytesIO:
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    p.setFont("Helvetica-Bold", 14)
    p.drawString(40, height-40, title)
    p.setFont("Helvetica", 11)
    y = height-70
    wrapper = textwrap.wrap(body, width=100)
    for line in wrapper:
        if y < 100:
            p.showPage()
            y = height-40
            p.setFont("Helvetica",11)
        p.drawString(40, y, line)
        y -= 16
    for url in image_urls:
        try:
            img_resp = requests.get(url)
            img_data = io.BytesIO(img_resp.content)
            p.showPage()
            p.drawInlineImage(img_data,50,height//2-100,width=500,height=300)
        except:
            continue
    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer

# ----------------------------
# BUSCAR NOTICIAS EN TIEMPO REAL
# ----------------------------
def search_news(query: str) -> str:
    if not NEWSAPI_KEY:
        return "NewsAPI no configurada."
    url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&apiKey={NEWSAPI_KEY}&language=es&pageSize=3"
    r = requests.get(url)
    data = r.json()
    articles = data.get("articles", [])
    summary = ""
    for art in articles:
        summary += f"- {art.get('title')} ({art.get('source', {}).get('name')})\n"
    return summary or "No se encontraron noticias recientes."

# ----------------------------
# ENDPOINT PRINCIPAL
# ----------------------------
@app.post("/assistant/stream")
async def assistant_stream(
    command: str = Form(...),
    upload_files: Optional[List[UploadFile]] = File(None),
    search_news_query: Optional[str] = Form(None)
):
    """
    Backend completo: streaming, análisis de archivos, generación de imágenes, PDF y noticias.
    """
    async def event_generator():
        text_parts = []
        image_urls = []

        # Archivos subidos
        if upload_files:
            for file in upload_files:
                contents = await file.read()
                if file.content_type.startswith("image/"):
                    caption = process_image_caption(contents)
                    text_parts.append(caption)
                elif file.filename.endswith((".pdf",".docx")):
                    summary = analyze_document_bytes(contents, file.filename)
                    text_parts.append(summary)

        # Búsqueda de noticias
        if search_news_query:
            news_text = search_news(search_news_query)
            text_parts.append("Noticias recientes:\n" + news_text)

        # GPT-4-mini genera texto principal
        prompt = f"Como asistente educativo, realiza: {command}\n"
        if text_parts:
            prompt += "\nInformación de archivos y noticias:\n" + "\n".join(text_parts)

        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=1500
        )
        main_text = resp.choices[0].message.content if hasattr(resp.choices[0],"message") else resp.choices[0].text
        main_text = limpiar_texto(main_text)

        # Streaming de texto
        for char in main_text:
            yield json_bytes({"delta": char}) + b"\n"
            await asyncio.sleep(0.01)

        # Generar imágenes si se mencionan
        if "imagen" in command.lower() or "imágenes" in command.lower():
            lines = main_text.split("\n")
            for line in lines:
                if "imagen" in line.lower():
                    url = generate_image(line)
                    image_urls.append(url)

        # Crear PDF final
        pdf_buffer = create_pdf_report(title=command[:50], body=main_text, image_urls=image_urls)
        yield json_bytes({"delta": "\n✅ PDF generado listo para descarga.", "pdf_ready": True}) + b"\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
