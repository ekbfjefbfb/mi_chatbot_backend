from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import Base, engine
import models
from auth import router as auth_router
from assistant import router as assistant_router
import os

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Asistente Completo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("ALLOWED_ORIGINS","*").split(",")] if "*" not in os.getenv("ALLOWED_ORIGINS","*") else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

app.include_router(auth_router)
app.include_router(assistant_router)
