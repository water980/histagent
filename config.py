import os
from dotenv import load_dotenv

load_dotenv()


TG_TOKEN = os.getenv("TG_TOKEN")
LLM_TOKEN = os.getenv("LLM_TOKEN")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")


if not TG_TOKEN:
    raise ValueError("TG_TOKEN is missing from .env")

if not LLM_TOKEN:
    raise ValueError("LLM_TOKEN is missing from .env")
