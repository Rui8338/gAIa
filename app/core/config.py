import os
from dotenv import load_dotenv

# Carrega as variáveis do ficheiro .env
load_dotenv()

class Settings:
    PROJECT_NAME: str = "gAIa - Smart Plant Doctor"
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    MODEL_PATH: str = "models/gaia_resnet50_v1.pth"
    MAPPING_PATH: str = "models/class_mapping.json"

settings = Settings()