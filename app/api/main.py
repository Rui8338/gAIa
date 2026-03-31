import io
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from app.services.vision_service import VisionService
from app.services.llm_service import LLMService
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    import os
    print("CWD:", os.getcwd())
    print("FILES:", os.listdir("."))
    print("MODELS:", os.listdir("models") if os.path.exists("models") else "pasta não existe")
    app.state.vision = VisionService(str(settings.MODEL_PATH), str(settings.MAPPING_PATH))
    app.state.llm = LLMService()
    yield

app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

# 1. ADICIONA O CORS (Essencial para o Streamlit funcionar)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def home():
    return {"status": "online", "message": f"Bem-vindo ao {settings.PROJECT_NAME}"}

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    # Validação básica
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="O ficheiro deve ser uma imagem.")

    try:
        # 2. LER OS BYTES DA IMAGEM
        image_bytes = await file.read()
        img_io = io.BytesIO(image_bytes)

        # 3. ACEDER AOS SERVIÇOS VIA app.state (Injetados pelo lifespan)
        vision_service = request.app.state.vision
        llm_service = request.app.state.llm

        # Predição de Visão
        prediction_result = vision_service.predict(img_io)
        disease = prediction_result.get("disease")
        confidence = prediction_result.get("confidence")

        # Análise do Gemini
        if confidence > 0.60:
            analysis = llm_service.get_plant_care_advice(disease)
        else:
            analysis = {
                "description": "Confiança baixa. Tire uma foto mais nítida da zona afetada.",
                "treatment": [],
                "prevention": []
            }
        analysis = llm_service.get_plant_care_advice(disease)

        return {
            "filename": file.filename,
            "prediction": disease,
            "confidence": f"{confidence * 100:.2f}%",
            "analysis": analysis
        }

    except Exception as e:
        print(f"Erro no processamento: {e}")
        raise HTTPException(status_code=500, detail=str(e))