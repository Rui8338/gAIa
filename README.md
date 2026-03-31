# 🌿 gAIa — Smart Plant Doctor AI

An end-to-end AI system that detects plant diseases from images and provides treatment recommendations using Computer Vision and a Large Language Model.

**Live Demo**: [gaia.streamlit.app](https://gaia.streamlit.app) · **API Docs**: [gaia-api.onrender.com/docs](https://gaia-api.onrender.com/docs)

---

## Overview

gAIa was built as a production-like AI engineering project, focusing on the full lifecycle after model training: API design, LLM integration, containerization, and cloud deployment.

Upload a photo of a plant leaf and gAIa will:
1. Classify the disease using a fine-tuned ResNet-50 (99.2% accuracy)
2. Generate a botanical report with treatment and prevention tips via Gemini

---

## Architecture

```
User (Streamlit UI)
        │
        ▼
FastAPI Backend (Render)
        ├── VisionService   → ResNet-50 (PyTorch) — classifies disease
        └── LLMService      → Gemini 2.0 Flash — generates report
                │
                ▼
        Structured JSON Response
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Computer Vision | PyTorch · ResNet-50 (transfer learning) |
| API | FastAPI · Pydantic · Uvicorn |
| LLM | Google Gemini 2.0 Flash |
| Frontend | Streamlit |
| Containerization | Docker · Docker Compose |
| Deployment | Render (API) · Streamlit Cloud (UI) |

---

## Supported Classes

| Plant | Condition |
|---|---|
| Apple | Apple Scab |
| Apple | Healthy |
| Potato | Early Blight |
| Potato | Healthy |
| Tomato | Late Blight |
| Tomato | Healthy |

---

## API

### `POST /predict`

Upload a plant image and receive a diagnosis.

**Request**
```
Content-Type: multipart/form-data
file: <image>
```

**Response**
```json
{
  "filename": "leaf.jpg",
  "prediction": "Potato___Early_blight",
  "confidence": "99.70%",
  "analysis": {
    "description": "Early blight is a fungal disease caused by Alternaria solani...",
    "treatment": ["Remove infected leaves", "Apply copper-based fungicide"],
    "prevention": ["Use certified seeds", "Practice crop rotation"]
  }
}
```

### `GET /health`
```json
{ "status": "ok" }
```

---

## Run Locally

**Prerequisites**: Docker Desktop

```bash
git clone https://github.com/<your-username>/gAIa.git
cd gAIa
```

Create a `.env` file:
```
GEMINI_API_KEY=your_key_here
MODEL_PATH=models/gaia_resnet50_v1.pth
MAPPING_PATH=models/class_mapping.json
```

```bash
docker-compose up --build
```

- API: http://localhost:8000/docs
- UI: http://localhost:8501

---

## Project Structure

```
gAIa/
├── app/
│   ├── api/
│   │   └── main.py           # FastAPI app, lifespan, routes
│   ├── core/
│   │   └── config.py         # Pydantic settings
│   └── services/
│       ├── vision_service.py # CV inference
│       └── llm_service.py    # LLM integration
├── models/
│   ├── gaia_resnet50_v1.pth  # Trained weights
│   └── class_mapping.json    # Index → class name
├── notebooks/
│   └── training_v1.ipynb     # Model training
├── Dockerfile
├── docker-compose.yml
└── requirements-api.txt
```

---

## Model

- **Architecture**: ResNet-50 (pretrained on ImageNet, last layer fine-tuned)
- **Dataset**: PlantVillage (6,927 images · 6 classes)
- **Training**: 10 epochs · Adam optimizer · CrossEntropyLoss
- **Results**: 99.5% train accuracy · 99.2% validation accuracy
- **Device**: CUDA (GPU accelerated)

---

## What I Learned

This project was focused on the engineering side of AI — not just training a model, but shipping it:

- Serving a PyTorch model via a REST API
- Integrating an LLM to generate structured JSON output
- Managing services with Docker and Docker Compose
- Deploying to cloud platforms (Render + Streamlit Cloud)
- Structuring a Python project for production readability
