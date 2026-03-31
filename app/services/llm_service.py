import json
import logging
import google.generativeai as genai
from app.core.config import settings

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        try:
            if not settings.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY não encontrada no .env")

            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.model_name = "models/gemini-2.5-flash"
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"LLMService inicializado com: {self.model_name}")

        except Exception as e:
            logger.error(f"Erro ao inicializar o LLMService: {str(e)}")
            self.model = None

    def get_plant_care_advice(self, plant_disease: str) -> dict:
        if not self.model:
            return {"error": "Serviço de diagnóstico offline."}

        prompt = f"""
        És o gAIa, um assistente botânico de IA.
        O diagnóstico é: {plant_disease}.

        Responde APENAS com JSON válido, sem markdown, neste formato exato:
        {{
          "description": "O que é esta condição em 2 frases.",
          "treatment": ["passo 1", "passo 2"],
          "prevention": ["dica 1", "dica 2"]
        }}
        """

        try:
            response = self.model.generate_content(prompt)
            texto_limpo = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(texto_limpo)
        except json.JSONDecodeError:
            logger.error("Gemini não devolveu JSON válido")
            return {"error": "Erro ao processar resposta do modelo."}
        except Exception as e:
            logger.error(f"Erro na geração do Gemini: {str(e)}")
            return {"error": "Erro ao gerar recomendações."}