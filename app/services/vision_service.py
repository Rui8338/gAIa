import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
from pathlib import Path

class VisionService:
    def __init__(self, model_path: str, mapping_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_mapping = self._load_mapping(mapping_path)
        self.model = self._init_model(model_path)
        
        # As transformações devem ser IDÊNTICAS às do treino
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_mapping(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _init_model(self, path):
        num_classes = len(self.class_mapping)
        # Recriar a arquitetura exata do ResNet50
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        # Carregar os pesos guardados no notebook
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def predict(self, image_bytes):
        img = Image.open(image_bytes).convert("RGB")
        img_t = self.transform(img).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(img_t)
            # outputs tem o formato -> Lote de 1, com 6 classes
            
            # Aplicamos o softmax na dimensão 1 (as classes)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Tiramos o valor máximo e o índice desse máximo
            conf, pred = torch.max(probabilities, dim=1)

        return {
            # .item() converte o Tensor de 1 elemento num número normal do Python
            "disease": self.class_mapping[str(pred.item())],
            "confidence": float(conf.item())
        }