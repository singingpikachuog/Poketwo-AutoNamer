import os
import json
import random
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO

class Prediction:
    def __init__(self):
        # Load your model
        self.model_path = "model/pokemon_cnn.pt"
        self.labels_path = "model/labels.txt"
        self.data_path = "pokemon_data.txt"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(self.model_path, map_location=self.device)
        self.model.eval()

        # Load labels
        with open(self.labels_path, "r") as f:
            self.labels = [line.strip() for line in f]

        # Load Pokémon metadata
        if os.path.exists(self.data_path):
            with open(self.data_path, "r", encoding="utf-8") as f:
                self.pokemon_data = json.load(f)
        else:
            self.pokemon_data = {}

        # Transform for input images
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def predict(self, url):
        try:
            # Load image from URL
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img = self.transform(img).unsqueeze(0).to(self.device)

            # Run prediction
            with torch.no_grad():
                outputs = self.model(img)
                probs = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, 1)

            # Get label (dex + name)
            full_label = self.labels[predicted.item()]
            dex_number = full_label.split("_")[0]
            clean_name = full_label[5:].replace("_", " ")

            # Look up in pokemon_data.txt
            pokemon_info = None
            random_name = None
            if self.pokemon_data:
                for entry in self.pokemon_data.values():
                    if str(entry.get("dex_number")).zfill(3) == dex_number:
                        pokemon_info = entry
                        random_name = random.choice(list(entry["names"].values())) if "names" in entry else None
                        break

            return {
                "dex": dex_number,
                "name": clean_name,
                "confidence": round(confidence.item() * 100, 2),
                "random_name": random_name
            }

        except Exception as e:
            return {"error": str(e)}
