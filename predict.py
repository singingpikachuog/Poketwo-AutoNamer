
import onnxruntime as ort
import numpy as np
import requests
from PIL import Image
import io
import os
import json
import random
import re

SUBMODULE_PATH = os.path.dirname(os.path.realpath(__file__))  

ONNX_PATH = os.path.join(SUBMODULE_PATH, "model/pokemon_cnn.onnx")
LABELS_PATH = os.path.join(SUBMODULE_PATH, "model/labels.txt")
SAVE_PATH = os.path.join(SUBMODULE_PATH, "data/commands/pokemon/images")
POKEMON_DATA_PATH = os.path.join(SUBMODULE_PATH, "pokemon_data.txt")

class Prediction:
    def __init__(self, onnx_path=ONNX_PATH, labels_path=LABELS_PATH, save_path=SAVE_PATH, pokemon_data_path=POKEMON_DATA_PATH):
        self.onnx_path = onnx_path
        self.labels_path = labels_path
        self.save_path = save_path
        self.pokemon_data_path = pokemon_data_path
        self.class_names = self.load_class_names()
        self.ort_session = ort.InferenceSession(self.onnx_path)
        self.pokemon_data = self.load_pokemon_data()

    def load_pokemon_data(self):
        """Load the pokemon data from pokemon_data.txt"""
        try:
            with open(self.pokemon_data_path, "r", encoding="utf-8") as f:
                content = f.read()
                # Parse the JSON-like structure
                return json.loads(content)
        except Exception as e:
            print(f"Error loading pokemon data: {e}")
            return {}

    def generate_labels_file_from_save_path(self):
        if not os.path.exists(self.save_path):
            raise FileNotFoundError(f"SAVE_PATH does not exist: {self.save_path}")

        class_names = sorted([
            d for d in os.listdir(self.save_path)
            if os.path.isdir(os.path.join(self.save_path, d))
        ])

        os.makedirs(os.path.dirname(self.labels_path), exist_ok=True)
        with open(self.labels_path, "w", encoding="utf-8") as f:
            f.write("\n".join(class_names))

        return class_names

    def load_class_names(self):
        if not os.path.exists(self.labels_path):
            return self.generate_labels_file_from_save_path()

        with open(self.labels_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def extract_dex_number(self, label_name):
        """Extract dex number from label name (first 4 digits, remove leading zeros)"""
        match = re.match(r'^(\d{4})', label_name)
        if match:
            dex_str = match.group(1)
            return int(dex_str.lstrip('0')) if dex_str.lstrip('0') else 0
        return None

    def find_random_pokemon_by_dex(self, dex_number):
        """Find a random pokemon with the given dex number from pokemon_data.txt"""
        matching_pokemon = []
        
        for key, pokemon in self.pokemon_data.items():
            if pokemon.get("dex_number") == dex_number:
                matching_pokemon.append(pokemon["name"])
        
        if matching_pokemon:
            return random.choice(matching_pokemon)
        return None

    def preprocess_image_from_url(self, url):
        try:
            response = requests.get(url)
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to load image from URL: {e}")

        image = image.resize((224, 224))
        image = np.array(image).astype(np.float32) / 255.0
        image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        image = np.transpose(image, (2, 0, 1))  # CHW
        image = np.expand_dims(image, axis=0).astype(np.float32)  # NCHW
        return image

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def predict(self, url):
        image = self.preprocess_image_from_url(url)
        inputs = {self.ort_session.get_inputs()[0].name: image}
        outputs = self.ort_session.run(None, inputs)
        logits = outputs[0][0]
        pred_idx = int(np.argmax(logits))
        prob = float(np.max(self.softmax(logits)))
        
        predicted_label = self.class_names[pred_idx] if pred_idx < len(self.class_names) else f"unknown_{pred_idx}"
        
        # Extract Pokemon name from label (remove dex number and formatting)
        pokemon_name_parts = predicted_label.split('_')[1:]  # Remove first part (dex number)
        pokemon_name = ' '.join(pokemon_name_parts)
        
        # Extract dex number from the predicted label
        dex_number = self.extract_dex_number(predicted_label)
        
        # Find random pokemon with same dex number
        random_name = None
        if dex_number is not None:
            random_name = self.find_random_pokemon_by_dex(dex_number)
        
        if random_name is None:
            random_name = "Unknown"
        
        confidence = f"{prob * 100:.2f}%"
        
        return pokemon_name, random_name, dex_number, confidence

def main():
    try:
        predictor = Prediction()
    except Exception as e:
        print(f"Initialization error: {e}")
        return

    while True:
        url = input("🔍 Who's that Pokémon? Enter image URL (or 'q' to quit): ").strip()
        if url.lower() == 'q':
            break
        try:
            pokemon_name, random_name, dex_number, confidence = predictor.predict(url)
            print(f"Pokémon Name: {pokemon_name}")
            print(f"Random Name: {random_name}")
            print(f"Dex Number: {dex_number}")
            print(f"Confidence: {confidence}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
