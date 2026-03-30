import os
from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer

# ✅ Load CLIP model with OpenVINO backend
model = SentenceTransformer(
    "clip-ViT-B-32",
    backend="openvino"   # 🔥 enables OpenVINO acceleration
)

def get_image_embeddings(folder_path):
    embeddings = []
    image_paths = []

    for file in os.listdir(folder_path):
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
            img_path = os.path.join(folder_path, file)
            try:
                image = Image.open(img_path).convert("RGB")
                
                # Encode image → embedding
                emb = model.encode(image, convert_to_numpy=True)
                
                embeddings.append(emb)
                image_paths.append(img_path)

            except Exception as e:
                print(f"Error processing {file}: {e}")

    return embeddings, model
