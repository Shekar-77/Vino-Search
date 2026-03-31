import os
from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer

# ✅ Load CLIP model with OpenVINO backend
import os
from PIL import Image
from sentence_transformers import SentenceTransformer


def load_clip_model(device="AUTO"):
    return SentenceTransformer(
        "clip-ViT-B-32",
        backend="openvino",
        model_kwargs={"device": device}  
    )


def get_image_embeddings(folder_path, batch_size=32):
    model = load_clip_model("AUTO")

    images = []
    image_paths = []

    # 📂 Collect images
    for file in os.listdir(folder_path):
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
            img_path = os.path.join(folder_path, file)
            try:
                image = Image.open(img_path).convert("RGB")
                images.append(image)
                image_paths.append(img_path)
            except Exception as e:
                print(f"Error loading {file}: {e}")

    # 🚀 Batch encoding (much faster than loop)
    embeddings = model.encode(
        images,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    return embeddings, model