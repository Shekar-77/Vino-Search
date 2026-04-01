## 🧠 Vino Search

### OpenVINO Deep Search AI Assistant on Multimodal Personal Database for AIPC

### 📌 Overview

Vino Search
OpenVINO Deep Search AI Assistant on Multimodal Personal Database for AIPC
Deep Search, as one of the core functions of a personal AI assistant, significantly enhances the user experience by providing information extraction capabilities for various file types (such as Word, PowerPoint, PDF, images, and videos) and supporting multi-dimensional information queries. The localized personal knowledge base not only improves the accuracy and relevance of answers but also protects data security and provides personalized search results based on the user's private data. This project aims to develop a desktop AI localized personal knowledge base search assistant for AI PCs. By building a multimodal personal database and using Retrieval Augmented Generation (RAG) technology, this project leverages this private multimodal data to enhance local large language models (LLMs). Users can interact with the OpenVINO instant messaging AI assistant, ask questions, and perform fuzzy searches using multimodal data.

<img width="896" height="466" alt="image" src="https://github.com/user-attachments/assets/537589e7-bb2f-43d6-938d-0c17ddea6d13" />

###Video Example:
https://github.com/user-attachments/assets/2fddb0c8-05d4-4d32-947d-5a9b78427c2e


Abstract of the Solution

1. VinoSearch is a multimodal retrieval system designed for audio, document, image, and video understanding.  
2. It uses **Qdrant** as the vector database to store embeddings with a strong focus on privacy and local execution.  
3. The system currently uses memory-based storage, with future support planned for Docker and persistent storage.  
4. Lightweight and efficient embedding models such as **MiniLM** and **CLIP** are used for fast semantic retrieval.  
5. OpenVINO integration enables optimized inference on CPU, GPU, and NPU for supported models.  
6. The document pipeline leverages **Docling** to extract text, tables, images, and structured content from files.  
7. Each data type (text, tables, images) is embedded separately to improve retrieval accuracy and structure.  
8. Queries are processed using **multi-vector search** across text, table, and image embeddings.  
9. Results are fused using **Reciprocal Rank Fusion (RRF)** to ensure high-quality retrieval across modalities.  
10. The image pipeline converts images into semantic captions using vision-language models (VLMs).  
11. Caption embeddings are stored to enable natural language querying of visual content.  
12. The audio pipeline uses **Whisper models** to convert speech into text for semantic indexing.  
13. Video analysis is performed by combining visual captions and audio transcripts at chunk level.  
14. The system supports flexible model selection including **Gemma and Qwen-based LLMs** via Hugging Face.  
15. VinoSearch provides a unified, scalable, and privacy-preserving multimodal search experience across all data types.


### 🚀 Goal

To build a **privacy-preserving, multimodal deep search AI assistant** that:

<img width="570" height="650" alt="image" src="https://github.com/user-attachments/assets/ab3ee7fe-c9c4-4590-b4a1-5f4addd92fb3" />



## 🧠 Hardware Acceleration (CPU / GPU / NPU)

This project leverages **Intel OpenVINO** alongside **PyTorch-based models** to enable efficient execution across multiple hardware backends: **CPU, GPU, and NPU**.

---

## ⚙️ Device Configuration

The recommended configuration is:

```python
device = "AUTO"
```

This allows OpenVINO to:

* Automatically select the best available hardware
* Seamlessly fall back to other devices when needed

---

## 🖥️ CPU Support

* ✅ Fully supported across all components
* 🔁 Acts as the default fallback device
* 🛡️ Ensures maximum compatibility and stability

**Used by:**

* Whisper (OpenVINO)
* Embedding models (SentenceTransformer)
* Video processing (OpenCV)
* Qdrant vector database

---

## ⚡ GPU Support (Intel iGPU)

* ✅ Supported via OpenVINO `"GPU"` backend
* 🚀 Provides significant performance improvements

**Used by:**

* Whisper (speech-to-text)
* Embedding models (MiniLM, CLIP)
* Master llm

**Behavior:**

* Automatically selected when using `"AUTO"`
* Falls back to CPU if required

---

## 🧩 NPU Support (Intel Core Ultra)

* Only select OpenVINO-optimized models can utilize NPU
* Master llm

---

This ensures:

* ✅ No crashes due to unsupported hardware
* ✅ Stable execution across all devices

---

Reproducing and running the inference:

markdown
## 🚀 Getting Started

Follow these steps to set up the environment and run the project locally.

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Shekar-77/Vino-Search
cd VinoSearch

2️⃣ Environment Setup
We recommend using Conda with Python 3.12 for the best compatibility with OpenVINO and PyTorch.
bash
conda create -n vinosearch python=3.12 -y
conda activate vinosearch

3️⃣ Install Dependencies

pip install -r requirements.txt
Use code with caution.

🛠️ How to Run
🌐 Web Interface (Gradio)
To launch the interactive web dashboard, run:
python VinoSearch_sample_website.py

Once running, the local URL (usually http://127.0.0.1:7860) will be displayed in your terminal.

💻 Code-Based Inference
To run a sample inference directly through the script:
python VinoSearch_Sample_Run.py


analyzer = OpencVino_DeepSearchAgent(
    model_id="OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov", 
    device="CPU"  # or 'GPU' or 'CPU' or 'NPU
)

You can select any of the gemma and qwen models from the official openvino huggingface repositry: https://huggingface.co/collections/OpenVINO/llm

Running retrival inferences:
You can run retrival inference along all pipelines to visualize the data being retrieved. You can find the code in src
