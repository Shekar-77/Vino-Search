## 🧠 Vino Search

### OpenVINO Deep Search AI Assistant on Multimodal Personal Database for AIPC

### 📌 Overview

**Vino Search** is a desktop-based AI assistant designed to enable **deep search over a localized multimodal personal knowledge base**. It leverages **OpenVINO** and **Retrieval Augmented Generation (RAG)** to provide intelligent, secure, and personalized information retrieval directly on AI PCs.

---

### 🎯 Problem Statement

Modern AI assistants often rely heavily on cloud-based data, which leads to:

* Limited personalization
* Privacy concerns
* Inability to deeply search personal files

Users need a system that can:

* Understand and search **private, local data**
* Work across **multiple file formats**
* Provide **context-aware and accurate responses**

---

### 💡 Proposed Solution

Vino Search builds a **multimodal personal database** and integrates it with a **local LLM** to enable deep, context-aware querying.

The system supports:

* 📄 Documents: Word, PDF, PowerPoint
* 🖼️ Images
* 🎥 Videos

Using **RAG (Retrieval Augmented Generation)**:

* Relevant data is retrieved from the local database
* The LLM generates responses grounded in user-specific content

---

### ⚙️ Key Features

* 🔍 **Deep Search Capability**
  Perform semantic and fuzzy searches across personal data

* 🧠 **Multimodal Understanding**
  Extract and analyze information from text, images, and videos

* 🔒 **Privacy-Focused**
  All processing happens locally on the user's device

* ⚡ **Optimized with OpenVINO**
  Efficient inference on AI PCs with hardware acceleration

* 💬 **Interactive AI Assistant**
  Chat-based interface for natural interaction

---

### 🏗️ System Workflow

1. **Data Ingestion**

   * User uploads or indexes local files
   * Files are processed into embeddings

2. **Multimodal Database Creation**

   * Text, image, and video data stored in a unified format

3. **Query Processing**

   * User asks a question via chat interface

4. **Retrieval (RAG)**

   * Relevant data is fetched from the database

5. **Response Generation**

   * Local LLM generates a contextual, accurate response

---

### 🚀 Goal

To build a **fully local, privacy-preserving, multimodal AI assistant** that:

* Enhances productivity
* Provides deep insights from personal data
* Runs efficiently on AI-powered PCs

**Current Directory**
├── audio/
│   └── Audio_to_text.py
├── documents/
│   └── main.py
├── Images/
│   ├── get_image_captions.py
│   ├── Image_embedding.py
│   └── Image_vqa.py
├── Sample_audio/
├── Sample_documents/
│   ├── Current Essentials of Medicine(1)(1).pdf
│   ├── doc.pdf
│   └── KEVLAR 1.docx
├── Sample_images/
│   └── spongebob-cartoon-png-32.png
├── Sample_video/
│   ├── new.mp4
│   └── new2.mp4
├── src/
│   ├── audio_analysis_inference.py
│   ├── document_inference.py
│   ├── image_inference.py
│   ├── inference.py
│   ├── sample_inference.py
│   └── video_inference_sample.py
├── temp_audio/
├── video/
│   ├── blip_model_video_analysis.py
│   ├── smolvlm_video_analysis.py
│   └── Video_analysis_inference.py
├── .gitignore
├── .txt
├── audio_retrival.py
├── main.py
└── requirements.txt
|__ VinoSearch_sample_website.py
|__ VinoSearch_incode_inference.py


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
git clone https://github.com/Shekar-77
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
