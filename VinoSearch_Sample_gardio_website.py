import os
import shutil
import gradio as gr
import torch
import openvino as ov

# ── Model registry ─────────────────────────────────────────────────────────────
MODEL_MAP = {
    "1.5B · INT4":  "OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov",
    "1.5B · INT8":  "OpenVINO/Qwen2.5-1.5B-Instruct-int8-ov",
    "1.5B · FP16":  "OpenVINO/Qwen2.5-1.5B-Instruct-fp16-ov",
    "7B · INT4":    "OpenVINO/Qwen2.5-7B-Instruct-int4-ov",
    "7B · INT8":    "OpenVINO/Qwen2.5-7B-Instruct-int8-ov",
    "7B · FP16":    "OpenVINO/Qwen2.5-7B-Instruct-fp16-ov",
    "14B · INT4":   "OpenVINO/Qwen2.5-14B-Instruct-int4-ov",
    "14B · INT8":   "OpenVINO/Qwen2.5-14B-Instruct-int8-ov",
    "14B · FP16":   "OpenVINO/Qwen2.5-14B-Instruct-fp16-ov",
}

MODE_TO_MODALITY = {
    "Documents": "document",
    "Images":    "image",
    "Audio":     "audio",
    "Video":     "video",
}

MODE_EXTENSIONS = {
    "Documents": {".pdf", ".txt", ".docx", ".md", ".csv"},
    "Images":    {".jpg", ".jpeg", ".png", ".webp", ".bmp"},
    "Audio":     {".mp3", ".wav", ".m4a", ".flac", ".ogg"},
    "Video":     {".mp4", ".avi", ".mov", ".mkv", ".webm"},
}

IMAGE_MODEL_MAP = {
    "phi":   "OpenVINO/Phi-3.5-vision-instruct-int4-ov",
    "llava": "OpenVINO/InternVL2-1B-int4-ov",
}

VIDEO_MODEL_MAP = {
    "blip": "Salesforce/blip-image-captioning-base",
    "smol": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
}


# ══════════════════════════════════════════════════════════════════════════════
# Core engine
# ══════════════════════════════════════════════════════════════════════════════

class DeepSearchEngine:
    def __init__(self, model_id: str, device: str):
        from optimum.intel.openvino import OVModelForCausalLM
        from transformers import AutoTokenizer

        self.device   = self._resolve_device(device)
        self.model_id = model_id
        print(f"[Engine] Loading {model_id} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model     = OVModelForCausalLM.from_pretrained(
            model_id, device=self.device, trust_remote_code=True,
            ov_config={"PERFORMANCE_HINT": "LATENCY"},
        )

        self._pipeline  = None
        self._modality  = None
        self._folder    = None

    @staticmethod
    def _resolve_device(requested: str) -> str:
        core      = ov.Core()
        available = core.available_devices
        if requested == "GPU" and "GPU" not in available:
            print("[Engine] Intel GPU unavailable — falling back to CPU.")
            return "CPU"
        if requested == "NPU" and "NPU" not in available:
            return "CPU"
        return requested

    def ingest(self, modality: str, folder_path: str, image_model: str = None, video_model: str = None) -> str:

        modality = modality.lower()
        self._modality    = modality
        self._folder      = folder_path
        self._image_model = image_model
        self._video_model = video_model

        if modality == "document":
            from documents.main import Document_Storage
            self._pipeline = Document_Storage(
                collection_name="docs",
                document_folder_path=folder_path,
            )
            self._pipeline.retrieval(limit=1, query="index")

        elif modality == "image":
            from Images.Image import Image_vector_store
            self._pipeline = Image_vector_store(
                collection_name="image_search",
                folder_path=folder_path,
                device=self.device,
                model=image_model or IMAGE_MODEL_MAP["phi"],
            )
            self._pipeline.creating_vector_store()

        elif modality == "audio":
            from audio.Audio_to_text import AudioSearchEngine
            self._pipeline = AudioSearchEngine(device=self.device)
            self._pipeline.process_folder(folder_path, input_type="audio")

        elif modality == "video":
            from video.Video_analysis_inference import video_inference
            self._pipeline = video_inference(
                folder_path=folder_path,
                model=video_model or VIDEO_MODEL_MAP["blip"],
                device=self.device,
                collection_name="video_inference",
            )
            self._pipeline.response()

        else:
            print(f"Unsupported modality: {modality}")


    def search(self, query: str) -> str:
        if self._pipeline is None or self._modality is None:
            return "⚠️ Please ingest files first."

        modality = self._modality

        if modality == "document":
            combined_data, image_data = self._pipeline.retrieval(limit=3, query=query)
            context = "Text: " + str(combined_data)  + "\nImage data: " + str(image_data)

        elif modality == "image":
            captions = self._pipeline.image_retrieval(query=query)
            context  = f"Visual Description: {captions}"

        elif modality == "audio":
            hits    = self._pipeline.search(query)
            context = " ".join(h.payload["text"] for h in hits)

        elif modality == "video":
            context = self._pipeline.retrival(query=query)

        else:
            return f"Unsupported modality: {modality}"

        return self._infer(context, query)

    def _infer(self, context: str, query: str) -> str:
        prompt = (
            "Use the following context to answer the user query.\n"
            "Context: " + str(context) + "\nQuery: " + str(query)
        )
        messages = [{"role": "user", "content": prompt}]
        text     = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs   = self.tokenizer(text, return_tensors="pt")
        outputs  = self.model.generate(
            **inputs, max_new_tokens=256,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        answer_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()


# ── Engine singleton cache ─────────────────────────────────────────────────────
_engine_cache: dict[tuple, DeepSearchEngine] = {}

def _get_engine(model_label: str, device: str) -> DeepSearchEngine:
    key = (MODEL_MAP[model_label], device)
    if key not in _engine_cache:
        _engine_cache[key] = DeepSearchEngine(MODEL_MAP[model_label], device)
    return _engine_cache[key]


def _dir_fingerprint(path: str) -> str:
    if not os.path.exists(path):
        return ""
    parts = []
    for f in sorted(os.listdir(path)):
        fp = os.path.join(path, f)
        if os.path.isfile(fp):
            parts.append(f"{f}:{os.path.getsize(fp)}")
    return "|".join(parts)

_last_fingerprint: dict[str, str] = {}


def save_uploaded_files(files, save_path: str, mode: str) -> list[str]:
    os.makedirs(save_path, exist_ok=True)
    for f in os.listdir(save_path):
        fp = os.path.join(save_path, f)
        if os.path.isfile(fp):
            os.remove(fp)
    allowed = MODE_EXTENSIONS[mode]
    saved   = []
    for f in (files or []):
        src  = f.name if hasattr(f, "name") else str(f)
        name = os.path.basename(src)
        if os.path.splitext(name)[1].lower() not in allowed:
            continue
        shutil.copy2(src, os.path.join(save_path, name))
        saved.append(name)
    return saved


def update_path(m):
    return {
        "Documents": "data/uploads/documents",
        "Images":    "data/uploads/images",
        "Audio":     "data/uploads/audio",
        "Video":     "data/uploads/video",
    }[m]


def ingest_files(files, mode, save_path, collection, image_model, video_model,
                 model_label, device, max_tokens):
    print("Ingesting the files")
    if not files:
        return "⚠️ Please upload at least one file.", {}, [], []

    saved = save_uploaded_files(files, save_path, mode)
    if not saved:
        allowed = ", ".join(sorted(MODE_EXTENSIONS[mode]))
        return f"⚠️ No valid {mode} files. Allowed: {allowed}", {}, [], []

    engine   = _get_engine(model_label, device)
    modality = MODE_TO_MODALITY[mode]

    selected_image_model = IMAGE_MODEL_MAP.get(image_model, IMAGE_MODEL_MAP["phi"])
    selected_video_model = VIDEO_MODEL_MAP.get(video_model, VIDEO_MODEL_MAP["blip"])

    try:
        print("Starting ingestion")
        engine.ingest(modality, save_path,
                                   image_model=selected_image_model,
                                   video_model=selected_video_model)
    except Exception as e:
        print(f"The error:{e}")
        return f"❌ Ingest error: {e}", {}, [], []

    _last_fingerprint[save_path] = _dir_fingerprint(save_path)
    status_msg="done"

    session = {
        "model_label": model_label,
        "device":      device,
        "mode":        mode,
        "save_path":   save_path,
        "ready":       True,
        "image_model": selected_image_model,
        "video_model": selected_video_model,
    }

    print(f"The status message:{status_msg}")
    welcome = (
        f"**{len(saved)} file(s) indexed** — ready for questions.\n\n"
        f"**Mode:** {mode} &nbsp;·&nbsp; **Model:** {model_label} &nbsp;·&nbsp; **Device:** {engine.device}\n\n"
        "Ask me anything about your files."
    )
    # FIX: history must be list of dicts with role/content keys
    history = [{"role": "assistant", "content": welcome}]
    return status_msg, session, history, history


def user_send(user_message, history, session, _model_label, _device, max_tokens):
    if not user_message.strip():
        yield "", history, history
        return

    if not session or not session.get("ready"):
        h = history + [{"role": "assistant",
                        "content": "⚠️ Please upload and ingest files first using the sidebar."}]
        yield "", h, h
        return

    history = history + [{"role": "user", "content": user_message}]
    yield "", history, history

    model_label = session["model_label"]
    device      = session["device"]
    mode        = session["mode"]
    save_path   = session["save_path"]
    engine      = _get_engine(_model_label, _device)

    current_fp = _dir_fingerprint(save_path)
    if current_fp != _last_fingerprint.get(save_path, ""):
        try:
            print("Running ingestion")
            engine.ingest(MODE_TO_MODALITY[mode], save_path)
            _last_fingerprint[save_path] = current_fp
        except Exception as e:
            reply = f"❌ Re-ingest error: {e}"
            history = history + [{"role": "assistant", "content": reply}]
            yield "", history, history
            return

    try:
        print(f"Sending the user querry:{user_message}")
        reply = engine.search(query=user_message)
    except Exception as e:
        reply = f"❌ Search error: {e}"

    history = history + [{"role": "assistant", "content": reply}]
    yield "", history, history


def clear_chat(session):
    if not session or not session.get("ready"):
        return [], []
    h = [{"role": "assistant",
          "content": "Chat cleared — vector store is still loaded. Ask away!"}]
    return h, h


# ══════════════════════════════════════════════════════════════════════════════
# CSS — Clean professional dark UI, properly scoped for Gradio
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# CSS — Clean, professional, well-organized Gradio layout
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# CSS — Clean, professional, well-organized Gradio layout with better visibility
# ══════════════════════════════════════════════════════════════════════════════
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Color Variables with better contrast ── */
:root {
    --primary: #667eea;
    --primary-dark: #5a67d8;
    --primary-light: #7c8ef0;
    --secondary: #764ba2;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    
    /* Light mode - High contrast */
    --bg-page: #f8fafc;
    --bg-card: #ffffff;
    --bg-input: #ffffff;
    --text-primary: #0f172a;
    --text-secondary: #334155;
    --text-tertiary: #475569;
    --text-muted: #64748b;
    --border-light: #e2e8f0;
    --border-medium: #cbd5e1;
    --hover-bg: #f1f5f9;
}

/* ── Global Reset & Base ── */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--bg-page);
    color: var(--text-primary);
    font-size: 14px;
    line-height: 1.5;
}

/* ── Gradio Container Overrides ── */
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding: 1.5rem !important;
    background: var(--bg-page) !important;
}

/* ── Header ── */
.app-header {
    background: var(--bg-card);
    border-radius: 16px;
    padding: 1rem 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    border: 1px solid var(--border-light);
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.app-header-left {
    display: flex;
    align-items: center;
    gap: 1rem;
}

.app-logo {
    width: 48px;
    height: 48px;
    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 1.2rem;
    color: white;
}

.app-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: black !important;
    background: white !important;
    padding: 6px 10px;
    border-radius: 6px;
}

.app-title span {
    background: none !important;
    -webkit-background-clip: unset !important;
    background-clip: unset !important;
    color: black !important;
}

.app-subtitle {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 0.25rem;
}

.app-badges {
    display: flex;
    gap: 0.5rem;
    align-items: center;
}

.badge {
    font-size: 0.7rem;
    font-weight: 500;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    background: var(--hover-bg);
    color: var(--text-secondary);
    font-family: 'JetBrains Mono', monospace;
}

.badge.green {
    background: #10b98110;
    color: #059669;
    border: 1px solid #10b98130;
}

.pulse {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10b981;
    animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.2); }
}

/* ── Markdown Styling for visibility ── */
.markdown-text, .gr-markdown, .prose {
    color: var(--text-primary) !important;
}

.markdown-text h1, .gr-markdown h1,
.markdown-text h2, .gr-markdown h2,
.markdown-text h3, .gr-markdown h3 {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    margin-bottom: 0.5rem !important;
}

.markdown-text p, .gr-markdown p {
    color: var(--text-secondary) !important;
    line-height: 1.6 !important;
}

/* ── Mode Selector Styling ── */
.mode-selector {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    margin-bottom: 1rem !important;
}

.mode-selector .gr-radio-group {
    display: flex !important;
    gap: 0.75rem !important;
    flex-wrap: wrap !important;
}

.mode-selector .gr-radio-row {
    flex: 1 !important;
    min-width: 100px !important;
}

.mode-selector .gr-radio-row label {
    display: block !important;
    padding: 0.625rem 1rem !important;
    background: var(--bg-input) !important;
    border: 2px solid var(--border-light) !important;
    border-radius: 10px !important;
    text-align: center !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
}

.mode-selector .gr-radio-row label:hover {
    background: var(--hover-bg) !important;
    border-color: var(--primary-light) !important;
    transform: translateY(-1px) !important;
    color: var(--text-primary) !important;
}

.mode-selector .gr-radio-row label:has(input:checked) {
    background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
    border-color: var(--primary) !important;
    color: white !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
}

.mode-selector input[type="radio"] {
    display: none !important;
}

/* ── Sidebar Components ── */
:root {
    --bg-card: #000000;
    --border-light: #222222;
    --text-light: #ffffff;
}

.sidebar-card, .gr-form, .gr-box {
    background: var(--bg-card) !important;
    border-radius: 12px !important;
    border: 1px solid var(--border-light) !important;
    padding: 1rem !important;
    margin-bottom: 1rem !important;
    color: var(--text-light) !important;
}

/* Title styling */
.sidebar-card-title {
    color: var(--text-light);
    font-weight: 600;
}

/* Radio buttons */
.gr-radio label {
    color: var(--text-light) !important;
}

/* Optional: hover effect for nicer UI */
.gr-radio input:checked + span {
    color: #ffffff !important;
    font-weight: 500;
}

/* Form Labels - Make them visible */
label, .gr-form label, .gr-label {
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    margin-bottom: 0.5rem !important;
    display: block !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

/* Input Fields */
input[type="text"], 
input[type="number"], 
textarea,
.gr-textarea textarea {
    background: var(--bg-input) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 8px !important;
    padding: 0.625rem 0.75rem !important;
    font-size: 0.875rem !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
    transition: all 0.15s !important;
}

input::placeholder,
textarea::placeholder {
    color: var(--text-muted) !important;
}

input:focus,
textarea:focus {
    border-color: var(--primary) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
}

/* Dropdowns */
.gr-dropdown select,
.gr-dropdown .wrap-inner {
    background: var(--bg-input) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 8px !important;
    padding: 0.5rem !important;
    font-size: 0.875rem !important;
    color: var(--text-primary) !important;
}

.gr-dropdown select option {
    color: var(--text-primary) !important;
    background: var(--bg-card) !important;
}

/* File Upload */
.gr-file {
    background: var(--bg-input) !important;
    border: 2px dashed var(--border-medium) !important;
    border-radius: 10px !important;
    padding: 1.5rem !important;
    text-align: center !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    color: var(--text-secondary) !important;
}

.gr-file:hover {
    border-color: var(--primary) !important;
    background: var(--hover-bg) !important;
}

/* File list display */
.gr-file .file-list {
    color: var(--text-primary) !important;
}

/* Accordion */
.accordion {
    margin-bottom: 0.5rem !important;
}

.accordion > .label-wrap {
    background: var(--hover-bg) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 10px !important;
    padding: 0.75rem 1rem !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    color: var(--text-primary) !important;
}

.accordion > .label-wrap:hover {
    background: var(--bg-input) !important;
}

.accordion .accordion-content {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
}

/* Slider */
input[type=range] {
    accent-color: var(--primary) !important;
}

/* ── Buttons ── */
.gr-button {
    font-weight: 500 !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}

.gr-button-primary {
    background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
    border: none !important;
    color: white !important;
}

.gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
}

#ingest-btn > button {
    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%) !important;
    width: 100% !important;
    padding: 0.75rem !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    border: none !important;
    color: white !important;
}

#send-btn > button {
    background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
    padding: 0.625rem 1.25rem !important;
    font-weight: 600 !important;
    border: none !important;
    color: white !important;
}

#clear-btn > button {
    background: var(--bg-input) !important;
    border: 1px solid var(--border-light) !important;
    color: var(--text-secondary) !important;
    padding: 0.625rem 1rem !important;
}

#clear-btn > button:hover {
    background: var(--hover-bg) !important;
    border-color: var(--border-medium) !important;
    color: var(--text-primary) !important;
}

.suggestion-pill > button {
    background: var(--bg-input) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 20px !important;
    padding: 0.375rem 1rem !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
}

.suggestion-pill > button:hover {
    background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
    border-color: var(--primary) !important;
    color: white !important;
    transform: translateY(-1px) !important;
}

/* ── Status Box ── */
#status-box textarea {
    background: #fef9e3 !important;
    border-color: #fde047 !important;
    color: #854d0e !important;
    font-weight: 500 !important;
}

/* ── Chatbot ── */
#chatbot {
    background: var(--bg-card) !important;
    border-radius: 12px !important;
    border: 1px solid var(--border-light) !important;
    overflow: hidden !important;
}

#chatbot .wrap {
    background: var(--bg-card) !important;
}

/* User Message */
#chatbot .message.user {
    margin-bottom: 1rem !important;
}

#chatbot .message.user > div {
    background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
    color: white !important;
    border-radius: 12px 12px 4px 12px !important;
    padding: 0.75rem 1rem !important;
    max-width: 70% !important;
    margin-left: auto !important;
}

/* Assistant Message */
#chatbot .message.bot {
    margin-bottom: 1rem !important;
}

#chatbot .message.bot > div {
    background: var(--hover-bg) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 12px 12px 12px 4px !important;
    padding: 0.75rem 1rem !important;
    max-width: 80% !important;
    color: var(--text-primary) !important;
}

#chatbot .message.bot p,
#chatbot .message.bot div {
    color: var(--text-primary) !important;
}

#chatbot .message.bot code {
    background: var(--border-light) !important;
    padding: 0.125rem 0.25rem !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--text-primary) !important;
}

#chatbot .message.bot pre {
    background: var(--text-primary) !important;
    color: var(--bg-card) !important;
    padding: 0.75rem !important;
    border-radius: 8px !important;
    overflow-x: auto !important;
}

/* Chat Input */
#msg-input textarea {
    background: var(--bg-input) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 10px !important;
    padding: 0.75rem !important;
    font-size: 0.875rem !important;
    color: var(--text-primary) !important;
}

#msg-input textarea::placeholder {
    color: var(--text-muted) !important;
}

#msg-input textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--hover-bg);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: var(--border-medium);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--text-muted);
}

/* ── Footer ── */
footer {
    display: none !important;
}

/* ── Responsive Design ── */
@media (max-width: 768px) {
    .gradio-container {
        padding: 1rem !important;
    }
    
    .app-header {
        flex-direction: column;
        gap: 1rem;
        text-align: center;
    }
    
    .gr-row {
        flex-direction: column !important;
    }
    
    .mode-selector .gr-radio-row {
        min-width: calc(50% - 0.5rem) !important;
    }
    
    #chatbot .message.user > div,
    #chatbot .message.bot > div {
        max-width: 90% !important;
    }
}

/* ── Ensure all text is visible ── */
.gr-box, .gr-panel, .gr-form, .gr-group {
    color: var(--text-primary) !important;
}

.gr-box span, .gr-panel span, .gr-form span {
    color: inherit !important;
}

.gr-form .gr-form-label {
    color: var(--text-secondary) !important;
}

/* File names in upload zone */
.gr-file .file-name {
    color: var(--text-primary) !important;
    font-weight: 500 !important;
}

/* Info text */
.info-text, .help-text {
    color: var(--text-muted) !important;
    font-size: 0.75rem !important;
}
"""

THEME = gr.themes.Base(
    font=[gr.themes.GoogleFont("Sora"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
).set(
    body_background_fill="#0b0f1a",
    block_background_fill="#111827",
    block_border_color="rgba(99,179,237,0.15)",
    block_label_text_color="#64748b",
    input_background_fill="#1a2235",
    input_border_color="rgba(99,179,237,0.15)",
    button_primary_background_fill="linear-gradient(135deg, #4299e1, #63b3ed)",
    button_primary_text_color="#0b0f1a"
)

# ══════════════════════════════════════════════════════════════════════════════
# Build UI
# ══════════════════════════════════════════════════════════════════════════════
with gr.Blocks(
    css=CSS,
    theme=THEME,
    title="OpenVINO DeepSearch Agent",
    fill_width=True,
) as demo:

    session_state = gr.State({})
    history_state = gr.State([])

    # ── Header ──────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="app-header">
      <div class="app-header-left">
        <div class="app-logo">OV</div>
        <div>
          <div class="app-title">Vino Search</div>
          <div class="app-subtitle">MULTIMODAL · ON-DEVICE ·</div>
        </div>
      </div>
      <div class="app-badges">
        <span class="badge">Vino Search</span>
        <div class="pulse" title="System ready"></div>
      </div>
    </div>
    """)

    # ── Main layout: sidebar + chat ─────────────────────────────────────────────
    with gr.Row(equal_height=True):

        # ══ SIDEBAR ════════════════════════════════════════════════════════════
        with gr.Column(min_width=290, scale=0):
            gr.HTML("""<div class="sidebar-card">
              <div class="sidebar-card-title">Analysis Mode</div>
            </div>""")

            mode = gr.Radio(
                ["Documents", "Images", "Audio", "Video"],
                value="Documents",
                label="Select mode",
                container=True,
            )

            gr.HTML("""<div class="sidebar-card" style="margin-top:8px">
              <div class="sidebar-card-title">Upload Files</div>
            </div>""")

            files_input = gr.File(
                file_count="multiple",
                label="Drop files here",
                file_types=[".pdf", ".docx", ".txt", ".md", ".csv",
                            ".jpg", ".jpeg", ".png", ".webp",
                            ".mp3", ".wav", ".m4a",
                            ".mp4", ".mov", ".avi", ".mkv"],
            )

            save_path = gr.Textbox(
                value="data/uploads/documents",
                label="Save path",
                visible=False,
            )

            with gr.Accordion("⚙  Advanced settings", open=False):
                collection  = gr.Textbox(value="chatbot_session", label="Collection name")
                image_model = gr.Dropdown(
                    ["phi", "llava"], value="phi", label="Image model")
                video_model = gr.Dropdown(
                    ["blip", "smol"], value="blip", label="Video model")
                model_label = gr.Dropdown(
                    list(MODEL_MAP.keys()), value="1.5B · INT4", label="Qwen2.5 variant")
                device = gr.Dropdown(
                    ["CPU", "GPU", "NPU", "AUTO"], value="CPU", label="Inference device")
                max_tokens = gr.Slider(
                    50, 512, value=256, step=10, label="Max new tokens")

            ingest_btn = gr.Button(
                "⚡  Ingest & Start Chat",
                variant="primary",
                elem_id="ingest-btn",
            )

            status_box = gr.Textbox(
                label="Status",
                interactive=False,
                placeholder="Ready — upload files and click Ingest",
                elem_id="status-box",
                lines=2,
            )

            gr.HTML("""
            <div style="margin-top:16px; padding:12px 14px;
                        background:rgba(99,179,237,0.04);
                        border:1px solid rgba(99,179,237,0.1);
                        border-radius:10px;
                        font-size:0.72rem;
                        color:#475569;
                        font-family:'JetBrains Mono',monospace;
                        line-height:1.7">
              Vector store persists across queries.<br>
              Re-ingest to switch files or mode.
            </div>
            """)

        # ══ CHAT AREA ══════════════════════════════════════════════════════════
        with gr.Column(scale=1):

            chatbot = gr.Chatbot(
                label="",
                elem_id="chatbot",
                height=520,
                show_label=False,
                placeholder=(
                    "### Vino Search - OpenVino's deepsearch agent\n"
                    "Upload files in the sidebar and click **Ingest** to begin."
                ),
                avatar_images=(None, None)
            )

            # ── Input row ──────────────────────────────────────────────────────
            with gr.Row(equal_height=True):
                msg_input = gr.Textbox(
                    placeholder="Ask anything about your files…",
                    show_label=False,
                    lines=1,
                    scale=5,
                    container=False,
                    autofocus=True,
                    elem_id="msg-input",
                )
                send_btn  = gr.Button("Send ↵", scale=0, elem_id="send-btn",  min_width=90)
                clear_btn = gr.Button("Clear",  scale=0, elem_id="clear-btn", min_width=75)

            # ── Suggestion pills ───────────────────────────────────────────────
            with gr.Row():
                sq1 = gr.Button("Summarise key points", elem_classes=["suggestion-pill"], size="sm")
                sq2 = gr.Button("Main topics",           elem_classes=["suggestion-pill"], size="sm")
                sq3 = gr.Button("Tables & numbers",      elem_classes=["suggestion-pill"], size="sm")
                sq4 = gr.Button("Key conclusions",       elem_classes=["suggestion-pill"], size="sm")
                sq5 = gr.Button("Explain simply",        elem_classes=["suggestion-pill"], size="sm")

    # ── Event wiring ────────────────────────────────────────────────────────────

    mode.change(update_path, inputs=mode, outputs=save_path)

    ingest_btn.click(
        ingest_files,
        inputs=[files_input, mode, save_path, collection, image_model,
                video_model, model_label, device, max_tokens],
        outputs=[status_box, session_state, chatbot, history_state],
    )

    def _send(msg, hist, sp, ml, dev, mt):
        yield from user_send(msg, hist, sp, ml, dev, mt)

    send_btn.click(
        _send,
        inputs=[msg_input, history_state, session_state, model_label, device, max_tokens],
        outputs=[msg_input, chatbot, history_state],
    )
    msg_input.submit(
        _send,
        inputs=[msg_input, history_state, session_state, model_label, device, max_tokens],
        outputs=[msg_input, chatbot, history_state],
    )

    clear_btn.click(
        clear_chat,
        inputs=[session_state],
        outputs=[chatbot, history_state],
    )

    SUGGESTIONS = [
        "Summarise the key points from the uploaded content.",
        "What are the main topics covered?",
        "Are there any tables or numerical data? Summarise them.",
        "What key conclusions can be drawn from this material?",
        "Explain the most technical part in simple terms.",
    ]
    for btn, text in zip([sq1, sq2, sq3, sq4, sq5], SUGGESTIONS):
        btn.click(lambda t=text: t, outputs=[msg_input])


# ── Launch ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for path in ["data/uploads/documents", "data/uploads/images",
                 "data/uploads/audio", "data/uploads/video"]:
        os.makedirs(path, exist_ok=True)

    demo.queue(max_size=10).launch()
