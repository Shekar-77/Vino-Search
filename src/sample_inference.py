import os
import torch
from transformers import AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM
import openvino as ov
from video.Video_analysis_inference import video_inference

# Custom module imports
from audio.Audio_to_text import AudioSearchEngine
from Images.Image import Image_vector_store
from documents.main import Document_Storage

class OpencVino_DeepSearchAgent:

    def __init__(self, model_id, device="CPU"):
        """
        Initializes the Intel-optimized engine.
        :param model_id: Full string of the OpenVINO model (e.g., 'OpenVINO/Qwen2.5-7B-Instruct-int4-ov')
        :param device: 'CPU', 'GPU', or 'NPU'
        """
        # Normalize device name for OpenVINO
        self.final_device = self._validate_device(device.upper())

        print(f"--- Loading Engine: {model_id} on {self.final_device} ---")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = OVModelForCausalLM.from_pretrained(
            model_id,
            device=self.final_device,
            trust_remote_code=True,
            ov_config={"PERFORMANCE_HINT": "LATENCY"}
        )
        self.context = ""

    def _validate_device(self, requested):
        """Ensures OpenVINO doesn't target NVIDIA GPUs."""
        core = ov.Core()
        available = core.available_devices
        # If NVIDIA is present, OpenVINO cannot use 'GPU'
        if requested == "GPU" and (torch.cuda.is_available() or "GPU" not in available):
            print("Notice: NVIDIA detected or Intel GPU missing. Falling back to CPU.")
            return "CPU"
        if requested == "NPU" and "NPU" not in available:
            return "CPU"
        if requested == "AUTO":
            return "AUTO"
        
        return requested
    
    def analyze(self, modality, folder_path, query, image_model:str = None, video_model:str = None):
        """
        One function to rule them all. 
        :param modality: 'document', 'image', 'audio', or 'video'
        :param folder_path: Path to the data
        :param query: The user question
        """

        if modality.lower() == 'document':

            engine = Document_Storage(
                collection_name="docs", 
                document_folder_path=folder_path, 
            )
            engine.create_vector_store()
            combined_data, image_bytes = engine.retrieval(limit=3,query=query)
            self.context_data = f"The documents retrieved data:{combined_data}" 

        elif modality.lower() == 'image':

            engine = Image_vector_store(collection_name='Image_search',folder_path=folder_path, device=self.final_device, model = image_model)
            engine.creating_vector_store()
            captions = engine.image_retrieval(query=query)
            self.context_data = f"Visual Description: {captions}" 

        elif modality.lower() == 'audio':

            engine = AudioSearchEngine()
            engine.process_folder(folder_path,input_type='audio')
            hits = engine.search(query)
            self.context_data = " ".join([h.payload['text'] for h in hits]) 
            print(f"The context data:{self.context_data}")

        elif modality.lower() == 'video':

            engine = video_inference(folder_path='Sample_video', model=video_model,device = self.final_device, collection_name='video_inference')
            engine.response()
            self.context_data = engine.retrival(query=query)
            

        else:
            return "Unsupported modality. Choose: document, image, audio, or video."

        return self._run_inference(query)

    def _run_inference(self, query):
        """Internal helper to generate the final text response."""
        prompt = f"Use the following context to answer the user query.\nContext: {self.context_data}\nQuery: {query}"
        
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt")

        outputs = self.model.generate(**inputs, max_new_tokens=256)
        
        # Trim input from output to return only the answer
        answer_ids = outputs[0][inputs.input_ids.shape[-1]:]
        return self.tokenizer.decode(answer_ids, skip_special_tokens=True)
