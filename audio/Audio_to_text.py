import os
import uuid
from pathlib import Path
from optimum.intel import OVModelForSpeechSeq2Seq
from transformers import AutoProcessor, pipeline
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

class AudioSearchEngine:
    
    def __init__(self,device:str, model_id="distil-whisper/distil-large-v3", save_path="./ov_distil_whisper"):
        self.model_id = model_id
        self.save_path = save_path
        self.collection_name = "audio_search_index"
        self.embed_model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            backend="openvino",
            model_kwargs={"device": 'AUTO'}   # ✅ OpenVINO acceleration
        )
        self.client = QdrantClient(":memory:") # Use "http://localhost:6333" for persistent
        
        self._initialize_whisper()
        self._setup_qdrant()
        if(device == 'NPU'):
            self.device = 'CPU'
        else:
            self.device = device

    def _initialize_whisper(self):
        """Loads or exports the OpenVINO Whisper model."""
        if os.path.exists(self.save_path) and len(os.listdir(self.save_path)) > 0:
            print(f"Loading local model from {self.save_path}...")
            model = OVModelForSpeechSeq2Seq.from_pretrained(self.save_path, compile=True)
        else:
            print(f"First-time setup: Exporting {self.model_id} to OpenVINO...")
            model = OVModelForSpeechSeq2Seq.from_pretrained(self.model_id, export=True, compile=False)
            model.save_pretrained(self.save_path)

        model.compile(device=self.device)

        processor = AutoProcessor.from_pretrained(self.model_id)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=30,
            batch_size=4,
        )

    def _setup_qdrant(self):
        """Initializes the Qdrant collection."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                self.collection_name,
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
            )

    def process_folder(self, folder_path, input_type):
        """Iterates through a folder and indexes all audio files."""

        supported_ext = ('.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac')
        audio_files = [f for f in Path(folder_path).iterdir() if f.suffix.lower() in supported_ext]
        
        print(f"Found {len(audio_files)} audio files in {folder_path}")
        result_list=[]
        for audio_file in audio_files:
            print(f"\nProcessing: {audio_file.name}...")
            result = self.get_result(audio_file)
            if(input_type=="audio"):
                self.index_file(str(audio_file), result=result)
            else:
                result_list.append(result)
                return result_list

    def get_result(self,file_path):
        result = self.pipe(str(file_path), return_timestamps=True)
        return result

    def index_file(self, file_path, result):
        """Transcribes, vectorizes, and uploads a single file to Qdrant."""
        points = []

        for chunk in result["chunks"]:
            text = chunk["text"].strip()
            if not text: continue

            vector = self.embed_model.encode(text).tolist()
            start, end = chunk["timestamp"]

            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "text": text,
                        "start_time": start,
                        "end_time": end,
                        "source": os.path.basename(file_path)
                    }
                )
            )

        if points:
            self.client.upsert(self.collection_name, points=points)
            print(f"Indexed {len(points)} segments from {os.path.basename(file_path)}")

    def search(self, query, limit=3):
        
        """Performs a semantic search across all indexed audio."""
        query_vector = self.embed_model.encode(query).tolist()
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit
        )

        return results.points


