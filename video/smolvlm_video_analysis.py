import os
import cv2
import torch
import whisper
from PIL import Image
from moviepy import VideoFileClip
from transformers import AutoModelForImageTextToText, AutoProcessor

class SmolVLM2VideoPipeline:

    def __init__(self, device="cpu"):
        self.device = torch.device("cuda" if device=="cuda" and torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device!="cpu" else torch.float32,
            trust_remote_code=True
        ).to(self.device)

        # Whisper for audio transcription
        self.whisper_model = whisper.load_model("base")

    def get_video_duration(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return frames/fps if fps>0 else 0

    def extract_audio_chunks(self, video_path, chunk_duration=30):
        clip = VideoFileClip(video_path)
        dur = int(clip.duration)
        os.makedirs("temp_audio", exist_ok=True)
        for f in os.listdir("temp_audio"):
            os.remove(os.path.join("temp_audio",f))
        chunks = []
        for start in range(0, dur, chunk_duration):
            end = min(start+chunk_duration, dur)
            sub = clip.subclipped(start,end)
            outp = f"temp_audio/audio_{start}.wav"
            sub.audio.write_audiofile(outp, fps=16000, logger=None)
            chunks.append((start,end,outp))
        clip.close()
        return chunks

    def transcribe_audio(self, audio_path):
        res = self.whisper_model.transcribe(audio_path)
        return res["text"].strip()

    def describe_chunk_video(self, video_path, start, end):
        """Run SmolVLM2 on the raw video input for this segment."""
        # SmolVLM2 supports direct video input in the chat template:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": video_path, "start": start, "end": end},
                    {"type": "text", "text": f"Describe the video content between {start}s and {end}s in detail."}
                ]
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )

        # 2. Move to device AND cast to the correct dtype (float16/bfloat16)
        # This is the critical line to fix your error
        inputs = {k: v.to(device=self.device, dtype=self.model.dtype) if torch.is_floating_point(v) else v.to(self.device) 
                for k, v in inputs.items()}

        # 3. Generate should now work
        generated = self.model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False
        )

        out_text = self.processor.batch_decode(
            generated,
            skip_special_tokens=True
        )[0]

        return out_text

    def process_video(self, video_path, chunk_duration=30):
        dur = self.get_video_duration(video_path)
        result = {"video_path": video_path, "duration": dur, "chunks":[]}

        aud_chunks = self.extract_audio_chunks(video_path, chunk_duration)
        for start,end,audio_path in aud_chunks:
            # video-level text (SmolVLM)
            visual_text = self.describe_chunk_video(video_path,start,end)

            # audio transcription
            audio_text = self.transcribe_audio(audio_path)

            combined = (visual_text + " " + audio_text).strip()

            result["chunks"].append({
                "start_time": start,
                "end_time": end,
                "timestamp": f"[{int(start//60):02d}:{int(start%60):02d} - {int(end//60):02d}:{int(end%60):02d}]",
                "video_text": visual_text,
                "audio_text": audio_text,
                "combined_text": combined
            })

        return result

    def process_folder(self, folder_path, chunk_duration=30):
        all_out = []
        formats = (".mp4",".mkv",".avi",".mov")
        for file in os.listdir(folder_path):
            if file.lower().endswith(formats):
                p = os.path.join(folder_path,file)
                print(f"Processing {file}")
                all_out.append(self.process_video(p,chunk_duration))
        return all_out
    
