import os
from audio.Audio_to_text import AudioSearchEngine
from moviepy import VideoFileClip
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client import models
import os
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import subprocess
import json
import os
import shutil
import cv2
import whisper


class Video_analysis():

    def __init__(self, folder_path:str, device: str = 'cpu'):

        self.input_folder = folder_path
        self.extract_audio_folder = "extracted_audio"
        self.extracted_frames_folder = "extracted_frames"
        self.client = QdrantClient(":memory:")
        self.model, self.processor = self.get_models()
        self.device = device
        self.whisper_model = whisper.load_model("base")

    def extract_audio(self):

        # Create output folder if it doesn't exist
        if not os.path.exists(self.extract_audio_folder):
            os.makedirs(self.extract_audio_folder)

        # 2. Define supported video extensions
        video_extensions = (".mp4", ".mkv", ".mov", ".avi", ".wmv")

        # 3. Iterate through every file in the folder
        for filename in os.listdir(self.input_folder):
            if filename.lower().endswith(video_extensions):
                video_path = os.path.join(self.input_folder, filename)
                
                # Define output filename (e.g., video.mp4 -> video.mp3)
                audio_filename = os.path.splitext(filename)[0] + ".mp3"
                audio_path = os.path.join(self.extract_self.extract_audio_folder, audio_filename)

                try:
                    print(f"Processing: {filename}...")
                    # Load video and extract audio
                    clip = VideoFileClip(video_path)
                    
                    if clip.audio is not None:
                        clip.audio.write_audiofile(audio_path)
                        print(f"Successfully saved to: {audio_filename}")
                    else:
                        print(f"Skipping {filename}: No audio track found.")
                    
                    # Close clips to free up system memory
                    clip.close()
                    
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

        print("Done! All videos processed.")

    def get_models(self):

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            torch_dtype="auto",
            device_map="auto"
        )

        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    
        return model, processor

    def extract_chunk_frames(self,video_path, output_path, start, duration, fps_sample=1, threshold=0.4):

        os.makedirs(output_path, exist_ok=True)

        cap = cv2.VideoCapture(video_path)

        # ---- VIDEO INFO ----
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(start * video_fps)
        end_frame = int((start + duration) * video_fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        prev_gray = None
        frame_id = 0
        saved_count = 0

        sample_interval = int(video_fps / fps_sample)  # e.g. 30 for 1 FPS

        current_frame = start_frame

        while cap.isOpened() and current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            # ---- CONVERT TO GRAYSCALE ----
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            save_frame = False

            # ---- SCENE CHANGE DETECTION ----
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                score = diff.mean()

                if score > threshold:
                    save_frame = True

            # ---- FPS SAMPLING (fallback) ----
            if frame_id % sample_interval == 0:
                save_frame = True

            # ---- SAVE FRAME ----
            if save_frame:
                filename = os.path.join(output_path, f"{saved_count:05d}.jpg")
                cv2.imwrite(filename, frame)
                saved_count += 1

            prev_gray = gray
            frame_id += 1
            current_frame += 1

        cap.release()

        print(f"✅ Saved {saved_count} frames from {start}s to {start+duration}s")

    def get_video_duration(self, video_path):

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("❌ Error opening video")
            return 0

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        cap.release()

        if fps == 0:
            return 0

        duration = frame_count / fps
        return duration
    
    def format_timestamp(self,seconds):
        
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"

    def get_frame_list(self,frames_path, max_frames=16):

        frames = []
        print("Processing has started")
        for file in sorted(os.listdir(frames_path)):
            if file.endswith(".jpg"):
                img = Image.open(os.path.join(frames_path, file)).convert("RGB")
                frames.append(img)
        
        if(len(frames)==0):
            print("got in..")
            img = Image.open('Sample_images/spongebob-cartoon-png-32.png').convert("RGB")
            frames.append(img)
        frames = frames[:max_frames]
        print(f"{len(frames)} frames selected")
        return frames

    def audio_to_text(self, audio_path):
        import torch

        try:
            result = self.whisper_model.transcribe(
                audio_path,
                fp16=torch.cuda.is_available()
            )
            return result["text"].strip()

        except Exception as e:
            print(f"❌ Whisper error: {e}")
            return ""
        
    def query_video(self, video_path, frames_path,chunk_duration=30):

        if os.path.exists(self.extract_audio_folder):
            shutil.rmtree(self.extract_audio_folder)
        os.makedirs(self.extract_audio_folder, exist_ok=True)

        total_duration = self.get_video_duration(video_path)
        results = []
        print(f"Total durarion is {total_duration}")
        print("Hey in")

        for start in range(0, int(total_duration), chunk_duration):
            print("Hey in here as well")

            end = min(start + chunk_duration, total_duration)

            # ---- AUDIO ----
            audio_path = os.path.join(self.extract_audio_folder, f"audio_{start}_{int(end)}.wav")

            os.system(
                f'ffmpeg -ss {start} -t {chunk_duration} -i "{video_path}" '
                f'-vn -acodec pcm_s16le -ar 16000 -ac 1 "{audio_path}"'
            )

            # ---- TRANSCRIBE ----
            audio_text = self.audio_to_text(audio_path)

            # ---- FRAMES ----
            frames_path = f"temp_frames_{start}"
            print(video_path, frames_path)
            self.extract_chunk_frames(video_path, frames_path, start, chunk_duration)

            frames = self.get_frame_list(frames_path)
            if not frames:
                continue

            frames = frames[:20]

            # ---- MODEL ----
            messages = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": frames, "fps": 1.0},
                    {"type": "text", "text": f"Describe video from {start}s to {end}s"}
                ]
            }]

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=128)

            trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, output_ids)
            ]

            caption = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]

            # ---- STORE STRUCTURED ----
            print(f"The caption is :{caption}")
            results.append({
                "video_path": video_path,
                "start_time": start,
                "end_time": end,
                "caption": caption,
                "audio_text": audio_text
            })

            shutil.rmtree(frames_path, ignore_errors=True)

        return results
    
    def process_video_folder(self):

        results = []

        print("Processing folder...")

        for file in os.listdir(self.input_folder):

            if not file.lower().endswith((".mp4", ".mkv", ".mov", ".avi")):
                continue

            print("\n==============================")
            print(f"Processing video: {file}")

            video_path = os.path.join(self.input_folder, file)

            # Unique temp folder per video
            frames_folder = os.path.join(
                self.input_folder,
                f"temp_frames_{os.path.splitext(file)[0]}"
            )

            try:
                # ---- CLEAN FRAMES FOLDER ----
                if os.path.exists(frames_folder):
                    shutil.rmtree(frames_folder)

                os.makedirs(frames_folder, exist_ok=True)

                # ---- PROCESS VIDEO ----
                description = self.query_video(video_path = video_path,frames_path=frames_folder)
                print(f"The description:{description}")

                # ---- STORE RESULT ----
                results.append({
                    "video": file,
                    "description": description
                })

                print(f"✅ Finished: {file}")

            except Exception as e:
                print(f"❌ Error processing {file}: {e}")

        print("\n==============================")
        print("All videos processed.")

        return results
            




# print("All processing complete.")
