import os
import cv2
import torch
import whisper
from PIL import Image
from moviepy import VideoFileClip
from transformers import BlipProcessor, BlipForConditionalGeneration


class BlipVideoCaptionPipeline:

    def __init__(self, device="cpu"):

        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        print(f"Using device: {self.device}")

        # ---- BLIP ----
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)

        # ---- WHISPER ----
        self.whisper_model = whisper.load_model("base")

    # =========================
    # 🎥 VIDEO DURATION
    # =========================
    def get_video_duration(self, video_path):
        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        cap.release()

        if fps == 0:
            return 0

        return frames / fps

    # =========================
    # 🎞 FRAME EXTRACTION
    # =========================
    def extract_frames(self, video_path, start, duration, fps_sample=1, threshold=30):

        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(start * fps)
        end_frame = int((start + duration) * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        prev_gray = None
        frames = []

        frame_id = 0
        interval = int(fps / fps_sample) if fps > 0 else 30
        current = start_frame

        while cap.isOpened() and current < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            save = False

            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                if diff.mean() > threshold:
                    save = True

            if frame_id % interval == 0:
                save = True

            if save:
                frames.append(frame)

            prev_gray = gray
            frame_id += 1
            current += 1

        cap.release()
        return frames

    # =========================
    # 🧠 BLIP CAPTIONING
    # =========================
    def caption_frames(self, frames):

        captions = []

        for frame in frames[:15]:

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = self.processor(image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=30)

            caption = self.processor.decode(out[0], skip_special_tokens=True)
            captions.append(caption)

        return list(set(captions))

    def extract_audio_chunks(self, video_path, chunk_duration=30):

        clip = VideoFileClip(video_path)

        total_duration = int(clip.duration)
        audio_paths = []

        os.makedirs("temp_audio", exist_ok=True)

        # clear old files
        for f in os.listdir("temp_audio"):
            os.remove(os.path.join("temp_audio", f))

        for start in range(0, total_duration, chunk_duration):

            end = min(start + chunk_duration, total_duration)

            subclip = clip.subclipped(start, end)

            audio_path = f"temp_audio/audio_{start}.wav"

            subclip.audio.write_audiofile(
                audio_path,
                fps=16000,
                logger=None
            )

            audio_paths.append((start, end, audio_path))

        clip.close()
        return audio_paths

    # =========================
    # 🗣 WHISPER TRANSCRIPTION
    # =========================
    def transcribe_audio(self, audio_path):

        result = self.whisper_model.transcribe(audio_path)
        return result["text"].strip()

    # =========================
    # 🕒 TIMESTAMP FORMAT
    # =========================
    def format_timestamp(self, sec):
        m = int(sec // 60)
        s = int(sec % 60)
        return f"{m:02d}:{s:02d}"

    # =========================
    # 🎬 MAIN PIPELINE
    # =========================
    def describe_video(self, video_path, chunk_duration=30):

        total_duration = self.get_video_duration(video_path)

        if total_duration <= 0:
            return {}

        video_result = {
            "video_path": video_path,
            "duration": total_duration,
            "chunks": []
        }

        audio_chunks = self.extract_audio_chunks(video_path, chunk_duration)

        for start, end, audio_path in audio_chunks:

            print(f"\nProcessing: {start}s → {end}s")

            # ---- VIDEO ----
            frames = self.extract_frames(video_path, start, end - start)

            video_caption = ""
            if frames:
                captions = self.caption_frames(frames)
                video_caption = " ".join(captions)

            # ---- AUDIO ----
            audio_text = self.transcribe_audio(audio_path)

            video_result["chunks"].append({
                "start_time": start,
                "end_time": end,
                "timestamp": f"[{self.format_timestamp(start)} - {self.format_timestamp(end)}]",
                "video_description": video_caption,
                "audio_text": audio_text
            })

        return video_result

    # =========================
    # 📂 PROCESS FOLDER
    # =========================
    def process_folder(self, folder_path):

        results = []
        formats = (".mp4", ".mkv", ".avi", ".mov")

        for file in os.listdir(folder_path):

            if file.lower().endswith(formats):

                video_path = os.path.join(folder_path, file)

                print(f"\n🎥 Processing {file}")

                result = self.describe_video(video_path)

                if result:
                    results.append(result)

        return results


# # =========================
# # 🚀 RUN
# # =========================
# if __name__ == "__main__":

#     folder_path = "Sample_video"

#     pipeline = MultimodalVideoPipeline(device="cpu")

#     results = pipeline.process_folder(folder_path)

#     print("\n🎯 FINAL OUTPUT:\n")
#     print(results)