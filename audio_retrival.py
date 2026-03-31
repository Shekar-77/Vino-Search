import os
import shutil

def clear_audio_folder(audio_folder="extracted_audio"):
    if os.path.exists(audio_folder):
        shutil.rmtree(audio_folder)
    os.makedirs(audio_folder, exist_ok=True)

def extract_audio_chunks(video_path, audio_folder,video_duration, chunk_duration=30):
    os.makedirs(audio_folder, exist_ok=True)

    total_duration = video_duration(video_path)
    audio_files = []

    for start in range(0, int(total_duration), chunk_duration):
        end = min(start + chunk_duration, total_duration)

        output_file = os.path.join(
            audio_folder,
            f"audio_{start}_{int(end)}.wav"
        )

        cmd = (
            f'ffmpeg -ss {start} -t {chunk_duration} -i "{video_path}" '
            f'-vn -acodec pcm_s16le -ar 16000 -ac 1 "{output_file}"'
        )

        os.system(cmd)

        audio_files.append((start, end, output_file))

    return audio_files