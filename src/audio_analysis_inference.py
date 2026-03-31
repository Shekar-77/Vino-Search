import os
from audio.Audio_to_text import AudioSearchEngine
from pydub import AudioSegment
from pydub.playback import play

# 1. Initialize Engine
engine = AudioSearchEngine(device = 'CPU')

# 2. Process Folder (Indexing)
FOLDER_TO_SCAN = "Sample_audio" 

if os.path.exists(FOLDER_TO_SCAN):
    # Only index if needed; if using persistent Qdrant, you can skip this after the first run
    engine.process_folder(FOLDER_TO_SCAN, input_type="audio")

    # 3. Search
    query = "what is this about?"
    hits = engine.search(query, limit=3) # Get top 3 matches

    print(f"\n--- Search Results & Playback for: '{query}' ---")
    
    for hit in hits:
        p = hit.payload
        file_name = p['source']
        # Ensure we have the full path to the file
        file_path = os.path.join(FOLDER_TO_SCAN, file_name)
        
        start_ms = p['start_time'] * 1000  # pydub uses milliseconds
        end_ms = p['end_time'] * 1000
        
        print(f"\n[MATCH FOUND]")
        print(f"File: {file_name} ({p['start_time']}s - {p['end_time']}s)")
        print(f"Text: {p['text']}")
        print(f"Score: {hit.score:.4f}")

        # 4. Audio Playback Logic
        if os.path.exists(file_path):
            print(f"--- Playing segment from {file_name} ---")
            try:
                audio = AudioSegment.from_file(file_path)
                segment = audio[start_ms:end_ms]
                play(segment)
            except Exception as e:
                print(f"Playback error: {e}. Ensure ffmpeg is installed.")
        else:
            print(f"Warning: File {file_path} not found for playback.")
            
else:
    print(f"Folder '{FOLDER_TO_SCAN}' not found.")