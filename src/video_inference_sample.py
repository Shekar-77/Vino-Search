from video.Video_analysis_inference import video_inference

def play_top_result(results):
    import os

    if not results:
        print("No results to play.")
        return

    # ---- GET TOP RESULT ----
    top_result = results[0].payload

    video_path = top_result.get("video_path")
    start_time = top_result.get("start_time")
    end_time = top_result.get("end_time")

    if not video_path:
        print("Invalid result payload.")
        return

    print("\n🎯 Playing Best Match")
    print(f"Video: {video_path}")
    print(f"Start Time: {start_time}s → {end_time}s")

    # ---- PLAY VIDEO FROM TIMESTAMP ----
    cmd = f'ffplay -ss {start_time} "{video_path}"'
    os.system(cmd)

engine = video_inference(folder_path='Sample_video',model='blip' ,collection_name='video_inference')
engine.response()
context_data = engine.retrival(query="What is the video about? ")
play_top_result(results = context_data)
