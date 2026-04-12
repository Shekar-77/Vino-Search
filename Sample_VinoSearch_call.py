import time
from src.sample_inference import OpencVino_DeepSearchAgent

analyzer = OpencVino_DeepSearchAgent(
    model_id="OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov", 
    device="GPU"  # or 'GPU' or 'CPU' or 'NPU, recommended auto as openvino selects the best device available itself.
)

# 2. Run analysis on any data type via the single function
# The function handles the extraction, retrieval, and LLM final answer

#Select either: OpenVINO/InternVL2-1B-int4-ov or OpenVINO/Phi-3.5-vision-instruct-int4-ov
start_time = time.perf_counter()

# result = analyzer.analyze(
#     modality="image", 
#     folder_path="Sample_images", 
#     query="What do all these images have in common?",
#     image_model = "blip-ov"
#     )

end_time = time.perf_counter()

# result = analyzer.analyze(
#     modality="document", 
#     folder_path="Sample_documents", 
#     query="Kevlar"
# )

# result = analyzer.analyze(
#     modality="audio", 
#     folder_path="Sample_documents", 
#     query="What is this audio about?"
# )

#Select either: Salesforce/blip-image-captioning-base or HuggingFaceTB/SmolVLM2-500M-Video-Instruct
result = analyzer.analyze(
    modality="video", 
    folder_path="Sample_video", 
    query="What is this audio about?",
    video_model="Salesforce/blip-image-captioning-base"
)

duration = end_time - start_time

print(f"\nFINAL ANSWER:\n{result}")
print(f"Total Pipeline Latency: {duration:.2f} seconds")
