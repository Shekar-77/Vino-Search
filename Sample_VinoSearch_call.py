from src.sample_inference import OpencVino_DeepSearchAgent

analyzer = OpencVino_DeepSearchAgent(
    model_id="OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov", 
    device="AUTO"  # or 'GPU' or 'CPU' or 'NPU, recommended auto as openvino selects the best device available itself.
)

# 2. Run analysis on any data type via the single function
# The function handles the extraction, retrieval, and LLM final answer

#Select either: OpenVINO/InternVL2-1B-int4-ov or OpenVINO/Phi-3.5-vision-instruct-int4-ov
# result = analyzer.analyze(
#     modality="image", 
#     folder_path="Sample_images", 
#     query="What is this image about?",
#     image_model = "OpenVINO/Phi-3.5-vision-instruct-int4-ov"
#     )

# result = analyzer.analyze(
#     modality="document", 
#     folder_path="Sample_documents", 
#     query="Kevlar?"
# )

# result = analyzer.analyze(
#     modality="audio", 
#     folder_path="Sample_audio", 
#     query="What is this audio about?"
# )

#Select either: Salesforce/blip-image-captioning-base or HuggingFaceTB/SmolVLM2-500M-Video-Instruct
result = analyzer.analyze(
    modality="video", 
    folder_path="Sample_video", 
    query="What is this video about?",
    video_model="Salesforce/blip-image-captioning-base"
)


print(f"\nFINAL ANSWER:\n{result}")
