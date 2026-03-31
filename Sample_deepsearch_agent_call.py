from src.sample_inference import OpencVino_DeepSearchAgent

analyzer = OpencVino_DeepSearchAgent(
    model_id="OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov", 
    device="CPU"  # or 'GPU' or 'CPU'
)

# 2. Run analysis on any data type via the single function
# The function handles the extraction, retrieval, and LLM final answer
result = analyzer.analyze(
    modality="audio", 
    folder_path="Sample_audio", 
    query="What is this audio about?"
)

print(f"\nFINAL ANSWER:\n{result}")