from Images.blip_weights.main import Blip_model_captioning

from PIL import Image
import numpy as np
import time

device = "CPU"
raw_image = Image.open("image_c33ecd5f.png").convert("RGB")
ov_model, processor = Blip_model_captioning(device = device).load_pipe()
inputs = processor(raw_image, "Describe the image?", return_tensors="pt")

print("Warming up GPU/CPU...")
# Use a very short generation to trigger kernel loading
ov_model.generate_answer(**inputs, max_new_tokens=1)

# --- STEP 5: MEASURED INFERENCE ---
print(f"--- Running Timed Inference on {device} ---")
# --- STEP 5: MEASURED INFERENCE ---
print(f"--- Running Timed Inference on {device} ---")

start = time.perf_counter()

# This will now be much faster
out = ov_model.generate_answer(**inputs, max_length=20)

end = time.perf_counter()
# --------------------------------------------------

duration = end - start

# 1. Flatten for decoding
token_ids = np.array(out).flatten().tolist()
caption = processor.decode(token_ids, skip_special_tokens=True)

# 2. Calculate TPS
# Subtract the input prompt tokens if 'out' includes them, 
# but usually 'generate_answer' returns only the new tokens.
num_tokens = len(token_ids)
tps = num_tokens / duration if duration > 0 else 0

print(f"\n" + "="*40)
print(f"Result: {caption}")
print("="*40)
print(f"Tokens Generated  : {num_tokens}")
print(f"Optimized Latency : {duration:.2f}s")
print(f"Throughput        : {tps:.2f} TPS")
print("="*40)