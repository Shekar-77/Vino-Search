from pathlib import Path
from PIL import Image
from PIL import Image
from pathlib import Path
from PIL import Image 
from optimum.intel.openvino import OVModelForVisualCausalLM
from transformers import AutoProcessor, TextStreamer
from pathlib import Path
import openvino_genai as ov_genai
import openvino as ov
import numpy as np
from Images.blip_weights.main import Blip_model_captioning


class get_image_caption():
    
    def __init__(self,question:str, folder_path:str, model:str,device:str):
        self.question = question
        self.folder_path = Path(folder_path)
        self.model_name = model
        self.device = device
        self.load_model() 
    
    def load_model(self):

        if  self.model_name == "blip-ov":

            self.ov_model, self.processor = Blip_model_captioning(device = self.device).load_pipe()
            raw_image = Image.open("Sample_images/image_c33ecd5f.png").convert("RGB")
            inputs = self.processor(raw_image, "Describe the image?", return_tensors="pt")
            self.ov_model.generate_answer(**inputs, max_new_tokens=1) # Warming up the hardware on any image


        elif self.model_name == "InternVL2-1B-int4-ov":

            self.model = ov_genai.VLMPipeline(self.model_name, self.device)
            raw_image = Image.open("Sample_images/spongebob-cartoon-png-32.png").convert("RGB").resize((448, 448))
            image_data = np.array(raw_image)[None] # Add batch dimension [1, H, W, 3]
            image_tensor = ov.Tensor(image_data)
            self.model.generate(self.question, images=[image_tensor], max_new_tokens=1)

        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
    def caption_generation(self):
        extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        caption=[]

        print(f"folder path:{self.folder_path}")

        for file_path in self.folder_path.iterdir():
            if file_path.suffix.lower() in extensions:
                with Image.open(file_path) as img:
                    # Do something with the image
                    print(f"Opened: {file_path.name} | Size: {img.size}")
                    # img.show() # Uncomment to physically open the system viewer
                    if self.model_name == "blip-ov" :
                        raw_image = img.convert("RGB")
                        inputs = self.processor(raw_image, "Describe the image?", return_tensors="pt")
                        out = self.ov_model.generate_answer(**inputs, max_length=20)
                        token_ids = np.array(out).flatten().tolist()
                        response_text = self.processor.decode(token_ids, skip_special_tokens=True)
                        caption.append([response_text,str(file_path)])

                    elif self.model_name == 'InternVL2-1B-int4-ov':

                        raw_image = img.convert("RGB").resize((448, 448))
                        image_data = np.array(raw_image)[None] # Add batch dimension [1, H, W, 3]
                        image_tensor = ov.Tensor(image_data)

                        output_text = self.model.generate(self.question, images=[image_tensor], max_new_tokens=150)
                        response_text = output_text.texts[0]
                        caption.append([response_text,str(file_path)])
                    
                    else:

                        raise(f"The model:{self.model} is not available pleese select either of the given two options")
    
        return caption
