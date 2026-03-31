from pathlib import Path
from PIL import Image
from PIL import Image
from pathlib import Path
from PIL import Image 
from optimum.intel.openvino import OVModelForVisualCausalLM
from transformers import AutoProcessor, TextStreamer, AutoTokenizer
from pathlib import Path



# Fetch `notebook_utils` module
import requests

class get_image_caption():
    
    def __init__(self,question:str, folder_path:str, model:str,device:str):
        self.question = question
        self.folder_path = Path(folder_path)
        self.model_name = model
        self.device = device
        self.load_model() 
    
    def load_model(self):
        if  self.model_name == "OpenVINO/Phi-3.5-vision-instruct-int4-ov":
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True
            )

            self.model = OVModelForVisualCausalLM.from_pretrained(
                self.model_name,
                device=self.device,   # 🔥 key fix
                trust_remote_code=True
            )

        elif self.model_name == "OpenVINO/InternVL2-1B-int4-ov":

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )

            self.model = OVModelForVisualCausalLM.from_pretrained(
                self.model_name,
                device=self.device,   # 🔥 key fix
                trust_remote_code=True
            )

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
                    if self.model_name=="OpenVINO/Phi-3.5-vision-instruct-int4-ov" :
                            
                            inputs = self.model.preprocess_inputs(text=self.question, image=img, processor=self.processor)

                            generation_args = { 
                                "max_new_tokens": 50, 
                                "temperature": 0.0, 
                                "do_sample": False,
                                "streamer": TextStreamer(self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
                            } 

                            generate_ids = self.model.generate(**inputs, 
                            eos_token_id=self.processor.tokenizer.eos_token_id, 
                            **generation_args
                            )

                            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
                            response = self.processor.batch_decode(generate_ids, 
                            skip_special_tokens=True, 
                            clean_up_tokenization_spaces=False)[0]
                            caption.append((response, str(file_path))) 

                    elif self.model_name == 'OpenVINO/InternVL2-1B-int4-ov':

                        caption = []

                        prompt = "Describe the image."

                        inputs = self.model.preprocess_inputs(
                            text=prompt,
                            image=img,
                            tokenizer=self.tokenizer,
                            config=self.model.config
                        )

                        generation_args = {
                            "max_new_tokens": 100
                        }

                        generated_ids = self.model.generate(**inputs, **generation_args)

                        generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]

                        output_text = self.tokenizer.batch_decode(
                            generated_ids,
                            skip_special_tokens=True
                        )[0]

                        caption.append([output_text,str(file_path)])
                    
                    else:
                        raise(f"The model:{self.model} is not available pleese select either of the given two options")
    
        return caption
