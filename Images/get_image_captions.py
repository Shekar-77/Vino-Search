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
        self.model = model
        self.device = device

    def phi_model(self):
        processor = AutoProcessor.from_pretrained(self.model, trust_remote_code=True)
        ov_model = OVModelForVisualCausalLM.from_pretrained(self.model, trust_remote_code=True)
        return ov_model,processor
    
    def intern_model(self):
        model_id = "OpenVINO/InternVL2-1B-int4-ov"

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        model = OVModelForVisualCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True
        )

        return tokenizer, model
    
    def caption_generation(self):
        extensions = {".jpg", ".jpeg", ".png", ".bmp",".avif"}
        caption=[]

        print(f"folder path:{self.folder_path}")

        for file_path in self.folder_path.iterdir():
            if file_path.suffix.lower() in extensions:
                with Image.open(file_path) as img:
                    # Do something with the image
                    print(f"Opened: {file_path.name} | Size: {img.size}")
                    # img.show() # Uncomment to physically open the system viewer
                    if self.model=="OpenVINO/Phi-3.5-vision-instruct-int4-ov" :
                            
                            model, processor = self.phi_model()
                            inputs = model.preprocess_inputs(text=self.question, image=img, processor=processor)

                            generation_args = { 
                                "max_new_tokens": 50, 
                                "temperature": 0.0, 
                                "do_sample": False,
                                "streamer": TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
                            } 

                            generate_ids = model.generate(**inputs, 
                            eos_token_id=processor.tokenizer.eos_token_id, 
                            **generation_args
                            )

                            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
                            response = processor.batch_decode(generate_ids, 
                            skip_special_tokens=True, 
                            clean_up_tokenization_spaces=False)[0]
                            caption.append((response, str(file_path))) 

                    elif self.model == 'OpenVINO/InternVL2-1B-int4-ov':

                        tokenizer, model = self.intern_model()

                        caption = []

                        prompt = "Describe the image."

                        inputs = model.preprocess_inputs(
                            text=prompt,
                            image=img,
                            tokenizer=tokenizer,
                            config=model.config
                        )

                        generation_args = {
                            "max_new_tokens": 100
                        }

                        generated_ids = model.generate(**inputs, **generation_args)

                        generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]

                        output_text = tokenizer.batch_decode(
                            generated_ids,
                            skip_special_tokens=True
                        )[0]

                        caption.append([output_text,str(file_path)])
                    
                    else:
                        raise(f"The model:{self.model} is not available pleese select either of the given two options")
    
        return caption
