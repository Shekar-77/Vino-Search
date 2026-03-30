import os
from pathlib import Path
from storage.data_retrival.get_page_images import get_page_images
from storage.pdf_vector_store import create_vectore_store
from models.model import initialize_model
from models.processor import initialize_processor
from llama_index.core import SimpleDirectoryReader
#from video.storing_images_audio.store_images_audio import store_images_audio_of_video
from video.storing_images_audio.storing_images_text_vector_store import storing_images_text_vector_store
from Images.creat_collection import Create_class
from Images.retrive_images import retrive_similar_images

#Initial data
PDF_PATH = "Current Essentials of Medicine(1)(1).pdf"
STORAGE_PATH = "qdrant_storage"
COLLECTION_NAME = "pdf_rag"
document_type="pdf"
model_path = "qwen2.5-vl-ov-int4"

output_video_path = "./video_data/"
output_folder = "./mixed_data/"
output_audio_path = "./mixed_data/output_audio.wav"

filepath = output_video_path + "input_vid.mp4"
Path(output_folder).mkdir(parents=True, exist_ok=True)

#Loading on basis of document type
if(document_type=="pdf"):
        filtered_list, client = create_vectore_store(pdf_path=PDF_PATH,COLLECTION_NAME=COLLECTION_NAME,STORAGE_PATH=STORAGE_PATH,document_type=document_type)
        images_extracted = get_page_images(pdf_path=PDF_PATH,filtered_list=filtered_list)

# elif(document_type=='video'):
#         store_images_audio_of_video(output_folder=output_folder,output_audio_path=output_audio_path)
#         img , filtered_list = storing_images_text_vector_store(output_folder=output_folder, query="hello")
#         images_extracted = SimpleDirectoryReader(
#             input_dir=output_folder, input_files=img
#         ).load_data()
        #Share the image document to the llm
#Initializing the current models, current processor

elif(document_type=="image"):
        recieving_client = Create_class()
        client = recieving_client.create_client("images")
        reuslt = retrive_similar_images(client=client)

        #Image result can be seen this way, send the result to the model accordingly
        # for i, point in enumerate(results.points):
        #     img = Image.open(point.payload["filename"]).convert("RGB")
        #     plt.subplot(1, len(results.points), i + 1)
        #     plt.imshow(img)
        #     plt.axis("off")
        #     plt.title(f"{point.score:.2f}")

        # plt.tight_layout()
        # plt.show()
        
current_model = initialize_model(model_path=model_path)
current_processor = initialize_processor(model_path=model_path)


#starting the inference

def inference():
        messages = [
            {
                "role": "user",
                "content": [
                    # Inject retrieved PIL Image objects
                    *[
                        {"type": "image", "image": img}
                        for img in images_extracted
                    ],
                    # Inject retrieved text chunks with source citations
                    *[
                        {"type": "text", "text": f"[Page {res['page']}] {res['text']}"}
                        for res in filtered_list
                    ],
                    # Final instruction guiding the synthesis and grounding
                    {
                        "type": "text",
                        "text": "Summarize the following content from the document in 3-5 sentences, making sure to cite the page number for any claims."
                    }
                ]
            }
        ]

        inputs = current_processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(current_model.device)
        generated_ids = current_model.generate(**inputs, max_new_tokens=1668)
        # 3. Decode and Print Result
        print(generated_ids)
        output_text = current_processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )
        print(output_text[0])
        client.close()

inference()