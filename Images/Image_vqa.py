from .get_image_captions import get_image_caption
from .Image_embedding import get_image_embeddings

# captions = get_image_caption(question="what is the imsge about?",folder_path="Sample_images",model="qwen",device="cpu")

# reponse = captions.caption_generation()
# print(reponse)

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

class Image_vector_store():

    def __init__(self,collection_name:str,folder_path:str,model:str='OpenVINO/Phi-3.5-vision-instruct-int4-ov'):

        self.collection_name = collection_name
        self.folder_path = folder_path
        self.text_embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", 
            backend="openvino"
        )
        self.client = QdrantClient(":memory:")
        self.model = model
    
    def get_caption(self):

        captions = get_image_caption(question="Describe the image in full detail." ,folder_path=self.folder_path,model=self.model,device="cpu")
        reponse = captions.caption_generation()   
        print(f"The reponse is:{reponse}")
        return reponse
    
    def creating_vector_store(self):
        collection_name = self.collection_name

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "image": models.VectorParams(
                    size=512,  # CLIP image embedding size
                    distance=models.Distance.COSINE,
                ),
                "text": models.VectorParams(
                    size=self.text_embedding_model.get_sentence_embedding_dimension(),  # e.g., 384
                    distance=models.Distance.COSINE,
                ),
            },
        )
        captions = self.get_caption()

# 👇 returns (image_embeddings, image_paths)
        image_embeddings, self.clip_model = get_image_embeddings(folder_path=self.folder_path)

        cleaned_captions = []
        payloads = []

        for text, file_path in captions:
            if "assistant\n" in text:
                description = text.split("assistant\n")[-1].strip()
            else:
                description = text.strip()

            cleaned_captions.append(description)

            payloads.append({
                "text": description,
                "image_path": file_path
            })

        # ✅ Text embeddings
        text_embeddings = self.text_embedding_model.encode(cleaned_captions)

        self.client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=idx,
                    vector={
                        "image": image_embeddings[idx].tolist(),
                        "text": text_embeddings[idx].tolist(),
                    },
                    payload=payloads[idx]
                )
                for idx in range(len(payloads))
            ]
        )

        # 2. Encode query with OpenVINO backend
        # Ensure it matches the 384-dimensional vector space
        

        # 3. Perform the search

    def image_retrieval(self,query, limit=5):

        text_vector = self.text_embedding_model.encode(query).tolist()

        image_vector = self.clip_model.encode([query])[0].tolist()

        results = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(
                        query=text_vector,
                        using="text",
                        limit=limit * 5
                    ),
                    models.Prefetch(
                        query=image_vector,
                        using="image",
                        limit=limit * 5
                    )
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True
            ).points
        
        print(f"The result is :{results}")
        return results



