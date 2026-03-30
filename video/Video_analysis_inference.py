from video.main import Video_analysis
from audio.Audio_to_text import AudioSearchEngine
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client import models
from video.blip_model_video_analysis import BlipVideoCaptionPipeline
from video.smolvlm_video_analysis import SmolVLM2VideoPipeline


class video_inference():
        
        def __init__(self, folder_path:str, model:str, collection_name:str='Video_analysis'):

            self.folder_path = folder_path
            self.client = QdrantClient(":memory:")
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2', backend='openvino')
            self.collection_name =  collection_name
            self.audio_results = []
            self.result = []
            self.model = model
        
        def get_smolvlm_results(self):
        
            result = SmolVLM2VideoPipeline().process_video_folder(folder_path = 'Sample_video')

            return result

        
        def get_blip_results(self):

            pipeline = BlipVideoCaptionPipeline()

            results = pipeline.process_folder(
                self.folder_path,
            )

            return results
              

               
        def response(self):

            if self.model == 'Salesforce/blip-image-captioning-base':
                  all_results = self.get_blip_results()

            elif self.model == 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct':
                 all_results = self.get_smolvlm_results()

            # ---- RECREATE COLLECTION ----
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "caption_vector": models.VectorParams(size=384, distance=models.Distance.COSINE),
                    "audio_vector": models.VectorParams(size=384, distance=models.Distance.COSINE)
                },
            )

            points = []
            point_id = 0

            # ---- ITERATE OVER VIDEOS ----
            for video in all_results:

                video_path = video.get("video_path", "")

                # ---- ITERATE OVER CHUNKS ----
                for chunk in video.get("chunks", []):

                    caption_text = chunk.get("video_description", "")
                    audio_text = chunk.get("audio_text", "")

                    start_time = chunk.get("start_time", 0)
                    end_time = chunk.get("end_time", 0)
                    timestamp = chunk.get("timestamp", "")

                    # ---- CREATE POINT ----
                    points.append(
                        models.PointStruct(
                            id=point_id,
                            vector={
                                "caption_vector": self.encoder.encode(caption_text).tolist(),
                                "audio_vector": self.encoder.encode(audio_text).tolist()
                            },
                            payload={
                                # ---- VIDEO INFO ----
                                "video_path": video_path,

                                # ---- TIMESTAMP INFO ----
                                "start_time": start_time,
                                "end_time": end_time,
                                "timestamp": timestamp,

                                # ---- TEXT DATA ----
                                "caption_text": caption_text,
                                "audio_text": audio_text
                            }
                        )
                    )

                    point_id += 1

            # ---- SINGLE UPSERT ----
            if points:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )

            print(f"✅ Indexed {len(points)} video segments into Qdrant.")
        
        def retrival(self, query:str, limit=3):

            """
            Multimodal Retrieval using RRF (caption + audio + combined)
            """

            query_vector = self.encoder.encode(query).tolist()

            results = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(
                        query=query_vector,
                        using="caption_vector",
                        limit=limit * 5
                    ),
                    models.Prefetch(
                        query=query_vector,
                        using="audio_vector",
                        limit=limit * 5
                    )
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True
            ).points

            print(f"\n🔍 sSearch Results for: '{query}'\n" + "="*65)

            formatted_results = []

            for i, res in enumerate(results):

                payload = res.payload

                formatted = {
                    "rank": i + 1,
                    "score": res.score,

                    # ---- VIDEO INFO ----
                    "video_path": payload.get("video_path"),
                    "start_time": payload.get("start_time"),
                    "end_time": payload.get("end_time"),
                    "timestamp": payload.get("timestamp"),

                    # ---- CONTENT ----
                    "caption": payload.get("caption_text"),
                    "audio": payload.get("audio_text"),
                    "combined": payload.get("combined_text"),
                }

                formatted_results.append(formatted)

                # ---- PRINT ----
                print(f"\nRank {i+1} | Score: {res.score:.4f}")
                print(f"🎬 {formatted['timestamp']}")
                print(f"📹 {formatted['video_path']}")
                print(f"🖼 Caption: {formatted['caption']}")
                print(f"🔊 Audio: {formatted['audio']}")

            return formatted_results

# reponse = Video_analysis(folder_path='Sample_video')

# result = reponse.process_video_folder()

# # result = ["The videp depicts the view of tom laughing and pointing at something"]
# reponse.extract_audio()

# extract_audio_folder="extracted_audio"

# audio_engine = AudioSearchEngine()
# audio_results = audio_engine.process_folder(folder_path=extract_audio_folder, input_type="video")

# print(f"The audio results are:{audio_results}")



# encoder = SentenceTransformer('all-MiniLM-L6-v2')


# collection_name = "video_multimodal_store"

# client = QdrantClient(":memory:")

# client.recreate_collection(
#     collection_name=collection_name,
#     vectors_config={
#         "caption_vector": models.VectorParams(size=384, distance=models.Distance.COSINE),
#         "audio_vector": models.VectorParams(size=384, distance=models.Distance.COSINE),
#     },
# )

# points = []
# point_id = 0
        
#         # 3. Iterate through each video's results
# for i in range (len(result)):
            
#         print("Checking for timestamps.....")

#         final_audio_results = audio_results[0]['text']
#         # 4. Create the dual-vector point
#         points.append(models.PointStruct(
#             id=point_id,
#             vector={
#                 "caption_vector": encoder.encode(result[i]).tolist(),
#                 "audio_vector": encoder.encode(final_audio_results.strip()).tolist()
#             },
#             payload={
#                 "caption_text": result[i],
#                 "audio_text": final_audio_results.strip()
#             }
#         ))
#         point_id += 1

#         client.upsert(collection_name=collection_name, points=points)
#         print(f"Indexed {len(points)} timestamped segments into Qdrant.")



# query_text = "What is this video about?"
# """
# Performs Reciprocal Rank Fusion (RRF) across caption and audio vectors.
# """
# # 1. Setup client and encode query
# collection_name = "video_multimodal_store"

# # 2. Execute Fusion Query
# # This combines the ranks from both vector spaces
# # 1. Encode query for both modalities
# # (Assuming you are using one encoder for both text/image descriptions)
# query_vector = encoder.encode(query_text).tolist()
# limit=3
# # 2. Search with RRF Fusion using your logic
# results = client.query_points(
#     collection_name=collection_name,
#     prefetch=[
#         models.Prefetch(query=query_vector, using="caption_vector", limit=limit*5),
#         models.Prefetch(query=query_vector, using="audio_vector", limit=limit*5),
#     ],
#     query=models.FusionQuery(fusion=models.Fusion.RRF),
#     limit=limit,
#     with_payload=True
# ).points

# # 3. Print results using your formatting
# print(f"\n🔍 Search Results for: '{query_text}'\n" + "="*65)
# print(results)

# response = video_inference(folder_path='Sample_video').response()
# print(f"The final response is:{response}")