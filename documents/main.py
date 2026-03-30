import os
import uuid
import io
import base64, easyocr 
import numpy as np
from pathlib import Path
from PIL import Image
from IPython.display import display
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling_core.types.doc import PictureItem, TextItem, TableItem, SectionHeaderItem, FormulaItem, CodeItem
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct, VectorParams, Distance
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption, PowerpointFormatOption, ExcelFormatOption




class Document_Storage():
    
    def __init__(self,collection_name:str,storage_type:str,document_folder_path:str):

        self.collection_name = collection_name
        self.storage_type = storage_type
        self.document_folder_path = document_folder_path
        self.client = QdrantClient(":memory:")
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2', backend="openvino")
        self.clip_model = SentenceTransformer('clip-ViT-B-32', backend='openvino')
    
    def img_to_base64(self,pil_img):
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    
    def generate_converter(self):

        master_opts = PdfPipelineOptions(do_ocr=True, generate_picture_images=True)
        master_opts.do_table_structure = True 

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=master_opts),
                InputFormat.DOCX: WordFormatOption(pipeline_options=master_opts),
                InputFormat.PPTX: PowerpointFormatOption(pipeline_options=master_opts),
                InputFormat.XLSX: ExcelFormatOption(pipeline_options=master_opts),
            }
        )
        return converter
    
    def create_vector_store(self):

        reader = easyocr.Reader(['en'])

        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "text_vec": VectorParams(size=384, distance=Distance.COSINE),
                    "image_vec": VectorParams(size=512, distance=Distance.COSINE),
                    "table_content": VectorParams(size=384, distance=Distance.COSINE),
                    "formula_content": VectorParams(size=384, distance=Distance.COSINE),
                    "code_content": VectorParams(size=384, distance=Distance.COSINE)# Dedicated Table Vector
                }
            )
        #Load models
        #Load the converter
        converter = self.generate_converter()
        document_path = Path(self.document_folder_path)

        for file in document_path.iterdir():
            if file.suffix.lower() not in [".pdf", ".docx", ".doc", ".pptx", ".xlsx"]: continue
            print(f"🚀 Indexing: {file.name}")

            try:
                    result = converter.convert(file)
                    points = []
                    current_section = "General"

                    for element, _level in result.document.iterate_items():
                        # A. Track Sections
                        if isinstance(element, SectionHeaderItem):
                            current_section = element.text.strip()

                        # B. Handle Tables (Using the new 'table_content' vector)
                        elif isinstance(element, TableItem):
                            # FIXED: Pass result.document to fix deprecation warning
                            table_md = element.export_to_markdown(result.document)
                            if table_md.strip():
                                content = f"[{current_section}] Table: {table_md}"
                                t_vec = self.text_model.encode(content, normalize_embeddings=True).tolist()
                                points.append(PointStruct(
                                    id=str(uuid.uuid4()),
                                    vector={"table_content": t_vec}, # Specific vector for tables
                                    payload={"content": table_md, "type": "table", "source": file.name, "section": current_section}
                                ))

                        # C. Handle Images (OCR + CLIP)
                        elif isinstance(element, PictureItem):
                            pil_image = element.get_image(result.document)
                            if pil_image:
                                i_vec = self.image_model.encode(pil_image, normalize_embeddings=True).tolist()
                                img_array = np.array(pil_image.convert('RGB'))
                                ocr_text = " ".join(reader.readtext(img_array, detail=0)).strip()
                                t_vec = self.text_model.encode(ocr_text or "diagram", normalize_embeddings=True).tolist()

                                points.append(PointStruct(
                                    id=str(uuid.uuid4()),
                                    vector={"image_vec": i_vec, "text_vec": t_vec},
                                    payload={"content": ocr_text, "type": "image", "source": file.name, "base64": self.img_to_base64(pil_image)}
                                ))

                        # D. Handle Standard Text
                        elif isinstance(element, TextItem):
                            txt = element.text.strip()
                            if txt:
                                t_vec = self.text_model.encode(f"[{current_section}] {txt}", normalize_embeddings=True).tolist()
                                points.append(PointStruct(
                                    id=str(uuid.uuid4()),
                                    vector={"text_vec": t_vec},
                                    payload={"content": txt, "type": "text", "source": file.name, "section": current_section}
                                ))
                        
                        # --- Retrieve Formulas ---
                        elif isinstance(element, FormulaItem):
                            latex_content = element.text.strip()
                            if latex_content:
                                # Pre-pending "Formula:" helps the text model understand context
                                content = f"[{current_section}] Formula: {latex_content}"
                                f_vec = self.text_model.encode(content, normalize_embeddings=True).tolist()
                                
                                points.append(PointStruct(
                                    id=str(uuid.uuid4()),
                                    vector={"formula_content": f_vec}, # You can also use a dedicated 'formula_vec' if defined
                                    payload={
                                        "content": latex_content, 
                                        "type": "formula", 
                                        "source": file.name, 
                                        "section": current_section
                                    }
                                ))

                        # --- Retrieve Code Blocks ---
                        elif isinstance(element, CodeItem):
                            code_text = element.text.strip()
                            # Docling identifies the language in the 'label' or custom attributes
                            lang = getattr(element, 'language', 'code') 
                            
                            if code_text:
                                content = f"[{current_section}] Code ({lang}):\n{code_text}"
                                c_vec = self.text_model.encode(content, normalize_embeddings=True).tolist()
                                
                                points.append(PointStruct(
                                    id=str(uuid.uuid4()),
                                    vector={"code_content": c_vec}, # Specific 'code_vec' is an option for multi-vector collections
                                    payload={
                                        "content": code_text, 
                                        "type": "code", 
                                        "language": lang,
                                        "source": file.name, 
                                        "section": current_section
                                    }
                                ))

                    if points:
                        self.client.upsert(collection_name=self.collection_name, points=points)
                        print(f"✅ Indexed {len(points)} elements.")

            except Exception as e:
                    print(f"❌ Failed {file.name}: {e}")


    def retrieval(self,limit,query:str):
        # Create vector store,and add the files
        # Generate Query Embeddings
        t_vec = self.text_model.encode(query, normalize_embeddings=True).tolist()
        i_vec = self.image_model.encode(query, normalize_embeddings=True).tolist()

        # Search with RRF Fusion
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(query=i_vec, using="image_vec", limit=limit*5),
                models.Prefetch(query=t_vec, using="text_vec", limit=limit*5),
                models.Prefetch(query=t_vec, using="table_content", limit=limit*5),
                models.Prefetch(query=t_vec, using="formula_content", limit=limit*5),
                models.Prefetch(query=t_vec, using="code_content", limit=limit*5),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True
        ).points

        print(f"\n🔍 Search Results for: '{query}'\n" + "="*65)

        table_content_list = []
        img_data_list = []
        text_data_list = []
        formula_data_list = []
        code_data_list = []

        # Assuming 'hits' is your list of results from Qdrant
        for hit in results:
            payload = hit.payload
            dtype = payload.get("type")
            content = payload.get("content", "")

            if dtype == "table":
                print(f"📊 Table: {content[:50]}...")
                table_content_list.append(content)

            elif dtype == "image":
                print("🖼️ Image/Graph found.")
                img_b64 = payload.get("base64")
                if img_b64:
                    img_data = base64.b64decode(img_b64)
                    img_data_list.append(img_data)

            elif dtype == "formula":
                print(f"➗ Formula: {content}")
                formula_data_list.append(content)

            elif dtype == "code":
                lang = payload.get("language", "unknown")
                print(f"💻 Code ({lang}): {content[:50]}...")
                code_data_list.append(content)

            else:
                # Standard text and others
                print(f"📄 Text: {content[:50]}...")
                text_data_list.append(content)

        # --- The Single Combined List ---
        # This flattens all extracted content into one master list for downstream RAG or display
        combined_data_list = (
            text_data_list + 
            table_content_list + 
            formula_data_list + 
            code_data_list
            # Note: We usually keep img_data (bytes) separate or handle them differently
            # unless your downstream model accepts multimodal inputs.
        )
        
        return combined_data_list, img_data_list

        
            