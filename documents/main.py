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
    
    def __init__(self,collection_name:str,document_folder_path:str):

        self.collection_name = collection_name
        self.document_folder_path = document_folder_path
        self.client = QdrantClient(":memory:")
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2', backend="openvino",model_kwargs={"device": 'AUTO'})
        self.clip_model = SentenceTransformer('clip-ViT-B-32', backend='openvino',model_kwargs={"device": 'AUTO'})
    
    def img_to_base64(self,pil_img):
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    
    def generate_converter(self):

        master_opts = PdfPipelineOptions(
                do_ocr=True,
                generate_picture_images=True,
                do_table_structure=True,
                images_scale=1.0,          # ✅ Default is 2.0 — halving this cuts RAM by 4x
                generate_page_images=False, # ✅ Don't store full page images in memory
        )
        master_opts.ocr_options = EasyOcrOptions(use_gpu=False)  
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
        import gc
        import pypdf

        reader = easyocr.Reader(['en'])

        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "text_vec":        VectorParams(size=384, distance=Distance.COSINE),
                    "image_vec":       VectorParams(size=512, distance=Distance.COSINE),
                    "table_content":   VectorParams(size=384, distance=Distance.COSINE),
                    "formula_content": VectorParams(size=384, distance=Distance.COSINE),
                    "code_content":    VectorParams(size=384, distance=Distance.COSINE),
                }
            )

        converter     = self.generate_converter()
        document_path = Path(self.document_folder_path).resolve()
        CHUNK_SIZE    = 20  # pages per chunk for large PDFs

        for file in document_path.iterdir():
            if file.suffix.lower() not in [".pdf", ".docx", ".doc", ".pptx", ".xlsx"]:
                continue

            abs_file = file.resolve()
            print(f"🚀 Indexing: {abs_file.name}")

            try:
                # ── PDFs: chunk by page range to avoid OOM ──────────────────────
                if abs_file.suffix.lower() == ".pdf":
                    try:
                        with open(abs_file, "rb") as f:
                            total_pages = len(pypdf.PdfReader(f).pages)
                    except Exception:
                        total_pages = 9999  # fallback — convert whole file

                    print(f"   📄 Total pages: {total_pages}, chunk size: {CHUNK_SIZE}")

                    for start in range(0, total_pages, CHUNK_SIZE):
                        end = min(start + CHUNK_SIZE - 1, total_pages - 1)
                        # Convert to 1-indexed for docling's page_range
                        docling_start = start + 1
                        docling_end = end + 1
                        print(f"   ⚙️  Processing pages {start + 1}–{end + 1}...")

                        try:
                            result = converter.convert(
                                abs_file,
                                page_range=(docling_start, docling_end),  # 1-indexed page range
                            )
                            points = self._extract_points(result, reader, abs_file)

                            if points:
                                self.client.upsert(
                                    collection_name=self.collection_name, points=points)
                                print(f"   ✅ Indexed {len(points)} elements "
                                    f"from pages {start + 1}–{end + 1}")

                        except MemoryError:
                            print(f"   ⚠️  OOM on pages {start + 1}–{end + 1}, skipping chunk")

                        finally:
                            gc.collect()  # free memory between chunks

                # ── Other formats: convert in one shot ──────────────────────────
                else:
                    result = converter.convert(abs_file)
                    points = self._extract_points(result, reader, abs_file)
                    if points:
                        self.client.upsert(
                            collection_name=self.collection_name, points=points)
                        print(f"✅ Indexed {len(points)} elements.")

            except Exception as e:
                print(f"❌ Failed {abs_file.name}: {e}")

            finally:
                gc.collect()  # free memory between files


    def _extract_points(self, result, reader, abs_file: Path) -> list:
        """Extract PointStructs from a docling ConversionResult."""
        points          = []
        current_section = "General"

        for element, _level in result.document.iterate_items():

            # A. Section headers — update running context
            if isinstance(element, SectionHeaderItem):
                current_section = element.text.strip()

            # B. Tables
            elif isinstance(element, TableItem):
                table_md = element.export_to_markdown(result.document)
                if table_md.strip():
                    content = f"[{current_section}] Table: {table_md}"
                    t_vec   = self.text_model.encode(content, normalize_embeddings=True).tolist()
                    points.append(PointStruct(
                        id=str(uuid.uuid4()),
                        vector={"table_content": t_vec},
                        payload={"content": table_md, "type": "table",
                                "source": abs_file.name, "section": current_section}
                    ))

            # C. Images — CLIP + EasyOCR
            elif isinstance(element, PictureItem):
                pil_image = element.get_image(result.document)
                if pil_image:
                    i_vec     = self.clip_model.encode(pil_image, normalize_embeddings=True).tolist()
                    img_array = np.array(pil_image.convert('RGB'))
                    ocr_text  = " ".join(reader.readtext(img_array, detail=0)).strip()
                    t_vec     = self.text_model.encode(
                        ocr_text or "diagram", normalize_embeddings=True).tolist()
                    points.append(PointStruct(
                        id=str(uuid.uuid4()),
                        vector={"image_vec": i_vec, "text_vec": t_vec},
                        payload={"content": ocr_text, "type": "image",
                                "source": abs_file.name, "base64": self.img_to_base64(pil_image)}
                    ))

            # D. Plain text
            elif isinstance(element, TextItem):
                txt = element.text.strip()
                if txt:
                    t_vec = self.text_model.encode(
                        f"[{current_section}] {txt}", normalize_embeddings=True).tolist()
                    points.append(PointStruct(
                        id=str(uuid.uuid4()),
                        vector={"text_vec": t_vec},
                        payload={"content": txt, "type": "text",
                                "source": abs_file.name, "section": current_section}
                    ))

            # E. Formulas
            elif isinstance(element, FormulaItem):
                latex = element.text.strip()
                if latex:
                    f_vec = self.text_model.encode(
                        f"[{current_section}] Formula: {latex}", normalize_embeddings=True).tolist()
                    points.append(PointStruct(
                        id=str(uuid.uuid4()),
                        vector={"formula_content": f_vec},
                        payload={"content": latex, "type": "formula",
                                "source": abs_file.name, "section": current_section}
                    ))

            # F. Code blocks
            elif isinstance(element, CodeItem):
                code = element.text.strip()
                lang = getattr(element, 'language', 'code')
                if code:
                    c_vec = self.text_model.encode(
                        f"[{current_section}] Code ({lang}):\n{code}",
                        normalize_embeddings=True).tolist()
                    points.append(PointStruct(
                        id=str(uuid.uuid4()),
                        vector={"code_content": c_vec},
                        payload={"content": code, "type": "code", "language": lang,
                                "source": abs_file.name, "section": current_section}
                    ))
        return points


    def retrieval(self,limit,query:str):
        # Create vector store,and add the files
        # Generate Query Embeddings
        t_vec = self.text_model.encode(query, normalize_embeddings=True).tolist()
        i_vec = self.clip_model.encode(query, normalize_embeddings=True).tolist()

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

        
            
