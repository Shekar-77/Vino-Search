from documents.main import Document_Storage

result = Document_Storage(collection_name="document",document_folder_path="Sample_documents")
result.create_vector_store()
combined_data_list, img_data_list = result.retrieval(limit=3, query="kevlar")

print(table_content_list)