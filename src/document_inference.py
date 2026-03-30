from documents.main import Document_Storage

result = Document_Storage(collection_name="document",storage_type=":memory:",document_folder_path="Sample_documents")
result.create_vector_store()
img_data_list, table_content_list = result.retrieval(limit=3, query="kevlar")

print(table_content_list, img_data_list)