from app.services.ingestion import load_file, chunk_documents
from pathlib import Path
from app.rag.vectorstore import add_documents

docs = load_file(Path("data/sample_kb/company_it_handbook.md"))

for doc in docs:
    print(doc.page_content)

chunk_doc = chunk_documents(docs)

print(len(chunk_doc))

print(add_documents(chunk_doc))