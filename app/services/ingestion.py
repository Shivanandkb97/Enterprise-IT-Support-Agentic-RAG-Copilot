
import hashlib
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from docx import Document as DocxDocument


SUPPORTED_FILE_FORMATS = {".pdf", ".txt", ".docx", ".md"}


# Generate a unique SHA-256 hash for the uploaded file
def calculate_file_hash(path: Path) -> str:
    file_bytes = path.read_bytes()
    return hashlib.sha256(file_bytes).hexdigest()


# Load the file
def load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return PyPDFLoader(str(path)).load()

    if suffix in {".txt", ".md"}:
        return TextLoader(str(path), encoding="utf-8").load()

    if suffix == ".docx":
        doc = DocxDocument(str(path))
        text = "\n".join(
            paragraph.text
            for paragraph in doc.paragraphs
            if paragraph.text.strip()
        )
        return [
            Document(
                page_content=text,
                metadata={"source": str(path)}
            )
        ]

    raise ValueError(f"Unsupported file type: {suffix}")


# Split documents into chunks
def chunk_documents(docs: Iterable[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120,
        add_start_index=True
    )

    return splitter.split_documents(list(docs))
