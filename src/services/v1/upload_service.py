import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
import ollama
from fastapi import HTTPException, UploadFile
from langchain.schema import Document
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    CSVLoader,
    JSONLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredExcelLoader,
    UnstructuredFileLoader,
    UnstructuredHTMLLoader,
    UnstructuredPowerPointLoader,
    UnstructuredRTFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredXMLLoader,
)

from api.v1.schema.chat import UploadFileChunkResponse
from config.settings_config import get_settings
from core.qdrant import add_documents_to_qdrant, delete_documents
from db.prisma.utils import get_db

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    # Text files
    ".txt": TextLoader,
    ".md": TextLoader,
    ".markdown": TextLoader,
    ".rst": TextLoader,
    ".rtf": UnstructuredRTFLoader,
    # PDF files
    ".pdf": PyMuPDFLoader,
    # CSV and data files
    ".csv": CSVLoader,
    ".tsv": CSVLoader,  # Tab-separated values
    # Microsoft Office documents
    ".docx": UnstructuredWordDocumentLoader,
    ".doc": UnstructuredWordDocumentLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".xls": UnstructuredExcelLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".ppt": UnstructuredPowerPointLoader,
    # Structured data
    ".json": JSONLoader,
    ".jsonl": JSONLoader,
    ".xml": UnstructuredXMLLoader,
    ".yaml": UnstructuredFileLoader,
    ".yml": UnstructuredFileLoader,
    # Web formats
    ".html": UnstructuredHTMLLoader,
    ".htm": UnstructuredHTMLLoader,
    # Email formats
    ".eml": UnstructuredEmailLoader,
    ".msg": UnstructuredEmailLoader,
    # OpenDocument formats (LibreOffice)
    ".odt": UnstructuredFileLoader,  # OpenDocument Text
    ".ods": UnstructuredFileLoader,  # OpenDocument Spreadsheet
    ".odp": UnstructuredFileLoader,  # OpenDocument Presentation
    # Code files (treated as text)
    ".py": TextLoader,
    ".js": TextLoader,
    ".ts": TextLoader,
    ".java": TextLoader,
    ".cpp": TextLoader,
    ".c": TextLoader,
    ".cs": TextLoader,
    ".php": TextLoader,
    ".rb": TextLoader,
    ".go": TextLoader,
    ".rs": TextLoader,
    ".sql": TextLoader,
    # Configuration files
    ".ini": TextLoader,
    ".cfg": TextLoader,
    ".conf": TextLoader,
    ".env": TextLoader,
    # Log files
    ".log": TextLoader,
    # Subtitle files
    ".srt": TextLoader,
    ".vtt": TextLoader,
}


def _get_loader(file_path: str, file_extension: str):
    """Get appropriate loader for file type"""
    loader_class = SUPPORTED_EXTENSIONS.get(file_extension.lower())
    if not loader_class:
        raise ValueError(f"Unsupported file type: {file_extension}")

    # Special handling for different file types
    if file_extension.lower() == ".json":
        return loader_class(file_path, jq_schema=".", text_content=False)
    elif file_extension.lower() == ".jsonl":
        return loader_class(
            file_path, jq_schema=".", text_content=False, json_lines=True
        )
    elif file_extension.lower() in [".tsv"]:
        # Handle tab-separated values
        return CSVLoader(file_path, csv_args={"delimiter": "\t"})
    elif file_extension.lower() in [".yaml", ".yml", ".odt", ".ods", ".odp"]:
        return UnstructuredFileLoader(file_path)
    elif file_extension.lower() in [
        ".py",
        ".js",
        ".ts",
        ".java",
        ".cpp",
        ".c",
        ".cs",
        ".php",
        ".rb",
        ".go",
        ".rs",
        ".sql",
        ".ini",
        ".cfg",
        ".conf",
        ".env",
        ".log",
        ".srt",
        ".vtt",
        ".md",
        ".markdown",
        ".rst",
    ]:
        # Handle code and text files with encoding
        return TextLoader(file_path, encoding="utf-8")

    return loader_class(file_path)


def _get_language_from_extension(file_extension: str) -> Language | None:
    """Map file extensions to Language enum values for code-aware splitting"""
    language_map = {
        ".py": Language.PYTHON,
        ".js": Language.JS,
        ".ts": Language.TS,
        ".java": Language.JAVA,
        ".cpp": Language.CPP,
        ".c": Language.C,
        ".cs": Language.CSHARP,
        ".php": Language.PHP,
        ".rb": Language.RUBY,
        ".go": Language.GO,
        ".rs": Language.RUST,
        ".html": Language.HTML,
        ".htm": Language.HTML,
        ".md": Language.MARKDOWN,
        ".markdown": Language.MARKDOWN,
    }
    return language_map.get(file_extension.lower())


def _process_file(
    user_id: str,
    file_id: str,
    file_path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    """Process a file and return chunked documents"""
    file_extension = Path(file_path).suffix

    try:
        loader = _get_loader(file_path, file_extension)
        documents = loader.load()
    except Exception as e:
        # Fallback to TextLoader for unsupported file types
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            documents = loader.load()
        except Exception:
            raise ValueError(f"Could not process file {file_path}: {str(e)}")

    # Add metadata to documents
    for doc in documents:
        doc.metadata.update(metadata or {})
        doc.metadata["user_id"] = user_id
        doc.metadata["file_id"] = file_id
        doc.metadata["file_name"] = Path(file_path).name
        doc.metadata["file_extension"] = file_extension
        doc.metadata["processed_at"] = datetime.now().isoformat()

    # Choose appropriate text splitter based on file type
    language = _get_language_from_extension(file_extension)

    if language:
        # Use code-aware splitter for programming languages
        text_splitter = RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=1000,
            chunk_overlap=200,
        )
    else:
        # Use default text splitter for other file types
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    chunks = text_splitter.split_documents(documents)

    # Add chunk metadata
    for i, split in enumerate(chunks):
        split.metadata["chunk_index"] = i
        split.metadata["total_chunks"] = len(chunks)

    return chunks


def _get_description(documents: List[Document]):
    file_text = "\n\n".join(doc.page_content for doc in documents[:10])

    prompt = (
        "Summarize the following text in no more than 300 characters. "
        "The summary will be reused as context:\n\n"
        f"{file_text}"
    )

    response = ollama.generate(
        model=get_settings().chat_upload_file_description_model,
        prompt=prompt,
    )
    description = re.sub(
        r"<think>.*?</think>", "", response["response"], flags=re.DOTALL
    ).strip()

    return description


async def upload_file_chunks(
    chunk: UploadFile,
    user_id: str,
    file_id: str,
    file_name: str,
    chunk_index: int,
    total_chunks: int,
) -> UploadFileChunkResponse:
    temp_dir = Path(get_settings().rag_agent_upload_temp_dir) / file_id
    chunks_dir = temp_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Save the incoming chunk
    chunk_path = chunks_dir / f"chunk_{chunk_index}"
    async with aiofiles.open(chunk_path, "wb") as f:
        content = await chunk.read()
        await f.write(content)

    # If it's the last chunk, merge all parts
    if chunk_index == total_chunks - 1:
        final_path = temp_dir / file_name
        async with aiofiles.open(final_path, "wb") as outfile:
            for i in range(total_chunks):
                part = chunks_dir / f"chunk_{i}"
                if not part.exists():
                    raise HTTPException(400, f"Missing chunk {i}")
                async with aiofiles.open(part, "rb") as infile:
                    data = await infile.read()
                    await outfile.write(data)

        try:
            docs = _process_file(user_id, file_id, str(final_path))
            add_documents_to_qdrant(docs)
            description = _get_description(docs)

            db = await get_db()
            await db.uploadfile.create(
                data={
                    "id": file_id,
                    "filename": file_name,
                    "description": description,
                    "userId": user_id,
                }
            )
        except Exception as e:
            logger.error(f"Processing error: {e}")
            raise HTTPException(500, "File processing failed")
        finally:
            shutil.rmtree(temp_dir)

        return UploadFileChunkResponse(
            file_name=file_name, file_id=file_id, complete=True
        )

    return UploadFileChunkResponse(file_name=file_name, file_id=file_id, complete=False)


async def delete_uploaded_file(user_id: str, file_id: str) -> None:
    db = await get_db()
    upload_file = await db.uploadfile.find_first(
        where={"userId": user_id, "id": file_id}
    )
    if not upload_file:
        raise HTTPException(status_code=400, detail="File not found")

    delete_documents({"user_id": user_id, "file_id": file_id})

    await db.uploadfile.delete(where={"id": file_id})
