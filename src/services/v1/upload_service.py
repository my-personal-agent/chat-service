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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    CSVLoader,
    JSONLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

from api.v1.schema.chat import UploadFileChunkResponse
from config.settings_config import get_settings
from core.qdrant import add_documents_to_qdrant, delete_documents
from db.prisma.utils import get_db

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".txt": TextLoader,
    ".pdf": PyMuPDFLoader,
    ".csv": CSVLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".doc": UnstructuredWordDocumentLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".xls": UnstructuredExcelLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".ppt": UnstructuredPowerPointLoader,
    ".json": JSONLoader,
}


def get_loader(file_path: str, file_extension: str):
    """Get appropriate loader for file type"""
    loader_class = SUPPORTED_EXTENSIONS.get(file_extension.lower())
    if not loader_class:
        raise ValueError(f"Unsupported file type: {file_extension}")

    # Special handling for JSON files
    if file_extension.lower() == ".json":
        return loader_class(file_path, jq_schema=".", text_content=False)

    return loader_class(file_path)


def process_file(
    user_id: str,
    file_id: str,
    file_path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    """Process a file and return chunked documents"""
    file_extension = Path(file_path).suffix

    loader = get_loader(file_path, file_extension)
    documents = loader.load()

    # Add metadata to documents
    for doc in documents:
        doc.metadata.update(metadata or {})
        doc.metadata["user_id"] = user_id
        doc.metadata["file_id"] = file_id
        doc.metadata["file_name"] = Path(file_path).name
        doc.metadata["file_extension"] = file_extension
        doc.metadata["processed_at"] = datetime.now().isoformat()

    # Split documents into chunks
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


def get_description(documents: List[Document]):
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
            docs = process_file(user_id, file_id, str(final_path))
            add_documents_to_qdrant(docs)
            description = get_description(docs)

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
