import logging
import uuid
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile, status

from api.v1.schema.chat import UploadFileChunkResponse
from services.v1.upload_service import delete_uploaded_file, upload_file_chunks

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chats/upload/chunks", response_model=UploadFileChunkResponse)
async def upload_chunks(
    chunk: UploadFile = File(...),
    filename: str = Form(...),
    chunk_index: int = Form(...),
    total_chunks: int = Form(...),
    file_id: Optional[str] = Form(None),
):
    user_id = "user_id"  # TODO:

    if not file_id:
        file_id = str(uuid.uuid4())

    return await upload_file_chunks(
        chunk=chunk,
        user_id=user_id,
        file_id=file_id,
        file_name=filename,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
    )


@router.delete("/chats/upload/{file_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_upload(file_id: str):
    # todo
    user_id = "user_id"

    await delete_uploaded_file(user_id, file_id)
