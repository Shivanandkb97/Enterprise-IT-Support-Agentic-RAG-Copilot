
import json
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.rag.workflow import agent_graph
from app.rag.state import AgentState
from app.rag.vectorstore import add_documents
from app.services.ingestion import (
    load_file,
    chunk_documents,
    calculate_file_hash,
    SUPPORTED_FILE_FORMATS,
)
from app.services.audit import (
    write_audit,
    is_document_duplicate,
    register_document,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")
settings = get_settings()


class CharRequest(BaseModel):
    question: str = Field(min_length=2, max_length=3000)


def sse_event(data: dict) -> str:
    """Format data as a Server-Sent Event."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.get("/health")
def health():
    return {
        "status": "Ok",
        "service": settings.app_name
    }


@router.post("/chat")
async def chat(payload: CharRequest):

    initial: AgentState = {
        "question": payload.question,
        "current_query": payload.question,
        "kb_docs": [],
        "web_results": "",
        "kb_grade": "",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "retry_count": 0,
        "trace": [],
        "citations": [],
    }

    async def event_stream():

        final_data = {
            "answer": "",
            "source_used": "",
            "citations": [],
            "trace": [],
            "current_query": payload.question,
        }

        last_trace = ""

        try:
            yield sse_event({
                "type": "status",
                "message": "Analyzing your question..."
            })

            async for mode, chunk in agent_graph.astream(
                initial,
                stream_mode=["updates", "messages"],
            ):

                if mode == "messages":

                    message, metadata = chunk
                    node = metadata.get("langgraph_node", "")

                    if node in {
                        "generate_from_kb",
                        "generate_from_web",
                        "direct_answer",
                    }:

                        content = message.content

                        if isinstance(content, str):
                            token = content

                        elif isinstance(content, list):
                            token = "".join(
                                block.get("text", "")
                                for block in content
                                if isinstance(block, dict)
                                and block.get("type") == "text"
                            )

                        else:
                            token = ""

                        if token:
                            final_data["answer"] += token

                            yield sse_event({
                                "type": "token",
                                "content": token,
                            })

                elif mode == "updates":

                    for node, output in chunk.items():

                        if not isinstance(output, dict):
                            continue

                        for key in (
                            "answer",
                            "source_used",
                            "citations",
                            "current_query",
                        ):
                            if key in output:
                                final_data[key] = output[key]

                        node_trace = output.get("trace", [])

                        if node_trace:
                            latest_trace = node_trace[-1]

                            if latest_trace != last_trace:
                                last_trace = latest_trace
                                final_data["trace"] = node_trace

                                yield sse_event({
                                    "type": "trace",
                                    "message": latest_trace,
                                })

                        statuses = {
                            "route_question": "Understanding your question...",
                            "retrieve_kb": "Searching company knowledge...",
                            "grade_kb": "Evaluating company information...",
                            "search_web": "Searching external sources...",
                            "grade_web": "Checking external evidence...",
                            "rewrite_query": "Refining your search...",
                            "generate_from_kb": "Preparing your answer...",
                            "generate_from_web": "Preparing your answer...",
                            "direct_answer": "Preparing your answer...",
                            "insufficient": "Checking available information...",
                        }

                        if node in statuses:
                            yield sse_event({
                                "type": "status",
                                "message": statuses[node],
                            })

            write_audit(
                payload.question,
                final_data["source_used"],
                final_data["trace"],
            )

            yield sse_event({
                "type": "final",
                **final_data,
            })

        except Exception:
            logger.exception("Streaming chat failed")

            yield sse_event({
                "type": "error",
                "message": "Something went wrong while generating the response."
            })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    x_admin_key: str = Header(default="")
):
    # 1. Validate admin key
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid admin key"
        )

    # 2. Validate file extension
    original_name = Path(file.filename or "").name
    suffix = Path(original_name).suffix.lower()

    if suffix not in SUPPORTED_FILE_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Supported: {', '.join(sorted(SUPPORTED_FILE_FORMATS))}"
        )

    # 3. Read uploaded file
    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty."
        )

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Use a unique internal filename to prevent overwriting
    # another document with the same original filename.
    stored_path = upload_dir / f"{uuid.uuid4().hex}{suffix}"

    try:
        stored_path.write_bytes(file_bytes)

        # 4. Calculate SHA-256 hash
        file_hash = calculate_file_hash(stored_path)

        # 5. Check whether document already exists
        if is_document_duplicate(file_hash):
            logger.info("Duplicate document detected: %s", original_name)

            return {
                "status": "duplicate",
                "duplicate": True,
                "message": "This document has already been uploaded and indexed.",
                "file": original_name,
                "chunks": 0,
                "ids_created": 0,
            }

        # 6. Load and chunk document
        docs = load_file(stored_path)

        # Preserve the original filename in document metadata
        for doc in docs:
            doc.metadata["source"] = original_name

        chunks = chunk_documents(docs)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No readable content found in the uploaded document."
            )

        # 7. Add document chunks to Pinecone
        ids = add_documents(chunks)

        # 8. Register document only after successful indexing
        registered = register_document(original_name, file_hash)

        if not registered:
            logger.warning(
                "Document was indexed, but its hash was already registered: %s",
                original_name
            )

        logger.info(
            "Document indexed successfully: %s (%s chunks)",
            original_name,
            len(chunks)
        )

        return {
            "status": "success",
            "duplicate": False,
            "message": "Document indexed successfully.",
            "file": original_name,
            "chunks": len(chunks),
            "ids_created": len(ids),
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception("Document ingestion failed: %s", original_name)

        raise HTTPException(
            status_code=500,
            detail="Something went wrong while processing the document."
        )

    finally:
        await file.close()

        # Remove temporary file if it was not successfully registered.
        # Successfully indexed documents remain in the upload directory.
        if stored_path.exists():
            try:
                if not is_document_duplicate(
                    calculate_file_hash(stored_path)
                ):
                    stored_path.unlink()
            except Exception:
                logger.exception(
                    "Could not clean up uploaded file: %s",
                    stored_path
                )
