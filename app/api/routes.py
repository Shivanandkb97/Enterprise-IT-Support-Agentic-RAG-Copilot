
import json
import logging
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
    SUPPORTED_FILE_FORMATS
)
from app.services.audit import write_audit

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

                # Receive generated LLM tokens
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

                # Receive completed node updates
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

                        # Send workflow trace updates
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

                        # Send workflow status
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

            # Record completed request
            write_audit(
                payload.question,
                final_data["source_used"],
                final_data["trace"],
            )

            # Send final response metadata
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
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid admin key"
        )

    suffix = Path(file.filename or "").suffix.lower()

    if suffix not in SUPPORTED_FILE_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Supported: {', '.join(sorted(SUPPORTED_FILE_FORMATS))}"
        )

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    dest = upload_dir / Path(file.filename).name
    dest.write_bytes(await file.read())

    docs = load_file(dest)
    chunks = chunk_documents(docs)
    ids = add_documents(chunks)

    return {
        "message": "Document indexed",
        "file": dest.name,
        "chunks": len(chunks),
        "ids_created": len(ids)
    }
