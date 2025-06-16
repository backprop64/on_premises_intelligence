from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote

from fastapi import FastAPI, File, HTTPException, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse

from src.rag import RAGPipeline
import asyncio
import json
import traceback
import config

# ------------------------------------------------------------------
# Bootstrapping
# ------------------------------------------------------------------

pipeline = RAGPipeline()

# Print configuration on startup
print("\n" + "="*60)
print("🚀 STARTING ON-PREMISES INTELLIGENCE API")
print("="*60)
config.print_config()

# Lazily load heavy models on first use to reduce startup crashes (macOS Torch/FAISS
# occasionally segfault when initialised in the main thread before the event
# loop starts).

_pipeline_loaded: bool = False
_load_lock = asyncio.Lock()

# Semaphore to limit how many generation threads can run simultaneously
_GEN_SEMAPHORE = asyncio.Semaphore(1)

async def _ensure_pipeline_loaded():
    global _pipeline_loaded
    if _pipeline_loaded:
        return
    async with _load_lock:
        if not _pipeline_loaded:
            # Off-thread to avoid blocking the event loop and to isolate any
            # OpenMP initialisation shenanigans.
            await asyncio.to_thread(pipeline.load)
            _pipeline_loaded = True

app = FastAPI(title="On-Premises Intelligence API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Streaming helpers
# ------------------------------------------------------------------

def _sse_format(event: str | None, data: str) -> str:
    """Return a Server-Sent Events formatted string."""
    if event is not None:
        return f"event: {event}\ndata: {data}\n\n"
    return f"data: {data}\n\n"


@app.get("/chat_stream")
async def chat_stream(prompt: str, history: str = ""):
    """Stream tokens back to the client using Server-Sent Events (SSE).

    1. Retrieve docs + filenames (sent as a *meta* event).
    2. Stream tokens from the generator as *token* events.
    3. Emit a final *done* event.
    """

    try:
        await _ensure_pipeline_loaded()
        # Acquire semaphore here to serialize generations
        await _GEN_SEMAPHORE.acquire()
        
        # Step 1 – retrieval (non-streaming)
        print(f"[API] Processing query: {prompt[:100]}...")
        q_emb = pipeline.embedder.embed(prompt)
        print(f"[API] Query embedding shape: {q_emb.shape}")
        
        docs_info = pipeline.retrieve_docs(q_emb, k=config.RETRIEVAL_K)
        print(f"[API] Retrieved {len(docs_info)} documents")
        
        # Build context and collect filenames with chunks
        context_parts = []
        filenames = []
        chunks_by_file = {}
        
        for i, d in enumerate(docs_info):
            chunk = d.get("chunk", "").strip()
            fname = d.get("filename", "unknown")
            score = d.get("score", 0.0)
            
            if chunk:  # Only add non-empty chunks
                context_parts.append(f"Document {i+1} ({fname}):\n{chunk}")
            
            if fname and fname != "unknown" and fname not in filenames:
                filenames.append(fname)
            
            # Group chunks by filename for UI
            if fname not in chunks_by_file:
                chunks_by_file[fname] = []
            if chunk:
                chunks_by_file[fname].append({
                    "text": chunk,
                    "score": score,
                    "index": i + 1
                })
        
        context = "\n\n".join(context_parts)
        
        # ------------------------------------------------------------------
        # 🔗 Assemble full prompt with chat history
        # ------------------------------------------------------------------

        # Clean incoming history string and then trim if necessary
        history = history.strip()

        # -- NEW: Trim history to fit within model context window ------------
        tokenizer = getattr(pipeline.generator, "tokenizer", None)
        if tokenizer is not None:
            try:
                max_ctx_len = config.MAX_CONTEXT_LENGTH
                prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
                context_tokens = len(tokenizer.encode(context, add_special_tokens=False))
                history_tokens = tokenizer.encode(history, add_special_tokens=False) if history else []

                # Calculate how many tokens we can allocate to history so that
                # prompt + context + answer (MAX_TOKENS) fit inside window.
                allowed_history_tokens = max_ctx_len - context_tokens - prompt_tokens - config.MAX_TOKENS

                if allowed_history_tokens < 0:
                    # Context alone is too big – in this scenario we leave
                    # history empty and keep full context (as requested)
                    history_tokens = []
                elif len(history_tokens) > allowed_history_tokens:
                    # Drop oldest tokens to keep the most recent turns
                    # (i.e. remove from start of list)
                    history_tokens = history_tokens[-allowed_history_tokens:]

                # Decode back to text for downstream call
                history = tokenizer.decode(history_tokens, skip_special_tokens=True)
            except Exception as e:
                if config.ENABLE_DEBUG:
                    print(f"[API] Failed to trim history: {e}")
        # ------------------------------------------------------------------

        # Show both character and token counts for better monitoring
        context_chars = len(context)
        try:
            if hasattr(pipeline.embedder, 'tokenizer') and pipeline.embedder.tokenizer:
                context_tokens = len(pipeline.embedder.tokenizer.encode(context, add_special_tokens=False))
                print(f"[API] Built context with {len(context_parts)} parts: {context_chars} chars, ~{context_tokens} tokens")
            else:
                print(f"[API] Built context with {len(context_parts)} parts: {context_chars} chars (tokenizer unavailable)")
        except:
            print(f"[API] Built context with {len(context_parts)} parts: {context_chars} chars (token count failed)")
            
        print(f"[API] Retrieved files: {filenames}")

        # Build a synchronous token generator for SSE with robust error handling
        def _token_gen():
            try:
                # Send metadata first with chunks information
                yield _sse_format("meta", json.dumps({
                    "files": filenames, 
                    "num_docs": len(docs_info),
                    "chunks": chunks_by_file
                }))

                # Even if no documents were retrieved, fall back to normal LLM answering with an empty context.

                # ------------------------------------------------------------------
                # 🚀 Stream tokens from the language model
                # ------------------------------------------------------------------
                # Use the SmolLM2 chat_with_history method (default text model)
                token_stream = pipeline.generator.chat_with_history(
                    prompt=prompt,
                    context=context,
                    history=history,
                    max_tokens=config.MAX_TOKENS,
                    stream=True,
                )

                token_count = 0
                
                try:
                    for tok in token_stream:
                        yield _sse_format("token", tok)
                        token_count += 1
                finally:
                    # Ensure background generation thread is cleaned up
                    t = getattr(token_stream, "_thread", None)
                    if t and t.is_alive():
                        t.join(timeout=0.5)
                    # Release semaphore slot once streaming is finished or errored
                    _GEN_SEMAPHORE.release()

                print(f"[API] Generated {token_count} tokens")
                yield _sse_format("done", "")
                
            except Exception as exc:  # pragma: no cover
                # Log full traceback for debugging
                print(f"[API] Error in token generation: {exc}")
                traceback.print_exc()
                # Send error event so frontend can display the message
                yield _sse_format("error", f"Generation failed: {str(exc)}")
                _GEN_SEMAPHORE.release()

        # ------------------------------------------------------------------
        # 🔍 Debug output – show the full prompt that is sent to the LLM
        # ------------------------------------------------------------------
        if config.ENABLE_DEBUG:
            print("\n" + "="*80)
            print("📝 PROMPT SENT TO TEXT MODEL (including history & chunks)")
            print("="*80)
            print(f"History length: {len(history)} chars")
            print(f"Context length: {len(context)} chars") 
            print(f"Current prompt: {prompt}")
            if history:
                print(f"Previous conversation:\n{history}")
            if context:
                print(f"Retrieved context:\n{context[:500]}{'...' if len(context) > 500 else ''}")
            print("="*80)

        return StreamingResponse(_token_gen(), media_type="text/event-stream")
        
    except Exception as exc:  # pragma: no cover
        print(f"[API] Error in chat_stream: {exc}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc

# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.post("/chat")
async def chat(payload: dict = Body(..., example={"prompt": "Hello"})):
    """Return a generated answer for *prompt*.

    The heavy RAG computation is executed in a worker thread so the asyncio
    event loop remains responsive and we sidestep macOS fork-safety issues
    that can trigger segfaults when running Torch/FAISS on the main thread.
    """

    try:
        await _ensure_pipeline_loaded()
        prompt = str(payload.get("prompt", ""))
        answer, files = await asyncio.to_thread(pipeline.chat, prompt)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"answer": answer, "files": files}


UPLOAD_DIR = Path("opi_file_system")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Static file serving for uploaded files (images/documents)
# ------------------------------------------------------------------

@app.get("/files/{filename}")
async def get_uploaded_file(filename: str):
    """Return the raw bytes for an uploaded file stored in *opi_file_system*. This is
    mainly used by the frontend to render image previews in the sidebar.

    A simple security check ensures the resolved path stays within the upload
    directory so that arbitrary filesystem traversal is not possible.
    """
    # Decode percent-encoding (spaces, commas, etc.) from the URL so we can match
    # the actual filename on disk.
    decoded_name = unquote(filename)

    # Prevent directory traversal attacks
    safe_path = (UPLOAD_DIR / decoded_name).resolve()
    if not str(safe_path).startswith(str(UPLOAD_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not safe_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(safe_path)

@app.get("/config")
async def get_config():
    """Return current configuration for frontend."""
    return {
        "api_host": config.API_HOST,
        "api_port": config.API_PORT,
        "client_timeout": config.CLIENT_TIMEOUT * 1000,  # Convert to milliseconds for JS
        "max_file_size": config.MAX_FILE_SIZE,
        "allowed_extensions": config.ALLOWED_EXTENSIONS,
        "retrieval_k": config.RETRIEVAL_K,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "streaming_enabled": config.ENABLE_STREAMING,
    }

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Store *file* on disk and schedule ingestion."""
    dest = UPLOAD_DIR / file.filename
    
    print(f"[API] Starting upload for file: {file.filename}")
    print(f"[API] File size: {file.size if hasattr(file, 'size') else 'unknown'}")
    print(f"[API] Content type: {file.content_type}")
    
    try:
        # Write file to disk
        with dest.open("wb") as fh:
            content = await file.read()
            fh.write(content)
        
        print(f"[API] File saved to: {dest}")
        
        await _ensure_pipeline_loaded()
        
        # Check current index state before ingestion
        total_before = pipeline.db_interface.index.index.ntotal if pipeline.db_interface.index.index else 0
        print(f"[API] Index vectors before ingestion: {total_before}")
        
        # Ingest the uploaded file in a background thread
        print(f"[API] Starting ingestion for: {dest}")
        await asyncio.to_thread(pipeline.db_interface.ingest_file, dest)
        
        # Check index state after ingestion
        total_after = pipeline.db_interface.index.index.ntotal if pipeline.db_interface.index.index else 0
        new_vectors = total_after - total_before
        print(f"[API] Index vectors after ingestion: {total_after} (+{new_vectors})")
        
        # Verify we can retrieve some records
        if total_after > 0:
            print(f"[API] Testing retrieval with sample query...")
            test_query = "test"
            q_emb = pipeline.embedder.embed(test_query)
            test_docs = pipeline.retrieve_docs(q_emb, k=3)
            print(f"[API] Test retrieval returned {len(test_docs)} docs")
        
        return {
            "filename": file.filename, 
            "status": "processed",
            "vectors_added": new_vectors,
            "total_vectors": total_after
        }
        
    except Exception as exc:  # pragma: no cover
        print(f"[API] Upload/ingestion error: {exc}")
        traceback.print_exc()
        
        # Clean up file if ingestion failed
        if dest.exists():
            try:
                dest.unlink()
                print(f"[API] Cleaned up failed upload: {dest}")
            except Exception as cleanup_err:
                print(f"[API] Failed to clean up file: {cleanup_err}")
        
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(exc)}") from exc

# ------------------------------------------------------------------
# Search endpoint for sidebar file search
# ------------------------------------------------------------------

@app.get("/search_docs")
async def search_docs(query: str, k: int | None = None):
    """Return filenames and chunk previews relevant to *query* without generation.

    The response schema mirrors the *meta* event used by /chat_stream so that
    the frontend can reuse the same rendering logic:

        {
            "files": ["file1.txt", "file2.md", ...],
            "chunks": {
                "file1.txt": [{"text": "...", "score": 0.93, "index": 1}, ...],
                ...
            }
        }
    """

    if not query.strip():
        return {"files": [], "chunks": {}}

    await _ensure_pipeline_loaded()

    # Embed query and perform retrieval
    q_emb = pipeline.embedder.embed(query)
    docs_info = pipeline.retrieve_docs(q_emb, k=k or config.RETRIEVAL_K)

    filenames: list[str] = []
    chunks_by_file: dict[str, list[dict]] = {}

    for idx, d in enumerate(docs_info, start=1):
        fname = d.get("filename", "unknown")
        if fname not in filenames:
            filenames.append(fname)

        if fname not in chunks_by_file:
            chunks_by_file[fname] = []

        chunks_by_file[fname].append({
            "text": d.get("chunk", ""),
            "score": d.get("score", 0.0),
            "index": idx,
        })

    return {"files": filenames, "chunks": chunks_by_file}

# ------------------------------------------------------------------
# Maintenance
# ------------------------------------------------------------------


# Notify backend to reload indices in case multiple workers are used
# Note: Reload endpoint removed due to threading issues - SQLite is now thread-safe


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "api.opi_api:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=False,
        log_level="info",
    ) 