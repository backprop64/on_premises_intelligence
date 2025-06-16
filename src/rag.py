from __future__ import annotations

from typing import List

import numpy as np

from src.database.interface import IngestionInterface
from src.models import SmolVLM2, SmolLM2, TextEmbedder
import config


class RAGPipeline:
    """End-to-end Retrieval-Augmented Generation pipeline (stub)."""

    def __init__(self) -> None:
        self.db_interface = IngestionInterface()
        self.embedder = TextEmbedder()
        # Generator is lazily instantiated in .load() to avoid macOS segfaults
        # Start with text-only model by default, switch to VLM when images are uploaded
        self.generator: SmolVLM2 | SmolLM2 | None = None
        self.current_model_type: str = "text"  # Track current model type
        self._use_debug = config.ENABLE_DEBUG  # Use config for debug output

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def load(self) -> None:
        self.db_interface.load()
        self.embedder.load()

        # Always start with text-only model for faster startup
        if self.generator is None:
            print(f"[RAG] Loading SmolLM2 text model: {config.TEXT_MODEL_SIZE}")
            self.generator = SmolLM2(size=config.TEXT_MODEL_SIZE, device=config.MODEL_DEVICE)
            self.current_model_type = "text"
        self.generator.load()

    def enable_debug(self, enabled: bool = True):
        """Enable/disable debug output for context and prompts."""
        self._use_debug = enabled

    def switch_to_vision_model(self):
        """Switch to vision model for image+text tasks."""
        if self.current_model_type != "vision":
            print(f"[RAG] Switching to SmolVLM2 vision model: {config.VISION_MODEL_SIZE}")
            # Unload current model to free memory
            if self.generator is not None:
                del self.generator
                import gc
                gc.collect()
            
            # Load vision model
            self.generator = SmolVLM2(size=config.VISION_MODEL_SIZE, device=config.MODEL_DEVICE)
            self.generator.load()
            self.current_model_type = "vision"
            print("[RAG] Successfully switched to vision model")

    def switch_to_text_model(self):
        """Switch back to text-only model."""
        if self.current_model_type != "text":
            print(f"[RAG] Switching to SmolLM2 text model: {config.TEXT_MODEL_SIZE}")
            # Unload current model to free memory
            if self.generator is not None:
                del self.generator
                import gc
                gc.collect()
            
            # Load text model
            self.generator = SmolLM2(size=config.TEXT_MODEL_SIZE, device=config.MODEL_DEVICE)
            self.generator.load()
            self.current_model_type = "text"
            print("[RAG] Successfully switched to text model")

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------
    def embed_query(self, query: str) -> np.ndarray:
        return self.embedder.embed(query)

    def retrieve_docs(self, query_embedding: np.ndarray, *, k: int = None):
        """Return list of dicts containing chunk, filename, score."""
        if k is None:
            k = config.RETRIEVAL_K
            
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        if self._use_debug:
            print(f"[RAG] Query embedding shape for search: {query_embedding.shape}")
            
        try:
            scores, ids = self.db_interface.index.search(query_embedding, k=k)
            if self._use_debug:
                print(f"[RAG] Search returned scores shape: {scores.shape}, ids shape: {ids.shape}")
                print(f"[RAG] Index total vectors: {self.db_interface.index.index.ntotal if self.db_interface.index.index else 0}")
                
        except Exception as e:
            print(f"[RAG] Error during vector search: {e}")
            return []
        
        docs = []
        for score, _id in zip(scores.flatten(), ids.flatten()):
            if _id == -1:  # FAISS returns -1 for invalid/empty results
                continue
                
            try:
                rec = self.db_interface.store.get_record(int(_id))
                if rec:  # Make sure record exists
                    docs.append({
                        "chunk": rec.get("chunk", ""),
                        "filename": rec.get("filename", "unknown"),
                        "score": float(score),
                        "id": int(_id)
                    })
            except Exception as e:
                if self._use_debug:
                    print(f"[RAG] Error retrieving record {_id}: {e}")
                continue
                
        if self._use_debug:
            print(f"[RAG] Retrieved {len(docs)} valid docs for query embedding. Top scores: {scores.flatten()[:min(5, len(scores.flatten()))] if scores.size else '[]'}")
            
            # Show each retrieved document separately
            print("Retrieved context (all docs):")
            for idx, d in enumerate(docs, start=1):
                header = f"Document {idx} ({d['filename']}):"
                print(header)
                print(d["chunk"])
                print("-" * 40)
            
        return docs

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate_response(self, query: str, docs_info) -> str:
        context = "\n\n".join(d["chunk"] for d in docs_info)
        
        if self._use_debug:
            print("\n" + "="*80)
            print(f"🔍 DEBUG: CONTEXT AND PROMPT SENT TO {self.current_model_type.upper()} MODEL")
            print("="*80)
            print(f"Query: {query}")
            
            # Show both character and token counts for better monitoring
            context_chars = len(context)
            # Use the generator's tokenizer for token counting to avoid 512-token warning
            gen_tokenizer = getattr(self.generator, 'tokenizer', None)
            if gen_tokenizer is not None:
                try:
                    context_tokens = len(gen_tokenizer.encode(context, add_special_tokens=False))
                    print(f"Context: {context_chars} characters, ~{context_tokens} tokens")
                except Exception:
                    print(f"Context: {context_chars} characters (token count unavailable)")
            else:
                print(f"Context: {context_chars} characters (tokenizer not available)")
                
            print(f"Number of chunks: {len(docs_info)}")
            print("\nFull Context:")
            print("-" * 40)
            print(context)
            print("-" * 40)

            # ------------------------------------------------------------------
            # Show the actual user prompt that will be sent to the language model
            # ------------------------------------------------------------------
            assembled_prompt = (
                f"Here's some relevant information:\n\n{context}\n\nQuestion: {query}"
            )

            print("\nAssembled user prompt (full, no truncation):")
            print("=" * 40)
            print(assembled_prompt)
            print("=" * 40)
            print("=" * 80)
        
        if self.generator is None:
            raise RuntimeError("RAGPipeline not loaded")
        
        # Use appropriate generation method based on current model type
        if self.current_model_type == "text":
            return self.generator.generate(prompt=query, context=context, max_tokens=config.MAX_TOKENS)
        else:  # vision model
            return self.generator.generate(prompt=query, context=context, max_tokens=config.MAX_TOKENS)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def chat(self, query: str, *, k: int = None):
        q_emb = self.embed_query(query)
        docs_info = self.retrieve_docs(q_emb, k=k)
        answer = self.generate_response(query, docs_info)
        # collect filenames ordered by score
        filenames = []
        for d in docs_info:
            fname = d["filename"]
            if fname not in filenames:
                filenames.append(fname)
        return answer, filenames

    def chat_stream(self, query: str, *, k: int = None):
        """Stream tokens in real-time while generating response."""
        q_emb = self.embed_query(query)
        docs_info = self.retrieve_docs(q_emb, k=k)
        
        # Build context
        context = "\n\n".join(d["chunk"] for d in docs_info)
        
        if self._use_debug:
            print("\n" + "="*80)
            print(f"🔍 DEBUG: STREAMING - CONTEXT AND PROMPT SENT TO {self.current_model_type.upper()} MODEL")
            print("="*80)
            print(f"Query: {query}")
            
            # Show both character and token counts for better monitoring
            context_chars = len(context)
            # Use the generator's tokenizer for token counting to avoid 512-token warning
            gen_tokenizer = getattr(self.generator, 'tokenizer', None)
            if gen_tokenizer is not None:
                try:
                    context_tokens = len(gen_tokenizer.encode(context, add_special_tokens=False))
                    print(f"Context: {context_chars} characters, ~{context_tokens} tokens")
                except Exception:
                    print(f"Context: {context_chars} characters (token count unavailable)")
            else:
                print(f"Context: {context_chars} characters (tokenizer not available)")
                
            print(f"Number of chunks: {len(docs_info)}")
            print("\nFull Context:")
            print("-" * 40)
            print(context)
            print("-" * 40)

            # ------------------------------------------------------------------
            # Show the actual user prompt that will be sent to the language model
            # ------------------------------------------------------------------
            assembled_prompt = (
                f"Here's some relevant information:\n\n{context}\n\nQuestion: {query}"
            )

            print("\nAssembled user prompt (full, no truncation):")
            print("=" * 40)
            print(assembled_prompt)
            print("=" * 40)
            print("=" * 80)
        
        # Collect filenames
        filenames = []
        for d in docs_info:
            fname = d["filename"]
            if fname not in filenames:
                filenames.append(fname)
        
        if self.generator is None:
            raise RuntimeError("RAGPipeline not loaded")
            
        # Use appropriate generation method based on current model type
        if self.current_model_type == "text":
            token_stream = self.generator.generate(prompt=query, context=context, max_tokens=config.MAX_TOKENS, stream=True)
        else:  # vision model
            token_stream = self.generator.generate(prompt=query, context=context, max_tokens=config.MAX_TOKENS, stream=True)
            
        return token_stream, filenames


if __name__ == "__main__":  # pragma: no cover
    import sys
    from pathlib import Path

    # Simple test flow – ensure pipeline works after ingesting sample files
    root = Path(__file__).resolve().parents[1]
    pdf_path = root / "test_text.pdf"
    img_path = root / "test_img.png"

    pipeline = RAGPipeline()
    pipeline.load()

    # Ingest once if index empty
    if pipeline.db_interface.index.index is None or pipeline.db_interface.index.index.ntotal == 0:
        pipeline.db_interface.ingest_file(pdf_path)
        pipeline.db_interface.ingest_file(img_path)

    question = "What does the uploaded PDF talk about?"
    answer, filenames = pipeline.chat(question)
    print("Q:", question)
    print("A:", answer)
    print("Filenames:", filenames) 