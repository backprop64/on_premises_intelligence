from pathlib import Path
from typing import Union, Literal

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, TextIteratorStreamer


MODEL_MAP = {
    "2.2b": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "500m": "HuggingFaceTB/SmolVLM2-500M-Instruct",
    "256m": "HuggingFaceTB/SmolVLM2-256M-Instruct",
}


class SmolVLM2:
    """Multimodal chat wrapper for SmolVLM2."""

    def __init__(
        self,
        *,
        device: str = "cpu",
        size: Literal["2.2b", "500m", "256m"] = "256m",
    ) -> None:
        self.device = device
        self.model_name = MODEL_MAP[size]

        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        ).to(device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Chat API
    # ------------------------------------------------------------------
    def chat(
        self,
        *,
        parts: list[Union[str, Path, Image.Image]],
        max_tokens: int = 1024,
        stream: bool = False,
    ) -> str:
        """Chat with arbitrary interleaved text & images.

        Each element of *parts* can be:
        • str  – treated as text fragment
        • Path | PIL.Image – treated as image
        Items keep order, enabling multi-image prompts like

        parts=["Compare these: ", img1, img2]
        """

        content = []
        for p in parts:
            if isinstance(p, (str,)):
                content.append({"type": "text", "text": p})
            else:
                if isinstance(p, (str, Path)):
                    p = Image.open(p)
                content.append({"type": "image", "image": p})
  
        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        if stream:
            # Return a *live* iterator of tokens instead of blocking until completion.
            streamer = TextIteratorStreamer(self.processor.tokenizer, skip_special_tokens=True)

            def _run():
                self.model.generate(
                    **inputs, 
                    max_new_tokens=max_tokens, 
                    streamer=streamer, 
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )

            import threading

            t = threading.Thread(target=_run, daemon=True, name="SmolVLM2Generate")
            t.start()
            # Attach the thread reference to the streamer for later cleanup
            streamer._thread = t

            # The caller can iterate over *streamer* directly to receive tokens.
            return streamer

        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs, 
                max_new_tokens=max_tokens, 
                do_sample=True,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )

        return self.processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()

    # ------------------------------------------------------------------
    # CLI helper
    # ------------------------------------------------------------------
    def _cli(self) -> None:  # pragma: no cover
        img = Path("test_img.png")
        parts = [ img, "be very specific, what kind of pants and T-shirt would go well with this jacket?"
        ]
        print("Streaming answer:\n")
        self.chat(parts=parts, stream=True)

    # ------------------------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Placeholder for API consistency.

        SmolVLM2 initialises its weights during __init__, so additional work
        is not required. The *RAGPipeline* nevertheless expects every model
        to expose a *load()* method – we provide a simple pass-through here
        to avoid AttributeError.
        """

        # Nothing to do: model is already loaded during __init__.
        return

    # ------------------------------------------------------------------
    # Text-only convenience wrapper (RAG compatibility)
    # ------------------------------------------------------------------
    def generate(
        self,
        *,
        prompt: str,
        context: str = "",
        max_tokens: int = 2048,
        stream: bool = False,
    ):
        """Generate an answer for *prompt* given *context*.

        This is a thin wrapper around :py:meth:`chat` so the *RAGPipeline*
        can treat `SmolVLM2` like a normal text-only LLM.
        """

        full_prompt = (
            ("Context:\n" + context + "\n\n" if context else "")
            + "Question: "
            + prompt
        )

        # Streamed generation → return iterator of tokens
        if stream:
            token_stream = self.chat(parts=[full_prompt], max_tokens=max_tokens, stream=True)
            # Simple pass-through - let the frontend handle any formatting
            return token_stream

        # Regular blocking generation
        full_response = self.chat(parts=[full_prompt], max_tokens=max_tokens, stream=False)
        
        # Clean up the response by removing any echoed prompt content
        answer = full_response.strip()
        
        # Remove common prompt echoes that might appear at the start
        prefixes_to_remove = [
            "User: Context:",
            "Assistant: User:",
            "Context:",
            "Question:",
            "Please answer:",
            "Based on the following information:",
            "Previous conversation:"
        ]
        
        for prefix in prefixes_to_remove:
            if answer.startswith(prefix):
                # Find the end of this section and remove it
                lines = answer.split('\n')
                # Skip lines that start with prompt-like content
                clean_lines = []
                skip_mode = True
                for line in lines:
                    line_stripped = line.strip()
                    # If we find a line that looks like actual content (not prompt echo)
                    if skip_mode and line_stripped and not any(line_stripped.startswith(p) for p in prefixes_to_remove):
                        skip_mode = False
                    if not skip_mode:
                        clean_lines.append(line)
                answer = '\n'.join(clean_lines).strip()
                break
                 
        return answer

    # Reduce default max_tokens to 128 to avoid excessive memory footprint on
    # resource-constrained macOS setups.


if __name__ == "__main__":  # pragma: no cover
    SmolVLM2(size="500m")._cli() 