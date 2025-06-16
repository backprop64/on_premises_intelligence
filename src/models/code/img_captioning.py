from pathlib import Path
from typing import Union, Literal
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText


MODEL_MAP = {
    "2.2b": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "500m": "HuggingFaceTB/SmolVLM2-500M-Instruct",
    "256m": "HuggingFaceTB/SmolVLM2-256M-Instruct",
}


class ImageCaptioner:
    """Caption images with SmolVLM2.

    The strategy is simple: we embed the image bytes under a short prompt and
    ask the model to return a richly-detailed caption suitable for retrieval.
    The implementation keeps things lightweight by converting the image to a
    base64 string before passing it to the model — this is a *placeholder*
    until proper VLM image embedding is wired in.
    """

    def __init__(
        self,
        *,
        device: str = "cpu",
        size: Literal["2.2b", "500m", "256m"] = "2.2b",
    ) -> None:
        self.device = device
        self.model_name = MODEL_MAP[size.lower()]
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        ).to(device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def caption(
        self,
        image: Union[str, Path, Image.Image],
        *,
        prompt: str = "Describe the following image in exhaustive detail for retrieval:",
        max_tokens: int = 512,
        stream: bool = False,
    ) -> str:
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        # Build chat-like messages expected by SmolVLM2 template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        if stream:
            from transformers import TextIteratorStreamer
            streamer = TextIteratorStreamer(self.processor.tokenizer, skip_special_tokens=True)

            # Run generation in background to allow real-time printing
            import threading

            def _gen():
                self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    streamer=streamer,
                )

            threading.Thread(target=_gen, daemon=True).start()

            partial = ""
            for token in streamer:
                print(token, end="", flush=True)
                partial += token
            print()
            return partial.strip()
        else:
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)

            caption = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            return caption.strip()


# ------------------------------------------------------------------
# CLI sanity check
# ------------------------------------------------------------------


def _cli(img_path: str | Path, size: str = "2.2b") -> None:  # pragma: no cover
    cap = ImageCaptioner(size=size)
    print("Streaming caption:")
    cap.caption(img_path, stream=True)


if __name__ == "__main__":  # pragma: no cover
    import sys

    img = sys.argv[1] if len(sys.argv) > 1 else "test_img.png"
    size = sys.argv[2] if len(sys.argv) > 2 else "2.2b"
    _cli(img, size) 