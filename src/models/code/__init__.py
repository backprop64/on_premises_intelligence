"""Wrappers around third-party model libraries.

Each wrapper exposes a minimal common interface so the rest of the
codebase can stay framework-agnostic.
"""

from .vllm import SmolVLM2
from .text_embedding import TextEmbedder
from .img_captioning import ImageCaptioner

__all__: list[str] = [
    "SmolVLM2",
    "TextEmbedder",
    "ImageCaptioner",
] 