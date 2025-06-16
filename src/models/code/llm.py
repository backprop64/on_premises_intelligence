from typing import Literal
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import threading


MODEL_MAP = {
    "1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "360m": "HuggingFaceTB/SmolLM2-360M-Instruct", 
    "135m": "HuggingFaceTB/SmolLM2-135M-Instruct",
}


class SmolLM2:
    """Text-only chat wrapper for SmolLM2 models."""

    def __init__(
        self,
        *,
        device: str = "cpu",
        size: Literal["1.7b", "360m", "135m"] = "360m",
    ) -> None:
        self.device = device
        self.model_name = MODEL_MAP[size]
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        """Load the model and tokenizer."""
        print(f"[SmolLM2] Loading {self.model_name} on {self.device}")
        
        # Load tokenizer with proper context length (SmolLM2 supports 8k tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            model_max_length=8192  # Set to 8k context length
        )
        
        # Explicitly override the tokenizer's model_max_length to ensure it's used
        self.tokenizer.model_max_length = 8192
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        
        # Verify the configuration
        actual_max_length = getattr(self.model.config, 'max_position_embeddings', 'unknown')
        print(f"[SmolLM2] Model loaded successfully with {self.tokenizer.model_max_length} token context")
        print(f"[SmolLM2] Model config max_position_embeddings: {actual_max_length}")
        
        # If the model config shows 512, we need to update it
        if hasattr(self.model.config, 'max_position_embeddings') and self.model.config.max_position_embeddings < 8192:
            print(f"[SmolLM2] WARNING: Model config shows {self.model.config.max_position_embeddings} tokens, but SmolLM2 supports 8192. Updating config...")
            self.model.config.max_position_embeddings = 8192

    def chat(
        self,
        *,
        messages: list[dict],
        max_tokens: int = 1024,
        stream: bool = False,
    ):
        """Chat with the model using proper message format.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Apply chat template
        input_text = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=False
        )
        
        inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        
        # Check if input is too long and truncate if necessary
        max_input_tokens = 8192 - max_tokens  # Reserve space for generation
        if inputs.shape[1] > max_input_tokens:
            print(f"[SmolLM2] WARNING: Input length ({inputs.shape[1]} tokens) exceeds maximum ({max_input_tokens} tokens). Truncating...")
            inputs = inputs[:, -max_input_tokens:]  # Keep the last max_input_tokens

        if stream:
            # Return a streaming iterator
            streamer = TextIteratorStreamer(
                self.tokenizer, 
                skip_special_tokens=True,
                skip_prompt=True  # Only return generated tokens, not the input
            )

            def _run():
                self.model.generate(
                    inputs,
                    max_new_tokens=max_tokens,
                    streamer=streamer,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            thread = threading.Thread(target=_run, daemon=True, name="SmolLM2Generate")
            thread.start()
            # Attach thread reference for cleanup
            streamer._thread = thread
            return streamer

        # Non-streaming generation
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode only the generated part (skip the input)
        generated_tokens = outputs[0][len(inputs[0]):]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response.strip()

    def generate(
        self,
        *,
        prompt: str,
        context: str = "",
        max_tokens: int = 1024,
        stream: bool = False,
    ):
        """Generate a response for a prompt with optional context.
        
        This method builds the conversation history and context into proper
        message format for the chat model.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Build the user message with context if provided
        user_content = prompt
        if context.strip():
            user_content = f"Here's some relevant information:\n\n{context}\n\nQuestion: {prompt}"

        messages = [{"role": "user", "content": user_content}]
        
        return self.chat(messages=messages, max_tokens=max_tokens, stream=stream)

    def chat_with_history(
        self,
        *,
        prompt: str,
        context: str = "",
        history: str = "",
        max_tokens: int = 1024,
        stream: bool = False,
    ):
        """Generate a response with conversation history.
        
        Args:
            prompt: Current user question
            context: Retrieved document context
            history: Previous conversation history in "User: ... Assistant: ..." format
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        messages = []
        
        # Parse history into proper message format
        if history.strip():
            # Split history into turns
            turns = []
            current_turn = ""
            current_role = None
            
            for line in history.split('\n'):
                line = line.strip()
                if line.startswith('User: '):
                    if current_turn and current_role:
                        turns.append({"role": current_role, "content": current_turn.strip()})
                    current_role = "user"
                    current_turn = line[6:]  # Remove "User: "
                elif line.startswith('Assistant: '):
                    if current_turn and current_role:
                        turns.append({"role": current_role, "content": current_turn.strip()})
                    current_role = "assistant"
                    current_turn = line[11:]  # Remove "Assistant: "
                elif current_turn:
                    current_turn += "\n" + line
            
            # Add the last turn
            if current_turn and current_role:
                turns.append({"role": current_role, "content": current_turn.strip()})
            
            messages.extend(turns)

        # Build current user message with context
        user_content = prompt
        if context.strip():
            user_content = f"Here's some relevant information:\n\n{context}\n\nQuestion: {prompt}"
        
        messages.append({"role": "user", "content": user_content})
        
        return self.chat(messages=messages, max_tokens=max_tokens, stream=stream)


if __name__ == "__main__":  # pragma: no cover
    # Simple test
    model = SmolLM2(size="135m", device="cpu")
    model.load()
    
    response = model.generate(prompt="What is the capital of France?")
    print("Response:", response) 