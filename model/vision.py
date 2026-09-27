import gc
from typing import Dict, Any, Optional
import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

import config


class VisionModel:
    """
    Lightweight Vision-Language Model abstraction for local screenshot observation.
    Uses Qwen2-VL-2B-Instruct quantized in 4-bit NF4 to operate within RTX 3050 6GB VRAM bounds.
    Includes strict memory management, tensor cleanup, and cache clearing.
    """

    def __init__(
        self,
        model_name: str = config.VISION_MODEL_NAME,
        load_in_4bit: bool = config.VISION_LOAD_IN_4BIT,
        device_map: str = "cuda:0" if torch.cuda.is_available() else "cpu",
        max_new_tokens: int = config.VISION_MAX_NEW_TOKENS,
        temperature: float = config.VISION_TEMPERATURE,
    ):
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._is_loaded = False
        self.model: Optional[Qwen2VLForConditionalGeneration] = None
        self.processor: Optional[AutoProcessor] = None

    def _ensure_loaded(self) -> None:
        """Lazily load the vision model only when observation is requested."""
        if self._is_loaded and self.model is not None and self.processor is not None:
            return

        self._inspect_memory("Pre-load")
        print(f"[ZIA VISION] Loading processor for {self.model_name}...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        load_kwargs: Dict[str, Any] = {
            "device_map": self.device_map,
            "trust_remote_code": True,
        }

        if self.load_in_4bit and torch.cuda.is_available():
            compute_dtype = torch.float16
            print(f"[ZIA VISION] Configuring 4-bit NF4 quantization for {self.model_name}...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            load_kwargs["quantization_config"] = quantization_config
        elif torch.cuda.is_available():
            load_kwargs["torch_dtype"] = torch.float16
        else:
            load_kwargs["torch_dtype"] = torch.float32

        print(f"[ZIA VISION] Loading vision model {self.model_name} onto {self.device_map}...")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            **load_kwargs,
        )
        self._is_loaded = True
        self._inspect_memory("Post-load")

    def _inspect_memory(self, phase: str) -> None:
        if torch.cuda.is_available():
            stats = self.get_memory_stats()
            print(
                f"[ZIA VISION GPU] {phase} Memory: "
                f"Allocated = {stats['allocated_mb']:.1f} MB, "
                f"Reserved = {stats['reserved_mb']:.1f} MB, "
                f"Free = {stats['free_mb']:.1f} MB / Total = {stats['total_mb']:.1f} MB"
            )

    def get_memory_stats(self) -> Dict[str, float]:
        """Return current GPU memory statistics in MB."""
        if not torch.cuda.is_available():
            return {"total_mb": 0.0, "allocated_mb": 0.0, "reserved_mb": 0.0, "free_mb": 0.0}
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        return {
            "total_mb": total_bytes / (1024**2),
            "allocated_mb": torch.cuda.memory_allocated(0) / (1024**2),
            "reserved_mb": torch.cuda.memory_reserved(0) / (1024**2),
            "free_mb": free_bytes / (1024**2),
        }

    def describe_screen(
        self,
        image: Image.Image,
        query: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """
        Analyze screenshot and return natural language description of visible UI, windows,
        terminal text, or errors.
        Ensures strict GPU tensor deallocation and cache clearing after inference.
        """
        self._ensure_loaded()
        assert self.model is not None and self.processor is not None

        # Build prompt focused on screen observation
        if query and query.strip():
            user_instruction = (
                f"Examine this screenshot carefully and address this specific query: {query.strip()}\n"
                "Also identify the active application/window, visible error messages, and main UI layout."
            )
        else:
            user_instruction = (
                "Describe what is currently visible on this computer screen.\n"
                "Specifically identify:\n"
                "1. The active or foreground window and application name.\n"
                "2. Visible terminal output or commands if any.\n"
                "3. Any visible error messages, dialogs, or warnings.\n"
                "4. Key UI elements or content currently displayed."
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_instruction},
                ],
            }
        ]

        # Prepare multimodal inputs via processor
        text_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = None, None
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
        except Exception:
            # Fallback if qwen_vl_utils is unavailable
            image_inputs = [image]

        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        device = next(self.model.parameters()).device
        inputs = inputs.to(device)

        gen_tokens = max_new_tokens or self.max_new_tokens

        self._inspect_memory("Pre-inference")

        try:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=gen_tokens,
                    temperature=self.temperature if self.temperature > 0 else None,
                    do_sample=True if self.temperature > 0 else False,
                )

            # Extract generated tokens (excluding prompt)
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            return output_text

        finally:
            # Explicit tensor deallocation and GPU memory cleanup
            del inputs
            if "generated_ids" in locals():
                del generated_ids
            if "generated_ids_trimmed" in locals():
                del generated_ids_trimmed
            if image_inputs is not None:
                del image_inputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._inspect_memory("Post-inference")

    def unload(self) -> None:
        """Completely release model and tokenizer resources from VRAM."""
        print("[ZIA VISION] Unloading vision model and clearing GPU cache...")
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._is_loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[ZIA VISION] Unload complete.")


_GLOBAL_VISION_MODEL: Optional[VisionModel] = None


def get_vision_model() -> VisionModel:
    """Singleton getter for shared VisionModel instance."""
    global _GLOBAL_VISION_MODEL
    if _GLOBAL_VISION_MODEL is None:
        _GLOBAL_VISION_MODEL = VisionModel()
    return _GLOBAL_VISION_MODEL
