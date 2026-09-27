import gc
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import config


class QwenModel:
    """
    Optimized wrapper for Qwen/Qwen3-4B model running locally with 4-bit NF4 quantization
    on NVIDIA RTX 3050 (6GB VRAM) via BitsAndBytes.
    """

    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        device_map: str = config.DEVICE_MAP,
        torch_dtype: str = config.TORCH_DTYPE,
        load_in_4bit: bool = config.LOAD_IN_4BIT,
        bnb_4bit_quant_type: str = config.BNB_4BIT_QUANT_TYPE,
        bnb_4bit_use_double_quant: bool = config.BNB_4BIT_USE_DOUBLE_QUANT,
        bnb_4bit_compute_dtype: str = config.BNB_4BIT_COMPUTE_DTYPE,
        max_context_length: int = config.MAX_CONTEXT_LENGTH,
    ):
        self.model_name = model_name
        self.device_map = device_map
        self.max_context_length = max_context_length
        self.load_in_4bit = load_in_4bit

        # Inspect pre-load GPU memory
        self._inspect_pre_load_memory()

        # Tokenizer setup
        print(f"[ZIA] Loading tokenizer for {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Model loading with 4-bit quantization or fallback
        load_kwargs: Dict[str, Any] = {
            "device_map": self.device_map,
            "trust_remote_code": True,
        }

        if self.load_in_4bit:
            compute_dtype = (
                torch.float16
                if bnb_4bit_compute_dtype == "float16"
                else (torch.bfloat16 if bnb_4bit_compute_dtype == "bfloat16" else torch.float32)
            )
            print(
                f"[ZIA] Configuring 4-bit NF4 quantization (quant_type={bnb_4bit_quant_type}, "
                f"double_quant={bnb_4bit_use_double_quant}, compute_dtype={compute_dtype})..."
            )
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            load_kwargs["quantization_config"] = quantization_config
        else:
            dtype = torch.float16 if torch_dtype == "float16" else torch.bfloat16
            load_kwargs["torch_dtype"] = dtype

        print(f"[ZIA] Loading model {self.model_name} onto {self.device_map}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **load_kwargs,
        )

        self.device = next(self.model.parameters()).device
        self._verify_device_placement()
        self._report_post_load_memory()

    def _inspect_pre_load_memory(self) -> None:
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            free_mb = free_bytes / (1024**2)
            total_mb = total_bytes / (1024**2)
            print(f"[ZIA GPU] Pre-load GPU Memory: Free = {free_mb:.1f} MB / Total = {total_mb:.1f} MB")
        else:
            print("[ZIA GPU WARNING] CUDA is not available! Model will run on CPU.")

    def _verify_device_placement(self) -> None:
        if hasattr(self.model, "hf_device_map"):
            offloaded = {k: v for k, v in self.model.hf_device_map.items() if v in ("cpu", "disk")}
            if offloaded:
                print(f"[ZIA WARNING] CPU/disk offloading detected: {offloaded}")
            else:
                print(f"[ZIA GPU] All model layers successfully placed on GPU: {self.device_map}")
        else:
            print(f"[ZIA GPU] Model loaded on primary device: {self.device}")

    def _report_post_load_memory(self) -> None:
        if torch.cuda.is_available():
            stats = self.get_memory_stats()
            print(f"[ZIA GPU] Post-load GPU Memory:")
            print(f"  • Total VRAM:     {stats['total_mb']:.1f} MB")
            print(f"  • Allocated VRAM: {stats['allocated_mb']:.1f} MB")
            print(f"  • Reserved VRAM:  {stats['reserved_mb']:.1f} MB")
            print(f"  • Free VRAM:      {stats['free_mb']:.1f} MB")

    def get_memory_stats(self) -> Dict[str, float]:
        """Return current GPU memory allocation stats in MB."""
        if not torch.cuda.is_available():
            return {"total_mb": 0.0, "allocated_mb": 0.0, "reserved_mb": 0.0, "free_mb": 0.0}
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        return {
            "total_mb": total_bytes / (1024**2),
            "allocated_mb": torch.cuda.memory_allocated(0) / (1024**2),
            "reserved_mb": torch.cuda.memory_reserved(0) / (1024**2),
            "free_mb": free_bytes / (1024**2),
        }

    def generate_response(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int = config.MAX_NEW_TOKENS,
        temperature: float = config.TEMPERATURE,
        top_p: float = config.TOP_P,
        max_context_length: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        enable_thinking: bool = config.ENABLE_THINKING,
    ) -> str:
        """
        Generate text response given standard chat messages format.
        Supports structured tool schemas via tokenizer chat template.
        Ensures context length bounds are strictly enforced to preserve GPU VRAM.
        """
        ctx_limit = max_context_length or self.max_context_length

        template_kwargs: Dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if tools:
            template_kwargs["tools"] = tools

        # Apply enable_thinking if supported by template
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                enable_thinking=enable_thinking,
                **template_kwargs,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages,
                **template_kwargs,
            )

        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"]

        # Enforce context length: leave headroom for max_new_tokens
        max_prompt_tokens = max(1, ctx_limit - max_new_tokens)
        if input_ids.shape[1] > max_prompt_tokens:
            print(
                f"[ZIA WARNING] Input tokens ({input_ids.shape[1]}) exceed prompt limit "
                f"({max_prompt_tokens}). Truncating to fit within {ctx_limit} max context."
            )
            input_ids = input_ids[:, -max_prompt_tokens:]
            attention_mask = inputs["attention_mask"][:, -max_prompt_tokens:]
            inputs = {"input_ids": input_ids.to(self.device), "attention_mask": attention_mask.to(self.device)}
        else:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Slice off prompt tokens
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response.strip()

    def unload(self) -> None:
        """Cleanly releases model resources from GPU VRAM."""
        print("[ZIA] Unloading model and freeing GPU cache...")
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[ZIA] Unload complete.")
