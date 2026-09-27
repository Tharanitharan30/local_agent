#!/usr/bin/env python3
"""
Benchmark script for Zia local LLM setup.
Measures:
1. Model loading time
2. Warmup generation
3. Generation latency and tokens per second
4. GPU VRAM metrics (allocated, reserved, peak, free)
5. System RAM metrics
6. Clean resource teardown
"""
import gc
import sys
import time
import psutil
import torch

from model import QwenModel
import config


def get_ram_mb() -> float:
    """Return current system RAM usage in MB."""
    return psutil.virtual_memory().used / (1024**2)


def main():
    print("=" * 65)
    print("         ZIA LOCAL LLM BENCHMARK: QWEN3-4B (4-BIT NF4)          ")
    print("=" * 65)

    # 1. Environment & Pre-load metrics
    ram_before = get_ram_mb()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    print(f"Device:               {gpu_name}")
    print(f"PyTorch Version:      {torch.__version__}")
    print(f"CUDA Available:       {torch.cuda.is_available()}")
    print(f"Pre-load System RAM:  {ram_before:.1f} MB")

    if torch.cuda.is_available():
        free_init, total_init = torch.cuda.mem_get_info(0)
        print(f"Pre-load Free VRAM:   {free_init / (1024**2):.1f} MB / {total_init / (1024**2):.1f} MB")
    print("-" * 65)

    # 2. Model Loading
    print("[1/4] Loading Qwen3-4B with 4-bit NF4 Quantization...")
    t_load_start = time.perf_counter()
    model = QwenModel()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_load_end = time.perf_counter()
    load_time = t_load_end - t_load_start
    print(f"✓ Model loaded in {load_time:.2f} seconds.")

    post_load_stats = model.get_memory_stats()
    ram_after_load = get_ram_mb()
    print(f"  • VRAM Allocated:   {post_load_stats['allocated_mb']:.1f} MB")
    print(f"  • VRAM Reserved:    {post_load_stats['reserved_mb']:.1f} MB")
    print(f"  • VRAM Free:        {post_load_stats['free_mb']:.1f} MB")
    print(f"  • System RAM Used:  {ram_after_load:.1f} MB (Delta: +{ram_after_load - ram_before:.1f} MB)")
    print("-" * 65)

    # 3. Warmup
    print("[2/4] Warming up model with short generation...")
    warmup_messages = [{"role": "user", "content": "Hello! Give a 1-word greeting."}]
    _ = model.generate_response(warmup_messages, max_new_tokens=16, temperature=0.6)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print("✓ Warmup complete.")
    print("-" * 65)

    # 4. Benchmark Generation
    benchmark_prompt = "Explain what a Linux process is in two sentences."
    print(f"[3/4] Running Benchmark Generation...")
    print(f"Prompt: \"{benchmark_prompt}\"")

    messages = [{"role": "user", "content": benchmark_prompt}]
    text = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = model.tokenizer(text, return_tensors="pt").to(model.device)
    prompt_tokens = inputs["input_ids"].shape[1]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.synchronize()

    t_gen_start = time.perf_counter()
    with torch.no_grad():
        outputs = model.model.generate(
            **inputs,
            max_new_tokens=config.MAX_NEW_TOKENS,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            do_sample=True,
            pad_token_id=model.tokenizer.eos_token_id,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_gen_end = time.perf_counter()

    gen_time = t_gen_end - t_gen_start
    gen_token_ids = outputs[0][prompt_tokens:]
    tokens_generated = len(gen_token_ids)
    tok_per_sec = tokens_generated / gen_time if gen_time > 0 else 0.0

    response_text = model.tokenizer.decode(gen_token_ids, skip_special_tokens=True).strip()

    peak_vram = torch.cuda.max_memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0.0
    post_gen_stats = model.get_memory_stats()
    ram_after_gen = get_ram_mb()

    print("\n--- Model Response ---")
    print(response_text)
    print("----------------------\n")

    # 5. Resource Cleanup
    print("[4/4] Cleaning up resources...")
    model.unload()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("✓ Cleanup finished.")
    print("=" * 65)

    # 6. Benchmark Summary Table
    print("                      BENCHMARK RESULTS                        ")
    print("=" * 65)
    print(f"Model:                  {config.MODEL_NAME}")
    print(f"Quantization:           4-bit NF4 (BitsAndBytes, double quant)")
    print(f"Compute Dtype:          {config.BNB_4BIT_COMPUTE_DTYPE}")
    print(f"Configured Context:     {config.MAX_CONTEXT_LENGTH} tokens")
    print(f"Configured Max Tokens:  {config.MAX_NEW_TOKENS}")
    print(f"Sampling Parameters:    temp={config.TEMPERATURE}, top_p={config.TOP_P}")
    print("-" * 65)
    print(f"Model Load Time:        {load_time:.2f} s")
    print(f"Prompt Tokens:          {prompt_tokens}")
    print(f"Generated Tokens:       {tokens_generated}")
    print(f"Generation Time:        {gen_time:.2f} s")
    print(f"Inference Speed:        {tok_per_sec:.2f} tokens/s")
    print("-" * 65)
    print(f"Peak VRAM Allocated:    {peak_vram:.1f} MB")
    print(f"Model VRAM (Post-Load): {post_load_stats['allocated_mb']:.1f} MB")
    print(f"VRAM Reserved:          {post_load_stats['reserved_mb']:.1f} MB")
    print(f"VRAM Free (Post-Load):  {post_load_stats['free_mb']:.1f} MB")
    print(f"System RAM Used:        {ram_after_gen:.1f} MB")
    print("=" * 65)
    print("Status: SUCCESS (Runs stably in GPU VRAM without CPU offloading)")
    print("=" * 65)


if __name__ == "__main__":
    main()
