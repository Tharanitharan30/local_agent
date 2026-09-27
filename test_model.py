#!/usr/bin/env python3
"""
Lightweight test script for Zia Qwen3-4B local model.
Loads the model once using QwenModel with 4-bit NF4 quantization on CUDA.
"""
from model import QwenModel


def main():
    print("=" * 60)
    print("ZIA MODEL TEST: Qwen3-4B (4-bit NF4)")
    print("=" * 60)

    # Initialize model wrapper (loads model once)
    model = QwenModel()

    prompt = "Explain what a Linux process is in two sentences."
    print(f"\nPrompt: {prompt}")

    messages = [
        {"role": "user", "content": prompt}
    ]

    print("\nGenerating response...")
    response = model.generate_response(
        messages=messages,
        max_new_tokens=128,
        temperature=0.6,
        top_p=0.8,
    )

    print("\n" + "=" * 60)
    print("Response from Qwen:")
    print("=" * 60)
    print(response)
    print("=" * 60)

    # Clean release of resources
    model.unload()


if __name__ == "__main__":
    main()
