import importlib
import os
from unittest.mock import MagicMock, patch
import pytest
import torch

import config
from model.qwen import QwenModel


def test_default_context_configuration():
    """Verify that default context length is 8192 and generation limit is 512."""
    assert config.MAX_CONTEXT_LENGTH == 8192
    assert config.MAX_NEW_TOKENS == 512
    max_prompt_tokens = config.MAX_CONTEXT_LENGTH - config.MAX_NEW_TOKENS
    assert max_prompt_tokens == 7680


def test_environment_variable_overrides():
    """Verify that ZIA_MAX_CONTEXT_LENGTH and ZIA_MAX_NEW_TOKENS can override defaults."""
    try:
        with patch.dict(os.environ, {"ZIA_MAX_CONTEXT_LENGTH": "16384", "ZIA_MAX_NEW_TOKENS": "1024"}):
            importlib.reload(config)
            assert config.MAX_CONTEXT_LENGTH == 16384
            assert config.MAX_NEW_TOKENS == 1024
    finally:
        # Restore original config outside of patch.dict
        importlib.reload(config)


def test_qwen_context_headroom_and_truncation(capsys):
    """
    Verify prompt truncation behavior with 8192 context:
    1. A prompt with > 3840 tokens (e.g. 5000 tokens) must NOT be truncated.
    2. A prompt with > 7680 tokens (e.g. 8000 tokens) MUST be truncated to 7680 tokens.
    """
    # Create QwenModel without calling __init__
    qwen = object.__new__(QwenModel)
    qwen.model_name = "test-model"
    qwen.device = torch.device("cpu")
    qwen.max_context_length = config.MAX_CONTEXT_LENGTH  # 8192

    # Mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "dummy prompt text"
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 0
    mock_tokenizer.decode.return_value = "response text"
    qwen.tokenizer = mock_tokenizer

    # Mock inner torch model
    mock_inner_model = MagicMock()
    # Output includes prompt + 1 generated token
    mock_inner_model.generate.return_value = torch.zeros((1, 10), dtype=torch.long)
    qwen.model = mock_inner_model

    # Case 1: Prompt longer than old limit (3840), e.g. 5000 tokens
    token_count = 5000
    mock_input_ids = torch.zeros((1, token_count), dtype=torch.long)
    mock_attention = torch.ones((1, token_count), dtype=torch.long)
    mock_tokenizer.return_value = {"input_ids": mock_input_ids, "attention_mask": mock_attention}

    capsys.readouterr()  # Clear stdout
    _ = qwen.generate_response(messages=[{"role": "user", "content": "test"}])
    captured = capsys.readouterr()

    # Must NOT warn or truncate at 3840
    assert "[ZIA WARNING] Input tokens" not in captured.out
    # Check tensor shape passed into model.generate
    call_kwargs = mock_inner_model.generate.call_args.kwargs
    assert call_kwargs["input_ids"].shape[1] == 5000

    # Case 2: Prompt exceeding new headroom (7680), e.g. 8000 tokens
    token_count_large = 8000
    mock_input_ids_large = torch.zeros((1, token_count_large), dtype=torch.long)
    mock_attention_large = torch.ones((1, token_count_large), dtype=torch.long)
    mock_tokenizer.return_value = {"input_ids": mock_input_ids_large, "attention_mask": mock_attention_large}

    _ = qwen.generate_response(messages=[{"role": "user", "content": "test"}])
    captured = capsys.readouterr()

    # Must log warning showing truncation to fit within 8192 max context
    assert "[ZIA WARNING] Input tokens (8000) exceed prompt limit (7680). Truncating to fit within 8192 max context." in captured.out
    # Verify input_ids passed into model.generate is truncated to 7680
    call_kwargs_large = mock_inner_model.generate.call_args.kwargs
    assert call_kwargs_large["input_ids"].shape[1] == 7680
