import os
from unittest.mock import MagicMock, patch
from PIL import Image
import pytest
import torch

import config
from model.vision import VisionModel
from tools.screen import (
    ScreenTool,
    capture_screen_image,
    preprocess_screenshot,
)
from agent import Agent


def test_screen_tool_initialization():
    """Requirement 1: Test screen capture tool initialization and schemas."""
    tool = ScreenTool()
    assert tool.name == "screen.capture"
    assert "screen" in tool.aliases
    assert "screen_capture" in tool.aliases
    schema = tool.to_tool_schema()
    assert schema["function"]["name"] == "screen.capture"
    assert "display" in schema["function"]["parameters"]["properties"]
    assert "query" in schema["function"]["parameters"]["properties"]


def test_successful_screenshot_capture():
    """Requirement 2: Test live screenshot capture returns a valid PIL Image."""
    success, img, err = capture_screen_image()
    assert success is True, f"Capture failed: {err}"
    assert isinstance(img, Image.Image)
    assert img.width > 0
    assert img.height > 0
    # Clean up image
    del img


def test_correct_screenshot_dimensions():
    """Requirement 3: Test screenshot preprocessing preserves aspect ratio and bounds."""
    # Test 1: Image already within bounds should not be resized
    small_img = Image.new("RGB", (800, 600), color="blue")
    processed_small = preprocess_screenshot(small_img, max_width=1280, max_height=720)
    assert processed_small.size == (800, 600)

    # Test 2: 1920x1080 image should be resized down to 1280x720 preserving 16:9 aspect ratio
    hd_img = Image.new("RGB", (1920, 1080), color="red")
    processed_hd = preprocess_screenshot(hd_img, max_width=1280, max_height=720)
    assert processed_hd.size == (1280, 720)

    # Test 3: Ultra-wide image 3840x1080
    wide_img = Image.new("RGB", (3840, 1080), color="green")
    processed_wide = preprocess_screenshot(wide_img, max_width=1280, max_height=720)
    assert processed_wide.width == 1280
    assert processed_wide.height == 360  # 1280 / (3840/1080) = 360

    # Test 4: Format conversion to RGB
    rgba_img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
    processed_rgba = preprocess_screenshot(rgba_img)
    assert processed_rgba.mode == "RGB"


def test_vision_model_initialization():
    """Requirement 4: Test vision model initialization without premature loading."""
    vision = VisionModel(model_name="Qwen/Qwen2-VL-2B-Instruct", load_in_4bit=True)
    assert vision.model_name == "Qwen/Qwen2-VL-2B-Instruct"
    assert vision.load_in_4bit is True
    # Verify lazy loading: model is not yet placed in VRAM until describe_screen is called
    assert vision._is_loaded is False
    assert vision.model is None
    assert vision.processor is None

    stats = vision.get_memory_stats()
    assert "total_mb" in stats
    assert "free_mb" in stats


def test_simple_screenshot_description_mock():
    """Requirement 5: Test simple screenshot description through ScreenTool with mock vision model."""
    mock_vision = MagicMock()
    mock_vision.describe_screen.return_value = (
        "Visible on screen: Antigravity IDE code editor with test files open and no visible errors."
    )

    test_img = Image.new("RGB", (1920, 1080), color="black")
    with patch("tools.screen.capture_screen_image", return_value=(True, test_img, None)):
        tool = ScreenTool(vision_model=mock_vision)
        result = tool.execute(display="primary", query="What is on my screen?")

        assert result["success"] is True
        assert result["display"] == "primary"
        assert result["width"] == 1920
        assert result["height"] == 1080
        assert "Antigravity IDE" in result["description"]
        mock_vision.describe_screen.assert_called_once()


def test_error_handling_when_display_capture_fails():
    """Requirement 6: Test graceful error handling when screen capture fails."""
    with patch("tools.screen.capture_screen_image", return_value=(False, None, "Display capture denied")):
        tool = ScreenTool()
        result = tool.execute(display="primary")

        assert result["success"] is False
        assert "Display capture denied" in result["error"]
        assert result["width"] == 0
        assert result["height"] == 0


def test_error_handling_when_vision_inference_fails():
    """Test error handling when vision inference raises an exception."""
    mock_vision = MagicMock()
    mock_vision.describe_screen.side_effect = RuntimeError("Out of memory in vision forward pass")

    test_img = Image.new("RGB", (100, 100), color="gray")
    with patch("tools.screen.capture_screen_image", return_value=(True, test_img, None)):
        tool = ScreenTool(vision_model=mock_vision)
        result = tool.execute()

        assert result["success"] is False
        assert "Vision model inference error" in result["error"]
        assert "Out of memory" in result["error"]


def test_gpu_memory_cleanup():
    """Requirement 7: Test GPU memory tracking and clean unloading."""
    vision = VisionModel()
    stats_before = vision.get_memory_stats()
    assert isinstance(stats_before["free_mb"], (int, float))

    # Unload should cleanly execute even if not loaded
    vision.unload()
    assert vision._is_loaded is False
    assert vision.model is None


def test_agent_integration_with_screen_tool():
    """Test end-to-end agent interaction with ScreenTool."""
    mock_model = MagicMock()
    # First response: call screen.capture tool
    mock_model.generate_response.side_effect = [
        '<tool_call>{"name": "screen.capture", "arguments": {"query": "What is on screen?"}}</tool_call>',
        'Based on the screen capture, you have the Antigravity IDE open.'
    ]

    mock_vision = MagicMock()
    mock_vision.describe_screen.return_value = "Screen shows Antigravity IDE editor."

    test_img = Image.new("RGB", (1920, 1080), color="blue")
    with patch("tools.screen.capture_screen_image", return_value=(True, test_img, None)):
        screen_tool = ScreenTool(vision_model=mock_vision)
        agent = Agent(model=mock_model)
        agent.register_tool(screen_tool)

        response = agent.run("What is currently on my screen?")
        assert "Antigravity IDE" in response
        assert len(agent.messages) >= 3
        # Ensure tool call message was recorded
        assert any(m.get("role") == "tool" and m.get("name") == "screen.capture" for m in agent.messages)
