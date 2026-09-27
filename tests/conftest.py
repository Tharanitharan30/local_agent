import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: mark test as requiring the real GPU model."
    )
