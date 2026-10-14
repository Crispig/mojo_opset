import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic_mode(monkeypatch):
    monkeypatch.setenv("MOJO_DETERMINISTIC", "1")
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)
