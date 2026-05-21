from __future__ import annotations

import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def stable_native_threading() -> None:
    torch.set_num_threads(1)
