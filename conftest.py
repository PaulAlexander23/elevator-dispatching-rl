import pytest

import env


@pytest.fixture(autouse=True)
def no_render_sleep(monkeypatch):
    """Human rendering sleeps 1s per frame; skip that in tests."""
    monkeypatch.setattr(env, "sleep", lambda seconds: None)
