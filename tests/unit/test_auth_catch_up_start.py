"""The auth catch-up starts only when its switch is on and Supabase is configured."""
from __future__ import annotations

import pytest

from app.services import auth_data_migration as m


@pytest.fixture
def started(monkeypatch):
    calls = []
    monkeypatch.setattr(m.threading, "Thread", lambda **kw: type("T", (), {"start": lambda self: calls.append(kw["name"])})())
    monkeypatch.setattr(m.settings, "auth_supabase_url", "https://example.supabase.co")
    monkeypatch.setattr(m.settings, "auth_supabase_service_key", "key")
    monkeypatch.setattr(m.settings, "auth_supabase_catchup", True)
    return calls


def test_starts_when_on_and_configured(started):
    m.start_catch_up_in_background()
    assert started == ["auth-catch-up"]


def test_does_nothing_when_switched_off(started, monkeypatch):
    monkeypatch.setattr(m.settings, "auth_supabase_catchup", False)
    m.start_catch_up_in_background()
    assert started == []


def test_does_nothing_without_supabase_credentials(started, monkeypatch):
    monkeypatch.setattr(m.settings, "auth_supabase_service_key", "")
    m.start_catch_up_in_background()
    assert started == []
