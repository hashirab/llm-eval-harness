import io
import json
import urllib.error

import pytest

from evalharness import providers
from evalharness.providers import MockProvider, ProviderError, _post_json
from evalharness.runner import load_config, run

from .test_pipeline import MOCK_CONFIG


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_error(code):
    return urllib.error.HTTPError("https://x", code, "err", {}, io.BytesIO(b'{"error": "nope"}'))


@pytest.fixture
def no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(providers, "_sleep", sleeps.append)
    return sleeps


def _script(monkeypatch, outcomes):
    """Make urlopen return or raise each item in `outcomes` in turn."""
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(req)
        outcome = outcomes[len(calls) - 1]
        if isinstance(outcome, BaseException):
            raise outcome
        return FakeResponse(json.dumps(outcome).encode())

    monkeypatch.setattr(providers.urllib.request, "urlopen", fake_urlopen)
    return calls


def test_timeout_is_retried_then_succeeds(monkeypatch, no_sleep):
    calls = _script(monkeypatch, [TimeoutError("read timed out"), {"ok": True}])
    assert _post_json("https://x", {}) == {"ok": True}
    assert len(calls) == 2
    assert no_sleep == [2.0]


def test_rate_limit_and_server_errors_are_retried_with_backoff(monkeypatch, no_sleep):
    calls = _script(monkeypatch, [_http_error(429), _http_error(503), {"ok": True}])
    assert _post_json("https://x", {}) == {"ok": True}
    assert len(calls) == 3
    assert no_sleep == [2.0, 4.0]


def test_client_errors_are_not_retried(monkeypatch, no_sleep):
    calls = _script(monkeypatch, [_http_error(404)])
    with pytest.raises(ProviderError, match="HTTP 404"):
        _post_json("https://x", {})
    assert len(calls) == 1 and no_sleep == []


def test_persistent_timeout_becomes_provider_error(monkeypatch, no_sleep):
    _script(monkeypatch, [TimeoutError("t")] * 3)
    with pytest.raises(ProviderError, match="after 3 attempts"):
        _post_json("https://x", {})


def test_unexpected_exception_is_recorded_and_run_continues():
    calls = {"n": 0}

    def flaky(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("read timed out")
        return "fine"

    config = load_config(MOCK_CONFIG)
    data = run(config, providers=[MockProvider("flaky", responder=flaky)], limit=2, progress=False)
    errors = [r["error"] for r in data["results"]]
    assert errors[0].startswith("Unexpected TimeoutError")
    assert all(e is None for e in errors[1:])
