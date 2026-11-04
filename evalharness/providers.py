"""Model providers.

Each provider takes a prompt and returns a Completion with the text and how long it took.
Only the standard library is used for HTTP, so the harness has a single dependency (PyYAML).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable


@dataclass
class Completion:
    text: str
    latency_ms: float


class ProviderError(RuntimeError):
    pass


def _post_json(url: str, payload: dict, headers: dict | None = None, timeout: float = 120.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise ProviderError(f"HTTP {exc.code} from {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise ProviderError(f"Could not reach {url}: {exc.reason}") from exc


class Provider:
    """Base class. Subclasses implement _generate and return the raw text."""

    def __init__(self, model: str, temperature: float = 0.0, **options):
        self.model = model
        self.temperature = temperature
        self.options = options

    @property
    def name(self) -> str:
        return f"{self.__class__.__name__.replace('Provider', '').lower()}:{self.model}"

    def complete(self, prompt: str) -> Completion:
        start = time.perf_counter()
        text = self._generate(prompt)
        latency_ms = (time.perf_counter() - start) * 1000
        return Completion(text=text.strip(), latency_ms=latency_ms)

    def _generate(self, prompt: str) -> str:  # pragma: no cover - abstract
        raise NotImplementedError


class OllamaProvider(Provider):
    """Local models served by Ollama (https://ollama.com)."""

    def _generate(self, prompt: str) -> str:
        host = self.options.get("host") or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        return _post_json(f"{host.rstrip('/')}/api/generate", payload)["response"]


class GeminiProvider(Provider):
    """Google Gemini via the Generative Language REST API. Needs GEMINI_API_KEY."""

    def _generate(self, prompt: str) -> str:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ProviderError("GEMINI_API_KEY is not set")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": self.temperature},
        }
        data = _post_json(url, payload, headers={"x-goog-api-key": key})
        try:
            return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError) as exc:
            raise ProviderError(f"Unexpected Gemini response: {str(data)[:300]}") from exc


class OpenAIProvider(Provider):
    """Any OpenAI-compatible chat completions endpoint. Needs OPENAI_API_KEY."""

    def _generate(self, prompt: str) -> str:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ProviderError("OPENAI_API_KEY is not set")
        base = self.options.get("base_url") or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        data = _post_json(f"{base.rstrip('/')}/chat/completions", payload, headers={"Authorization": f"Bearer {key}"})
        return data["choices"][0]["message"]["content"]


class MockProvider(Provider):
    """Deterministic provider for tests and dry runs.

    By default it echoes the first `echo_chars` characters of the prompt's incident text,
    which gives realistic-looking scores without any network access. Pass `responder` to
    control the output exactly.
    """

    def __init__(self, model: str = "mock", temperature: float = 0.0,
                 responder: Callable[[str], str] | None = None, **options):
        super().__init__(model, temperature, **options)
        self.responder = responder

    def _generate(self, prompt: str) -> str:
        if self.responder is not None:
            return self.responder(prompt)
        marker = "INCIDENT:"
        body = prompt.split(marker, 1)[1] if marker in prompt else prompt
        return body.strip()[: int(self.options.get("echo_chars", 400))]


PROVIDERS = {
    "ollama": OllamaProvider,
    "gemini": GeminiProvider,
    "openai": OpenAIProvider,
    "mock": MockProvider,
}


def build_provider(spec: dict) -> Provider:
    """Build a provider from a config entry like {"provider": "ollama", "model": "llama3.1:8b"}."""
    spec = dict(spec)
    kind = spec.pop("provider")
    if kind not in PROVIDERS:
        raise ValueError(f"Unknown provider '{kind}'. Choose from: {', '.join(PROVIDERS)}")
    model = spec.pop("model", "mock")
    temperature = spec.pop("temperature", 0.0)
    return PROVIDERS[kind](model=model, temperature=temperature, **spec)
