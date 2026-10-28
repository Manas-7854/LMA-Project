"""Config-driven LLM client with caching and JSON-schema-validated outputs.

Backends:
  hf      local transformers model (CPU smoke tests)
  openai  any OpenAI-compatible server, e.g. vLLM serving Qwen3-8B
  mock    deterministic stub for pipeline tests (no model)
Parse and validation failures are returned in the result, never dropped.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import jsonschema

from ..utils.cache import Cache, cache_key

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class LLMConfig:
    backend: str = "mock"
    model: str = "mock"
    revision: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    seed: int = 13
    extra: dict = field(default_factory=dict)


@dataclass
class LLMResult:
    text: str
    parsed: dict | None
    error: str | None
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cached: bool = False


def extract_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = _JSON_OBJ.search(text)
    if not m:
        raise ValueError("no JSON object in output")
    return json.loads(m.group(0))


def default_mock(messages: list[dict]) -> str:
    """Always 'true', citing the first passage id in the prompt (pipeline tests only)."""
    ids = re.findall(r"^\[([^\]]+)\]$", messages[-1]["content"], flags=re.MULTILINE)
    return json.dumps({"verdict": "true", "cited_passages": ids[:1], "justification": "mock"})


class LLMClient:
    def __init__(self, cfg: LLMConfig, cache: Cache | None = None):
        self.cfg, self.cache = cfg, cache
        self._model = self._tok = None

    # -- backends -----------------------------------------------------------------------
    def _hf(self, messages: list[dict], temperature: float, seed: int) -> tuple[str, int, int]:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if self._model is None:
            torch.manual_seed(self.cfg.seed)
            self._tok = AutoTokenizer.from_pretrained(self.cfg.model, revision=self.cfg.revision)
            self._model = AutoModelForCausalLM.from_pretrained(self.cfg.model, revision=self.cfg.revision,
                                                               dtype=torch.float32)
            self._model.eval()
        ids = self._tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                            return_dict=True, **self.cfg.extra.get("chat_template_kwargs", {}))
        with torch.no_grad():
            torch.manual_seed(seed)
            sampling = {"do_sample": True, "temperature": temperature, "top_p": self.cfg.top_p} if temperature > 0                 else {"do_sample": False}
            out = self._model.generate(**ids, max_new_tokens=self.cfg.max_tokens, **sampling)
        gen = out[0, ids["input_ids"].shape[1]:]
        return self._tok.decode(gen, skip_special_tokens=True), int(ids["input_ids"].shape[1]), int(len(gen))

    def _openai(self, messages: list[dict], temperature: float, seed: int) -> tuple[str, int, int]:
        import httpx
        body = {"model": self.cfg.model, "messages": messages, "temperature": temperature,
                "top_p": self.cfg.top_p, "max_tokens": self.cfg.max_tokens, "seed": seed,
                **self.cfg.extra}
        r = httpx.post(f"{self.cfg.base_url.rstrip('/')}/chat/completions", json=body, timeout=300)
        r.raise_for_status()
        d = r.json()
        u = d.get("usage", {})
        return d["choices"][0]["message"]["content"], u.get("prompt_tokens", 0), u.get("completion_tokens", 0)

    def _mock(self, messages: list[dict], temperature: float, seed: int) -> tuple[str, int, int]:
        return self.cfg.extra.get("mock_fn", default_mock)(messages), 0, 0

    # -- public -------------------------------------------------------------------------
    def generate_json(self, messages: list[dict], schema: dict, prompt_version: str,
                      temperature: float | None = None, sample: int = 0) -> LLMResult:
        """temperature overrides the profile's; sample k uses seed + k (self-consistency draws)."""
        temperature = self.cfg.temperature if temperature is None else temperature
        seed = self.cfg.seed + sample
        key = cache_key(model=self.cfg.model, revision=self.cfg.revision, backend=self.cfg.backend,
                        temperature=temperature, top_p=self.cfg.top_p, max_tokens=self.cfg.max_tokens,
                        seed=seed, prompt_version=prompt_version, messages=messages,
                        extra={k: v for k, v in self.cfg.extra.items() if k != "mock_fn"})
        if self.cache and self.cfg.backend != "mock" and (hit := self.cache.get(key)):
            return LLMResult(**{**hit, "cached": True})
        t0 = time.perf_counter()
        try:
            text, pt, ct = getattr(self, f"_{self.cfg.backend}")(messages, temperature, seed)
        except Exception as e:  # transport/runtime failure: recorded, not raised
            return LLMResult("", None, f"call_error: {type(e).__name__}: {e}", 0, 0, time.perf_counter() - t0)
        latency = time.perf_counter() - t0
        parsed, error = None, None
        try:
            parsed = extract_json(text)
            jsonschema.validate(parsed, schema)
        except (ValueError, json.JSONDecodeError) as e:
            error = f"parse_error: {e}"
        except jsonschema.ValidationError as e:
            error = f"schema_error: {e.message}"
        res = LLMResult(text, parsed if error is None else None, error, pt, ct, latency)
        if self.cache and self.cfg.backend != "mock":
            self.cache.put(key, {k: v for k, v in res.__dict__.items() if k != "cached"})
        return res
