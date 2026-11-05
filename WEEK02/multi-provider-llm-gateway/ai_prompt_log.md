# AI Prompt Log — Multi-Provider LLM Gateway

## 1. Tool Used
Claude (model: Opus 5.5) — date: 2026-11-04

## 2. Initial Prompt
You are an expert Python AI Infrastructure Engineer. I need an asynchronous, streaming multi-provider SDK wrapper supporting both OpenAI (e.g., gpt-4o-mini, gpt-4o) and Anthropic (e.g., claude-3-5-sonnet, claude-3-5-haiku).

Requirements:

Environment: Python 3.11+, typed with Pydantic v2.
Architecture:
   - A unified class UnifiedLLMGateway with an async generator method:

     async def stream(self, prompt: str, model: str, temperature: float = 0.7) -> AsyncGenerator[StreamChunk, None]:

Use official async clients (AsyncOpenAI and AsyncAnthropic).
Normalization:
   - Define a Pydantic schema StreamChunk containing:

delta_text: str (the new text token)

provider: str (e.g., "openai" or "anthropic")

model: str

is_final: bool

telemetry: Optional[ChunkTelemetry]

Normalize streaming events so that differences in chunk structure between OpenAI and Anthropic are completely hidden from the consumer.
Token & Cost Telemetry:
   - Include a static pricing dictionary with USD rates per 1,000,000 input and output tokens for supported models.

For OpenAI: enable stream usage tracking (stream_options={"include_usage": True}) and use tiktoken for prompt token verification.

For Anthropic: capture message_start for input tokens and accumulate output tokens from stream events.

On the final chunk (is_final=True), provide exact input tokens, output tokens, total latency (seconds), Time-to-First-Token (TTFT), tokens per second, and calculated total request cost ($USD).

Error Handling:
   - Handle rate limits, invalid API keys, and connection errors by raising typed, custom domain exceptions (e.g., GatewayProviderError, RateLimitExceededError).

Do not leak raw API keys or unhandled stack traces in logs or outputs.

Provide clean, production-ready code split into schemas.py, pricing.py, and gateway.py.

## 3. Raw AI Output (unedited)
I’ll build the three modules, check them against mocked OpenAI and Anthropic streams, and send them over.

I’ve written schemas.py, pricing.py and gateway.py. I tested them against mocked OpenAI and Anthropic streams and error responses run through the real SDKs, not against the live APIs, and every test passed.

📦 What each file holds
File	What’s in it
schemas.py	StreamChunk and ChunkTelemetry (Pydantic v2, read-only). A chunk is rejected unless telemetry appears on exactly the final chunk.
pricing.py	The price table per 1M tokens, including cached-token prices. Costs are calculated with Decimal to avoid rounding errors. Pinned model versions like gpt-4o-2024-08-06 are accepted; unknown models are rejected rather than priced wrong.
gateway.py	UnifiedLLMGateway.stream(), the custom error types and the key redaction.
🔍 Behaviour worth knowing
Telemetry:
TTFT runs from the start of the request to the first text token.
Tokens/sec is output tokens divided by the time from the first token to the end, so connection setup isn’t counted.
Cached tokens are billed at their own (cheaper) rates.
usage_source says whether the counts came from the provider or had to be estimated locally.
tiktoken check: for OpenAI, tiktoken’s prompt count is compared with OpenAI’s and a warning is logged if they differ. Billing always uses OpenAI’s number.
Errors: the custom types are RateLimitExceededError (includes the retry_after wait time), AuthenticationError, ProviderConnectionError / ProviderTimeoutError, ProviderBadRequestError and UnsupportedModelError. All of them inherit from GatewayProviderError or GatewayError.
No leaks: errors are raised from None, so no stack trace comes along. Key patterns are scrubbed from messages, and error text from failed logins is never repeated. The tests confirmed no key showed up in errors, logs or the gateway’s repr.
Early exit: if the caller stops reading partway through, the connection still closes properly.
⚠️ Things I had to work around, and one you should check
Anthropic SDK 2.x no longer has a temperature argument. The Claude 3.5 models still accept it, so I send it with extra_body, which works on both old and new SDK versions.
Both SDKs now use httpx2 instead of httpx. The connection-error handling catches errors from both libraries, so it doesn’t matter which version is installed.
The Claude 3.5 models are being retired by Anthropic. As far as I know, 3.5 Sonnet is already gone and Haiku is on its way out. The gateway will raise ProviderBadRequestError for a model that no longer exists. You’ll probably want to add current Claude models to MODEL_REGISTRY with today’s prices.
tiktoken check: the sandbox blocked tiktoken’s one-time download of its encoding file, so this check didn’t run here. It skipped itself without breaking the request. On a normal machine it downloads once and then works.

To try it: pip install openai anthropic tiktoken pydantic, set OPENAI_API_KEY and ANTHROPIC_API_KEY, then run python gateway.py gpt-4o-mini.

## 4. Audit Findings

### 4.1 Checklist items that passed (verified, no change needed)
| Check | Result | Evidence |
|---|---|---|
| Async event-loop blocking | ✅ Pass | Uses `AsyncOpenAI` / `AsyncAnthropic`; tiktoken runs in `asyncio.to_thread` |
| OpenAI final usage chunk (`choices=[]`, `content=None`) | ✅ Pass | `if not event.choices: continue` and `if text:` guard |
| Anthropic event types | ✅ Pass | Handles `message_start`, `content_block_delta`, `message_delta` |
| TTFT measured at first non-empty delta | ✅ Pass | `first_token_at` set only when text is non-empty |
| Cost precision | ✅ Pass | `Decimal`, quantized to 1e-8 USD (stricter than `round(cost, 6)`) |
| Persistent clients | ✅ Pass | Created once and cached on `self` |
| Secret leakage | ✅ Pass | Regex redaction, `raise ... from None`, `__repr__` hides keys |

### 4.2 Bugs found and fixed
| # | Bug | Severity | How found |
|---|---|---|---|
| B1 | Retired Claude 3.5 model IDs | 🔴 Breaks all Anthropic calls | Deprecations page + `UnsupportedModelError` |
| B2 | Absolute imports break inside `src` package | 🔴 Tests/app cannot import | `import src.gateway` → `ModuleNotFoundError` |
| B3 | `.env` never loaded | 🔴 Keys ignored | Live run → "No OpenAI API key configured" |
| B4 | Field/exception names differ from spec; no malformed-key check | 🟡 Spec mismatch | Compared against assignment Tests 3–4 |
| B5 | `temperature` forced via `extra_body`; Claude 5.x rejects it | 🔴 Breaks all Anthropic calls | Live run → `400 temperature: Extra inputs are not permitted` |

#### B1 — Retired Anthropic model IDs
- **Problem:** Registered `claude-3-5-sonnet-latest` / `claude-3-5-haiku-latest`, retired on 2025-10-28 and 2026-02-19.
- **Fix:** Replaced with `claude-sonnet-5-5` and `claude-haiku-5-5` at current prices (checked 2026-11-04).
  Haiku 5.5's tiered pricing (>100K-token prompts) is documented as a known limitation.
- **Diff:**
```diff
--- a/WEEK02/multi-provider-llm-gateway/src/pricing.py
+++ b/WEEK02/multi-provider-llm-gateway/src/pricing.py
@@ -62,13 +62,17 @@
         ),
         # ---- Anthropic ---------------------------------------------------
-        "claude-3-5-sonnet": ModelSpec(
+        # claude-3-5-sonnet (retired 2025-10-28) and claude-3-5-haiku (retired 2026-02-19)
+        # were replaced with current models. Prices checked 2026-11-04.
+        "claude-sonnet-5-5": ModelSpec(
             provider="anthropic",
-            api_model="claude-3-5-sonnet-latest",
-            pricing=ModelPricing(D("3.00"), D("15.00"), cached_input_per_mtok=D("0.30"), cache_write_per_mtok=D("3.75")),
+            api_model="claude-sonnet-5-5",
+            pricing=ModelPricing(D("2.00"), D("10.00"), cached_input_per_mtok=D("0.20"), cache_write_per_mtok=D("2.50")),
         ),
-        "claude-3-5-haiku": ModelSpec(
+        # NOTE: Haiku 5.5 has tiered pricing. Rates below are for prompts <= 100K tokens.
+        # Prompts > 100K tokens cost $0.50 in / $2.50 out per MTok (not modeled here).
+        "claude-haiku-5-5": ModelSpec(
             provider="anthropic",
-            api_model="claude-3-5-haiku-latest",
-            pricing=ModelPricing(D("0.80"), D("4.00"), cached_input_per_mtok=D("0.08"), cache_write_per_mtok=D("1.00")),
+            api_model="claude-haiku-5-5",
+            pricing=ModelPricing(D("0.10"), D("0.50"), cached_input_per_mtok=D("0.01"), cache_write_per_mtok=D("0.125")),
         ),
     }
```

#### B2 — Absolute imports inside the `src` package
- **Problem:** `from pricing import ...` works only when running the file directly; `import src.gateway` failed.
- **Fix:** Relative imports (`from .pricing`, `from .schemas`) + `[tool.pytest.ini_options] pythonpath = ["."]`.
  Demo now runs as `python -m src.gateway`.
- **Diff:**
```diff
--- a/WEEK02/multi-provider-llm-gateway/src/pricing.py
+++ b/WEEK02/multi-provider-llm-gateway/src/pricing.py
@@ -16,5 +16,5 @@
 from typing import Final, Mapping, Optional
 
-from schemas import Provider
+from .schemas import Provider
 
 _PER_MILLION: Final = Decimal(1_000_000)
--- a/WEEK02/multi-provider-llm-gateway/src/gateway.py
+++ b/WEEK02/multi-provider-llm-gateway/src/gateway.py
@@ -30,6 +30,6 @@
 from pydantic import SecretStr
 
-from pricing import ResolvedModel, calculate_cost, resolve_model, supported_models
-from schemas import ChunkTelemetry, Provider, StreamChunk, UsageSource
+from .pricing import ResolvedModel, calculate_cost, resolve_model, supported_models
+from .schemas import ChunkTelemetry, Provider, StreamChunk, UsageSource
 
 __all__ = [
--- a/WEEK02/multi-provider-llm-gateway/pyproject.toml
+++ b/WEEK02/multi-provider-llm-gateway/pyproject.toml
@@ -20,2 +20,6 @@
     "uvicorn>=0.54.0",
 ]
+
+[tool.pytest.ini_options]
+pythonpath = ["."]
+asyncio_mode = "auto"
```

#### B3 — `.env` never loaded
- **Problem:** Gateway reads only `os.environ`; nothing called `load_dotenv()`.
- **Fix:** Load `.env` at the entry point (`__main__` demo, later `app.py`), not inside the library,
  so the gateway has no hidden side effects and tests can control env vars.
- **Diff:**
```diff
--- a/WEEK02/multi-provider-llm-gateway/src/gateway.py
+++ b/WEEK02/multi-provider-llm-gateway/src/gateway.py
@@ -628,4 +628,7 @@
 if __name__ == "__main__":
     import sys
+    from dotenv import find_dotenv, load_dotenv
+
+    load_dotenv(find_dotenv(usecwd=True))
 
     async def _demo() -> None:
```

#### B4 — Names don't match the spec; malformed keys not detected
- **Problem:** Draft used `ttft_s`, `cost_usd`, `InvalidAPIKeyError`; spec tests use `ttft_ms`,
  `total_cost_usd`, `AuthenticationError`. A key like `abc` was only rejected after a network call.
- **Fix:** Renamed fields (TTFT now in ms), renamed the exception (kept the old name as an alias),
  added local prefix checks (`sk-` / `sk-ant-`). Error messages never include the key.
- **Note:** A mistake during this edit was caught immediately because `ChunkTelemetry` uses
  `extra="forbid"`: Pydantic raised `ValidationError` instead of silently dropping a field.
- **Diff:**
```diff
--- a/WEEK02/multi-provider-llm-gateway/src/schemas.py
+++ b/WEEK02/multi-provider-llm-gateway/src/schemas.py
@@ -27,8 +27,8 @@
 
     latency_s: float = Field(ge=0, description="Wall-clock seconds from request start to stream end.")
-    ttft_s: Optional[float] = Field(default=None, ge=0, description="Seconds until the first text token. None if no text was produced.")
+    ttft_ms: Optional[float] = Field(default=None, ge=0, description="Milliseconds until the first text token. None if no text was produced.")
     tokens_per_second: float = Field(ge=0, description="Decode throughput: output tokens / (end - first token).")
 
-    cost_usd: float = Field(ge=0, description="Total request cost in USD, from the static pricing table.")
+    total_cost_usd: float = Field(ge=0, description="Total request cost in USD, from the static pricing table.")
 
     usage_source: UsageSource = Field(description="'provider' if counts came from the API, 'estimated' if reconstructed locally.")
--- a/WEEK02/multi-provider-llm-gateway/src/gateway.py
+++ b/WEEK02/multi-provider-llm-gateway/src/gateway.py
@@ -40,5 +40,5 @@
     "GatewayProviderError",
     "RateLimitExceededError",
-    "InvalidAPIKeyError",
+    "AuthenticationError",
     "ProviderConnectionError",
     "ProviderTimeoutError",
@@ -122,5 +122,5 @@
 
 
-class InvalidAPIKeyError(GatewayProviderError):
+class AuthenticationError(GatewayProviderError):
     """Missing, invalid, or insufficiently-permissioned API key."""
 
@@ -466,5 +466,7 @@
         if self._openai is None:
             if self._openai_key is None:
-                raise InvalidAPIKeyError("No OpenAI API key configured (set OPENAI_API_KEY).", provider="openai")
+                raise AuthenticationError("No OpenAI API key configured (set OPENAI_API_KEY).", provider="openai")
+            if not self._openai_key.get_secret_value().startswith("sk-"):
+                raise AuthenticationError("OPENAI_API_KEY looks malformed (expected prefix 'sk-').", provider="openai")
             self._openai = AsyncOpenAI(
                 api_key=self._openai_key.get_secret_value(),
@@ -477,5 +479,7 @@
         if self._anthropic is None:
             if self._anthropic_key is None:
-                raise InvalidAPIKeyError("No Anthropic API key configured (set ANTHROPIC_API_KEY).", provider="anthropic")
+                raise AuthenticationError("No Anthropic API key configured (set ANTHROPIC_API_KEY).", provider="anthropic")
+            if not self._anthropic_key.get_secret_value().startswith("sk-ant-"):
+                raise AuthenticationError("ANTHROPIC_API_KEY looks malformed (expected prefix 'sk-ant-').", provider="anthropic")
             self._anthropic = AsyncAnthropic(
                 api_key=self._anthropic_key.get_secret_value(),
@@ -535,7 +539,7 @@
             cache_write_tokens=cache_write_tokens,
             latency_s=round(latency, 4),
-            ttft_s=round(ttft, 4) if ttft is not None else None,
+            ttft_ms=round(ttft * 1000, 1) if ttft is not None else None,
             tokens_per_second=round(tps, 2),
-            cost_usd=float(cost),
+            total_cost_usd=float(cost),
             usage_source=usage_source,
             prompt_tokens_estimate=prompt_estimate,
@@ -610,5 +614,5 @@
     if isinstance(exc, (sdk.AuthenticationError, sdk.PermissionDeniedError)):
         # Deliberately do not echo the provider message: it may include a key fragment.
-        return InvalidAPIKeyError("Authentication with provider failed; check the API key and its permissions.", **common)
+        return AuthenticationError("Authentication with provider failed; check the API key and its permissions.", **common)
     if isinstance(exc, sdk.APITimeoutError):
         return ProviderTimeoutError("Provider request timed out.", retryable=True, **common)
```

#### B5 — AI workaround hid a removed parameter
- **Problem:** The Anthropic SDK no longer accepts `temperature`. Instead of treating that as a signal,
  the AI bypassed it with `extra_body`, claiming Claude 3.5 "still accepts it". Claude 5.x returned
  `400 invalid_request_error: temperature: Extra inputs are not permitted`.
- **Fix:** Added `supports_temperature` per model; temperature is sent only when supported.
- **Lesson:** An SDK signature change is information, not an obstacle to route around.
- **Diff:**
```diff
--- a/WEEK02/multi-provider-llm-gateway/src/pricing.py
+++ b/WEEK02/multi-provider-llm-gateway/src/pricing.py
@@ -35,4 +35,5 @@
     api_model: str  # ID sent to the API when the caller uses the bare registry name
     pricing: ModelPricing
+    supports_temperature: bool = True
 
 
@@ -44,4 +45,5 @@
     api_model: str
     pricing: ModelPricing
+    supports_temperature: bool = True
 
 
@@ -67,4 +69,5 @@
             provider="anthropic",
             api_model="claude-sonnet-5-5",
+            supports_temperature=False,
             pricing=ModelPricing(D("2.00"), D("10.00"), cached_input_per_mtok=D("0.20"), cache_write_per_mtok=D("2.50")),
         ),
@@ -74,4 +77,5 @@
             provider="anthropic",
             api_model="claude-haiku-5-5",
+            supports_temperature=False,
             pricing=ModelPricing(D("0.10"), D("0.50"), cached_input_per_mtok=D("0.01"), cache_write_per_mtok=D("0.125")),
         ),
@@ -94,5 +98,5 @@
     spec = MODEL_REGISTRY.get(key)
     if spec is not None:
-        return ResolvedModel(model, key, spec.provider, spec.api_model, spec.pricing)
+        return ResolvedModel(model, key, spec.provider, spec.api_model, spec.pricing, spec.supports_temperature)
 
     match = _SNAPSHOT_SUFFIX.search(key)
@@ -102,5 +106,5 @@
         if spec is not None:
             # Caller pinned a specific snapshot: send it to the API verbatim.
-            return ResolvedModel(model, base, spec.provider, key, spec.pricing)
+            return ResolvedModel(model, base, spec.provider, key, spec.pricing, spec.supports_temperature)
 
     raise KeyError(model)
--- a/WEEK02/multi-provider-llm-gateway/src/gateway.py
+++ b/WEEK02/multi-provider-llm-gateway/src/gateway.py
@@ -400,13 +400,14 @@
         async with self._provider_errors("anthropic", resolved.api_model):
             client = self._get_anthropic_client()
-            stream = await client.messages.create(
-                model=resolved.api_model,
-                max_tokens=self._anthropic_max_tokens,
-                messages=[{"role": "user", "content": prompt}],
-                stream=True,
-                # anthropic>=2 removed `temperature` from the typed signature; the
-                # API still accepts it, so send it on the wire for all SDK versions.
-                extra_body={"temperature": temperature},
-            )
+            request: dict[str, Any] = {
+                "model": resolved.api_model,
+                "max_tokens": self._anthropic_max_tokens,
+                "messages": [{"role": "user", "content": prompt}],
+                "stream": True,
+            }
+            # Newer Claude models (5.x) reject `temperature` with a 400; only send it when supported.
+            if resolved.supports_temperature:
+                request["extra_body"] = {"temperature": temperature}
+            stream = await client.messages.create(**request)
             try:
                 async for event in stream:
```

### 4.3 Verified results after fixes
| Model | In | Out | TTFT | Tokens/sec | Cost (USD) |
|---|---|---|---|---|---|
| gpt-4o-mini | 15 | 59 | 860.9 ms | 108.21 | 0.00003765 |
| claude-haiku-5-5 | 23 | 104 | 505.6 ms | 182.85 | 0.00005430 |

tiktoken prompt estimate (15) matched OpenAI's reported prompt tokens (15) exactly.