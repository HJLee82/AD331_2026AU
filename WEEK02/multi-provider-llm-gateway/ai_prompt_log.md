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
Errors: the custom types are RateLimitExceededError (includes the retry_after wait time), InvalidAPIKeyError, ProviderConnectionError / ProviderTimeoutError, ProviderBadRequestError and UnsupportedModelError. All of them inherit from GatewayProviderError or GatewayError.
No leaks: errors are raised from None, so no stack trace comes along. Key patterns are scrubbed from messages, and error text from failed logins is never repeated. The tests confirmed no key showed up in errors, logs or the gateway’s repr.
Early exit: if the caller stops reading partway through, the connection still closes properly.
⚠️ Things I had to work around, and one you should check
Anthropic SDK 2.x no longer has a temperature argument. The Claude 3.5 models still accept it, so I send it with extra_body, which works on both old and new SDK versions.
Both SDKs now use httpx2 instead of httpx. The connection-error handling catches errors from both libraries, so it doesn’t matter which version is installed.
The Claude 3.5 models are being retired by Anthropic. As far as I know, 3.5 Sonnet is already gone and Haiku is on its way out. The gateway will raise ProviderBadRequestError for a model that no longer exists. You’ll probably want to add current Claude models to MODEL_REGISTRY with today’s prices.
tiktoken check: the sandbox blocked tiktoken’s one-time download of its encoding file, so this check didn’t run here. It skipped itself without breaking the request. On a normal machine it downloads once and then works.

To try it: pip install openai anthropic tiktoken pydantic, set OPENAI_API_KEY and ANTHROPIC_API_KEY, then run python gateway.py gpt-4o-mini.

## 4. Audit Findings
### B1 — Retired Anthropic model IDs
- **Problem:** The draft registered `claude-3-5-sonnet-latest` and `claude-3-5-haiku-latest`.
  Both are retired (2025-10-28 and 2026-02-19), so any Anthropic request returns 404.
- **How found:** Checked Anthropic's model deprecations page; `claude-haiku-5-5` was rejected as UnsupportedModelError.
- **Fix:** Replaced them with `claude-sonnet-5-5` and `claude-haiku-5-5` at current prices.
  Documented Haiku 5.5's tiered pricing (>100K tokens) as a known limitation.
- **Diff:** diff --git a/WEEK02/multi-provider-llm-gateway/src/pricing.py b/WEEK02/multi-provider-llm-gateway/src/pricing.py
index 649717f..84ea96b 100644
--- a/WEEK02/multi-provider-llm-gateway/src/pricing.py
+++ b/WEEK02/multi-provider-llm-gateway/src/pricing.py
@@ -61,22 +61,25 @@ MODEL_REGISTRY: Final[Mapping[str, ModelSpec]] = MappingProxyType(
             pricing=ModelPricing(D("2.50"), D("10.00"), cached_input_per_mtok=D("1.25")),
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
 )
 
 # Accept pinned snapshots of registered models, e.g. "gpt-4o-2024-08-06",
-# "claude-3-5-haiku-20241022", "claude-3-5-sonnet-latest". Anything else
-# (e.g. "gpt-4o-audio-preview") is rejected rather than silently mis-priced.
+# "claude-3-5-haiku-20241022", "claude-3-5-sonnet-latest". Anything else is rejected rather than silently mis-priced.
 _SNAPSHOT_SUFFIX: Final = re.compile(r"-(?:\d{4}-\d{2}-\d{2}|\d{8}|latest)$")

 ### B2 — Absolute imports break inside the `src` package
- **Problem:** `gateway.py` used `from pricing import ...` and `from schemas import ...`.
  Importing the package (`import src.gateway`, as the tests and app must) raised
  `ModuleNotFoundError: No module named 'pricing'`.
- **How found:** Ran `python -c "import src.gateway"` from the project root.
- **Fix:** Switched to relative imports (`from .pricing`, `from .schemas`) and added
  `[tool.pytest.ini_options] pythonpath = ["."]` so tests resolve `src`.
- **Diff:**
```diff
(git --no-pager diff src/ pyproject.toml | pbcopy)
```
 
 ### B3 — `.env` file never loaded
- **Problem:** The gateway reads keys only from `os.environ`, and no code called `load_dotenv()`.
  With valid keys in `.env`, running the demo failed:
  `Gateway error: [openai] No OpenAI API key configured (set OPENAI_API_KEY).`
- **How found:** Ran `uv run python -m src.gateway gpt-4o-mini` after setting up `.env`.
- **Fix:** Load `.env` at the entry point (`__main__` demo; later `app.py`), not inside the
  library class, so the gateway stays side-effect free and tests can control env vars.
- **Diff:**
```diff
(git --no-pager diff src/gateway.py | pbcopy)
```

### B5 — Claude 5.x rejects `temperature` (AI workaround hid the problem)
- **Problem:** The draft sent `temperature` via `extra_body` to bypass the Anthropic SDK's
  signature, which no longer accepts it. Live call to `claude-haiku-5-5` returned
  `400 invalid_request_error: temperature: Extra inputs are not permitted`.
- **How found:** First live Anthropic request after fixing B1.
- **Lesson:** The AI treated an SDK signature change as a typing nuisance instead of a signal
  that the parameter was removed.
- **Fix:** Added `supports_temperature` per model; temperature is only sent when supported.
- **Diff:**
```diff
(git --no-pager diff src/ | pbcopy)
```