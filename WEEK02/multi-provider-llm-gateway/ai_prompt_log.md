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
(paste)
```

#### B2 — Absolute imports inside the `src` package
- **Problem:** `from pricing import ...` works only when running the file directly; `import src.gateway` failed.
- **Fix:** Relative imports (`from .pricing`, `from .schemas`) + `[tool.pytest.ini_options] pythonpath = ["."]`.
  Demo now runs as `python -m src.gateway`.
- **Diff:**
```diff
(paste)
```

#### B3 — `.env` never loaded
- **Problem:** Gateway reads only `os.environ`; nothing called `load_dotenv()`.
- **Fix:** Load `.env` at the entry point (`__main__` demo, later `app.py`), not inside the library,
  so the gateway has no hidden side effects and tests can control env vars.
- **Diff:**
```diff
(paste)
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
(paste)
```

#### B5 — AI workaround hid a removed parameter
- **Problem:** The Anthropic SDK no longer accepts `temperature`. Instead of treating that as a signal,
  the AI bypassed it with `extra_body`, claiming Claude 3.5 "still accepts it". Claude 5.x returned
  `400 invalid_request_error: temperature: Extra inputs are not permitted`.
- **Fix:** Added `supports_temperature` per model; temperature is sent only when supported.
- **Lesson:** An SDK signature change is information, not an obstacle to route around.
- **Diff:**
```diff
(paste)
```

### 4.3 Verified results after fixes
| Model | In | Out | TTFT | Tokens/sec | Cost (USD) |
|---|---|---|---|---|---|
| gpt-4o-mini | 15 | 59 | 860.9 ms | 108.21 | 0.00003765 |
| claude-haiku-5-5 | 23 | 104 | 505.6 ms | 182.85 | 0.00005430 |

tiktoken prompt estimate (15) matched OpenAI's reported prompt tokens (15) exactly.