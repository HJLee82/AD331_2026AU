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
- **Diff:** (paste output of `git diff src/pricing.py`)