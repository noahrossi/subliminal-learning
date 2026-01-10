"""Async OpenAI client with rate limiting and retry logic."""

import asyncio
import os
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI, RateLimitError, APIError


@dataclass
class TokenUsage:
    """Track token usage across API calls."""

    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def cost(self, input_price_per_1m: float, output_price_per_1m: float) -> float:
        """Calculate cost in dollars."""
        return (self.input_tokens / 1_000_000) * input_price_per_1m + (
            self.output_tokens / 1_000_000
        ) * output_price_per_1m


class AsyncClient:
    """Async OpenAI client with rate limiting and retry logic."""

    def __init__(
        self,
        concurrency: int = 50,
        max_retries: int = 5,
        base_delay: float = 1.0,
    ) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self._client = AsyncOpenAI(api_key=api_key)
        self._semaphore = asyncio.Semaphore(concurrency)
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._usage = TokenUsage()
        self._lock = asyncio.Lock()

    @property
    def usage(self) -> TokenUsage:
        """Get accumulated token usage."""
        return self._usage

    def reset_usage(self) -> None:
        """Reset token usage counters."""
        self._usage = TokenUsage()

    async def _add_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Thread-safe usage tracking."""
        async with self._lock:
            self._usage.add(input_tokens, output_tokens)

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        n: int = 1,
    ) -> dict[str, Any]:
        """
        Make a chat completion request with rate limiting and retries.

        Args:
            messages: Chat messages
            model: Model ID
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            n: Number of completions to return (default 1)

        Returns:
            dict with "contents" (list of strings) and "finish_reasons" (list)
        """
        async with self._semaphore:
            for attempt in range(self._max_retries):
                try:
                    response = await self._client.chat.completions.create(
                        model=model,
                        messages=messages,  # type: ignore
                        temperature=temperature,
                        max_tokens=max_tokens,
                        n=n,
                    )

                    # Track usage
                    if response.usage:
                        await self._add_usage(
                            response.usage.prompt_tokens,
                            response.usage.completion_tokens,
                        )

                    return {
                        "contents": [c.message.content for c in response.choices],
                        "finish_reasons": [c.finish_reason for c in response.choices],
                    }

                except RateLimitError:
                    if attempt == self._max_retries - 1:
                        raise
                    delay = self._base_delay * (2**attempt)
                    await asyncio.sleep(delay)

                except APIError:
                    if attempt == self._max_retries - 1:
                        raise
                    delay = self._base_delay * (2**attempt)
                    await asyncio.sleep(delay)

        # Should never reach here
        raise RuntimeError("Unexpected state in chat_completion")


# Global client instance (lazy initialization)
_client: AsyncClient | None = None


def get_client(concurrency: int = 50) -> AsyncClient:
    """Get or create the global async client."""
    global _client
    if _client is None:
        _client = AsyncClient(concurrency=concurrency)
    return _client


def reset_client() -> None:
    """Reset the global client (useful for testing)."""
    global _client
    _client = None
