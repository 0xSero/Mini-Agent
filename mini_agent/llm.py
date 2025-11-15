"""LLM client for MiniMax M2 via Anthropic-compatible API (with stub fallback)."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from .retry import RetryConfig as RetryConfigBase
from .retry import async_retry
from .schema import FunctionCall, LLMResponse, Message, ToolCall

logger = logging.getLogger(__name__)


class LLMClient:
    """MiniMax M2 LLM Client via Anthropic-compatible endpoint.

    Supported models:
    - MiniMax-M2
    """

    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.minimax.io/anthropic",
        model: str = "MiniMax-M2",
        retry_config: RetryConfigBase | None = None,
        *,
        offline_mode: bool | None = None,
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.retry_config = retry_config or RetryConfigBase()
        self.retry_callback = None
        self._offline = offline_mode if offline_mode is not None else api_key.upper().startswith("TEST_")
        self._fake_call_counter = 0

    async def _make_api_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute API request (core method that can be retried)

        Args:
            payload: Request payload

        Returns:
            API response result

        Raises:
            Exception: API call failed
        """
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.api_base}/v1/messages",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01",
                },
                json=payload,
            )

            result = response.json()

        # Check for errors (Anthropic format)
        if result.get("type") == "error":
            error_info = result.get("error", {})
            error_msg = f"API Error ({error_info.get('type')}): {error_info.get('message')}"
            raise Exception(error_msg)

        # Check for MiniMax base_resp errors
        if "base_resp" in result:
            base_resp = result["base_resp"]
            status_code = base_resp.get("status_code")
            status_msg = base_resp.get("status_msg")

            if status_code not in [0, 1000, None]:
                error_msg = f"MiniMax API Error (code {status_code}): {status_msg}"
                if status_code == 1008:
                    error_msg += "\n\n⚠️  Insufficient account balance, please recharge on MiniMax platform"
                elif status_code == 2013:
                    error_msg += f"\n\n⚠️  Model '{self.model}' is not supported"
                raise Exception(error_msg)

        return result

    async def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Generate response from LLM."""
        if self._offline:
            return await self._fake_response(messages, tools)
        # Extract system message (Anthropic requires it separately)
        system_message = None
        api_messages = []

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
                continue

            # For user and assistant messages
            if msg.role in ["user", "assistant"]:
                # Handle assistant messages with thinking or tool calls
                if msg.role == "assistant" and (msg.thinking or msg.tool_calls):
                    # Build content blocks for assistant with thinking and/or tool calls
                    content_blocks = []

                    # Add thinking block if present
                    if msg.thinking:
                        content_blocks.append({"type": "thinking", "thinking": msg.thinking})

                    # Add text content if present
                    if msg.content:
                        content_blocks.append({"type": "text", "text": msg.content})

                    # Add tool use blocks
                    if msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            content_blocks.append(
                                {
                                    "type": "tool_use",
                                    "id": tool_call.id,
                                    "name": tool_call.function.name,
                                    "input": tool_call.function.arguments,
                                }
                            )

                    api_messages.append({"role": "assistant", "content": content_blocks})
                else:
                    api_messages.append({"role": msg.role, "content": msg.content})

            # For tool result messages
            elif msg.role == "tool":
                # Anthropic uses user role with tool_result content blocks
                api_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id,
                                "content": msg.content,
                            }
                        ],
                    }
                )

        # Build request payload
        payload = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": 16384,  # Increased to handle longer outputs
        }

        # Add system message if present
        if system_message:
            payload["system"] = system_message

        # Add tools if provided
        if tools:
            payload["tools"] = tools

        # Make API request with retry logic
        if self.retry_config.enabled:
            # Apply retry logic
            retry_decorator = async_retry(config=self.retry_config, on_retry=self.retry_callback)
            api_call = retry_decorator(self._make_api_request)
            result = await api_call(payload)
        else:
            # Don't use retry
            result = await self._make_api_request(payload)

        # Parse Anthropic response format
        content_blocks = result.get("content", [])
        stop_reason = result.get("stop_reason", "stop")

        # Extract text content, thinking, and tool calls
        text_content = ""
        thinking_content = ""
        tool_calls = []

        for block in content_blocks:
            if block.get("type") == "text":
                text_content += block.get("text", "")
            elif block.get("type") == "thinking":
                thinking_content += block.get("thinking", "")
            elif block.get("type") == "tool_use":
                # Parse Anthropic tool_use block
                tool_calls.append(
                    ToolCall(
                        id=block.get("id"),
                        type="function",
                        function=FunctionCall(
                            name=block.get("name"),
                            arguments=block.get("input", {}),
                        ),
                    )
                )

        return LLMResponse(
            content=text_content,
            thinking=thinking_content if thinking_content else None,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason=stop_reason,
        )

    async def _fake_response(self, messages: list[Message], tools: list[dict[str, Any]] | None) -> LLMResponse:
        """Deterministic offline responses used when API key starts with TEST_."""
        # Helper functions
        def last_user_text() -> str:
            for msg in reversed(messages):
                if msg.role == "user" and isinstance(msg.content, str):
                    return msg.content
            return ""

        def has_tool_result(name: str) -> bool:
            return any(msg.role == "tool" and msg.name == name for msg in messages)

        def tool_call(name: str, arguments: dict[str, Any]) -> list[ToolCall]:
            self._fake_call_counter += 1
            return [
                ToolCall(
                    id=f"stub-{self._fake_call_counter}",
                    type="function",
                    function=FunctionCall(name=name, arguments=arguments),
                )
            ]

        text = last_user_text().lower()

        if "say 'hello, mini agent!'" in text or "hello, mini agent" in text:
            return LLMResponse(content="Hello, Mini Agent!", thinking=None, tool_calls=None, finish_reason="stop")

        if "calculate 123 + 456" in text and tools:
            return LLMResponse(
                content="",
                thinking=None,
                tool_calls=tool_call("calculator", {"operation": "add", "a": 123, "b": 456}),
                finish_reason="tool_use",
            )

        if "test.txt" in text and "hello from agent" in text:
            if has_tool_result("write_file"):
                return LLMResponse(content="Created test.txt with the requested content.", thinking=None, tool_calls=None, finish_reason="stop")
            return LLMResponse(
                content="",
                thinking=None,
                tool_calls=tool_call("write_file", {"path": "test.txt", "content": "Hello from Agent!"}),
                finish_reason="tool_use",
            )

        if "list all files" in text and "bash" in text:
            if has_tool_result("bash"):
                return LLMResponse(content="Listed the files in the workspace.", thinking=None, tool_calls=None, finish_reason="stop")
            return LLMResponse(
                content="",
                thinking=None,
                tool_calls=tool_call("bash", {"command": "ls", "timeout": 30, "run_in_background": False}),
                finish_reason="tool_use",
            )

        # Generic fallback
        fallback = last_user_text() or "Hello!"
        return LLMResponse(content=f"Stub response: {fallback.strip()}", thinking=None, tool_calls=None, finish_reason="stop")
