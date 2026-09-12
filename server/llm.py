"""
LLM（大语言模型）Provider 抽象层与实现。

设计思想来自 Local Live 项目的 packages/providers/llm/base.py：
- 统一的 ChatMessage / LLMRequest / LLMChunk 数据结构
- 流式输出（AsyncIterator[LLMChunk]）
- 统一的错误归一化（ProviderError，不暴露 SDK 原始信息）

精简版保留两种实现：
1. OpenAICompatibleLLM —— 任何 OpenAI 兼容的 LLM 接口（默认，支持 DeepSeek、
   SiliconFlow、OpenRouter、本地 vLLM 等）
2. OllamaLLM —— 本地 Ollama 模型（离线可用示例）
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypedDict


# ============================================================
# 数据结构
# ============================================================

class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


@dataclass(frozen=True)
class LLMRequest:
    messages: Sequence[ChatMessage]
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMChunk:
    text: str
    provider: str
    model: str | None = None
    finish_reason: str | None = None


class LLMProviderError(Exception):
    """安全的 LLM 错误，不包含 SDK 原始信息或密钥。"""

    def __init__(
        self,
        provider: str,
        kind: str,
        message: str,
        *,
        retryable: bool = False,
        fallback_allowed: bool = True,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.kind = kind
        self.message = message
        self.retryable = retryable
        self.fallback_allowed = fallback_allowed
        self.status_code = status_code


# ============================================================
# Provider 抽象接口
# ============================================================

class LLMProvider(Protocol):
    name: str

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMChunk]:
        ...

    async def close(self) -> None:
        ...


# ============================================================
# 实现一：OpenAI 兼容 LLM
# ============================================================

class OpenAICompatibleLLM:
    """
    OpenAI 兼容的流式 LLM Provider。

    适用于：
    - OpenAI 官方（gpt-4o, gpt-4o-mini）
    - DeepSeek（deepseek-chat, deepseek-reasoner）
    - SiliconFlow（Qwen, GLM, Llama 等）
    - OpenRouter（聚合数百种模型）
    - 本地 vLLM / Ollama（开启 OpenAI 兼容模式）
    - 任何兼容 /v1/chat/completions 接口的服务

    核心特性：
    - 真正的流式输出（SSE 解析）
    - 支持中断（asyncio.CancelledError 会正确关闭连接）
    - 统一的错误归一化
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        default_temperature: float = 0.7,
        default_max_tokens: int = 1024,
        timeout: float = 60.0,
    ) -> None:
        self.name = f"openai-compat:{model}"
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens
        self._timeout = timeout

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMChunk]:
        import aiohttp

        payload: dict[str, Any] = {
            "model": request.model or self._model,
            "messages": list(request.messages),
            "stream": True,
            "temperature": request.temperature
            if request.temperature is not None
            else self._default_temperature,
            "max_tokens": request.max_tokens or self._default_max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self._timeout),
                ) as resp:
                    if resp.status != 200:
                        err_text = await resp.text()
                        raise LLMProviderError(
                            self.name,
                            "http_error",
                            f"LLM HTTP {resp.status}",
                            retryable=resp.status >= 500,
                            fallback_allowed=True,
                            status_code=resp.status,
                        )

                    # 解析 SSE 流
                    async for line in resp.content:
                        line = line.decode("utf-8").strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        choices = obj.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        finish_reason = choices[0].get("finish_reason")

                        if content:
                            yield LLMChunk(
                                text=content,
                                provider=self.name,
                                model=obj.get("model"),
                                finish_reason=finish_reason,
                            )
                        if finish_reason:
                            yield LLMChunk(
                                text="",
                                provider=self.name,
                                model=obj.get("model"),
                                finish_reason=finish_reason,
                            )
        except asyncio.CancelledError:
            # 中断是正常操作（用户打断），不包装为错误
            raise
        except LLMProviderError:
            raise
        except Exception as e:
            raise LLMProviderError(
                self.name,
                "unknown",
                f"LLM request failed: {type(e).__name__}",
                retryable=True,
                fallback_allowed=True,
            )

    async def close(self) -> None:
        # aiohttp 使用 with 语句自动管理，无需额外关闭
        pass


# ============================================================
# 实现二：Ollama 本地 LLM
# ============================================================

class OllamaLLM:
    """
    本地 Ollama LLM Provider。

    优点：完全离线、免费、无 API 调用限制、数据不出本机。
    缺点：需要较好的 GPU 才能达到流畅速度。

    安装：
    1. 下载 Ollama：https://ollama.com/
    2. 拉取模型：ollama pull qwen2.5:7b
    3. 启动服务：ollama serve（默认 http://localhost:11434）
    """

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b",
        default_temperature: float = 0.7,
        default_max_tokens: int = 1024,
    ) -> None:
        self.name = f"ollama:{model}"
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMChunk]:
        import aiohttp

        payload = {
            "model": request.model or self._model,
            "messages": list(request.messages),
            "stream": True,
            "options": {
                "temperature": request.temperature
                if request.temperature is not None
                else self._default_temperature,
                "num_predict": request.max_tokens or self._default_max_tokens,
            },
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._base_url}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120.0),
                ) as resp:
                    if resp.status != 200:
                        err_text = await resp.text()
                        raise LLMProviderError(
                            self.name,
                            "http_error",
                            f"Ollama HTTP {resp.status}",
                            retryable=resp.status >= 500,
                            status_code=resp.status,
                        )

                    async for line in resp.content:
                        line = line.decode("utf-8").strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        message = obj.get("message", {})
                        content = message.get("content", "")
                        done = obj.get("done", False)

                        if content:
                            yield LLMChunk(
                                text=content,
                                provider=self.name,
                                model=self._model,
                            )
                        if done:
                            yield LLMChunk(
                                text="",
                                provider=self.name,
                                model=self._model,
                                finish_reason="stop",
                            )
        except asyncio.CancelledError:
            raise
        except LLMProviderError:
            raise
        except Exception as e:
            raise LLMProviderError(
                self.name,
                "unknown",
                f"Ollama request failed: {type(e).__name__}",
                retryable=True,
            )

    async def close(self) -> None:
        pass
