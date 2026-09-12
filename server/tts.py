"""
TTS（文字转语音）Provider 抽象层与实现。

设计思想来自 Local Live 项目的 packages/providers/tts/base.py：
- 统一的 TTSChunk 数据结构（PCM 音频块）
- 流式合成（输入文本流，输出音频流）
- 统一的错误归一化

精简版保留两种实现：
1. OpenAICompatibleTTS —— OpenAI 兼容的 TTS 接口（默认）
2. EdgeTTS —— 微软 Edge 浏览器的免费 TTS（离线可用示例，音质好）
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol


# ============================================================
# 数据结构
# ============================================================

@dataclass(frozen=True)
class TTSChunk:
    audio: bytes
    provider: str
    sample_rate: int = 24000
    num_channels: int = 1
    audio_format: str = "pcm"
    metadata: dict[str, Any] = field(default_factory=dict)


class TTSProviderError(Exception):
    """安全的 TTS 错误，不包含 SDK 原始信息或密钥。"""

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

class TTSProvider(Protocol):
    name: str

    def synthesize_stream(
        self,
        text_stream: AsyncIterator[str],
        *,
        voice: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[TTSChunk]:
        ...

    async def close(self) -> None:
        ...


# ============================================================
# 实现一：OpenAI 兼容 TTS
# ============================================================

class OpenAICompatibleTTS:
    """
    OpenAI 兼容的 TTS Provider。

    支持两种模式：
    - mode="speech"（默认）：标准 /v1/audio/speech 接口（OpenAI官方、SiliconFlow等）
    - mode="chat"：通过 /v1/chat/completions 流式返回base64 PCM（xiaomimimo mimo-tts等）
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "tts-1",
        voice: str = "alloy",
        sample_rate: int = 24000,
        mode: str = "speech",
    ) -> None:
        self.name = f"openai-compat:{model}"
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._voice = voice
        self.sample_rate = sample_rate
        self._mode = mode  # "speech" | "chat"

    async def _synch_speech(self, text: str, voice: str) -> tuple[bytes, str]:
        """标准 /audio/speech 接口，返回 (音频字节, 格式)"""
        import aiohttp
        payload = {
            "model": self._model,
            "input": text,
            "voice": voice,
            "response_format": "mp3",
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._base_url}/audio/speech",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    raise TTSProviderError(
                        self.name, "http_error",
                        f"TTS HTTP {resp.status}",
                        retryable=resp.status >= 500,
                        status_code=resp.status,
                    )
                return await resp.read(), "mp3"

    async def _synch_chat(self, text: str, voice: str) -> tuple[bytes, str]:
        """chat.completions 流式TTS（xiaomimimo等），返回 (PCM16字节, 格式)"""
        chunks = []
        async for pcm in self._stream_chat_pcm(text, voice):
            chunks.append(pcm)
        return b"".join(chunks), "pcm"

    async def _stream_chat_pcm(self, text: str, voice: str):
        """流式拉取 chat TTS 的 PCM 分片。"""
        import base64
        import json
        import aiohttp

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [
                {"role": "user", "content": "用自然友好的语气朗读，适合语音助手场景"},
                {"role": "assistant", "content": text},
            ],
            "audio": {
                "format": "pcm16",
                "voice": voice,
                "sample_rate": self.sample_rate,
            },
            "stream": True,
        }

        total = 0
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise TTSProviderError(
                        self.name, "http_error",
                        f"TTS HTTP {resp.status}: {body[:100]}",
                        retryable=resp.status >= 500,
                        status_code=resp.status,
                    )

                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue
                    # 兼容 "data:" 与 "data: "
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                    else:
                        continue
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices") or [{}]
                    delta = choices[0].get("delta", {}) if choices else {}
                    audio_data = delta.get("audio")
                    if not audio_data:
                        # 兼容部分实现把音频放在 message 里
                        audio_data = (chunk.get("message") or {}).get("audio")

                    b64 = ""
                    if isinstance(audio_data, dict):
                        b64 = audio_data.get("data", "") or audio_data.get("audio", "")
                    elif isinstance(audio_data, str):
                        b64 = audio_data

                    if b64:
                        try:
                            pcm = base64.b64decode(b64)
                        except Exception:
                            continue
                        if pcm:
                            total += len(pcm)
                            yield pcm

        print(f"[TTS] chat 合成完成: {total} bytes (~{total / (self.sample_rate * 2):.2f}s @ {self.sample_rate}Hz)")

    async def synthesize_stream(
        self,
        text_stream: AsyncIterator[str],
        *,
        voice: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[TTSChunk]:
        # chat 模式：整段一次合成，避免多段拼接导致末尾被截断
        if self._mode == "chat":
            full_text = ""
            async for delta in text_stream:
                full_text += delta
            if not full_text.strip():
                return
            try:
                async for pcm in self._stream_chat_pcm(full_text, voice or self._voice):
                    if pcm:
                        yield TTSChunk(
                            audio=pcm,
                            provider=self.name,
                            sample_rate=self.sample_rate,
                            audio_format="pcm",
                        )
            except asyncio.CancelledError:
                raise
            except TTSProviderError:
                raise
            except Exception as e:
                raise TTSProviderError(
                    self.name, "unknown",
                    f"TTS failed: {type(e).__name__}",
                    retryable=True,
                )
            return

        # speech 模式：按句子分段合成（兼容标准 /audio/speech）
        buffer = ""
        sentence_endings = "。！？.!?；;\n"

        async def _synth(text: str) -> tuple[bytes, str]:
            if not text.strip():
                return b"", "mp3"
            return await self._synch_speech(text, voice or self._voice)

        try:
            async for delta in text_stream:
                buffer += delta
                while any(c in buffer for c in sentence_endings):
                    idx = min(
                        (buffer.find(c) for c in sentence_endings if c in buffer),
                        default=-1,
                    )
                    if idx < 0:
                        break
                    sentence = buffer[: idx + 1]
                    buffer = buffer[idx + 1:]
                    if sentence.strip():
                        audio, fmt = await _synth(sentence)
                        if audio:
                            yield TTSChunk(
                                audio=audio,
                                provider=self.name,
                                sample_rate=self.sample_rate,
                                audio_format=fmt,
                            )

            if buffer.strip():
                audio, fmt = await _synth(buffer)
                if audio:
                    yield TTSChunk(
                        audio=audio,
                        provider=self.name,
                        sample_rate=self.sample_rate,
                        audio_format=fmt,
                    )
        except asyncio.CancelledError:
            raise
        except TTSProviderError:
            raise
        except Exception as e:
            raise TTSProviderError(
                self.name, "unknown",
                f"TTS failed: {type(e).__name__}",
                retryable=True,
            )

    async def close(self) -> None:
        pass


# ============================================================
# 实现二：Edge TTS（微软免费 TTS）
# ============================================================

class EdgeTTS:
    """
    微软 Edge 浏览器的免费 TTS Provider。

    优点：完全免费、音质好、支持多种语言和声音、无需 API Key。
    缺点：依赖微软服务（非完全离线）、非官方接口可能有稳定性风险。

    安装：pip install edge-tts
    中文声音推荐：
    - zh-CN-XiaoxiaoNeural（晓晓，女声，自然）
    - zh-CN-YunxiNeural（云希，男声，沉稳）
    - zh-CN-YunjianNeural（云健，男声，磁性）
    """

    def __init__(
        self,
        *,
        voice: str = "zh-CN-XiaoxiaoNeural",
        rate: str = "+0%",
        volume: str = "+0%",
        sample_rate: int = 24000,
    ) -> None:
        self.name = f"edge-tts:{voice}"
        self._voice = voice
        self._rate = rate
        self._volume = volume
        self.sample_rate = sample_rate

    async def synthesize_stream(
        self,
        text_stream: AsyncIterator[str],
        *,
        voice: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[TTSChunk]:
        import edge_tts

        # 累积完整文本（edge-tts 不支持真正的流式输入）
        full_text = ""
        async for delta in text_stream:
            full_text += delta

        if not full_text.strip():
            return

        try:
            communicate = edge_tts.Communicate(
                full_text,
                voice or self._voice,
                rate=self._rate,
                volume=self._volume,
            )
            audio_buffer = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_buffer.extend(chunk["data"])
            if audio_buffer:
                yield TTSChunk(
                    audio=bytes(audio_buffer),
                    provider=self.name,
                    sample_rate=self.sample_rate,
                    audio_format="mp3",
                )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            raise TTSProviderError(
                self.name, "unknown",
                f"Edge TTS failed: {type(e).__name__}",
                retryable=True,
            )

    async def close(self) -> None:
        pass
