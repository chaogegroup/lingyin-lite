"""
STT（语音转文字）Provider 抽象层与实现。

设计思想来自 Local Live 项目的 packages/providers/stt/base.py：
- 统一的 Provider 接口（Protocol）
- 流式事件输出（START_OF_SPEECH / PARTIAL / FINAL / END_OF_SPEECH）
- 统一的错误归一化（不暴露 SDK 原始错误信息）

精简版保留两种实现：
1. OpenAICompatibleSTT —— 任何 OpenAI 兼容的 STT 接口（默认）
2. FasterWhisperSTT —— 本地 faster-whisper 实现（离线可用示例）
"""

from __future__ import annotations

import asyncio
import io
import time
import wave
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


def pcm16_to_wav(pcm_data: bytes, sample_rate: int = 16000) -> bytes:
    """把裸 PCM16 单声道数据包装成合法 WAV 字节（带 44 字节头）。"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


# ============================================================
# 数据结构
# ============================================================

class STTEventType(str, Enum):
    START_OF_SPEECH = "start_of_speech"
    PARTIAL_TRANSCRIPT = "partial_transcript"
    FINAL_TRANSCRIPT = "final_transcript"
    END_OF_SPEECH = "end_of_speech"
    ERROR = "error"


@dataclass(frozen=True)
class STTEvent:
    type: STTEventType
    provider: str
    text: str = ""
    language: str | None = None
    timestamp: float = field(default_factory=time.time)
    latency_ms: float | None = None


class STTProviderError(Exception):
    """安全的 STT 错误，不包含 SDK 原始信息或密钥。"""

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

class STTProvider(Protocol):
    """所有 STT Provider 必须实现的接口。"""

    name: str

    async def start(self, *, language: str | None = None, sample_rate: int = 16000) -> None:
        """启动识别会话。"""
        ...

    async def send_audio(self, chunk: bytes) -> None:
        """发送 PCM16 音频块。"""
        ...

    def events(self) -> AsyncIterator[STTEvent]:
        """异步事件流。"""
        ...

    async def stop(self) -> str:
        """停止识别会话，返回最终识别文本。"""
        ...


# ============================================================
# 实现一：OpenAI 兼容 STT
# ============================================================

class OpenAICompatibleSTT:
    """
    OpenAI 兼容的流式 STT Provider。

    适用于：OpenAI Whisper API、SiliconFlow、Groq、以及任何兼容
    OpenAI audio/transcriptions 接口的服务。

    注意：OpenAI 官方的 audio/transcriptions 是非流式的（一次上传返回全文），
    本实现采用"分段发送+累积返回"的方式模拟流式效果。
    真正的流式 STT 建议使用支持 WebSocket 流式的服务（如腾讯云实时语音、
    Deepgram 等），可参照本接口自行实现。
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "whisper-1",
        language: str = "zh",
    ) -> None:
        self.name = f"openai-compat:{model}"
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._language = language
        self._events_queue: asyncio.Queue[STTEvent | None] = asyncio.Queue()
        self._audio_buffer = bytearray()
        self._sample_rate = 16000
        self._running = False
        self._worker_task: asyncio.Task | None = None

    async def start(self, *, language: str | None = None, sample_rate: int = 16000) -> None:
        self._sample_rate = sample_rate
        if language:
            self._language = language
        self._audio_buffer.clear()
        self._events_queue = asyncio.Queue()
        self._running = True
        self._worker_task = asyncio.create_task(self._transcribe_worker())

    async def send_audio(self, chunk: bytes) -> None:
        if not self._running:
            return
        self._audio_buffer.extend(chunk)

    def events(self) -> AsyncIterator[STTEvent]:
        async def _gen():
            while True:
                evt = await self._events_queue.get()
                if evt is None:
                    return
                yield evt
        return _gen()

    async def stop(self) -> str:
        """停止识别会话。冲刷缓冲区，返回最终识别文本。"""
        self._running = False
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

        final_text = ""
        if len(self._audio_buffer) >= 1600:  # >= 50ms
            final_text = await self._transcribe_bytes(bytes(self._audio_buffer))
            self._audio_buffer.clear()

        await self._events_queue.put(None)
        return final_text

    async def _transcribe_bytes(self, audio_data: bytes) -> str:
        """上传一段 PCM16 音频并返回识别文本。"""
        import aiohttp

        if len(audio_data) < 1600:
            return ""

        try:
            async with aiohttp.ClientSession() as session:
                wav_data = pcm16_to_wav(audio_data, self._sample_rate)
                form = aiohttp.FormData()
                form.add_field(
                    "file",
                    wav_data,
                    filename="audio.wav",
                    content_type="audio/wav",
                )
                form.add_field("model", self._model)
                form.add_field("language", self._language)

                headers = {"Authorization": f"Bearer {self._api_key}"}
                duration = len(audio_data) / (self._sample_rate * 2)
                print(f"[STT] 发送转录请求: {len(wav_data)} bytes ({duration:.1f}s), model={self._model}")
                async with session.post(
                    f"{self._base_url}/audio/transcriptions",
                    data=form,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=20),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        text = data.get("text", "").strip()
                        print(f"[STT] 识别结果: {text}")
                        return text
                    else:
                        err_text = await resp.text()
                        print(f"[STT] HTTP {resp.status}: {err_text[:200]}")
                        await self._events_queue.put(STTEvent(
                            type=STTEventType.ERROR,
                            provider=self.name,
                            text=f"STT error {resp.status}: {err_text[:200]}",
                        ))
                        return ""
        except Exception as e:
            print(f"[STT] 异常: {type(e).__name__}: {e}")
            await self._events_queue.put(STTEvent(
                type=STTEventType.ERROR,
                provider=self.name,
                text=f"STT exception: {type(e).__name__}",
            ))
            return ""

    async def _transcribe_worker(self) -> None:
        """后台 worker：整段累积，stop() 时统一识别；期间仅作可选中间检测。"""
        last_partial = time.time()

        while self._running:
            await asyncio.sleep(0.4)

            audio_duration = len(self._audio_buffer) / (self._sample_rate * 2)
            elapsed = time.time() - last_partial

            # 每约 2.5s 做一次中间识别（PARTIAL），便于前端实时显示
            if audio_duration >= 1.2 and elapsed >= 2.5 and len(self._audio_buffer) >= 3200:
                snapshot = bytes(self._audio_buffer)
                last_partial = time.time()
                text = await self._transcribe_bytes(snapshot)
                if text:
                    await self._events_queue.put(STTEvent(
                        type=STTEventType.PARTIAL_TRANSCRIPT,
                        provider=self.name,
                        text=text,
                        language=self._language,
                    ))


# ============================================================
# 实现二：Faster Whisper 本地 STT
# ============================================================

class FasterWhisperSTT:
    """
    本地 faster-whisper STT Provider。

    注意：WhisperModel 非线程安全，所有 transcribe 必须串行；
    使用专用单线程 executor，避免与默认线程池死锁。
    """

    def __init__(
        self,
        *,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "int8",
        language: str = "zh",
    ) -> None:
        self.name = f"faster-whisper:{model_size}"
        self._model_size = model_size
        # Windows CPU 上 auto 偶发异常，无 GPU 时直接 cpu 更稳
        self._device = device if device and device != "auto" else "cpu"
        self._compute_type = compute_type
        self._language = language
        self._model = None
        self._events_queue: asyncio.Queue[STTEvent | None] = asyncio.Queue()
        self._audio_buffer = bytearray()
        self._sample_rate = 16000
        self._running = False
        self._worker_task: asyncio.Task | None = None
        self._executor = None  # 懒加载单线程池
        self._transcribe_lock = None  # threading.Lock

    def _ensure_executor(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fw-stt")
        if self._transcribe_lock is None:
            self._transcribe_lock = threading.Lock()
        return self._executor

    def _load_model(self):
        import os

        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

        from faster_whisper import WhisperModel

        last_err = None
        try:
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
                local_files_only=True,
            )
            print(f"[STT] Faster Whisper 已从本地缓存加载: {self._model_size} ({self._device})")
            return
        except Exception as e:
            last_err = e

        try:
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )
            print(f"[STT] Faster Whisper 在线加载成功: {self._model_size} ({self._device})")
            return
        except Exception as e:
            last_err = e

        raise RuntimeError(
            f"Faster Whisper 模型加载失败（{self._model_size}）。"
            f"请检查网络或改用 OpenAI 兼容 STT。详情: {last_err}"
        )

    def _transcribe_sync(self, audio_data: bytes) -> str:
        """在线程池中串行执行，禁止重入。"""
        import numpy as np

        if self._model is None or len(audio_data) < 1600:
            return ""
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        with self._transcribe_lock:
            # vad_filter 内部再加载 silero，可能拖慢/异常；短音频关掉更稳
            segments, info = self._model.transcribe(
                audio_np,
                language=self._language,
                beam_size=1,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            text = "".join(seg.text for seg in segments).strip()
            print(f"[STT] Faster Whisper 结果: {text!r} (lang={getattr(info, 'language', '?')})")
            return text

    async def start(self, *, language: str | None = None, sample_rate: int = 16000) -> None:
        self._sample_rate = sample_rate
        if language:
            self._language = language
        self._audio_buffer.clear()
        self._events_queue = asyncio.Queue()
        self._running = True
        self._ensure_executor()

        if self._model is None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self._load_model)

        # 只缓冲，不做并发 partial 转写，避免与 stop() 抢模型
        self._worker_task = asyncio.create_task(self._idle_worker())

    async def send_audio(self, chunk: bytes) -> None:
        if not self._running:
            return
        self._audio_buffer.extend(chunk)

    def events(self) -> AsyncIterator[STTEvent]:
        async def _gen():
            while True:
                evt = await self._events_queue.get()
                if evt is None:
                    return
                yield evt
        return _gen()

    async def _idle_worker(self) -> None:
        """占位 worker：仅保持会话，真正识别在 stop() 串行执行。"""
        try:
            while self._running:
                await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            pass

    async def stop(self) -> str:
        """停止识别会话。冲刷缓冲区，返回最终识别文本。"""
        self._running = False

        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

        final_text = ""
        audio_data = bytes(self._audio_buffer)
        self._audio_buffer.clear()

        if len(audio_data) >= 1600 and self._model is not None:
            try:
                loop = asyncio.get_running_loop()
                self._ensure_executor()
                print(f"[STT] Faster Whisper 开始转写: {len(audio_data)} bytes "
                      f"({len(audio_data) / (self._sample_rate * 2):.2f}s)")
                final_text = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, self._transcribe_sync, audio_data),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                print("[STT] Faster Whisper 转写超时(30s)")
                await self._events_queue.put(STTEvent(
                    type=STTEventType.ERROR,
                    provider=self.name,
                    text="Whisper transcribe timeout",
                ))
            except Exception as e:
                print(f"[STT] Faster Whisper 转写异常: {type(e).__name__}: {e}")
                await self._events_queue.put(STTEvent(
                    type=STTEventType.ERROR,
                    provider=self.name,
                    text=f"Whisper error: {type(e).__name__}",
                ))

        await self._events_queue.put(None)
        return final_text

    def __del__(self):
        try:
            if self._executor is not None:
                self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
