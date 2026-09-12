"""
语音对话 Pipeline —— 全链路状态机。

这是整个项目的核心，负责串联：
    音频输入 → VAD → STT → LLM → TTS → 音频输出

并实现**打断机制**：用户说话时立即停止当前播放和生成。

设计思想来自 Local Live 项目的 packages/livekit_bridge/llm.py：
- 状态机驱动（IDLE → LISTENING → THINKING → SPEAKING）
- 流式串联（STT 输出喂给 LLM，LLM 输出喂给 TTS）
- 可中断（asyncio.Task 取消 + 状态重置）

本模块不直接依赖 WebRTC，通过回调接口与音频层解耦。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

from .llm import ChatMessage, LLMChunk, LLMProvider, LLMRequest
from .stt import STTEvent, STTEventType, STTProvider
from .tts import TTSChunk, TTSProvider
from .vad import SileroVAD, VADConfig, VADState

logger = logging.getLogger("voice_pipeline")


# ============================================================
# 状态定义
# ============================================================

class PipelineState(str, Enum):
    IDLE = "idle"              # 空闲，等待用户说话
    LISTENING = "listening"    # 正在监听（VAD 检测到说话）
    THINKING = "thinking"      # STT 完成，正在调用 LLM
    SPEAKING = "speaking"      # LLM 完成，正在 TTS 播放
    INTERRUPTED = "interrupted"  # 被打断（瞬态，立即回到 LISTENING）


# ============================================================
# 回调接口
# ============================================================

@dataclass
class PipelineCallbacks:
    """
    Pipeline 与外部（WebRTC/UI）的回调接口。

    所有回调都是 async 的，由 Pipeline 在对应事件发生时调用。
    """
    # 音频输出：TTS 合成的音频块通过此回调发送给播放器
    # 签名: (audio: bytes, sample_rate: int, audio_format: str)
    on_audio: Callable[..., Awaitable[None]] | None = None
    # 状态变化：用于 UI 显示当前状态
    on_state_change: Callable[[PipelineState], Awaitable[None]] | None = None
    # STT 中间结果：用于实时显示识别文字
    on_stt_partial: Callable[[str], Awaitable[None]] | None = None
    # STT 最终结果
    on_stt_final: Callable[[str], Awaitable[None]] | None = None
    # LLM 输出：用于实时显示回复文字
    on_llm_chunk: Callable[[str], Awaitable[None]] | None = None
    # LLM 完成
    on_llm_done: Callable[[], Awaitable[None]] | None = None
    # 打断事件
    on_interrupt: Callable[[], Awaitable[None]] | None = None
    # 错误事件
    on_error: Callable[[str], Awaitable[None]] | None = None


# ============================================================
# Pipeline 配置
# ============================================================

@dataclass
class PipelineConfig:
    """Pipeline 配置。"""
    system_prompt: str = (
        "你是一个友好、简洁的语音助手。"
        "回答要口语化、简短，适合语音播报。"
        "不要使用 markdown 格式，不要输出代码块。"
    )
    max_history: int = 10           # 最大对话轮数（每轮=user+assistant）
    language: str = "zh"            # STT 语言
    sample_rate: int = 16000        # 音频采样率
    tts_voice: str = "alloy"        # TTS 声音
    interrupt_on_speech: bool = True  # 用户说话时是否打断当前回复
    vad_config: VADConfig = field(default_factory=VADConfig)


# ============================================================
# 核心 Pipeline
# ============================================================

class VoicePipeline:
    """
    实时语音对话 Pipeline。

    使用方法：
        pipeline = VoicePipeline(stt, llm, tts, config)
        pipeline.set_callbacks(callbacks)
        await pipeline.start()

        # 音频输入（从麦克风/WebRTC 收到的 PCM16 数据）
        await pipeline.feed_audio(audio_chunk)

        # 停止
        await pipeline.stop()
    """

    def __init__(
        self,
        stt: STTProvider,
        llm: LLMProvider,
        tts: TTSProvider,
        config: PipelineConfig | None = None,
    ) -> None:
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._config = config or PipelineConfig()
        self._callbacks = PipelineCallbacks()

        self._vad = SileroVAD(self._config.vad_config)
        self._state = PipelineState.IDLE
        self._history: list[ChatMessage] = []

        self._stt_task: asyncio.Task | None = None
        self._llm_task: asyncio.Task | None = None
        self._tts_task: asyncio.Task | None = None
        self._listen_timeout_task: asyncio.Task | None = None
        self._audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

        self._running = False
        self._current_user_text = ""
        self._stopping = False
        self._session_start_time = 0.0
        # 说话前音频预回填（解决开头丢字）：保留最近 ~500ms
        self._preroll_ms = 500
        self._preroll = bytearray()
        self._preroll_max_bytes = max(1, int(self._config.sample_rate * 2 * self._preroll_ms / 1000))

    # --------------------------------------------------------
    # 生命周期
    # --------------------------------------------------------

    def set_callbacks(self, callbacks: PipelineCallbacks) -> None:
        self._callbacks = callbacks

    def apply_runtime_config(
        self,
        *,
        system_prompt: str | None = None,
        max_history: int | None = None,
        interrupt_on_speech: bool | None = None,
        vad_config: VADConfig | None = None,
        tts_voice: str | None = None,
        stt: STTProvider | None = None,
        llm: LLMProvider | None = None,
        tts: TTSProvider | None = None,
    ) -> None:
        """热更新运行中 Pipeline 的配置与 Provider（下一轮对话生效）。"""
        if system_prompt is not None:
            self._config.system_prompt = system_prompt
        if max_history is not None:
            self._config.max_history = max_history
        if interrupt_on_speech is not None:
            self._config.interrupt_on_speech = interrupt_on_speech
        if tts_voice is not None:
            self._config.tts_voice = tts_voice
        if vad_config is not None:
            self._config.vad_config = vad_config
            # 原地更新 VAD，无需重建
            self._vad.config = vad_config
        if stt is not None:
            self._stt = stt
        if llm is not None:
            self._llm = llm
        if tts is not None:
            self._tts = tts
        logger.info("Pipeline runtime config applied")

    async def start(self) -> None:
        """启动 Pipeline，开始监听。"""
        if self._running:
            return
        self._running = True
        self._vad.start()
        self._session_start_time = time.time()
        await self._set_state(PipelineState.IDLE)
        logger.info("Pipeline started")

    async def stop(self) -> None:
        """停止 Pipeline，取消所有任务。"""
        self._running = False
        for task in (self._stt_task, self._llm_task, self._tts_task, self._listen_timeout_task):
            if task and not task.done():
                task.cancel()
        try:
            await self._stt.stop()
        except Exception:
            pass
        await self._set_state(PipelineState.IDLE)
        logger.info("Pipeline stopped")

    # --------------------------------------------------------
    # 音频输入
    # --------------------------------------------------------

    async def feed_audio(self, audio_chunk: bytes) -> None:
        """
        输入 PCM16 音频块（从麦克风/WebRTC 收到）。

        这是 Pipeline 的主入口，所有音频数据都通过这里进入。
        """
        if not self._running or not audio_chunk:
            return

        # VAD 检测
        vad_event = self._vad.feed(audio_chunk)

        if vad_event == "speech_start":
            # 先取走「本帧之前」的预回填，再进入 LISTENING
            await self._on_speech_start()
        elif vad_event == "speech_end":
            # 本帧仍算语音尾部，先送 STT 再结束
            if self._state == PipelineState.LISTENING:
                try:
                    await self._stt.send_audio(audio_chunk)
                except Exception:
                    pass
            await self._on_speech_end()
            return

        if self._state == PipelineState.LISTENING:
            try:
                await self._stt.send_audio(audio_chunk)
            except Exception as e:
                logger.error(f"STT send_audio error: {e}")
        else:
            # 非监听阶段维护预回填，供下一次 speech_start 使用
            self._append_preroll(audio_chunk)

    def _append_preroll(self, chunk: bytes) -> None:
        self._preroll.extend(chunk)
        overflow = len(self._preroll) - self._preroll_max_bytes
        if overflow > 0:
            del self._preroll[:overflow]

    def _take_preroll(self) -> bytes:
        data = bytes(self._preroll)
        self._preroll.clear()
        return data

    # --------------------------------------------------------
    # VAD 事件处理
    # --------------------------------------------------------

    async def _on_speech_start(self) -> None:
        """检测到用户开始说话。"""
        logger.info("Speech start detected")

        # 如果正在播放回复，用更严格条件确认后打断
        if self._config.interrupt_on_speech and self._state in (
            PipelineState.THINKING,
            PipelineState.SPEAKING,
        ):
            asyncio.create_task(self._delayed_interrupt())
            return

        if self._state == PipelineState.IDLE:
            await self._start_listening()

    async def _delayed_interrupt(self) -> None:
        """
        延迟打断：要求持续且足够响的说话，降低环境音/回声误打断。
        """
        try:
            # 播放中打断更保守：等待更久，且 VAD 仍判定为强语音
            await asyncio.sleep(self._config.vad_config.interrupt_min_duration)
            if self._state not in (PipelineState.THINKING, PipelineState.SPEAKING):
                return
            if self._vad.is_strong_interrupt():
                logger.info("Strong speech interrupt accepted")
                await self._interrupt()
            else:
                logger.info("Interrupt rejected (not strong enough)")
        except asyncio.CancelledError:
            pass

    async def _on_speech_end(self) -> None:
        """检测到用户停止说话。"""
        logger.info("Speech end detected")
        if self._state == PipelineState.LISTENING:
            # 停止 STT，获取最终识别结果
            await self._stop_listening_and_think()

    # --------------------------------------------------------
    # 状态转换
    # --------------------------------------------------------

    async def _start_listening(self) -> None:
        """开始监听：启动 STT，并回填说话前音频，避免开头丢字。"""
        await self._set_state(PipelineState.LISTENING)
        self._current_user_text = ""
        self._stopping = False

        try:
            await self._stt.start(
                language=self._config.language,
                sample_rate=self._config.sample_rate,
            )
            # VAD 起跳前已有约 min_speech_duration 的语音，用预回填补上
            preroll = self._take_preroll()
            if preroll:
                await self._stt.send_audio(preroll)
                duration = len(preroll) / (self._config.sample_rate * 2)
                print(f"[PIPELINE] 预回填 {len(preroll)} bytes (~{duration:.2f}s) 到 STT")
            self._stt_task = asyncio.create_task(self._consume_stt_events())
            self._listen_timeout_task = asyncio.create_task(self._listen_timeout())
        except Exception as e:
            logger.error(f"STT start error: {e}")
            await self._emit_error(f"STT 启动失败: {e}")
            await self._set_state(PipelineState.IDLE)

    async def _listen_timeout(self) -> None:
        """监听超时兜底：避免 VAD 未检测到结束导致一直 LISTENING。"""
        try:
            await asyncio.sleep(15.0)
            if self._running and self._state == PipelineState.LISTENING and not self._stopping:
                logger.info("Listening timeout, forcing stop and think")
                await self._stop_listening_and_think()
        except asyncio.CancelledError:
            pass

    async def _stop_listening_and_think(self) -> None:
        """停止监听，启动 LLM 思考。"""
        if self._stopping:
            return
        self._stopping = True

        if self._listen_timeout_task and not self._listen_timeout_task.done():
            self._listen_timeout_task.cancel()
            self._listen_timeout_task = None

        # 本地 STT 转写可能要几秒，先切到 thinking，避免 UI 一直停在“聆听”
        await self._set_state(PipelineState.THINKING)

        # 停止 STT 并获取最终识别文本（在 stop 内冲刷缓冲区）
        try:
            final_text = await self._stt.stop()
        except Exception as e:
            logger.error(f"STT stop error: {e}")
            final_text = ""

        if final_text and final_text.strip():
            self._current_user_text = final_text

        # 关闭事件消费任务（避免重复触发）
        if self._stt_task and not self._stt_task.done():
            self._stt_task.cancel()
            try:
                await self._stt_task
            except asyncio.CancelledError:
                pass
            self._stt_task = None

        user_text = self._current_user_text.strip()
        if not user_text:
            logger.info("Empty transcript, back to IDLE")
            self._stopping = False
            await self._set_state(PipelineState.IDLE)
            return

        logger.info(f"User: {user_text}")
        await self._emit_stt_final(user_text)

        # 启动 LLM
        await self._start_thinking(user_text)

    async def _start_thinking(self, user_text: str) -> None:
        """启动 LLM 生成回复。"""
        await self._set_state(PipelineState.THINKING)

        # 构建消息
        messages: list[ChatMessage] = [
            {"role": "system", "content": self._config.system_prompt}
        ]
        messages.extend(self._history[-self._config.max_history * 2:])
        messages.append({"role": "user", "content": user_text})

        request = LLMRequest(messages=messages)

        self._llm_task = asyncio.create_task(self._run_llm_and_tts(request, user_text))

    async def _run_llm_and_tts(self, request: LLMRequest, user_text: str) -> None:
        """
        运行 LLM 流式生成，并将输出实时喂给 TTS。

        这是核心的流式串联逻辑：
        LLM 输出文本 → 按句子分段 → TTS 合成 → 音频输出
        """
        full_response = ""
        tts_text_queue: asyncio.Queue[str | None] = asyncio.Queue()

        # 启动 TTS 消费任务
        self._tts_task = asyncio.create_task(
            self._run_tts(tts_text_queue)
        )

        try:
            async for chunk in self._llm.stream(request):
                if chunk.text:
                    full_response += chunk.text
                    await self._emit_llm_chunk(chunk.text)

                    # 按句子分段喂给 TTS
                    tts_text_queue.put_nowait(chunk.text)

                if chunk.finish_reason:
                    break

            # 通知 TTS 输入结束
            await tts_text_queue.put(None)

            # 等待 TTS 完成
            if self._tts_task:
                await self._tts_task

            # 保存对话历史
            self._history.append({"role": "user", "content": user_text})
            self._history.append({"role": "assistant", "content": full_response})

            await self._emit_llm_done()
            await self._set_state(PipelineState.IDLE)
            logger.info(f"Assistant: {full_response[:100]}...")

        except asyncio.CancelledError:
            # 被打断，取消 TTS
            if self._tts_task and not self._tts_task.done():
                self._tts_task.cancel()
            await tts_text_queue.put(None)
            raise
        except Exception as e:
            logger.error(f"LLM/TTS error: {e}")
            await self._emit_error(f"生成回复失败: {e}")
            await self._set_state(PipelineState.IDLE)

    async def _run_tts(self, text_queue: asyncio.Queue[str | None]) -> None:
        """
        TTS 消费任务：从队列读取文本，合成音频，通过回调输出。

        采用"句子级分段合成"降低首字延迟：
        - 累积文本直到遇到句子结束符
        - 立即合成并输出
        - 同时继续接收 LLM 输出
        """
        await self._set_state(PipelineState.SPEAKING)

        buffer = ""
        sentence_endings = "。！？.!?；;\n"

        async def _text_generator():
            nonlocal buffer
            while True:
                item = await text_queue.get()
                if item is None:
                    if buffer.strip():
                        yield buffer
                        buffer = ""
                    return
                buffer += item
                # 遇到句子结束符，输出一段
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
                        yield sentence

        try:
            async for chunk in self._tts.synthesize_stream(
                _text_generator(),
                voice=self._config.tts_voice,
            ):
                if chunk.audio and self._callbacks.on_audio:
                    await self._callbacks.on_audio(
                        chunk.audio,
                        chunk.sample_rate,
                        chunk.audio_format,
                    )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"TTS error: {e}")
            await self._emit_error(f"语音合成失败: {e}")

    # --------------------------------------------------------
    # 打断机制
    # --------------------------------------------------------

    async def _interrupt(self) -> None:
        """
        打断当前回复：停止 TTS 播放，取消 LLM 生成，回到监听状态。

        这是实时语音对话的关键功能——用户随时可以插话。
        """
        logger.info("=== INTERRUPT ===")
        self._stopping = True
        await self._set_state(PipelineState.INTERRUPTED)

        # 取消 LLM 和 TTS 任务
        for task in (self._llm_task, self._tts_task, self._listen_timeout_task):
            if task and not task.done():
                task.cancel()

        # 清空音频队列（停止播放）
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # 重置 VAD 和 STT
        self._vad.reset()
        try:
            await self._stt.stop()
        except Exception:
            pass
        if self._stt_task and not self._stt_task.done():
            self._stt_task.cancel()
        self._stt_task = None
        self._current_user_text = ""

        if self._callbacks.on_interrupt:
            await self._callbacks.on_interrupt()

        # 立即开始监听新的输入
        self._stopping = False
        await self._start_listening()

    # --------------------------------------------------------
    # STT 事件消费
    # --------------------------------------------------------

    async def _consume_stt_events(self) -> None:
        """消费 STT 事件流，累积识别文本。"""
        try:
            async for event in self._stt.events():
                print(f"[STT-EVENT] {event.type}: {event.text}")
                if event.type == STTEventType.PARTIAL_TRANSCRIPT:
                    self._current_user_text = event.text
                    if self._callbacks.on_stt_partial:
                        await self._callbacks.on_stt_partial(event.text)
                elif event.type == STTEventType.FINAL_TRANSCRIPT:
                    self._current_user_text = event.text
                elif event.type == STTEventType.ERROR:
                    logger.error(f"STT error: {event.text}")
                    await self._emit_error(f"语音识别错误: {event.text}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"STT event consumer error: {e}")

    # --------------------------------------------------------
    # 辅助方法
    # --------------------------------------------------------

    async def _set_state(self, state: PipelineState) -> None:
        self._state = state
        if self._callbacks.on_state_change:
            await self._callbacks.on_state_change(state)

    async def _emit_stt_final(self, text: str) -> None:
        if self._callbacks.on_stt_final:
            await self._callbacks.on_stt_final(text)

    async def _emit_llm_chunk(self, text: str) -> None:
        if self._callbacks.on_llm_chunk:
            await self._callbacks.on_llm_chunk(text)

    async def _emit_llm_done(self) -> None:
        if self._callbacks.on_llm_done:
            await self._callbacks.on_llm_done()

    async def _emit_error(self, message: str) -> None:
        logger.error(message)
        if self._callbacks.on_error:
            await self._callbacks.on_error(message)

    @property
    def state(self) -> PipelineState:
        return self._state

    @property
    def history(self) -> list[ChatMessage]:
        return list(self._history)

    def clear_history(self) -> None:
        """清空对话历史。"""
        self._history.clear()
