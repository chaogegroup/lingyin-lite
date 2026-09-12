"""
VAD（Voice Activity Detection，语音活动检测）模块。

功能：判断用户什么时候开始说话、什么时候停止说话。
这是实时语音对话的关键组件——没有 VAD，系统不知道何时把音频发给 STT，
也不知道何时停止收听并开始回复。

精简版使用 silero-vad：
- 轻量（单文件 ONNX 模型，约 2MB）
- 准确率高（工业级）
- 速度快（CPU 上实时）
- 支持 PCM16 输入

安装：pip install silero-vad
模型会自动下载。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable


class VADState(str, Enum):
    IDLE = "idle"              # 静默，等待说话
    SPEAKING = "speaking"      # 正在说话
    SILENCE = "silence"        # 说话暂停，可能结束


@dataclass
class VADConfig:
    """VAD 参数配置。"""
    threshold: float = 0.62         # 开始说话阈值
    min_speech_duration: float = 0.45  # 连续说满该时长才触发 speech_start
    min_silence_duration: float = 0.85  # 静默该秒数判定说完
    interrupt_threshold: float = 0.78  # 播放中打断阈值（更严）
    interrupt_min_duration: float = 0.8  # 播放中需持续说话时长（秒）
    pending_grace: float = 0.28     # 低于阈值的短暂间隙不重置累计
    sample_rate: int = 16000         # 采样率


class SileroVAD:
    """
    Silero VAD 封装。

    使用方法：
        vad = SileroVAD()
        vad.start()
        for audio_chunk in audio_stream:
            event = vad.feed(audio_chunk)
            if event == "speech_start":
                print("开始说话")
            elif event == "speech_end":
                print("结束说话")
    """

    def __init__(self, config: VADConfig | None = None) -> None:
        self.config = config or VADConfig()
        self._model = None
        self._state = VADState.IDLE
        self._speech_buffer = bytearray()
        self._silence_duration = 0.0
        self._speech_duration = 0.0
        self._pending_speech = 0.0
        self._pending_quiet = 0.0
        self._last_speech_prob = 0.0
        # Silero 16k 推理窗口：512 采样
        self._sample_buf = bytearray()
        self._silero_window = 512

    def start(self) -> None:
        """加载模型并重置状态。"""
        if self._model is None:
            self._load_model()
        self._state = VADState.IDLE
        self._speech_buffer.clear()
        self._silence_duration = 0.0
        self._speech_duration = 0.0
        self._pending_speech = 0.0
        self._pending_quiet = 0.0
        self._sample_buf.clear()

    def _load_model(self) -> None:
        """
        加载 Silero VAD。

        注意：Windows 中文路径下 torch.jit.load(path) 可能 fopen 失败，
        因此统一读成 bytes 再从内存加载。
        """
        # 1) silero_vad 官方包
        try:
            model = self._load_silero_from_package()
            if model is not None:
                self._model = model
                print("[VAD] Silero VAD 已加载（silero_vad 包 / 内存）")
                return
        except Exception as e:
            print(f"[VAD] silero_vad 包加载失败: {e}")

        # 2) torch.hub
        try:
            import torch
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
            )
            self._model = model
            self._get_speech_timestamps = utils[0]
            print("[VAD] Silero VAD 已加载（torch.hub）")
            return
        except Exception as e:
            print(f"[VAD] Silero 加载失败，退化为能量检测: {e}")
            self._model = None

    @staticmethod
    def _load_silero_from_package():
        """从已安装的 silero_vad 包读取 JIT 模型（兼容中文路径）。"""
        import io
        from pathlib import Path

        try:
            import silero_vad
            pkg_dir = Path(silero_vad.__file__).resolve().parent
        except Exception:
            return None

        jit_path = pkg_dir / "data" / "silero_vad.jit"
        if not jit_path.exists():
            return None

        import torch
        # Python 读文件支持中文路径；再交给 torch 从内存加载
        return torch.jit.load(io.BytesIO(jit_path.read_bytes()), map_location="cpu")

    def feed(self, audio_chunk: bytes) -> str | None:
        """
        输入 PCM16 音频块，返回 VAD 事件。

        返回值：
        - "speech_start"：检测到说话开始
        - "speech_end"：检测到说话结束
        - None：状态无变化
        """
        if not audio_chunk:
            return None

        chunk_duration = len(audio_chunk) / (self.config.sample_rate * 2)
        speech_prob = self._detect_speech(audio_chunk)
        self._last_speech_prob = speech_prob

        event = None

        if self._state == VADState.IDLE:
            if speech_prob > self.config.threshold:
                self._pending_speech += chunk_duration
                self._pending_quiet = 0.0
                if self._pending_speech >= self.config.min_speech_duration:
                    self._state = VADState.SPEAKING
                    self._speech_duration = self._pending_speech
                    self._silence_duration = 0.0
                    self._pending_speech = 0.0
                    self._pending_quiet = 0.0
                    event = "speech_start"
            else:
                # 允许短暂低于阈值，避免一个气口就清零
                self._pending_quiet += chunk_duration
                if self._pending_quiet >= self.config.pending_grace:
                    self._pending_speech = 0.0

        elif self._state == VADState.SPEAKING:
            self._speech_duration += chunk_duration
            if speech_prob < self.config.threshold:
                self._silence_duration += chunk_duration
                if self._silence_duration >= self.config.min_silence_duration:
                    if self._speech_duration >= self.config.min_speech_duration:
                        self._state = VADState.IDLE
                        event = "speech_end"
                    else:
                        self._state = VADState.IDLE
                        self._speech_duration = 0.0
            else:
                self._silence_duration = 0.0

        return event

    def is_strong_interrupt(self) -> bool:
        """播放中是否满足明确打断条件。"""
        if self._state != VADState.SPEAKING:
            return False
        return (
            self._last_speech_prob >= self.config.interrupt_threshold
            and self._speech_duration >= self.config.interrupt_min_duration
        )

    def _detect_speech(self, audio_chunk: bytes) -> float:
        """检测音频块的语音概率（0-1）。"""
        if self._model is not None:
            try:
                import torch
                import numpy as np
                self._sample_buf.extend(audio_chunk)
                window_bytes = self._silero_window * 2
                if len(self._sample_buf) < window_bytes:
                    # 窗口未满，用能量粗判，避免卡死
                    return self._energy_prob(audio_chunk)

                # 取满窗数据推理，剩余保留
                raw = bytes(self._sample_buf[:window_bytes])
                del self._sample_buf[:window_bytes]
                audio_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                audio_tensor = torch.from_numpy(audio_np)
                with torch.no_grad():
                    out = self._model(audio_tensor, self.config.sample_rate)
                if hasattr(out, "item"):
                    return float(out.item())
                return float(out)
            except Exception:
                pass

        return self._energy_prob(audio_chunk)

    def _energy_prob(self, audio_chunk: bytes) -> float:
        """退化方案：基于 RMS 能量的简单检测。"""
        import array
        samples = array.array("h")
        samples.frombytes(audio_chunk[: len(audio_chunk) - len(audio_chunk) % 2])
        if not samples:
            return 0.0
        mean_sq = sum(s * s for s in samples) / len(samples)
        rms = (mean_sq ** 0.5) / 32768.0
        return min(1.0, rms * 20.0)

    @property
    def state(self) -> VADState:
        return self._state

    @property
    def last_speech_prob(self) -> float:
        return self._last_speech_prob

    def reset(self) -> None:
        """重置 VAD 状态（用于打断后重新开始监听）。"""
        self._state = VADState.IDLE
        self._speech_buffer.clear()
        self._silence_duration = 0.0
        self._speech_duration = 0.0
        self._pending_speech = 0.0
        self._pending_quiet = 0.0
