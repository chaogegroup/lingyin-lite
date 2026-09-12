"""
WebRTC 音频处理层。

使用 aiortc（Python WebRTC 库）实现浏览器与服务端之间的实时音频传输。
不需要单独部署 LiveKit 或其他 WebRTC 服务器——FastAPI 同时承担信令和媒体处理。

工作流程：
1. 浏览器打开页面，通过 WebSocket 发起 WebRTC 连接
2. 交换 SDP（会话描述）和 ICE candidate（网络候选）
3. 建立 WebRTC 连接后：
   - 浏览器采集麦克风音频，通过 WebRTC 发给服务端
   - 服务端把音频喂给 VoicePipeline
   - Pipeline 的 TTS 输出通过 WebRTC 发给浏览器播放

设计思想来自 Local Live 项目的 packages/livekit_bridge/，
但用 aiortc 替代 LiveKit，降低部署门槛。
"""

from __future__ import annotations

import asyncio
import json
import logging
from fractions import Fraction
from typing import Any

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay
from av import AudioFrame
from av.audio.resampler import AudioResampler
import numpy as np

from .pipeline import PipelineCallbacks, PipelineState, VoicePipeline

logger = logging.getLogger("webrtc")


# ============================================================
# 音频轨道：从 WebRTC 接收麦克风音频
# ============================================================

class MicAudioTrack:
    """
    封装从浏览器收到的音频轨道，转换为 PCM16 字节流供 Pipeline 使用。

    aiortc 收到的是 AudioFrame（包含 numpy 数组），需要转为 PCM16 bytes。
    """

    def __init__(self, track: MediaStreamTrack) -> None:
        self._track = track
        self._relay = MediaRelay()
        self._relayed = self._relay.subscribe(track)
        self._resampler = AudioResampler(format="s16", layout="mono", rate=16000)

    async def recv_pcm16(self) -> bytes | None:
        """
        接收一帧音频，返回 PCM16 bytes（16kHz 单声道）。

        返回值：
        - bytes：有效 PCM（可能为空，表示重采样器缓冲中）
        - None：轨道结束/异常
        """
        try:
            frame: AudioFrame = await self._relayed.recv()
        except Exception:
            return None

        try:
            # 统一重采样为 s16 / mono / 16kHz，避免把 int16 误当成 float 裁剪
            frames = self._resampler.resample(frame)
        except Exception as e:
            logger.warning(f"Audio resample failed: {e}")
            return b""

        out = bytearray()
        for f in frames:
            arr = f.to_ndarray()
            if arr.ndim > 1:
                arr = arr.reshape(-1)
            if arr.dtype != np.int16:
                arr = np.clip(arr, -1.0, 1.0)
                arr = (arr * 32767).astype(np.int16)
            out.extend(arr.tobytes())
        return bytes(out)


# ============================================================
# 音频轨道：TTS 输出通过 WebRTC 发给浏览器
# ============================================================

class TTSAudioTrack(MediaStreamTrack):
    """
    自定义音频轨道，把 Pipeline 的 TTS 输出通过 WebRTC 发给浏览器。

    aiortc 要求 MediaStreamTrack 实现 recv() 方法返回 AudioFrame。
    我们从一个 asyncio.Queue 读取 PCM16 数据，封装为 AudioFrame。

    使用 48kHz（WebRTC/Opus 常用采样率），避免 24kHz 帧在浏览器侧被截断或变速。
    """

    kind = "audio"

    def __init__(self, sample_rate: int = 48000) -> None:
        super().__init__()
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._sample_rate = sample_rate
        self._buffer = bytearray()
        self._pts = 0
        # 20ms 帧
        self._frame_samples = sample_rate // 50

    def push_audio(self, pcm16: bytes) -> None:
        """推入 PCM16 音频数据（由 Pipeline 回调调用）。"""
        if pcm16:
            self._queue.put_nowait(pcm16)

    def stop(self) -> None:
        """停止轨道。"""
        super().stop()
        self._queue.put_nowait(None)

    async def recv(self) -> AudioFrame:
        """
        返回一帧音频。aiortc 会持续调用此方法获取音频数据。

        如果队列为空，返回静音帧（保持连接活跃）。
        """
        frame_bytes = self._frame_samples * 2  # int16

        while len(self._buffer) < frame_bytes:
            try:
                data = await asyncio.wait_for(self._queue.get(), timeout=0.02)
            except asyncio.TimeoutError:
                # 超时，填充静音补齐本帧（不要丢掉已有数据）
                need = frame_bytes - len(self._buffer)
                self._buffer.extend(b"\x00" * need)
                break
            if data is None:
                need = frame_bytes - len(self._buffer)
                self._buffer.extend(b"\x00" * max(need, 0))
                break
            self._buffer.extend(data)

        chunk = bytes(self._buffer[:frame_bytes])
        self._buffer = self._buffer[frame_bytes:]

        audio = np.frombuffer(chunk, dtype=np.int16)
        frame = AudioFrame.from_ndarray(audio.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = self._sample_rate
        frame.pts = self._pts
        frame.time_base = Fraction(1, self._sample_rate)
        self._pts += len(audio)

        return frame


# ============================================================
# WebRTC 连接管理器
# ============================================================

class WebRTCManager:
    """
    管理 WebRTC 连接，将音频流与 VoicePipeline 对接。

    每个浏览器连接创建一个 PeerConnection 和一个 VoicePipeline 实例。
    """

    def __init__(self, pipeline_factory: Any) -> None:
        """
        Args:
            pipeline_factory: 调用工厂函数，返回 (VoicePipeline, config)
        """
        self._pipeline_factory = pipeline_factory
        self._connections: dict[str, dict[str, Any]] = {}

    async def handle_offer(
        self,
        offer_sdp: str,
        offer_type: str = "offer",
        websocket: Any = None,
    ) -> dict[str, str]:
        """
        处理浏览器发来的 SDP offer，返回 answer。

        这是 WebRTC 信令的核心：收到 offer → 创建 answer → 返回。
        """
        pc = RTCPeerConnection()
        conn_id = str(id(pc))

        # 创建 TTS 输出轨道
        tts_track = TTSAudioTrack()
        pc.addTrack(tts_track)

        # 创建 Pipeline
        pipeline, config = self._pipeline_factory()

        # 本段回复是否已开始推送音频
        audio_active = False
        audio_bytes_total = 0

        async def send_ws(payload: dict) -> None:
            if websocket is None:
                return
            try:
                await websocket.send_json(payload)
            except Exception as e:
                logger.debug(f"WebSocket send failed: {e}")

        async def send_pcm(pcm: bytes) -> None:
            if websocket is None or not pcm:
                return
            try:
                await websocket.send_bytes(pcm)
            except Exception as e:
                logger.debug(f"WebSocket send_bytes failed: {e}")

        async def on_stt_partial(text: str) -> None:
            await send_ws({"type": "stt_partial", "text": text})

        async def on_stt_final(text: str) -> None:
            await send_ws({"type": "stt_final", "text": text})

        async def on_llm_chunk(text: str) -> None:
            await send_ws({"type": "llm_chunk", "text": text})

        async def on_llm_done() -> None:
            nonlocal audio_active, audio_bytes_total
            await send_ws({"type": "llm_done"})
            if audio_active:
                await send_ws({"type": "audio_end"})
                print(f"[TTS] 本段推送完成: {audio_bytes_total} bytes (~{audio_bytes_total / (24000 * 2):.2f}s @24kHz)")
                audio_active = False
                audio_bytes_total = 0

        async def on_state(state: PipelineState) -> None:
            nonlocal audio_active
            await send_ws({"type": "state", "state": state.value})
            logger.debug(f"Connection {conn_id} state: {state}")
            # 用户开始说话 / 新一轮生成 / 打断时，停掉浏览器尚未播完的 TTS
            # 注意：idle 表示合成结束，不能停播，否则尾音会被掐掉
            if state.value in ("listening", "thinking", "interrupted"):
                await send_ws({"type": "audio_stop"})
                audio_active = False

        async def on_interrupt() -> None:
            nonlocal audio_active
            await send_ws({"type": "interrupt"})
            await send_ws({"type": "audio_stop"})
            audio_active = False

        async def on_error(msg: str) -> None:
            logger.error(f"Pipeline error: {msg}")
            await send_ws({"type": "error", "message": msg})

        async def on_audio(audio: bytes, sample_rate: int = 24000, audio_format: str = "pcm") -> None:
            """
            TTS 音频优先通过 WebSocket 发给浏览器 Web Audio 播放。

            WebRTC 轨道只保持静音占位，避免 Opus 链路导致播放被截断。
            """
            nonlocal audio_active, audio_bytes_total
            if not audio:
                return
            try:
                pcm16 = prepare_tts_pcm(
                    audio,
                    source_rate=sample_rate or 24000,
                    audio_format=audio_format or "pcm",
                )
                if not pcm16:
                    print(f"[TTS] 解码结果为空，跳过 fmt={audio_format} raw={len(audio)}")
                    return

                rate = sample_rate or 24000
                if not audio_active:
                    await send_ws({
                        "type": "audio_start",
                        "sample_rate": rate,
                        "format": "pcm16",
                    })
                    audio_active = True
                    audio_bytes_total = 0

                await send_pcm(pcm16)
                audio_bytes_total += len(pcm16)
                duration = len(pcm16) / (rate * 2)
                print(f"[TTS] WebSocket 推送: {len(pcm16)} bytes, ~{duration:.2f}s @ {rate}Hz fmt={audio_format}")

                # WebRTC 轨道推静音占位
                tts_track.push_audio(b"\x00" * min(len(pcm16), 4800))
            except Exception as e:
                logger.error(f"TTS audio send error: {e}")

        callbacks = PipelineCallbacks(
            on_audio=on_audio,
            on_state_change=on_state,
            on_stt_partial=on_stt_partial,
            on_stt_final=on_stt_final,
            on_llm_chunk=on_llm_chunk,
            on_llm_done=on_llm_done,
            on_interrupt=on_interrupt,
            on_error=on_error,
        )
        pipeline.set_callbacks(callbacks)

        # 处理浏览器发来的音频轨道
        @pc.on("track")
        async def on_track(track):
            if track.kind == "audio":
                mic = MicAudioTrack(track)
                await pipeline.start()
                # 持续读取麦克风音频，喂给 Pipeline
                asyncio.create_task(self._audio_loop(mic, pipeline))

        @pc.on("connectionstatechange")
        async def on_state_change():
            logger.info(f"WebRTC state: {pc.connectionState}")
            if pc.connectionState == "closed":
                await pipeline.stop()
                self._connections.pop(conn_id, None)

        # 设置远程描述（offer）
        offer = RTCSessionDescription(sdp=offer_sdp, type=offer_type)
        await pc.setRemoteDescription(offer)

        # 创建并设置本地描述（answer）
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        # 保存连接
        self._connections[conn_id] = {
            "pc": pc,
            "pipeline": pipeline,
            "tts_track": tts_track,
            "websocket": websocket,
        }

        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "connection_id": conn_id,
        }

    async def _audio_loop(self, mic: MicAudioTrack, pipeline: VoicePipeline) -> None:
        """持续读取麦克风音频，喂给 Pipeline。"""
        while True:
            pcm16 = await mic.recv_pcm16()
            if pcm16 is None:
                break
            if not pcm16:
                continue
            await pipeline.feed_audio(pcm16)

    async def close_all(self) -> None:
        """关闭所有连接。"""
        for conn in self._connections.values():
            await conn["pc"].close()
            await conn["pipeline"].stop()
        self._connections.clear()

    def apply_config_to_all(self, **kwargs) -> int:
        """把热更新配置推到所有活跃连接，返回连接数。"""
        count = 0
        for conn in self._connections.values():
            pipeline = conn.get("pipeline")
            if pipeline is not None:
                pipeline.apply_runtime_config(**kwargs)
                count += 1
        return count


def prepare_tts_pcm(
    audio: bytes,
    *,
    source_rate: int = 24000,
    audio_format: str = "pcm",
) -> bytes:
    """把 TTS 输出整理为可直接播放的 PCM16 mono。"""
    if not audio:
        return b""

    fmt = (audio_format or "").lower()
    if fmt in ("mp3", "wav", "ogg", "m4a", "aac", "webm"):
        # 已知容器格式：强制解码（Edge TTS 帧头是 FFF3，不是 FFFB）
        return decode_audio_to_pcm16(audio, sample_rate=source_rate, force=True)

    is_container = (
        audio[:3] == b"ID3"
        or audio[:4] == b"RIFF"
        or audio[:4] == b"OggS"
        or (len(audio) >= 2 and audio[0] == 0xFF and (audio[1] & 0xE0) == 0xE0)
    )
    if is_container:
        return decode_audio_to_pcm16(audio, sample_rate=source_rate, force=True)
    return audio


def decode_audio_to_pcm16(
    audio_data: bytes,
    sample_rate: int = 24000,
    force: bool = False,
) -> bytes:
    """
    将音频数据（MP3/WAV/OGG 等）解码为 PCM16 bytes。

    Edge TTS 的 MP3 帧头常为 0xFFF3，已知格式必须 force 解码，
    否则会被当成裸 PCM 播成杂音。
    """
    import io
    import av

    if len(audio_data) < 4:
        return audio_data

    header = audio_data[:4]
    is_container = force or (
        header[:2] == b"\xff\xfb"
        or header[:3] == b"ID3"
        or header[:4] == b"RIFF"
        or header[:4] == b"OggS"
        or (audio_data[0] == 0xFF and (audio_data[1] & 0xE0) == 0xE0)
    )
    if not is_container:
        return audio_data

    input_buffer = io.BytesIO(audio_data)
    output_buffer = io.BytesIO()

    try:
        container = av.open(input_buffer)
        resampler = av.audio.resampler.AudioResampler(
            format="s16",
            layout="mono",
            rate=sample_rate,
        )

        for frame in container.decode(audio=0):
            frame.pts = None
            resampled = resampler.resample(frame)
            for r in resampled:
                output_buffer.write(r.to_ndarray().tobytes())

        try:
            for r in resampler.resample(None):
                output_buffer.write(r.to_ndarray().tobytes())
        except Exception:
            pass

        container.close()
        pcm = output_buffer.getvalue()
        if not pcm:
            logger.warning("Audio decode produced empty PCM")
        return pcm
    except Exception as e:
        logger.warning(f"Audio decode failed ({e})")
        # 强制模式绝不把压缩数据当 PCM 播出去
        return b"" if force else audio_data


# ============================================================
# ICE candidate 处理
# ============================================================

async def handle_ice_candidate(
    pc: RTCPeerConnection,
    candidate: dict[str, Any],
) -> None:
    """
    处理浏览器发来的 ICE candidate。

    ICE candidate 是 WebRTC 的网络地址信息，需要双方交换才能建立连接。
    """
    from aiortc import RTCIceCandidate

    ice_candidate = RTCIceCandidate(
        component=candidate.get("component", 1),
        foundation=candidate.get("foundation", ""),
        ip=candidate.get("ip", ""),
        port=candidate.get("port", 0),
        priority=candidate.get("priority", 0),
        protocol=candidate.get("protocol", "udp"),
        type=candidate.get("type", "host"),
    )
    await pc.addIceCandidate(ice_candidate)


# ============================================================
# 音频解码辅助
# ============================================================

def resample_pcm16(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """线性重采样 PCM16 mono。"""
    if not pcm or src_rate == dst_rate:
        return pcm
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return pcm
    dst_len = max(1, int(round(samples.size * dst_rate / src_rate)))
    indices = np.linspace(0, samples.size - 1, dst_len)
    resampled = np.interp(indices, np.arange(samples.size), samples)
    return np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()

