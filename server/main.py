"""
FastAPI 入口 —— 实时语音 AI 助手精简版。

启动方式：
    uvicorn server.main:app --host 0.0.0.0 --port 8000

然后浏览器打开 http://localhost:8000 即可使用。

功能：
- WebRTC 实时音频传输（浏览器麦克风采集 + 扬声器播放）
- VAD 语音活动检测
- 流式 STT（语音转文字）
- 流式 LLM（大模型对话）
- 流式 TTS（文字转语音）
- 打断机制（用户说话时立即停止回复）
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .llm import LLMProvider, OpenAICompatibleLLM, OllamaLLM
from .pipeline import PipelineConfig, VoicePipeline
from .stt import FasterWhisperSTT, OpenAICompatibleSTT, STTProvider
from .tts import EdgeTTS, OpenAICompatibleTTS, TTSProvider
from .webrtc import WebRTCManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("main")

# 国内环境默认 HuggingFace 镜像（faster-whisper 下载模型用）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


# ============================================================
# 配置加载
# ============================================================

def load_config() -> dict[str, Any]:
    """加载 config.yaml，如果不存在则用 config.example.yaml。"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    example_path = Path(__file__).parent.parent / "config.example.yaml"

    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    elif example_path.exists():
        logger.warning("config.yaml 不存在，使用 config.example.yaml")
        with open(example_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    else:
        logger.warning("配置文件不存在，使用默认配置")
        return {}


CONFIG = load_config()


# ============================================================
# Provider 工厂
# ============================================================

def create_stt_provider() -> STTProvider:
    """根据配置创建 STT Provider。"""
    stt_config = CONFIG.get("stt", {})
    provider_type = stt_config.get("provider", "openai")

    if provider_type == "openai":
        return OpenAICompatibleSTT(
            api_key=stt_config.get("api_key", os.environ.get("OPENAI_API_KEY", "")),
            base_url=stt_config.get("base_url", "https://api.openai.com/v1"),
            model=stt_config.get("model", "whisper-1"),
            language=stt_config.get("language", "zh"),
        )
    elif provider_type == "faster-whisper":
        return FasterWhisperSTT(
            model_size=stt_config.get("model_size", "base"),
            device=stt_config.get("device", "auto"),
            compute_type=stt_config.get("compute_type", "int8"),
            language=stt_config.get("language", "zh"),
        )
    else:
        raise ValueError(f"未知 STT provider: {provider_type}")


def create_llm_provider() -> LLMProvider:
    """根据配置创建 LLM Provider。"""
    llm_config = CONFIG.get("llm", {})
    provider_type = llm_config.get("provider", "openai")

    if provider_type == "openai":
        return OpenAICompatibleLLM(
            api_key=llm_config.get("api_key", os.environ.get("OPENAI_API_KEY", "")),
            base_url=llm_config.get("base_url", "https://api.openai.com/v1"),
            model=llm_config.get("model", "gpt-4o-mini"),
            default_temperature=llm_config.get("temperature", 0.7),
            default_max_tokens=llm_config.get("max_tokens", 1024),
        )
    elif provider_type == "ollama":
        return OllamaLLM(
            base_url=llm_config.get("base_url", "http://localhost:11434"),
            model=llm_config.get("model", "qwen2.5:7b"),
            default_temperature=llm_config.get("temperature", 0.7),
            default_max_tokens=llm_config.get("max_tokens", 1024),
        )
    else:
        raise ValueError(f"未知 LLM provider: {provider_type}")


def create_tts_provider() -> TTSProvider:
    """根据配置创建 TTS Provider。"""
    tts_config = CONFIG.get("tts", {})
    provider_type = tts_config.get("provider", "edge-tts")

    if provider_type == "openai":
        return OpenAICompatibleTTS(
            api_key=tts_config.get("api_key", os.environ.get("OPENAI_API_KEY", "")),
            base_url=tts_config.get("base_url", "https://api.openai.com/v1"),
            model=tts_config.get("model", "tts-1"),
            voice=tts_config.get("voice", "alloy"),
            sample_rate=tts_config.get("sample_rate", 24000),
            mode=tts_config.get("mode", "speech"),
        )
    elif provider_type == "edge-tts":
        return EdgeTTS(
            voice=tts_config.get("voice", "zh-CN-XiaoxiaoNeural"),
            rate=tts_config.get("rate", "+0%"),
            volume=tts_config.get("volume", "+0%"),
            sample_rate=tts_config.get("sample_rate", 24000),
        )
    else:
        raise ValueError(f"未知 TTS provider: {provider_type}")


def create_pipeline() -> tuple[VoicePipeline, PipelineConfig]:
    """创建 VoicePipeline 实例。"""
    from .vad import VADConfig

    vad_raw = CONFIG.get("vad") or {}
    vad_config = VADConfig(
        threshold=float(vad_raw.get("threshold", 0.62)),
        min_speech_duration=float(vad_raw.get("min_speech_duration", 0.45)),
        min_silence_duration=float(vad_raw.get("min_silence_duration", 0.85)),
        interrupt_threshold=float(vad_raw.get("interrupt_threshold", 0.78)),
        interrupt_min_duration=float(vad_raw.get("interrupt_min_duration", 0.8)),
        pending_grace=float(vad_raw.get("pending_grace", 0.28)),
        sample_rate=int(CONFIG.get("sample_rate", 16000)),
    )

    pipeline_config = PipelineConfig(
        system_prompt=CONFIG.get("system_prompt", PipelineConfig.system_prompt),
        max_history=CONFIG.get("max_history", 10),
        language=CONFIG.get("language", "zh"),
        sample_rate=CONFIG.get("sample_rate", 16000),
        tts_voice=CONFIG.get("tts", {}).get("voice", "alloy"),
        interrupt_on_speech=CONFIG.get("interrupt_on_speech", True),
        vad_config=vad_config,
    )
    pipeline = VoicePipeline(
        stt=create_stt_provider(),
        llm=create_llm_provider(),
        tts=create_tts_provider(),
        config=pipeline_config,
    )
    return pipeline, pipeline_config


# ============================================================
# MP3 解码辅助
# ============================================================

def decode_mp3_to_pcm16(mp3_data: bytes, sample_rate: int = 24000) -> bytes:
    """
    将 MP3 字节解码为 PCM16 bytes。

    使用 av（PyAV，aiortc 的依赖）解码，不需要额外安装 ffmpeg。
    """
    import av

    input_buffer = io.BytesIO(mp3_data)
    output_buffer = io.BytesIO()

    container = av.open(input_buffer, format="mp3")
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

    container.close()
    return output_buffer.getvalue()


# ============================================================
# FastAPI 应用
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化，关闭时清理。"""
    logger.info("Starting Voice AI Assistant Lite...")
    app.state.webrtc_manager = WebRTCManager(create_pipeline)
    yield
    logger.info("Shutting down...")
    await app.state.webrtc_manager.close_all()


app = FastAPI(
    title="Voice AI Assistant Lite",
    description="实时语音 AI 助手精简版 —— 从 Local Live 项目提取的核心语音链路",
    version="1.0.0",
    lifespan=lifespan,
)

# 静态文件：前端页面
client_dir = Path(__file__).parent.parent / "client"
app.mount("/static", StaticFiles(directory=str(client_dir)), name="static")


@app.get("/")
async def index():
    """首页：返回测试页面。"""
    return FileResponse(str(client_dir / "index.html"))


@app.get("/health")
async def health():
    """健康检查。"""
    return {"status": "ok", "service": "voice-ai-assistant-lite"}


@app.get("/config")
async def get_config():
    """获取当前配置（隐藏 API Key）。"""
    safe_config = {}
    for key, value in CONFIG.items():
        if isinstance(value, dict):
            safe_config[key] = {
                k: ("***" if "key" in k.lower() or "token" in k.lower() else v)
                for k, v in value.items()
            }
        else:
            safe_config[key] = value
    return safe_config


def _mask_secret(value: str) -> str:
    if not value or len(value) < 8:
        return "***" if value else ""
    return value[:6] + "..." + value[-4:]


@app.get("/api/settings")
async def get_settings():
    """完整设置（API Key 脱敏，前端用 has_api_key 判断是否已配置）。"""
    out: dict[str, Any] = {
        "host": CONFIG.get("host", "0.0.0.0"),
        "port": CONFIG.get("port", 8000),
        "language": CONFIG.get("language", "zh"),
        "sample_rate": CONFIG.get("sample_rate", 16000),
        "system_prompt": CONFIG.get("system_prompt", ""),
        "max_history": CONFIG.get("max_history", 10),
        "interrupt_on_speech": CONFIG.get("interrupt_on_speech", True),
        "vad": {
            "threshold": 0.62,
            "min_speech_duration": 0.45,
            "min_silence_duration": 0.85,
            "interrupt_threshold": 0.78,
            "interrupt_min_duration": 0.8,
            "pending_grace": 0.28,
            **(CONFIG.get("vad") or {}),
        },
        "stt": dict(CONFIG.get("stt") or {}),
        "llm": dict(CONFIG.get("llm") or {}),
        "tts": dict(CONFIG.get("tts") or {}),
    }
    for section in ("stt", "llm", "tts"):
        key = out[section].get("api_key", "")
        out[section]["api_key"] = ""
        out[section]["api_key_masked"] = _mask_secret(key)
        out[section]["has_api_key"] = bool(key)
    return out


def _merge_section(old: dict, new: dict) -> dict:
    """合并配置段：空 api_key 保留旧值，避免界面清空后覆盖。"""
    merged = dict(old or {})
    for k, v in (new or {}).items():
        if k in ("api_key_masked", "has_api_key"):
            continue
        if k == "api_key" and (v is None or str(v).strip() == ""):
            continue
        if v is None:
            continue
        merged[k] = v
    return merged


def save_config(data: dict[str, Any]) -> None:
    """写回 config.yaml。"""
    path = Path(__file__).parent.parent / "config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, width=1000)


def hot_reload_runtime() -> dict[str, Any]:
    """根据全局 CONFIG 热更新所有活跃连接。"""
    from .vad import VADConfig

    vad_raw = CONFIG.get("vad") or {}
    vad_config = VADConfig(
        threshold=float(vad_raw.get("threshold", 0.62)),
        min_speech_duration=float(vad_raw.get("min_speech_duration", 0.45)),
        min_silence_duration=float(vad_raw.get("min_silence_duration", 0.85)),
        interrupt_threshold=float(vad_raw.get("interrupt_threshold", 0.78)),
        interrupt_min_duration=float(vad_raw.get("interrupt_min_duration", 0.8)),
        pending_grace=float(vad_raw.get("pending_grace", 0.28)),
        sample_rate=int(CONFIG.get("sample_rate", 16000)),
    )

    try:
        stt = create_stt_provider()
        llm = create_llm_provider()
        tts = create_tts_provider()
        provider_error = None
    except Exception as e:
        stt = llm = tts = None
        provider_error = str(e)
        logger.error(f"Provider recreate failed: {e}")

    manager: WebRTCManager = app.state.webrtc_manager
    applied = 0
    if provider_error is None:
        applied = manager.apply_config_to_all(
            system_prompt=CONFIG.get("system_prompt"),
            max_history=CONFIG.get("max_history"),
            interrupt_on_speech=CONFIG.get("interrupt_on_speech", True),
            tts_voice=CONFIG.get("tts", {}).get("voice"),
            vad_config=vad_config,
            stt=stt,
            llm=llm,
            tts=tts,
        )
    else:
        # Provider 失败时仍热更 VAD / 提示词
        applied = manager.apply_config_to_all(
            system_prompt=CONFIG.get("system_prompt"),
            max_history=CONFIG.get("max_history"),
            interrupt_on_speech=CONFIG.get("interrupt_on_speech", True),
            vad_config=vad_config,
        )

    return {
        "applied_connections": applied,
        "provider_error": provider_error,
    }


@app.put("/api/settings")
async def update_settings(payload: dict[str, Any]):
    """
    更新配置并热加载。

    payload 为部分配置，会与现有 config 合并后写盘并推送到活跃连接。
    """
    global CONFIG

    new_config = dict(CONFIG)
    for key in (
        "language", "sample_rate", "system_prompt", "max_history",
        "interrupt_on_speech", "host", "port",
    ):
        if key in payload and payload[key] is not None:
            new_config[key] = payload[key]

    if "vad" in payload and isinstance(payload["vad"], dict):
        new_config["vad"] = _merge_section(CONFIG.get("vad") or {}, payload["vad"])

    for section in ("stt", "llm", "tts"):
        if section in payload and isinstance(payload[section], dict):
            new_config[section] = _merge_section(CONFIG.get(section) or {}, payload[section])

    # 数值类型校验
    try:
        new_config["max_history"] = int(new_config.get("max_history", 10))
        new_config["sample_rate"] = int(new_config.get("sample_rate", 16000))
        if isinstance(new_config.get("vad"), dict):
            vad = new_config["vad"]
            for vk in (
                "threshold", "min_speech_duration", "min_silence_duration",
                "interrupt_threshold", "interrupt_min_duration", "pending_grace",
            ):
                if vk in vad:
                    vad[vk] = float(vad[vk])
        if isinstance(new_config.get("llm"), dict):
            llm = new_config["llm"]
            if "temperature" in llm:
                llm["temperature"] = float(llm["temperature"])
            if "max_tokens" in llm:
                llm["max_tokens"] = int(llm["max_tokens"])
        if isinstance(new_config.get("interrupt_on_speech"), str):
            new_config["interrupt_on_speech"] = new_config["interrupt_on_speech"].lower() in (
                "1", "true", "yes", "on",
            )
    except (TypeError, ValueError) as e:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": f"参数类型错误: {e}"},
        )

    CONFIG = new_config
    try:
        save_config(CONFIG)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": f"写入 config.yaml 失败: {e}"},
        )

    reload_info = hot_reload_runtime()
    logger.info(f"Settings hot-reloaded, applied={reload_info.get('applied_connections')}")

    return {
        "ok": True,
        "message": "配置已保存并热更新",
        "applied_connections": reload_info.get("applied_connections", 0),
        "provider_error": reload_info.get("provider_error"),
    }


@app.post("/api/settings/reload")
async def reload_settings():
    """仅从磁盘重载 config.yaml 并热更新。"""
    global CONFIG
    CONFIG = load_config()
    info = hot_reload_runtime()
    return {"ok": True, **info}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket 信令端点：处理 WebRTC 的 SDP 交换和 ICE candidate。

    消息格式（JSON）：
    - {"type": "offer", "sdp": "..."} → 返回 {"type": "answer", "sdp": "..."}
    - {"type": "ice", "candidate": {...}} → 添加 ICE candidate
    """
    await websocket.accept()
    pc = None

    try:
        while True:
            data = await websocket.receive_text()
            message = __import__("json").loads(data)
            msg_type = message.get("type")

            if msg_type == "offer":
                # 处理 SDP offer
                manager: WebRTCManager = app.state.webrtc_manager
                result = await manager.handle_offer(
                    offer_sdp=message["sdp"],
                    offer_type=message.get("sdp_type", "offer"),
                    websocket=websocket,
                )
                await websocket.send_json({
                    "type": "answer",
                    "sdp": result["sdp"],
                    "connection_id": result["connection_id"],
                })

            elif msg_type == "ice":
                # 处理 ICE candidate（简化处理，实际需要关联到具体 pc）
                # 精简版中，单连接场景下可以忽略 ICE 的复杂处理
                logger.debug("ICE candidate received")

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


# ============================================================
# 启动入口
# ============================================================

if __name__ == "__main__":
    import uvicorn

    host = CONFIG.get("host", "0.0.0.0")
    port = CONFIG.get("port", 8000)
    uvicorn.run(app, host=host, port=port)
