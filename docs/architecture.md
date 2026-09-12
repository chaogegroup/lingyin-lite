# 架构说明

## 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                        浏览器 (Client)                        │
│  ┌──────────────┐          ┌──────────────────────────┐     │
│  │  麦克风采集    │          │  WebRTC PeerConnection   │     │
│  │  (getUserMedia)│─────────▶│  (发送音频/接收音频)     │     │
│  └──────────────┘          └──────────┬───────────────┘     │
│                                       │                       │
│  ┌──────────────┐                     │                       │
│  │  扬声器播放    │◀────────────────────┘                       │
│  │  (audio elem) │                                             │
│  └──────────────┘                                             │
└───────────────────────────────┬─────────────────────────────┘
                                │ WebRTC (SRTP/DTLS)
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                     服务端 (FastAPI + aiortc)                  │
│                                                               │
│  ┌─────────────┐   ┌─────────┐   ┌──────────────────────┐   │
│  │  WebSocket  │   │ WebRTC  │   │   VoicePipeline      │   │
│  │  (信令交换)  │──▶│ (音频)  │──▶│   (核心状态机)        │   │
│  └─────────────┘   └────┬────┘   └──────────┬───────────┘   │
│                          │                   │               │
│                          ▼                   ▼               │
│                   ┌─────────────┐    ┌──────────────┐       │
│                   │ MicAudioTrack│    │ TTSAudioTrack │       │
│                   │ (PCM16输出)  │    │ (PCM16输入)   │       │
│                   └──────┬──────┘    └──────┬───────┘       │
│                          │                  │                │
│                          ▼                  ▼                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                   Pipeline 内部                        │   │
│  │                                                      │   │
│  │  音频 → VAD → STT → LLM → TTS → 音频                  │   │
│  │         │     │     │     │                          │   │
│  │         └─────┴─────┴─────┴── 打断机制 ──────────┐    │   │
│  │                                                  │    │   │
│  │  Provider 抽象层:                                │    │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐            │    │   │
│  │  │  STT    │ │  LLM    │ │  TTS    │            │    │   │
│  │  │ Provider│ │ Provider│ │ Provider│            │    │   │
│  │  └─────────┘ └─────────┘ └─────────┘            │    │   │
│  │       │           │           │                  │    │   │
│  │  ┌────┴────┐ ┌────┴────┐ ┌────┴────┐            │    │   │
│  │  │OpenAI   │ │OpenAI   │ │Edge TTS │            │    │   │
│  │  │Whisper  │ │Compat   │ │(免费)   │            │    │   │
│  │  ├─────────┤ ├─────────┤ ├─────────┤            │    │   │
│  │  │Faster   │ │Ollama   │ │OpenAI   │            │    │   │
│  │  │Whisper  │ │(本地)   │ │TTS      │            │    │   │
│  │  └─────────┘ └─────────┘ └─────────┘            │    │   │
│  └──────────────────────────────────────────────────┘    │   │
└───────────────────────────────────────────────────────────┘
```

## 核心模块说明

### 1. main.py - FastAPI 入口

- 提供 HTTP 服务（静态文件 + API）
- 提供 WebSocket 信令端点（/ws）
- 加载配置，创建 Provider 实例
- 管理 WebRTC 连接生命周期

### 2. webrtc.py - WebRTC 音频处理

- `MicAudioTrack`: 从浏览器接收音频，转为 PCM16 bytes
- `TTSAudioTrack`: 自定义音频轨道，把 TTS 输出发给浏览器
- `WebRTCManager`: 管理 PeerConnection，与 Pipeline 对接
- `decode_audio_to_pcm16`: MP3→PCM 解码（TTS 输出通常是 MP3）

### 3. vad.py - 语音活动检测

- 使用 Silero VAD 模型（工业级，轻量）
- 维护状态机：IDLE → SPEAKING → SILENCE
- 输出 `speech_start` / `speech_end` 事件
- 退化方案：Silero 加载失败时用 RMS 能量检测

### 4. stt.py - 语音转文字

- `STTProvider` Protocol: 统一接口（start/send_audio/events/stop）
- `OpenAICompatibleSTT`: OpenAI 兼容接口（非流式，分段模拟）
- `FasterWhisperSTT`: 本地 Whisper（完全离线）
- 事件类型：START_OF_SPEECH / PARTIAL / FINAL / END_OF_SPEECH / ERROR

### 5. llm.py - 大语言模型

- `LLMProvider` Protocol: 统一接口（stream/close）
- `OpenAICompatibleLLM`: OpenAI 兼容接口（真正的 SSE 流式）
- `OllamaLLM`: 本地 Ollama（完全离线）
- 支持中断（asyncio.CancelledError 正确处理）

### 6. tts.py - 文字转语音

- `TTSProvider` Protocol: 统一接口（synthesize_stream/close）
- `OpenAICompatibleTTS`: OpenAI TTS（句子级分段合成降低延迟）
- `EdgeTTS`: 微软免费 TTS（音质好，中文声音丰富）
- 输入是文本流，输出是音频块流

### 7. pipeline.py - 核心状态机

这是整个项目的核心，负责串联所有模块。

#### 状态定义

| 状态 | 说明 |
|:---|:---|
| `IDLE` | 空闲，等待用户说话 |
| `LISTENING` | VAD 检测到说话，音频发给 STT |
| `THINKING` | STT 返回最终文本，正在调用 LLM |
| `SPEAKING` | LLM 返回文本，正在 TTS 播放 |
| `INTERRUPTED` | 被打断（瞬态，立即回到 LISTENING） |

#### 核心流程

```
1. 用户开始说话 (VAD speech_start)
   → 状态: IDLE → LISTENING
   → 启动 STT，持续发送音频

2. 用户停止说话 (VAD speech_end)
   → 停止 STT，获取识别文本
   → 状态: LISTENING → THINKING
   → 调用 LLM 流式生成

3. LLM 输出文本
   → 按句子分段喂给 TTS
   → 状态: THINKING → SPEAKING
   → TTS 合成音频，通过 WebRTC 播放

4. TTS 播放完成
   → 保存对话历史
   → 状态: SPEAKING → IDLE

5. 用户打断 (VAD speech_start，当前在 THINKING/SPEAKING)
   → 取消 LLM 任务
   → 取消 TTS 任务
   → 清空音频队列
   → 重置 VAD/STT
   → 状态: → INTERRUPTED → LISTENING
```

#### 打断机制实现

打断是实时语音对话的关键功能。实现要点：

1. **LLM 中断**：用 `asyncio.Task.cancel()` 取消生成任务，aiohttp 会正确关闭连接
2. **TTS 中断**：取消 TTS 任务，同时清空 `TTSAudioTrack` 的音频队列
3. **STT 重置**：停止当前 STT 会话，启动新的会话
4. **VAD 重置**：重置 VAD 状态，避免误判
5. **立即监听**：打断后立即开始监听新输入，不需要等当前回复完全停止

## 数据流

### 音频输入流

```
浏览器麦克风 → WebRTC → MicAudioTrack.recv_pcm16() → Pipeline.feed_audio()
                                                          ↓
                                                    VAD.feed() → speech_start/end
                                                          ↓
                                              (LISTENING时) STT.send_audio()
                                                          ↓
                                              STT.events() → 识别文本
```

### 音频输出流

```
LLM.stream() → 文本chunk → 按句子分段 → TTS.synthesize_stream()
                                                    ↓
                                              TTSChunk (MP3)
                                                    ↓
                                          decode_audio_to_pcm16()
                                                    ↓
                                              TTSAudioTrack.push_audio()
                                                    ↓
                                              WebRTC → 浏览器扬声器
```

## 设计决策

### 为什么用 aiortc 而不是 LiveKit？

Local Live 完整版用 LiveKit 做 WebRTC SFU，但精简版选择 aiortc：
- **部署简单**：不需要单独部署 LiveKit 服务器，FastAPI 同时承担信令和媒体
- **依赖少**：aiortc 是纯 Python 库，pip install 即可
- **适合教学**：代码更透明，容易理解 WebRTC 原理
- **单连接够用**：精简版不需要 SFU 的多房间/大规模转发能力

### 为什么 TTS 用句子级分段而不是字符合成？

- OpenAI TTS API 不支持真正的流式输入（一次请求返回完整音频）
- 句子级分段可以降低首字延迟（LLM 输出第一句话就开始合成）
- 同时避免了逐字合成导致的语音不连贯

### 为什么 VAD 在服务端而不是浏览器端？

- 服务端 VAD 可以复用 Silero 模型（准确率高）
- 浏览器端 VAD 通常用简单的能量检测，误判率高
- 服务端可以统一处理打断逻辑，前端更简单

## 性能指标（参考）

| 环节 | 延迟（本地测试） | 说明 |
|:---|:---|:---|
| VAD 检测 | < 100ms | Silero VAD，CPU 实时 |
| STT 首字 | 300-800ms | 取决于 API 和音频长度 |
| LLM 首字 | 200-500ms | 取决于模型和 API |
| TTS 首字 | 200-500ms | Edge TTS 较快 |
| **端到端** | **1-2秒** | 从说话结束到听到回复 |

> 本地部署（Ollama + faster-whisper + Edge TTS）延迟可能更高，取决于硬件。

## 扩展方向

如果要从精简版升级到生产级产品，需要增加：

1. **记忆系统**：长期记忆、用户画像、记忆蒸馏（参考 Local Live 的 packages/memory）
2. **工具调用**：让 AI 能执行操作（参考 Local Live 的 packages/tools）
3. **多用户管理**：用户认证、会话管理、限流
4. **监控告警**：延迟监控、错误追踪、用量统计
5. **生产级 WebRTC**：用 LiveKit 或 mediasoup 替代 aiortc
6. **完整 UI**：对话历史、设置面板、多语言
