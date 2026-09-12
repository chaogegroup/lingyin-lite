# Voice AI Assistant Lite

> **实时语音 AI 助手核心引擎 —— 能听、能想、能说、能被打断。**
>
> A minimal, hackable realtime voice AI engine: WebRTC audio in → VAD → STT → LLM → TTS → audio out, with true barge-in interruption.

---

## 这是什么

一个**可以直接运行、可以学习、可以二次开发**的实时语音对话引擎。

它实现了让 AI "能听、能想、能说、能被打断"的最小完整技术链路：

```
麦克风 → WebRTC → VAD(预回填) → STT → LLM → TTS → WebSocket → Web Audio → 扬声器
         ↑                                                                    ↓
         └──────────────────── 打断机制（说话即停） ──────────────────────────┘
```

没有臃肿框架、没有强制外部 SFU、没有黑盒。代码结构清晰，注释完整，适合在一个周末读完核心链路。

## 核心特性

| 能力 | 实现 |
|:---|:---|
| **实时语音对话** | 浏览器原生 WebRTC 采集麦克风；云端 API 典型延迟约 1–2 秒 |
| **语音活动检测** | Silero VAD（中文路径下内存加载）+ 500ms 预回填，减少开头丢字 |
| **语音识别 STT** | OpenAI 兼容 Whisper API / Faster-Whisper 本地离线 |
| **大语言模型 LLM** | 任意 OpenAI 兼容接口 + Ollama 本地模型，SSE 真流式 |
| **语音合成 TTS** | Edge TTS（免费）/ OpenAI TTS / Chat 模式；长回复不截断 |
| **自然打断 Barge-in** | AI 说话时插话 → 停止生成与发声 → 立即重新监听 |
| **设置热更新** | 网页设置面板改 Provider / VAD / 提示词，保存即生效 |
| **Provider 抽象层** | STT/LLM/TTS 统一接口，配置切换，不改核心代码 |

## 快速开始

### 方式一：一键启动（推荐）

**Windows：** 双击 `启动.bat`

```powershell
# 或 PowerShell
powershell -ExecutionPolicy Bypass -File ".\启动.ps1"
```

**macOS / Linux：**

```bash
chmod +x start.sh
./start.sh
```

一键脚本会：检查 Python → 创建项目 venv → 安装依赖 → 生成配置 → 自检 → 启动并打开浏览器。

### 方式二：手动启动

```bash
# 1. 克隆
git clone https://github.com/chaogegroup/voice-ai-assistant-lite.git
cd voice-ai-assistant-lite

# 2. 依赖（Python 3.10+）
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

# 3. 配置
cp config.example.yaml config.yaml
# 编辑 config.yaml 填入 API Key（本地 Provider 可不填 Key）

# 4. 启动
python -m server.main
# 浏览器打开 http://localhost:8000
```

首次运行后也可在网页「设置」里改 Provider 并热更新。

## 配置一瞥

```yaml
stt:
  provider: "openai"          # openai | faster-whisper
  # api_key / base_url / model
  # 或本地: model_size: base

llm:
  provider: "openai"          # openai | ollama
  base_url: "https://api.deepseek.com/v1"
  model: "deepseek-chat"

tts:
  provider: "edge-tts"        # edge-tts | openai
  voice: "zh-CN-XiaoxiaoNeural"
```

已验证可接：DeepSeek、SiliconFlow、智谱、月之暗面、Ollama 等 OpenAI 兼容接口。

## 架构速览

```
浏览器                         FastAPI 服务端
┌────────────┐               ┌──────────────────────┐
│ 麦克风 WebRTC │ ──音频──►    │ MicAudioTrack (16k)  │
│ Web Audio    │ ◄──PCM──WS── │ STT → LLM → TTS      │
│ 对话 UI/Orb  │ ◄──JSON────  │ 状态机 + 打断          │
└────────────┘               └──────────────────────┘
```

状态机：`IDLE → LISTENING → THINKING → SPEAKING →（可 INTERRUPTED）→ IDLE`。

技术栈：Python 3.10+ · FastAPI · aiortc · Silero VAD · PyAV · Edge TTS / faster-whisper。

详见 [docs/architecture.md](docs/architecture.md)。

## 项目结构

```
voice-ai-assistant-lite/
├── README.md
├── LICENSE
├── requirements.txt
├── config.example.yaml
├── 启动.bat / 启动.ps1 / start.sh
├── run.py
├── server/          # FastAPI + Pipeline + Providers
├── client/          # Web UI（对话 / Orb / 设置）
└── docs/            # 架构说明与截图
```

## 适合谁

- 想快速搭语音对话原型的开发者
- 想在自己产品里集成语音能力的独立开发者
- 想吃透实时语音系统架构的学习者

## 限制（请先读）

- **非成品 App**：需要会装 Python、会改配置
- **单用户/小团队向**：无生产级多用户网关与高并发治理
- **STT 非逐字流式**：整句识别后再进 LLM（本地/标准 Whisper API）
- **记忆与 Agent**：本 lite 版不含长期记忆、工具调用

## 常见问题

**Q: 可以商用吗？**  
A: 可以。MIT 协议。第三方 API 遵守各自服务条款。

**Q: 和普通语音 demo 的区别？**  
A: 支持真打断（barge-in），并可在网页热切换云端/本地 Provider。

**Q: 能完全离线吗？**  
A: STT 可用 faster-whisper，LLM 可用 Ollama。Edge TTS 需网络（免费）；彻底离线需自建 TTS。

**Q: 开源了为什么还有人卖 199 元服务包？**  
A: 代码开源免费；有人卖的是**教程 + 1对1 指导**，不是代码本身。

## License

MIT — 见 [LICENSE](LICENSE)

---

如果这个项目帮到了你，欢迎点 Star。
