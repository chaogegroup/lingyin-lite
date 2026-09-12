# Lingyin Lite v1.0.0

## What

灵音（Local Live）开源精简版 — a hackable realtime voice AI engine: WebRTC → VAD → STT → LLM → TTS, with true barge-in.

## Highlights

- Browser WebRTC mic capture + WebSocket/Web Audio TTS playback (long replies not truncated)
- Silero VAD with 500ms pre-roll (less dropped sentence starts)
- STT: OpenAI-compatible Whisper or local faster-whisper
- LLM: any OpenAI-compatible API or Ollama
- TTS: Edge TTS (free) or OpenAI/chat TTS (MP3 force-decoded)
- Web settings panel with hot reload (providers / VAD / system prompt)
- One-click Windows / PowerShell / shell start scripts; project venv isolation

## Known limitations

- STT is utterance-level (not token-streaming ASR)
- Single-user / small-team oriented; no production multi-tenant gateway
- No long-term memory or agent tool-calling in this lite build
- Edge TTS requires network; fully offline TTS needs a self-hosted backend

## License

MIT
