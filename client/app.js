/**
 * 实时语音 AI 助手 - 前端
 * WebRTC 麦克风 + WebSocket 文本/TTS + 滚动对话 + Orb 状态动画
 */

// ============================================================
// 状态
// ============================================================

let peerConnection = null;
let websocket = null;
let localStream = null;
let remoteAudio = null;
let isRunning = false;

let appState = 'idle'; // idle | listening | thinking | speaking | interrupted
let chatHistory = [];   // {role, text, streaming?}
let assistantBuffer = '';
let liveAssistantIndex = -1;

// TTS Web Audio
let audioCtx = null;
let nextPlayTime = 0;
let activeSources = [];
let ttsSampleRate = 24000;

const statusText = document.getElementById('statusText');
const chatLog = document.getElementById('chatLog');
const chatEmpty = document.getElementById('chatEmpty');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const configInfo = document.getElementById('configInfo');

// ============================================================
// Orb 动画
// ============================================================

const orbCanvas = document.getElementById('orbCanvas');
const orbCtx = orbCanvas ? orbCanvas.getContext('2d') : null;
let orbT = 0;
let orbAmp = 0.15;
let orbTargetAmp = 0.15;

const ORB_STYLES = {
    idle:        { core: [61, 224, 255],  glow: [139, 92, 255],  amp: 0.12, speed: 0.8 },
    listening:   { core: [61, 224, 255],  glow: [0, 180, 255],   amp: 0.55, speed: 2.2 },
    thinking:    { core: [139, 92, 255],  glow: [180, 120, 255], amp: 0.35, speed: 1.4 },
    speaking:    { core: [61, 255, 176],  glow: [0, 220, 160],   amp: 0.7,  speed: 2.8 },
    interrupted: { core: [255, 193, 77],  glow: [255, 120, 80],  amp: 0.4,  speed: 1.8 },
};

function setAppState(state) {
    appState = state || 'idle';
    orbTargetAmp = (ORB_STYLES[appState] || ORB_STYLES.idle).amp;
    updateStatusPill();
}

function updateStatusPill() {
    const map = {
        idle: '点击「开始对话」开始语音交流',
        listening: '正在聆听…',
        thinking: '正在思考…',
        speaking: '正在回复…',
        interrupted: '已打断，继续聆听…',
    };
    statusText.textContent = map[appState] || map.idle;
    statusText.className = 'status-pill' + (appState !== 'idle' ? ' ' + appState : '');
}

function resizeOrbCanvas() {
    if (!orbCanvas) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cssSize = orbCanvas.clientWidth || 148;
    orbCanvas.width = Math.floor(cssSize * dpr);
    orbCanvas.height = Math.floor(cssSize * dpr);
    orbCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function drawOrb() {
    if (!orbCtx || !orbCanvas) return;
    const style = ORB_STYLES[appState] || ORB_STYLES.idle;
    orbAmp += (orbTargetAmp - orbAmp) * 0.08;
    orbT += 0.016 * style.speed;

    const w = orbCanvas.clientWidth || 148;
    const h = orbCanvas.clientHeight || 148;
    const cx = w / 2;
    const cy = h / 2;
    const baseR = w * 0.28;

    orbCtx.clearRect(0, 0, w, h);

    // 外发光
    const glow = orbCtx.createRadialGradient(cx, cy, baseR * 0.2, cx, cy, baseR * (1.6 + orbAmp * 0.4));
    glow.addColorStop(0, `rgba(${style.glow.join(',')},${0.35 + orbAmp * 0.25})`);
    glow.addColorStop(0.55, `rgba(${style.glow.join(',')},0.12)`);
    glow.addColorStop(1, 'rgba(0,0,0,0)');
    orbCtx.fillStyle = glow;
    orbCtx.beginPath();
    orbCtx.arc(cx, cy, baseR * (1.7 + orbAmp * 0.35), 0, Math.PI * 2);
    orbCtx.fill();

    // 波形环
    orbCtx.beginPath();
    const points = 72;
    for (let i = 0; i <= points; i++) {
        const a = (i / points) * Math.PI * 2;
        const n =
            Math.sin(a * 3 + orbT * 2.1) * 0.55 +
            Math.sin(a * 5 - orbT * 1.4) * 0.3 +
            Math.sin(a * 2 + orbT * 0.9) * 0.2;
        const r = baseR * (1 + orbAmp * n * 0.22);
        const x = cx + Math.cos(a) * r;
        const y = cy + Math.sin(a) * r;
        if (i === 0) orbCtx.moveTo(x, y);
        else orbCtx.lineTo(x, y);
    }
    orbCtx.closePath();
    orbCtx.strokeStyle = `rgba(${style.core.join(',')},${0.55 + orbAmp * 0.35})`;
    orbCtx.lineWidth = 2;
    orbCtx.stroke();

    // 内核
    const core = orbCtx.createRadialGradient(cx - baseR * 0.15, cy - baseR * 0.2, baseR * 0.1, cx, cy, baseR);
    core.addColorStop(0, `rgba(255,255,255,0.95)`);
    core.addColorStop(0.25, `rgba(${style.core.join(',')},0.95)`);
    core.addColorStop(1, `rgba(${style.glow.join(',')},0.55)`);
    orbCtx.fillStyle = core;
    orbCtx.beginPath();
    orbCtx.arc(cx, cy, baseR * (0.86 + orbAmp * 0.08), 0, Math.PI * 2);
    orbCtx.fill();

    // 高光
    orbCtx.beginPath();
    orbCtx.ellipse(cx - baseR * 0.28, cy - baseR * 0.32, baseR * 0.22, baseR * 0.12, -0.5, 0, Math.PI * 2);
    orbCtx.fillStyle = 'rgba(255,255,255,0.28)';
    orbCtx.fill();

    requestAnimationFrame(drawOrb);
}

window.addEventListener('resize', resizeOrbCanvas);
resizeOrbCanvas();
requestAnimationFrame(drawOrb);
setAppState('idle');

// ============================================================
// 对话滚动窗
// ============================================================

function ensureEmptyHidden() {
    if (!chatEmpty) return;
    chatEmpty.style.display = chatHistory.length ? 'none' : 'flex';
}

function scrollChatToBottom(force) {
    if (!chatLog) return;
    const nearBottom =
        chatLog.scrollHeight - chatLog.scrollTop - chatLog.clientHeight < 80;
    if (force || nearBottom) {
        chatLog.scrollTop = chatLog.scrollHeight;
    }
}

function renderChat() {
    if (!chatLog) return;
    ensureEmptyHidden();

    // 重建简单可靠（消息量通常不大）
    const frag = document.createDocumentFragment();
    chatHistory.forEach((msg, idx) => {
        const el = document.createElement('div');
        el.className = `msg ${msg.role}` + (msg.streaming ? ' streaming' : '');
        const roleLabel =
            msg.role === 'user' ? '你' :
            msg.role === 'assistant' ? 'AI' : '系统';
        el.innerHTML = `
            <div class="meta">${roleLabel}</div>
            <div class="bubble"></div>
        `;
        el.querySelector('.bubble').textContent = msg.text || (msg.streaming ? '' : '（空）');
        frag.appendChild(el);
        if (idx === chatHistory.length - 1) {
            // last
        }
    });

    // 保留 empty 节点
    chatLog.innerHTML = '';
    if (chatHistory.length === 0) {
        chatLog.appendChild(chatEmpty);
        chatEmpty.style.display = 'flex';
    } else {
        chatLog.appendChild(frag);
        scrollChatToBottom(true);
    }
}

function appendMessage(role, text, streaming) {
    chatHistory.push({ role, text: text || '', streaming: !!streaming });
    renderChat();
    return chatHistory.length - 1;
}

function updateMessage(index, text, streaming) {
    if (index < 0 || index >= chatHistory.length) return;
    chatHistory[index].text = text;
    chatHistory[index].streaming = !!streaming;

    const nodes = chatLog.querySelectorAll('.msg');
    const node = nodes[index];
    if (node) {
        const bubble = node.querySelector('.bubble');
        if (bubble) bubble.textContent = text || '';
        node.classList.toggle('streaming', !!streaming);
        scrollChatToBottom(false);
    } else {
        renderChat();
    }
}

function finishMessage(index) {
    if (index < 0 || index >= chatHistory.length) return;
    chatHistory[index].streaming = false;
    renderChat();
}

function clearChat() {
    chatHistory = [];
    assistantBuffer = '';
    liveAssistantIndex = -1;
    renderChat();
}

function addSystemNote(text) {
    appendMessage('system', text, false);
}

// ============================================================
// 音频播放
// ============================================================

function ensureAudioCtx() {
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === 'suspended') {
        audioCtx.resume().catch(() => {});
    }
    return audioCtx;
}

function stopTtsPlayback() {
    for (const src of activeSources) {
        try { src.stop(); } catch (e) {}
        try { src.disconnect(); } catch (e) {}
    }
    activeSources = [];
    nextPlayTime = 0;
}

function playPcmChunk(arrayBuffer) {
    const ctx = ensureAudioCtx();
    const pcm = new Int16Array(arrayBuffer);
    if (!pcm.length) return;

    const f32 = new Float32Array(pcm.length);
    for (let i = 0; i < pcm.length; i++) f32[i] = pcm[i] / 32768;

    const buffer = ctx.createBuffer(1, f32.length, ttsSampleRate);
    buffer.copyToChannel(f32, 0);

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);

    const now = ctx.currentTime;
    if (nextPlayTime < now + 0.02) nextPlayTime = now + 0.05;
    source.start(nextPlayTime);
    nextPlayTime += buffer.duration;
    activeSources.push(source);
    source.onended = () => {
        activeSources = activeSources.filter(s => s !== source);
    };
}

// ============================================================
// 配置摘要
// ============================================================

async function loadConfig() {
    try {
        const resp = await fetch('/config');
        const config = await resp.json();
        configInfo.innerHTML = `
            STT：${config.stt?.provider || '-'} / ${config.stt?.model || '-'}<br>
            LLM：${config.llm?.provider || '-'} / ${config.llm?.model || '-'}<br>
            TTS：${config.tts?.provider || '-'} / ${config.tts?.voice || '-'}
        `;
    } catch (e) {
        configInfo.textContent = '配置加载失败';
    }
}

loadConfig();
loadSettings(false);
initSettingsUI();

// ============================================================
// 会话
// ============================================================

async function startConversation() {
    try {
        startBtn.disabled = true;
        statusText.textContent = '正在初始化...';

        localStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
            },
            video: false,
        });

        peerConnection = new RTCPeerConnection({
            iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
        });

        localStream.getAudioTracks().forEach(track => {
            peerConnection.addTrack(track, localStream);
        });

        peerConnection.ontrack = (event) => {
            if (event.track.kind === 'audio') {
                remoteAudio = document.createElement('audio');
                remoteAudio.srcObject = event.streams[0];
                remoteAudio.autoplay = true;
                remoteAudio.play().catch(() => {});
            }
        };

        peerConnection.onicecandidate = (event) => {
            if (event.candidate && websocket && websocket.readyState === WebSocket.OPEN) {
                websocket.send(JSON.stringify({ type: 'ice', candidate: event.candidate }));
            }
        };

        peerConnection.onconnectionstatechange = () => {
            if (peerConnection.connectionState === 'connected') {
                isRunning = true;
                stopBtn.disabled = false;
                setAppState('idle');
                statusText.textContent = '连接成功，请开始说话';
                ensureAudioCtx();
            } else if (peerConnection.connectionState === 'failed') {
                statusText.textContent = '连接失败，请刷新重试';
                stopConversation();
            }
        };

        const offer = await peerConnection.createOffer();
        await peerConnection.setLocalDescription(offer);
        await connectWebSocket(offer);
    } catch (e) {
        console.error('Start error:', e);
        statusText.textContent = '启动失败: ' + e.message;
        startBtn.disabled = false;
    }
}

async function connectWebSocket(offer) {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    websocket = new WebSocket(`${wsProtocol}//${window.location.host}/ws`);
    websocket.binaryType = 'arraybuffer';

    websocket.onopen = () => {
        ensureAudioCtx();
        websocket.send(JSON.stringify({
            type: 'offer',
            sdp: offer.sdp,
            sdp_type: offer.type,
        }));
    };

    websocket.onmessage = async (event) => {
        if (event.data instanceof ArrayBuffer) {
            playPcmChunk(event.data);
            return;
        }

        const message = JSON.parse(event.data);

        if (message.type === 'answer') {
            await peerConnection.setRemoteDescription(
                new RTCSessionDescription({ sdp: message.sdp, type: 'answer' })
            );
            statusText.textContent = '正在建立连接...';
            return;
        }

        if (message.type === 'audio_start') {
            ttsSampleRate = message.sample_rate || 24000;
            ensureAudioCtx();
            return;
        }
        if (message.type === 'audio_stop') {
            stopTtsPlayback();
            return;
        }
        if (message.type === 'audio_end') {
            return;
        }

        if (message.type === 'state') {
            const s = message.state;
            if (s === 'listening') {
                assistantBuffer = '';
                if (liveAssistantIndex >= 0) finishMessage(liveAssistantIndex);
                liveAssistantIndex = -1;
                setAppState('listening');
            } else if (s === 'thinking') {
                if (liveAssistantIndex >= 0) finishMessage(liveAssistantIndex);
                assistantBuffer = '';
                liveAssistantIndex = appendMessage('assistant', '…', true);
                setAppState('thinking');
            } else if (s === 'speaking') {
                setAppState('speaking');
            } else if (s === 'interrupted') {
                stopTtsPlayback();
                if (liveAssistantIndex >= 0) {
                    const t = chatHistory[liveAssistantIndex].text;
                    updateMessage(liveAssistantIndex, (t || '') + '（已打断）', false);
                    liveAssistantIndex = -1;
                }
                setAppState('interrupted');
            } else {
                setAppState('idle');
            }
            return;
        }

        if (message.type === 'stt_partial') {
            // partial 只更新最新 user 行
            let idx = -1;
            for (let i = chatHistory.length - 1; i >= 0; i--) {
                if (chatHistory[i].role === 'user' && chatHistory[i].streaming) { idx = i; break; }
            }
            if (idx < 0) idx = appendMessage('user', message.text, true);
            else updateMessage(idx, message.text, true);
            return;
        }

        if (message.type === 'stt_final') {
            let idx = -1;
            for (let i = chatHistory.length - 1; i >= 0; i--) {
                if (chatHistory[i].role === 'user' && chatHistory[i].streaming) { idx = i; break; }
            }
            if (idx >= 0) updateMessage(idx, message.text, false);
            else appendMessage('user', message.text, false);
            assistantBuffer = '';
            liveAssistantIndex = appendMessage('assistant', '…', true);
            return;
        }

        if (message.type === 'llm_chunk') {
            assistantBuffer += message.text;
            if (liveAssistantIndex < 0) {
                liveAssistantIndex = appendMessage('assistant', assistantBuffer, true);
            } else {
                updateMessage(liveAssistantIndex, assistantBuffer, true);
            }
            return;
        }

        if (message.type === 'llm_done') {
            if (liveAssistantIndex >= 0) {
                if (!assistantBuffer) updateMessage(liveAssistantIndex, '（无回复）', false);
                else finishMessage(liveAssistantIndex);
                liveAssistantIndex = -1;
            }
            return;
        }

        if (message.type === 'interrupt') {
            stopTtsPlayback();
            return;
        }

        if (message.type === 'error') {
            console.error('Pipeline error:', message.message);
            addSystemNote('错误：' + message.message);
            statusText.textContent = '出错: ' + message.message;
            return;
        }
    };

    websocket.onerror = () => {
        statusText.textContent = '信令连接失败';
    };
    websocket.onclose = () => {
        console.log('WebSocket closed');
    };
}

function stopConversation() {
    isRunning = false;
    setAppState('idle');
    stopTtsPlayback();
    if (liveAssistantIndex >= 0) finishMessage(liveAssistantIndex);
    liveAssistantIndex = -1;

    if (peerConnection) { peerConnection.close(); peerConnection = null; }
    if (websocket) { websocket.close(); websocket = null; }
    if (localStream) {
        localStream.getTracks().forEach(t => t.stop());
        localStream = null;
    }
    if (remoteAudio) { remoteAudio.pause(); remoteAudio = null; }

    startBtn.disabled = false;
    stopBtn.disabled = true;
    statusText.textContent = '已停止，点击「开始对话」重新开始';
}

document.addEventListener('keydown', (e) => {
    if (e.code === 'Escape') {
        if (isSettingsOpen()) {
            e.preventDefault();
            closeSettings();
            return;
        }
        if (isRunning) {
            e.preventDefault();
            stopConversation();
        }
        return;
    }
    if (e.code === 'Space' && !isRunning && !isSettingsOpen()) {
        e.preventDefault();
        startConversation();
    }
});
window.addEventListener('beforeunload', () => {
    if (isRunning) stopConversation();
});

// ============================================================
// 设置面板（与 /api/settings 对接）
// ============================================================

function toggleSettings() {
    const modal = document.getElementById('settingsModal');
    if (!modal) return;
    if (modal.classList.contains('open')) {
        closeSettings();
    } else {
        openSettings();
    }
}

function openSettings() {
    const modal = document.getElementById('settingsModal');
    if (!modal) return;
    modal.classList.add('open');
    loadSettings(false);
}

function closeSettings() {
    const modal = document.getElementById('settingsModal');
    if (!modal) return;
    modal.classList.remove('open');
}

function onSettingsBackdrop(event) {
    if (event.target && event.target.id === 'settingsModal') {
        closeSettings();
    }
}

function isSettingsOpen() {
    const modal = document.getElementById('settingsModal');
    return !!(modal && modal.classList.contains('open'));
}

function initSettingsUI() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            const pane = document.getElementById('tab-' + btn.dataset.tab);
            if (pane) pane.classList.add('active');
        });
    });

    const bindRange = (id) => {
        const el = document.getElementById(id);
        const val = document.getElementById(id + '_val');
        if (!el || !val) return;
        const sync = () => { val.textContent = el.value; };
        el.addEventListener('input', sync);
        sync();
    };
    [
        'llm_temperature',
        'vad_threshold', 'vad_min_speech_duration', 'vad_min_silence_duration',
        'vad_interrupt_threshold', 'vad_interrupt_min_duration', 'vad_pending_grace',
    ].forEach(bindRange);

    // Provider 切换：显示对应字段并填默认值
    const sttSel = document.getElementById('stt_provider');
    const llmSel = document.getElementById('llm_provider');
    const ttsSel = document.getElementById('tts_provider');
    if (sttSel) sttSel.addEventListener('change', () => applyProviderFields('stt', true));
    if (llmSel) llmSel.addEventListener('change', () => applyProviderFields('llm', true));
    if (ttsSel) ttsSel.addEventListener('change', () => applyProviderFields('tts', true));
}

function setVal(id, v) {
    const el = document.getElementById(id);
    if (el && v !== undefined && v !== null && v !== '') el.value = v;
}

function applyProviderFields(section, applyDefaults) {
    const sel = document.getElementById(`${section}_provider`);
    if (!sel) return;
    const p = sel.value;

    const show = (cls, on) => {
        document.querySelectorAll('.' + cls).forEach(el => {
            el.style.display = on ? '' : 'none';
        });
    };

    if (section === 'stt') {
        show('stt-openai-only', p === 'openai');
        show('stt-fw-only', p === 'faster-whisper');
        if (applyDefaults && p === 'faster-whisper') {
            setVal('stt_model_size', 'base');
            setVal('stt_device', 'auto');
            setVal('stt_compute_type', 'int8');
            setVal('stt_fw_language', document.getElementById('stt_fw_language')?.value || 'zh');
        }
        if (applyDefaults && p === 'openai') {
            setVal('stt_language', document.getElementById('stt_language')?.value || 'zh');
        }
    }

    if (section === 'llm') {
        show('llm-openai-only', p === 'openai');
        show('llm-ollama-only', p === 'ollama');
        if (applyDefaults && p === 'ollama') {
            const url = document.getElementById('llm_base_url');
            if (url && (!url.value || url.value.includes('api.openai.com') || url.value.includes('agnes') || url.value.includes('siliconflow') || url.value.includes('deepseek'))) {
                url.value = 'http://localhost:11434';
            }
            const model = document.getElementById('llm_model');
            if (model && (!model.value || model.value.includes('gpt') || model.value.includes('flash') || model.value.includes('deepseek'))) {
                model.value = 'qwen2.5:7b';
            }
        }
        if (applyDefaults && p === 'openai') {
            const url = document.getElementById('llm_base_url');
            if (url && url.value.includes('localhost:11434')) {
                url.value = 'https://api.openai.com/v1';
            }
            const model = document.getElementById('llm_model');
            if (model && model.value.includes('qwen')) {
                model.value = 'gpt-4o-mini';
            }
        }
    }

    if (section === 'tts') {
        show('tts-openai-only', p === 'openai');
        show('tts-edge-only', p === 'edge-tts');
        if (applyDefaults && p === 'edge-tts') {
            setVal('tts_edge_voice', document.getElementById('tts_edge_voice')?.value || 'zh-CN-XiaoxiaoNeural');
            setVal('tts_rate', '+0%');
            setVal('tts_volume', '+0%');
            const sr = document.getElementById('tts_sample_rate')?.value || '24000';
            setVal('tts_edge_sample_rate', sr);
        }
        if (applyDefaults && p === 'openai') {
            setVal('tts_voice', document.getElementById('tts_voice')?.value || 'alloy');
        }
    }
}

function setSettingsMsg(text, isErr) {
    const el = document.getElementById('settingsMsg');
    if (!el) return;
    el.textContent = text || '';
    el.className = 'settings-msg' + (isErr ? ' err' : '');
}

function fillSettingsForm(data) {
    const setVal = (id, v) => {
        const el = document.getElementById(id);
        if (el && v !== undefined && v !== null) el.value = v;
    };
    const setChk = (id, v) => {
        const el = document.getElementById(id);
        if (el) el.checked = !!v;
    };

    setVal('system_prompt', data.system_prompt);
    setVal('language', data.language || 'zh');
    setVal('max_history', data.max_history ?? 10);
    setChk('interrupt_on_speech', data.interrupt_on_speech !== false);

    const stt = data.stt || {};
    setVal('stt_provider', stt.provider || 'openai');
    setVal('stt_model', stt.model || '');
    setVal('stt_base_url', stt.base_url || '');
    setVal('stt_language', stt.language || 'zh');
    setVal('stt_model_size', stt.model_size || 'base');
    setVal('stt_device', stt.device || 'auto');
    setVal('stt_compute_type', stt.compute_type || 'int8');
    setVal('stt_fw_language', stt.language || 'zh');
    const sttKey = document.getElementById('stt_api_key');
    if (sttKey) sttKey.value = '';
    const sttHint = document.getElementById('stt_key_hint');
    if (sttHint) sttHint.textContent = stt.has_api_key ? `已配置 ${stt.api_key_masked || ''}，留空保持` : '未配置';

    const llm = data.llm || {};
    setVal('llm_provider', llm.provider || 'openai');
    setVal('llm_model', llm.model || '');
    setVal('llm_base_url', llm.base_url || '');
    setVal('llm_temperature', llm.temperature ?? 0.7);
    setVal('llm_max_tokens', llm.max_tokens ?? 1024);
    const llmKey = document.getElementById('llm_api_key');
    if (llmKey) llmKey.value = '';
    const llmHint = document.getElementById('llm_key_hint');
    if (llmHint) llmHint.textContent = llm.has_api_key ? `已配置 ${llm.api_key_masked || ''}，留空保持` : '未配置';

    const tts = data.tts || {};
    setVal('tts_provider', tts.provider || 'openai');
    setVal('tts_mode', tts.mode || 'chat');
    setVal('tts_model', tts.model || '');
    setVal('tts_voice', tts.voice || '');
    setVal('tts_base_url', tts.base_url || '');
    setVal('tts_sample_rate', tts.sample_rate ?? 24000);
    setVal('tts_edge_voice', tts.voice || 'zh-CN-XiaoxiaoNeural');
    setVal('tts_rate', tts.rate || '+0%');
    setVal('tts_volume', tts.volume || '+0%');
    setVal('tts_edge_sample_rate', tts.sample_rate ?? 24000);
    const ttsKey = document.getElementById('tts_api_key');
    if (ttsKey) ttsKey.value = '';
    const ttsHint = document.getElementById('tts_key_hint');
    if (ttsHint) ttsHint.textContent = tts.has_api_key ? `已配置 ${tts.api_key_masked || ''}，留空保持` : '未配置';

    // 按当前 provider 显示字段
    applyProviderFields('stt', false);
    applyProviderFields('llm', false);
    applyProviderFields('tts', false);

    const vad = data.vad || {};
    setVal('vad_threshold', vad.threshold ?? 0.62);
    setVal('vad_min_speech_duration', vad.min_speech_duration ?? 0.45);
    setVal('vad_min_silence_duration', vad.min_silence_duration ?? 0.85);
    setVal('vad_interrupt_threshold', vad.interrupt_threshold ?? 0.78);
    setVal('vad_interrupt_min_duration', vad.interrupt_min_duration ?? 0.8);
    setVal('vad_pending_grace', vad.pending_grace ?? 0.28);

    [
        'llm_temperature',
        'vad_threshold', 'vad_min_speech_duration', 'vad_min_silence_duration',
        'vad_interrupt_threshold', 'vad_interrupt_min_duration', 'vad_pending_grace',
    ].forEach(id => {
        const el = document.getElementById(id);
        const val = document.getElementById(id + '_val');
        if (el && val) val.textContent = el.value;
    });
}

async function loadSettings(openPanel) {
    try {
        const resp = await fetch('/api/settings');
        const data = await resp.json();
        fillSettingsForm(data);
        if (openPanel) setSettingsMsg('已从服务器重新加载');
    } catch (e) {
        setSettingsMsg('加载设置失败: ' + e.message, true);
    }
}

function collectSettingsPayload() {
    const num = (id, fallback) => {
        const v = document.getElementById(id)?.value;
        if (v === undefined || v === null || v === '') return fallback;
        const n = Number(v);
        return Number.isFinite(n) ? n : fallback;
    };
    const str = (id, fallback = '') => {
        const v = document.getElementById(id)?.value;
        return v === undefined || v === null ? fallback : v;
    };
    const sttKey = str('stt_api_key').trim();
    const llmKey = str('llm_api_key').trim();
    const ttsKey = str('tts_api_key').trim();

    const sttProvider = str('stt_provider', 'openai');
    const llmProvider = str('llm_provider', 'openai');
    const ttsProvider = str('tts_provider', 'openai');

    let stt;
    if (sttProvider === 'faster-whisper') {
        stt = {
            provider: 'faster-whisper',
            model_size: str('stt_model_size', 'base'),
            device: str('stt_device', 'auto'),
            compute_type: str('stt_compute_type', 'int8'),
            language: str('stt_fw_language', 'zh'),
        };
    } else {
        stt = {
            provider: 'openai',
            model: str('stt_model'),
            base_url: str('stt_base_url'),
            language: str('stt_language', 'zh'),
            ...(sttKey ? { api_key: sttKey } : {}),
        };
    }

    let llm;
    if (llmProvider === 'ollama') {
        llm = {
            provider: 'ollama',
            model: str('llm_model') || 'qwen2.5:7b',
            base_url: str('llm_base_url') || 'http://localhost:11434',
            temperature: num('llm_temperature', 0.7),
            max_tokens: Math.round(num('llm_max_tokens', 1024)),
        };
    } else {
        llm = {
            provider: 'openai',
            model: str('llm_model'),
            base_url: str('llm_base_url'),
            temperature: num('llm_temperature', 0.7),
            max_tokens: Math.round(num('llm_max_tokens', 1024)),
            ...(llmKey ? { api_key: llmKey } : {}),
        };
    }

    let tts;
    if (ttsProvider === 'edge-tts') {
        tts = {
            provider: 'edge-tts',
            voice: str('tts_edge_voice') || 'zh-CN-XiaoxiaoNeural',
            rate: str('tts_rate') || '+0%',
            volume: str('tts_volume') || '+0%',
            sample_rate: Math.round(num('tts_edge_sample_rate', 24000)),
        };
    } else {
        tts = {
            provider: 'openai',
            mode: str('tts_mode', 'chat'),
            model: str('tts_model'),
            voice: str('tts_voice'),
            base_url: str('tts_base_url'),
            sample_rate: Math.round(num('tts_sample_rate', 24000)),
            ...(ttsKey ? { api_key: ttsKey } : {}),
        };
    }

    return {
        language: str('language', 'zh'),
        system_prompt: str('system_prompt'),
        max_history: Math.max(1, Math.round(num('max_history', 10))),
        interrupt_on_speech: !!document.getElementById('interrupt_on_speech')?.checked,
        stt,
        llm,
        tts,
        vad: {
            threshold: num('vad_threshold', 0.62),
            min_speech_duration: num('vad_min_speech_duration', 0.45),
            min_silence_duration: num('vad_min_silence_duration', 0.85),
            interrupt_threshold: num('vad_interrupt_threshold', 0.78),
            interrupt_min_duration: num('vad_interrupt_min_duration', 0.8),
            pending_grace: num('vad_pending_grace', 0.28),
        },
    };
}

async function saveSettings() {
    const btn = document.getElementById('saveSettingsBtn');
    if (btn) btn.disabled = true;
    setSettingsMsg('保存中...');
    try {
        const payload = collectSettingsPayload();
        const resp = await fetch('/api/settings', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const data = await resp.json();
        if (!resp.ok || data.ok === false) {
            throw new Error(data.error || data.detail || ('HTTP ' + resp.status));
        }
        const applied = data.applied_connections ?? 0;
        let msg = `已保存并热更新（连接 ${applied}）`;
        if (data.provider_error) {
            setSettingsMsg(msg + `；警告: ${data.provider_error}`, true);
        } else {
            setSettingsMsg(msg);
        }
        await loadConfig();
        await loadSettings(false);
    } catch (e) {
        setSettingsMsg('保存失败: ' + e.message, true);
    } finally {
        if (btn) btn.disabled = false;
    }
}
