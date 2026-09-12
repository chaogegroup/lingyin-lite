#!/usr/bin/env python3
"""
实时语音 AI 助手 - 一键启动脚本

功能：
1. 检查 Python 版本
2. 创建/复用项目 venv（始终用 venv 内 Python 安装与启动，避免污染系统环境）
3. 安装依赖到 venv
4. 检查配置文件
5. 自检
6. 用 venv 启动服务并打开浏览器

用法：
    python run.py
    python run.py --check
    python run.py --install
    python run.py --recreate-venv
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

try:
    os.system("")
    COLOR = True
except Exception:
    COLOR = False


def c(text: str, color: str) -> str:
    if not COLOR:
        return text
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "cyan": "\033[96m",
        "bold": "\033[1m",
        "reset": "\033[0m",
    }
    return f"{colors.get(color, '')}{text}{colors['reset']}"


def info(msg: str) -> None:
    print(c(f"[INFO] {msg}", "blue"), flush=True)


def ok(msg: str) -> None:
    print(c(f"[OK]   {msg}", "green"), flush=True)


def warn(msg: str) -> None:
    print(c(f"[WARN] {msg}", "yellow"), flush=True)


def error(msg: str) -> None:
    print(c(f"[ERROR] {msg}", "red"), flush=True)


def step(msg: str) -> None:
    print(c(f"\n>>> {msg}", "cyan"), flush=True)


BASE_DIR = Path(__file__).parent.resolve()
REQ_FILE = BASE_DIR / "requirements.txt"
CONFIG_EXAMPLE = BASE_DIR / "config.example.yaml"
CONFIG_FILE = BASE_DIR / "config.yaml"
VENV_DIR = BASE_DIR / "venv"

# 国内网络：HuggingFace 镜像（faster-whisper 等）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def get_venv_python() -> Path:
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def get_venv_pip() -> Path:
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "pip.exe"
    return VENV_DIR / "bin" / "pip"


def in_project_venv() -> bool:
    """当前解释器是否就是本项目的 venv。"""
    try:
        return Path(sys.prefix).resolve() == VENV_DIR.resolve()
    except Exception:
        return False


def check_python_version() -> bool:
    step("检查 Python 版本")
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        error(f"Python 版本过低：{version_str}，需要 >= 3.10")
        return False
    ok(f"Python {version_str} 满足要求 (>= 3.10)")
    info(f"当前解释器：{sys.executable}")
    return True


def recreate_venv() -> Path:
    step("重建虚拟环境")
    if VENV_DIR.exists():
        warn(f"删除旧 venv：{VENV_DIR}")
        shutil.rmtree(VENV_DIR, ignore_errors=True)
    return create_venv()


def create_venv() -> Path:
    info(f"创建虚拟环境：{VENV_DIR}")
    # --upgrade-deps 确保 venv 自带可用 pip
    cmd = [sys.executable, "-m", "venv", str(VENV_DIR)]
    try:
        subprocess.run(cmd, check=True, cwd=str(BASE_DIR))
    except subprocess.CalledProcessError:
        info("重试：不带额外参数创建 venv...")
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True, cwd=str(BASE_DIR))

    venv_python = get_venv_python()
    if not venv_python.exists():
        error(f"venv 创建后找不到 python：{venv_python}")
        sys.exit(1)
    ok(f"虚拟环境就绪：{venv_python}")
    return venv_python


def ensure_pip(venv_python: Path) -> None:
    """确保 venv 里有 pip（有些创建方式会缺）。"""
    try:
        r = subprocess.run(
            [str(venv_python), "-m", "pip", "--version"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0:
            ok(f"pip 可用：{r.stdout.strip()}")
            return
    except Exception:
        pass
    warn("venv 缺少 pip，正在 bootstrap...")
    subprocess.run([str(venv_python), "-m", "ensurepip", "--upgrade"], check=False, cwd=str(BASE_DIR))
    subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], check=False, cwd=str(BASE_DIR))


def ensure_venv(recreate: bool = False) -> Path:
    """保证返回本项目 venv 的 python 路径（不依赖 os.execv）。"""
    step("检查虚拟环境")

    if recreate:
        return recreate_venv()

    venv_python = get_venv_python()
    if VENV_DIR.exists() and venv_python.exists():
        ok(f"使用项目 venv：{venv_python}")
        return venv_python

    if VENV_DIR.exists() and not venv_python.exists():
        warn("venv 目录存在但不完整，将重建")
        shutil.rmtree(VENV_DIR, ignore_errors=True)

    return create_venv()


def run_in_venv(venv_python: Path, args: list[str], **kwargs) -> subprocess.CompletedProcess:
    """始终用 venv 内 Python 执行命令。"""
    cmd = [str(venv_python)] + args
    return subprocess.run(cmd, cwd=str(BASE_DIR), **kwargs)


def install_dependencies(venv_python: Path, upgrade: bool = False) -> bool:
    step("安装 Python 依赖（写入项目 venv，不污染系统 Python）")

    if not REQ_FILE.exists():
        error(f"依赖文件不存在：{REQ_FILE}")
        return False

    # 去掉 BOM，避免个别环境把第一行当成非法 requirement
    try:
        raw = REQ_FILE.read_bytes()
        if raw.startswith(b"\xef\xbb\xbf"):
            REQ_FILE.write_bytes(raw[3:])
            warn("requirements.txt 含 UTF-8 BOM，已清除")
    except Exception as e:
        warn(f"清理 BOM 失败（可忽略）：{e}")

    info(f"目标解释器：{venv_python}")
    info("正在安装依赖，这可能需要几分钟...")

    cmd = [str(venv_python), "-m", "pip", "install", "-r", str(REQ_FILE)]
    if upgrade:
        cmd.append("--upgrade")

    try:
        result = subprocess.run(cmd, cwd=str(BASE_DIR))
        if result.returncode != 0:
            warn("默认源安装失败，尝试清华镜像...")
            cmd_mirror = cmd + ["-i", "https://pypi.tuna.tsinghua.edu.cn/simple"]
            result = subprocess.run(cmd_mirror, cwd=str(BASE_DIR))
            if result.returncode != 0:
                error("依赖安装失败，请检查网络连接")
                return False
        ok("依赖安装完成（已写入项目 venv）")
        return True
    except Exception as e:
        error(f"依赖安装出错：{e}")
        return False


def ensure_config() -> bool:
    step("检查配置文件")
    if CONFIG_FILE.exists():
        ok(f"配置文件已存在：{CONFIG_FILE}")
        return True
    if not CONFIG_EXAMPLE.exists():
        error(f"配置模板不存在：{CONFIG_EXAMPLE}")
        return False
    shutil.copy2(CONFIG_EXAMPLE, CONFIG_FILE)
    ok(f"已创建配置文件：{CONFIG_FILE}")
    warn("请编辑 config.yaml 填入你的 API Key（或用网页设置面板）")
    return True


def self_check(venv_python: Path) -> dict:
    step("运行自检（使用 venv 解释器）")
    results: dict = {}

    required_modules = [
        ("fastapi", "FastAPI Web 框架"),
        ("uvicorn", "ASGI 服务器"),
        ("aiortc", "WebRTC 库"),
        ("aiohttp", "HTTP 客户端"),
        ("yaml", "配置解析 (PyYAML)"),
        ("numpy", "数值计算"),
        ("av", "音频解码 (PyAV)"),
        ("torch", "PyTorch（Silero VAD）"),
        ("silero_vad", "Silero VAD 包"),
        ("edge_tts", "Edge TTS（免费语音合成）"),
        ("faster_whisper", "Faster Whisper（本地 STT）"),
    ]
    optional_modules = []

    all_mods = required_modules + optional_modules
    # 只传模块名列表，避免把字符串解包成 (mod, desc) 导致自检失败
    check_script = (
        "import importlib, json\n"
        "mods = json.loads(%s)\n"
        "out = {}\n"
        "for mod in mods:\n"
        "    try:\n"
        "        importlib.import_module(mod)\n"
        "        out[mod] = True\n"
        "    except Exception:\n"
        "        out[mod] = False\n"
        "print(json.dumps(out))\n"
    ) % json.dumps(json.dumps([m for m, _ in all_mods]))

    try:
        result = subprocess.run(
            [str(venv_python), "-c", check_script],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(BASE_DIR), timeout=60,
        )
        stdout = (result.stdout or "").strip().splitlines()
        mod_results = json.loads(stdout[-1]) if stdout else {}
    except Exception as e:
        warn(f"依赖检查失败：{e}")
        mod_results = {}

    info("检查必需依赖...")
    deps_ok = True
    for mod, desc in required_modules:
        if mod_results.get(mod):
            ok(f"  {desc} ({mod})")
        else:
            error(f"  {desc} ({mod}) - 未安装到 venv")
            deps_ok = False
    results["dependencies"] = deps_ok

    if optional_modules:
        info("检查可选依赖...")
        for mod, desc in optional_modules:
            if mod_results.get(mod):
                ok(f"  {desc}")
            else:
                warn(f"  {desc} - 未安装（可选）")

    # 确认安装位置在 venv
    info("确认包安装位置...")
    try:
        # 子进程直接判断是否在 venv 内，避免中文路径经控制台编码后无法比对
        probe = (
            "import fastapi, sys\n"
            "from pathlib import Path\n"
            "p = Path(fastapi.__file__).resolve()\n"
            "v = Path(sys.prefix).resolve()\n"
            "try:\n"
            "    p.relative_to(v)\n"
            "    print('IN_VENV')\n"
            "except ValueError:\n"
            "    print('OUTSIDE')\n"
        )
        loc = subprocess.run(
            [str(venv_python), "-c", probe],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20,
        )
        flag = (loc.stdout or "").strip().splitlines()[-1] if (loc.stdout or "").strip() else ""
        if flag == "IN_VENV":
            ok("  fastapi 已安装在项目 venv 内")
        elif flag == "OUTSIDE":
            warn("  fastapi 不在项目 venv（可能装到系统 Python 了）")
        else:
            warn(f"  位置检查无结果：{flag or (loc.stderr or '')[:120]}")
    except Exception as e:
        warn(f"  位置检查失败：{e}")

    info("检查配置...")
    if not CONFIG_FILE.exists():
        error("  config.yaml 不存在")
        results["config"] = False
    else:
        try:
            import yaml
            with open(CONFIG_FILE, encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            api_issues = []
            for provider_type in ["stt", "llm", "tts"]:
                provider_config = config.get(provider_type) or {}
                provider_name = provider_config.get("provider", "")
                api_key = provider_config.get("api_key", "") or ""
                if provider_name == "openai" and api_key:
                    if "你的" in api_key or "xxx" in api_key.lower() or len(api_key) < 10:
                        api_issues.append(f"{provider_type.upper()} API Key 未正确填写")
            if api_issues:
                for issue in api_issues:
                    warn(f"  {issue}")
                results["api_key"] = False
            else:
                ok("  API Key 配置检查通过")
                results["api_key"] = True
            results["config"] = True
        except Exception as e:
            error(f"  配置文件解析失败：{e}")
            results["config"] = False

    info("检查端口...")
    port = 8000
    try:
        import yaml
        with open(CONFIG_FILE, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
            port = int(config.get("port", 8000))
    except Exception:
        pass

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("0.0.0.0", port))
        sock.close()
        ok(f"  端口 {port} 可用")
        results["port"] = True
    except OSError:
        warn(f"  端口 {port} 已被占用")
        results["port"] = False

    print()
    failed = [k for k, v in results.items() if v is False]
    if not failed:
        ok("自检全部通过！")
    else:
        warn(f"未通过项：{', '.join(failed)}")

    results["port_number"] = port
    return results


def start_server(venv_python: Path, port: int, host: str = "0.0.0.0", open_browser: bool = True):
    step("启动服务（venv）")
    ok(f"解释器：{venv_python}")
    url = f"http://localhost:{port}"
    ok(f"服务地址：{url}")
    info("按 Ctrl+C 停止服务")
    print()

    if open_browser:
        def open_browser_delayed():
            time.sleep(2.5)
            try:
                webbrowser.open(url)
            except Exception:
                pass
        import threading
        threading.Thread(target=open_browser_delayed, daemon=True).start()

    cmd = [
        str(venv_python), "-m", "uvicorn", "server.main:app",
        "--host", host, "--port", str(port),
    ]
    try:
        subprocess.run(cmd, cwd=str(BASE_DIR))
    except KeyboardInterrupt:
        print()
        info("服务已停止")
    except Exception as e:
        error(f"服务启动失败：{e}")


def main():
    parser = argparse.ArgumentParser(description="实时语音 AI 助手 - 一键启动")
    parser.add_argument("--check", action="store_true", help="仅自检")
    parser.add_argument("--install", action="store_true", help="仅安装依赖")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--upgrade", action="store_true", help="升级依赖")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--recreate-venv", action="store_true", help="删除并重建 venv 后再安装")
    args = parser.parse_args()

    print(c("=" * 60, "cyan"))
    print(c("  实时语音 AI 助手 - 一键启动", "bold"))
    print(c("  Voice AI Assistant Lite v1.03", "cyan"))
    print(c("=" * 60, "cyan"))

    if not check_python_version():
        sys.exit(1)

    # 始终用项目 venv，不再 os.execv（Windows + 中文路径不可靠）
    venv_python = ensure_venv(recreate=args.recreate_venv)
    ensure_pip(venv_python)

    if not args.skip_install:
        if not install_dependencies(venv_python, upgrade=args.upgrade):
            if not args.check:
                error("依赖安装失败，无法继续")
                sys.exit(1)

    if args.install:
        ok("依赖安装完成")
        return

    ensure_config()
    check_results = self_check(venv_python)

    if args.check:
        return

    port = args.port or check_results.get("port_number", 8000)
    if not check_results.get("config", True):
        warn("配置文件有问题，但仍尝试启动...")

    print()
    info("3 秒后启动服务，按 Ctrl+C 取消...")
    try:
        time.sleep(3)
    except KeyboardInterrupt:
        info("已取消")
        return

    start_server(
        venv_python,
        port=port,
        host=args.host,
        open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
