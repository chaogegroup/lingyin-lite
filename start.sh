#!/bin/bash
# 实时语音 AI 助手 - 一键启动脚本 (macOS/Linux)

echo "============================================================"
echo "  实时语音 AI 助手 - 一键启动"
echo "  Lingyin Lite v1.0.0"
echo "============================================================"
echo ""

# 检查 Python3
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未检测到 python3，请先安装 Python 3.10+"
    echo "macOS: brew install python@3.11"
    echo "Ubuntu: sudo apt install python3.10 python3.10-venv"
    exit 1
fi

# 检查版本
PYVER=$(python3 --version)
echo "[信息] $PYVER"
echo ""

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 运行启动脚本
python3 "$SCRIPT_DIR/run.py" "$@"
