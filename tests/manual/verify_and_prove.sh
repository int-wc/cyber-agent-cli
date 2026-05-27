#!/bin/bash
# FuckCSDN 一键验证 —— 编译 + 逻辑证明 + 端到端实测
# 用法: bash tests/manual/verify_and_prove.sh
set -e
cd "$(dirname "$0")/../.."
export PYTHONPATH=src

echo "========================================"
echo "  Step 1/3: 编译检查"
echo "========================================"
python -c "compile(open('src/cyber_agent/tools/search.py').read(),'search.py','exec');print('OK')"

echo ""
echo "========================================"
echo "  Step 2/3: 逻辑证明 (无网络)"
echo "========================================"
python tests/manual/prove_fuck_csdn.py --dry 2>&1 | tail -20

echo ""
echo "========================================"
echo "  Step 3/3: 端到端实测"
echo "========================================"
python tests/manual/prove_fuck_csdn.py --live
