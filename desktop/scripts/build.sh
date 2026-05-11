#!/usr/bin/env bash
set -euo pipefail

# Cyber Agent IDE build script
# Builds the Tauri desktop app for Linux (AppImage) and Windows (MSI)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=== 1. 安装前端依赖 ==="
npm install

echo "=== 2. 构建 Vite 前端 ==="
npm run build

echo "=== 3. 构建 Tauri 桌面应用 ==="
npx tauri build "$@"

echo ""
echo "=== 构建完成 ==="
echo "产物位置: src-tauri/target/release/bundle/"
ls -la src-tauri/target/release/bundle/ 2>/dev/null || echo "(请检查构建输出)"
