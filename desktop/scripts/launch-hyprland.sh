#!/bin/bash
# Cyber Agent IDE — Hyprland (Wayland) Launch Script
# Enables native Wayland support for Liquid Glass effects

ELECTRON_FLAGS=(
  --enable-features=UseOzonePlatform
  --ozone-platform=wayland
  --enable-features=WebRTCPipeWireCapturer
  --enable-gpu-rasterization
  --enable-zero-copy
)

# Ensure Electron uses Wayland
export ELECTRON_OZONE_PLATFORM_HINT=wayland
export GDK_BACKEND=wayland
export QT_QPA_PLATFORM=wayland

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESKTOP_DIR="$(dirname "$SCRIPT_DIR")"

# Check if electron is available
if command -v electron &> /dev/null; then
  ELECTRON_BIN="electron"
elif [ -f "$DESKTOP_DIR/node_modules/.bin/electron" ]; then
  ELECTRON_BIN="$DESKTOP_DIR/node_modules/.bin/electron"
else
  echo "Electron not found. Run: cd $DESKTOP_DIR && npm install"
  exit 1
fi

echo "Starting Cyber Agent IDE with Wayland/Hyprland support..."
exec "$ELECTRON_BIN" "${ELECTRON_FLAGS[@]}" "$DESKTOP_DIR" "$@"
