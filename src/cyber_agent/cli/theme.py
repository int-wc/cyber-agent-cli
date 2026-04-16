"""统一维护 CLI 与 TUI 共用的交互配色。"""

WINDOW_BG = "#0f172a"
SURFACE_BG = "#111827"
PANEL_BORDER = "#475569"

TEXT_PRIMARY = "#e2e8f0"
TEXT_MUTED = "#94a3b8"
TEXT_SOFT = "#cbd5e1"

USER_BORDER_COLOR = "#f59e0b"
USER_TEXT_COLOR = "#fef3c7"
ASSISTANT_BORDER_COLOR = "#14b8a6"
ASSISTANT_TEXT_COLOR = "#ccfbf1"
SYSTEM_BORDER_COLOR = "#64748b"
ERROR_BORDER_COLOR = "#ef4444"
ERROR_TEXT_COLOR = "#fee2e2"

COMMAND_NAME_COLOR = "#f59e0b"
COMMAND_DESC_COLOR = "#cbd5e1"
KEYCAP_COLOR = "#14b8a6"

MODE_AUTHORIZED_COLOR = "#14b8a6"
MODE_STANDARD_COLOR = "#38bdf8"
APPROVAL_PROMPT_COLOR = "#f59e0b"
APPROVAL_AUTO_COLOR = "#22c55e"
APPROVAL_NEVER_COLOR = "#ef4444"
SERVICE_VALUE_COLOR = "#67e8f9"
MODEL_VALUE_COLOR = "#c4b5fd"
CWD_VALUE_COLOR = "#cbd5e1"

ROLE_STYLES: dict[str, dict[str, str]] = {
    "user": {
        "title": "用户输入",
        "border_style": USER_BORDER_COLOR,
        "text_style": f"bold {USER_TEXT_COLOR}",
    },
    "assistant": {
        "title": "智能体输出",
        "border_style": ASSISTANT_BORDER_COLOR,
        "text_style": ASSISTANT_TEXT_COLOR,
    },
    "system": {
        "title": "系统提示",
        "border_style": SYSTEM_BORDER_COLOR,
        "text_style": TEXT_PRIMARY,
    },
    "error": {
        "title": "运行错误",
        "border_style": ERROR_BORDER_COLOR,
        "text_style": f"bold {ERROR_TEXT_COLOR}",
    },
}

SYSTEM_LABEL_STYLE = TEXT_MUTED
SYSTEM_VALUE_STYLE = TEXT_PRIMARY
SYSTEM_VALUE_STYLES: dict[str, str] = {
    "authorized": f"bold {MODE_AUTHORIZED_COLOR}",
    "standard": f"bold {MODE_STANDARD_COLOR}",
    "prompt": f"bold {APPROVAL_PROMPT_COLOR}",
    "auto": f"bold {APPROVAL_AUTO_COLOR}",
    "never": f"bold {APPROVAL_NEVER_COLOR}",
    "service": f"bold {SERVICE_VALUE_COLOR}",
    "model": f"bold {MODEL_VALUE_COLOR}",
    "cwd": CWD_VALUE_COLOR,
}

COMMAND_NAME_STYLE = f"bold {COMMAND_NAME_COLOR}"
COMMAND_DESC_STYLE = COMMAND_DESC_COLOR
HINT_TITLE_STYLE = f"bold {TEXT_PRIMARY}"
KEYCAP_STYLE = f"bold {KEYCAP_COLOR}"
