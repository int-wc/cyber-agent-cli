import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOCAL_CONFIG_FILENAME = ".cyber-agent-cli.json"
CYBER_DIR_NAME = ".cyber"
CYBER_AGENT_HOME_ENV = "CYBER_AGENT_HOME"


def get_application_home() -> Path:
    """返回 Cyber Agent 的应用主目录，用于存放默认配置和运行数据。"""
    configured_home = os.getenv(CYBER_AGENT_HOME_ENV, "").strip()
    if configured_home:
        return Path(configured_home).expanduser().resolve()

    source_path = Path(__file__).resolve()
    for parent in source_path.parents:
        if (
            (parent / "pyproject.toml").is_file()
            and (parent / "src" / "cyber_agent").is_dir()
        ):
            return parent
    return Path.home() / ".cyber-agent-cli"


def get_application_env_file() -> Path:
    """返回默认环境配置文件路径。"""
    return get_application_home() / ".env"


@dataclass(slots=True)
class LocalCliConfig:
    """表示当前工作目录下的本地 CLI 配置。"""

    allow_paths: list[Path]


def normalize_allowed_roots(allowed_roots: list[Path | str]) -> list[Path]:
    """轻量规范化允许路径，避免读取本地配置时导入完整工具集合。"""
    normalized_roots: list[Path] = []
    seen_roots: set[Path] = set()

    for raw_root in allowed_roots:
        root_path = Path(raw_root).expanduser().resolve()
        if root_path in seen_roots:
            continue
        normalized_roots.append(root_path)
        seen_roots.add(root_path)

    return normalized_roots


def get_local_config_path(base_dir: Path | None = None) -> Path:
    """
    返回当前工作目录对应的本地配置文件路径。
    优先查找当前目录及父目录中已存在的配置文件；
    若都不存在，则在当前工作目录下创建。
    """
    resolved_base_dir = (base_dir or Path.cwd()).resolve()
    return resolved_base_dir / LOCAL_CONFIG_FILENAME


def find_local_config_file(base_dir: Path | None = None) -> Path | None:
    """在当前目录及父目录中查找已存在的本地配置文件。"""
    current = (base_dir or Path.cwd()).resolve()
    for _ in range(10):  # 最多向上搜索 10 层
        candidate = current / LOCAL_CONFIG_FILENAME
        if candidate.exists():
            return candidate
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def get_global_config_path() -> Path:
    """返回用户家目录下的全局配置文件路径。"""
    return Path.home() / LOCAL_CONFIG_FILENAME


def load_config_with_fallback(base_dir: Path | None = None) -> LocalCliConfig:
    """按优先级加载配置：本地目录 → 父目录回溯 → 全局配置 → 空配置。"""
    # 1. 当前目录的直接配置
    local_path = get_local_config_path(base_dir)
    if local_path.exists():
        return load_local_cli_config(base_dir)

    # 2. 父目录回溯
    found = find_local_config_file(base_dir)
    if found is not None:
        return load_local_cli_config(found.parent)

    # 3. 全局配置
    global_path = get_global_config_path()
    if global_path.exists():
        return load_local_cli_config(global_path.parent)

    # 4. 空配置
    return LocalCliConfig(allow_paths=[])


def _normalize_allow_paths(raw_value: Any) -> list[Path]:
    """将配置中的允许路径列表规范化为绝对路径数组。"""
    if raw_value is None:
        return []
    if not isinstance(raw_value, list):
        raise ValueError("本地配置中的 allow_paths 必须为数组。")

    path_items: list[Path | str] = []
    for item in raw_value:
        if not isinstance(item, str):
            raise ValueError("本地配置中的 allow_paths 仅允许包含字符串路径。")
        stripped_item = item.strip()
        if not stripped_item:
            continue
        path_items.append(stripped_item)

    return normalize_allowed_roots(path_items)


def load_local_cli_config(base_dir: Path | None = None) -> LocalCliConfig:
    """读取当前工作目录下的本地配置；若不存在则返回空配置。"""
    config_path = get_local_config_path(base_dir)
    if not config_path.exists():
        return LocalCliConfig(allow_paths=[])

    try:
        raw_data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"本地配置文件不是合法 JSON：{config_path}") from exc

    if not isinstance(raw_data, dict):
        raise ValueError(f"本地配置文件内容必须为对象：{config_path}")

    return LocalCliConfig(
        allow_paths=_normalize_allow_paths(raw_data.get("allow_paths")),
    )


def save_local_cli_config(
    config: LocalCliConfig,
    base_dir: Path | None = None,
) -> Path:
    """保存本地 CLI 配置。"""
    config_path = get_local_config_path(base_dir)
    serialized_data = {
        "allow_paths": [str(path) for path in normalize_allowed_roots(config.allow_paths)],
    }
    config_path.write_text(
        json.dumps(serialized_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return config_path


def find_cyber_dir(base_dir: Path | None = None) -> Path:
    """返回集中应用数据目录 .cyber/。

    无显式 base_dir 时固定使用应用主目录，避免在任意启动目录创建隐藏数据。
    有显式 base_dir 时保留原项目回溯策略：
    - 若找到已有 .cyber/ → 直接返回
    - 若找到含 .git/ 的目录但无 .cyber/ → 在其下创建 .cyber/ 并返回
    - 若都没找到 → 在 base_dir（或 cwd）下创建 .cyber/ 并返回
    """
    if base_dir is None:
        target = get_application_home() / CYBER_DIR_NAME
        target.mkdir(parents=True, exist_ok=True)
        return target

    start = base_dir.resolve()
    candidate_git_root: Path | None = None
    current = start
    for _ in range(10):
        if (current / ".git").exists():
            candidate_git_root = current
        if (current / CYBER_DIR_NAME).is_dir():
            return current / CYBER_DIR_NAME
        parent = current.parent
        if parent == current:
            break
        current = parent

    target = (candidate_git_root or start) / CYBER_DIR_NAME
    target.mkdir(parents=True, exist_ok=True)
    return target


def find_data_dir(dirname: str, base_dir: Path | None = None) -> Path:
    """查找数据目录，默认集中到应用主目录的 .cyber/<sub>。

    当 base_dir 显式传入时，始终优先使用 base_dir 本身，方便测试或显式隔离。
    dirname 示例: ".cyber-agent-cli-sessions", ".cyber-agent-cli-capabilities"
    """
    if base_dir is not None:
        return base_dir.resolve() / dirname

    if dirname.startswith(".cyber-agent-cli-"):
        short_name = dirname[len(".cyber-agent-cli-"):]
        return find_cyber_dir(None) / short_name

    return get_application_home() / dirname


def merge_allow_paths(*path_groups: list[Path | str]) -> list[Path]:
    """合并多组允许路径，并按顺序去重。"""
    merged_paths: list[Path | str] = []
    for path_group in path_groups:
        merged_paths.extend(path_group)
    return normalize_allowed_roots(merged_paths)


def add_allow_path_to_local_config(
    path: Path | str,
    base_dir: Path | None = None,
) -> tuple[Path, bool, Path]:
    """
    将目录写入本地配置。
    返回规范化后的目录路径、本次是否新增，以及配置文件路径。
    """
    target_path = Path(path).expanduser().resolve()
    if not target_path.exists():
        raise ValueError(f"目录不存在：{target_path}")
    if not target_path.is_dir():
        raise ValueError(f"目标路径不是目录：{target_path}")

    local_config = load_local_cli_config(base_dir)
    if target_path in local_config.allow_paths:
        return target_path, False, get_local_config_path(base_dir)

    local_config.allow_paths.append(target_path)
    config_path = save_local_cli_config(local_config, base_dir)
    return target_path, True, config_path
