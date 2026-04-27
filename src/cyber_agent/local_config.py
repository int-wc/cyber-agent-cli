import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOCAL_CONFIG_FILENAME = ".cyber-agent-cli.json"


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
    当前仓库暂无统一的全局配置体系，因此先采用工作目录级配置，
    避免不同项目之间相互污染授权目录。
    """
    resolved_base_dir = (base_dir or Path.cwd()).resolve()
    return resolved_base_dir / LOCAL_CONFIG_FILENAME


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
