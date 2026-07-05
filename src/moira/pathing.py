from __future__ import annotations

from pathlib import Path


def get_project_root(config_path: str | Path) -> Path:
    resolved_config_path = Path(config_path).resolve()
    start_dir = (
        resolved_config_path
        if resolved_config_path.is_dir()
        else resolved_config_path.parent
    )
    for candidate in (start_dir, *start_dir.parents):
        if (candidate / "src" / "moira").is_dir() and (
            (candidate / "pyproject.toml").is_file()
            or (candidate / ".git").exists()
        ):
            return candidate
    return resolved_config_path.parent


def resolve_project_path(path_value: str | Path, *, config_path: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (get_project_root(config_path) / path).resolve()
