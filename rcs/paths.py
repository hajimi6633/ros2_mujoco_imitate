"""路径工具：定位仓库根（config/ 与 models/ 数据文件新旧代码共享）。"""
from __future__ import annotations
import os
from pathlib import Path
import yaml

# 仓库根：rcs/paths.py 的上一级
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(rel: str) -> str:
    """相对仓库根的路径 → 绝对路径；已是绝对路径则原样返回。"""
    if os.path.isabs(rel) or (len(rel) >= 2 and rel[1] == ':'):
        return rel
    return str(PROJECT_ROOT / rel)


def load_yaml(rel: str) -> dict:
    with open(project_path(rel), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
