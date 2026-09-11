"""pytest 收集本目录测试时注入 legacy 路径，
使旧测试的 `from src.xxx` 导入以 legacy/ 为根解析。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
