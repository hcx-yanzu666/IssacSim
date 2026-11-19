"""Compatibility entry. Implementation: perception/yolo_worker.py."""
from pathlib import Path as _CompatPath
_target = _CompatPath(__file__).resolve().parents[1] / 'perception/yolo_worker.py'
__file__ = str(_target)
exec(compile(_target.read_text(encoding="utf-8"), str(_target), "exec"), globals())
