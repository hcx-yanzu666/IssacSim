"""Compatibility entry. Implementation: src/bottle_sim/ros2_bridge.py."""
from pathlib import Path as _CompatPath
_target = _CompatPath(__file__).resolve().parents[1] / 'src/bottle_sim/ros2_bridge.py'
__file__ = str(_target)
exec(compile(_target.read_text(encoding="utf-8"), str(_target), "exec"), globals())
