"""Franka 实例几何显示兼容处理。

现象：不同 Isaac Sim/RTX 组合下，实例化的 articulation link geometry
可能不被视口正常绘制。为保证连杆、手掌和夹爪使用一致的渲染路径，
在 session layer 取消这些 geometry prim 的实例化；不保存或改写官方资产。
"""

from pathlib import Path as _ProjectPath
import sys as _sys
PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in _sys.path:
    _sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pxr import Usd
import omni.timeline


def prepare_franka_visuals(stage, robot_path='/World/Franka'):
    link_names = tuple(f'panda_link{i}' for i in range(8))
    paths = [f'{robot_path}/{name}/geometry'
             for name in (*link_names, 'panda_hand', 'panda_leftfinger', 'panda_rightfinger')]
    pending = [path for path in paths if stage.GetPrimAtPath(path)
               and stage.GetPrimAtPath(path).IsInstance()]
    if not pending:
        return []
    timeline = omni.timeline.get_timeline_interface()
    if timeline.is_playing():
        raise RuntimeError('请先停止仿真再应用几何实例兼容处理。')
    if not timeline.is_stopped():
        # 改实例结构会使现有物理视图失效；暂停态先Stop，后续重新初始化。
        print('[显示兼容] 首次处理需Stop重建物理，场景会恢复起始状态。')
        timeline.stop()
    changed = []
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for path in pending:
            prim = stage.GetPrimAtPath(path)
            if prim and prim.IsInstance():
                prim.SetInstanceable(False)
                changed.append(str(prim.GetPath()))
    if changed:
        print(f'[显示兼容] 会话层取消 {len(changed)} 个机器人连杆/末端几何实例化；关节和模型尺寸不变。')
    return changed
