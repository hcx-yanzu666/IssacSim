"""当前 Isaac Sim 5.1 会话中观察到的 Franka 实例几何显示兼容处理。

现象：物理/ USD 连杆位置正确，画面中手掌和手指却断开或不显示。
实测将这三处 geometry 取消实例化后恢复。这里只确认该绕行办法有效，
未证明底层渲染器的具体缺陷。改在 session layer，不保存或改写官方资产。
"""
from pxr import Usd
import omni.timeline


def prepare_franka_visuals(stage, robot_path='/World/Franka'):
    paths = [f'{robot_path}/{name}/geometry'
             for name in ('panda_hand', 'panda_leftfinger', 'panda_rightfinger')]
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
        print('[显示兼容] 会话层取消手掌/手指几何实例化；关节和模型尺寸不变。')
    return changed
