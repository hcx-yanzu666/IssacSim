"""第 3 课：给驱动器下目标，观察真实夹爪开合。

在 Isaac Sim Script Editor 用 runpy.run_path(..., run_name='__main__') 运行。
自动播放：张开 → 闭合 → 再张开 → 暂停（保留结果供观察）。
当前是空夹爪实验，没有抓取方块，也不保存 USD。
"""
import asyncio
import builtins
import runpy
from pathlib import Path

import numpy as np
import omni.kit.app
import omni.timeline
import omni.usd
from isaacsim.core.utils.types import ArticulationAction
from pxr import UsdPhysics

# 【你的实验】默认每根手指目标 30 mm。之后改成 0.015，观察开口变化。
# 单根手指位移不等于整个夹爪开口。本资产每根手指范围为 0～0.04 m。
OPEN_POSITION_M = 0.03
HOLD_SECONDS = 3.0
_TASK_KEY = '_isaac_lesson03_gripper_task'


class GripperLesson:
    def __init__(self):
        # 复用上一课初始化；模块不是 __main__，不会额外打印上一课整张表。
        lesson02 = runpy.run_path(str(Path(__file__).with_name('02_read_joint_state.py')))
        self.robot, prim = lesson02['initialize_robot']()
        names = list(self.robot.dof_names)
        self.driver = names.index('panda_finger_joint1')
        self.follower = names.index('panda_finger_joint2')
        self.stage = omni.usd.get_context().get_stage()
        joint = UsdPhysics.PrismaticJoint(
            self.stage.GetPrimAtPath('/World/Franka/panda_hand/panda_finger_joint1')
        )
        self.limits = (joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get())
        self.controller = self.robot.get_articulation_controller()

    def command(self, target_m):
        if not np.isfinite(target_m) or not self.limits[0] <= target_m <= self.limits[1]:
            raise ValueError(f'目标 {target_m} 超出关节范围 {self.limits}')
        # 关键 API：只控制第一个手指，第二个手指由已确认的 mimic 约束跟随。
        # joint_indices 与 joint_positions 等长，这里只发送一个关节目标。
        action = ArticulationAction(
            joint_positions=np.array([target_m], dtype=np.float32),
            joint_indices=np.array([self.driver], dtype=np.int32),
        )
        self.controller.apply_action(action)
        # apply_action 只是下指令；必须继续推进物理步才会产生运动。
        # 不使用 set_joint_positions 把手指直接瞬移到目标。

    def report(self, target_m):
        q = self.robot.get_joint_positions()
        actual = [float(q[self.driver]), float(q[self.follower])]
        passed = all(abs(value - target_m) < 0.001 for value in actual)
        print(f'[第3课] 目标={target_m * 1000:.1f} mm/手指，'
              f'实际=({actual[0] * 1000:.2f}, {actual[1] * 1000:.2f}) mm，'
              f'两指误差均小于1mm={passed}')
        return {'target_m': target_m, 'actual_m': actual, 'passed': passed}


async def demo():
    timeline = omni.timeline.get_timeline_interface()
    if timeline.is_playing():
        raise RuntimeError('请先暂停现有仿真再运行，避免两个控制程序同时工作。')
    lesson = GripperLesson()
    if not lesson.limits[0] <= OPEN_POSITION_M <= lesson.limits[1]:
        raise ValueError('OPEN_POSITION_M 超出夹爪关节限制。')
    if not np.isfinite(HOLD_SECONDS) or HOLD_SECONDS <= 0:
        raise ValueError('HOLD_SECONDS 必须为正数。')
    app = omni.kit.app.get_app()
    try:
        # Script Editor 由 Kit 自己推进帧；异步让出执行权，不能用阻塞 while/sleep。
        timeline.play()
        for label, target in [('张开', OPEN_POSITION_M), ('闭合', 0.0), ('再张开', OPEN_POSITION_M)]:
            lesson.command(target)
            print(f'[第3课] {label}，请观察两根手指。')
            start_time = timeline.get_current_time()
            while timeline.get_current_time() - start_time < HOLD_SECONDS:
                await app.next_update_async()
                if omni.usd.get_context().get_stage() != lesson.stage:
                    print('[第3课] 场景已切换，结束实验。')
                    return
                if not timeline.is_playing():
                    print('[第3课] 用户暂停或停止，结束实验；重新运行可再做一次。')
                    return
            lesson.report(target)
        print('[第3课] 完成，已暂停在张开状态。这验证了驱动与联动，尚未验证抓取。')
    finally:
        # Pause 保留末态；Stop 会使运行时句柄失效，下一次运行会重新初始化。
        if omni.usd.get_context().get_stage() == lesson.stage and timeline.is_playing():
            timeline.pause()


def start():
    bottle_task = getattr(builtins, '_isaac_lesson09_task', None)
    if bottle_task is not None and not bottle_task.done():
        print('[第3课] 瓶子任务正在运行，请先暂停并等待结束。')
        return
    batch_task = getattr(builtins, '_isaac_lesson07_task', None)
    if batch_task is not None and not batch_task.done():
        print('[第3课] 批量实验正在运行，请先暂停并等待结束。')
        return
    previous = getattr(builtins, _TASK_KEY, None)
    if previous is not None and not previous.done():
        print('[第3课] 实验已在运行，请等待完成，或按 Pause 中止。')
        return previous
    task = asyncio.ensure_future(demo())
    setattr(builtins, _TASK_KEY, task)

    def show_error(done):
        if not done.cancelled() and done.exception() is not None:
            print(f'[第3课] 未完成：{done.exception()}')

    task.add_done_callback(show_error)
    return task


# TODO：你把 OPEN_POSITION_M 改成 0.015，先预测结果再运行。
# TODO（下节）：机械臂末端移动到方块上方，再讨论 IK 和运动生成。
if __name__ == '__main__':
    start()
