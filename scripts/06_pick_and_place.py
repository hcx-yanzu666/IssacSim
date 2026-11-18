"""第6课：复用第五课抓取，再搬运 → 放下 → 松开 → 撤离 → 检查。

运行于 Isaac Sim 5.1 Script Editor。先暂停旧任务；启动时 Stop 恢复起点。
先读 tick() 的五个新阶段。方块只靠接触/摩擦运动，不吸附、不瞬移。
TODO（你的练习）：将 PLACE_XY_M 的 Y 从0.20改为0.15，比较报告位置。
"""
import asyncio
import builtins
import json
import runpy
from pathlib import Path

import numpy as np
import omni.kit.app
import omni.physx
import omni.timeline
import omni.usd
from isaacsim.core.utils.rotations import quat_to_rot_matrix

PLACE_XY_M = (0.18, 0.20)  # B点世界坐标XY，单位米；Z根据桌面和方块尺寸计算。
RELEASE_CLEARANCE_M = 0.003  # 松开时留3mm名义间隙，避免把物体向桌面压。
RETREAT_HEIGHT_M = 0.12
PLACE_TOLERANCE_M = 0.015
SETTLE_SECONDS = 1.0
TASK_KEY = '_isaac_lesson06_task'
RESULT_KEY = '_isaac_lesson06_result'
OUTPUT_PATH = Path(__file__).resolve().parent.parent / 'outputs' / 'lesson06_latest.json'
lesson05 = runpy.run_path(str(Path(__file__).with_name('05_pick_cube.py')))


class PickAndPlaceLesson(lesson05['PickLesson']):
    def __init__(self):
        xy = np.asarray(PLACE_XY_M, dtype=float)
        if xy.shape != (2,) or not np.all(np.isfinite(xy)):
            raise ValueError('PLACE_XY_M需要两个有限数值，单位米。')
        super().__init__()
        self.place_center = np.r_[xy, self.table_z + self.cube_half_size[2]]
        # 本课只支持当前轴对齐桌面和5cm方块，B点须留足桌边余量。
        available = self.table.size[:2]/2 - self.cube_half_size[:2] - 0.05
        if np.any(np.abs(xy-self.table.center[:2]) > available):
            raise ValueError('B点太靠近桌边或不在桌面范围内。')
        if np.linalg.norm(xy-self.cube_start[:2]) < 0.08:
            raise ValueError('本课A/B点至少相隔8cm，便于观察搬运。')
        self.transport_reference = None
        self.transport_max_drift_m = 0.0
        self.settle_time = 0.0
        self.settle_reference = None
        self.metrics = {}
        self.pick_verified = False
        self.release_commanded = False
        # 终点可达检查不等于整条路径无碰撞；仍使用第四课的桌面代理。
        self.check_targets([
            self.place_center + [0., 0., lesson05['LIFT_HEIGHT_M']],
            self.place_center + [0., 0., RELEASE_CLEARANCE_M],
            self.place_center + [0., 0., RETREAT_HEIGHT_M],
        ])
        print(f'[第6课] A={self.cube_start.round(4)} → B={self.place_center.round(4)}')

    def check_targets(self, targets):
        for target in targets:
            _, ok = self.kinematics.compute_inverse_kinematics(
                target-self.target_r @ self.tcp_in_gripper, self.target_q,
                position_tolerance=0.001, orientation_tolerance=0.01)
            if not ok:
                raise RuntimeError('放置/撤离目标IK未收敛，请缩短B点距离。')

    def transition(self, phase):
        self.phase, self.phase_time, self.stable_seconds = phase, 0.0, 0.0
        self.phase_trace.append({'phase': phase, 'time': self.elapsed})
        print(f'[第6课] 阶段：{phase}')

    def begin_transfer(self):
        # 抓住后记录物体相对TCP的实际偏移，放置时补偿；不更改物体位置。
        tcp, _ = self.actual_pose()
        cube_p, _ = self.cube.get_world_pose()
        self.transport_reference = cube_p-tcp
        self.transfer_tcp = self.place_center + [0., 0., lesson05['LIFT_HEIGHT_M']] - self.transport_reference
        self.place_tcp = self.place_center + [0., 0., RELEASE_CLEARANCE_M] - self.transport_reference
        self.retreat_tcp = self.place_tcp + [0., 0., RETREAT_HEIGHT_M]
        self.check_targets([self.transfer_tcp, self.place_tcp, self.retreat_tcp])
        self.pick_verified = True
        self.status, self.detail = 'Running', ''
        self.transition('Transfer')

    def placement_metrics(self, tcp, cube_p, cube_q):
        bottom = float(cube_p[2] - (np.abs(quat_to_rot_matrix(cube_q)) @ self.cube_half_size)[2])
        return {
            'xy_error_m': float(np.linalg.norm(cube_p[:2]-self.place_center[:2])),
            'center_height_error_m': float(abs(cube_p[2]-self.place_center[2])),
            'bottom_clearance_m': bottom-self.table_z,
            'linear_speed_m_s': float(np.linalg.norm(self.cube.get_linear_velocity())),
            'angular_speed_rad_s': float(np.linalg.norm(self.cube.get_angular_velocity())),
            'tcp_above_cube_m': float(tcp[2]-cube_p[2]),
            'finger_position_m': float(self.robot.get_joint_positions()[self.finger]),
        }

    def tick(self, dt):
        if self.status != 'Running':
            return
        try:
            # ① 原来的抓取五阶段完全复用。Hold成功只是本课的中间里程碑。
            if self.phase in ('Hover', 'Descend', 'Close', 'Lift', 'Hold'):
                super().tick(dt)
                if self.status == 'Succeeded':
                    self.begin_transfer()
                return

            if omni.usd.get_context().get_stage() != self.stage or not self.robot.handles_initialized:
                raise RuntimeError('场景/物理句柄已变化')
            dt = float(dt)
            self.elapsed += dt
            self.phase_time += dt
            tcp, rotation = self.actual_pose()
            cube_p, cube_q = self.cube.get_world_pose()
            if not np.all(np.isfinite(np.r_[tcp, cube_p, cube_q])):
                raise RuntimeError('物体/末端位姿出现非有限数值')
            self.metrics = self.placement_metrics(tcp, cube_p, cube_q)
            if not np.all(np.isfinite(list(self.metrics.values()))):
                raise RuntimeError('物理状态出现非有限数值')
            self.max_lift_m = max(self.max_lift_m, float(cube_p[2]-self.cube_start[2]))
            if self.metrics['bottom_clearance_m'] < -0.005 or tcp[2] < self.table_z+0.012:
                raise RuntimeError('方块或TCP过低，停止任务')
            if self.phase_time > 18.0 or self.elapsed > 90.0:
                self.status, self.detail = 'TimedOut', f'{self.phase}阶段未按时完成'
                return

            # 松开之前监测物体是否仍在夹爪中，搬运滑落不能算成功。
            if self.phase in ('Transfer', 'Lower'):
                drift = float(np.linalg.norm(cube_p-tcp-self.transport_reference))
                self.transport_max_drift_m = max(self.transport_max_drift_m, drift)
                if drift > 0.015:
                    raise RuntimeError('搬运/下降时方块相对夹爪滑动超过1.5cm')

            # ② 本课新增的五个阶段。
            if self.phase == 'Transfer':
                self.set_tcp(self.transfer_tcp, dt)
                if self.close_to(tcp, rotation, self.transfer_tcp, dt):
                    self.transition('Lower')
            elif self.phase == 'Lower':
                self.set_tcp(self.place_tcp, dt)
                if self.close_to(tcp, rotation, self.place_tcp, dt):
                    self.grip(0.035)
                    self.release_commanded = True
                    self.transition('Release')
            elif self.phase == 'Release':
                # 检查实际张开；仅下发张开指令不代表已释放。
                if self.phase_time >= 1.0 and self.metrics['finger_position_m'] > 0.032:
                    self.transition('Retreat')
            elif self.phase == 'Retreat':
                self.set_tcp(self.retreat_tcp, dt)
                if self.close_to(tcp, rotation, self.retreat_tcp, dt):
                    self.transition('Settle')
            elif self.phase == 'Settle':
                # ③ 观察物体与夹爪的实际状态，而不是仅看命令执行结束。
                m = self.metrics
                valid = (self.pick_verified and self.release_commanded
                         and m['xy_error_m'] < PLACE_TOLERANCE_M
                         and m['center_height_error_m'] < 0.006
                         and abs(m['bottom_clearance_m']) < 0.004
                         and m['linear_speed_m_s'] < 0.01
                         and m['angular_speed_rad_s'] < 0.1
                         and m['tcp_above_cube_m'] > 0.08
                         and m['finger_position_m'] > 0.032)
                if valid:
                    if self.settle_reference is None or np.linalg.norm(cube_p-self.settle_reference) > 0.002:
                        self.settle_reference, self.settle_time = cube_p.copy(), 0.0
                    self.settle_time += dt
                    if self.settle_time >= SETTLE_SECONDS:
                        self.status, self.detail = 'Succeeded', '方块已搬到B点，松开并撤离后在桌面稳定保持1秒'
                else:
                    self.settle_time, self.settle_reference = 0.0, None
                if self.phase_time >= 5.0 and self.status == 'Running':
                    raise RuntimeError('放置未达标：检查目标误差、桌面高度、速度与夹爪撤离距离')

            self.controller.apply_action(self.motion.get_next_articulation_action(physics_dt=dt))
        except Exception as exc:
            self.status, self.detail = 'Failed', str(exc)

    def report(self, save=True):
        result = {
            'status': self.status, 'phase': self.phase, 'detail': self.detail,
            'elapsed_seconds': self.elapsed, 'source_center_m': self.cube_start.tolist(),
            'place_target_m': self.place_center.tolist(), 'pick_verified': self.pick_verified,
            'release_commanded': self.release_commanded, 'max_lift_m': self.max_lift_m,
            'transport_max_drift_m': self.transport_max_drift_m,
            'placement': self.metrics, 'stable_seconds': self.settle_time, 'phases': self.phase_trace,
            'configuration': {'place_tolerance_m': PLACE_TOLERANCE_M,
                              'release_clearance_m': RELEASE_CLEARANCE_M,
                              'retreat_height_m': RETREAT_HEIGHT_M,
                              'move_speed_m_s': lesson05['MOVE_SPEED_M_S'],
                              'cube_mass_kg': lesson05['CUBE_MASS_KG'],
                              'physics_setup': 'lesson05.prepare_contact_material; collision-estimated inertia'},
        }
        if omni.usd.get_context().get_stage() == self.stage and self.robot.handles_initialized:
            try:
                p, _ = self.cube.get_world_pose()
                result['cube_position_m'] = p.tolist()
            except Exception as exc:
                result['final_pose_unavailable'] = str(exc)
        if save:
            save_result(result)
        return result


def save_result(result):
    setattr(builtins, RESULT_KEY, result)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[第6课] '+json.dumps(result, ensure_ascii=False))


async def demo():
    timeline = omni.timeline.get_timeline_interface()
    if timeline.is_playing():
        raise RuntimeError('请先暂停现有任务。')
    timeline.stop()
    await omni.kit.app.get_app().next_update_async()
    lesson05['prepare_contact_material']()
    await omni.kit.app.get_app().next_update_async()
    lesson = PickAndPlaceLesson()
    subscription = omni.physx.get_physx_interface().subscribe_physics_step_events(lesson.tick)
    try:
        timeline.play()
        while lesson.status == 'Running':
            await omni.kit.app.get_app().next_update_async()
            if lesson.status == 'Running' and (omni.usd.get_context().get_stage() != lesson.stage or not timeline.is_playing()):
                lesson.status, lesson.detail = 'Cancelled', '用户暂停/停止或切换场景'
    finally:
        subscription = None
        if omni.usd.get_context().get_stage() == lesson.stage and timeline.is_playing():
            timeline.pause()
        lesson.report()


def start():
    for key in (TASK_KEY, '_isaac_lesson03_gripper_task', '_isaac_lesson04_task', '_isaac_lesson05_task', '_isaac_lesson07_task', '_isaac_lesson09_task'):
        previous = getattr(builtins, key, None)
        if previous is not None and not previous.done():
            print('[第6课] 前一任务尚未结束，请先暂停并等待。')
            return
    save_result({'status': 'Running', 'phase': 'Initialize'})
    task = asyncio.ensure_future(demo())
    setattr(builtins, TASK_KEY, task)
    def finished(t):
        if t.cancelled():
            save_result({'status': 'Cancelled', 'phase': 'Task', 'detail': '异步任务被取消'})
        elif t.exception() is not None:
            save_result({'status': 'Failed', 'phase': 'Initialize', 'detail': str(t.exception())})
    task.add_done_callback(finished)
    return task


if __name__ == '__main__':
    start()
