"""第5课：到上方 → 下降 → 合拢 → 抬起 → 保持。

先看 PickLesson.tick 的阶段切换。运动学沿用第四课，不用重学求解器。
运行于 Isaac Sim 5.1 Script Editor。自动 Stop 后从场景起始状态重新实验。
只用物理接触/摩擦抓取，不绑定方块、不在运行中设置方块位置。
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
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.types import ArticulationAction

LIFT_HEIGHT_M = 0.12  # 【你的实验】成功后改为0.16，观察方块实际抬升高度。
CUBE_MASS_KG = 0.10
MOVE_SPEED_M_S = 0.04
CLOSE_SECONDS = 2.0
HOLD_SECONDS = 1.0
TASK_KEY = '_isaac_lesson05_task'
RESULT_KEY = '_isaac_lesson05_result'
OUTPUT_PATH = Path(__file__).resolve().parent.parent / 'outputs' / 'lesson05_latest.json'

lesson04 = runpy.run_path(str(Path(__file__).with_name('04_hover_above_cube.py')))


def prepare_contact_material():
    """在Stop后、物理初始化前明确实验参数，仅写会话层，不保存USD。"""
    stage = omni.usd.get_context().get_stage()
    cube = stage.GetPrimAtPath('/World/PracticeCube') if stage else None
    if not cube or not cube.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError('请打开包含动态方块的Franka练习场景。')
    compat = runpy.run_path(str(Path(__file__).with_name('franka_visual_compat.py')))
    compat['prepare_franka_visuals'](stage)
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        # 当前资产连杆惯量约1e-6、手指约1e-10 kg*m²，接触时发生抖动。
        # 保留质量，惯量用0哨兵让PhysX按碰撞几何重新估算；不是实机标定值。
        for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/Franka')):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                mass = UsdPhysics.MassAPI.Apply(prim)
                mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0))
                mass.CreatePrincipalAxesAttr(Gf.Quatf(1))
        articulation = PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath('/World/Franka'))
        articulation.CreateSolverVelocityIterationCountAttr(8)
        UsdPhysics.MassAPI.Apply(cube).CreateMassAttr(CUBE_MASS_KG)
        material = UsdShade.Material.Define(stage, '/World/Lesson05GripMaterial')
        physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics.CreateStaticFrictionAttr(0.8)
        physics.CreateDynamicFrictionAttr(0.6)
        physics.CreateRestitutionAttr(0.0)
        for path in ['/World/PracticeCube',
                     '/World/Franka/panda_leftfinger/geometry/panda_leftfinger',
                     '/World/Franka/panda_rightfinger/geometry/panda_rightfinger']:
            prim = stage.GetPrimAtPath(path)
            if not prim:
                raise RuntimeError(f'缺少碰撞对象：{path}')
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose='physics')
            collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            collision.CreateContactOffsetAttr(0.002)
            collision.CreateRestOffsetAttr(0.0)


class PickLesson(lesson04['HoverLesson']):
    def __init__(self):
        if not 0.08 <= LIFT_HEIGHT_M <= 0.20:
            raise ValueError('本课抬升高度限制为0.08～0.20m。')
        super().__init__(hover_height=0.15)
        # 第四课已经准备好robot、controller、cube、motion及TCP坐标变换。
        self.phase = 'Hover'
        self.phase_time = 0.0
        self.phase_trace = [{'phase': 'Hover', 'time': 0.0}]
        self.stable_seconds = 0.0
        self.table_z = float(self.table.center[2] + self.table.size[2] / 2)
        size = lesson04['ExistingCuboid'](self.stage, '/World/PracticeCube').size
        if np.max(np.abs(size - 0.05)) > 0.002:
            raise ValueError('本课抓取点仅针对当前5cm方块，请勿直接套用到其他物体。')
        self.cube_half_size = size / 2
        if abs(self.cube_start[2] - (self.table_z + size[2]/2)) > 0.01:
            raise ValueError('初始方块不在桌面上，请恢复场景起始状态。')
        self.grasp_tcp = self.cube_start.copy()  # 抓取中心对准方块中心
        self.lift_tcp = self.grasp_tcp + np.array([0., 0., LIFT_HEIGHT_M])
        self.command_tcp = self.target_tcp.copy()
        self.max_lift_m = 0.0
        self.hold_reference = None
        self.hold_drift_m = 0.0
        # 抓取点与抬升终点也做IK预检查；IK只检查可达，不执行瞬移。
        for label, target in [('grasp', self.grasp_tcp), ('lift', self.lift_tcp)]:
            _, ok = self.kinematics.compute_inverse_kinematics(
                target - self.target_r @ self.tcp_in_gripper, self.target_q,
                position_tolerance=0.001, orientation_tolerance=0.01)
            if not ok:
                raise RuntimeError(f'{label}目标IK未收敛，未启动抓取。')
        self.grip(0.035)
        print('[第5课] 到上方 → 下降 → 合拢 → 抬起 → 保持；物体质量0.10kg。')

    def grip(self, position):
        self.controller.apply_action(ArticulationAction(
            joint_positions=np.array([position]), joint_indices=np.array([self.finger])))

    def transition(self, phase):
        self.phase = phase
        self.phase_time = 0.0
        self.stable_seconds = 0.0
        self.phase_trace.append({'phase': phase, 'time': self.elapsed})
        print(f'[第5课] 阶段：{phase}')

    def set_tcp(self, goal, dt):
        # 让笛卡尔目标每步最多移动 speed*dt，避免突然把目标跳到桌边。
        delta = goal - self.command_tcp
        distance = float(np.linalg.norm(delta))
        if distance > 0:
            self.command_tcp += delta * min(1.0, MOVE_SPEED_M_S * dt / distance)
        model_goal = self.command_tcp - self.target_r @ self.tcp_in_gripper
        self.policy.set_end_effector_target(model_goal, self.target_q)

    def close_to(self, tcp, rotation, goal, dt):
        reached = (np.linalg.norm(tcp-goal) < 0.006 and
                   lesson04['angle_error'](rotation, self.target_r) < np.deg2rad(3))
        self.stable_seconds = self.stable_seconds + dt if reached else 0.0
        return self.stable_seconds >= 0.3

    def tick(self, dt):
        if self.status != 'Running':
            return
        try:
            if omni.usd.get_context().get_stage() != self.stage or not self.robot.handles_initialized:
                raise RuntimeError('场景/物理句柄已变化')
            self.elapsed += float(dt)
            self.phase_time += float(dt)
            tcp, rotation = self.actual_pose()
            cube_p, cube_q = self.cube.get_world_pose()
            if not np.all(np.isfinite(np.r_[tcp, cube_p])):
                raise RuntimeError('位置出现非有限数值')
            lift = float(cube_p[2] - self.cube_start[2])
            self.max_lift_m = max(self.max_lift_m, lift)
            # 方块底面高度考虑旋转，避免把中心高度误当作底面离桌高度。
            bottom = float(cube_p[2] - (np.abs(quat_to_rot_matrix(cube_q)) @ self.cube_half_size)[2])
            if bottom < self.table_z - 0.005:
                raise RuntimeError('方块穿入/掉到桌面以下')
            if tcp[2] < self.table_z + 0.012:
                raise RuntimeError('抓取中心过低，停止下降')
            if self.phase_time > 18.0 or self.elapsed > 50.0:
                self.status, self.detail = 'TimedOut', f'{self.phase}阶段未按时完成'
                return

            # 【本课核心】根据阶段决定目标，不需要重新写运动学算法。
            if self.phase == 'Hover':
                if self.close_to(tcp, rotation, self.target_tcp, dt):
                    self.transition('Descend')
            elif self.phase == 'Descend':
                self.set_tcp(self.grasp_tcp, dt)
                if self.close_to(tcp, rotation, self.grasp_tcp, dt):
                    self.grip(0.0)  # 合拢目标为0，但碰到方块后实际位置不应为0。
                    self.transition('Close')
            elif self.phase == 'Close':
                if self.phase_time >= CLOSE_SECONDS:
                    q = self.robot.get_joint_positions()
                    if float(q[self.finger]) < 0.010:
                        raise RuntimeError('夹爪几乎空合，没有夹住预期5cm方块')
                    self.transition('Lift')
            elif self.phase == 'Lift':
                self.set_tcp(self.lift_tcp, dt)
                if self.close_to(tcp, rotation, self.lift_tcp, dt):
                    self.hold_reference = cube_p - tcp
                    self.transition('Hold')
            elif self.phase == 'Hold':
                # 成功依据物体，而不是只看手臂到达或夹爪合拢。
                relative = cube_p - tcp
                self.hold_drift_m = max(self.hold_drift_m, float(np.linalg.norm(relative-self.hold_reference)))
                if lift < LIFT_HEIGHT_M * 0.65 or np.linalg.norm(relative) > 0.04:
                    raise RuntimeError('手臂已抬起，但方块没有随夹爪稳定抬起')
                if self.hold_drift_m > 0.01:
                    raise RuntimeError('保持阶段方块明显滑动')
                if self.phase_time >= HOLD_SECONDS:
                    self.status = 'Succeeded'
                    self.detail = '方块被提起并稳定保持；尚未搬运和放置'
                    return
            self.controller.apply_action(self.motion.get_next_articulation_action(physics_dt=float(dt)))
        except Exception as exc:
            self.status, self.detail = 'Failed', str(exc)

    def report(self):
        result = {'status': self.status, 'phase': self.phase, 'detail': self.detail,
                  'elapsed_seconds': self.elapsed, 'lift_target_m': LIFT_HEIGHT_M,
                  'cube_mass_kg': CUBE_MASS_KG, 'max_lift_m': self.max_lift_m,
                  'hold_drift_m': self.hold_drift_m, 'phases': self.phase_trace}
        result['physics_configuration'] = {
            'inertia_source': 'estimated_from_collision_geometry',
            'static_friction': 0.8, 'dynamic_friction': 0.6,
            'contact_offset_m': 0.002, 'rest_offset_m': 0.0,
            'articulation_velocity_iterations': 8,
        }
        if omni.usd.get_context().get_stage() == self.stage and self.robot.handles_initialized:
            p, _ = self.cube.get_world_pose()
            q = self.robot.get_joint_positions()
            result.update(cube_position_m=p.tolist(), actual_lift_m=float(p[2]-self.cube_start[2]),
                          finger_position_m=float(q[self.finger]))
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        setattr(builtins, RESULT_KEY, result)
        print('[第5课] '+json.dumps(result, ensure_ascii=False))
        return result


async def demo():
    timeline = omni.timeline.get_timeline_interface()
    if timeline.is_playing():
        raise RuntimeError('请先暂停前一任务。')
    # 本课每次从Stop的场景起始状态运行；不是瞬移抓取，只在开始恢复环境。
    timeline.stop()
    await omni.kit.app.get_app().next_update_async()
    prepare_contact_material()
    # 场景结构/物理属性变更通知需先处理，再建立新的物理句柄。
    await omni.kit.app.get_app().next_update_async()
    lesson = PickLesson()
    subscription = omni.physx.get_physx_interface().subscribe_physics_step_events(lesson.tick)
    try:
        timeline.play()
        while lesson.status == 'Running':
            await omni.kit.app.get_app().next_update_async()
            if omni.usd.get_context().get_stage() != lesson.stage or not timeline.is_playing():
                lesson.status, lesson.detail = 'Cancelled', '用户暂停/停止或切换场景'
                break
    finally:
        subscription = None
        if omni.usd.get_context().get_stage() == lesson.stage and timeline.is_playing():
            timeline.pause()
        lesson.report()


def start():
    for key in (TASK_KEY, '_isaac_lesson03_gripper_task', '_isaac_lesson04_task', '_isaac_lesson07_task', '_isaac_lesson09_task'):
        previous = getattr(builtins, key, None)
        if previous is not None and not previous.done():
            print('[第5课] 前一任务尚未结束，请先暂停并等待。')
            return
    setattr(builtins, RESULT_KEY, {'status': 'Running', 'phase': 'Initialize'})
    task = asyncio.ensure_future(demo())
    setattr(builtins, TASK_KEY, task)
    def finished(t):
        if not t.cancelled() and t.exception() is not None:
            result = {'status':'Failed', 'phase':'Initialize', 'detail':str(t.exception())}
            setattr(builtins, RESULT_KEY, result)
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            print('[第5课] '+str(result))
    task.add_done_callback(finished)
    return task


if __name__ == '__main__':
    start()
