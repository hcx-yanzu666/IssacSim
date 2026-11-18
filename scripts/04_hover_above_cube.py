"""第4课：Lula 验证 IK，RMPflow 生成运动，到方块上方悬停。

入口：Isaac Sim 5.1 Script Editor，runpy.run_path(..., run_name='__main__')。
不用自行求解 IK，不移动方块，不保存场景。要求暂停且无其他控制器运行。
先读 HoverLesson.__init__ 中的 ①②③，再读 tick 中的 ④。
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
from pxr import Usd, UsdGeom
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.rotations import quat_to_rot_matrix, rot_matrix_to_quat
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot_motion.motion_generation import (
    LulaKinematicsSolver, ArticulationKinematicsSolver,
    RmpFlow, ArticulationMotionPolicy, interface_config_loader,
)

# 【小实验】默认抓取中心位于方块中心上方15cm；跑通后改成0.20。
HOVER_HEIGHT_M = 0.15
POSITION_TOLERANCE_M = 0.005
ORIENTATION_TOLERANCE_RAD = np.deg2rad(3.0)
TIMEOUT_SECONDS = 20.0
TASK_KEY = '_isaac_lesson04_task'
RESULT_KEY = '_isaac_lesson04_result'


def angle_error(a, b):
    """两个旋转矩阵之间的夹角，单位rad。"""
    return float(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1)))


class ExistingCuboid:
    """为RMPflow提供已有Cube的只读几何接口，不重新创建资产或修改材质。

    使用已核对的5.1 add_cuboid接口，当前只支持无父级缩放的轴对齐Cube。
    这里用世界AABB作为保守障碍代理；未来复杂模型需另外构建代理。
    """
    def __init__(self, stage, path):
        self.prim_path = path
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsA(UsdGeom.Cube):
            raise ValueError(f'需要现有Cube：{path}')
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        self.center = np.array(bounds.GetMidpoint(), dtype=float)
        self.size = np.array(bounds.GetSize(), dtype=float)

    def get_size(self):
        return 1.0

    def get_local_scale(self):
        return self.size.copy()

    def get_world_pose(self):
        return self.center.copy(), np.array([1., 0., 0., 0.])


class HoverLesson:
    def __init__(self, hover_height=HOVER_HEIGHT_M, object_path='/World/PracticeCube',
                 target_offset=None, hand_rotation=None):
        if not np.isfinite(hover_height) or not 0.10 <= hover_height <= 0.30:
            raise ValueError('本课悬停高度限定在0.10～0.30m；先不做下降抓取。')
        lesson02 = runpy.run_path(str(Path(__file__).with_name('02_read_joint_state.py')))
        self.robot, _ = lesson02['initialize_robot']()
        self.stage = omni.usd.get_context().get_stage()
        if abs(UsdGeom.GetStageMetersPerUnit(self.stage) - 1.0) > 1e-6:
            raise ValueError('本课要求场景单位为米。')
        self.controller = self.robot.get_articulation_controller()
        self.hand = SingleRigidPrim('/World/Franka/panda_hand', reset_xform_properties=False)
        # 默认保持原方块课程；瓶子课程可指定已有刚体路径，复用运动学初始化。
        self.cube = SingleRigidPrim(object_path, reset_xform_properties=False)
        self.hand.initialize()
        self.cube.initialize()
        tool = self.stage.GetPrimAtPath('/World/Franka/panda_hand/tool_center')
        if not tool:
            raise ValueError('缺少tool_center，请先核对抓取中心。')
        self.tcp_local = np.array(UsdGeom.Xformable(tool).GetLocalTransformation().ExtractTranslation())

        # ① 加载官方Franka模型。求解器的base必须放到实际机器人底座位置。
        config = interface_config_loader.load_supported_motion_policy_config('Franka', 'RMPflow')
        self.frame = config['end_effector_frame_name']  # right_gripper
        self.solver = LulaKinematicsSolver(config['robot_description_path'], config['urdf_path'])
        self.base_p, self.base_q = self.robot.get_world_pose()
        self.solver.set_robot_base_pose(self.base_p, self.base_q)
        self.kinematics = ArticulationKinematicsSolver(self.robot, self.solver, self.frame)
        q = self.kinematics.get_joints_subset().get_joint_positions()
        gp, gr = self.solver.compute_forward_kinematics(self.frame, q)
        hp, hr = self.solver.compute_forward_kinematics('panda_hand', q)
        actual_hp, actual_hq = self.hand.get_world_pose()
        # 验证当前USD资产与官方运动学模型相符；不匹配时不能直接套用配置。
        if np.linalg.norm(hp - actual_hp) > 0.005 or angle_error(hr, quat_to_rot_matrix(actual_hq)) > 0.03:
            raise RuntimeError('USD手掌位姿与模型FK不一致，先检查模型与底座变换。')

        # ② 显式把场景TCP目标转换成模型right_gripper目标。
        # 模型末端与手掌有固定旋转和位置关系，不能只对世界Z随便加减。
        self.hand_to_gripper_r = hr.T @ gr
        self.tcp_in_gripper = gr.T @ (hp + hr @ self.tcp_local - gp)
        self.cube_start, _ = self.cube.get_world_pose()
        self.target_tcp = self.cube_start + (np.array([0., 0., hover_height]) if target_offset is None else np.asarray(target_offset))
        desired_hand_r = np.diag([1., -1., -1.]) if hand_rotation is None else np.asarray(hand_rotation)
        self.target_r = desired_hand_r @ self.hand_to_gripper_r
        self.target_q = rot_matrix_to_quat(self.target_r)  # w,x,y,z
        self.target_model = self.target_tcp - self.target_r @ self.tcp_in_gripper

        # ③ 先计算IK，再用FK核对解；不把IK终点解直接瞬移给机器人。
        action, success = self.kinematics.compute_inverse_kinematics(
            self.target_model, self.target_q, position_tolerance=0.001, orientation_tolerance=0.01
        )
        if not success:
            raise RuntimeError('IK无解/未收敛，本次不启动运动。')
        solved_p, solved_r = self.solver.compute_forward_kinematics(self.frame, action.joint_positions)
        self.ik_error = float(np.linalg.norm(solved_p + solved_r @ self.tcp_in_gripper - self.target_tcp))
        if self.ik_error > 0.002 or angle_error(solved_r, self.target_r) > 0.02:
            raise RuntimeError('IK返回成功，但FK误差过大，本次不启动运动。')

        self.policy = RmpFlow(**config)
        self.policy.set_robot_base_pose(self.base_p, self.base_q)
        # 桌面必须显式加入策略环境：USD有碰撞不代表RMPflow自动知道它。
        self.table = ExistingCuboid(self.stage, '/World/DeskTop')
        self.policy.add_cuboid(self.table, static=True)
        self.policy.update_world()
        self.policy.set_end_effector_target(self.target_model, self.target_q)
        self.motion = ArticulationMotionPolicy(self.robot, self.policy, 1.0 / 60.0)
        self.finger = list(self.robot.dof_names).index('panda_finger_joint1')
        self.controller.apply_action(ArticulationAction(
            joint_positions=np.array([0.03]), joint_indices=np.array([self.finger])))
        self.elapsed = 0.0
        self.stable_seconds = 0.0
        self.status = 'Running'
        self.detail = ''
        self.min_tcp_z = float('inf')
        self.max_cube_displacement = 0.0
        print(f'[第4课] TCP目标(m)={np.round(self.target_tcp, 4)}')
        print(f'[第4课] 模型末端→TCP偏移(m)={np.round(self.tcp_in_gripper, 6)}')
        print(f'[第4课] IK成功，FK验证误差={self.ik_error * 1000:.3f}mm；准备连续运动。')

    def actual_pose(self):
        # 通过实际物理手掌位姿算TCP，避免只用同一求解器自我验证。
        hp, hq = self.hand.get_world_pose()
        hr = quat_to_rot_matrix(hq)
        return hp + hr @ self.tcp_local, hr @ self.hand_to_gripper_r

    def tick(self, dt):
        """④ 每物理步：读实际状态 → 判断到达 → RMPflow给下一步动作。"""
        if self.status != 'Running':
            return
        try:
            if omni.usd.get_context().get_stage() != self.stage:
                raise RuntimeError('场景已切换')
            if not self.robot.handles_initialized:
                raise RuntimeError('物理句柄已失效，请重新运行')
            self.elapsed += float(dt)
            tcp, rotation = self.actual_pose()
            if not np.all(np.isfinite(tcp)):
                raise RuntimeError('末端位置包含非有限数值')
            self.min_tcp_z = min(self.min_tcp_z, float(tcp[2]))
            cube_p, _ = self.cube.get_world_pose()
            self.max_cube_displacement = max(self.max_cube_displacement, float(np.linalg.norm(cube_p-self.cube_start)))
            # 仅为本课保守中止条件，不是全机械臂碰撞检测。
            if tcp[2] < self.table.center[2] + self.table.size[2] / 2 + 0.06:
                raise RuntimeError('TCP过于接近桌面，停止悬停练习')
            if self.max_cube_displacement > 0.01:
                raise RuntimeError('方块发生明显位移，需检查接触情况')
            error = float(np.linalg.norm(tcp - self.target_tcp))
            orientation_error = angle_error(rotation, self.target_r)
            reached = error < POSITION_TOLERANCE_M and orientation_error < ORIENTATION_TOLERANCE_RAD
            self.stable_seconds = self.stable_seconds + dt if reached else 0.0
            if self.stable_seconds >= 0.5:
                self.status = 'Succeeded'
                return
            if self.elapsed >= TIMEOUT_SECONDS:
                self.status, self.detail = 'TimedOut', '末端未在时限内稳定到达目标'
                return
            # API生成小步关节位置/速度目标；apply_action让物理驱动器跟踪。
            action = self.motion.get_next_articulation_action(physics_dt=float(dt))
            self.controller.apply_action(action)
        except Exception as exc:
            self.status, self.detail = 'Failed', str(exc)

    def report(self):
        result = {'status': self.status, 'detail': self.detail, 'elapsed_seconds': self.elapsed,
                  'target_tcp_m': self.target_tcp.tolist(), 'ik_fk_error_m': self.ik_error,
                  'minimum_tcp_height_m': self.min_tcp_z if np.isfinite(self.min_tcp_z) else None,
                  'max_cube_displacement_m': self.max_cube_displacement}
        if omni.usd.get_context().get_stage() == self.stage and self.robot.handles_initialized:
            p, r = self.actual_pose()
            result.update(actual_tcp_m=p.tolist(), position_error_m=float(np.linalg.norm(p-self.target_tcp)),
                          orientation_error_degrees=float(np.rad2deg(angle_error(r,self.target_r))))
        print('[第4课] '+json.dumps(result, ensure_ascii=False))
        setattr(builtins, RESULT_KEY, result)
        return result


async def demo():
    timeline = omni.timeline.get_timeline_interface()
    if timeline.is_playing():
        raise RuntimeError('请先暂停现有任务。')
    lesson = HoverLesson(hover_height=HOVER_HEIGHT_M)
    # PhysX步回调才是真正的控制节拍；异步循环只监视状态并让UI继续更新。
    subscription = omni.physx.get_physx_interface().subscribe_physics_step_events(lesson.tick)
    try:
        timeline.play()
        while lesson.status == 'Running':
            await omni.kit.app.get_app().next_update_async()
            if omni.usd.get_context().get_stage() != lesson.stage or not timeline.is_playing():
                lesson.status, lesson.detail = 'Cancelled', '用户暂停/停止或切换场景'
                break
    finally:
        subscription = None  # 释放回调，避免重复运行后多个控制器同时写关节
        if omni.usd.get_context().get_stage() == lesson.stage and timeline.is_playing():
            timeline.pause()
        lesson.report()


def start():
    bottle_task = getattr(builtins, '_isaac_lesson09_task', None)
    if bottle_task is not None and not bottle_task.done():
        print('[第4课] 瓶子任务正在运行，请先暂停并等待结束。')
        return
    old = getattr(builtins, TASK_KEY, None)
    previous_gripper = getattr(builtins, '_isaac_lesson03_gripper_task', None)
    batch_task = getattr(builtins, '_isaac_lesson07_task', None)
    if any(task is not None and not task.done() for task in (old, previous_gripper, batch_task)):
        print('[第4课] 前一课任务仍在运行，请先暂停并等它结束。')
        return
    task = asyncio.ensure_future(demo())
    setattr(builtins, TASK_KEY, task)
    def finished(t):
        if not t.cancelled() and t.exception() is not None:
            result = {'status':'Failed','detail':str(t.exception())}
            setattr(builtins, RESULT_KEY, result)
            print('[第4课] '+str(result))
    task.add_done_callback(finished)
    return task


# TODO（你的实验）：HOVER_HEIGHT_M从0.15改为0.20，观察目标和实际TCP的Z。
# TODO（下一课）：加入下降、夹紧与抬起，按物体实际离桌高度评价抓取。
if __name__ == '__main__':
    start()
