"""第9课：已知位置的瓶子抓取放置基线，Isaac Sim 6.1 / PhysX。

教学重点：①瓶子几何适配 ②状态机 ③实际物体结果。
复用第四课运动学初始化，以Lula IK生成连续参考；物体仅由物理接触搬运。
start() 默认读取场景瓶子位姿；start(bottle_position_world_m=[x, y, z])
使用外部提供的瓶子 Body 世界位置作为抓取参考，不会移动瓶子。
当前仅用参考位置的 X/Y；TCP 高度仍由桌面和 GRASP_HEIGHT_M 决定。
保留官方603g质量。摩擦是明确的仿真假设，尚非实测标定。
TODO：先只改PLACE_XY_M的Y为0.15，对照落点与报告；再研究抓取高度。
"""
import asyncio
import builtins
import json
from pathlib import Path
import runpy
import time

import numpy as np
import omni.kit.app
import omni.physx
import omni.timeline
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.types import ArticulationAction

BODY_PATH = '/World/PracticeBottle/Body'
MESH_PATH = BODY_PATH + '/_06_mustard_bottle'
GRASP_HEIGHT_M = 0.11  # TCP相对桌面高度；实测可减小搬运期间的瓶身倾斜与滑移。
GRASP_X_OFFSET_M = 0.0  # 加长夹指后，TCP对准瓶身几何中心。
APPROACH_YAW_DEGREES = 0.0
PLACE_XY_M = (0.18, 0.20)
LIFT_HEIGHT_M = 0.12
MOVE_SPEED_M_S = 0.035
MAX_ARM_TARGET_SPEED_RAD_S = 0.8
OPEN_POSITION_M = 0.04
STATIC_FRICTION = 1.5  # 高摩擦指垫的实验假设，尚未实物标定。
DYNAMIC_FRICTION = 1.5
GRIP_DRIVE_MAX_FORCE_N = 60.0  # 驱动器作用力上限，不是实测指尖力。
FINGER_LENGTH_SCALE = 1.8  # 自定义加长夹指：外观与碰撞几何同步缩放。
TCP_EXTENSION_M = 0.036
GRIP_DRIVE_STIFFNESS = 5000.0
GRIP_DRIVE_DAMPING = 100.0
PHYSICS_HZ = 240  # 高刚度夹持需要更小物理步长，降低接触与mimic耦合振荡。
WALL_TIMEOUT_SECONDS = 600  # 编辑器低实时率时留出运行时间；仿真100秒/阶段22秒限制仍保留。
TASK_KEY = '_isaac_lesson09_task'
RESULT_KEY = '_isaac_lesson09_result'
OUTPUT_PATH = Path(__file__).resolve().parent.parent / 'outputs' / 'lesson09_latest.json'
lesson04 = runpy.run_path(str(Path(__file__).with_name('04_hover_above_cube.py')))


def validate_bottle_position(position):
    """外部输入为场景世界坐标系下瓶子 Body 原点，单位米。"""
    if position is None:
        return None
    try:
        result = np.asarray(position, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('瓶子位置必须是三个有限数值，单位米') from exc
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError('瓶子位置必须是三个有限数值，单位米')
    return result.copy()


def prepare_physics():
    """Stop后、初始化前设置会话参数；不写入官方资产/场景文件。"""
    stage = omni.usd.get_context().get_stage()
    if not stage or not stage.GetPrimAtPath(BODY_PATH):
        raise RuntimeError('请打开Isaac_Desk_Arm_Bottle.usda')
    if not omni.timeline.get_timeline_interface().is_stopped():
        raise RuntimeError('配置物理参数前请Stop')
    mesh = stage.GetPrimAtPath(MESH_PATH)
    if UsdGeom.Xformable(mesh).GetLocalTransformation().ExtractTranslation().GetLength() > 0.001:
        raise RuntimeError('瓶身内部Mesh出现偏移，请先运行08_repair_bottle_transform.py修复')
    compat = runpy.run_path(str(Path(__file__).with_name('franka_visual_compat.py')))
    compat['prepare_franka_visuals'](stage)
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                PhysxSchema.PhysxSceneAPI.Apply(prim).CreateTimeStepsPerSecondAttr(PHYSICS_HZ)
        bottle_api = PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath(BODY_PATH))
        bottle_api.CreateSolverPositionIterationCountAttr(32)
        bottle_api.CreateSolverVelocityIterationCountAttr(8)
        # 与第五课一致：当前机器人资产惯量偏小，按碰撞几何估算。
        for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/Franka')):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                mass = UsdPhysics.MassAPI.Apply(prim)
                mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0))
                mass.CreatePrincipalAxesAttr(Gf.Quatf(1))
        PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath('/World/Franka')).CreateSolverVelocityIterationCountAttr(8)
        # 原7.2N/400驱动在603g瓶子测试中滑脱；这是仿真实验配置，非实机标定。
        drive = UsdPhysics.DriveAPI(stage.GetPrimAtPath('/World/Franka/panda_hand/panda_finger_joint1'), 'linear')
        drive.GetMaxForceAttr().Set(GRIP_DRIVE_MAX_FORCE_N)
        drive.GetStiffnessAttr().Set(GRIP_DRIVE_STIFFNESS)
        drive.GetDampingAttr().Set(GRIP_DRIVE_DAMPING)
        other = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath('/World/Franka/panda_hand/panda_finger_joint2'), 'linear')
        other.CreateMaxForceAttr(0.)
        other.CreateStiffnessAttr(0.)
        other.CreateDampingAttr(0.)
        # 每次从根层资产读取原尺度，避免重复运行时累计拉长。
        original = Usd.Stage.Open(stage.GetRootLayer())
        for side in ('left', 'right'):
            path = f'/World/Franka/panda_{side}finger/geometry/panda_{side}finger'
            original_scale = original.GetPrimAtPath(path).GetAttribute('xformOp:scale').Get()
            scale_attr = stage.GetPrimAtPath(path).GetAttribute('xformOp:scale')
            value = (original_scale[0], original_scale[1], original_scale[2]*FINGER_LENGTH_SCALE)
            scale_attr.Set(Gf.Vec3f(*value) if scale_attr.GetTypeName() == 'float3' else Gf.Vec3d(*value))
        tool = stage.GetPrimAtPath('/World/Franka/panda_hand/tool_center')
        tool.GetAttribute('xformOp:translate').Set(Gf.Vec3d(0., 0., 0.1034+TCP_EXTENSION_M))
        material = UsdShade.Material.Define(stage, '/World/Lesson09GripMaterial')
        api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        api.CreateStaticFrictionAttr(STATIC_FRICTION)
        api.CreateDynamicFrictionAttr(DYNAMIC_FRICTION)
        api.CreateRestitutionAttr(0.0)
        for path in (MESH_PATH, '/World/Franka/panda_leftfinger/geometry/panda_leftfinger',
                     '/World/Franka/panda_rightfinger/geometry/panda_rightfinger'):
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.HasAPI(UsdPhysics.CollisionAPI):
                raise RuntimeError(f'缺少碰撞体：{path}')
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose='physics')
            collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            collision.CreateContactOffsetAttr(0.002)
            collision.CreateRestOffsetAttr(0.0)
            # 指垫有有限接触面积，允许接触面产生抗扭摩擦，非点接触。
            # 10mm为加长指垫接触斑半径的仿真假设，非实测标定。
            collision.CreateTorsionalPatchRadiusAttr(0.01)
            collision.CreateMinTorsionalPatchRadiusAttr(0.01)
            decomposition = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexDecomposition')
            decomposition.CreateShrinkWrapAttr(True)
            decomposition.CreateErrorPercentageAttr(2. if path == MESH_PATH else 1.)
            decomposition.CreateMaxConvexHullsAttr(64 if path == MESH_PATH else 16)


class BottleLesson(lesson04['HoverLesson']):
    def __init__(self, bottle_position_world_m=None, place_xy_world_m=None,
                 progress_callback=None):
        # 中部侧向接近，避免手掌从顶部下降时被瓶盖挡住。
        external_position = validate_bottle_position(bottle_position_world_m)
        requested_place_xy = validate_place_position(place_xy_world_m)
        stage = omni.usd.get_context().get_stage()
        table = lesson04['ExistingCuboid'](stage, '/World/DeskTop')
        table_z = float(table.center[2]+table.size[2]/2)
        body_position = np.array(UsdGeom.Xformable(stage.GetPrimAtPath(BODY_PATH)).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation(), dtype=float)
        reference_position = body_position if external_position is None else external_position
        # 手指沿世界+X伸入，开合轴沿世界Y；加长夹指给手掌留空间。
        yaw = np.deg2rad(APPROACH_YAW_DEGREES)
        c, s = np.cos(yaw), np.sin(yaw)
        self.approach_direction = np.array([c, s, 0.])
        hand_r = np.array([[0., -s, c], [0., c, s], [-1., 0., 0.]])
        super().__init__(object_path=BODY_PATH,
                         target_offset=[reference_position[0]-body_position[0]+GRASP_X_OFFSET_M-0.10*c,
                                        reference_position[1]-body_position[1]-0.10*s,
                                        table_z+GRASP_HEIGHT_M+0.18-body_position[2]],
                         hand_rotation=hand_r)
        self.bottle = self.cube  # 父类历史命名仅作兼容，物理句柄实际指向瓶子Body。
        self.start_p, self.start_q = self.bottle.get_world_pose()
        self.start_p, self.start_q = self.start_p.copy(), self.start_q.copy()
        self.reference_position = self.start_p.copy() if external_position is None else external_position.copy()
        self.target_source = 'scene_pose' if external_position is None else 'external_world_position'
        self.table_z = float(self.table.center[2]+self.table.size[2]/2)
        body = self.stage.GetPrimAtPath(BODY_PATH)
        mesh = self.stage.GetPrimAtPath(MESH_PATH)
        body_world = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        mesh_local = UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) * body_world.GetInverse()
        self.points = np.array([mesh_local.Transform(Gf.Vec3d(v)) for v in UsdGeom.Mesh(mesh).GetPointsAttr().Get()])
        if not len(self.points) or not np.all(np.isfinite(self.points)):
            raise RuntimeError('瓶身几何无效')
        self.local_lo, self.local_hi = self.points.min(axis=0), self.points.max(axis=0)
        # 原资产-Y朝瓶盖，+Y为瓶底；用实际瓶底区域建立放置参考。
        self.base_center_local = self.points[self.points[:, 1] > self.local_hi[1]-0.002].mean(axis=0)
        r = quat_to_rot_matrix(self.start_q)
        # 仅适配当前直立、薄侧沿世界Y的瓶子。换朝向时须重新计算夹爪朝向。
        if np.dot(r @ np.array([0., -1., 0.]), [0., 0., 1.]) < np.cos(np.deg2rad(5)):
            raise RuntimeError('瓶子初始不直立，请Stop恢复或修复场景')
        narrow = r @ np.array([0., 0., 1.])
        if abs(narrow[1]) < 0.99:
            raise RuntimeError('本课要求瓶身薄侧沿世界Y，不能直接夹9.6cm宽侧')
        self.body_size = self.local_hi-self.local_lo
        if self.body_size[2] >= 2*OPEN_POSITION_M-0.006:
            raise RuntimeError('薄侧宽度超出当前演示夹爪开口')
        self.mass_kg = float(UsdPhysics.MassAPI(body).GetMassAttr().Get())
        if abs(self.mass_kg-0.603) > 0.005:
            raise RuntimeError('本课要求保留官方约603g质量，配置已变化')
        self.initial_bottom = float((self.points @ r.T+self.start_p)[:, 2].min())
        if abs(self.initial_bottom-self.table_z) > 0.008:
            raise RuntimeError('瓶子初始不在桌面附近')
        self.grasp_tcp = self.reference_position.copy()
        self.grasp_tcp[0] += GRASP_X_OFFSET_M
        self.grasp_tcp[2] = self.table_z+GRASP_HEIGHT_M
        self.approach_start = self.grasp_tcp-0.10*self.approach_direction
        if not 0.085 <= GRASP_HEIGHT_M <= 0.115:
            raise ValueError('本课瓶身中部抓取高度限定0.085～0.115m')
        self.lift_tcp = self.grasp_tcp+[0., 0., LIFT_HEIGHT_M]
        self.place_body = np.r_[PLACE_XY_M if requested_place_xy is None else requested_place_xy,
                                self.start_p[2] + self.table_z-self.initial_bottom]
        self.place_target_source = 'default_configuration' if requested_place_xy is None else 'ros2_world_xy'
        self.requested_place_xy_world_m = (None if requested_place_xy is None
                                            else requested_place_xy.tolist())
        self.progress_callback = progress_callback
        available = self.table.size[:2]/2 - 0.07
        if np.any(np.abs(self.place_body[:2]-self.table.center[:2]) > available):
            raise ValueError('B点太靠近桌边')
        self.command_tcp = self.target_tcp.copy()
        self.phase, self.phase_time = 'PreGrasp', 0.0
        self.phase_trace = [{'phase': self.phase, 'time': 0.0}]
        self.metrics = {}
        self.phase_diagnostics = []
        self._last_diagnostic_time = {}
        self.max_lift_m = self.max_slip_m = self.stable_seconds = 0.0
        self.relative_reference = None
        self.pick_verified = self.release_commanded = False
        self.stable_reference = None
        # 水平姿态的RMPflow会落入腕部限位附近的局部解。
        # 本课使用Lula IK连续参考+限速关节驱动；参考从上一命令续算，
        # 不把夹持接触引起的实际速度抖动反馈到参考轨迹中。
        subset = self.kinematics.get_joints_subset()
        self.arm_indices = np.array(subset.joint_indices, dtype=int)
        self.arm_command = self.robot.get_joint_positions()[self.arm_indices].astype(float)
        ready, ok = self.solver.compute_inverse_kinematics(
            self.frame, self.target_model, self.target_q, warm_start=self.arm_command,
            position_tolerance=0.001, orientation_tolerance=0.01)
        if not ok:
            raise RuntimeError('侧向预抓取姿态IK失败')
        self.arm_goal = np.asarray(ready)
        for target in (self.grasp_tcp, self.lift_tcp,
                       self.grasp_tcp + (self.place_body-self.start_p),
                       self.lift_tcp + (self.place_body-self.start_p)):
            self.check_target(target)
        self.grip(OPEN_POSITION_M)
        print(f'[第9课] 质量{self.mass_kg:.3f}kg，TCP抓取高度{GRASP_HEIGHT_M:.3f}m；薄侧夹持')
        self.emit_progress(self.phase, '抓放状态机已初始化')

    def emit_progress(self, phase, detail=''):
        if self.progress_callback is not None:
            self.progress_callback(phase, detail)

    def check_target(self, tcp):
        _, ok = self.kinematics.compute_inverse_kinematics(
            tcp-self.target_r @ self.tcp_in_gripper, self.target_q,
            position_tolerance=0.001, orientation_tolerance=0.01)
        if not ok:
            raise RuntimeError(f'目标IK未收敛：{tcp}')

    def grip(self, position):
        self.controller.apply_action(ArticulationAction(
            joint_positions=np.array([position]), joint_indices=np.array([self.finger])))

    def set_tcp(self, goal, dt):
        delta = goal-self.command_tcp
        distance = float(np.linalg.norm(delta))
        if distance:
            self.command_tcp += delta*min(1., MOVE_SPEED_M_S*dt/distance)
        attempts = []
        for position_tolerance, orientation_tolerance in ((0.0005, 0.005), (0.001, 0.01), (0.002, 0.02)):
            q, ok = self.solver.compute_inverse_kinematics(
                self.frame, self.command_tcp-self.target_r @ self.tcp_in_gripper, self.target_q,
                warm_start=self.arm_command.copy(), position_tolerance=position_tolerance,
                orientation_tolerance=orientation_tolerance)
            delta_q = float(np.max(np.abs(q-self.arm_command))) if ok else None
            attempts.append({'position_tolerance_m': position_tolerance,
                             'orientation_tolerance_rad': orientation_tolerance,
                             'converged': bool(ok), 'max_joint_delta_rad': delta_q,
                             'joint_delta_rad': (np.asarray(q-self.arm_command).tolist() if ok else None)})
            # Lula 在这段冗余姿态附近会出现约0.19rad的有效解变化；
            # 关节参考仍由 apply_arm 限速，避免把IK目标变化直接施加到物理关节。
            if ok and delta_q < 0.25:
                self.arm_goal = np.asarray(q)
                return
        raise RuntimeError('侧向路径IK未收敛或分支跳变，中止运动；尝试结果='+json.dumps(attempts, ensure_ascii=False))

    def apply_arm(self, dt):
        # 下发位置/速度驱动目标；不设置实际关节角度，不移动瓶子。
        previous = self.arm_command.copy()
        delta = self.arm_goal-previous
        scale = min(1., MAX_ARM_TARGET_SPEED_RAD_S*dt/max(float(np.max(np.abs(delta))), 1e-9))
        self.arm_command += delta*scale
        actual = self.robot.get_joint_positions()[self.arm_indices]
        if np.max(np.abs(self.arm_command-actual)) > 0.25:
            raise RuntimeError('关节未跟上参考轨迹，可能接触受阻')
        self.controller.apply_action(ArticulationAction(
            joint_positions=self.arm_command, joint_velocities=(self.arm_command-previous)/dt,
            joint_indices=self.arm_indices))

    def reached(self, tcp, rotation, goal, dt):
        good = (np.linalg.norm(self.command_tcp-goal) < 0.0001
                and np.linalg.norm(tcp-goal) < 0.006
                and lesson04['angle_error'](rotation, self.target_r) < np.deg2rad(3))
        self.stable_seconds = self.stable_seconds+dt if good else 0.0
        return self.stable_seconds >= 0.3

    def transition(self, phase):
        self.phase, self.phase_time, self.stable_seconds = phase, 0., 0.
        self.phase_trace.append({'phase': phase, 'time': self.elapsed})
        print('[第9课] '+phase)
        self.emit_progress(phase, '进入抓放阶段')

    def measure(self, tcp, p, q):
        r = quat_to_rot_matrix(q)
        bottom = float((self.points @ r.T+p)[:, 2].min())
        tilt = float(np.rad2deg(np.arccos(np.clip(np.dot(r @ [0., -1., 0.], [0., 0., 1.]), -1., 1.))))
        return {'lift_m': float(p[2]-self.start_p[2]), 'bottom_clearance_m': bottom-self.table_z,
                'tilt_degrees': tilt, 'xy_error_m': float(np.linalg.norm(p[:2]-self.place_body[:2])),
                'height_error_m': float(abs(p[2]-self.place_body[2])),
                'tcp_xy_clearance_m': float(np.linalg.norm(tcp[:2]-p[:2])),
                'linear_speed_m_s': float(np.linalg.norm(self.bottle.get_linear_velocity())),
                'angular_speed_rad_s': float(np.linalg.norm(self.bottle.get_angular_velocity())),
                'finger_position_m': float(self.robot.get_joint_positions()[self.finger]),
                'tcp_above_bottle_top_m': float(tcp[2]-(self.points @ r.T+p)[:, 2].max())}

    def record_phase_diagnostic(self, phase, tcp, rotation, bottle_p, bottle_q):
        if phase not in ('Transfer', 'Lower', 'Release', 'RetreatClear'):
            return
        previous = self._last_diagnostic_time.get(phase, -1.0)
        if self.phase_time-previous < 0.5:
            return
        goal = {
            'Transfer': getattr(self, 'transfer_tcp', None),
            'Lower': getattr(self, 'lower_goal', None),
            'Release': getattr(self, 'place_tcp', None),
            'RetreatClear': getattr(self, 'retreat_clear_tcp', None),
        }[phase]
        if goal is None:
            return
        actual_joints = self.robot.get_joint_positions()[self.arm_indices]
        metrics = self.measure(tcp, bottle_p, bottle_q)
        slip = None
        if self.relative_reference is not None:
            slip = float(np.linalg.norm(bottle_p-tcp-self.relative_reference))
        self.phase_diagnostics.append({
            'phase': phase,
            'phase_time_s': float(self.phase_time),
            'tcp_world_m': tcp.tolist(),
            'tcp_command_world_m': self.command_tcp.tolist(),
            'tcp_goal_world_m': np.asarray(goal, dtype=float).tolist(),
            'tcp_goal_error_m': float(np.linalg.norm(tcp-goal)),
            'command_goal_error_m': float(np.linalg.norm(self.command_tcp-goal)),
            'orientation_error_degrees': float(np.rad2deg(
                lesson04['angle_error'](rotation, self.target_r))),
            'max_joint_tracking_error_rad': float(np.max(
                np.abs(self.arm_command-actual_joints))),
            'arm_command_rad': self.arm_command.tolist(),
            'bottle_world_m': bottle_p.tolist(),
            'bottle_tilt_degrees': metrics['tilt_degrees'],
            'bottle_linear_speed_m_s': metrics['linear_speed_m_s'],
            'bottle_angular_speed_rad_s': metrics['angular_speed_rad_s'],
            'finger_position_m': metrics['finger_position_m'],
            'grasp_slip_m': slip,
        })
        self._last_diagnostic_time[phase] = float(self.phase_time)

    def tick(self, dt):
        if self.status != 'Running':
            return
        try:
            if omni.usd.get_context().get_stage() != self.stage or not self.robot.handles_initialized:
                raise RuntimeError('场景或物理句柄失效')
            dt = float(dt)
            self.elapsed += dt
            self.phase_time += dt
            phase_at_tick = self.phase
            tcp, rotation = self.actual_pose()
            p, q = self.bottle.get_world_pose()
            self.metrics = m = self.measure(tcp, p, q)
            if not np.all(np.isfinite(np.r_[tcp, p, q, list(m.values())])):
                raise RuntimeError('物理状态出现非有限数值')
            self.max_lift_m = max(self.max_lift_m, m['lift_m'])
            if m['bottom_clearance_m'] < -0.006:
                raise RuntimeError('瓶子穿入或掉到桌面以下')
            if m['tilt_degrees'] > 20:
                raise RuntimeError('瓶子倾倒/倾斜超过20度')
            if self.elapsed > 100 or self.phase_time > 22:
                self.status, self.detail = 'TimedOut', self.phase+'阶段未按时完成'
                return
            if self.phase in ('Hold', 'Transfer', 'Lower'):
                slip = float(np.linalg.norm(p-tcp-self.relative_reference))
                self.max_slip_m = max(self.max_slip_m, slip)
                if slip > 0.015:
                    raise RuntimeError('物体相对TCP滑移超过1.5cm')
                if self.phase in ('Hold', 'Transfer') and m['lift_m'] < LIFT_HEIGHT_M*0.65:
                    raise RuntimeError('机械臂已抬起，但瓶子未保持抬升')

            if self.phase == 'PreGrasp':
                if self.reached(tcp, rotation, self.target_tcp, dt):
                    self.transition('Align')
            elif self.phase == 'Align':
                self.set_tcp(self.approach_start, dt)
                if self.reached(tcp, rotation, self.approach_start, dt):
                    self.transition('Approach')
            elif self.phase == 'Approach':
                self.set_tcp(self.grasp_tcp, dt)
                if self.reached(tcp, rotation, self.grasp_tcp, dt):
                    self.grip(0.)
                    self.transition('Close')
            elif self.phase == 'Close':
                if self.phase_time >= 2.:
                    if m['finger_position_m'] > OPEN_POSITION_M-0.004:
                        raise RuntimeError('夹爪未实际合拢，不能开始抬升')
                    if m['finger_position_m'] < 0.003:
                        raise RuntimeError('夹爪空合，未夹住预期瓶身')
                    self.transition('Lift')
            elif self.phase == 'Lift':
                self.set_tcp(self.lift_tcp, dt)
                if self.reached(tcp, rotation, self.lift_tcp, dt):
                    if m['lift_m'] < LIFT_HEIGHT_M*0.65:
                        raise RuntimeError('抬升失败：瓶子没有随夹爪升高')
                    self.relative_reference = p-tcp
                    self.transition('Hold')
            elif self.phase == 'Hold':
                if self.phase_time >= 1.:
                    self.pick_verified = True
                    self.transfer_tcp = self.place_body+[0., 0., LIFT_HEIGHT_M]-self.relative_reference
                    self.place_tcp = self.place_body+[0., 0., 0.003]-self.relative_reference
                    self.retreat_clear_tcp = self.place_tcp-0.12*self.approach_direction
                    for target in (self.transfer_tcp, self.place_tcp):
                        self.check_target(target)
                    self.transition('Transfer')
            elif self.phase == 'Transfer':
                self.set_tcp(self.transfer_tcp, dt)
                if self.reached(tcp, rotation, self.transfer_tcp, dt):
                    self.transition('Lower')
            elif self.phase == 'Lower':
                # 下降时按瓶身Body原点修正XY，与最终放置误差使用同一参考。
                # 若按瓶底顶点中心追踪，瓶身倾斜会让目标横移并把瓶子推偏。
                self.lower_goal = self.place_tcp.copy()
                self.lower_goal[:2] = tcp[:2]+self.place_body[:2]-p[:2]
                self.set_tcp(self.lower_goal, dt)
                # 支撑接触后不再要求把瓶子硬压到理想TCP高度；
                # 由实际瓶底/放置位置确认接近桌面，再松爪，最终仍需独立放稳验收。
                supported = (abs(self.command_tcp[2]-self.place_tcp[2]) < 0.0001
                             and abs(m['bottom_clearance_m']) < 0.003
                             and m['xy_error_m'] < 0.005
                             and m['height_error_m'] < 0.008
                             and m['tilt_degrees'] < 15)
                if supported:
                    self.retreat_clear_tcp = tcp-0.12*self.approach_direction
                    self.release_commanded = True
                    self.release_start = float(self.robot.get_joint_positions()[self.finger])
                    self.transition('Release')
            elif self.phase == 'Release':
                # 缓慢张开，避免高刚度目标突变将瓶身弹向一侧。
                self.grip(min(OPEN_POSITION_M, self.release_start+0.008*self.phase_time))
                if self.phase_time >= 2.5 and m['finger_position_m'] > 0.038:
                    self.transition('RetreatClear')
            elif self.phase == 'RetreatClear':
                self.set_tcp(self.retreat_clear_tcp, dt)
                if self.reached(tcp, rotation, self.retreat_clear_tcp, dt):
                    self.transition('Settle')
            elif self.phase == 'Settle':
                tcp_clear = (m['tcp_xy_clearance_m'] > 0.07
                             or m['tcp_above_bottle_top_m'] > 0.04)
                valid = (self.pick_verified and self.release_commanded and m['xy_error_m'] < 0.015
                         and m['height_error_m'] < 0.008 and abs(m['bottom_clearance_m']) < 0.005
                         and m['tilt_degrees'] < 8 and m['linear_speed_m_s'] < 0.01
                         and m['angular_speed_rad_s'] < 0.1 and m['finger_position_m'] > 0.032
                         and tcp_clear)
                if valid:
                    if self.stable_reference is None or np.linalg.norm(p-self.stable_reference) > 0.002:
                        self.stable_reference, self.stable_seconds = p.copy(), 0.
                    self.stable_seconds += dt
                    if self.stable_seconds >= 1.:
                        self.status, self.detail = 'Succeeded', '瓶子已抓起、搬运、松开并直立放稳'
                else:
                    self.stable_reference, self.stable_seconds = None, 0.
                if self.phase_time >= 5 and self.status == 'Running':
                    raise RuntimeError('放置未达标：检查位置、倾角、速度和撤离距离')
            self.record_phase_diagnostic(phase_at_tick, tcp, rotation, p, q)
            self.apply_arm(dt)
        except Exception as exc:
            self.status, self.detail = 'Failed', str(exc)

    def report(self):
        result = {'status': self.status, 'phase': self.phase, 'detail': self.detail,
                  'elapsed_seconds': self.elapsed, 'start_position_m': self.start_p.tolist(),
                  'place_target_m': self.place_body.tolist(), 'pick_verified': self.pick_verified,
                  'release_commanded': self.release_commanded, 'max_lift_m': self.max_lift_m,
                  'max_hold_transport_slip_m': self.max_slip_m, 'metrics': self.metrics,
                  'phases': self.phase_trace, 'phase_diagnostics': self.phase_diagnostics,
                  'configuration': {
                      'object': BODY_PATH, 'mass_kg': self.mass_kg, 'mass_source': 'official_asset_not_measured',
                      'grasp_height_m': GRASP_HEIGHT_M, 'lift_height_m': LIFT_HEIGHT_M,
                      'approach': 'horizontal_midbody_from_world_minus_x', 'grasp_x_offset_m': GRASP_X_OFFSET_M,
                      'approach_yaw_degrees': APPROACH_YAW_DEGREES,
                      'gripper_geometry': 'custom_lengthened_fingers', 'finger_length_scale': FINGER_LENGTH_SCALE,
                      'tcp_extension_m': TCP_EXTENSION_M, 'torsional_contact_patch_radius_m': 0.01,
                      'collision_proxy': 'shrinkwrapped_convex_decomposition_bottle_and_fingers',
                      'arm_control': 'lula_ik_continuous_reference_and_rate_limited_joint_targets',
                      'max_arm_target_speed_rad_s': MAX_ARM_TARGET_SPEED_RAD_S, 'physics_hz': PHYSICS_HZ,
                      'wall_timeout_seconds': WALL_TIMEOUT_SECONDS,
                      'tcp_position_tolerance_m': 0.006, 'tcp_orientation_tolerance_degrees': 3,
                      'move_speed_m_s': MOVE_SPEED_M_S, 'static_friction': STATIC_FRICTION,
                      'dynamic_friction': DYNAMIC_FRICTION, 'friction_source': 'simulation_assumption',
                      'gripper_drive_max_force_n': GRIP_DRIVE_MAX_FORCE_N,
                      'gripper_drive_stiffness': GRIP_DRIVE_STIFFNESS,
                      'gripper_drive_damping': GRIP_DRIVE_DAMPING,
                      'gripper_force_source': 'drive_limit_not_measured_contact_force',
                      'robot_inertia': 'estimated_from_collision_geometry', 'bottle_inertia': 'asset_configuration',
                      'sensor_input': self.target_source,
                      'bottle_reference_position_world_m': self.reference_position.tolist(),
                      'place_target_source': self.place_target_source,
                      'requested_place_xy_world_m': self.requested_place_xy_world_m,
                      'planned_place_xy_world_m': self.place_body[:2].tolist(),
                      'place_height_source': 'table_and_bottle_geometry',
                      'actual_bottle_pose_used_for_evaluation': True}}
        if omni.usd.get_context().get_stage() == self.stage and self.robot.handles_initialized:
            try:
                result['final_position_m'] = self.bottle.get_world_pose()[0].tolist()
            except Exception:
                pass
        save_result(result)
        return result


def save_result(result):
    setattr(builtins, RESULT_KEY, result)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[第9课] '+json.dumps(result, ensure_ascii=False))


def validate_place_position(position):
    """外部放置目标为 world 坐标系下的 XY，单位米；Z由场景几何决定。"""
    if position is None:
        return None
    try:
        result = np.asarray(position, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('放置目标必须是两个有限数值，单位米') from exc
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise ValueError('放置目标必须是两个有限数值，单位米')
    return result.copy()


async def demo(bottle_position_world_m=None, place_xy_world_m=None,
               progress_callback=None, cancel_requested=None):
    timeline = omni.timeline.get_timeline_interface()
    if timeline.is_playing():
        raise RuntimeError('请先暂停现有任务')
    timeline.stop()
    await omni.kit.app.get_app().next_update_async()
    prepare_physics()
    await omni.kit.app.get_app().next_update_async()
    lesson = BottleLesson(bottle_position_world_m=bottle_position_world_m,
                          place_xy_world_m=place_xy_world_m,
                          progress_callback=progress_callback)
    subscription = omni.physx.get_physx_interface().subscribe_physics_step_events(lesson.tick)
    started = time.monotonic()
    try:
        timeline.play()
        while lesson.status == 'Running':
            await omni.kit.app.get_app().next_update_async()
            if lesson.status == 'Running' and cancel_requested and cancel_requested():
                lesson.status, lesson.detail = 'Cancelled', '收到ROS2取消请求'
                if timeline.is_playing():
                    timeline.pause()
                break
            if lesson.status == 'Running' and (omni.usd.get_context().get_stage() != lesson.stage or not timeline.is_playing()):
                lesson.status, lesson.detail = 'Cancelled', '用户暂停/停止或切换场景'
            if lesson.status == 'Running' and time.monotonic()-started > WALL_TIMEOUT_SECONDS:
                lesson.status, lesson.detail = 'TimedOut', f'墙钟时间超过{WALL_TIMEOUT_SECONDS}秒'
    finally:
        subscription = None
        if omni.usd.get_context().get_stage() == lesson.stage and timeline.is_playing():
            timeline.pause()
        lesson.report()


def start(bottle_position_world_m=None, place_xy_world_m=None,
          progress_callback=None, cancel_requested=None):
    bottle_position_world_m = validate_bottle_position(bottle_position_world_m)
    place_xy_world_m = validate_place_position(place_xy_world_m)
    for key in (TASK_KEY, '_isaac_lesson03_gripper_task', '_isaac_lesson04_task',
                '_isaac_lesson05_task', '_isaac_lesson06_task', '_isaac_lesson07_task'):
        previous = getattr(builtins, key, None)
        if previous is not None and not previous.done():
            print('[第9课] 前一任务尚未结束，请先暂停并等待')
            return
    save_result({'status': 'Running', 'phase': 'Initialize'})
    task = asyncio.ensure_future(demo(bottle_position_world_m=bottle_position_world_m,
                                      place_xy_world_m=place_xy_world_m,
                                      progress_callback=progress_callback,
                                      cancel_requested=cancel_requested))
    setattr(builtins, TASK_KEY, task)
    def finished(t):
        if t.cancelled():
            save_result({'status': 'Cancelled', 'phase': 'Task', 'detail': '任务被取消'})
        elif t.exception() is not None:
            save_result({'status': 'Failed', 'phase': 'Initialize', 'detail': str(t.exception())})
    task.add_done_callback(finished)
    return task


if __name__ == '__main__':
    start()
