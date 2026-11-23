"""Single fixed-coordinate grasp/lift test for the imported water bottle.

The physics values below are temporary test assumptions, not measured bottle data.
All USD overrides are authored in the session layer and restored after the run.
"""

from pathlib import Path as _ProjectPath
import sys as _sys
PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in _sys.path:
    _sys.path.insert(0, str(PROJECT_ROOT / "src"))

import asyncio
import builtins
import json
from pathlib import Path
import time

import numpy as np
import omni.kit.app
import omni.physx
import omni.timeline
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.types import ArticulationAction


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "fixed_pose_grasp"
BODY_PATH = "/World/PracticeBottle/bottle014/body"
LID_PATH = "/World/PracticeBottle/bottle014/lid"
LID_JOINT_PATH = LID_PATH + "/D6Joint"
TABLE_PATH = "/World/DeskTop"
TASK_KEY = "_isaac_waterbottle_fixed_grasp_task"
RESULT_KEY = "_isaac_waterbottle_fixed_grasp_result"

TEST_BODY_MASS_KG = 0.54  # 560g满瓶基线减去约20g瓶盖；待实物称重后替换。
TEST_LID_MASS_KG = 0.02
TEST_BOTTLE_RADIUS_M = 0.038
TEST_BOTTLE_HEIGHT_M = 0.24
TEST_LID_RADIUS_M = 0.020
TEST_LID_HEIGHT_M = 0.020
GRASP_HEIGHT_ABOVE_TABLE_M = 0.28
GRASP_TCP_X_OFFSET_M = 0.0
GRASP_TCP_Y_OFFSET_M = 0.0
APPROACH_DISTANCE_M = 0.10
LIFT_HEIGHT_M = 0.08
OPEN_POSITION_M = 0.045
GRIP_CLOSE_SPEED_M_S = 0.02
GRIP_CLOSE_TIMEOUT_S = 4.0
GRIP_CONTACT_STABLE_S = 0.25
GRIP_HOLD_PRELOAD_M = 0.006
EXPECTED_NECK_DIAMETER_M = 0.039
GRIP_CONTACT_POSITION_TOLERANCE_M = 0.004
GRIPPER_FINGER_LENGTH_SCALE = 1.8
TCP_EXTENSION_M = 0.036
GRIP_MAX_FORCE_N = 300.0
GRIP_STIFFNESS = 20000.0
GRIP_DAMPING = 100.0
GRIP_SURFACE_FRICTION = 6.0
FINGER_LENGTH_SCALE = 1.8
MAX_ARM_TARGET_SPEED_RAD_S = 0.8
MOVE_SPEED_M_S = 0.035
PHYSICS_HZ = 240
WALL_TIMEOUT_S = 180
HOLD_SECONDS = 1.0
MAX_SLIP_M = 0.015

motion_module = None


def _inertia_cylinder(mass, radius, height):
    transverse = mass * (3.0 * radius**2 + height**2) / 12.0
    axial = mass * radius**2 / 2.0
    return Gf.Vec3f(transverse, transverse, axial)


def _temporary_physics_setup(stage):
    """Apply temporary test overrides and restore the exact session layer afterward."""
    session_layer = stage.GetSessionLayer()
    session_layer_snapshot = session_layer.ExportToString()
    material_path = "/World/FixedPoseGraspPhysicsMaterial"

    def set_attr(attr, value):
        if not attr or not attr.IsValid():
            raise RuntimeError("缺少临时物理配置属性")
        attr.Set(value)

    body = stage.GetPrimAtPath(BODY_PATH)
    lid = stage.GetPrimAtPath(LID_PATH)
    joint_prim = stage.GetPrimAtPath(LID_JOINT_PATH)
    if not body or not lid or not joint_prim:
        raise RuntimeError("场景缺少水瓶刚体或瓶盖关节")

    body_collision_root = stage.GetPrimAtPath(BODY_PATH + "/Collisions")
    collision_prims = [prim for prim in Usd.PrimRange(body_collision_root)
                       if prim.IsA(UsdGeom.Mesh)]

    body_rb = UsdPhysics.RigidBodyAPI(body)
    lid_rb = UsdPhysics.RigidBodyAPI(lid)
    body_mass = UsdPhysics.MassAPI(body)
    lid_mass = UsdPhysics.MassAPI(lid)
    finger1 = stage.GetPrimAtPath("/World/Franka/panda_hand/panda_finger_joint1")
    finger2 = stage.GetPrimAtPath("/World/Franka/panda_hand/panda_finger_joint2")
    tool = stage.GetPrimAtPath("/World/Franka/panda_hand/tool_center")
    finger_mesh_paths = (
        "/World/Franka/panda_leftfinger/geometry/panda_leftfinger",
        "/World/Franka/panda_rightfinger/geometry/panda_rightfinger",
    )
    if not finger1 or not finger2 or not tool:
        raise RuntimeError("场景缺少 Franka 夹爪关节或 tool_center")
    try:
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            set_attr(body_rb.GetKinematicEnabledAttr(), False)
            set_attr(body_mass.GetMassAttr(), TEST_BODY_MASS_KG)
            set_attr(body_mass.GetDiagonalInertiaAttr(),
                     _inertia_cylinder(TEST_BODY_MASS_KG, TEST_BOTTLE_RADIUS_M, TEST_BOTTLE_HEIGHT_M))
            set_attr(body_mass.GetPrincipalAxesAttr(), Gf.Quatf(1))
            set_attr(lid_rb.GetKinematicEnabledAttr(), False)
            set_attr(lid_mass.GetMassAttr(), TEST_LID_MASS_KG)
            set_attr(lid_mass.GetDiagonalInertiaAttr(),
                     _inertia_cylinder(TEST_LID_MASS_KG, TEST_LID_RADIUS_M, TEST_LID_HEIGHT_M))
            set_attr(lid_mass.GetPrincipalAxesAttr(), Gf.Quatf(1))

            drive = UsdPhysics.DriveAPI(finger1, "linear")
            set_attr(drive.GetMaxForceAttr(), GRIP_MAX_FORCE_N)
            set_attr(drive.GetStiffnessAttr(), GRIP_STIFFNESS)
            set_attr(drive.GetDampingAttr(), GRIP_DAMPING)
            for finger_joint in (finger1, finger2):
                set_attr(UsdPhysics.PrismaticJoint(finger_joint).GetUpperLimitAttr(), OPEN_POSITION_M)

            for path in finger_mesh_paths:
                mesh = stage.GetPrimAtPath(path)
                scale_attr = mesh.GetAttribute("xformOp:scale") if mesh else None
                if not scale_attr or not scale_attr.IsValid():
                    raise RuntimeError(f"夹爪指尖缺少缩放属性: {path}")
                old_scale = scale_attr.Get()
                scaled = (old_scale[0], old_scale[1], old_scale[2] * GRIPPER_FINGER_LENGTH_SCALE)
                set_attr(scale_attr, Gf.Vec3f(*scaled) if scale_attr.GetTypeName() == "float3"
                         else Gf.Vec3d(*scaled))
            tool_translate = tool.GetAttribute("xformOp:translate")
            old_tool = tool_translate.Get()
            extended_tool = Gf.Vec3d(old_tool[0], old_tool[1], old_tool[2] + TCP_EXTENSION_M)
            set_attr(tool_translate, extended_tool)

            material = UsdShade.Material.Define(stage, material_path)
            physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
            physics_material.CreateStaticFrictionAttr(GRIP_SURFACE_FRICTION)
            physics_material.CreateDynamicFrictionAttr(GRIP_SURFACE_FRICTION)
            physics_material.CreateRestitutionAttr(0.0)
            binding_prims = collision_prims + [
                stage.GetPrimAtPath("/World/Franka/panda_leftfinger/geometry/panda_leftfinger"),
                stage.GetPrimAtPath("/World/Franka/panda_rightfinger/geometry/panda_rightfinger"),
                stage.GetPrimAtPath(TABLE_PATH),
            ]
            for prim in binding_prims:
                if not prim:
                    raise RuntimeError("缺少物理材质绑定目标")
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    material, materialPurpose="physics")
                collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                set_attr(collision.CreateContactOffsetAttr(0.002), 0.002)
                set_attr(collision.CreateRestOffsetAttr(0.0), 0.0)
            for path in finger_mesh_paths:
                prim = stage.GetPrimAtPath(path)
                collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                set_attr(collision.CreateTorsionalPatchRadiusAttr(0.01), 0.01)
                set_attr(collision.CreateMinTorsionalPatchRadiusAttr(0.01), 0.01)

            for axis in ("transX", "transY", "transZ", "rotX", "rotY", "rotZ"):
                limit = UsdPhysics.LimitAPI(joint_prim, axis)
                set_attr(limit.GetLowAttr(), 0.0)
                set_attr(limit.GetHighAttr(), 0.0)

            for prim in stage.Traverse():
                if prim.IsA(UsdPhysics.Scene):
                    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(prim)
                    set_attr(physx_scene.GetTimeStepsPerSecondAttr(), PHYSICS_HZ)
    except Exception:
        session_layer.ImportFromString(session_layer_snapshot)
        raise

    def restore():
        if not session_layer.ImportFromString(session_layer_snapshot):
            raise RuntimeError("未能恢复抓取测试前的 USD session layer")

    return restore


def _table_top(stage):
    table = motion_module["ExistingCuboid"](stage, TABLE_PATH)
    return float(table.center[2] + table.size[2] / 2.0)


def _collision_section_center(stage, z_world):
    collision_root = stage.GetPrimAtPath(BODY_PATH + "/Collisions")
    section_points = []
    for prim in Usd.PrimRange(collision_root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get() or []
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        for point in points:
            world = transform.Transform(Gf.Vec3d(point))
            if abs(float(world[2]) - z_world) <= 0.011:
                section_points.append((float(world[0]), float(world[1])))
    if not section_points:
        raise RuntimeError(f"水瓶碰撞网格在抓取高度 {z_world:.3f}m 没有截面点")
    section = np.asarray(section_points, dtype=float)
    return (section.min(axis=0) + section.max(axis=0)) / 2.0


class WaterBottleGrasp:
    def __init__(self, stage):
        global motion_module
        if motion_module is None:
            motion_module = __import__("runpy").run_path(str(PROJECT_ROOT / 'src/bottle_sim/motion.py'))

        body_xf = stage.GetPrimAtPath(BODY_PATH)
        # Use the scene world transform; the body is nested under the imported asset.
        body_position = np.array(
            UsdGeom.Xformable(body_xf).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation(),
            dtype=float,
        )
        table_z = _table_top(stage)
        grasp_tcp = body_position.copy()
        grasp_tcp[2] = table_z + GRASP_HEIGHT_ABOVE_TABLE_M
        collision_center_xy = _collision_section_center(stage, grasp_tcp[2])
        grasp_tcp[:2] = collision_center_xy + np.array(
            [GRASP_TCP_X_OFFSET_M, GRASP_TCP_Y_OFFSET_M], dtype=float)
        approach_direction = np.array([1.0, 0.0, 0.0])
        approach_tcp = grasp_tcp - approach_direction * APPROACH_DISTANCE_M
        pregrasp_tcp = approach_tcp + np.array([0.0, 0.0, 0.18])
        hand_rotation = np.array([[0.0, 0.0, 1.0],
                                  [0.0, 1.0, 0.0],
                                  [-1.0, 0.0, 0.0]])
        self.base = motion_module["HoverController"](
            object_path=BODY_PATH,
            target_offset=pregrasp_tcp - body_position,
            hand_rotation=hand_rotation,
        )
        self.stage = stage
        self.robot = self.base.robot
        self.controller = self.base.controller
        self.body = self.base.cube
        self.hand_rotation = self.base.target_r
        self.hand_quaternion = self.base.target_q
        self.solver = self.base.solver
        self.kinematics = self.base.kinematics
        self.frame = self.base.frame
        self.tcp_in_gripper = self.base.tcp_in_gripper
        self.hand_to_gripper_r = self.base.hand_to_gripper_r
        self.finger = self.base.finger
        self.table_z = table_z
        self.grasp_tcp = grasp_tcp
        self.pregrasp_tcp = pregrasp_tcp
        self.approach_tcp = approach_tcp
        self.lift_tcp = grasp_tcp + np.array([0.0, 0.0, LIFT_HEIGHT_M])
        self.approach_direction = approach_direction
        self.start_body_p, self.start_body_q = self.body.get_world_pose()
        self.start_lid_p = np.array(
            UsdGeom.Xformable(stage.GetPrimAtPath(LID_PATH))
            .ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation(), dtype=float
        )
        self.arm_indices = np.asarray(self.kinematics.get_joints_subset().joint_indices, dtype=int)
        self.arm_command = self.robot.get_joint_positions()[self.arm_indices].astype(float)
        self.command_tcp = self.pregrasp_tcp.copy()
        self.motion_goal = self.pregrasp_tcp.copy()
        self.arm_goal = self._solve_arm_goal(self.pregrasp_tcp)
        self._set_gripper(OPEN_POSITION_M)
        self.phase = "PreGrasp"
        self.phase_time = 0.0
        self.elapsed = 0.0
        self.stable_seconds = 0.0
        self.hold_seconds = 0.0
        self.last_finger_position = OPEN_POSITION_M
        self.finger_stable_seconds = 0.0
        self.grip_hold_position = None
        self.relative_reference = None
        self.max_slip_m = 0.0
        self.max_lift_m = 0.0
        self.status = "Running"
        self.detail = ""
        self.trace = [{"phase": self.phase, "elapsed_s": 0.0}]
        self.contact_geometry = {}
        self.initial_body_position = self.start_body_p.copy()
        print(f"[水瓶固定坐标抓取] body_start={np.round(self.start_body_p, 4).tolist()}, grasp_tcp={np.round(self.grasp_tcp, 4).tolist()}")

    def _solve_arm_goal(self, tcp):
        model_target = tcp - self.hand_rotation @ self.tcp_in_gripper
        q, ok = self.solver.compute_inverse_kinematics(
            self.frame, model_target, self.hand_quaternion, warm_start=self.arm_command,
            position_tolerance=0.001, orientation_tolerance=0.01,
        )
        if not ok:
            raise RuntimeError(f"固定目标 IK 无解: {np.round(tcp, 4).tolist()}")
        return np.asarray(q, dtype=float)

    def _set_gripper(self, position):
        self.controller.apply_action(ArticulationAction(
            joint_positions=np.array([position]), joint_indices=np.array([self.finger])))

    def _set_tcp_goal(self, goal):
        self.motion_goal = np.asarray(goal, dtype=float).copy()
        self.stable_seconds = 0.0

    def _advance_tcp(self, dt):
        delta = self.motion_goal - self.command_tcp
        distance = float(np.linalg.norm(delta))
        if distance:
            self.command_tcp += delta * min(1.0, MOVE_SPEED_M_S * dt / distance)
        attempts = []
        for position_tolerance, orientation_tolerance in ((0.0005, 0.005), (0.001, 0.01), (0.002, 0.02)):
            q, ok = self.solver.compute_inverse_kinematics(
                self.frame, self.command_tcp - self.hand_rotation @ self.tcp_in_gripper,
                self.hand_quaternion, warm_start=self.arm_command.copy(),
                position_tolerance=position_tolerance, orientation_tolerance=orientation_tolerance,
            )
            delta_q = float(np.max(np.abs(q - self.arm_command))) if ok else None
            attempts.append((bool(ok), delta_q))
            if ok and delta_q < 0.25:
                self.arm_goal = np.asarray(q, dtype=float)
                return
        raise RuntimeError(f"逐步 IK 未收敛或解分支跳变: {attempts}")

    def _actual_pose(self):
        hp, hq = self.base.hand.get_world_pose()
        hr = quat_to_rot_matrix(hq)
        return hp + hr @ self.base.tcp_local, hr @ self.hand_to_gripper_r

    def _apply_arm(self, dt):
        previous = self.arm_command.copy()
        delta = self.arm_goal - previous
        scale = min(1.0, MAX_ARM_TARGET_SPEED_RAD_S * dt /
                    max(float(np.max(np.abs(delta))), 1e-9))
        self.arm_command += delta * scale
        actual = self.robot.get_joint_positions()[self.arm_indices]
        if np.max(np.abs(self.arm_command - actual)) > 0.25:
            raise RuntimeError("机械臂关节未跟上限速目标")
        self.controller.apply_action(ArticulationAction(
            joint_positions=self.arm_command,
            joint_velocities=(self.arm_command - previous) / dt,
            joint_indices=self.arm_indices,
        ))

    def _reached(self, tcp, rotation, goal, dt):
        good = (np.linalg.norm(self.command_tcp - goal) < 1e-4
                and np.linalg.norm(tcp - goal) < 0.006
                and motion_module["angle_error"](rotation, self.hand_rotation) < np.deg2rad(3.0))
        self.stable_seconds = self.stable_seconds + dt if good else 0.0
        return self.stable_seconds >= 0.25

    def _transition(self, phase):
        self.phase = phase
        self.phase_time = 0.0
        self.stable_seconds = 0.0
        self.trace.append({"phase": phase, "elapsed_s": self.elapsed})
        print(f"[水瓶固定坐标抓取] {phase}")

    def _capture_contact_geometry(self, label):
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        tcp, _ = self._actual_pose()
        paths = (
            BODY_PATH,
            "/World/Franka/panda_hand",
            "/World/Franka/panda_leftfinger/geometry/panda_leftfinger",
            "/World/Franka/panda_rightfinger/geometry/panda_rightfinger",
        )
        snapshot = {}
        for path in paths:
            prim = self.stage.GetPrimAtPath(path)
            if not prim:
                snapshot[path] = None
                continue
            bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            snapshot[path] = {
                "center_world_m": np.asarray(bounds.GetMidpoint(), dtype=float).tolist(),
                "size_world_m": np.asarray(bounds.GetSize(), dtype=float).tolist(),
            }
        collision_root = self.stage.GetPrimAtPath(BODY_PATH + "/Collisions")
        section_points = []
        for prim in Usd.PrimRange(collision_root):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get() or []
            transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            for point in points:
                world = transform.Transform(Gf.Vec3d(point))
                if abs(float(world[2]) - float(tcp[2])) <= 0.011:
                    section_points.append((float(world[0]), float(world[1])))
        if section_points:
            section = np.asarray(section_points, dtype=float)
            snapshot["bottle_collision_section_at_tcp_z"] = {
                "z_world_m": float(tcp[2]),
                "point_count": int(len(section)),
                "xy_min_world_m": section.min(axis=0).tolist(),
                "xy_max_world_m": section.max(axis=0).tolist(),
                "xy_size_world_m": (section.max(axis=0) - section.min(axis=0)).tolist(),
            }
        snapshot["tcp_world_m"] = np.asarray(tcp, dtype=float).tolist()
        snapshot["finger_position_m"] = float(self.robot.get_joint_positions()[self.finger])
        snapshot["finger_efforts_n"] = self.robot.get_measured_joint_efforts()[-2:].tolist()
        self.contact_geometry[label] = snapshot

    def tick(self, dt):
        if self.status != "Running":
            return
        try:
            dt = float(dt)
            self.elapsed += dt
            self.phase_time += dt
            tcp, rotation = self._actual_pose()
            body_p, body_q = self.body.get_world_pose()
            lid_p = np.array(
                UsdGeom.Xformable(self.stage.GetPrimAtPath(LID_PATH))
                .ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation(), dtype=float
            )
            finger_position = float(self.robot.get_joint_positions()[self.finger])
            lift_m = float(body_p[2] - self.start_body_p[2])
            self.max_lift_m = max(self.max_lift_m, lift_m)
            if not np.all(np.isfinite(np.r_[tcp, body_p, body_q, lid_p, finger_position])):
                raise RuntimeError("检测到非有限物理状态")
            if self.elapsed > 90.0:
                raise RuntimeError("抓取测试超时")
            if self.phase in ("PreGrasp", "Align", "Approach", "Close") and np.linalg.norm(body_p - self.start_body_p) > 0.01:
                raise RuntimeError("夹持前水瓶已明显移动，检查桌面接触与碰撞体")
            lid_relative = np.linalg.norm((lid_p - body_p) - (self.start_lid_p - self.start_body_p))
            if self.phase not in ("PreGrasp", "Approach") and lid_relative > 0.008:
                raise RuntimeError(f"瓶盖相对瓶身移动 {lid_relative * 1000:.1f}mm")
            if self.phase in ("Lift", "Hold") and self.relative_reference is not None:
                slip = float(np.linalg.norm(body_p - tcp - self.relative_reference))
                self.max_slip_m = max(self.max_slip_m, slip)
                if slip > MAX_SLIP_M:
                    raise RuntimeError(f"夹持滑移超过 {MAX_SLIP_M * 100:.1f}cm")

            if self.phase == "PreGrasp":
                if self._reached(tcp, rotation, self.pregrasp_tcp, dt):
                    self._set_tcp_goal(self.approach_tcp)
                    self._transition("Align")
            elif self.phase == "Align":
                self._advance_tcp(dt)
                if self._reached(tcp, rotation, self.approach_tcp, dt):
                    self._set_tcp_goal(self.grasp_tcp)
                    self._transition("Approach")
            elif self.phase == "Approach":
                self._advance_tcp(dt)
                if self._reached(tcp, rotation, self.grasp_tcp, dt):
                    self._capture_contact_geometry("before_close")
                    self._transition("Close")
            elif self.phase == "Close":
                close_target = max(0.0, OPEN_POSITION_M - GRIP_CLOSE_SPEED_M_S * self.phase_time)
                self._set_gripper(close_target)
                if (close_target < finger_position - 0.001
                        and abs(finger_position - self.last_finger_position) < 0.0002):
                    self.finger_stable_seconds += dt
                else:
                    self.finger_stable_seconds = 0.0
                self.last_finger_position = finger_position
                # Franka finger joint position is the half-width of the opening.
                expected_position = EXPECTED_NECK_DIAMETER_M / 2.0
                valid_contact_range = (expected_position - GRIP_CONTACT_POSITION_TOLERANCE_M
                                       < finger_position
                                       < expected_position + GRIP_CONTACT_POSITION_TOLERANCE_M)
                if self.finger_stable_seconds >= GRIP_CONTACT_STABLE_S and valid_contact_range:
                    self._capture_contact_geometry("after_close")
                    self.grip_hold_position = max(0.0, finger_position - GRIP_HOLD_PRELOAD_M)
                    self.relative_reference = body_p - tcp
                    self._set_tcp_goal(self.lift_tcp)
                    self._transition("Lift")
                elif self.phase_time >= GRIP_CLOSE_TIMEOUT_S:
                    self._capture_contact_geometry("after_close")
                    raise RuntimeError(
                        f"闭爪未在瓶颈预期位置稳定接触，finger={finger_position:.4f}m，"
                        f"expected≈{expected_position:.4f}m")
            elif self.phase == "Lift":
                self._set_gripper(self.grip_hold_position)
                self._advance_tcp(dt)
                object_lifted = lift_m >= LIFT_HEIGHT_M * 0.90
                object_still_grasped = self.max_slip_m <= MAX_SLIP_M
                if object_lifted and object_still_grasped:
                    self._transition("Hold")
                elif self._reached(tcp, rotation, self.lift_tcp, dt):
                    raise RuntimeError(f"夹爪到达抬升点，但瓶身仅升高 {lift_m * 1000:.1f}mm")
            elif self.phase == "Hold":
                self.hold_seconds += dt
                if self.hold_seconds >= HOLD_SECONDS:
                    self.status = "Succeeded"
                    self.detail = "固定坐标侧向夹持并保持抬升"
                    return
            self._apply_arm(dt)
        except Exception as exc:
            self.status = "Failed"
            self.detail = str(exc)

    def report(self):
        body_p, body_q = self.body.get_world_pose()
        tcp, _ = self._actual_pose()
        result = {
            "status": self.status,
            "phase": self.phase,
            "detail": self.detail,
            "elapsed_s": self.elapsed,
            "target_body_path": BODY_PATH,
            "start_body_world_m": self.start_body_p.tolist(),
            "final_body_world_m": np.asarray(body_p).tolist(),
            "grasp_tcp_world_m": self.grasp_tcp.tolist(),
            "final_tcp_world_m": np.asarray(tcp).tolist(),
            "lift_m": float(body_p[2] - self.start_body_p[2]),
            "max_lift_m": self.max_lift_m,
            "max_slip_m": self.max_slip_m,
            "finger_position_m": float(self.robot.get_joint_positions()[self.finger]),
            "grip_hold_position_m": self.grip_hold_position,
            "contact_geometry": self.contact_geometry,
            "phases": self.trace,
            "physics_assumptions": {
                "body_mass_kg": TEST_BODY_MASS_KG,
                "lid_mass_kg": TEST_LID_MASS_KG,
                "bottle_mass_is_measured": False,
                "lid_joint_locked_for_test": True,
                "grasp_height_above_table_m": GRASP_HEIGHT_ABOVE_TABLE_M,
                "grasp_tcp_x_offset_m": GRASP_TCP_X_OFFSET_M,
                "grasp_tcp_y_offset_m": GRASP_TCP_Y_OFFSET_M,
                "lift_height_m": LIFT_HEIGHT_M,
                "gripper_opening_m": OPEN_POSITION_M * 2.0,
                "gripper_close_speed_m_s": GRIP_CLOSE_SPEED_M_S,
                "gripper_opening_is_stock_franka": False,
                "gripper_max_force_n": GRIP_MAX_FORCE_N,
                "grip_surface_friction": GRIP_SURFACE_FRICTION,
                "finger_length_scale": GRIPPER_FINGER_LENGTH_SCALE,
                "tcp_extension_m": TCP_EXTENSION_M,
                "grip_hold_preload_m": GRIP_HOLD_PRELOAD_M,
                "expected_neck_diameter_m": EXPECTED_NECK_DIAMETER_M,
            },
        }
        setattr(builtins, RESULT_KEY, result)
        print("[水瓶固定坐标抓取] " + json.dumps(result, ensure_ascii=False))
        return result


async def run_test():
    timeline = omni.timeline.get_timeline_interface()
    if timeline.is_playing():
        raise RuntimeError("请先 Stop 当前仿真")
    timeline.stop()
    await omni.kit.app.get_app().next_update_async()
    stage = omni.usd.get_context().get_stage()
    restore_settings = None
    controller = None
    subscription = None
    started = time.monotonic()
    result = {"status": "Failed", "phase": "Initialize", "detail": "抓取控制初始化失败"}
    try:
        restore_settings = _temporary_physics_setup(stage)
        await omni.kit.app.get_app().next_update_async()
        controller = WaterBottleGrasp(stage)
        subscription = omni.physx.get_physx_interface().subscribe_physics_step_events(controller.tick)
        timeline.play()
        while controller.status == "Running":
            await omni.kit.app.get_app().next_update_async()
            if time.monotonic() - started > WALL_TIMEOUT_S:
                controller.status, controller.detail = "TimedOut", "墙钟超时"
                break
    finally:
        subscription = None
        if timeline.is_playing():
            timeline.pause()
        if controller is not None:
            try:
                result = controller.report()
            except Exception as exc:
                result.update({"phase": controller.phase, "detail": f"结果读取失败: {exc}"})
        timeline.stop()
        await omni.kit.app.get_app().next_update_async()
        if restore_settings is not None:
            restore_settings()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"waterbottle_fixed_grasp_{time.strftime('%Y%m%d_%H%M%S')}.json"
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[水瓶固定坐标抓取] 报告: {output_path}")


def start():
    previous = getattr(builtins, TASK_KEY, None)
    if previous is not None and not previous.done():
        raise RuntimeError("水瓶固定坐标抓取测试正在运行")
    task = asyncio.ensure_future(run_test())
    setattr(builtins, TASK_KEY, task)
    return task


if __name__ == "__main__":
    start()
