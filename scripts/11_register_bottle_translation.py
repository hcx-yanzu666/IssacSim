"""RGB-D bottle translation estimate using a known USD mesh template.

Run in Isaac Sim's Script Editor after wrist_rgbd_01 has been captured.
The bottle orientation is assumed fixed and read as a setup prior from USD;
the bottle translation is estimated from the mask-filtered depth cloud.
Ground-truth translation is read only after fitting, for simulation evaluation.
"""
import json
import runpy
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import omni.usd
from pxr import Gf, Usd, UsdGeom
from isaacsim.core.utils.rotations import rot_matrix_to_quat
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
    interface_config_loader,
)


DATA_DIR = Path(__file__).resolve().parent.parent / "outputs" / "wrist_rgbd_01"
# Current primary asset.  Keep the old mustard-bottle path out of the
# perception path; it is no longer present after the bottle014 replacement.
BODY_PATH = "/World/PracticeBottle/bottle014/body"
MESH_PATH = BODY_PATH + "/Visuals"
CAMERA_PATH = "/World/Franka/panda_hand/WristCamera"
SAMPLE_STRIDE = 4
MODEL_SAMPLE_COUNT = 4000
ICP_ITERATIONS = 30
INITIAL_RAY_OFFSET_M = 0.03


def erode_binary(mask, radius=3):
    """Remove silhouette pixels where foreground/background depth can mix."""
    height, width = mask.shape
    result = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = np.zeros_like(mask)
            y0, y1 = max(0, dy), min(height, height + dy)
            x0, x1 = max(0, dx), min(width, width + dx)
            shifted[y0:y1, x0:x1] = mask[y0-dy:y1-dy, x0-dx:x1-dx]
            result &= shifted
    return result


def camera_depth_to_world(mask, depth, intrinsics, camera_to_world):
    """Back-project valid masked pixels; USD camera forward is local -Z."""
    valid = erode_binary(mask) & np.isfinite(depth) & (depth > 0)
    rows, cols = np.where(valid)
    keep = (rows % SAMPLE_STRIDE == 0) & (cols % SAMPLE_STRIDE == 0)
    rows, cols = rows[keep], cols[keep]
    z = depth[rows, cols]
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    points_camera = np.column_stack((
        (cols - cx) * z / fx,
        -(rows - cy) * z / fy,
        -z,
        np.ones_like(z),
    ))
    return (points_camera @ camera_to_world)[:, :3]


def fit_translation_icp(observed_world, model_body, body_rotation, camera_forward):
    """Point-to-point ICP with fixed model rotation and robust median updates."""
    rng = np.random.default_rng(7)
    if len(model_body) > MODEL_SAMPLE_COUNT:
        model_body = model_body[rng.choice(len(model_body), MODEL_SAMPLE_COUNT, replace=False)]
    model_world_zero = model_body @ body_rotation

    # The observed cloud is only the visible surface. A short camera-ray offset
    # gives ICP an initial estimate near the object's interior.
    translation = np.median(observed_world, axis=0) + camera_forward * INITIAL_RAY_OFFSET_M
    for _ in range(ICP_ITERATIONS):
        delta_points = observed_world[:, None, :] - (model_world_zero[None, :, :] + translation)
        distances_sq = np.einsum("ijk,ijk->ij", delta_points, delta_points)
        nearest = distances_sq.argmin(axis=1)
        residual = observed_world - (model_world_zero[nearest] + translation)
        residual_norm = np.linalg.norm(residual, axis=1)
        cutoff = max(float(np.percentile(residual_norm, 75)), 0.02)
        inliers = residual_norm <= cutoff
        update = np.median(residual[inliers], axis=0)
        translation += update
        if np.linalg.norm(update) < 1e-5:
            break

    final_delta = observed_world[:, None, :] - (model_world_zero[None, :, :] + translation)
    final_distances = np.sqrt(np.einsum("ijk,ijk->ij", final_delta, final_delta).min(axis=1))
    return translation, final_distances


def precheck_grasp_ik(body_translation):
    """Check the lesson's approach/grasp TCP poses without commanding motion."""
    script_dir = Path(__file__).resolve().parent
    lesson04 = runpy.run_path(str(script_dir / "04_hover_above_cube.py"))
    lesson02 = runpy.run_path(str(script_dir / "02_read_joint_state.py"))
    robot, _ = lesson02["initialize_robot"]()

    config = interface_config_loader.load_supported_motion_policy_config("Franka", "RMPflow")
    frame = config["end_effector_frame_name"]
    solver = LulaKinematicsSolver(config["robot_description_path"], config["urdf_path"])
    base_position, base_orientation = robot.get_world_pose()
    solver.set_robot_base_pose(base_position, base_orientation)
    kinematics = ArticulationKinematicsSolver(robot, solver, frame)
    arm_q = kinematics.get_joints_subset().get_joint_positions()
    gripper_position, gripper_rotation = solver.compute_forward_kinematics(frame, arm_q)
    hand_position, hand_rotation_fk = solver.compute_forward_kinematics("panda_hand", arm_q)

    stage = omni.usd.get_context().get_stage()
    tcp_prim = stage.GetPrimAtPath("/World/Franka/panda_hand/tool_center")
    tcp_local = np.asarray(
        UsdGeom.Xformable(tcp_prim).GetLocalTransformation().ExtractTranslation(), dtype=float
    )
    hand_to_gripper_rotation = hand_rotation_fk.T @ gripper_rotation
    tcp_in_gripper = gripper_rotation.T @ (
        hand_position + hand_rotation_fk @ tcp_local - gripper_position
    )

    # Match the existing lesson's fixed side-approach grasp orientation.
    desired_hand_rotation = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    target_rotation = desired_hand_rotation @ hand_to_gripper_rotation
    target_quaternion = rot_matrix_to_quat(target_rotation)
    table = lesson04["ExistingCuboid"](stage, "/World/DeskTop")
    table_z = float(table.center[2] + table.size[2] / 2.)
    grasp_tcp = np.asarray(body_translation, dtype=float).copy()
    grasp_tcp[2] = table_z + 0.10
    approach_tcp = grasp_tcp + np.array([-0.10, 0., 0.])

    checks = []
    warm_start = np.asarray(arm_q, dtype=float)
    for name, tcp in (("approach", approach_tcp), ("grasp", grasp_tcp)):
        model_target = tcp - target_rotation @ tcp_in_gripper
        solution, success = solver.compute_inverse_kinematics(
            frame, model_target, target_quaternion, warm_start=warm_start,
            position_tolerance=0.001, orientation_tolerance=0.01,
        )
        item = {"name": name, "tcp_world_m": tcp.tolist(), "ik_success": bool(success)}
        if success:
            arm_q = np.asarray(solution, dtype=float)
            fk_position, fk_rotation = solver.compute_forward_kinematics(frame, arm_q)
            fk_tcp = fk_position + fk_rotation @ tcp_in_gripper
            item["fk_tcp_world_m"] = fk_tcp.tolist()
            item["fk_position_error_m"] = float(np.linalg.norm(fk_tcp - tcp))
            item["joint_solution_rad"] = arm_q.tolist()
            warm_start = arm_q
        checks.append(item)
    return {
        "moves_robot": False,
        "estimated_bottle_body_translation_world_m": np.asarray(body_translation).tolist(),
        "table_top_z_m": table_z,
        "checks": checks,
        "all_ik_succeeded": all(item["ik_success"] for item in checks),
        "note": "IK reachability precheck only; no collision-free path or grasp success is implied.",
    }


def main(data_dir=DATA_DIR, depth_path=None):
    data_dir = Path(data_dir)
    depth_path = Path(depth_path) if depth_path is not None else data_dir / "depth_m.npy"
    stage = omni.usd.get_context().get_stage()
    body = stage.GetPrimAtPath(BODY_PATH)
    mesh = stage.GetPrimAtPath(MESH_PATH)
    camera = stage.GetPrimAtPath(CAMERA_PATH)
    if not body or not mesh or not camera:
        raise RuntimeError("当前场景缺少瓶子 Body、瓶子 Mesh 或 WristCamera")

    mask = np.asarray(Image.open(data_dir / "bottle_mask.png").convert("L")) > 0
    depth = np.load(depth_path).astype(np.float64)
    rgb = np.asarray(Image.open(data_dir / "rgb.png").convert("RGB"))
    with (data_dir / "camera_info.json").open("r", encoding="utf-8") as stream:
        camera_info = json.load(stream)
    if mask.shape != depth.shape or rgb.shape[:2] != depth.shape:
        raise RuntimeError("RGB、mask、depth 尺寸不一致")

    camera_to_world = np.asarray(
        camera_info["camera_to_world_matrix_usd_row_vector_convention"], dtype=np.float64
    )
    forward = -camera_to_world[2, :3]
    forward /= np.linalg.norm(forward)
    observed_world = camera_depth_to_world(
        mask, depth, camera_info["intrinsics_px"], camera_to_world
    )
    if len(observed_world) < 50:
        raise RuntimeError(f"有效深度点不足：{len(observed_world)}")

    # Convert USD mesh vertices into the rigid body's local frame. This keeps
    # mesh scale/offset while treating the root-body translation as unknown.
    body_world = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    mesh_world = UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    mesh_to_body = mesh_world * body_world.GetInverse()
    raw_vertices = UsdGeom.Mesh(mesh).GetPointsAttr().Get()
    model_body = np.asarray([
        mesh_to_body.Transform(Gf.Vec3d(float(v[0]), float(v[1]), float(v[2])))
        for v in raw_vertices
    ], dtype=np.float64)

    # Fixed-orientation prior for this demonstration; translation is estimated.
    body_rotation = np.asarray([[float(body_world[i][j]) for j in range(3)] for i in range(3)])
    estimated_translation, residuals = fit_translation_icp(
        observed_world, model_body, body_rotation, forward
    )

    # Simulation ground truth is used only to report the estimator's error.
    ground_truth = np.asarray(body_world.ExtractTranslation(), dtype=np.float64)
    error = estimated_translation - ground_truth
    metrics = {
        "estimated_body_translation_world_m": estimated_translation.tolist(),
        "ground_truth_body_translation_world_m_evaluation_only": ground_truth.tolist(),
        "translation_error_m_evaluation_only": float(np.linalg.norm(error)),
        "translation_error_xyz_m_evaluation_only": error.tolist(),
        "orientation_source": "fixed setup prior from USD; not estimated by this step",
        "translation_estimation": "mask/depth back-projection + known USD mesh point-to-point ICP",
        "observed_depth_points": int(len(observed_world)),
        "model_vertices": int(len(model_body)),
        "median_nearest_surface_residual_m": float(np.median(residuals)),
        "p90_nearest_surface_residual_m": float(np.percentile(residuals, 90)),
        "initial_ray_offset_m": INITIAL_RAY_OFFSET_M,
        "depth_input_file": depth_path.name,
    }
    metrics["grasp_ik_precheck"] = precheck_grasp_ik(estimated_translation)
    (data_dir / "bottle_pose_estimate.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Overlay the estimated body origin in the image for a visual sanity check.
    camera_point = np.r_[estimated_translation, 1.0] @ np.linalg.inv(camera_to_world)
    if camera_point[2] < 0:
        k = camera_info["intrinsics_px"]
        u = int(round(k["fx"] * camera_point[0] / -camera_point[2] + k["cx"]))
        v = int(round(k["cy"] - k["fy"] * camera_point[1] / -camera_point[2]))
        overlay = Image.fromarray(rgb.copy(), "RGB")
        draw = ImageDraw.Draw(overlay)
        draw.line((u-8, v, u+8, v), fill=(0, 255, 0), width=3)
        draw.line((u, v-8, u, v+8), fill=(0, 255, 0), width=3)
        overlay.save(data_dir / "bottle_pose_estimate_overlay.png")

    print("Bottle RGB-D template registration complete")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


if __name__ == "__main__":
    main()
