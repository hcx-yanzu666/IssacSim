"""End-to-end RGB-D bottle pick using the estimated USD-template pose.

Run from Isaac Sim Script Editor while the simulation is stopped or paused.
The pose estimator uses the known bottle mesh and a fixed orientation prior;
it estimates translation from RGB-D and checks IK before starting lesson 09.
"""
import asyncio
import builtins
import json
from pathlib import Path

import numpy as np
import omni.timeline
import omni.usd
import omni.kit.app
import omni.replicator.core as rep
from PIL import Image
from pxr import Usd, UsdGeom


SCRIPT_DIR = Path(__file__).resolve().parent
ESTIMATOR_SCRIPT = SCRIPT_DIR / "11_register_bottle_translation.py"
PICK_SCRIPT = SCRIPT_DIR / "09_pick_and_place_bottle.py"
DATA_DIR = SCRIPT_DIR.parent / "outputs" / "wrist_rgbd_01"
ESTIMATE_PATH = DATA_DIR / "bottle_pose_estimate.json"
TASK_KEY = "_isaac_visual_bottle_pick_task"
RUNNER_RESULT = SCRIPT_DIR.parent / "outputs" / "visual_pick_runner_latest.json"


async def start_visual_pick(place_xy_world_m=None, progress_callback=None,
                           cancel_requested=None, data_dir=None,
                           depth_noise_sigma_m=0.0, depth_noise_seed=0,
                           prepare_scene_callback=None):
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    if not np.isfinite(depth_noise_sigma_m) or depth_noise_sigma_m < 0:
        raise ValueError("depth_noise_sigma_m must be finite and non-negative")

    def report_phase(phase, detail):
        if progress_callback is not None:
            progress_callback(phase, detail)

    def check_cancel():
        if cancel_requested is not None and cancel_requested():
            raise asyncio.CancelledError()

    timeline = omni.timeline.get_timeline_interface()
    if timeline.is_playing():
        raise RuntimeError("请先暂停/停止当前仿真，再启动视觉抓取")

    # The saved image is only a dataset artifact. Reset and recapture each run
    # so the RGB-D frame, object pose and wrist-camera pose are synchronized.
    timeline.stop()
    await omni.kit.app.get_app().next_update_async()
    check_cancel()
    if prepare_scene_callback is not None:
        prepare_scene_callback()
        await omni.kit.app.get_app().next_update_async()
    stage = omni.usd.get_context().get_stage()
    camera_path = "/World/Franka/panda_hand/WristCamera"
    # Capture the imported bottle014 asset.  The old mustard-bottle prim was
    # retained only in historical scripts and is not part of the active scene.
    bottle_path = "/World/PracticeBottle/bottle014"
    camera_prim = stage.GetPrimAtPath(camera_path)
    bottle_prim = stage.GetPrimAtPath(bottle_path)
    if not camera_prim or not bottle_prim:
        raise RuntimeError("场景缺少 WristCamera 或瓶子 Mesh")

    report_phase("capturing_rgbd", "采集手眼相机 RGB-D 和分割 mask")
    data_dir.mkdir(parents=True, exist_ok=True)
    render_product = rep.create.render_product(camera_path, (640, 360))
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    # instance_id_segmentation preserves the USD prim path for every visible
    # mesh.  The semantic annotator only returns generic UNLABELLED here,
    # which would make a valid bottle produce an empty mask.
    mask_annotator = rep.AnnotatorRegistry.get_annotator("instance_id_segmentation")
    for annotator in (rgb_annotator, depth_annotator, mask_annotator):
        annotator.attach([render_product])
    try:
        await rep.orchestrator.step_async(rt_subframes=4)
        check_cancel()
        rgb = np.asarray(rgb_annotator.get_data())[:, :, :3].astype(np.uint8)
        depth = np.asarray(depth_annotator.get_data(), dtype=np.float32)
        seg = mask_annotator.get_data()
        seg_data = np.asarray(seg["data"])
        if seg_data.ndim == 3 and seg_data.shape[-1] == 1:
            seg_data = seg_data[..., 0]
        labels = seg.get("info", {}).get("idToLabels", {})
        ids = [int(key) for key, value in labels.items()
               if isinstance(value, str)
               and (value == bottle_path or value.startswith(bottle_path + "/"))]
        mask = np.isin(seg_data, ids)
        if rgb.shape != (360, 640, 3) or depth.shape != (360, 640) or not mask.any():
            raise RuntimeError(
                f"RGB-D/mask采集异常：rgb={rgb.shape}, depth={depth.shape}, "
                f"mask_pixels={int(mask.sum())}, ids={ids}"
            )

        valid_depth = np.isfinite(depth) & (depth > 0)
        if depth_noise_sigma_m > 0:
            noise_rng = np.random.default_rng(int(depth_noise_seed))
            noisy_depth = depth.copy()
            noise = noise_rng.normal(0.0, depth_noise_sigma_m, size=depth.shape).astype(np.float32)
            noisy_depth[valid_depth] += noise[valid_depth]
            noisy_depth[valid_depth] = np.maximum(noisy_depth[valid_depth], 1e-4)
        else:
            noisy_depth = depth.copy()

        Image.fromarray(rgb, "RGB").save(data_dir / "rgb.png")
        np.save(data_dir / "depth_m.npy", depth)
        np.save(data_dir / "depth_noisy_m.npy", noisy_depth)
        Image.fromarray(mask.astype(np.uint8) * 255, "L").save(data_dir / "bottle_mask.png")

        camera = UsdGeom.Camera(camera_prim)
        focal = float(camera.GetFocalLengthAttr().Get())
        horizontal_aperture = float(camera.GetHorizontalApertureAttr().Get())
        vertical_aperture = float(camera.GetVerticalApertureAttr().Get())
        xform = UsdGeom.XformCache().GetLocalToWorldTransform(camera_prim)
        camera_to_world = [[float(xform[i][j]) for j in range(4)] for i in range(4)]
        bottle_world = UsdGeom.Xformable(bottle_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        joint_pose = None
        try:
            from isaacsim.core.prims import SingleArticulation
            articulation = SingleArticulation(
                prim_path="/World/Franka", name="visual_pick_pose_snapshot",
                reset_xform_properties=False,
            )
            articulation.initialize()
            joint_pose = [float(v) for v in articulation.get_joint_positions()]
        except Exception as exc:
            print("[视觉抓取] joint snapshot unavailable:", str(exc))

        metadata = {
            "camera_prim": camera_path,
            "resolution": [640, 360],
            "depth_encoding": "float32 meters; distance_to_image_plane",
            "intrinsics_px": {
                "fx": focal / horizontal_aperture * 640,
                "fy": focal / vertical_aperture * 360,
                "cx": 320.0,
                "cy": 180.0,
            },
            "usd_camera_mm": {
                "focal_length": focal,
                "horizontal_aperture": horizontal_aperture,
                "vertical_aperture": vertical_aperture,
            },
            "camera_to_world_matrix_usd_row_vector_convention": camera_to_world,
            "observation_joint_pose_rad": joint_pose,
            "bottle_body_translation_world_m_evaluation_only": [
                float(v) for v in bottle_world.ExtractTranslation()
            ],
        }
        (data_dir / "camera_info.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        mask_info = {
            "camera_prim": camera_path,
            "resolution": [640, 360],
            "annotator": "instance_segmentation",
            "target_prim_path": bottle_path,
            "instance_ids": ids,
            "white_pixel_count": int(mask.sum()),
            "white_pixel_ratio": float(mask.mean()),
        }
        (data_dir / "bottle_mask_info.json").write_text(
            json.dumps(mask_info, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("[视觉抓取] synchronized RGB-D captured; mask pixels:", int(mask.sum()))
    finally:
        for annotator in (rgb_annotator, depth_annotator, mask_annotator):
            annotator.detach([render_product])
        render_product.destroy()

    # Recompute from saved sensor outputs; do not reuse a stale estimate.
    check_cancel()
    report_phase("estimating_pose", "根据深度、mask 和瓶子模板估计抓取位置")
    import runpy
    estimator = runpy.run_path(str(ESTIMATOR_SCRIPT))
    estimate = estimator["main"](
        data_dir=data_dir,
        depth_path=data_dir / "depth_noisy_m.npy",
    )
    precheck = estimate["grasp_ik_precheck"]
    if not precheck["all_ik_succeeded"]:
        raise RuntimeError("视觉估计的预抓取/抓取位姿未通过 IK，任务未启动")

    report_phase("ik_check", "抓取位姿可达性检查通过")
    check_cancel()
    body_position = estimate["estimated_body_translation_world_m"]
    pick = runpy.run_path(str(PICK_SCRIPT))
    task = pick["start"](bottle_position_world_m=body_position,
                          place_xy_world_m=place_xy_world_m,
                          progress_callback=progress_callback,
                          cancel_requested=cancel_requested)
    if task is None:
        raise RuntimeError("抓取任务没有启动；检查是否有上一次控制任务仍在运行")
    print("[视觉抓取] RGB-D估计瓶子抓取位置：" + json.dumps(body_position))
    if place_xy_world_m is not None:
        print("[视觉抓取] 指定放置位置 world XY：" + json.dumps(place_xy_world_m))
    await task
    return getattr(builtins, "_isaac_lesson09_result", {})


if __name__ == "__main__":
    previous = getattr(builtins, TASK_KEY, None)
    if previous is not None and not previous.done():
        raise RuntimeError("视觉抓取入口已有任务运行中")
    task = asyncio.ensure_future(start_visual_pick())
    setattr(builtins, TASK_KEY, task)

    def report_runner_result(done_task):
        result = {"status": "Unknown", "detail": "抓取任务未返回结果"}
        if done_task.cancelled():
            result = {"status": "Cancelled", "detail": "视觉抓取入口任务被取消"}
        elif done_task.exception() is not None:
            result = {
                "status": "Failed",
                "detail": repr(done_task.exception()),
            }
            print("[视觉抓取入口] FAILED:", result["detail"])
        else:
            pick_result = done_task.result()
            result = {
                "status": pick_result.get("status", "Unknown"),
                "phase": pick_result.get("phase", ""),
                "detail": pick_result.get("detail", ""),
                "max_hold_transport_slip_m": pick_result.get("max_hold_transport_slip_m"),
                "place_target_m": pick_result.get("place_target_m"),
                "metrics": pick_result.get("metrics", {}),
            }
        RUNNER_RESULT.parent.mkdir(parents=True, exist_ok=True)
        RUNNER_RESULT.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    task.add_done_callback(report_runner_result)
    print("[视觉抓取入口] RGB-D采集与任务初始化已排队")
