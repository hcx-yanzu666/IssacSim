"""End-to-end RGB-D bottle pick using the estimated USD-template pose.

Run from Isaac Sim Script Editor while the simulation is stopped or paused.
The pose estimator uses the known bottle mesh and a fixed orientation prior;
it estimates translation from RGB-D and checks IK before starting the bottle pick-and-place controller.
"""

from pathlib import Path as _ProjectPath
import sys as _sys
PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in _sys.path:
    _sys.path.insert(0, str(PROJECT_ROOT / "src"))

import asyncio
import builtins
import json
import os
from pathlib import Path
import runpy
import subprocess
import time
import uuid

import numpy as np
import omni.timeline
import omni.usd
import omni.kit.app
import omni.replicator.core as rep
from PIL import Image
from pxr import Usd, UsdGeom


ESTIMATOR_SCRIPT = PROJECT_ROOT / 'src/bottle_sim/pose_estimation.py'
PICK_SCRIPT = PROJECT_ROOT / 'src/bottle_sim/pick_task.py'
DATA_DIR = PROJECT_ROOT / "outputs" / "wrist_rgbd_01"
ESTIMATE_PATH = DATA_DIR / "bottle_pose_estimate.json"
TASK_KEY = "_isaac_visual_bottle_pick_task"
RUNNER_RESULT = PROJECT_ROOT / "outputs" / "visual_pick_runner_latest.json"


async def infer_yolo_mask(data_dir, capture_id, check_cancel, weights=None, device='cpu', timeout=90):
    """Poll an isolated worker without blocking the Kit update loop."""
    root = PROJECT_ROOT
    python = root / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    weights = Path(weights) if weights else root / 'outputs/yolo_train/multiview500/weights/best.pt'
    if not python.is_file() or not weights.is_file():
        raise FileNotFoundError('YOLO .venv interpreter or trained weights missing')
    helper = runpy.run_path(str(PROJECT_ROOT / 'perception/yolo_worker.py'))
    request = {
        'capture_id': capture_id, 'rgb_sha256': helper['file_hash'](data_dir / 'rgb.png'),
        'resolution': [640, 360],
    }
    (data_dir / 'yolo_request.json').write_text(json.dumps(request, indent=2), encoding='utf-8')
    env = os.environ.copy()
    for key in ('PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV'):
        env.pop(key, None)
    env['PYTHONNOUSERSITE'] = '1'
    env['PYTHONUTF8'] = '1'
    if os.name == 'nt':
        windows = Path(env.get('SystemRoot', r'C:\Windows'))
        env['PATH'] = os.pathsep.join(map(str, [python.parent, windows / 'System32', windows]))
    command = [str(python), str(PROJECT_ROOT / 'perception/yolo_worker.py'),
               '--capture', str(data_dir), '--weights', str(weights), '--device', device]
    process = None
    with (data_dir / 'yolo_worker.log').open('w', encoding='utf-8') as log:
        try:
            process = subprocess.Popen(command, cwd=str(root), env=env, stdout=log,
                stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            started = time.monotonic()
            while process.poll() is None:
                check_cancel()
                if time.monotonic() - started > timeout:
                    raise TimeoutError(f'YOLO inference exceeded {timeout}s; see yolo_worker.log')
                await omni.kit.app.get_app().next_update_async()
            check_cancel()
            if not (data_dir / 'yolo_result.json').exists():
                raise RuntimeError(f'YOLO worker exited {process.returncode}; see yolo_worker.log')
            result = helper['validate_result'](data_dir, capture_id)
            if process.returncode != 0:
                raise RuntimeError(f'YOLO worker exited {process.returncode}')
            return result
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                await asyncio.to_thread(process.wait)


async def start_visual_pick(place_xy_world_m=None, progress_callback=None,
                           cancel_requested=None, data_dir=None,
                           depth_noise_sigma_m=0.0, depth_noise_seed=0,
                           prepare_scene_callback=None, mask_source='sim',
                           yolo_weights=None, yolo_device='cpu'):
    if mask_source not in ('sim', 'yolo'):
        raise ValueError('mask_source must be sim or yolo')
    capture_id = uuid.uuid4().hex
    if mask_source == 'yolo' and data_dir is None:
        data_dir = PROJECT_ROOT / 'outputs/yolo_pick' / capture_id
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
    # Prepare instance geometry before Replicator initializes physics. Doing it
    # later in the IK precheck invalidates the freshly created tensor view.
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError('Open the bottle scene before starting a visual pick')
    compat = runpy.run_path(str(PROJECT_ROOT / 'src/bottle_sim/visual_compat.py'))
    compat['prepare_franka_visuals'](stage, '/World/Franka')
    await omni.kit.app.get_app().next_update_async()
    check_cancel()
    if prepare_scene_callback is not None:
        prepare_scene_callback()
        await omni.kit.app.get_app().next_update_async()
    stage = omni.usd.get_context().get_stage()
    pick = runpy.run_path(str(PICK_SCRIPT))
    # Author all collision/drive changes before Replicator and the IK precheck
    # initialize PhysX articulation views. Reapplying these changes afterward
    # invalidates the deprecated SingleArticulation metadata in Isaac Sim 6.1.
    pick['prepare_physics']()
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

    report_phase("capturing_rgbd", "采集手眼相机 RGB-D，绑定本次采集编号")
    data_dir.mkdir(parents=True, exist_ok=True)
    render_product = rep.create.render_product(camera_path, (640, 360))
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    # instance_id_segmentation preserves the USD prim path for every visible
    # mesh.  The semantic annotator only returns generic UNLABELLED here,
    # which would make a valid bottle produce an empty mask.
    mask_annotator = (rep.AnnotatorRegistry.get_annotator("instance_id_segmentation")
                      if mask_source == 'sim' else None)
    annotators = [rgb_annotator, depth_annotator] + ([mask_annotator] if mask_annotator else [])
    for annotator in annotators:
        annotator.attach([render_product])
    try:
        await rep.orchestrator.step_async(rt_subframes=4)
        check_cancel()
        rgb = np.asarray(rgb_annotator.get_data())[:, :, :3].astype(np.uint8)
        depth = np.asarray(depth_annotator.get_data(), dtype=np.float32)
        mask, ids = None, []
        if mask_annotator is not None:
            seg = mask_annotator.get_data()
            seg_data = np.asarray(seg['data']).squeeze()
            labels = seg.get('info', {}).get('idToLabels', {})
            ids = [int(key) for key, value in labels.items() if isinstance(value, str)
                   and (value == bottle_path or value.startswith(bottle_path + '/'))]
            mask = np.isin(seg_data, ids)
            if not mask.any():
                raise RuntimeError('Empty simulator reference mask')
        if rgb.shape != (360, 640, 3) or depth.shape != (360, 640):
            raise RuntimeError(f'RGB-D shape mismatch: {rgb.shape}, {depth.shape}')

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
        if mask is not None:
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
            'schema_version': 2,
            'capture_id': capture_id,
            'mask_source': mask_source,
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
            "bottle_parent_translation_world_m_evaluation_only": [
                float(v) for v in bottle_world.ExtractTranslation()
            ],
            'bottle_body_translation_world_m_evaluation_only': [float(v) for v in
                UsdGeom.Xformable(stage.GetPrimAtPath(bottle_path + '/body'))
                .ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()],
        }
        (data_dir / "camera_info.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        mask_info = {
            "camera_prim": camera_path,
            "resolution": [640, 360],
            "annotator": 'instance_id_segmentation' if mask_source == 'sim' else 'yolo_pending',
            "target_prim_path": bottle_path,
            "instance_ids": ids,
            "white_pixel_count": int(mask.sum()) if mask is not None else None,
            "white_pixel_ratio": float(mask.mean()) if mask is not None else None,
        }
        (data_dir / "bottle_mask_info.json").write_text(
            json.dumps(mask_info, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print('[视觉抓取] synchronized RGB-D captured:', capture_id, 'mask source:', mask_source)
    finally:
        for annotator in annotators:
            annotator.detach([render_product])
        render_product.destroy()

    # Recompute from saved sensor outputs; do not reuse a stale estimate.
    check_cancel()
    mask_path = data_dir / 'bottle_mask.png'
    if mask_source == 'yolo':
        report_phase('yolo_inference', '外部 YOLO 识别 RGB，返回同编号瓶子 Mask')
        inference = await infer_yolo_mask(data_dir, capture_id, check_cancel, yolo_weights, yolo_device)
        mask_info.update(annotator='yolo11n-seg', capture_id=capture_id,
                         white_pixel_count=inference['mask_pixels'],
                         white_pixel_ratio=inference['mask_pixels'] / (640 * 360),
                         instance_ids=[], target_prim_path=None,
                         class_name=inference['class_name'], confidence=inference['confidence'])
        (data_dir / 'bottle_mask_info.json').write_text(
            json.dumps(mask_info, indent=2), encoding='utf-8')
        mask_path = data_dir / 'yolo_mask.png'
        if omni.usd.get_context().get_stage() != stage or timeline.is_playing():
            raise RuntimeError('Scene changed or simulation started during perception')
        current_camera = np.asarray(UsdGeom.Xformable(camera_prim)
            .ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        if not np.allclose(current_camera, camera_to_world, atol=1e-6, rtol=0):
            raise RuntimeError('Camera moved during inference; recapture required')
    report_phase("estimating_pose", "根据深度、mask 和瓶子模板估计抓取位置")
    estimator = runpy.run_path(str(ESTIMATOR_SCRIPT))
    estimate = estimator["main"](
        data_dir=data_dir,
        depth_path=data_dir / "depth_noisy_m.npy",
        mask_path=mask_path,
        fixed_body_rotation=np.eye(3) if mask_source == 'yolo' else None,
    )
    if mask_source == 'yolo':
        if (not np.all(np.isfinite(estimate['estimated_body_translation_world_m']))
                or estimate['median_nearest_surface_residual_m'] > 0.015
                or estimate['p90_nearest_surface_residual_m'] > 0.060):
            raise RuntimeError('Depth/template fit failed quality gate; task not started')
        table = runpy.run_path(str(PROJECT_ROOT / 'src/bottle_sim/motion.py'))['ExistingCuboid'](stage, '/World/DeskTop')
        xy = np.asarray(estimate['estimated_body_translation_world_m'])[:2]
        if np.any(np.abs(xy - table.center[:2]) > table.size[:2] / 2 - 0.07):
            raise RuntimeError('Estimated grasp XY outside tabletop working area')
    precheck = estimate["grasp_ik_precheck"]
    if not precheck["all_ik_succeeded"]:
        raise RuntimeError("视觉估计的预抓取/抓取位姿未通过 IK，任务未启动")

    report_phase("ik_check", "抓取位姿可达性检查通过")
    check_cancel()
    body_position = estimate["estimated_body_translation_world_m"]
    task = pick["start"](bottle_position_world_m=body_position,
                          place_xy_world_m=place_xy_world_m,
                          progress_callback=progress_callback,
                          cancel_requested=cancel_requested,
                          physics_prepared=True)
    if task is None:
        raise RuntimeError("抓取任务没有启动；检查是否有上一次控制任务仍在运行")
    print("[视觉抓取] RGB-D估计瓶子抓取位置：" + json.dumps(body_position))
    if place_xy_world_m is not None:
        print("[视觉抓取] 指定放置位置 world XY：" + json.dumps(place_xy_world_m))
    await task
    result = dict(getattr(builtins, "_isaac_bottle_pick_result", {}))
    perception = {
        'capture_id': capture_id,
        'mask_source': mask_source,
        'camera': metadata,
        'mask': mask_info,
        'yolo': ({key: inference.get(key) for key in (
            'weights_sha256', 'confidence', 'class_name', 'device',
            'ultralytics_version', 'torch_version', 'inference_speed_ms',
            'bbox_xyxy_px', 'mask_pixels', 'resolution',
        )} if mask_source == 'yolo' else None),
    }
    result.update(capture_id=capture_id, mask_source=mask_source,
                  capture_dir=str(data_dir), perception=perception)
    return result


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
