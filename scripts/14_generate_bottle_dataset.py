"""Generate a reproducible RGB-D bottle segmentation dataset without grasping.

Run from Isaac Sim Script Editor with the bottle scene open and the timeline
paused or stopped. Each accepted frame is saved in YOLO segmentation layout, alongside
the raw mask, depth image, and per-sample randomization metadata.
"""
import asyncio
import builtins
import json
from pathlib import Path
import time

import numpy as np
import omni.kit.app
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from PIL import Image
from pxr import Gf, Usd, UsdGeom, UsdPhysics


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "outputs" / "bottle_dataset"
TASK_KEY = "_isaac_bottle_dataset_task"

# Edit these values before running a new collection batch.
MASTER_SEED = 20261119
TARGET_SAMPLES = 50
TRAIN_FRACTION = 0.8
IMAGE_SIZE = (640, 360)  # width, height
MAX_ATTEMPTS_PER_SAMPLE = 4
MIN_MASK_PIXELS = 1200
MIN_BBOX_WIDTH_PX = 12
MIN_BBOX_HEIGHT_PX = 30
FRAME_EDGE_MARGIN_PX = 2
RANGES = {
    "bottle_dx_m": [-0.015, 0.015],
    "bottle_dy_m": [-0.015, 0.015],
    "ambient_intensity": [500.0, 900.0],
    "desk_intensity": [1300.0, 2300.0],
}

BOTTLE_WRAPPER_PATH = "/World/PracticeBottle"
BOTTLE_PRIM_PATH = "/World/PracticeBottle/bottle014"
LID_PRIM_PATH = BOTTLE_PRIM_PATH + "/lid"
CAMERA_PATH = "/World/Franka/panda_hand/WristCamera"
TABLE_PATH = "/World/DeskTop"
AMBIENT_LIGHT_PATH = "/World/AmbientLight"
DESK_LIGHT_PATH = "/World/DeskLight"
DEBUG_GROUP_PATH = "/World/Debug"
BASELINE_BOTTLE_XY_M = (0.18, -0.20)


def _get_translate_op(prim):
    for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            return op
    raise RuntimeError("PracticeBottle 外层缺少 translate xform op")


def _sample_parameters(rng, sample_index):
    return {
        "sample_index": int(sample_index),
        "bottle_dx_m": float(rng.uniform(*RANGES["bottle_dx_m"])),
        "bottle_dy_m": float(rng.uniform(*RANGES["bottle_dy_m"])),
        "ambient_intensity": float(rng.uniform(*RANGES["ambient_intensity"])),
        "desk_intensity": float(rng.uniform(*RANGES["desk_intensity"])),
        "bottle_orientation_randomized": False,
    }


def _apply_scene_parameters(stage, parameters, original_z, table_center, safe_xy):
    xy = np.asarray(BASELINE_BOTTLE_XY_M, dtype=np.float64) + np.array([
        parameters["bottle_dx_m"], parameters["bottle_dy_m"]
    ])
    if np.any(np.abs(xy - table_center) > safe_xy):
        raise ValueError("随机瓶子位置超出工作台安全范围")

    wrapper = stage.GetPrimAtPath(BOTTLE_WRAPPER_PATH)
    ambient = stage.GetPrimAtPath(AMBIENT_LIGHT_PATH)
    desk = stage.GetPrimAtPath(DESK_LIGHT_PATH)
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        _get_translate_op(wrapper).Set(Gf.Vec3d(float(xy[0]), float(xy[1]), original_z))
        ambient.GetAttribute("inputs:intensity").Set(parameters["ambient_intensity"])
        desk.GetAttribute("inputs:intensity").Set(parameters["desk_intensity"])
    parameters["bottle_xy_world_m"] = [float(xy[0]), float(xy[1]), float(original_z)]
    return parameters


def _extract_mask(segmentation, target_path):
    values = np.asarray(segmentation["data"])
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    labels = segmentation.get("info", {}).get("idToLabels", {})
    ids = [int(key) for key, value in labels.items()
           if isinstance(value, str)
           and (value == target_path or value.startswith(target_path + "/"))]
    body_ids = [int(key) for key, value in labels.items()
                if isinstance(value, str)
                and (value == target_path + "/body" or value.startswith(target_path + "/body/"))]
    lid_ids = [int(key) for key, value in labels.items()
               if isinstance(value, str)
               and (value == target_path + "/lid" or value.startswith(target_path + "/lid/"))]
    return np.isin(values, ids), ids, np.isin(values, body_ids), np.isin(values, lid_ids)


def _yolo_polygon(mask):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Isaac Python 环境缺少 cv2，无法生成 YOLO 分割多边形标签") from exc

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, max(1.0, 0.002 * perimeter), True)
    points = contour.reshape(-1, 2)
    height, width = mask.shape
    if len(points) < 3 or cv2.contourArea(contour) < 8:
        return None
    normalized = []
    for x, y in points:
        normalized.extend((min(1.0, max(0.0, float(x) / width)),
                           min(1.0, max(0.0, float(y) / height))))
    return "0 " + " ".join(f"{value:.6f}" for value in normalized)


def _camera_metadata(stage):
    camera_prim = stage.GetPrimAtPath(CAMERA_PATH)
    camera = UsdGeom.Camera(camera_prim)
    focal = float(camera.GetFocalLengthAttr().Get())
    horizontal_aperture = float(camera.GetHorizontalApertureAttr().Get())
    vertical_aperture = float(camera.GetVerticalApertureAttr().Get())
    width, height = IMAGE_SIZE
    xform = UsdGeom.XformCache().GetLocalToWorldTransform(camera_prim)
    camera_to_world = [[float(xform[i][j]) for j in range(4)] for i in range(4)]
    return {
        "camera_prim": CAMERA_PATH,
        "resolution": [width, height],
        "depth_encoding": "float32 meters; distance_to_image_plane",
        "intrinsics_px": {
            "fx": focal / horizontal_aperture * width,
            "fy": focal / vertical_aperture * height,
            "cx": width / 2.0,
            "cy": height / 2.0,
        },
        "camera_to_world_matrix_usd_row_vector_convention": camera_to_world,
    }


def _prepare_output_dirs(output_dir):
    for split in ("train", "val"):
        for folder in ("images", "labels", "masks", "depth"):
            (output_dir / folder / split).mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata").mkdir(parents=True, exist_ok=True)
    data_root = output_dir.resolve().as_posix()
    (output_dir / "data.yaml").write_text(
        f"path: '{data_root}'\ntrain: images/train\nval: images/val\nnames:\n  0: bottle\n",
        encoding="utf-8",
    )


async def generate_dataset():
    if TARGET_SAMPLES < 2:
        raise ValueError("TARGET_SAMPLES must be at least 2 so train and val both contain samples")
    if not 0.0 < TRAIN_FRACTION < 1.0:
        raise ValueError("TRAIN_FRACTION must be between 0 and 1")
    timeline = omni.timeline.get_timeline_interface()
    if timeline.is_playing():
        raise RuntimeError("请先暂停仿真，再启动数据集采集")
    await omni.kit.app.get_app().next_update_async()

    stage = omni.usd.get_context().get_stage()
    required = (BOTTLE_WRAPPER_PATH, BOTTLE_PRIM_PATH, LID_PRIM_PATH, CAMERA_PATH, TABLE_PATH,
                AMBIENT_LIGHT_PATH, DESK_LIGHT_PATH, DEBUG_GROUP_PATH)
    missing = [path for path in required if not stage.GetPrimAtPath(path)]
    if missing:
        raise RuntimeError("场景缺少必要 Prim: " + ", ".join(missing))

    wrapper = stage.GetPrimAtPath(BOTTLE_WRAPPER_PATH)
    translate_op = _get_translate_op(wrapper)
    original_translation = translate_op.Get()
    ambient_attr = stage.GetPrimAtPath(AMBIENT_LIGHT_PATH).GetAttribute("inputs:intensity")
    desk_attr = stage.GetPrimAtPath(DESK_LIGHT_PATH).GetAttribute("inputs:intensity")
    original_ambient = ambient_attr.Get()
    original_desk = desk_attr.Get()
    debug_imageable = UsdGeom.Imageable(stage.GetPrimAtPath(DEBUG_GROUP_PATH))
    debug_visibility_attr = debug_imageable.GetVisibilityAttr()
    original_debug_visibility = debug_visibility_attr.Get()
    lid_body = UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(LID_PRIM_PATH))
    lid_kinematic_attr = lid_body.GetKinematicEnabledAttr()
    original_lid_kinematic = lid_kinematic_attr.Get()

    table_data = __import__("runpy").run_path(str(SCRIPT_DIR / "04_hover_above_cube.py"))
    table = table_data["ExistingCuboid"](stage, TABLE_PATH)
    table_center = np.asarray(table.center[:2], dtype=np.float64)
    safe_xy = table.size[:2] / 2.0 - 0.12

    run_id = time.strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"batch_{MASTER_SEED}_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=False)
    _prepare_output_dirs(output_dir)
    rng = np.random.default_rng(MASTER_SEED)
    train_target = max(1, min(TARGET_SAMPLES - 1, int(TARGET_SAMPLES * TRAIN_FRACTION)))
    width, height = IMAGE_SIZE
    render_product = rep.create.render_product(CAMERA_PATH, IMAGE_SIZE)
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    mask_annotator = rep.AnnotatorRegistry.get_annotator("instance_id_segmentation")
    annotators = (rgb_annotator, depth_annotator, mask_annotator)
    for annotator in annotators:
        annotator.attach([render_product])

    manifest_path = output_dir / "samples.jsonl"
    accepted = 0
    attempts = 0
    rejected = []
    try:
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            debug_visibility_attr.Set(UsdGeom.Tokens.invisible)
            lid_kinematic_attr.Set(True)
        with manifest_path.open("w", encoding="utf-8") as manifest:
            while accepted < TARGET_SAMPLES and attempts < TARGET_SAMPLES * MAX_ATTEMPTS_PER_SAMPLE:
                sample_seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
                parameters = _sample_parameters(np.random.default_rng(sample_seed), accepted)
                parameters["seed"] = sample_seed
                parameters["attempt"] = attempts + 1
                _apply_scene_parameters(
                    stage, parameters, float(original_translation[2]), table_center, safe_xy
                )
                await omni.kit.app.get_app().next_update_async()
                await rep.orchestrator.step_async(rt_subframes=1, pause_timeline=True)
                attempts += 1

                rgb = np.asarray(rgb_annotator.get_data())[:, :, :3].astype(np.uint8)
                depth = np.asarray(depth_annotator.get_data(), dtype=np.float32)
                mask, instance_ids, body_mask, lid_mask = _extract_mask(
                    mask_annotator.get_data(), BOTTLE_PRIM_PATH
                )
                if rgb.shape != (height, width, 3) or depth.shape != (height, width):
                    rejected.append({"attempt": attempts, "reason": "unexpected image shape"})
                    continue
                if not mask.any():
                    rejected.append({"attempt": attempts, "reason": "empty bottle mask"})
                    continue
                if not body_mask.any() or not lid_mask.any():
                    rejected.append({
                        "attempt": attempts,
                        "reason": "bottle body or lid missing from mask",
                        "body_mask_pixels": int(body_mask.sum()),
                        "lid_mask_pixels": int(lid_mask.sum()),
                    })
                    continue
                ys, xs = np.where(mask)
                bbox_width = int(xs.max() - xs.min() + 1)
                bbox_height = int(ys.max() - ys.min() + 1)
                if int(mask.sum()) < MIN_MASK_PIXELS:
                    rejected.append({
                        "attempt": attempts, "reason": "mask too small",
                        "mask_pixels": int(mask.sum()),
                    })
                    continue
                if bbox_width < MIN_BBOX_WIDTH_PX or bbox_height < MIN_BBOX_HEIGHT_PX:
                    rejected.append({
                        "attempt": attempts, "reason": "bottle bbox too small",
                        "bbox_px": [bbox_width, bbox_height],
                    })
                    continue
                if (xs.min() <= FRAME_EDGE_MARGIN_PX or ys.min() <= FRAME_EDGE_MARGIN_PX
                        or xs.max() >= width - 1 - FRAME_EDGE_MARGIN_PX
                        or ys.max() >= height - 1 - FRAME_EDGE_MARGIN_PX):
                    rejected.append({
                        "attempt": attempts, "reason": "bottle mask clipped by image edge",
                        "bbox_xyxy_px": [int(xs.min()), int(ys.min()),
                                         int(xs.max()), int(ys.max())],
                    })
                    continue
                label_line = _yolo_polygon(mask)
                if label_line is None:
                    rejected.append({"attempt": attempts, "reason": "invalid mask contour"})
                    continue

                split = "train" if accepted < train_target else "val"
                sample_id = f"bottle_{accepted:06d}"
                Image.fromarray(rgb, "RGB").save(output_dir / "images" / split / f"{sample_id}.png")
                Image.fromarray(mask.astype(np.uint8) * 255, "L").save(
                    output_dir / "masks" / split / f"{sample_id}.png"
                )
                (output_dir / "labels" / split / f"{sample_id}.txt").write_text(
                    label_line + "\n", encoding="utf-8"
                )
                np.save(output_dir / "depth" / split / f"{sample_id}.npy", depth)

                camera_info = _camera_metadata(stage)
                record = {
                    "sample_id": sample_id,
                    "split": split,
                    "master_seed": MASTER_SEED,
                    "attempt_seed": parameters["seed"],
                    "parameters": parameters,
                    "mask_pixel_count": int(mask.sum()),
                    "mask_pixel_ratio": float(mask.mean()),
                    "body_mask_pixel_count": int(body_mask.sum()),
                    "lid_mask_pixel_count": int(lid_mask.sum()),
                    "instance_ids": instance_ids,
                    "mask_target_prim_prefix": BOTTLE_PRIM_PATH,
                    "camera": camera_info,
                    "files": {
                        "image": f"images/{split}/{sample_id}.png",
                        "label": f"labels/{split}/{sample_id}.txt",
                        "mask": f"masks/{split}/{sample_id}.png",
                        "depth": f"depth/{split}/{sample_id}.npy",
                    },
                }
                (output_dir / "metadata" / f"{sample_id}.json").write_text(
                    json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                manifest.flush()
                accepted += 1
                print(f"[Dataset] {accepted}/{TARGET_SAMPLES} {split} {sample_id} "
                      f"mask_pixels={int(mask.sum())}")
    finally:
        for annotator in annotators:
            annotator.detach([render_product])
        render_product.destroy()
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            translate_op.Set(original_translation)
            ambient_attr.Set(original_ambient)
            desk_attr.Set(original_desk)
            debug_visibility_attr.Set(original_debug_visibility)
            lid_kinematic_attr.Set(original_lid_kinematic)

    summary = {
        "status": "completed" if accepted == TARGET_SAMPLES else "incomplete",
        "master_seed": MASTER_SEED,
        "target_samples": TARGET_SAMPLES,
        "accepted_samples": accepted,
        "attempts": attempts,
        "train_samples": min(accepted, train_target),
        "validation_samples": max(0, accepted - train_target),
        "ranges": RANGES,
        "quality_thresholds": {
            "min_mask_pixels": MIN_MASK_PIXELS,
            "min_bbox_width_px": MIN_BBOX_WIDTH_PX,
            "min_bbox_height_px": MIN_BBOX_HEIGHT_PX,
            "frame_edge_margin_px": FRAME_EDGE_MARGIN_PX,
        },
        "rejected_attempts": rejected,
        "output_dir": str(output_dir),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[Dataset] " + json.dumps(summary, ensure_ascii=False))
    return summary


def start():
    previous = getattr(builtins, TASK_KEY, None)
    if previous is not None and not previous.done():
        raise RuntimeError("数据集采集任务已在运行")
    task = asyncio.ensure_future(generate_dataset())
    setattr(builtins, TASK_KEY, task)
    return task


if __name__ == "__main__":
    start()
