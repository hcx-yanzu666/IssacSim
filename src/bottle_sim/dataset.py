"""Generate a reproducible RGB-D bottle segmentation dataset without grasping.

Run from Isaac Sim Script Editor with the bottle scene open and the timeline
paused or stopped. Each accepted frame is saved in YOLO segmentation layout, alongside
the raw mask, depth image, and per-sample randomization metadata.
"""

from pathlib import Path as _ProjectPath
import sys as _sys
PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in _sys.path:
    _sys.path.insert(0, str(PROJECT_ROOT / "src"))

import asyncio
import builtins
import json
import runpy
import uuid
from pathlib import Path
import time

import numpy as np
import omni.kit.app
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from PIL import Image
from pxr import Gf, Usd, UsdGeom, UsdPhysics


OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "bottle_dataset"
TASK_KEY = "_isaac_bottle_dataset_task"
SUPPORT = runpy.run_path(str(PROJECT_ROOT / 'src/bottle_sim/randomization.py'))

# Randomization and batch options are in configs/dataset_randomization.json.
IMAGE_SIZE = (640, 360)
MIN_MASK_PIXELS = 1200
MIN_BBOX_WIDTH_PX = 12
MIN_BBOX_HEIGHT_PX = 30
FRAME_EDGE_MARGIN_PX = 2


BOTTLE_WRAPPER_PATH = "/World/PracticeBottle"
BOTTLE_PRIM_PATH = "/World/PracticeBottle/bottle014"
LID_PRIM_PATH = BOTTLE_PRIM_PATH + "/lid"
CAMERA_PATH = "/World/DatasetCaptureCamera"
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


def ensure_capture_camera(stage, config, reset_pose=False):
    """Independent synthetic-data camera; never edit the wrist sensor."""
    if stage.GetPrimAtPath(CAMERA_PATH) and not reset_pose:
        return UsdGeom.Camera(stage.GetPrimAtPath(CAMERA_PATH))
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        camera = UsdGeom.Camera.Define(stage, CAMERA_PATH)
        camera.CreateFocalLengthAttr(18.0)
        camera.CreateHorizontalApertureAttr(24.0)
        camera.CreateVerticalApertureAttr(13.5)
        camera.CreateClippingRangeAttr(Gf.Vec2f(.01,100.))
        wrapper = stage.GetPrimAtPath(BOTTLE_WRAPPER_PATH)
        z = float(_get_translate_op(wrapper).Get()[2]) if wrapper else .75
        target = [*BASELINE_BOTTLE_XY_M, z+config['camera_target_height_m']]
        parameters = {key:float(np.mean(config['ranges'][key])) for key in
                      ('camera_azimuth_deg','camera_elevation_deg','camera_distance_m')}
        position = SUPPORT['camera_position'](parameters,target)
        view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*position),Gf.Vec3d(*target),Gf.Vec3d(0,0,1))
        UsdGeom.Xformable(camera).MakeMatrixXform().Set(view.GetInverse())
    return camera


def _apply_scene_parameters(stage, parameters, original_z, table_center, safe_xy, config):
    xy = np.asarray(BASELINE_BOTTLE_XY_M) + np.array([
        parameters['bottle_dx_m'], parameters['bottle_dy_m']])
    if np.any(np.abs(xy-table_center) > safe_xy):
        raise ValueError('Randomized bottle position outside tabletop working area')
    target = [*BASELINE_BOTTLE_XY_M, original_z+config['light_target_height_m']]
    position = SUPPORT['light_position'](parameters, target)
    camera_target = [*xy, original_z+config['camera_target_height_m']]
    camera_position = SUPPORT['camera_position'](parameters,camera_target)
    camera_view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*camera_position),Gf.Vec3d(*camera_target),Gf.Vec3d(0,0,1))
    desk = stage.GetPrimAtPath(DESK_LIGHT_PATH)
    parent_world = UsdGeom.Xformable(desk.GetParent()).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    local_position = parent_world.GetInverse().Transform(Gf.Vec3d(*position))
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        _get_translate_op(stage.GetPrimAtPath(BOTTLE_WRAPPER_PATH)).Set(Gf.Vec3d(*xy, original_z))
        _get_translate_op(desk).Set(local_position)
        stage.GetPrimAtPath(AMBIENT_LIGHT_PATH).GetAttribute('inputs:intensity').Set(parameters['ambient_intensity'])
        desk.GetAttribute('inputs:intensity').Set(parameters['desk_intensity'])
        camera_prim = stage.GetPrimAtPath(CAMERA_PATH)
        camera_parent = UsdGeom.Xformable(camera_prim.GetParent()).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        UsdGeom.Xformable(camera_prim).MakeMatrixXform().Set(camera_view.GetInverse()*camera_parent.GetInverse())
    parameters.update(bottle_xy_world_m=xy.tolist(), bottle_translation_world_m=[*xy, original_z],
                      light_target_world_m=target, light_position_world_m=position,
                      light_type=desk.GetTypeName(), light_direction_method='sphere_light_position_on_hemisphere')
    parameters.update(camera_mode='independent_orbit',camera_position_world_m=camera_position,
                      camera_target_world_m=camera_target)
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
        "camera_mode": "independent_orbit",
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


async def generate_dataset(config=None, progress_callback=None, cancel_requested=None):
    config = SUPPORT['load_config'](config)
    target_samples = config['target_samples']
    timeline = omni.timeline.get_timeline_interface()
    if timeline.is_playing():
        raise RuntimeError('Pause the simulation before dataset capture')
    for key in ('_isaac_visual_bottle_pick_task', '_isaac_lesson09_task'):
        task = getattr(builtins, key, None)
        if task is not None and not task.done():
            raise RuntimeError('A robot task is still running')
    # Reset the arm and bottle from their last post-grasp poses before capturing.
    timeline.stop()
    for _ in range(3):
        await omni.kit.app.get_app().next_update_async()
    stage = omni.usd.get_context().get_stage()
    ensure_capture_camera(stage, config)
    required = (BOTTLE_WRAPPER_PATH, BOTTLE_PRIM_PATH, LID_PRIM_PATH,
                BOTTLE_PRIM_PATH+'/body', CAMERA_PATH, TABLE_PATH, AMBIENT_LIGHT_PATH, DESK_LIGHT_PATH)
    missing = [path for path in required if not stage.GetPrimAtPath(path)]
    if missing:
        raise RuntimeError('Missing prims: '+', '.join(missing))
    desk = stage.GetPrimAtPath(DESK_LIGHT_PATH)
    if desk.GetTypeName() != 'SphereLight':
        raise ValueError('This lighting preset requires a SphereLight')
    translate = _get_translate_op(stage.GetPrimAtPath(BOTTLE_WRAPPER_PATH))
    desk_translate = _get_translate_op(desk)
    ambient = stage.GetPrimAtPath(AMBIENT_LIGHT_PATH).GetAttribute('inputs:intensity')
    intensity = desk.GetAttribute('inputs:intensity')
    attrs = [translate.GetAttr(), desk_translate.GetAttr(), ambient, intensity]
    for path in (BOTTLE_PRIM_PATH+'/body', LID_PRIM_PATH):
        attrs.append(UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(path)).GetKinematicEnabledAttr())
    debug = stage.GetPrimAtPath(DEBUG_GROUP_PATH)
    debug_attr = UsdGeom.Imageable(debug).GetVisibilityAttr() if debug else None
    if debug_attr:
        attrs.append(debug_attr)
    environment_light = stage.GetPrimAtPath('/Environment/defaultLight')
    environment_visibility = (UsdGeom.Imageable(environment_light).GetVisibilityAttr()
                              if environment_light and config['mute_environment_light'] else None)
    if environment_visibility:
        attrs.append(environment_visibility)
    attrs.append(stage.GetPrimAtPath(CAMERA_PATH).GetAttribute('xformOp:transform'))
    originals = [(attr, attr.Get()) for attr in attrs]
    original_z = float(translate.Get()[2])
    table_data = runpy.run_path(str(PROJECT_ROOT / 'src/bottle_sim/motion.py'))
    table = table_data['ExistingCuboid'](stage, TABLE_PATH)
    table_center, safe_xy = np.asarray(table.center[:2]), table.size[:2]/2-.12
    run_id = time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
    output_dir = OUTPUT_ROOT / f"batch_{config['master_seed']}_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=False)
    _prepare_output_dirs(output_dir)
    (output_dir/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    seed_rng = np.random.default_rng(config['master_seed'])
    train_target = 1 if target_samples == 1 else max(1,min(target_samples-1,int(target_samples*config['train_fraction'])))
    records, rejected, accepted, attempts = [], [], 0, 0
    render_product, annotators = None, []
    status, error = 'running', None
    width, height = IMAGE_SIZE

    def notify(phase, parameters=None, detail=''):
        if progress_callback:
            progress_callback(dict(phase=phase, accepted=accepted, target=target_samples,
                attempts=attempts, rejected=len(rejected), parameters=parameters,
                output_dir=str(output_dir), detail=detail))

    def check_cancel():
        if cancel_requested and cancel_requested():
            raise asyncio.CancelledError()
        if omni.usd.get_context().get_stage() != stage:
            raise RuntimeError('Stage changed during capture')
        if timeline.is_playing():
            raise RuntimeError('Playback started during capture; pause and retry')

    try:
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            for attr in attrs[4:6]:
                attr.Set(True)
            if debug_attr:
                debug_attr.Set(UsdGeom.Tokens.invisible)
            if environment_visibility:
                environment_visibility.Set(UsdGeom.Tokens.invisible)
        render_product = rep.create.render_product(CAMERA_PATH, IMAGE_SIZE)
        for name in ('rgb','distance_to_image_plane','instance_id_segmentation'):
            annotator = rep.AnnotatorRegistry.get_annotator(name)
            annotator.attach([render_product])
            annotators.append(annotator)
        notify('started', detail='Preparing synchronized RGB-D and instance labels')
        with (output_dir/'samples.jsonl').open('w',encoding='utf-8') as manifest:
            while accepted < target_samples and attempts < target_samples*config['max_attempts_per_sample']:
                check_cancel()
                attempts += 1
                seed = int(seed_rng.integers(0,np.iinfo(np.uint32).max,dtype=np.uint32))
                params = SUPPORT['sample_parameters'](seed,accepted,config)
                params['attempt'] = attempts
                try:
                    _apply_scene_parameters(stage,params,original_z,table_center,safe_xy,config)
                except ValueError as exc:
                    rejected.append(dict(attempt=attempts,seed=seed,parameters=params,reason=str(exc)))
                    notify('rejected',params,str(exc))
                    continue
                notify('rendering',params,'Applying sampled scene parameters')
                await omni.kit.app.get_app().next_update_async()
                await rep.orchestrator.step_async(rt_subframes=config['render_subframes'],
                                                 delta_time=0.0, pause_timeline=True)
                check_cancel()
                rgb = np.asarray(annotators[0].get_data())[:,:,:3].astype(np.uint8).copy()
                depth = np.asarray(annotators[1].get_data(),dtype=np.float32).copy()
                mask, ids, body_mask, lid_mask = _extract_mask(annotators[2].get_data(),BOTTLE_PRIM_PATH)
                reason = None
                if rgb.shape != (height,width,3) or depth.shape != (height,width) or mask.shape != (height,width):
                    reason = 'unexpected image shape'
                elif not mask.any() or not body_mask.any() or not lid_mask.any():
                    reason = 'bottle body or lid missing from mask'
                else:
                    ys,xs = np.where(mask)
                    if mask.sum() < MIN_MASK_PIXELS or xs.max()-xs.min()+1 < MIN_BBOX_WIDTH_PX or ys.max()-ys.min()+1 < MIN_BBOX_HEIGHT_PX:
                        reason = 'bottle mask/bbox too small'
                    elif xs.min() <= FRAME_EDGE_MARGIN_PX or ys.min() <= FRAME_EDGE_MARGIN_PX or xs.max() >= width-1-FRAME_EDGE_MARGIN_PX or ys.max() >= height-1-FRAME_EDGE_MARGIN_PX:
                        reason = 'bottle mask clipped by image edge'
                    elif (np.isfinite(depth[mask]) & (depth[mask] > 0)).mean() < .95:
                        reason = 'insufficient valid bottle depth'
                label = None if reason else _yolo_polygon(mask)
                if reason is None and label is None:
                    reason = 'invalid mask contour'
                if reason:
                    rejected.append(dict(attempt=attempts,seed=seed,parameters=params,reason=reason))
                    notify('rejected',params,reason)
                    continue
                split = 'train' if accepted < train_target else 'val'
                sample_id = f'bottle_{accepted:06d}'
                files = {key:f'{folder}/{split}/{sample_id}{ext}' for key,folder,ext in
                         [('image','images','.png'),('label','labels','.txt'),('mask','masks','.png'),('depth','depth','.npy')]}
                Image.fromarray(rgb).save(output_dir/files['image'])
                Image.fromarray(mask.astype(np.uint8)*255).save(output_dir/files['mask'])
                (output_dir/files['label']).write_text(label+'\n',encoding='utf-8')
                np.save(output_dir/files['depth'],depth)
                record = dict(sample_id=sample_id,split=split,master_seed=config['master_seed'],
                    attempt_seed=seed,parameters=params,mask_pixel_count=int(mask.sum()),
                    mask_pixel_ratio=float(mask.mean()),body_mask_pixel_count=int(body_mask.sum()),
                    lid_mask_pixel_count=int(lid_mask.sum()),instance_ids=ids,
                    mask_target_prim_prefix=BOTTLE_PRIM_PATH,camera=_camera_metadata(stage),files=files,
                    bottle_mean_rgb=rgb[mask].mean(axis=0).tolist())
                files['preview'] = SUPPORT['save_preview'](output_dir,record,rgb,mask)
                (output_dir/'metadata'/f'{sample_id}.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
                manifest.write(json.dumps(record)+'\n'); manifest.flush()
                records.append(record)
                accepted += 1
                notify('saved',params,f'Saved {sample_id} ({split})')
                deadline = time.monotonic()+config['demo_hold_seconds']
                while time.monotonic() < deadline:
                    check_cancel()
                    await omni.kit.app.get_app().next_update_async()
        status = 'completed' if accepted == target_samples else 'incomplete'
    except asyncio.CancelledError:
        status, error = 'cancelled', 'Capture cancelled; accepted samples preserved'
    except Exception as exc:
        status, error = 'failed', f'{type(exc).__name__}: {exc}'
        raise
    finally:
        for annotator in annotators:
            annotator.detach([render_product])
        if render_product is not None:
            render_product.destroy()
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            for attr,value in originals:
                attr.Clear() if value is None else attr.Set(value)
        SUPPORT['build_gallery'](output_dir,records)
        summary = dict(status=status,error=error,master_seed=config['master_seed'],
            target_samples=target_samples,accepted_samples=accepted,attempts=attempts,
            train_samples=min(accepted,train_target),validation_samples=max(0,accepted-train_target),
            ranges=config['ranges'],config=config,rejected_attempts=rejected,
            quality_thresholds=dict(min_mask_pixels=MIN_MASK_PIXELS,min_bbox_width_px=MIN_BBOX_WIDTH_PX,
                min_bbox_height_px=MIN_BBOX_HEIGHT_PX,frame_edge_margin_px=FRAME_EDGE_MARGIN_PX,
                min_valid_depth_fraction=.95),
            output_dir=str(output_dir),created_at=time.strftime('%Y-%m-%d %H:%M:%S'),
            split_note='Same collection distribution; separate scene-condition test set still required',
            scene_parameters_restored=True)
        (output_dir/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        (OUTPUT_ROOT/'latest.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        notify(status,detail=error or 'Capture finished; original scene parameters restored')
        print('[Dataset] '+json.dumps(summary))
    return summary


def start(config=None, progress_callback=None, cancel_requested=None):
    previous = getattr(builtins,TASK_KEY,None)
    if previous is not None and not previous.done():
        raise RuntimeError('Dataset capture already running')
    task = asyncio.ensure_future(generate_dataset(config,progress_callback,cancel_requested))
    setattr(builtins,TASK_KEY,task)
    task.add_done_callback(lambda t: print('[Dataset task]', 'cancelled' if t.cancelled() else repr(t.exception())))
    return task


if __name__ == '__main__':
    start()
