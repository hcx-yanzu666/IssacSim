"""Compare YOLO and simulator masks on saved RGB-D, without moving the robot.

Run main with the training .venv. export_template(stage, path) may be called
once inside Isaac to export mesh-local geometry; it does not change the stage.
Translation is fitted, orientation remains an explicit upright setup prior.
"""

from pathlib import Path as _ProjectPath
import sys as _sys
PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in _sys.path:
    _sys.path.insert(0, str(PROJECT_ROOT / "src"))

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
import runpy

import numpy as np

ROOT = PROJECT_ROOT


def export_template(stage, output):
    """Export body-local mesh geometry ONLY, without current world pose."""
    from pxr import Gf, Usd, UsdGeom
    body_path = '/World/PracticeBottle/bottle014/body'
    mesh_path = body_path + '/Visuals'
    body = stage.GetPrimAtPath(body_path)
    mesh = stage.GetPrimAtPath(mesh_path)
    if not body or not mesh or not mesh.IsA(UsdGeom.Mesh):
        raise ValueError('Bottle body/visual mesh missing')
    body_world = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    mesh_world = UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    mesh_to_body = mesh_world * body_world.GetInverse()
    vertices = np.asarray([
        mesh_to_body.Transform(Gf.Vec3d(*map(float, v)))
        for v in UsdGeom.Mesh(mesh).GetPointsAttr().Get()
    ], dtype=np.float64)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, model_body=vertices)
    output.with_suffix('.json').write_text(json.dumps({
        'body_prim': body_path, 'mesh_prim': mesh_path,
        'vertices': len(vertices), 'units': 'meters',
        'contents': 'Body-local visual mesh vertices, no world translation or orientation',
        'stage_source': stage.GetRootLayer().identifier,
    }, indent=2), encoding='utf-8')
    return {'template': str(output), 'vertices': len(vertices)}


def write_ply(path, points):
    with path.open('w', encoding='ascii') as f:
        f.write(f'ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n')
        np.savetxt(f, points, fmt='%.8f')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import cv2
    from ultralytics import YOLO

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, default=ROOT / 'outputs/wrist_rgbd_01')
    parser.add_argument('--template', type=Path, default=ROOT / 'outputs/perception_templates/bottle014_body.npz')
    parser.add_argument('--weights', type=Path, default=ROOT / 'outputs/yolo_train/baseline50_run2/weights/best.pt')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', default='0')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--max-mask-induced-xy-mm', type=float, default=5.0)
    args = parser.parse_args()
    out = args.output or ROOT / 'outputs/yolo_pose_compare' / datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    report = {'status': 'started', 'moves_robot': False, 'capture': str(args.capture)}
    try:
        capture = args.capture
        rgb_path, depth_path = capture / 'rgb.png', capture / 'depth_m.npy'
        rgb = cv2.imread(str(rgb_path))
        depth = np.load(depth_path).astype(np.float64)
        camera = json.loads((capture / 'camera_info.json').read_text(encoding='utf-8'))
        transform = np.array(camera['camera_to_world_matrix_usd_row_vector_convention'])
        k = camera['intrinsics_px']
        if rgb is None or rgb.shape[:2] != depth.shape:
            raise ValueError('RGB and depth must be aligned')
        if 'distance_to_image_plane' not in camera['depth_encoding']:
            raise ValueError('Expected axial depth, not distance_to_camera')
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError('Invalid camera transform')
        estimator = runpy.run_path(str(PROJECT_ROOT / 'src/bottle_sim/pose_estimation.py'))
        model_body = np.load(args.template)['model_body']
        # Fixed upright orientation prior, never read current post-grasp pose.
        rotation = np.eye(3)
        forward = -transform[2, :3].copy()
        forward /= np.linalg.norm(forward)
        prediction = YOLO(str(args.weights)).predict(str(rgb_path), conf=args.conf,
                            imgsz=640, retina_masks=True, device=args.device, verbose=False)[0]
        candidates = [i for i, b in enumerate(prediction.boxes)
                      if prediction.names[int(b.cls.item())] == 'bottle']
        if not candidates or prediction.masks is None:
            raise ValueError('No bottle mask above confidence threshold')
        selected = max(candidates, key=lambda i: float(prediction.boxes.conf[i]))
        yolo_mask = prediction.masks.data[selected].cpu().numpy() > 0.5
        if yolo_mask.shape != depth.shape:
            raise ValueError('YOLO mask must be mapped to native RGB-D resolution')
        # Simulator mask and ground truth never participate in candidate selection.
        sim_mask = cv2.imread(str(capture / 'bottle_mask.png'), 0)
        if sim_mask is None or sim_mask.shape != depth.shape:
            raise ValueError('Missing/alignment error in reference mask')
        sim_mask = sim_mask > 0
        results = {}
        estimates = {}
        for name, mask in [('yolo', yolo_mask), ('sim', sim_mask)]:
            points = estimator['camera_depth_to_world'](mask, depth, k, transform)
            if len(points) < 50:
                raise ValueError(f'{name}: insufficient valid depth points ({len(points)})')
            estimate, residuals = estimator['fit_translation_icp'](points, model_body, rotation, forward)
            estimates[name] = estimate
            results[name] = {
                'estimated_body_translation_world_m': estimate.tolist(),
                'mask_pixels': int(mask.sum()), 'observed_points': len(points),
                'median_surface_residual_mm': float(np.median(residuals) * 1000),
                'p90_surface_residual_mm': float(np.percentile(residuals, 90) * 1000),
            }
            cv2.imwrite(str(out / (name + '_mask.png')), mask.astype(np.uint8) * 255)
            write_ply(out / (name + '_observed_world.ply'), points)
            write_ply(out / (name + '_fitted_template_world.ply'), model_body @ rotation + estimate)
        delta = estimates['yolo'] - estimates['sim']
        delta_xy = float(np.linalg.norm(delta[:2]) * 1000)
        # Saved estimator report contains actual body-origin GT. camera_info's
        # legacy "body" field actually records the bottle parent, NOT the body.
        previous_path = capture / 'bottle_pose_estimate.json'
        if previous_path.exists():
            previous = json.loads(previous_path.read_text(encoding='utf-8'))
            gt = np.array(previous['ground_truth_body_translation_world_m_evaluation_only'])
            for name in results:
                error = estimates[name] - gt
                results[name]['evaluation_error_xyz_mm'] = (error * 1000).tolist()
                results[name]['evaluation_error_xy_mm'] = float(np.linalg.norm(error[:2]) * 1000)
                results[name]['evaluation_error_3d_mm'] = float(np.linalg.norm(error) * 1000)
            report['evaluation_ground_truth'] = {
                'body_translation_world_m': gt.tolist(), 'source': str(previous_path),
                'note': 'Historical same-capture estimator report; evaluation only. Current live scene pose is not used.',
            }
            report['sim_reestimate_vs_historical_mm'] = float(np.linalg.norm(
                estimates['sim'] - np.array(previous['estimated_body_translation_world_m'])) * 1000)
        report.update({
            'status': 'completed', 'scope': 'Offline mask substitution and translation comparison; no IK or grasp execution',
            'inputs_sha256': {str(p): sha(p) for p in [rgb_path, depth_path, capture / 'camera_info.json', capture / 'bottle_mask.png', args.weights, args.template]},
            'orientation_prior': 'Identity body-to-world rotation (upright fixed setup); not estimated. Historical tiny tilt may differ.',
            'estimator': 'Unmodified NumPy backprojection and robust translation ICP from script 11; stride=4, erosion=3px',
            'model_vertices': len(model_body), 'confidence': float(prediction.boxes.conf[selected]),
            'bottle_candidates': len(candidates), 'candidate_selection': 'Highest confidence bottle; no reference mask or ground truth used',
            'mask_iou': float((yolo_mask & sim_mask).sum() / max(1, (yolo_mask | sim_mask).sum())),
            'results': results, 'yolo_minus_sim_xyz_mm': (delta * 1000).tolist(),
            'mask_induced_xy_difference_mm': delta_xy,
            'mask_induced_3d_difference_mm': float(np.linalg.norm(delta) * 1000),
            'mask_substitution_gate': {'threshold_xy_mm': args.max_mask_induced_xy_mm,
                'passed': delta_xy <= args.max_mask_induced_xy_mm,
                'meaning': 'Engineering comparison threshold only, not proof of absolute pose accuracy or safe grasp'},
            'limitations': ['One saved frame only', 'Known mesh and fixed orientation prior',
                'Body visual mesh template excludes lid while masks include lid',
                'No independent real-world evaluation', 'Legacy camera metadata parent-origin field ignored'],
        })
        overlay = rgb.copy()
        for name, color in [('sim', (0, 255, 0)), ('yolo', (0, 0, 255))]:
            mask = sim_mask if name == 'sim' else yolo_mask
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 1)
            pc = np.r_[estimates[name], 1.] @ np.linalg.inv(transform)
            if pc[2] < 0:
                u = int(round(k['fx'] * pc[0] / -pc[2] + k['cx']))
                v = int(round(k['cy'] - k['fy'] * pc[1] / -pc[2]))
                cv2.drawMarker(overlay, (u, v), color, cv2.MARKER_CROSS, 15, 2)
        overlay = cv2.copyMakeBorder(overlay, 50, 0, 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(overlay, 'Green: simulator mask/origin   Red: YOLO mask/origin', (8, 19), cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 1)
        cv2.putText(overlay, f'Mask IoU={report["mask_iou"]:.3f}  Pose XY difference={delta_xy:.2f} mm', (8, 40), cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 1)
        cv2.imwrite(str(out / 'comparison_overlay.png'), overlay)
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))
        print('OUTPUT_DIR=' + str(out))


if __name__ == '__main__':
    main()
