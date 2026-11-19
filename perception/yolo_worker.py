"""External YOLO worker. Reads RGB only; writes an ID/hash-bound native mask.

Called by Isaac using the training .venv, never import torch into Isaac.
The result validator has only stdlib dependencies and is usable on both sides.
"""

from pathlib import Path as _ProjectPath
import sys as _sys
PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in _sys.path:
    _sys.path.insert(0, str(PROJECT_ROOT / "src"))

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_result(capture_dir, capture_id):
    capture_dir = Path(capture_dir)
    request = json.loads((capture_dir / 'yolo_request.json').read_text(encoding='utf-8'))
    result = json.loads((capture_dir / 'yolo_result.json').read_text(encoding='utf-8'))
    if request['capture_id'] != capture_id or result.get('capture_id') != capture_id:
        raise ValueError('YOLO result belongs to a different capture')
    current_hash = file_hash(capture_dir / 'rgb.png')
    if request['rgb_sha256'] != current_hash or result.get('rgb_sha256') != current_hash:
        raise ValueError('RGB changed or stale YOLO result')
    if result.get('status') != 'succeeded':
        raise RuntimeError('YOLO inference failed: ' + result.get('error', 'unknown error'))
    if result.get('mask_file') != 'yolo_mask.png':
        raise ValueError('Unexpected YOLO output filename')
    mask_path = capture_dir / 'yolo_mask.png'
    if result.get('mask_sha256') != file_hash(mask_path):
        raise ValueError('Mask file does not match YOLO result')
    if result.get('resolution') != request['resolution']:
        raise ValueError('YOLO output resolution differs from capture')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--conf', type=float, default=0.5)
    args = parser.parse_args()
    request = json.loads((args.capture / 'yolo_request.json').read_text(encoding='utf-8'))
    result = {'capture_id': request['capture_id'], 'status': 'failed'}
    started = time.monotonic()
    try:
        import cv2
        import numpy as np
        from ultralytics import YOLO
        import ultralytics
        import torch
        # Limit CPU contention with the simulator rendering loop.
        torch.set_num_threads(2)
        rgb_path = args.capture / 'rgb.png'
        result['rgb_sha256'] = file_hash(rgb_path)
        if result['rgb_sha256'] != request['rgb_sha256']:
            raise ValueError('Input RGB hash differs from request')
        if not args.weights.is_file():
            raise FileNotFoundError(args.weights)
        pred = YOLO(str(args.weights)).predict(str(rgb_path), conf=args.conf, imgsz=640,
                retina_masks=True, device=args.device, verbose=False)[0]
        ids = [i for i, box in enumerate(pred.boxes)
               if pred.names[int(box.cls.item())] == 'bottle']
        if len(ids) != 1 or pred.masks is None:
            raise ValueError(f'Expected exactly one bottle above confidence {args.conf}, found {len(ids)}')
        index = ids[0]
        mask = pred.masks.data[index].cpu().numpy() > 0.5
        width, height = request['resolution']
        if mask.shape != (height, width):
            raise ValueError('Mask is not aligned with native RGB resolution')
        rows, cols = np.where(mask)
        if len(rows) < 1200:
            raise ValueError('Insufficient bottle mask pixels')
        if min(rows.min(), cols.min(), height-1-rows.max(), width-1-cols.max()) < 2:
            raise ValueError('Bottle mask touches image boundary')
        mask_path = args.capture / 'yolo_mask.png'
        if not cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255):
            raise RuntimeError('Failed to save mask')
        cv2.imwrite(str(args.capture / 'yolo_prediction.png'), pred.plot())
        result.update(status='succeeded', mask_file=mask_path.name,
            mask_sha256=file_hash(mask_path), resolution=[width, height],
            confidence=float(pred.boxes.conf[index]), bbox_xyxy_px=pred.boxes.xyxy[index].cpu().tolist(),
            mask_pixels=int(mask.sum()), candidates=len(ids), device=args.device,
            weights=str(args.weights), weights_sha256=file_hash(args.weights),
            ultralytics_version=ultralytics.__version__, torch_version=torch.__version__,
            inference_speed_ms=pred.speed, class_name='bottle')
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    result['worker_wall_seconds'] = time.monotonic() - started
    tmp = args.capture / 'yolo_result.tmp'
    tmp.write_text(json.dumps(result, indent=2), encoding='utf-8')
    os.replace(tmp, args.capture / 'yolo_result.json')
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'succeeded' else 1


if __name__ == '__main__':
    raise SystemExit(main())
