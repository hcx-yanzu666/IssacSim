"""Offline verification of an existing segmentation checkpoint; no simulator needed.

Run with the project's training .venv. This evaluates the existing validation
split, not an independent test set, and does not modify the source dataset.
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

import cv2
import numpy as np
from ultralytics import YOLO


def iou(a, b):
    return float(np.logical_and(a, b).sum() / max(1, np.logical_or(a, b).sum()))


def main():
    root = PROJECT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=root / 'outputs/bottle_dataset/batch_20261119_20261118_165034')
    parser.add_argument('--weights', type=Path, default=root / 'outputs/yolo_train/multiview500/weights/best.pt')
    parser.add_argument('--device', default='0')
    args = parser.parse_args()
    out = root / 'outputs/yolo_verify' / datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True)
    audit, hashes, images = [], {}, {}
    for split in ('train', 'val'):
        images[split] = sorted((args.dataset / 'images' / split).glob('*.png'))
        assert images[split], f'Empty split: {split}'
        hashes[split] = set()
        for path in images[split]:
            rgb = cv2.imread(str(path))
            assert rgb is not None, path
            h, w = rgb.shape[:2]
            hashes[split].add(hashlib.sha256(rgb.tobytes()).hexdigest())
            mask = cv2.imread(str(args.dataset / 'masks' / split / path.name), 0)
            assert mask is not None and mask.shape == (h, w) and mask.any(), path
            lines = (args.dataset / 'labels' / split / (path.stem + '.txt')).read_text().splitlines()
            assert len(lines) == 1, f'Expected one bottle: {path}'
            parts = np.array([float(v) for v in lines[0].split()])
            assert parts[0] == 0 and len(parts) >= 7 and len(parts) % 2 == 1
            assert np.isfinite(parts).all() and ((parts[1:] >= 0) & (parts[1:] <= 1)).all()
            polygon = np.rint(parts[1:].reshape(-1, 2) * [w, h]).astype(np.int32)
            raster = np.zeros((h, w), np.uint8)
            cv2.fillPoly(raster, [polygon], 1)
            audit.append({'image': str(path.relative_to(args.dataset)), 'polygon_mask_iou': iou(raster > 0, mask > 0)})
    overlap = len(hashes['train'] & hashes['val'])
    assert overlap == 0, 'Exact train/validation image duplicates found'
    model = YOLO(str(args.weights))
    metrics = model.val(data=str(args.dataset / 'data.yaml'), split='val', imgsz=640,
                        batch=1, device=args.device, workers=0, plots=True,
                        project=str(out), name='validation', exist_ok=True)
    rows, tiles = [], []
    for path in images['val']:
        result = model.predict(source=str(path), imgsz=640, conf=0.25, device=args.device,
                               retina_masks=True, verbose=False)[0]
        gt = cv2.imread(str(args.dataset / 'masks/val' / path.name), 0) > 0
        row = {'image': path.name, 'detections': len(result.boxes), 'confidence': None, 'mask_iou': 0.0}
        if len(result.boxes):
            # Select by confidence, never by ground-truth overlap.
            index = int(result.boxes.conf.argmax())
            row['confidence'] = float(result.boxes.conf[index])
            if result.masks is not None:
                pred = result.masks.data[index].cpu().numpy() > 0.5
                assert pred.shape == gt.shape
                row['mask_iou'] = iou(pred, gt)
        rows.append(row)
        annotated = result.plot()
        contours, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, contours, -1, (0, 255, 0), 1)
        cv2.imwrite(str(out / path.name), annotated)
        tile = cv2.copyMakeBorder(cv2.resize(annotated, (384, 288)), 32, 0, 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(tile, f'{path.stem} IoU={row["mask_iou"]:.3f}', (5, 21), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        tiles.append(tile)
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(out / 'prediction_contact_sheet.jpg'), np.vstack([np.hstack(tiles[i:i+2]) for i in range(0, len(tiles), 2)]))
    report = {
        'scope': 'Existing validation split; same asset, scene and narrow randomization ranges. Not independent test or real-world validation.',
        'weights': str(args.weights), 'weights_sha256': hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        'dataset': str(args.dataset), 'counts': {s: len(p) for s, p in images.items()},
        'exact_cross_split_duplicates': overlap,
        'label_mask_iou_min': min(a['polygon_mask_iou'] for a in audit),
        'label_mask_iou_mean': float(np.mean([a['polygon_mask_iou'] for a in audit])),
        'metrics': {k: float(v) for k, v in metrics.results_dict.items()},
        'prediction_conf_threshold': 0.25,
        'images_with_detection': sum(r['detections'] > 0 for r in rows),
        'extra_detections': sum(max(0, r['detections'] - 1) for r in rows),
        'mean_top_confidence_mask_iou': float(np.mean([r['mask_iou'] for r in rows])),
        'per_image': rows, 'label_audit': audit,
    }
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('per_image', 'label_audit')}, indent=2))
    print('OUTPUT_DIR=' + str(out))


if __name__ == '__main__':
    main()
