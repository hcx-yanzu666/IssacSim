"""Isaac Script Editor entry: capture -> external YOLO -> depth pose -> pick/place.

Start while paused/stopped. Outputs are isolated by run ID under outputs/yolo_pick.
No simulator segmentation is attached and there is no fallback to a true mask.
"""

from pathlib import Path as _ProjectPath
import sys as _sys
PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in _sys.path:
    _sys.path.insert(0, str(PROJECT_ROOT / "src"))

import asyncio
import builtins
from datetime import datetime
import json
from pathlib import Path
import runpy
import time
import uuid

ROOT = PROJECT_ROOT
TASK_KEY = '_isaac_visual_bottle_pick_task'

SUCCESS_DEBUG_FILES = (
    'depth_m.npy', 'depth_noisy_m.npy', 'bottle_mask.png',
    'camera_info.json', 'bottle_mask_info.json', 'yolo_prediction.png',
    'yolo_request.json', 'yolo_result.json', 'yolo_worker.log',
    'bottle_pose_estimate_overlay.png',
)


def prune_success_debug_files(run_dir):
    for name in SUCCESS_DEBUG_FILES:
        path = run_dir / name
        if path.is_file():
            path.unlink()


def start(place_xy_world_m=None, device='cpu', weights=None):
    for key in (TASK_KEY, '_isaac_lesson09_task'):
        previous = getattr(builtins, key, None)
        if previous is not None and not previous.done():
            raise RuntimeError('A pick task is already running')
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
    out = ROOT / 'outputs/yolo_pick' / run_id
    out.mkdir(parents=True, exist_ok=False)
    record = {'run_id': run_id, 'status': 'Running', 'phase': 'Queued', 'mask_source': 'yolo',
              'output_dir': str(out), 'events': [], 'started_at': datetime.now().isoformat()}
    started = time.monotonic()

    def save():
        tmp = out / 'result.tmp'
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.replace(out / 'result.json')

    def progress(phase, detail):
        record.update(phase=phase, detail=detail)
        record['events'].append(dict(phase=phase, detail=detail, wall_seconds=time.monotonic()-started))
        save()
        print('[YOLO pick]', phase, detail)

    async def run():
        try:
            module = runpy.run_path(str(PROJECT_ROOT / 'src/bottle_sim/visual_pick.py'))
            result = await module['start_visual_pick'](place_xy_world_m=place_xy_world_m,
                progress_callback=progress, data_dir=out, mask_source='yolo',
                yolo_device=device, yolo_weights=weights)
            record.update(status=result.get('status', 'Unknown'), phase=result.get('phase', ''),
                          detail=result.get('detail', ''), grasp_result=result)
            return result
        except asyncio.CancelledError:
            record.update(status='Cancelled', detail='YOLO pick task cancelled')
            raise
        except Exception as exc:
            record.update(status='Failed', detail=f'{type(exc).__name__}: {exc}')
            print('[YOLO pick] FAILED:', record['detail'])
            raise
        finally:
            record['wall_seconds'] = time.monotonic() - started
            save()
            if record.get('status') == 'Succeeded':
                prune_success_debug_files(out)
            try:
                report = runpy.run_path(str(ROOT / 'tools' / 'build_evaluation_report.py'))
                report_path, included, excluded = report['build_report']()
                print(f'[YOLO pick] HTML report: {report_path} '
                      f'({included} runs included, {excluded} excluded)')
            except Exception as report_error:
                print(f'[YOLO pick] HTML report generation failed: '
                      f'{type(report_error).__name__}: {report_error}')

    save()
    task = asyncio.ensure_future(run())
    setattr(builtins, TASK_KEY, task)
    # Consume/report exceptions also when launched from the Script Editor.
    task.add_done_callback(lambda t: print('[YOLO pick] finished:', record['status'],
                            '' if t.cancelled() else repr(t.exception())))
    print('[YOLO pick] output:', out)
    return task


if __name__ == '__main__':
    start()
