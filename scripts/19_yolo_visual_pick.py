"""Isaac Script Editor entry: capture -> external YOLO -> depth pose -> pick/place.

Start while paused/stopped. Outputs are isolated by run ID under outputs/yolo_pick.
No simulator segmentation is attached and there is no fallback to a true mask.
"""
import asyncio
import builtins
from datetime import datetime
import json
from pathlib import Path
import runpy
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
TASK_KEY = '_isaac_visual_bottle_pick_task'


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
        (ROOT / 'outputs/yolo_pick_latest.json').write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')

    def progress(phase, detail):
        record.update(phase=phase, detail=detail)
        record['events'].append(dict(phase=phase, detail=detail, wall_seconds=time.monotonic()-started))
        save()
        print('[YOLO pick]', phase, detail)

    async def run():
        try:
            module = runpy.run_path(str(ROOT / 'scripts/12_visual_bottle_pick.py'))
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
