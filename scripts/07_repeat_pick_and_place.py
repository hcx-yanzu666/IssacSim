"""第7课：同一配置重复实验，保留每轮报告并汇总。

教学主线：run_one恢复起点 → 创建新控制器 → 执行 → 清理回调 → 保存。
运动完全复用第六课。暂停/停止会结束整批，不会自动恢复播放。
TODO（你的实验）：将RUN_COUNT改为3，检查是否只有3份run报告。
"""
import asyncio
import builtins
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import runpy
import time
import uuid

RUN_COUNT = 5
WALL_TIMEOUT_SECONDS = 180.0  # 防止仿真时间不推进时永久等待。
TASK_KEY = '_isaac_lesson07_task'
RESULT_KEY = '_isaac_lesson07_result'
PROGRESS_KEY = '_isaac_lesson07_progress'
OUTPUT_ROOT = Path(__file__).resolve().parent.parent / 'outputs' / 'batches'


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def numeric_stats(values):
    valid = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return {'count': len(valid), 'mean': sum(valid)/len(valid), 'min': min(valid), 'max': max(valid)} if valid else None


def summarize(results, requested, status, batch_id):
    """只对成功轮次计算耗时/误差；取消轮次不参与成功率分母。"""
    counts = Counter(r['status'] for r in results)
    evaluated = sum(counts[k] for k in ('Succeeded', 'Failed', 'TimedOut'))
    successes = [r for r in results if r['status'] == 'Succeeded']
    return {
        'batch_id': batch_id, 'status': status, 'requested_runs': requested,
        'recorded_runs': len(results), 'evaluated_runs': evaluated,
        'status_counts': dict(counts),
        'success_rate': len(successes)/evaluated if evaluated else None,
        'successful_elapsed_seconds': numeric_stats([r.get('elapsed_seconds') for r in successes]),
        'successful_xy_error_m': numeric_stats([r.get('placement', {}).get('xy_error_m') for r in successes]),
        'failure_phases': dict(Counter(r.get('phase', 'Unknown') for r in results if r['status'] in ('Failed', 'TimedOut'))),
        'runs': [{'run_index': r['run_index'], 'status': r['status'],
                  'phase': r.get('phase'), 'report': r['report_file']} for r in results],
    }


def snapshot(lesson):
    """初始化后的实际状态，用来检查Stop是否确实恢复了同一起点。"""
    return {'cube_position_m': lesson.cube_start.tolist(),
            'joint_positions': lesson.robot.get_joint_positions().tolist()}


def check_reset(initial, reference):
    a, b = initial['joint_positions'], reference['joint_positions']
    if not a or len(a) != len(b) or len(initial['cube_position_m']) != 3 or len(reference['cube_position_m']) != 3:
        raise RuntimeError('重置状态维度不一致')
    if not all(math.isfinite(v) for v in a+b+initial['cube_position_m']+reference['cube_position_m']):
        raise RuntimeError('重置状态包含非有限数值')
    cube_error = math.dist(initial['cube_position_m'], reference['cube_position_m'])
    joint_error = max(abs(x-y) for x, y in zip(a, b))
    # 当前9DOF包含旋转关节(rad)和手指移动关节(m)，这里分别逐项用1e-3容差。
    if cube_error > 0.001 or joint_error > 0.001:
        raise RuntimeError(f'重置起点不一致：方块偏差{cube_error:.6f}m，最大关节偏差{joint_error:.6f}')
    return {'cube_position_error_m': cube_error, 'max_joint_position_error': joint_error}


async def run_one(module, stage, reference, run_index):
    import omni.kit.app
    import omni.physx
    import omni.timeline
    import omni.usd
    timeline = omni.timeline.get_timeline_interface()
    app = omni.kit.app.get_app()
    lesson = None
    subscription = None
    initial = None
    reset_check = None
    init_status, init_detail = 'Failed', ''
    try:
        if omni.usd.get_context().get_stage() != stage:
            raise RuntimeError('场景已切换，停止批次')
        # ① 先恢复场景，再处理USD变更，最后创建全新物理句柄/状态机。
        timeline.stop()
        await app.next_update_async()
        if omni.usd.get_context().get_stage() != stage:
            raise RuntimeError('初始化时场景已切换')
        module['lesson05']['prepare_contact_material']()
        await app.next_update_async()
        if omni.usd.get_context().get_stage() != stage:
            raise RuntimeError('初始化时场景已切换')
        lesson = module['PickAndPlaceLesson']()
        initial = snapshot(lesson)
        reset_check = check_reset(initial, reference or initial)
        subscription = omni.physx.get_physx_interface().subscribe_physics_step_events(lesson.tick)
        started = time.monotonic()
        timeline.play()
        # ② 控制仍在物理步回调中运行；此循环只负责监视生命周期。
        while lesson.status == 'Running':
            await app.next_update_async()
            setattr(builtins, PROGRESS_KEY, {'run_index': run_index, 'phase': lesson.phase,
                                             'elapsed_seconds': lesson.elapsed})
            if omni.usd.get_context().get_stage() != stage or not timeline.is_playing():
                lesson.status, lesson.detail = 'Cancelled', '用户暂停/停止或切换场景，整批结束'
                break
            if lesson.status == 'Running' and time.monotonic()-started > WALL_TIMEOUT_SECONDS:
                lesson.status, lesson.detail = 'TimedOut', '超过墙钟时间上限，检查仿真是否正常推进'
    except asyncio.CancelledError:
        init_status, init_detail = 'Cancelled', '批次任务被取消'
        if lesson:
            lesson.status, lesson.detail = init_status, init_detail
    except Exception as exc:
        init_detail = str(exc)
        if lesson:
            lesson.status, lesson.detail = 'Failed', init_detail
    finally:
        # ③ 在下一轮Stop之前释放旧回调，防止多个控制器同时下发动作。
        subscription = None
        if omni.usd.get_context().get_stage() == stage and timeline.is_playing():
            timeline.pause()
    result = lesson.report(save=False) if lesson else {'status': init_status, 'phase': 'Initialize', 'detail': init_detail}
    result['initial_state'] = initial
    result['reset_check'] = reset_check
    if reset_check is None:
        result['phase'] = 'Initialize'
    return result


async def batch(run_count, directory):
    import omni.timeline
    import omni.usd
    results, reference = [], None
    batch_status, detail = 'Running', ''
    started = datetime.now().astimezone().isoformat()
    stage = omni.usd.get_context().get_stage()
    stage_path = stage.GetRootLayer().identifier if stage else None

    def persist():
        report = summarize(results, run_count, batch_status, directory.name)
        report.update(started_at=started, stage=stage_path, detail=detail,
                      output_directory=str(directory))
        write_json(directory / 'summary.json', report)
        setattr(builtins, RESULT_KEY, report)
        return report

    try:
        if stage is None or omni.timeline.get_timeline_interface().is_playing():
            raise RuntimeError('请打开练习场景，并先暂停现有任务。')
        module = runpy.run_path(str(Path(__file__).with_name('06_pick_and_place.py')))
        persist()
        for index in range(1, run_count+1):
            setattr(builtins, PROGRESS_KEY, {'run_index': index, 'phase': 'Initialize'})
            print(f'[第7课] 第{index}/{run_count}轮')
            result = await run_one(module, stage, reference, index)
            result.update(run_index=index, report_file=f'run_{index:03d}.json')
            write_json(directory / result['report_file'], result)
            results.append(result)
            if reference is None and result.get('reset_check') is not None:
                reference = result['initial_state']
            persist()
            if result['status'] == 'Cancelled':
                batch_status, detail = 'Cancelled', result['detail']
                break
            if result['phase'] == 'Initialize' or omni.usd.get_context().get_stage() != stage:
                batch_status, detail = 'Aborted', result['detail']
                break
        else:
            batch_status = 'Completed'  # 表示计划轮数完成，不表示所有轮次成功。
    except asyncio.CancelledError:
        batch_status, detail = 'Cancelled', '批次任务被取消'
    except Exception as exc:
        batch_status, detail = 'Aborted', str(exc)
    finally:
        report = persist()
        setattr(builtins, PROGRESS_KEY, {'phase': batch_status, 'recorded_runs': len(results)})
        print('[第7课] '+json.dumps(report, ensure_ascii=False))
    return report


def start(run_count=RUN_COUNT):
    if isinstance(run_count, bool) or not isinstance(run_count, int) or not 1 <= run_count <= 20:
        raise ValueError('教学批次轮数限定为1～20的整数。')
    keys = (TASK_KEY, '_isaac_lesson03_gripper_task', '_isaac_lesson04_task',
            '_isaac_lesson05_task', '_isaac_lesson06_task', '_isaac_lesson09_task')
    for key in keys:
        previous = getattr(builtins, key, None)
        if previous is not None and not previous.done():
            print('[第7课] 前一任务尚未结束，请先暂停并等待。')
            return
    batch_id = datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
    directory = OUTPUT_ROOT / batch_id
    directory.mkdir(parents=True, exist_ok=False)
    setattr(builtins, RESULT_KEY, {'status': 'Running', 'output_directory': str(directory)})
    task = asyncio.ensure_future(batch(run_count, directory))
    setattr(builtins, TASK_KEY, task)
    return task


if __name__ == '__main__':
    start()
