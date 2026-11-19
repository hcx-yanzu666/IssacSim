"""Repeat the fixed neck-grasp test and write a batch summary.

Run inside Isaac Sim Script Editor while the bottle scene is stopped:

    import runpy
    ns = runpy.run_path(r"D:\\qtproject\\IssacSim Pj\\scripts\\16_repeat_fixed_pose_waterbottle_grasp.py")
    ns["start"](10)

The single-run controller remains in 15_fixed_pose_waterbottle_grasp.py.
Each run stops the timeline, restores its session-layer overrides, and writes
one independent JSON report before the next run starts.
"""
import asyncio
import builtins
import json
import runpy
import time
import uuid
from collections import Counter
from pathlib import Path

import omni.kit.app
import omni.timeline


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "outputs" / "fixed_pose_grasp_batches"
CONTROLLER_PATH = SCRIPT_DIR / "15_fixed_pose_waterbottle_grasp.py"
TASK_KEY = "_isaac_waterbottle_fixed_grasp_batch_task"
RESULT_KEY = "_isaac_waterbottle_fixed_grasp_batch_result"
PROGRESS_KEY = "_isaac_waterbottle_fixed_grasp_batch_progress"
DEFAULT_RUN_COUNT = 10
MAX_RUN_COUNT = 20


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _metric_values(results, field):
    values = []
    for result in results:
        value = result.get(field)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def summarize(results, requested, batch_id, started_at, directory):
    counts = Counter(result.get("status", "Unknown") for result in results)
    evaluated = counts.get("Succeeded", 0) + counts.get("Failed", 0) + counts.get("TimedOut", 0)
    successes = counts.get("Succeeded", 0)
    lifts = _metric_values(results, "max_lift_m")
    slips = _metric_values(results, "max_slip_m")
    elapsed = _metric_values(results, "elapsed_s")
    failure_phases = Counter(
        result.get("phase", "Unknown")
        for result in results
        if result.get("status") != "Succeeded"
    )

    def stats(values):
        if not values:
            return None
        return {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }

    return {
        "batch_id": batch_id,
        "status": "Completed" if len(results) == requested else "Incomplete",
        "requested_runs": requested,
        "recorded_runs": len(results),
        "evaluated_runs": evaluated,
        "success_count": successes,
        "success_rate": (successes / evaluated) if evaluated else None,
        "status_counts": dict(counts),
        "failure_phase_counts": dict(failure_phases),
        "max_lift_m": stats(lifts),
        "max_slip_m": stats(slips),
        "elapsed_s": stats(elapsed),
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "output_directory": str(directory),
        "controller": str(CONTROLLER_PATH),
        "acceptance": {
            "success_rate_target": 0.90,
            "minimum_lift_m": 0.072,
            "maximum_slip_m": 0.015,
        },
    }


async def _wait_for_controller(task, timeout_s=240.0):
    started = time.monotonic()
    while not task.done():
        await omni.kit.app.get_app().next_update_async()
        if time.monotonic() - started > timeout_s:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return {
                "status": "TimedOut",
                "phase": "BatchWait",
                "detail": "批量脚本等待单轮任务超过墙钟上限",
            }
    try:
        await task
    except Exception as exc:
        return {
            "status": "Failed",
            "phase": "ControllerTask",
            "detail": str(exc),
        }
    result = getattr(builtins, "_isaac_waterbottle_fixed_grasp_result", None)
    if not isinstance(result, dict):
        return {
            "status": "Failed",
            "phase": "Result",
            "detail": "单轮任务结束但没有发布结果对象",
        }
    return dict(result)


async def batch(run_count, directory, batch_id):
    timeline = omni.timeline.get_timeline_interface()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    results = []
    controller = runpy.run_path(str(CONTROLLER_PATH))
    setattr(builtins, PROGRESS_KEY, {"status": "Running", "run_index": 0, "run_count": run_count})

    try:
        for index in range(1, run_count + 1):
            timeline.stop()
            await omni.kit.app.get_app().next_update_async()
            setattr(builtins, PROGRESS_KEY, {
                "status": "Running",
                "run_index": index,
                "run_count": run_count,
                "phase": "Initialize",
            })
            print(f"[固定瓶颈抓取批量] 第 {index}/{run_count} 轮")

            try:
                task = controller["start"]()
                result = await _wait_for_controller(task)
            except Exception as exc:
                result = {
                    "status": "Failed",
                    "phase": "Initialize",
                    "detail": str(exc),
                }

            result["run_index"] = index
            result["batch_id"] = batch_id
            result_path = directory / f"run_{index:03d}.json"
            _write_json(result_path, result)
            results.append(result)
            setattr(builtins, PROGRESS_KEY, {
                "status": "Running",
                "run_index": index,
                "run_count": run_count,
                "phase": result.get("phase"),
                "run_status": result.get("status"),
            })
            print(
                f"[固定瓶颈抓取批量] 第 {index} 轮: "
                f"{result.get('status')} / {result.get('phase')} / {result.get('detail', '')}"
            )

        report = summarize(results, run_count, batch_id, started_at, directory)
    except Exception as exc:
        report = summarize(results, run_count, batch_id, started_at, directory)
        report.update({"status": "Aborted", "detail": str(exc)})
    finally:
        timeline.stop()
        await omni.kit.app.get_app().next_update_async()
        _write_json(directory / "summary.json", report)
        setattr(builtins, RESULT_KEY, report)
        setattr(builtins, PROGRESS_KEY, {
            "status": report["status"],
            "recorded_runs": report["recorded_runs"],
            "success_rate": report["success_rate"],
            "output_directory": str(directory),
        })
        print("[固定瓶颈抓取批量] " + json.dumps(report, ensure_ascii=False))
    return report


def start(run_count=DEFAULT_RUN_COUNT):
    if isinstance(run_count, bool) or not isinstance(run_count, int) or not 1 <= run_count <= MAX_RUN_COUNT:
        raise ValueError(f"批量轮数必须是 1～{MAX_RUN_COUNT} 的整数")
    previous = getattr(builtins, TASK_KEY, None)
    if previous is not None and not previous.done():
        raise RuntimeError("固定瓶颈抓取批量测试正在运行")
    batch_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    directory = OUTPUT_ROOT / batch_id
    directory.mkdir(parents=True, exist_ok=False)
    setattr(builtins, RESULT_KEY, {"status": "Running", "output_directory": str(directory)})
    task = asyncio.ensure_future(batch(run_count, directory, batch_id))
    setattr(builtins, TASK_KEY, task)
    return task


if __name__ == "__main__":
    start()
