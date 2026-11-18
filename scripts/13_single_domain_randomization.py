"""Run one reproducible visual-pick trial with scene/sensor randomization.

Run from Isaac Sim Script Editor with the bottle scene open and timeline stopped.
The randomized orientation remains fixed to match the current pose estimator.
"""
import asyncio
import builtins
import json
from pathlib import Path
import runpy
import time

import h5py
import numpy as np
import omni.kit.app
import omni.timeline
import omni.usd
from pxr import Gf, Usd, UsdGeom


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "outputs" / "domain_randomization"
TASK_KEY = "_isaac_domain_randomization_trial"
SUMMARY_PATH = OUTPUT_ROOT / "single_trial_latest.json"

MASTER_SEED = 20261118
RANGES = {
    "bottle_dx_m": [-0.015, 0.015],
    "bottle_dy_m": [-0.015, 0.015],
    "ambient_intensity": [500.0, 900.0],
    "desk_intensity": [1300.0, 2300.0],
    "depth_noise_sigma_m": [0.0, 0.003],
}

BOTTLE_WRAPPER_PATH = "/World/PracticeBottle"
AMBIENT_LIGHT_PATH = "/World/AmbientLight"
DESK_LIGHT_PATH = "/World/DeskLight"
BASELINE_BOTTLE_XY_M = (0.18, -0.20)


def sample_parameters(seed):
    rng = np.random.default_rng(seed)
    return {
        "seed": int(seed),
        "bottle_dx_m": float(rng.uniform(*RANGES["bottle_dx_m"])),
        "bottle_dy_m": float(rng.uniform(*RANGES["bottle_dy_m"])),
        "ambient_intensity": float(rng.uniform(*RANGES["ambient_intensity"])),
        "desk_intensity": float(rng.uniform(*RANGES["desk_intensity"])),
        "depth_noise_sigma_m": float(rng.uniform(*RANGES["depth_noise_sigma_m"])),
        "depth_noise_seed": int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32)),
    }


def apply_randomized_scene(parameters):
    stage = omni.usd.get_context().get_stage()
    wrapper = UsdGeom.Xformable(stage.GetPrimAtPath(BOTTLE_WRAPPER_PATH))
    ambient = stage.GetPrimAtPath(AMBIENT_LIGHT_PATH)
    desk = stage.GetPrimAtPath(DESK_LIGHT_PATH)
    if not wrapper or not ambient or not desk:
        raise RuntimeError("瓶子外层、AmbientLight 或 DeskLight 场景节点缺失")

    baseline_xy = np.asarray(BASELINE_BOTTLE_XY_M, dtype=float)
    randomized_xy = baseline_xy + np.array([
        parameters["bottle_dx_m"], parameters["bottle_dy_m"]
    ])
    table = runpy.run_path(str(SCRIPT_DIR / "04_hover_above_cube.py"))
    table_prim = table["ExistingCuboid"](stage, "/World/DeskTop")
    safe_xy = table_prim.size[:2] / 2.0 - 0.12
    if np.any(np.abs(randomized_xy - table_prim.center[:2]) > safe_xy):
        raise ValueError("随机瓶子位置超出工作台安全范围")

    with Usd.EditContext(stage, stage.GetSessionLayer()):
        translate = next((op for op in wrapper.GetOrderedXformOps()
                          if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
        if translate is None:
            raise RuntimeError("PracticeBottle 外层缺少 translate xform op")
        original_z = float(translate.Get()[2])
        translate.Set(Gf.Vec3d(float(randomized_xy[0]), float(randomized_xy[1]), original_z))
        ambient.GetAttribute("inputs:intensity").Set(float(parameters["ambient_intensity"]))
        desk.GetAttribute("inputs:intensity").Set(float(parameters["desk_intensity"]))

    parameters["bottle_xy_world_m"] = randomized_xy.tolist()
    parameters["bottle_orientation_randomized"] = False
    return parameters


def save_hdf5(output_path, parameters, result, data_dir):
    with h5py.File(output_path, "w") as output:
        output.attrs["schema"] = "virtusim_domain_randomization_single_trial_v1"
        output.attrs["master_seed"] = parameters["seed"]
        output.attrs["parameters_json"] = json.dumps(parameters, ensure_ascii=False)
        output.attrs["result_json"] = json.dumps(result, ensure_ascii=False)
        for name, filename in (("rgb", "rgb.png"), ("mask", "bottle_mask.png")):
            from PIL import Image
            array = np.asarray(Image.open(data_dir / filename))
            output.create_dataset(name, data=array, compression="gzip", shuffle=True)
        for name, filename in (("depth_raw_m", "depth_m.npy"),
                               ("depth_noisy_m", "depth_noisy_m.npy")):
            array = np.load(data_dir / filename)
            output.create_dataset(name, data=array, compression="gzip", shuffle=True)
        for name in ("camera_info.json", "bottle_mask_info.json", "bottle_pose_estimate.json"):
            output.attrs[name] = (data_dir / name).read_text(encoding="utf-8")


async def run_trial():
    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_stopped():
        raise RuntimeError("请先Stop仿真，再启动单轮域随机化")
    await omni.kit.app.get_app().next_update_async()

    trial_id = f"seed_{MASTER_SEED}_{time.strftime('%Y%m%d_%H%M%S')}"
    data_dir = OUTPUT_ROOT / trial_id
    data_dir.mkdir(parents=True, exist_ok=False)
    parameters = sample_parameters(MASTER_SEED)
    parameters["ranges"] = RANGES
    visual = runpy.run_path(str(SCRIPT_DIR / "12_visual_bottle_pick.py"))
    result = {"status": "Failed", "phase": "initialization", "detail": "trial did not finish"}
    try:
        result = await visual["start_visual_pick"](
            data_dir=data_dir,
            depth_noise_sigma_m=parameters["depth_noise_sigma_m"],
            depth_noise_seed=parameters["depth_noise_seed"],
            prepare_scene_callback=lambda: apply_randomized_scene(parameters),
        )
    except asyncio.CancelledError:
        result = {"status": "Cancelled", "phase": "task", "detail": "single trial cancelled"}
        raise
    except Exception as exc:
        result = {"status": "Failed", "phase": "visual_or_pick", "detail": repr(exc)}
        raise
    finally:
        estimate_path = data_dir / "bottle_pose_estimate.json"
        estimate = json.loads(estimate_path.read_text(encoding="utf-8")) if estimate_path.exists() else {}
        report = {
            "trial_id": trial_id,
            "seed": MASTER_SEED,
            "parameters": parameters,
            "status": result.get("status", "Unknown"),
            "failure_phase": result.get("phase", ""),
            "detail": result.get("detail", ""),
            "pose_estimate": estimate,
            "pick_place_result": result,
        }
        summary_path = data_dir / "result.json"
        summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        required = [data_dir / name for name in (
            "rgb.png", "bottle_mask.png", "depth_m.npy", "depth_noisy_m.npy",
            "camera_info.json", "bottle_mask_info.json", "bottle_pose_estimate.json",
        )]
        if all(path.exists() for path in required):
            save_hdf5(data_dir / "trial.h5", parameters, report, data_dir)
        print("[DomainRandomization] " + json.dumps(report, ensure_ascii=False))

    return result


def start():
    previous = getattr(builtins, TASK_KEY, None)
    if previous is not None and not previous.done():
        raise RuntimeError("单轮域随机化任务正在运行")
    task = asyncio.ensure_future(run_trial())
    setattr(builtins, TASK_KEY, task)
    return task


if __name__ == "__main__":
    start()
