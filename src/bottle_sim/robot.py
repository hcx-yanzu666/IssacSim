"""关节状态：读取实际关节状态。运行于 Isaac Sim 5.1 Script Editor / MCP。

通过 WATCH_JOINT 选择要检查的关节。
时间线需停止或暂停。首次物理初始化会执行预热步，并非纯 USD 只读。
不持续播放、不发送运动目标、不保存场景。
"""

from pathlib import Path as _ProjectPath
import sys as _sys
PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in _sys.path:
    _sys.path.insert(0, str(PROJECT_ROOT / "src"))

import math
import runpy
from pathlib import Path

import omni.timeline
import omni.usd
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Usd, UsdPhysics

ROBOT_PATH = "/World/Franka"

# 待检查的关节名称；可指定机械臂关节或夹爪关节。
# 运行前预测：单位会怎么变？这个数值代表什么？
WATCH_JOINT = "panda_finger_joint1"


def initialize_robot(allow_playing=False):
    """找到已有机械臂，为它建立物理运行时接口。"""
    if omni.timeline.get_timeline_interface().is_playing() and not allow_playing:
        raise RuntimeError("请先 Pause，再运行关节检查，避免干扰正在执行的任务。")
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("请先打开机械臂练习场景。")
    prim = stage.GetPrimAtPath(ROBOT_PATH)
    if not prim or not prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        raise RuntimeError(f"{ROBOT_PATH} 不是有效的关节系统根节点。")

    # 当前5.1资产曾出现实例几何显示与物理位置不一致，先处理会话层显示。
    # 单独放在辅助文件，不修改官方资产、关节锚点或驱动参数。
    compat = runpy.run_path(str(PROJECT_ROOT / 'src/bottle_sim/visual_compat.py'))
    compat['prepare_franka_visuals'](stage, ROBOT_PATH)

    # API 1：初始化物理视图；5.1 在需要时执行预热物理步。
    # 暂不创建 World，不重置整场景，也不启动持续 Play。
    SimulationManager.initialize_physics()

    # API 2：包装已有机械臂，不是创建另一台机器人。
    # 不指定 position，且保留 USD 原有变换操作结构。
    robot = SingleArticulation(
        prim_path=ROBOT_PATH, name="franka_robot", reset_xform_properties=False
    )
    robot.initialize()
    if not robot.handles_initialized:
        raise RuntimeError("物理句柄未就绪，请检查场景配置与控制台错误。")
    return robot, prim


def read_metadata(robot_prim):
    """USD 用来查关节类型及联动配置，不用遍历顺序决定控制索引。"""
    kinds, mimic = {}, []
    for prim in Usd.PrimRange(robot_prim):
        if prim.IsA(UsdPhysics.RevoluteJoint):
            kinds[prim.GetName()] = "rotation"
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            kinds[prim.GetName()] = "translation"
        for schema in prim.GetAppliedSchemas():
            if "mimic" in schema.lower():
                refs = {r.GetName(): [str(t) for t in r.GetTargets()]
                        for r in prim.GetRelationships() if "mimic" in r.GetName().lower()}
                mimic.append((prim.GetName(), schema, refs))
    return kinds, mimic


def inspect(watch_joint=WATCH_JOINT):
    robot, prim = initialize_robot()

    # API 3：名称与状态数组下标一一对应。必须按名称查索引，不能猜顺序。
    names = list(robot.dof_names)
    positions = robot.get_joint_positions()
    velocities = robot.get_joint_velocities()
    if positions is None or velocities is None:
        raise RuntimeError("未读取到运行时关节状态。")
    if len(names) != len(positions) or len(names) != len(velocities):
        raise RuntimeError("名称和状态数组长度不一致。")
    indices = {name: index for index, name in enumerate(names)}
    kinds, mimic = read_metadata(prim)

    print("\n========== 关节状态：实际关节状态 ==========")
    print(f"运行时 DOF 数量：{len(names)}（以实际输出为准）")
    for index, name in enumerate(names):
        unit = {"rotation": "rad", "translation": "m"}.get(kinds.get(name), "未知")
        print(f"[{index}] {name}: 位置={float(positions[index]):.6f} {unit}, "
              f"速度={float(velocities[index]):.6f} {unit}/s")

    if watch_joint not in indices:
        raise ValueError(f"找不到 {watch_joint}，请从上面的名称中选择。")
    index = indices[watch_joint]
    position = float(positions[index])
    if kinds.get(watch_joint) == "rotation":
        meaning = f"{position:.6f} rad，约 {math.degrees(position):.2f} 度"
    elif kinds.get(watch_joint) == "translation":
        meaning = f"{position:.6f} m，约 {position * 1000:.2f} mm（单个移动关节位移）"
    else:
        meaning = str(position)
    print(f"\n重点观察：{watch_joint} → 数组下标 {index} → {meaning}")
    print("\nUSD mimic 联动配置：")
    for name, schema, refs in mimic:
        print(f"  {name}: {schema}, 引用={refs}")
    if not mimic:
        print("  未发现 mimic schema；不能仅凭这一结果判断两指一定独立。")
    print("联动配置不等于动态开合验证；可用夹爪开合脚本检查。本脚本没有下发运动目标。")
    # 只返回数值快照，不长期保存物理句柄；Stop/切换场景后应重新初始化。
    return {"names": names, "positions": [float(v) for v in positions],
            "watch_joint": watch_joint, "watch_index": index}


if __name__ == "__main__":
    inspection_result = inspect()
