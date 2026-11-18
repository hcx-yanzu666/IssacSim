"""第一课：认识当前场景，只读检查，不改变仿真状态。

运行位置：Isaac Sim 5.1 的 Script Editor / MCP。
学习顺序：main → require_prim → describe_object → describe_joints。
"""

import math

import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics


# Prim Path 是场景中的对象路径，不是磁盘上的文件路径。
# 类比：通过对象路径找到 UE 世界中的一个对象，再读取它的组件/属性。
ROBOT_PATH = "/World/Franka"
CUBE_PATH = "/World/PracticeCube"
TABLE_PATH = "/World/DeskTop"


def require_prim(stage, path):
    """GetPrimAtPath：按路径查询场景对象；查询不到时尽早报出原因。"""
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid() or not prim.IsDefined():
        raise RuntimeError(f"未找到有效场景对象：{path}，请先打开机械臂练习场景。")
    return prim


def describe_object(prim, bbox_cache):
    """读取世界坐标和物理能力，不设置任何属性。"""
    # ComputeLocalToWorldTransform 会累积父节点变换。
    # 不能直接把局部 translate 当成世界坐标：机械臂连杆还有父节点变换。
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    position = matrix.ExtractTranslation()
    print(f"\n对象：{prim.GetPath()}")
    print(f"  世界位置（场景单位）：{tuple(round(v, 4) for v in position)}")
    print(f"  根节点有刚体：{prim.HasAPI(UsdPhysics.RigidBodyAPI)}")
    print(f"  根节点有碰撞：{prim.HasAPI(UsdPhysics.CollisionAPI)}")
    print(f"  根节点有 Articulation：{prim.HasAPI(UsdPhysics.ArticulationRootAPI)}")

    # 机械臂根节点没有 CollisionAPI 也不等于没有碰撞：碰撞通常在子 Mesh 上。
    # TraverseInstanceProxies 让检查覆盖实例化资产内部的几何体。
    collision_count = sum(
        p.HasAPI(UsdPhysics.CollisionAPI)
        for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies())
    )
    print(f"  含自身及子节点的碰撞对象数量：{collision_count}")

    if prim.IsA(UsdGeom.Cube):
        # Cube 的 size 与节点缩放共同决定实际尺寸。
        # 世界 AABB 是沿世界轴对齐的包围盒；旋转物体后它不等于物体本体边长。
        bounds = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        print(f"  世界 AABB 尺寸：{tuple(round(v, 4) for v in bounds.GetSize())}")


def describe_joints(robot):
    """读取 USD 关节配置。这里不是运行时关节控制接口。"""
    print("\n关节配置（此遍历顺序不能当成控制数组顺序）：")
    joint_count = 0
    for prim in Usd.PrimRange(robot):
        if prim.IsA(UsdPhysics.RevoluteJoint):
            joint = UsdPhysics.RevoluteJoint(prim)
            drive_name = "angular"
            # USD 旋转关节属性使用角度；Isaac 运行时关节控制通常使用弧度。
            low = math.radians(joint.GetLowerLimitAttr().Get())
            high = math.radians(joint.GetUpperLimitAttr().Get())
            unit = "rad（从 USD 的度转换）"
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            joint = UsdPhysics.PrismaticJoint(prim)
            drive_name = "linear"
            low = joint.GetLowerLimitAttr().Get()
            high = joint.GetUpperLimitAttr().Get()
            unit = "场景长度单位（本场景为米）"
        else:
            continue

        joint_count += 1
        has_drive = prim.HasAPI(UsdPhysics.DriveAPI, drive_name)
        print(f"  {prim.GetName()}: [{low:.4f}, {high:.4f}] {unit}，独立驱动={has_drive}")

    print(f"  共 {joint_count} 个旋转/移动关节。")
    print("  没有独立 drive 不一定是错误，也可能使用 mimic 等联动约束。")
    print("  下一课再读运行时 DOF 名称与实际位置，核对夹爪联动。")


def main():
    # get_context() 获取当前编辑器的 USD 上下文；get_stage() 获取已打开的场景。
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("当前没有打开场景。")

    print("\n========== 第 1 课：场景检查 ==========")
    print(f"场景文件：{stage.GetRootLayer().identifier}")
    print(f"每个场景单位对应的米数：{UsdGeom.GetStageMetersPerUnit(stage)}")
    print(f"向上轴：{UsdGeom.GetStageUpAxis(stage)}")

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    )
    for path in (TABLE_PATH, CUBE_PATH, ROBOT_PATH):
        describe_object(require_prim(stage, path), bbox_cache)
    describe_joints(require_prim(stage, ROBOT_PATH))
    print("\n检查完成：没有播放仿真或修改场景。")


# TODO（学习练习）：解释方块为什么同时需要 CollisionAPI 和 RigidBodyAPI。
# TODO（学习练习）：根据打印的位置，算出方块相对机械臂底座的位置。
# TODO（学习练习）：解释为什么机械臂根节点没有碰撞，手指仍然能碰到方块。
# 完成这三个问题后，再进入 02 文件。此处没有要求你补写底层物理引擎。

if __name__ == "__main__":
    main()
