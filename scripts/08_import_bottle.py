"""第8课第一步：在水瓶练习场景导入官方YCB芥末瓶。

仅在停止状态运行；已存在则只读取检查，不重复导入。
模型原始-Y方向是瓶盖，绕X旋转-90度使瓶盖朝上。
原始物理参数保留；这一步不改抓取控制器。
"""
import json
import omni.timeline
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdPhysics

ASSET_URL = 'https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd'
BOTTLE_PATH = '/World/PracticeBottle'
BODY_PATH = BOTTLE_PATH + '/Body'
POSITION_XY_M = (0.18, -0.20)


def main():
    stage = omni.usd.get_context().get_stage()
    if not stage or not stage.GetRootLayer().identifier.replace('\\', '/').endswith('/Isaac_Desk_Arm_Bottle.usda'):
        raise RuntimeError('请先打开单独的Isaac_Desk_Arm_Bottle.usda练习场景。')
    if not omni.timeline.get_timeline_interface().is_stopped():
        raise RuntimeError('请先Stop后再导入资产。')
    if UsdGeom.GetStageMetersPerUnit(stage) != 1.0 or UsdGeom.GetStageUpAxis(stage) != 'Z':
        raise RuntimeError('本练习要求米为单位、Z轴向上。')
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy'])
    table = stage.GetPrimAtPath('/World/DeskTop')
    if not table:
        raise RuntimeError('缺少桌面')
    table_bounds = cache.ComputeWorldBound(table).ComputeAlignedRange()
    table_z = table_bounds.GetMax()[2]
    if not stage.GetPrimAtPath(BOTTLE_PATH):
        with Usd.EditContext(stage, stage.GetRootLayer()):
            # 用外层Xform放置，避免清除资产内部原本的变换。
            wrapper = UsdGeom.Xform.Define(stage, BOTTLE_PATH)
            translate = wrapper.AddTranslateOp()
            wrapper.AddRotateXOp().Set(-90.0)
            body = stage.DefinePrim(BODY_PATH)
            body.GetReferences().AddReference(ASSET_URL)
            cache.Clear()
            bounds = cache.ComputeWorldBound(wrapper.GetPrim()).ComputeAlignedRange()
            if bounds.IsEmpty():
                raise RuntimeError('资产未成功加载')
            center = bounds.GetMidpoint()
            # 先在桌面上方留2mm；物理测试观察真实落稳。
            translate.Set(Gf.Vec3d(POSITION_XY_M[0]-center[0], POSITION_XY_M[1]-center[1],
                                  table_z-bounds.GetMin()[2]+0.002))
    cache.Clear()
    prim = stage.GetPrimAtPath(BOTTLE_PATH)
    bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    bodies, collisions = [], []
    for p in Usd.PrimRange(prim):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies.append(str(p.GetPath()))
        if p.HasAPI(UsdPhysics.CollisionAPI):
            collisions.append(str(p.GetPath()))
    if not bodies or not collisions:
        raise RuntimeError('模型缺少刚体或碰撞配置')
    result = {'asset_url': ASSET_URL, 'prim_path': BOTTLE_PATH,
              'rigid_bodies': bodies, 'colliders': collisions,
              'size_m': list(bounds.GetSize()), 'bounds_min_m': list(bounds.GetMin()),
              'bounds_max_m': list(bounds.GetMax()), 'table_height_m': table_z,
              'authored_mass_kg': UsdPhysics.MassAPI(stage.GetPrimAtPath(BODY_PATH)).GetMassAttr().Get()}
    stage.GetRootLayer().Save()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    main()
