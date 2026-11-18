"""恢复瓶身/刚体相对变换，解决只移动子网格导致的重心错位。

先Stop，备份后使用。仅清除瓶子Body与内部Mesh的本地变换覆盖，
让官方资产的变换重新生效；整体位置由外层PracticeBottle控制。
不改质量/摩擦，不保存官方资产。适用于当前指定瓶子场景。
"""
import json
import numpy as np
import omni.timeline
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdPhysics


def main():
    stage = omni.usd.get_context().get_stage()
    if not stage or not stage.GetRootLayer().identifier.replace('\\', '/').endswith('/Isaac_Desk_Arm_Bottle.usda'):
        raise RuntimeError('请打开水瓶练习场景')
    if not omni.timeline.get_timeline_interface().is_stopped():
        raise RuntimeError('请先Stop，等待更新后再运行')
    wrapper = stage.GetPrimAtPath('/World/PracticeBottle')
    body = stage.GetPrimAtPath('/World/PracticeBottle/Body')
    mesh = stage.GetPrimAtPath('/World/PracticeBottle/Body/_06_mustard_bottle')
    if not all((wrapper, body, mesh)):
        raise RuntimeError('瓶子节点缺失')
    removed = []
    for layer in (stage.GetSessionLayer(), stage.GetRootLayer()):
        with Usd.EditContext(stage, layer):
            # 临时侧倒实验的外层位姿仅在会话层；同时清除以恢复直立。
            targets = (wrapper, body, mesh) if layer == stage.GetSessionLayer() else (body, mesh)
            for prim in targets:
                spec = layer.GetPrimAtPath(prim.GetPath())
                names = [p.name for p in spec.properties] if spec else []
                for name in names:
                    if name == 'xformOpOrder' or name.startswith('xformOp:'):
                        prim.RemoveProperty(name)
                        removed.append(str(prim.GetPath())+'.'+name)
            # 清除运行时残留速度，保留原质量、碰撞和材质。
            rigid = UsdPhysics.RigidBodyAPI(body)
            rigid.CreateVelocityAttr(Gf.Vec3f(0))
            rigid.CreateAngularVelocityAttr(Gf.Vec3f(0))
    with Usd.EditContext(stage, stage.GetRootLayer()):
        x = UsdGeom.Xformable(wrapper)
        x.ClearXformOpOrder()
        translate = x.AddTranslateOp()
        x.AddRotateXOp().Set(-90.0)
        translate.Set(Gf.Vec3d(0))
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy'])
        bounds = cache.ComputeWorldBound(wrapper).ComputeAlignedRange()
        table_z = cache.ComputeWorldBound(stage.GetPrimAtPath('/World/DeskTop')).ComputeAlignedRange().GetMax()[2]
        center = bounds.GetMidpoint()
        translate.Set(Gf.Vec3d(0.18-center[0], -0.20-center[1], table_z-bounds.GetMin()[2]+0.002))

    # 核对碰撞网格在刚体局部坐标中是否仍居中。
    body_world = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    mesh_to_body = UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) * body_world.GetInverse()
    points = np.array([mesh_to_body.Transform(Gf.Vec3d(p)) for p in UsdGeom.Mesh(mesh).GetPointsAttr().Get()])
    center = (points.min(axis=0)+points.max(axis=0))/2
    com = np.array(UsdPhysics.MassAPI(body).GetCenterOfMassAttr().Get())
    error = float(np.linalg.norm(center-com))
    if error > 0.01:
        raise RuntimeError(f'瓶身与重心仍偏离{error}m，未保存')
    stage.GetRootLayer().Save()
    result = {'removed_overrides': removed, 'body_local_mesh_center_m': center.tolist(),
              'body_local_com_m': com.tolist(), 'mesh_center_to_com_m': error,
              'mass_kg': UsdPhysics.MassAPI(body).GetMassAttr().Get()}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    main()
