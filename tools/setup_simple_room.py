"""Build an optional furnished scene in Isaac; keep the original scene intact.

Run build() in Isaac Script Editor. Assets remain referenced from the local
Lightwheel library. The decoration does not add task collision geometry.
"""
from pathlib import Path
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = Path(r'D:\tools\Lightwheel_OpenSource\Lightwheel_OpenSource')
OUTPUT = ROOT / 'Isaac_Desk_Arm_Bottle_Room.usda'


def transform(prim, position=(0, 0, 0), scale=(1, 1, 1), yaw=0):
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*position))
    xf.AddRotateZOp().Set(float(yaw))
    xf.AddScaleOp().Set(Gf.Vec3f(*scale))


def bounds(prim):
    return UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render']).ComputeWorldBound(prim).ComputeAlignedRange()


def material(stage, name, color, roughness=0.65):
    mat = UsdShade.Material.Define(stage, '/World/Room/Looks/' + name)
    shader = UsdShade.Shader.Define(stage, str(mat.GetPath()) + '/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(roughness)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    return mat


def box(stage, name, position, size, mat):
    cube = UsdGeom.Cube.Define(stage, '/World/Room/' + name)
    cube.CreateSizeAttr(1)
    transform(cube.GetPrim(), position, size)
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)
    return cube.GetPrim()


def reference(stage, name, relative, center_xy, target_size=None, floor=0, yaw=0):
    asset = LIBRARY / relative
    if not asset.is_file():
        raise FileNotFoundError(asset)
    parent = UsdGeom.Xform.Define(stage, '/World/Room/' + name).GetPrim()
    model = stage.DefinePrim(str(parent.GetPath()) + '/Model')
    model.GetReferences().AddReference(asset.as_posix())
    # Author inert decoration overrides before this stage is opened by PhysX.
    for prim in list(Usd.PrimRange(model)):
        if prim.IsInstance():
            prim.SetInstanceable(False)
    for prim in list(Usd.PrimRange(model)):
        if prim.IsA(UsdPhysics.Joint):
            prim.SetActive(False)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
    rng = bounds(model)
    size, mid = rng.GetSize(), rng.GetMidpoint()
    scale = tuple(target_size[i] / size[i] for i in range(3)) if target_size else (1, 1, 1)
    # These selected models are centered in XY; yaw only used on centered assets.
    position = (center_xy[0] - mid[0]*scale[0], center_xy[1] - mid[1]*scale[1], floor-rng.GetMin()[2]*scale[2])
    transform(parent, position, scale, yaw)
    return parent


def build():
    if OUTPUT.exists():
        raise FileExistsError(f'Keep existing room scene: {OUTPUT}')
    stage = Usd.Stage.CreateNew(str(OUTPUT))
    stage.GetRootLayer().subLayerPaths = ['./Isaac_Desk_Arm_Bottle.usda']
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    UsdGeom.Xform.Define(stage, '/World/Room')
    floor_mat = material(stage, 'WarmConcrete', (0.28, 0.27, 0.24), 0.88)
    wall_mat = material(stage, 'WarmPlaster', (0.62, 0.60, 0.54), 0.85)
    accent_mat = material(stage, 'SagePaint', (0.20, 0.29, 0.27), 0.8)
    trim_mat = material(stage, 'DarkTrim', (0.065, 0.075, 0.07), 0.6)
    box(stage, 'Floor', (0, 0, -0.035), (6, 6, 0.06), floor_mat)
    box(stage, 'BackWall', (0, 2.55, 1.55), (6, 0.12, 3.1), wall_mat)
    box(stage, 'LeftWall', (-2.55, 0, 1.55), (0.12, 5.1, 3.1), accent_mat)
    box(stage, 'BackSkirting', (0, 2.47, 0.06), (5, 0.025, 0.12), trim_mat)
    box(stage, 'LeftSkirting', (-2.47, 0, 0.06), (0.025, 5, 0.12), trim_mat)
    reference(stage, 'Table', 'Locomotion/Apartment/Props/SM_table_02_0.usd', (0, 0), (1.4, 0.8, 0.75))
    table_model = '/World/Room/Table/Model'
    wood = UsdShade.Material.Get(stage, table_model + '/Looks/MI_wood_table_01')
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(table_model + '/SM_table_02_0/Section1')).Bind(wood)
    reference(stage, 'BackgroundCabinet', 'Locomotion/KitchenRoom/Kitchen_Cabinet001/Kitchen_Cabinet001.usd', (-0.35, 2.08))
    # Render-only furniture replaces the block desk visually. The original
    # Cube stays active for physics and ExistingCuboid's direct bounds query.
    for path in ['/World/DeskTop', '/World/DeskLeg_FL', '/World/DeskLeg_FR', '/World/DeskLeg_BL', '/World/DeskLeg_BR', '/World/groundPlane']:
        UsdGeom.Imageable(stage.GetPrimAtPath(path)).CreateVisibilityAttr('invisible')
    UsdLux.DomeLight(stage.GetPrimAtPath('/World/AmbientLight')).GetIntensityAttr().Set(700)
    old_light = UsdLux.SphereLight(stage.GetPrimAtPath('/World/DeskLight'))
    old_light.GetIntensityAttr().Set(600)
    old_light.CreateRadiusAttr(0.25)
    for prim in stage.Traverse():
        if prim.IsA(UsdLux.DistantLight):
            UsdLux.DistantLight(prim).GetIntensityAttr().Set(0)
    UsdGeom.Imageable(stage.GetPrimAtPath('/World/PracticeCube')).CreateVisibilityAttr('invisible')
    for name, pos, intensity, color, width, height in [
        ('KeySoftbox', (0.2, -0.4, 2.7), 4500, (1, 0.94, 0.84), 2.0, 1.5),
        ('RoomFill', (-1.4, 1.1, 2.8), 2500, (0.86, 0.93, 1), 1.3, 1.3),
    ]:
        light = UsdLux.RectLight.Define(stage, '/World/Room/' + name)
        light.CreateIntensityAttr(intensity)
        light.CreateColorAttr(Gf.Vec3f(*color))
        light.CreateWidthAttr(width)
        light.CreateHeightAttr(height)
        transform(light.GetPrim(), pos)
    camera = UsdGeom.Camera.Define(stage, '/World/Room/OverviewCamera')
    camera.CreateFocalLengthAttr(28)
    camera.CreateHorizontalApertureAttr(36)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.03, 100))
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(2.5, -3.6, 2.25), Gf.Vec3d(-0.1, 0.4, 0.9), Gf.Vec3d(0, 0, 1))
    camera.AddTransformOp().Set(view.GetInverse())
    stage.GetRootLayer().customLayerData = {
        'description': 'Simple room decoration; original desk collider and task transforms preserved.',
        'cameraSettings': {'boundCamera': '/World/Room/OverviewCamera'},
    }
    assert abs(bounds(stage.GetPrimAtPath('/World/DeskTop')).GetMax()[2] - 0.75) < 1e-6
    assert abs(bounds(stage.GetPrimAtPath('/World/Room/Table')).GetMax()[2] - 0.75) < 1e-5
    stage.GetRootLayer().Save()
    print('Saved furnished scene:', OUTPUT)
    return str(OUTPUT)


if __name__ == '__main__':
    build()
