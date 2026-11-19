"""Isaac in-app controls and split view for randomized dataset collection.

Script Editor: runpy.run_path(path, run_name='__main__'). No Qt dependency.
"""
import builtins
import json
import os
from pathlib import Path
import runpy

import omni.ui as ui
import omni.usd
from omni.kit.widget.viewport import ViewportWidget
from pxr import Gf, Usd, UsdGeom

ROOT = Path(__file__).resolve().parents[2]
PANEL_KEY = '_isaac_dataset_demo_panel'


class DatasetPanel:
    def __init__(self):
        self.runner = runpy.run_path(str(ROOT / 'src/bottle_sim/dataset.py'))
        self.config = self.runner['SUPPORT']['load_config']()
        self.models = {}
        self.task = None
        self.cancel_requested = False
        self.output_dir = None
        self.viewports = []
        self.runner['ensure_capture_camera'](omni.usd.get_context().get_stage(),self.config,reset_pose=True)
        self.setup_overview()
        # Live viewport textures must not be cached as a static UI raster.
        # Match Kit's ViewportWindow: AUTO can flicker when hover invalidation stops.
        self.window = ui.Window('Dataset Studio | RGB-D Randomization', width=1260, height=760,
                                raster_policy=ui.RasterPolicy.NEVER)
        self.window.set_visibility_changed_fn(self.on_visibility)
        with self.window.frame:
            with ui.VStack(spacing=6):
                with ui.HStack(height=30,spacing=8):
                    ui.Label('BOTTLE DATASET STUDIO',style={'font_size':20},width=300)
                    ui.Button('Preview 1',clicked_fn=lambda:self.start(preview=True))
                    ui.Button('Start capture',clicked_fn=self.start)
                    ui.Button('Stop',clicked_fn=self.stop)
                    ui.Button('Open gallery',clicked_fn=self.open_gallery)
                with ui.HStack(height=22):
                    ui.Label('LEFT  Independent capture camera',alignment=ui.Alignment.CENTER)
                    ui.Label('RIGHT  Scene overview',alignment=ui.Alignment.CENTER)
                with ui.HStack(height=330,spacing=6):
                    self.viewports.append(ViewportWidget(camera_path=self.runner['CAMERA_PATH'],resolution=(640,360)))
                    self.viewports.append(ViewportWidget(camera_path='/World/DatasetOverviewCamera',resolution=(640,360)))
                self.status = ui.Label('Ready | Pause/Stop the simulation before capture',height=24,style={'font_size':17})
                self.values = ui.Label('Position, light angles and intensities appear here during capture.',height=22)
                self.camera_values = ui.Label('Independent camera: random azimuth / elevation / distance, aimed at the bottle.',height=22)
                self.path_label = ui.Label('Output: outputs/bottle_dataset/',height=20,elided_text=True)
                with ui.CollapsableFrame('Capture settings (angles in degrees; intensity uses USD light units)',height=0,collapsed=False):
                    with ui.HStack(spacing=20):
                        with ui.VStack(spacing=4):
                            self.range_field('Bottle X offset (mm)','bottle_dx_m',1000)
                            self.range_field('Bottle Y offset (mm)','bottle_dy_m',1000)
                            self.range_field('Camera azimuth (deg)','camera_azimuth_deg')
                            self.range_field('Camera elevation (deg)','camera_elevation_deg')
                            self.range_field('Camera distance (m)','camera_distance_m')
                        with ui.VStack(spacing=4):
                            self.range_field('Light azimuth (deg)','light_azimuth_deg')
                            self.range_field('Light elevation (deg)','light_elevation_deg')
                            self.range_field('Desk light intensity','desk_intensity')
                            self.range_field('Ambient intensity','ambient_intensity')
                            self.scalar_field('Light distance (m)','light_distance_m')
                        with ui.VStack(spacing=4):
                            self.scalar_field('Hold each sample (s)','demo_hold_seconds')
                            self.scalar_field('Samples','target_samples',integer=True)
                            self.scalar_field('Master seed','master_seed',integer=True)
                            ui.Button('Save settings',height=26,clicked_fn=self.save_settings)
                            ui.Label('0 s hold = batch mode\n1.5 s hold = recording mode',height=44)
                ui.Label('Automatic labels: RGB / depth / mask / YOLO polygon / parameters. No grasp execution.',height=24)

    def setup_overview(self):
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError('Open the bottle scene first')
        with Usd.EditContext(stage,stage.GetSessionLayer()):
            camera = UsdGeom.Camera.Define(stage,'/World/DatasetOverviewCamera')
            camera.CreateFocalLengthAttr(22.0)
            camera.CreateHorizontalApertureAttr(24.0)
            camera.CreateVerticalApertureAttr(13.5)
            camera.CreateClippingRangeAttr(Gf.Vec2f(.01,100.))
            view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(1.55,-1.8,1.85),Gf.Vec3d(.10,0,.85),Gf.Vec3d(0,0,1))
            UsdGeom.Xformable(camera).MakeMatrixXform().Set(view.GetInverse())

    def range_field(self,label,key,scale=1):
        with ui.HStack(height=26,spacing=4):
            ui.Label(label,width=170)
            models=[]
            for value in self.config['ranges'][key]:
                model=ui.SimpleFloatModel(value*scale)
                ui.FloatField(model=model,width=78)
                models.append(model)
            self.models[key]=(models,scale)

    def scalar_field(self,label,key,integer=False):
        with ui.HStack(height=26,spacing=4):
            ui.Label(label,width=170)
            model=ui.SimpleIntModel(self.config[key]) if integer else ui.SimpleFloatModel(self.config[key])
            field=ui.IntField if integer else ui.FloatField
            field(model=model,width=130)
            self.models[key]=(model,integer)

    def read_config(self):
        config=dict(self.config)
        config['ranges']={}
        for key in self.config['ranges']:
            models,scale=self.models[key]
            config['ranges'][key]=[m.as_float/scale for m in models]
        for key in ('light_distance_m','demo_hold_seconds','target_samples','master_seed'):
            model,integer=self.models[key]
            config[key]=model.as_int if integer else model.as_float
        return self.runner['SUPPORT']['load_config'](config)

    def save_settings(self):
        try:
            config=self.read_config()
            (ROOT/'configs/dataset_randomization.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
            self.status.text='Settings saved to configs/dataset_randomization.json'
        except Exception as exc:
            self.status.text='Configuration error: '+str(exc)

    def start(self,preview=False):
        try:
            if self.task is not None and not self.task.done():
                raise RuntimeError('Capture is already running')
            config=self.read_config()
            if preview:
                config['target_samples']=1
                config['demo_hold_seconds']=max(3,config['demo_hold_seconds'])
            self.cancel_requested=False
            self.task=self.runner['start'](config=config,progress_callback=self.on_progress,
                cancel_requested=lambda:self.cancel_requested)
            self.task.add_done_callback(self.finished)
        except Exception as exc:
            self.status.text='Cannot start: '+str(exc)

    def on_progress(self,event):
        self.output_dir=Path(event['output_dir'])
        self.status.text=f"{event['phase'].upper()}  |  Saved {event['accepted']}/{event['target']}  |  Attempts {event['attempts']}  |  Rejected {event['rejected']}  |  {event['detail']}"
        self.path_label.text='Output: '+str(self.output_dir)
        p=event.get('parameters')
        if p:
            self.values.text=f"Bottle dx/dy: {p['bottle_dx_m']*1000:+.1f} / {p['bottle_dy_m']*1000:+.1f} mm   |   Light az/el: {p['light_azimuth_deg']:.1f} / {p['light_elevation_deg']:.1f} deg   |   Desk / ambient: {p['desk_intensity']:.0f} / {p['ambient_intensity']:.0f}"
            self.camera_values.text=f"Camera az/el: {p['camera_azimuth_deg']:.1f} / {p['camera_elevation_deg']:.1f} deg   |   Distance: {p['camera_distance_m']:.3f} m   |   Intrinsics + per-frame extrinsics saved"
        elif event['phase'] in ('completed','cancelled','incomplete','failed'):
            self.values.text='Scene restored. Open gallery to review captured RGB and annotations.'
            self.camera_values.text='Independent camera restored to its pre-capture pose. Wrist camera unchanged.'

    def finished(self,task):
        if not task.cancelled() and task.exception():
            self.status.text='FAILED: '+str(task.exception())

    def stop(self):
        self.cancel_requested=True
        self.status.text='Stop requested; finishing current render and restoring scene...'

    def open_gallery(self):
        if self.output_dir is None:
            latest=ROOT/'outputs/bottle_dataset/latest.json'
            if latest.exists():
                self.output_dir=Path(json.loads(latest.read_text())['output_dir'])
        if self.output_dir and (self.output_dir/'gallery.html').exists():
            os.startfile(str(self.output_dir/'gallery.html'))
        else:
            self.status.text='No gallery yet. Finish or stop a capture first.'

    def on_visibility(self,visible):
        if not visible:
            self.cancel_requested=True
        for viewport in self.viewports:
            # Hiding the widget alone leaves its Hydra render product updating.
            # Suspend rendering while closed and resume when shown again.
            viewport.viewport_api.updates_enabled=bool(visible)
            viewport.visible=bool(visible)

    def destroy(self):
        self.cancel_requested=True
        for viewport in self.viewports:
            viewport.destroy()
        self.viewports=[]
        self.window.destroy()


def show():
    previous=getattr(builtins,PANEL_KEY,None)
    if previous:
        if previous.task is not None and not previous.task.done():
            raise RuntimeError('Stop the running collection before reopening the panel')
        previous.destroy()
    panel=DatasetPanel()
    setattr(builtins,PANEL_KEY,panel)
    return panel


if __name__ == '__main__':
    show()
