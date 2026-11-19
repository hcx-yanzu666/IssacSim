"""Configuration, reproducible sampling and galleries (no Isaac imports)."""

from pathlib import Path as _ProjectPath
import sys as _sys
PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in _sys.path:
    _sys.path.insert(0, str(PROJECT_ROOT / "src"))

import copy
import html
import json
import math
from pathlib import Path

import numpy as np

ROOT = PROJECT_ROOT
CONFIG_PATH = ROOT / 'configs/dataset_randomization.json'


def load_config(overrides=None):
    config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    if overrides:
        unknown = set(overrides) - set(config)
        if unknown:
            raise ValueError(f'Unknown configuration keys: {unknown}')
        for key, value in overrides.items():
            if key == 'ranges':
                if set(value) - set(config['ranges']):
                    raise ValueError('Unknown randomization range')
                config['ranges'].update(copy.deepcopy(value))
            else:
                config[key] = value
    if not isinstance(config['master_seed'], int) or config['master_seed'] < 0:
        raise ValueError('Seed must be a non-negative integer')
    if not isinstance(config['mute_environment_light'], bool):
        raise ValueError('mute_environment_light must be boolean')
    if not isinstance(config['target_samples'], int) or config['target_samples'] < 1:
        raise ValueError('Sample count must be a positive integer')
    if not 0 < config['train_fraction'] < 1:
        raise ValueError('Train fraction must be between 0 and 1')
    for key in ('demo_hold_seconds', 'light_distance_m', 'light_target_height_m', 'camera_target_height_m'):
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f'Invalid {key}')
    if not .5 <= config['light_distance_m'] <= 4:
        raise ValueError('Light distance must be 0.5 to 4 meters')
    if not 0 <= config['demo_hold_seconds'] <= 30:
        raise ValueError('Demo hold must be 0 to 30 seconds')
    for key in ('render_subframes', 'max_attempts_per_sample'):
        if not isinstance(config[key], int) or not 1 <= config[key] <= 64:
            raise ValueError(f'{key} must be 1 to 64')
    for key, bounds in config['ranges'].items():
        if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
            raise ValueError(f'Invalid range: {key}')
        if 'intensity' in key and bounds[0] < 0:
            raise ValueError('Light intensity cannot be negative')
    if not 10 <= config['ranges']['light_elevation_deg'][0] <= config['ranges']['light_elevation_deg'][1] <= 85:
        raise ValueError('Light elevation must stay between 10 and 85 degrees')
    if not 5 <= config['ranges']['camera_elevation_deg'][0] <= config['ranges']['camera_elevation_deg'][1] <= 80:
        raise ValueError('Camera elevation must stay between 5 and 80 degrees')
    if not .35 <= config['ranges']['camera_distance_m'][0] <= config['ranges']['camera_distance_m'][1] <= 2:
        raise ValueError('Camera distance must stay between 0.35 and 2 meters')
    return config


def sample_parameters(seed, sample_index, config):
    # Stable child streams: changing the range of one parameter does not consume
    # random draws belonging to another parameter.
    # Append new parameters after the legacy six streams, so adding camera
    # randomization does not reshuffle the existing light/position samples.
    names = ('ambient_intensity', 'bottle_dx_m', 'bottle_dy_m', 'desk_intensity',
             'light_azimuth_deg', 'light_elevation_deg', 'camera_azimuth_deg',
             'camera_elevation_deg', 'camera_distance_m')
    streams = np.random.SeedSequence(int(seed)).spawn(len(names))
    result = {name: float(np.random.default_rng(stream).uniform(*config['ranges'][name]))
              for name, stream in zip(names, streams)}
    result.update(sample_index=int(sample_index), seed=int(seed),
                  bottle_orientation_randomized=False,
                  light_distance_m=config['light_distance_m'])
    return result


def light_position(parameters, target):
    az = math.radians(parameters['light_azimuth_deg'])
    el = math.radians(parameters['light_elevation_deg'])
    radius = parameters['light_distance_m']
    return (np.asarray(target, dtype=float) + radius * np.array([
        math.cos(el)*math.cos(az), math.cos(el)*math.sin(az), math.sin(el)])).tolist()


def camera_position(parameters, target):
    """Camera orbits the sampled bottle center; angles use world +Z up."""
    az = math.radians(parameters['camera_azimuth_deg'])
    el = math.radians(parameters['camera_elevation_deg'])
    radius = parameters['camera_distance_m']
    return (np.asarray(target, dtype=float) + radius * np.array([
        math.cos(el)*math.cos(az), math.cos(el)*math.sin(az), math.sin(el)])).tolist()


def save_preview(output_dir, record, rgb, mask):
    import cv2
    from PIL import Image, ImageDraw
    out = Path(output_dir) / 'previews'
    out.mkdir(exist_ok=True)
    overlay = rgb.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 230, 80), 2)
    panel = Image.new('RGB', (rgb.shape[1]*2, rgb.shape[0]+84), (20, 25, 32))
    panel.paste(Image.fromarray(rgb), (0, 84))
    panel.paste(Image.fromarray(overlay), (rgb.shape[1], 84))
    p = record['parameters']
    draw = ImageDraw.Draw(panel)
    draw.text((12, 8), f"{record['sample_id']}   RGB / automatic segmentation label", fill='white')
    draw.text((12, 28), f"XY offset: {p['bottle_dx_m']*1000:+.1f}, {p['bottle_dy_m']*1000:+.1f} mm  |  light az/el: {p['light_azimuth_deg']:.1f}/{p['light_elevation_deg']:.1f} deg", fill='white')
    draw.text((12, 46), f"Desk intensity: {p['desk_intensity']:.0f}  Ambient: {p['ambient_intensity']:.0f}  Seed: {record['attempt_seed']}", fill='white')
    if 'camera_azimuth_deg' in p:
        draw.text((12, 65), f"Camera az/el: {p['camera_azimuth_deg']:.1f}/{p['camera_elevation_deg']:.1f} deg  Distance: {p['camera_distance_m']:.3f} m", fill='white')
    rel = f"previews/{record['sample_id']}.jpg"
    panel.save(Path(output_dir) / rel, quality=92)
    return rel


def build_gallery(output_dir, records):
    from PIL import Image, ImageDraw
    output_dir = Path(output_dir)
    if not records:
        return
    shown = records[:min(24, len(records))]
    cols = 4
    sheet = Image.new('RGB', (cols*320, math.ceil(len(shown)/cols)*220), (20,25,32))
    draw = ImageDraw.Draw(sheet)
    for i, record in enumerate(shown):
        image = Image.open(output_dir / record['files']['image']).convert('RGB')
        image.thumbnail((320, 180))
        x, y = (i % cols)*320, (i//cols)*220
        sheet.paste(image, (x,y))
        p = record['parameters']
        draw.text((x+4,y+183), f"{record['sample_id']}  cam {p.get('camera_azimuth_deg',0):.0f}/{p.get('camera_elevation_deg',0):.0f} deg",fill='white')
        draw.text((x+4,y+200),f"Dist {p.get('camera_distance_m',0):.2f} m  Light {p['light_azimuth_deg']:.0f}/{p['light_elevation_deg']:.0f}  I={p['desk_intensity']:.0f}",fill='white')
    sheet.save(output_dir/'contact_sheet.jpg', quality=92)
    cards = ''.join(f'<button class="card" onclick="show({i})"><img src="{html.escape(r["files"]["image"])}"><span>{html.escape(r["sample_id"])}</span></button>' for i,r in enumerate(records))
    slides = json.dumps([r['files']['preview'] for r in records])
    page = '''<!doctype html><meta charset="utf-8"><title>Bottle dataset preview</title>
<style>body{background:#141920;color:#edf1f7;font:16px system-ui;margin:24px}button{cursor:pointer;background:#273447;color:white;border:1px solid #53657c;padding:10px;border-radius:6px}header{display:flex;gap:12px;align-items:center}#frame{width:100%;max-height:72vh;object-fit:contain;background:#0d1117;margin:12px 0}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.card img{width:100%;display:block}.card span{display:block;padding:5px}</style>
<header><b>RGB + automatic segmentation</b><span id="counter"></span><button onclick="show(index-1)">Previous</button><button onclick="show(index+1)">Next</button><button onclick="toggle()" id="play">Play 1.5s</button><a href="contact_sheet.jpg" style="color:#9bd2ff">Contact sheet</a></header>
<img id="frame"><div class="grid">CARDS</div>
<script>const slides=SLIDES;let index=0,timer=null;function show(i){index=(i+slides.length)%slides.length;document.getElementById('frame').src=slides[index];document.getElementById('counter').textContent=(index+1)+' / '+slides.length;}function toggle(){if(timer){clearInterval(timer);timer=null;document.getElementById('play').textContent='Play 1.5s'}else{timer=setInterval(()=>show(index+1),1500);document.getElementById('play').textContent='Pause'}}show(0);</script>'''
    (output_dir/'gallery.html').write_text(page.replace('CARDS',cards).replace('SLIDES',slides),encoding='utf-8')
