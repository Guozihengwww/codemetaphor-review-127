"""Regenerate the infographic locally from frozen data and generated coal assets.

Usage: PYTHON render.py
During the experiment, all complete renders are invoked through capture_render.py.
No network or model calls. Heights, leaders, labels and the scale are data-bound.
"""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parent
BRIEF = json.loads((ROOT / 'brief.json').read_text(encoding='utf-8'))
DATA = json.loads((ROOT / 'data.json').read_text(encoding='utf-8'))
W, H = int(BRIEF['width']), int(BRIEF['height'])
S = 2
NAVY = (16, 65, 91)
MUTED = (42, 93, 113)
BASELINE = 860.0
SCALE_HEIGHT = 420.0
AX_MIN = float(DATA['value_axis']['minimum'])
AX_MAX = float(DATA['value_axis']['maximum'])
PX_PER_UNIT = SCALE_HEIGHT / (AX_MAX - AX_MIN)
VALUES = DATA['series'][0]['values']
CATS = DATA['categories']
UNIT = DATA['series'][0]['unit']
DECIMALS = DATA['display_decimal_places']
assert len(VALUES) == len(CATS) == 10
assert AX_MIN == 0 and all(AX_MIN < v <= AX_MAX for v in VALUES)

# The selected sheet parts vary the rock arrangement, not the quantitative rule.
ASSET_IDS = [1, 2, 3, 4, 5, 6, 4, 5, 6, 3]
MIRRORS = [False, False, False, False, False, False, True, True, True, True]

def font(size, bold=False):
    fname = 'NimbusSans-Bold.otf' if bold else 'NimbusSans-Regular.otf'
    return ImageFont.truetype(str(ROOT / 'assets' / fname), round(size * S))

# A continuous cool-cyan studio surface; seeded fine grain stays reproducible.
yy, xx = np.mgrid[0:H*S, 0:W*S].astype(np.float32)
x = xx / (W*S-1)
y = yy / (H*S-1)
light = np.exp(-((x-0.22)**2 / 0.44 + (y-0.40)**2 / 0.60))
floor = np.exp(-((y-0.77)/0.22)**2)
noise = np.random.default_rng(913).normal(0, 0.42, (H*S, W*S)).astype(np.float32)
bg = np.empty((H*S, W*S, 3), dtype=np.uint8)
for k, (base, glow, flr) in enumerate([(145, 34, 6), (203, 24, 6), (216, 20, 4)]):
    bg[:,:,k] = np.clip(base + glow*light + flr*floor + noise, 0, 255).astype(np.uint8)
im = Image.fromarray(bg).convert('RGBA')
d = ImageDraw.Draw(im)
text_records = []

def txt(text, xy, size, bold=False, fill=NAVY, anchor='lt', role='label'):
    f = font(size, bold)
    pos = (round(xy[0]*S), round(xy[1]*S))
    d.text(pos, text, font=f, fill=fill, anchor=anchor)
    bbox = d.textbbox(pos, text, font=f, anchor=anchor)
    text_records.append({'text': text, 'role': role, 'bbox_xyxy': [round(v/S, 2) for v in bbox]})

def line(points, fill, width=1):
    d.line([(round(px*S), round(py*S)) for px,py in points], fill=fill, width=max(1,round(width*S)))

# Title, provenance cue, and succinct reading key.
txt('Solid-fuel import shares', (92, 91), 80, bold=True, role='title')
txt('By fictional partner · share of total imported solid-fuel mass', (95, 190), 31, role='subtitle')
d.rounded_rectangle((1546*S, 104*S, 1908*S, 164*S), radius=6*S, fill=NAVY)
txt('SYNTHETIC DATA', (1727, 135), 26, bold=True, fill=(212,242,244), anchor='mm', role='synthetic_marker')
txt(f'{len(CATS)} partners  /  {sum(VALUES):.{DECIMALS}f}% total', (1908, 201), 24, fill=MUTED, anchor='rt', role='total')
line([(94, 257), (1908, 257)], (102,165,183,255), 1)
txt('Each pile’s height shows its share.', (1068, 335), 28, role='encoding_key')
txt(f'Common zero · linear scale · {AX_MIN:g}–{AX_MAX:g}{UNIT}', (1068, 376), 23, fill=MUTED, role='encoding_key')

# A quiet ruler makes the shared quantitative scale unambiguous.
axis_x = 76
axis_top = BASELINE - SCALE_HEIGHT
line([(axis_x, axis_top), (axis_x, BASELINE)], (88,151,172,255), 1)
txt(UNIT, (axis_x, axis_top-30), 19, fill=MUTED, anchor='mm', role='axis_unit')
for tick in DATA['value_axis']['ticks']:
    py = BASELINE - (float(tick)-AX_MIN) * PX_PER_UNIT
    line([(axis_x-4,py),(axis_x+11,py)], (88,151,172,255), 1)
    txt(f'{tick:g}', (axis_x-13,py), 19, fill=MUTED, anchor='rm', role='axis_tick')
line([(axis_x, BASELINE), (1943, BASELINE)], (124,181,195,255), 1)

# Object widths accommodate the physical asset shapes. Height is the only
# quantitative readout; every object gets exactly the same linear y transform.
coal = []
for aid, mirror, value in zip(ASSET_IDS, MIRRORS, VALUES):
    a = Image.open(ROOT / f'assets/coal_v2_{aid:02d}.png').convert('RGBA')
    if mirror:
        a = ImageOps.mirror(a)
    alpha = np.asarray(a)[:,:,3]
    ys, xs = np.where(alpha >= 64)
    a = a.crop((int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)))
    h = (float(value)-AX_MIN) * PX_PER_UNIT
    coal.append({'image':a,'height':h,'natural_width':h*a.width/a.height,'asset':aid,'mirror':mirror})
left, right, gap = 125.0, 1942.0, 14.0
horizontal_fit = (right-left-gap*(len(coal)-1)) / sum(a['natural_width'] for a in coal)
for a in coal:
    a['width'] = a['natural_width'] * horizontal_fit
    a['left'] = left
    left += a['width'] + gap

# Soft floor shadows are a separate layer and never contribute to coal height.
shadow = Image.new('RGBA', im.size, (0,0,0,0))
sd = ImageDraw.Draw(shadow)
for a in coal:
    cx = a['left']+a['width']/2
    rx = a['width']*.51
    ry = 4 + a['width']*.010
    sd.ellipse(((cx-rx+6)*S,(BASELINE-ry*.6)*S,(cx+rx+6)*S,(BASELINE+ry*1.2)*S),fill=(34,72,81,76))
shadow = shadow.filter(ImageFilter.GaussianBlur(5*S))
im.alpha_composite(shadow)
contact = Image.new('RGBA', im.size, (0,0,0,0))
cd = ImageDraw.Draw(contact)
for a in coal:
    cx = a['left']+a['width']/2
    rx = a['width']*.47
    cd.ellipse(((cx-rx)*S,(BASELINE-4)*S,(cx+rx)*S,(BASELINE+3)*S),fill=(17,40,49,145))
contact = contact.filter(ImageFilter.GaussianBlur(1.6*S))
im.alpha_composite(contact)

geometry = []
for i, a in enumerate(coal):
    aw, ah = round(a['width']*S), round(a['height']*S)
    body = a['image'].resize((aw, ah), Image.Resampling.LANCZOS)
    arr = np.asarray(body).copy()
    # Small antialias fringes are preserved; all actual rock detail is generated.
    arr[:,:,3] = np.where(arr[:,:,3] < 12, 0, arr[:,:,3])
    body = Image.fromarray(arr)
    px = round(a['left']*S)
    py = round(BASELINE*S)-ah
    im.alpha_composite(body,(px,py))
    material = arr[:,:,3] >= 64
    ys, xs = np.where(material)
    top_rows = material[:min(5, ah)]
    apex_samples = np.where(top_rows)[1]
    if not len(apex_samples):
        apex_samples = xs[ys == ys.min()]
    apex = (px+float(np.median(apex_samples)))/S
    bbox = [(px+int(xs.min()))/S,(py+int(ys.min()))/S,(px+int(xs.max()+1))/S,(py+int(ys.max()+1))/S]
    geometry.append({'category':CATS[i],'value':VALUES[i],'asset':f"assets/coal_v2_{a['asset']:02d}.png",'mirrored':a['mirror'],
        'target_height_px':a['height'],'image_box_xywh':[px/S,py/S,aw/S,ah/S],
        'rendered_alpha64_bbox_xyxy':bbox,'apex_x':apex,'baseline_y':BASELINE,
        'height_from_common_zero_alpha64_px':BASELINE-bbox[1],
        'alpha64_bottom_y':bbox[3]})

# Typography and leaders are drawn after composition, all values from data.json.
d = ImageDraw.Draw(im)
for i, g in enumerate(geometry):
    top = BASELINE - g['target_height_px']
    apex = g['apex_x']
    label = f"{float(VALUES[i]):.{DECIMALS}f}{UNIT}"
    txt(label,(apex,top-81),46,anchor='mm',role='value')
    line([(apex,top-49),(apex,top-14)],NAVY,1.7)
    box = g['image_box_xywh']
    cx = box[0]+box[2]/2
    txt(CATS[i],(cx,913),25,anchor='mm',role='category')

# Footer remains on the same uninterrupted cyan surface.
line([(94, 1001), (1908, 1001)], (102,165,183,255), 1)
txt('Source: synthetic study data · fictional partners', (95,1030),25,role='source')
txt('Coal height encodes percentage; shadows and pile width carry no values.', (95,1069),22,fill=MUTED,role='method_note')
txt('ILLUSTRATIVE DATA', (1908,1042),19,bold=True,fill=MUTED,anchor='rt',role='footer_marker')

# Supersampling is reduced only once. No untracked complete preview is produced.
final = im.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
final.save(ROOT / 'final.png', optimize=True)
(ROOT / 'artifacts').mkdir(exist_ok=True)
report = {'canvas':[W,H], 'baseline_y':BASELINE,'axis':[AX_MIN,AX_MAX],
          'scale_pixels_per_percentage_point':PX_PER_UNIT,'horizontal_fit':horizontal_fit,
          'geometry':geometry,'text':text_records}
(ROOT / 'artifacts/render_geometry.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(f'Rendered {W}×{H}; {len(CATS)} coal heaps; {PX_PER_UNIT:g} px per percentage point.')
