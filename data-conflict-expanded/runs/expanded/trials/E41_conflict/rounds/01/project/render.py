"""Offline toilet-paper infographic. Usage: PYTHON render.py (capture wrapper required for this study)."""
from pathlib import Path
import json
import math
import hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
S = 4
RESAMPLE = Image.Resampling.LANCZOS

def read_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))

inputs = read_json('input.json')
brief = read_json(inputs['brief'])
data = read_json(inputs['data'])
core = read_json(inputs['critical_metaphor_contract'])
# All frozen local inputs remain part of the offline project. Reference pixels are never composited.
with Image.open(ROOT / inputs['reference']) as reference:
    reference.verify()
parts = read_json('assets/parts.json')
W, H = brief['width'], brief['height']
assert (W, H) == (2000, 1978)
assert data['value_axis'] == {'minimum': 0, 'maximum': 100, 'scale': 'linear'}
assert data['series'][0]['id'] == 'annual_use'
assert len(data['series']) == 1
cats = data['categories']
series = data['series'][0]
values = series['values']
assert len(cats) == len(values) == 10
assert all(0 < v <= 100 for v in values)

canvas = Image.new('RGB', (W*S, H*S), 'white')
draw = ImageDraw.Draw(canvas)
INK = '#28383D'
GREY = '#637176'
LIGHT = '#D7DFDF'
PALETTE = parts['palette']
FONT = ROOT / 'assets/fonts/NimbusSans-Regular.otf'
BOLD = ROOT / 'assets/fonts/NimbusSans-Bold.otf'
font_cache = {}
text_records = []

def font(n, bold=False):
    key = (n, bold)
    if key not in font_cache:
        font_cache[key] = ImageFont.truetype(str(BOLD if bold else FONT), round(n*S))
    return font_cache[key]

def text(x, y, value, size=28, fill=INK, bold=False, anchor='lt'):
    f = font(size, bold)
    draw.text((x*S, y*S), value, font=f, fill=fill, anchor=anchor)
    box = draw.textbbox((x*S, y*S), value, font=f, anchor=anchor)
    text_records.append({'text': value, 'box': [round(q/S, 2) for q in box]})

def tracked(x, y, value, size=20, tracking=2.0, fill=GREY, bold=False):
    f = font(size, bold)
    total = sum(draw.textlength(ch, font=f)/S for ch in value) + tracking*(len(value)-1)
    text_records.append({'text': value, 'box': [x,y,x+total,y+size]})
    for ch in value:
        draw.text((x*S,y*S),ch,font=f,fill=fill,anchor='lt')
        x += draw.textlength(ch,font=f)/S + tracking

def line(points, fill, width=1):
    draw.line([(round(x*S),round(y*S)) for x,y in points], fill=fill, width=max(1,round(width*S)), joint='curve')

def dashed(x1, x2, y, fill, width=1.0, dash=7, gap=7):
    x = x1
    while x < x2:
        line([(x,y),(min(x+dash,x2),y)], fill, width)
        x += dash+gap

def fmt(v):
    return format(v, 'g')

# Hierarchy and semantic scope.
draw.rounded_rectangle((108*S,82*S,410*S,128*S), radius=23*S, fill='#F6E9DF')
tracked(134, 96, data['visible_notice'], 19, 1.6, '#8C5435', True)
text(106, 164, data['title'].removeprefix('SYNTHETIC '), 126, INK, True)
text(110, 316, f"{series['label']}  ·  {series['unit']}", 33, GREY)
text(1924, 324, 'Ten synthetic population groups', 27, GREY, anchor='rt')
line([(110,379),(1924,379)], '#BBC8C9', 1.1)
tracked(210, 421, 'SYNTHETIC POPULATION GROUPS', 20, 1.8, GREY, True)
text(1924, 421, 'Same roll size · different hanging lengths', 22, GREY, anchor='rt')

# The ONLY value mapping: common zero to the average serrated-edge height.
Y0 = 606.0
AXIS_LENGTH = 1120.0
PPU = AXIS_LENGTH / 100.0
HEAD_WIDTH = 152.0
HEAD_HEIGHT = HEAD_WIDTH * parts['head_size'][1] / parts['head_size'][0]
X0, STEP = 210.0, 174.0
LEFT_OFFSET = HEAD_WIDTH * parts['strip_left_in_head'] / parts['head_size'][0]
RIGHT_OFFSET = HEAD_WIDTH * parts['strip_right_in_head'] / parts['head_size'][0]

# A quiet downward scale supports comparison without replacing the object chart.
text(143, 554, 'ROLLS', 17, GREY, anchor='rt')
line([(153,Y0),(153,Y0+AXIS_LENGTH)], '#CED8D8', 1)
for tick in range(0, 101, 20):
    yy = Y0 + tick*PPU
    text(132, yy, str(tick), 23, GREY, anchor='rm')
    line([(146,yy),(160,yy)], '#AABABB', 1.1)
    if tick == 0:
        dashed(165, 1924, yy, '#A9B7B8', 1.0, 5, 7)
    else:
        line([(166, yy),(1924,yy)], '#ECF0F0', 0.7)

texture = Image.open(ROOT / 'assets/paper_texture.png').convert('RGB')
geometry = []
mask_dir = ROOT / 'artifacts/masks'
mask_dir.mkdir(parents=True, exist_ok=True)

for i, (cat, value, color) in enumerate(zip(cats, values, PALETTE)):
    hx = X0 + i*STEP
    xl, xr = hx+LEFT_OFFSET, hx+RIGHT_OFFSET
    cx = (xl+xr)/2
    length = value*PPU
    end_y = Y0+length
    amplitude = min(3.4, length*0.20)
    teeth = 11
    # Each full tooth has zero mean displacement; endpoints also lie on the mean line.
    edge_points = [(xl,end_y)]
    for t in range(teeth):
        for phase, delta in [(0.25,-amplitude),(0.75,amplitude),(1,0)]:
            edge_points.append((xl+(t+phase)*(xr-xl)/teeth, end_y+delta))
    bx0, bx1 = math.floor(xl)-1, math.ceil(xr)+1
    by0, by1 = int(Y0), math.ceil(end_y+amplitude+2)
    pw, ph = (bx1-bx0)*S, (by1-by0)*S
    xx = np.arange(pw)/S + bx0 + 0.5/S
    ee = np.interp(xx, [p[0] for p in edge_points], [p[1] for p in edge_points])
    cover_x = np.clip((np.minimum(xx+0.5/S,xr)-np.maximum(xx-0.5/S,xl))*S,0,1)
    ytop = by0 + np.arange(ph)/S
    coverage = np.clip((ee[None,:]-ytop[:,None])*S,0,1)*cover_x[None,:]
    alpha = Image.fromarray(np.rint(coverage*255).astype('uint8'), 'L')
    # Tile the visible generated tissue grain; data length is provided only by alpha.
    tile_h = round(texture.height/texture.width*(xr-xl)*S)
    tile = texture.resize((pw,tile_h), RESAMPLE)
    paper = Image.new('RGB',(pw,ph),'white')
    for k in range(math.ceil(ph/tile_h)):
        paper.paste(tile if k%2 == 0 else tile.transpose(Image.Transpose.FLIP_TOP_BOTTOM), (0,k*tile_h))
    # Continuation of the generated winding shadow inside, and only inside, the data strip.
    pa = np.array(paper).astype(np.float32)
    shadow = (22*np.exp(-np.arange(ph)/(10*S)))[:,None,None]
    pa = np.clip(pa-shadow,0,255).astype('uint8')
    paper = Image.fromarray(pa,'RGB')
    canvas.paste(paper,(bx0*S,by0*S),alpha)
    # Outlines and perforations are code-native. Stroke is not included in data length.
    line([(xl,Y0),(xl,end_y)], '#596467', 1.15)
    line([(xr,Y0),(xr,end_y)], '#596467', 1.15)
    line(edge_points, '#596467', 1.15)
    py = Y0+76
    perf_count = 0
    while py < end_y-15:
        dashed(xl+3.5, xr-3.5, py, '#99A3A5', 1.15, 5.5, 5)
        py += 76
        perf_count += 1
    # Head has constant geometry. Its bottom meets the exact common data origin.
    cap = Image.open(ROOT / f'assets/roll_head_{chr(65+i)}.png').convert('RGBA')
    cap = cap.resize((round(HEAD_WIDTH*S),round(HEAD_HEIGHT*S)), RESAMPLE)
    cap_y = round(Y0*S)-cap.height
    canvas.paste(cap,(round(hx*S),cap_y),cap)
    text(hx+HEAD_WIDTH/2, 470, cat, 27, INK, True, 'mt')
    # Values stay explicit even when a data strip is only a fraction of a roll-head height.
    if value <= 5:
        value_y = 713.0
        line([(cx,end_y+8),(cx,value_y-13)], '#B6C3C4', 1.1)
    else:
        value_y = end_y + 29
    text(cx, value_y, fmt(value), 48, color, True, 'mt')
    # Actual raster-mask measurements, not just target coordinates.
    small_mask = alpha.resize((bx1-bx0, by1-by0), RESAMPLE)
    mask_path = f'artifacts/masks/group_{chr(65+i)}.png'
    small_mask.save(ROOT / mask_path)
    final_alpha = np.array(small_mask,dtype=float)/255
    measured_length = float(final_alpha.sum()/(xr-xl))
    subpixel_length = float(np.asarray(alpha,dtype=float).sum()/255/S/S/(xr-xl))
    geometry.append({'category':cat,'value':value,'unit':series['unit'],'head_box':[hx,cap_y/S,hx+HEAD_WIDTH,Y0],
        'strip_x':[xl,xr],'origin_y':Y0,'target_mean_edge_y':end_y,'target_length_px':length,
        'serration_amplitude_px':amplitude,'mask_mean_length_final_px':round(measured_length,6),
        'mask_mean_length_supersampled_px':round(subpixel_length,6),'mask_origin':[bx0,by0],
        'mask_path':mask_path,'value_label':fmt(value),'value_label_top':value_y,'perforations':perf_count})

# Explain the reading rule in the natural open space of the distribution.
tracked(1010, 1205, 'READING THE PAPER', 19, 1.9, '#367B7F', True)
text(1010, 1255, 'The hanging length is the measure.', 37, INK, True)
for j, label in enumerate([
    'Every strip starts at the same zero line.',
    'Read each torn edge at its average height.',
    'Roll heads and perforations do not encode values.',
]):
    text(1010, 1320+j*42, label, 26, GREY)
text(1010, 1478, '0.1 is kept at its true length — about one pixel.', 25, GREY)

# Local synthetic provenance, without pretending to be real-country statistics.
line([(110,1820),(1924,1820)], '#BBC8C9', 1.1)
text(110, 1858, 'Synthetic groups · no real population estimates', 25, INK)
text(110, 1902, 'Source: supplied data.json. All values are shown as supplied.', 22, GREY)
text(1924, 1858, 'One common linear scale: 0–100', 25, INK, anchor='rt')
text(1924, 1902, 'Rolls per person per year', 22, GREY, anchor='rt')

canvas = canvas.resize((W,H),RESAMPLE)
canvas.save(ROOT / 'final.png', 'PNG')
(ROOT / 'artifacts/geometry_actual.json').write_text(json.dumps({
    'canvas':[W,H],'mapping':{'origin_y':Y0,'axis_min':0,'axis_max':100,'axis_length_px':AXIS_LENGTH,'pixels_per_roll':PPU,'scale':'linear'},
    'measurements':geometry,'text_boxes':text_records,
    'frozen_input_sha256':{inputs[k]:hashlib.sha256((ROOT/inputs[k]).read_bytes()).hexdigest() for k in inputs},
    'measurement_note':'Mask area divided by strip width measures the mean torn-edge height at output resolution; visible edge checks are performed separately.'
},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'saved':'final.png','size':[W,H],'groups':len(geometry),'maximum_mask_error_px':max(abs(g['mask_mean_length_final_px']-g['target_length_px']) for g in geometry)}))
