#!/usr/bin/env python3
"""Render the frozen melon allocation from local data and generated materials.
Run through capture_render.py for every complete rendering attempt.
"""
from pathlib import Path
import json
import math
import hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / 'data.json').read_text(encoding='utf-8'))
BRIEF = json.loads((ROOT / 'brief.json').read_text(encoding='utf-8'))
W, H = BRIEF['width'], BRIEF['height']
S = 2
WS, HS = W*S, H*S
R, YS = 285.0, 0.96
LEFT, RIGHT = (380.0, 1100.0), (990.0, 1100.0)
DEPTH = 18.0
INK = (55, 65, 48)
MUTED = (102, 110, 89)
LINE = (118, 126, 105)
FONT = ROOT / 'assets/fonts'
SURFACE = Image.open(ROOT / 'assets/melon_surface.png').convert('RGB')
FLESH = Image.open(ROOT / 'assets/flesh_surface.png').convert('RGB')
CATS = DATA['categories']
VALS = DATA['series'][0]['values']
assert CATS == ['Category A', 'Category B', 'Category C', 'Category D', 'Category E', 'Category F']
assert sum(VALS) == 100 and VALS[0] == DATA['left_capacity'] == DATA['right_capacity']


def save_json(rel, value):
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def point(cx, cy, angle, radius=R):
    a = math.radians(angle)
    return (cx + radius*math.cos(a), cy + YS*radius*math.sin(a))


def make_piece(cat, value, center, start, sweep, distance=0):
    mid = start+sweep/2
    tx = distance*math.cos(math.radians(mid))
    ty = YS*distance*math.sin(math.radians(mid))
    cx, cy = center[0]+tx, center[1]+ty
    x0, y0 = math.floor(cx-R-3), math.floor(cy-YS*R-3)
    x1, y1 = math.ceil(cx+R+3), math.ceil(cy+YS*R+DEPTH+3)
    pw, ph = (x1-x0)*S, (y1-y0)*S
    x = (np.arange(pw, dtype=np.float64)+0.5)/S+x0
    y = (np.arange(ph, dtype=np.float64)+0.5)/S+y0
    u, v = (x[None, :]-cx)/R, (y[:, None]-cy)/(R*YS)
    rho = u*u+v*v
    deg = np.rad2deg(np.arctan2(v, u))
    in_angle = ((deg-start) % 360) < sweep if sweep < 360 else np.ones((ph, pw), dtype=bool)
    selected = (rho <= 1) & in_angle
    mask = Image.fromarray((selected*255).astype('uint8'))
    n = SURFACE.width
    # A single UV transform for every face: scale y by exactly 0.96, translate only.
    uv = (n/(2*R*S), 0, n*(x0-cx+R)/(2*R),
          0, n/(2*R*YS*S), n*(y0-cy+R*YS)/(2*R*YS))
    texture = SURFACE.transform((pw, ph), Image.Transform.AFFINE, uv, Image.Resampling.BICUBIC)
    angles = np.linspace(start, start+sweep, max(32, math.ceil(sweep*3)+1))
    arc = [point(cx, cy, float(a)) for a in angles]
    boundary = arc if sweep == 360 else [(cx, cy)]+arc
    return dict(cat=cat, value=value, start=start, sweep=sweep, translation=(tx, ty),
                center=(cx, cy), bbox=(x0, y0, x1, y1), mask=mask,
                texture=texture, boundary=boundary, area_pixels=int(selected.sum()),
                arc=arc, mid=mid)


# Clockwise input order B, C, D, E, F. Center the removed opening on the right.
opening = sum(VALS[2:])/DATA['right_capacity']*360
start = opening/2
pieces = [make_piece(CATS[0], VALS[0], LEFT, 0, 360)]
explode = [0, 85, 128, 95, 85]
for i, (cat, value) in enumerate(zip(CATS[1:], VALS[1:])):
    sweep = value/DATA['right_capacity']*360
    pieces.append(make_piece(cat, value, RIGHT, start, sweep, explode[i]))
    start += sweep

# A quiet warm studio backdrop; no external image is used.
yy, xx = np.mgrid[0:HS, 0:WS].astype(np.float32)
light = np.exp(-(((xx/S-600)/1150)**2+((yy/S-830)/1500)**2))
rng = np.random.default_rng(24)
noise = rng.normal(0, 0.23, (HS, WS)).astype(np.float32)
bg = np.empty((HS, WS, 3), np.uint8)
for k, base in enumerate((230, 232, 219)):
    bg[:, :, k] = np.clip(base+7*light+noise, 0, 255)
canvas = Image.fromarray(bg).convert('RGBA')
del bg, xx, yy, light, noise

# Soft shadows and slight contact shadows lie underneath every slab.
shadow_wide = Image.new('L', (WS, HS), 0)
shadow_close = Image.new('L', (WS, HS), 0)
for p in pieces:
    x0, y0, _, _ = p['bbox']
    shadow_wide.paste(p['mask'], (round((x0+8)*S), round((y0+25)*S)))
    shadow_close.paste(p['mask'], (round((x0+2)*S), round((y0+18)*S)))
for source, blur, opacity in [(shadow_wide, 20, 0.22), (shadow_close, 5, 0.19)]:
    alpha = source.filter(ImageFilter.GaussianBlur(blur*S)).point(lambda v: int(v*opacity))
    layer = Image.new('RGBA', (WS, HS), (57, 63, 34, 0))
    layer.putalpha(alpha)
    canvas.alpha_composite(layer)


def side_layer(p):
    x0, y0, _, _ = p['bbox']
    size = p['mask'].size
    layer = Image.new('RGBA', size, (0, 0, 0, 0))
    # The curved outer wall carries the same generated corky rind texture.
    for step in range(round(DEPTH*S), 0, -1):
        tone = 0.57+0.24*(1-step/(DEPTH*S))
        rind = ImageEnhance.Brightness(p['texture']).enhance(tone).convert('RGBA')
        rind.putalpha(p['mask'])
        layer.alpha_composite(rind, (0, step))
    if p['sweep'] < 360:
        # Only faces whose outward normal faces the extrusion direction are visible.
        radial_edges = [(p['center'], p['arc'][0]), (p['arc'][-1], p['center'])]
        for p0, p1 in radial_edges:
            ex, ey = p1[0]-p0[0], p1[1]-p0[1]
            if ex >= -1e-8:
                continue
            poly = [p0, p1, (p1[0], p1[1]+DEPTH), (p0[0], p0[1]+DEPTH)]
            mask = Image.new('L', size, 0)
            draw = ImageDraw.Draw(mask)
            draw.polygon([((x-x0)*S, (y-y0)*S) for x, y in poly], fill=255)
            fw, fh = FLESH.size
            # Affine map the generated cut-flesh swatch onto this exact side quadrilateral.
            u0 = (x0-p0[0])/ex
            affine = (fw/(ex*S), 0, fw*u0,
                      -fh*ey/(DEPTH*ex*S), fh/(DEPTH*S),
                      fh*((y0-p0[1])-ey*u0)/DEPTH)
            flesh = FLESH.transform(size, Image.Transform.AFFINE, affine, Image.Resampling.BICUBIC).convert('RGBA')
            flesh.putalpha(mask)
            layer.alpha_composite(flesh)
            dd = ImageDraw.Draw(layer)
            # A fine under-rind green band makes the transition from skin to flesh legible.
            dd.line([((p0[0]-x0)*S, (p0[1]-y0+1.9)*S),
                     ((p1[0]-x0)*S, (p1[1]-y0+1.9)*S)], fill=(119, 133, 47, 235), width=round(2.5*S))
            dd.line([((p0[0]-x0)*S, (p0[1]-y0+DEPTH)*S),
                     ((p1[0]-x0)*S, (p1[1]-y0+DEPTH)*S)], fill=(154, 147, 69, 165), width=S)
    return layer

# Paint all walls before their exact top masks; no top area is subtracted for a sidewall.
side_masks = []
for p in pieces:
    layer = side_layer(p)
    side_masks.append(layer.getchannel('A'))
    canvas.alpha_composite(layer, (p['bbox'][0]*S, p['bbox'][1]*S))

# Keep a pixel-level coverage map of exactly the visible generated top surfaces.
coverage = np.zeros((HS, WS), dtype=np.uint8)
overlap_pixels = 0
for i, p in enumerate(pieces, start=1):
    x0, y0, _, _ = p['bbox']
    tex = p['texture'].convert('RGBA')
    tex.putalpha(p['mask'])
    canvas.alpha_composite(tex, (x0*S, y0*S))
    m = np.asarray(p['mask']) > 0
    ph, pw = m.shape
    roi = coverage[y0*S:y0*S+ph, x0*S:x0*S+pw]
    overlap_pixels += int(((roi>0)&m).sum())
    roi[m] = i

D = ImageDraw.Draw(canvas)
text_boxes = []
leader_records = []

def font(size, face='regular'):
    names = {'regular':'NimbusSans-Regular.otf', 'number':'NimbusSansNarrow-Regular.otf',
             'cjk':'NotoSansCJK-Regular.ttc', 'cjk_medium':'NotoSansCJK-Medium.ttc'}
    return ImageFont.truetype(str(FONT/names[face]), round(size*S))


def text(txt, x, y, size, face='regular', fill=INK, align='left', tracking=0):
    f = font(size, face)
    if tracking:
        widths = [D.textlength(c, font=f) for c in txt]
        width = sum(widths)+(len(txt)-1)*tracking*S
    else:
        width = D.textlength(txt, font=f)
    px = x*S - (width/2 if align == 'center' else width if align == 'right' else 0)
    py = y*S
    if tracking:
        begin = px
        for c, cw in zip(txt, widths):
            D.text((px, py), c, font=f, fill=fill, anchor='lt')
            px += cw+tracking*S
        b = D.textbbox((begin, py), txt, font=f, anchor='lt')
        b = (begin, b[1], begin+width, b[3])
    else:
        D.text((px, py), txt, font=f, fill=fill, anchor='lt')
        b = D.textbbox((px, py), txt, font=f, anchor='lt')
    text_boxes.append({'text':txt, 'bbox':[round(v/S, 2) for v in b]})
    return [v/S for v in b]


def line(points, color=LINE, width=1.5):
    D.line([(round(x*S), round(y*S)) for x, y in points], fill=color, width=round(width*S), joint='curve')


def label(p, cx, y, size):
    value = str(p['value'])
    f = font(size, 'number')
    pf = font(size*0.40, 'number')
    vw = D.textlength(value, font=f)/S
    pw = D.textlength('%', font=pf)/S
    total = vw+pw+3
    top = y
    h = (f.getbbox(value, anchor='lt')[3])/S
    text(value, cx-total/2, top, size, 'number')
    text('%', cx-total/2+vw+3, top+h-(pf.getbbox('%', anchor='lt')[3])/S-2, size*0.40, 'number')
    cy = top+h+14
    text(p['cat'], cx, cy, 29, align='center')
    return cy+29


def leader(cat, pts):
    line(pts, width=1.35)
    ex, ey = pts[-1]
    D.ellipse(((ex-2)*S, (ey-2)*S, (ex+2)*S, (ey+2)*S), fill=LINE)
    leader_records.append({'category':cat, 'points':[[round(x,3),round(y,3)] for x,y in pts]})

# Editorial typography, with the mandatory notice and unit set in high contrast.
text(DATA['visible_notice'], 96, 113, 23, tracking=3.0, fill=MUTED)
text(DATA['title'], 92, 208, 75, face='cjk_medium')
text(DATA['series'][0]['label'], 96, 348, 31)
text('单位：占组合总量的百分比（%）', 1390, 351, 25, face='cjk', align='right', fill=MUTED)
line([(96, 412), (1390, 412)], color=(167, 174, 153), width=1)

# All six numeric/category labels are bound to their own material region.
label(pieces[0], 349, 615, 156)
label(pieces[1], 819, 565, 156)
label(pieces[2], 1170, 524, 112)
label(pieces[3], 1320, 678, 112)
label(pieces[4], 1307, 1516, 120)
label(pieces[5], 1001, 1516, 120)

# Leaders end just outside the corresponding rim so no label ink changes data area.
a_end = point(*pieces[0]['center'], -96, R+12)
b_end = point(*pieces[1]['center'], -120, R+12)
c_end = point(*pieces[2]['center'], pieces[2]['mid'], R+12)
d_end = point(*pieces[3]['center'], pieces[3]['mid'], R+12)
e_end = point(*pieces[4]['center'], pieces[4]['mid'], R+12)
f_end = point(*pieces[5]['center'], pieces[5]['mid'], R+12)
leader(CATS[0], [(349, 792), a_end])
leader(CATS[1], [(819, 743), (819, 815), b_end])
leader(CATS[2], [(1170, 666), (c_end[0], 666), c_end])
leader(CATS[3], [(1320, 820), (1320, 873), d_end])
leader(CATS[4], [(1307, 1490), (1307, 1463), (1414, 1463), (1414, e_end[1]), e_end])
leader(CATS[5], [(1001, 1490), (1001, 1428), (f_end[0], 1428), f_end])

line([(96, 1776), (1390, 1776)], color=(167, 174, 153), width=1)
text('两个瓜面，一份总量', 96, 1825, 31, face='cjk_medium')
text('每个完整瓜面代表 50%，两面等面积。', 96, 1889, 27, face='cjk')
text('仅顶面面积编码占比；分离空隙、侧壁与投影不计入。', 96, 1938, 27, face='cjk')
text('合成数据 · 六个类别 · 按输入顺序顺时针分区', 96, 2030, 21, face='cjk', fill=MUTED)

# Persist measured, concise geometry evidence, not an independent assessment.
visible_counts = [int((coverage==i).sum()) for i in range(1, len(pieces)+1)]
denom = sum(visible_counts)
checks = []
for p, count in zip(pieces, visible_counts):
    checks.append({'category':p['cat'], 'value_percent':p['value'],
                   'start_degrees_clockwise':round(p['start']%360,6),
                   'sweep_degrees':round(p['sweep'],6),
                   'translation_pixels':[round(v,6) for v in p['translation']],
                   'top_mask_pixels_at_2x':p['area_pixels'],
                   'visible_top_pixels_at_2x':count,
                   'measured_visible_percent':round(count/denom*100,6),
                   'error_percentage_points':round(count/denom*100-p['value'],6)})
# Test the endpoint direction after inverting the shared y transform.
endpoint_checks = []
for p, rec in zip(pieces, leader_records):
    ex, ey = rec['points'][-1]
    ux, uy = ex-p['center'][0], (ey-p['center'][1])/YS
    angle = math.degrees(math.atan2(uy,ux)) % 360
    inside = p['sweep'] == 360 or (angle-p['start'])%360 < p['sweep']
    endpoint_checks.append({'category':p['cat'], 'points_to_own_sector':bool(inside),
                            'distance_from_rim_pixels':round(math.hypot(ux,uy)-R,4)})
save_json('artifacts/geometry_checks.json', {
    'canvas':[W,H], 'supersampling':S, 'base_radius_pixels':R,
    'top_y_scale':YS, 'base_face_area_pixels':math.pi*R*R*YS,
    'top_overlap_pixels_at_2x':overlap_pixels, 'categories':checks,
    'leaders':endpoint_checks, 'text_boxes':text_boxes,
    'material_use':'Generated netted melon skin fills every data top face; generated flesh fills visible radial cut walls.'
})
# A small categorical mask is a diagnostic of top areas, not another complete chart.
colors=np.array([[0,0,0],[210,65,62],[40,112,169],[35,154,124],[233,183,38],[148,90,181],[231,110,32]],dtype=np.uint8)
Image.fromarray(colors[coverage]).resize((W//2,H//2), Image.Resampling.NEAREST).save(ROOT/'artifacts/top_area_mask.png')
canvas.convert('RGB').resize((W,H), Image.Resampling.LANCZOS).save(ROOT/'final.png')
print('Rendered final.png at',W,H,'with 6 exact, non-overlapping top masks.')
