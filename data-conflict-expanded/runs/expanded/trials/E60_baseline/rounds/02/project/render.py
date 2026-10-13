"""Offline, deterministic infographic renderer. Run through capture_render.py.

The generated photographs supply material only. All six top faces are clipped
by one exact area mapping: value / capacity * 360 degrees, then y *= 0.96.
Explosions are translations; walls and shadows are separate non-data layers.
"""
from pathlib import Path
import json
import math
import hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops, ImageEnhance

ROOT = Path(__file__).resolve().parent
INPUT = json.loads((ROOT / 'input.json').read_text(encoding='utf-8'))
BRIEF = json.loads((ROOT / INPUT['brief']).read_text(encoding='utf-8'))
DATA = json.loads((ROOT / INPUT['data']).read_text(encoding='utf-8'))
CORE = json.loads((ROOT / INPUT['critical_metaphor_contract']).read_text(encoding='utf-8'))
assert (ROOT / INPUT['reference']).is_file()
W, H = BRIEF['width'], BRIEF['height']
S = 2
SIZE = (W * S, H * S)
R = 278.0
YSCALE = 0.96
CENTERS = [(377.0, 1120.0), (955.0, 1120.0)]
THICKNESS = 21.0
INK = '#30352B'
SECONDARY = '#626657'
MUTED = '#858879'
GOLD = '#AA7609'
LINE = '#858979'
ART = ROOT / 'artifacts'
(ART / 'masks').mkdir(parents=True, exist_ok=True)

values = DATA['series'][0]['values']
categories = DATA['categories']
assert len(values) == len(categories) == 6
assert sum(values) == DATA['series_totals']['allocation'] == 100
assert values[0] == DATA['left_capacity'] == DATA['right_capacity'] == 50

def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def font(size, style='regular'):
    names = {'regular': 'NimbusSans-Regular.otf', 'bold': 'NimbusSans-Bold.otf', 'narrow': 'NimbusSansNarrow-Bold.otf'}
    return ImageFont.truetype(str(ROOT / 'assets/fonts' / names[style]), round(size * S))

def xy(points):
    return [(round(x * S), round(y * S)) for x, y in points]

def fmt(v):
    return f'{v:g}'

yy, xx = np.mgrid[:H, :W]
light = np.exp(-(((xx - 515) / 1060) ** 2 + ((yy - 970) / 1650) ** 2))
rng = np.random.default_rng(6001)
grain = rng.normal(0, .20, (H, W))
bg = np.stack([234 + 11 * light + grain, 234 + 10 * light + grain, 221 + 11 * light + grain], axis=-1)
canvas = Image.fromarray(np.uint8(np.clip(bg, 0, 255))).resize(SIZE, Image.Resampling.BICUBIC).convert('RGBA')
draw = ImageDraw.Draw(canvas)
text_mask = Image.new('L', SIZE, 0)
text_draw = ImageDraw.Draw(text_mask)
text_records = []

def text(txt, x, y, size, style='regular', fill=INK, align='left', tracking=0, record=True):
    f = font(size, style)
    if tracking:
        width = sum(draw.textlength(c, font=f) for c in txt) + max(0, len(txt) - 1) * tracking * S
    else:
        width = draw.textlength(txt, font=f)
    left = x * S - (width / 2 if align == 'center' else width if align == 'right' else 0)
    if tracking:
        baseline = y * S - f.getbbox(txt, anchor='ls')[1]
        for c in txt:
            draw.text((left, baseline), c, font=f, fill=fill, anchor='ls')
            text_draw.text((left, baseline), c, font=f, fill=255, anchor='ls')
            left += draw.textlength(c, font=f) + tracking * S
    else:
        draw.text((left, y * S), txt, font=f, fill=fill, anchor='lt')
        text_draw.text((left, y * S), txt, font=f, fill=255, anchor='lt')
    if record:
        text_records.append({'text': txt, 'size_px': size, 'position': [x, y], 'alignment': align})

def lines(txts, x, y, size, leading, **kw):
    for i, t in enumerate(txts):
        text(t, x, y + i * leading, size, **kw)

def wrap(txt, max_width, size=29, style='regular'):
    f = font(size, style)
    out, line = [], ''
    for word in txt.split():
        nxt = (line + ' ' + word).strip()
        if line and draw.textlength(nxt, font=f) > max_width * S:
            out.append(line)
            line = word
        else:
            line = nxt
    if line:
        out.append(line)
    return out

def percent(i, x, y, size, align='center'):
    number = fmt(values[i])
    nf, pf = font(size, 'narrow'), font(size * .45, 'narrow')
    nw, pw = draw.textlength(number, font=nf), draw.textlength('%', font=pf)
    gap = 4 * S
    total = nw + gap + pw
    left = x * S - (total / 2 if align == 'center' else total if align == 'right' else 0)
    nh = nf.getbbox(number)[3] - nf.getbbox(number)[1]
    ph = pf.getbbox('%')[3] - pf.getbbox('%')[1]
    draw.text((left, y * S), number, font=nf, fill=INK, anchor='lt')
    draw.text((left + nw + gap, y * S + nh - ph - 2 * S), '%', font=pf, fill=GOLD, anchor='lt')
    text_draw.text((left, y * S), number, font=nf, fill=255, anchor='lt')
    text_draw.text((left + nw + gap, y * S + nh - ph - 2 * S), '%', font=pf, fill=255, anchor='lt')
    text_records.append({'text': number + '%', 'category': categories[i], 'position': [x, y]})

def poly_mask(points):
    out = Image.new('L', SIZE, 0)
    ImageDraw.Draw(out).polygon(xy(points), fill=255)
    return out

def quantitative_mask(sp):
    """Analytic disk-sector coverage, sampled 4 x 4 per delivery pixel.

    Unlike inclusive polygon rasterization, this makes no outward half-pixel
    expansion of the very small sectors. No minimum sector size is applied.
    """
    cx, cy = sp['center']
    x0, x1 = math.floor((cx - R) * S) - 1, math.ceil((cx + R) * S) + 1
    y0, y1 = math.floor((cy - R * YSCALE) * S) - 1, math.ceil((cy + R * YSCALE) * S) + 1
    coverage = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    for ox in [.25, .75]:
        for oy in [.25, .75]:
            u = (np.arange(x0, x1)[None, :] + ox) / S - cx
            v = ((np.arange(y0, y1)[:, None] + oy) / S - cy) / YSCALE
            inside = u * u + v * v <= R * R
            if not sp['full']:
                angle = (np.degrees(np.arctan2(v, u)) - sp['start']) % 360
                inside &= angle < sp['end'] - sp['start']
            coverage += inside.astype(np.uint8)
    patch = Image.fromarray(np.uint8(np.rint(coverage.astype(float) * 255 / 4)))
    out = Image.new('L', SIZE, 0)
    out.paste(patch, (x0, y0))
    return out

def pt(center, angle, radius=R):
    theta = math.radians(angle)
    return (center[0] + radius * math.cos(theta), center[1] + radius * YSCALE * math.sin(theta))

def boundary(center, start, end, full=False):
    angles = np.linspace(start, end, max(20, math.ceil(abs(end - start) * 5) + 1))
    arc = [pt(center, float(a)) for a in angles]
    return arc if full else [center] + arc

# List order is exactly the frozen category order. The body runs clockwise
# through 244.8 degrees; the remaining four sectors continue around the circle.
specs = [{'i': 0, 'center': CENTERS[0], 'start': 0.0, 'end': 360.0, 'translation': [0, 0], 'full': True}]
start = 57.6
offsets = [(0, 0), (40, -50), (95, -24), (128, 16), (75, 66)]
for i, offset in enumerate(offsets, 1):
    end = start + values[i] / DATA['right_capacity'] * 360
    center = (CENTERS[1][0] + offset[0], CENTERS[1][1] + offset[1])
    specs.append({'i': i, 'center': center, 'start': start, 'end': end, 'translation': list(offset), 'full': False})
    start = end
assert abs(start - (57.6 + 360)) < 1e-9

skin = Image.open(ROOT / 'assets/skin_normalized.png').convert('RGB')
flesh = Image.open(ROOT / 'assets/flesh_texture.png').convert('RGB')
top_masks = []
wall_masks = []
for sp in specs:
    outline = boundary(sp['center'], sp['start'], sp['end'], sp['full'])
    sp['outline'] = outline
    top = quantitative_mask(sp)
    sp['mask'] = top
    top_masks.append(top)
    wall = Image.new('L', SIZE, 0)
    wall_draw = ImageDraw.Draw(wall)
    closed = outline + [outline[0]]
    for a, b in zip(closed, closed[1:]):
        wall_draw.polygon(xy([a, b, (b[0], b[1] + THICKNESS), (a[0], a[1] + THICKNESS)]), fill=255)
    wall_masks.append(wall)

# Soft object shadows precede all faces, so they never conceal any data face.
for sp, wall in zip(specs, wall_masks):
    shape = ImageChops.lighter(sp['mask'], wall)
    for blur, dx, dy, opacity in [(22, 8, 20, 42), (7, 2, 11, 51)]:
        shadow = ImageChops.offset(shape, dx * S, dy * S).filter(ImageFilter.GaussianBlur(blur * S))
        shadow = shadow.point(lambda a: round(a * opacity / 255))
        canvas.alpha_composite(Image.composite(Image.new('RGBA', SIZE, (63, 66, 37, 255)), Image.new('RGBA', SIZE), shadow))

# Each physical object is shaded independently. Texture placement uses its
# translated center and the SAME base radius and y scale as its geometric mask.
for sp, wall in zip(specs, wall_masks):
    cx, cy = sp['center']
    box = (round((cx - R) * S), round((cy - R * YSCALE) * S), round((cx + R) * S), round((cy + R * YSCALE) * S))
    tex = skin.resize((box[2] - box[0], box[3] - box[1]), Image.Resampling.LANCZOS)
    wall_tex = Image.new('RGBA', SIZE, (157, 125, 38, 255))
    outer = ImageEnhance.Brightness(tex).enhance(.70)
    wall_tex.paste(outer, (box[0], box[1] + round(THICKNESS * S)))
    canvas.paste(wall_tex, (0, 0), wall)
    if not sp['full']:
        for angle in [sp['start'], sp['end']]:
            tip = pt(sp['center'], angle)
            quad = [sp['center'], tip, (tip[0], tip[1] + THICKNESS), (cx, cy + THICKNESS)]
            cut_mask = poly_mask(quad)
            x0 = math.floor(min(p[0] for p in quad) * S)
            y0 = math.floor(min(p[1] for p in quad) * S)
            x1 = math.ceil(max(p[0] for p in quad) * S) + 1
            y1 = math.ceil(max(p[1] for p in quad) * S) + 1
            # High-frequency flesh material remains visible on the thin sides.
            patch = flesh.resize((max(1, x1 - x0), max(1, y1 - y0)), Image.Resampling.LANCZOS)
            cut = Image.new('RGBA', SIZE, (231, 214, 136, 255))
            cut.paste(patch, (x0, y0))
            canvas.paste(cut, (0, 0), cut_mask)
            side_draw = ImageDraw.Draw(canvas)
            side_draw.line(xy([(cx, cy + THICKNESS), (tip[0], tip[1] + THICKNESS)]), fill=(153, 140, 62, 255), width=2 * S)
            side_draw.line(xy([(cx, cy + 3), (tip[0], tip[1] + 3)]), fill=(232, 224, 150, 255), width=3 * S)
    top_texture = Image.new('RGBA', SIZE)
    top_texture.paste(tex, box[:2])
    canvas.paste(top_texture, (0, 0), sp['mask'])

draw = ImageDraw.Draw(canvas)

# Typographic hierarchy and a prominent synthetic-data notice.
draw.rounded_rectangle(xy([(88, 77), (370, 124)]), radius=5 * S, fill='#E5C466')
text(DATA['visible_notice'], 229, 91, 23, style='bold', align='center', tracking=1.3)
text('SIMULATED RESPONSE SHARE', 1394, 91, 19, align='right', tracking=1.8, fill=SECONDARY)
title = DATA['title'].removeprefix('SYNTHETIC ')
title_lines = title.replace(' women on panels', '\nwomen on panels').split('\n')
lines(title_lines, 88, 174, 104, 109, style='bold')
text(DATA['series'][0]['unit'].capitalize(), 92, 417, 30, fill=SECONDARY)
draw.line(xy([(88, 490), (1394, 490)]), fill='#C6CABB', width=S)

# All labels are derived from the frozen categories and values, with direct
# external leader lines; no label is burnt into the generated imagery.
label_specs = [
    (0, 332, 603, 128, 332, 720, 310, 'center'),
    (1, 797, 582, 122, 797, 695, 275, 'center'),
    (2, 1104, 589, 104, 1104, 695, 290, 'center'),
    (3, 1390, 773, 88, 1390, 857, 230, 'right'),
    (4, 1370, 1518, 100, 1370, 1617, 250, 'right'),
    (5, 946, 1518, 100, 946, 1617, 275, 'center'),
]
for i, nx, ny, ns, tx, ty, tw, align in label_specs:
    percent(i, nx, ny, ns, align=align)
    label_lines = categories[i].replace(' is ', ' is\n', 1).split('\n') if i == 5 else wrap(categories[i], tw, 29)
    lines(label_lines, tx, ty, 29, 36, align=align)

leaders = [
    [(332, 808), (332, 840)],
    [(797, 777), (797, 818), (839, 818), (839, 860)],
    [(1104, 782), (1104, 807), (1192, 807), (1192, 864)],
    [(1390, 941), (1356, 941), (1315, 960)],
    [(1370, 1492), (1418, 1492), (1418, 1111), (1380, 1111)],
    [(946, 1492), (946, 1462), (1261, 1462), (1261, 1358)],
]
for route in leaders:
    draw.line(xy(route), fill=LINE, width=2 * S, joint='curve')
    end = route[-1]
    rr = 3
    draw.ellipse(xy([(end[0] - rr, end[1] - rr), (end[0] + rr, end[1] + rr)]), fill=LINE)

text('TWO EQUAL HALVES', 100, 1525, 21, style='bold', fill=GOLD, tracking=1.8)
lines(['Together, the two complete', 'melon faces represent 100%.'], 100, 1574, 28, 37)
text('Each complete face = 50%', 100, 1673, 24, style='bold', fill=SECONDARY)

draw.line(xy([(88, 1810), (1394, 1810)]), fill='#C6CABB', width=S)
text('HOW TO READ THE MELONS', 91, 1853, 20, style='bold', tracking=1.3)
lines(['Top-surface area encodes response share.', 'The four slices are shifted outward for clarity.', 'Gaps, thickness and shadows carry no value.'], 91, 1902, 25, 36, fill=SECONDARY)
text('SYNTHETIC, NOT SURVEYED', 851, 1853, 20, style='bold', tracking=1.3)
lines(['One mutually exclusive simulated reason', 'per participant. This is not a real survey.'], 851, 1902, 25, 36, fill=SECONDARY)

# Persist exact rendered top masks and measure rasterized visible geometry.
# Mask sums are taken at delivery resolution, after the same downsampling used
# on the actual image. Sidewalls and shadows are never included in these sums.
mask_arrays = []
geometry = []
for sp in specs:
    i = sp['i']
    m = sp['mask'].resize((W, H), Image.Resampling.LANCZOS)
    arr = np.asarray(m, dtype=np.float64) / 255
    mask_arrays.append(arr)
    bbox = m.getbbox()
    m.crop(bbox).save(ART / 'masks' / f'top_{chr(65+i)}.png')
    geometry.append({'category': categories[i], 'value_percent': values[i],
                     'angle_degrees': sp['end'] - sp['start'],
                     'start_degrees_clockwise': sp['start'],
                     'end_degrees_clockwise': sp['end'],
                     'translation_px': sp['translation'], 'radius_px': R,
                     'y_scale': YSCALE, 'mask_bbox_px': list(bbox),
                     'rasterized_top_area_px2': round(float(arr.sum()), 4)})
base = geometry[0]['rasterized_top_area_px2']
stack = np.stack(mask_arrays)
text_arr = np.asarray(text_mask.resize((W, H), Image.Resampling.LANCZOS)) > 100
for i, g in enumerate(geometry):
    # Later opaque object layers would hide earlier faces. In this layout they
    # have no intersection; this is also checked against actual wall masks.
    later = np.zeros((H, W), dtype=bool)
    for j in range(i + 1, len(specs)):
        full = ImageChops.lighter(specs[j]['mask'], wall_masks[j]).resize((W, H), Image.Resampling.LANCZOS)
        later |= np.asarray(full) > 127
    g['top_occluded_by_later_object_px'] = int(np.count_nonzero((mask_arrays[i] > .5) & later))
    g['top_overlapped_by_text_px'] = int(np.count_nonzero((mask_arrays[i] > .5) & text_arr))
    g['measured_percent_of_two_base_faces'] = round(g['rasterized_top_area_px2'] / (2 * base) * 100, 5)
    g['absolute_percentage_point_error'] = round(abs(g['measured_percent_of_two_base_faces'] - values[i]), 5)
save_json(ART / 'geometry_check.json', {
    'canvas_px': [W, H], 'measurement': 'delivery-resolution alpha sums of masks actually composited',
    'base_face_area_px2': base, 'sum_measured_percent': round(sum(g['measured_percent_of_two_base_faces'] for g in geometry), 5),
    'top_mask_pairwise_overlap_px': int(np.count_nonzero((stack > .5).sum(axis=0) > 1)),
    'pieces': geometry,
    'right_reassembled_angle_sum_degrees': sum(g['angle_degrees'] for g in geometry[1:]),
    'mandatory_labels': text_records,
    'frozen_dependencies': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ['input.json','brief.json','data.json','core.json','reference.png']}
})
canvas.convert('RGB').resize((W, H), Image.Resampling.LANCZOS).save(ROOT / 'final.png')
print(f'Rendered {W} x {H}; six faces, four translated wedges; offline assets only.')
