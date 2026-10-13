"""Offline, deterministic data-bound banquet-table infographic renderer.

Run through capture_render.py for every complete rendering attempt.
All angles and mandatory labels are computed from the frozen local data.
"""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'assets'
ART = ROOT / 'artifacts'
ART.mkdir(exist_ok=True)
brief = json.loads((ROOT / 'brief.json').read_text(encoding='utf-8'))
data = json.loads((ROOT / 'data.json').read_text(encoding='utf-8'))
core = json.loads((ROOT / 'core.json').read_text(encoding='utf-8'))
inputs = json.loads((ROOT / 'input.json').read_text(encoding='utf-8'))
for dependency in inputs.values():
    assert (ROOT / dependency).is_file(), dependency

W, H = brief['width'], brief['height']
S = 2
CX, CY, R = 1030, 805, 465
values = np.array(data['series'][0]['values'], dtype=float)
total = int(data['total_count'])
assert sum(values) == total == data['series_totals']['preference']
assert len(values) == len(data['categories']) == 9
ends = np.cumsum(values) / total * 360.0
starts = np.r_[0.0, ends[:-1]]
midpoints = (starts + ends) / 2 - 90
material_names = ['ivory', 'sage', 'eucalyptus', 'rose', 'burgundy', 'navy', 'flax', 'orange', 'gold']
CREAM = '#f2edda'
MUTED = '#becbb7'
INK = '#223e2d'

def font(size, kind='sans'):
    filename = {'sans':'sans.ttf', 'bold':'sans_bold.ttf', 'serif':'serif.otf', 'cjk':'cjk.ttc'}[kind]
    return ImageFont.truetype(str(ASSETS / 'fonts' / filename), round(size * S))

def xy(x, y):
    return round(x * S), round(y * S)

def text(x, y, value, size=24, fill=CREAM, kind='sans', anchor=None):
    draw.text(xy(x, y), str(value), font=font(size, kind), fill=fill, anchor=anchor, stroke_width=0)

def tracked(x, y, value, size=20, spacing=2.0, fill=CREAM):
    f = font(size)
    for ch in value:
        draw.text(xy(x, y), ch, font=f, fill=fill)
        x += draw.textlength(ch, font=f) / S + spacing

def line(points, fill, width=1.0):
    draw.line([xy(*p) for p in points], fill=fill, width=max(1, round(width * S)), joint='curve')

def polar(radius, angle):
    a = math.radians(float(angle))
    return CX + radius * math.cos(a), CY + radius * math.sin(a)

# Quiet, tactile green backdrop. This is a procedural substrate, not data.
rng = np.random.default_rng(20261014)
yy, xx = np.mgrid[0:H*S, 0:W*S].astype(np.float32)
vignette = np.clip(((xx / S - 1040) / 1350) ** 2 + ((yy / S - 710) / 1200) ** 2, 0, 1.6)
noise = rng.normal(0, 0.75, (H*S, W*S)).astype(np.float32)
bg = np.empty((H*S, W*S, 3), dtype=np.uint8)
for c, base in enumerate((32, 65, 47)):
    bg[:, :, c] = np.clip(base + 8 * (1 - vignette) + noise, 0, 255).astype('uint8')
canvas = Image.fromarray(bg, 'RGB').convert('RGBA')
del bg, xx, yy, vignette, noise

# Circular tabletop shadow, non-data thickness and rim.
shadow = Image.new('RGBA', canvas.size)
sd = ImageDraw.Draw(shadow)
sd.ellipse((xy(CX-R-5, CY-R+15), xy(CX+R+5, CY+R+15)), fill=(0, 0, 0, 125))
shadow = shadow.filter(ImageFilter.GaussianBlur(20*S))
canvas.alpha_composite(shadow)
draw = ImageDraw.Draw(canvas)
draw.ellipse((xy(CX-R-2, CY-R+7), xy(CX+R+2, CY+R+9)), fill='#302f21')
draw.ellipse((xy(CX-R-1.3, CY-R-1.3), xy(CX+R+1.3, CY+R+1.3)), fill='#aa9564')

# One mapping: input order, clockwise from twelve. Pixel centers are evaluated
# analytically, so even the 0.6% slice has its exact angular width, without floors.
side = int(2 * R * S)
local_y, local_x = np.mgrid[0:side, 0:side].astype(np.float64)
dx, dy = (local_x + 0.5) / S - R, (local_y + 0.5) / S - R
radius = np.hypot(dx, dy)
theta = (np.degrees(np.arctan2(dy, dx)) + 90) % 360
ids = np.searchsorted(ends, theta, side='right').clip(0, len(values)-1)
inside = radius <= R
table = np.zeros((side, side, 4), dtype=np.uint8)
light = 1.015 - 0.032 * (dx + dy) / R - 0.036 * (radius / R) ** 5
materials = []
for i, name in enumerate(material_names):
    raw = np.asarray(Image.open(ASSETS / 'materials' / (name + '.png')).convert('RGB'))
    materials.append(tuple(np.median(raw.reshape(-1, 3), axis=0).astype(int)))
    # Mirrored repeat maintains a native-scale textile weave without hard tile seams.
    tx = ((local_x / S + i * 47) % (2 * raw.shape[1])).astype(int)
    ty = ((local_y / S + i * 29) % (2 * raw.shape[0])).astype(int)
    tx = np.where(tx >= raw.shape[1], 2 * raw.shape[1] - 1 - tx, tx)
    ty = np.where(ty >= raw.shape[0], 2 * raw.shape[0] - 1 - ty, ty)
    tex = np.clip(raw[ty, tx].astype(np.float32) * light[:, :, None], 0, 255).astype('uint8')
    select = inside & (ids == i)
    table[:, :, :3][select] = tex[select]
table[:, :, 3] = (inside * 255).astype('uint8')
canvas.alpha_composite(Image.fromarray(table), xy(CX-R, CY-R))
del table, local_x, local_y, tx, ty, tex, light

# Sub-pixel dark joints mark material boundaries without opening angular gaps.
seams = Image.new('RGBA', canvas.size)
seam_draw = ImageDraw.Draw(seams)
for boundary in starts:
    seam_draw.line([xy(CX, CY), xy(*polar(R-0.5, boundary-90))], fill=(34, 42, 30, 100), width=1)
canvas.alpha_composite(seams)

# Data-only raster and photographic occlusion mask support post-render checks.
geometry = Image.new('L', canvas.size, 0)
geometry.paste(Image.fromarray(np.where(inside, ids + 1, 0).astype('uint8')), xy(CX-R, CY-R))
occlusion = Image.new('L', canvas.size, 0)
placements = []

def part(name, center, width, rotation=0, shadow_opacity=95, shadow_blur=6, offset=(7, 10)):
    global canvas, occlusion
    src = Image.open(ASSETS / 'parts' / (name + '.png')).convert('RGBA')
    height = width * src.height / src.width
    sprite = src.resize((round(width*S), round(height*S)), Image.Resampling.LANCZOS)
    sprite = sprite.rotate(rotation, resample=Image.Resampling.BICUBIC, expand=True)
    px, py = round(center[0]*S - sprite.width/2), round(center[1]*S - sprite.height/2)
    alpha = sprite.getchannel('A')
    contact = Image.new('RGBA', canvas.size)
    ca = Image.new('L', canvas.size)
    ca.paste(alpha.point(lambda a: round(a * shadow_opacity / 255)), (px+round(offset[0]*S), py+round(offset[1]*S)))
    contact.putalpha(ca.filter(ImageFilter.GaussianBlur(shadow_blur*S)))
    canvas.alpha_composite(contact)
    canvas.alpha_composite(sprite, (px, py))
    single = Image.new('L', canvas.size)
    single.paste(alpha, (px, py))
    occlusion = Image.fromarray(np.maximum(np.asarray(occlusion), np.asarray(single)))
    placements.append({'asset':name, 'center':list(center), 'width_px':width, 'rotation_degrees':rotation})

# Generated objects rest on (or reach over) the quantitative tabletop.
part('gifts', (849, 537), 182, -15, shadow_opacity=100, shadow_blur=7)
part('champagne', (1178, 565), 172, 7, shadow_opacity=105, shadow_blur=5)
part('canapes', (1250, 937), 207, -17, shadow_opacity=90, shadow_blur=5)
part('dessert', (1160, 1090), 142, -8, shadow_opacity=85, shadow_blur=4)
part('guest_black', (641, 416), 235, 45, shadow_opacity=125, shadow_blur=9, offset=(10, 13))
part('guest_red', (480, 805), 237, 90, shadow_opacity=125, shadow_blur=9, offset=(10, 13))

draw = ImageDraw.Draw(canvas)

# Typographic hierarchy and statistical scope.
tracked(82, 45, data['visible_notice'], 19, 2.6, '#e6c46b')
tracked(1580, 48, 'HOLIDAY GIVING', 18, 1.8, MUTED)
text(77, 94, 'Gift budget preferences', 76, CREAM, 'serif')
text(82, 189, data['topic_zh'], 24, CREAM, 'cjk')
text(328, 197, 'One mutually exclusive choice per simulated participant.', 21, MUTED)
text(1580, 87, str(total), 68, CREAM, 'serif')
text(1730, 112, 'simulated', 22, CREAM)
text(1730, 142, 'responses', 22, CREAM)
text(1580, 195, '9 synthetic budget options', 20, MUTED)
line([(82, 244), (1918, 244)], '#657c64', 0.8)

# Central, large gold slice: full semantic label bound to its data.
text(693, 715, 'Price not', 37, INK, 'serif')
text(693, 756, 'a factor', 37, INK, 'serif')
text(686, 805, f'{values[8]/total*100:.1f}%', 74, INK, 'serif')
text(695, 892, f'{int(values[8])} responses', 23, INK)

# Fine leaders meet the actual slice mid-angles at the shared circular rim.
right_tops = [341, 549, 733, 939, 1121]
right_elbows = [(1390, 369), (1525, 577), (1527, 761), (1518, 967), (1422, 1149)]
left_tops = {5:1207, 6:1091, 7:974}

def metric_line(x, y, i, left=False):
    percent = f'{values[i] / total * 100:.1f}%'
    text(x, y, percent, 46, CREAM, 'serif')
    length = draw.textlength(percent, font=font(46, 'serif')) / S
    text(x+length+17, y+19, f'{int(values[i])} responses', 20, MUTED)

for i, category in enumerate(data['categories'][:-1]):
    color = tuple(int(v) for v in materials[i])
    anchor = polar(R-3, midpoints[i])
    if i < 5:
        x, top = 1580, right_tops[i]
        elbow = right_elbows[i]
        line([anchor, elbow, (1549, top+28)], '#adbca1', 1.2)
        draw.ellipse((xy(1560, top+22), xy(1570, top+32)), fill=color)
        parent, sub = category.split(' / ')
        text(x, top, parent, 30)
        text(x, top+38, sub, 22, MUTED)
        metric_line(x-1, top+70, i)
    else:
        x, top = 82, left_tops[i]
        if i == 5:
            outside = polar(R+45, midpoints[i])
            path = [(423, top+62), (783, top+62), outside, anchor]
        elif i == 6:
            outside = polar(R+48, midpoints[i])
            path = [(423, top+62), (580, top+62), outside, anchor]
        else:
            outside = polar(R+38, midpoints[i])
            path = [(423, top+62), (546, top+62), outside, anchor]
        line(path, '#adbca1', 1.2)
        draw.ellipse((xy(82, top+10), xy(92, top+20)), fill=color)
        text(x+22, top-1, category, 30)
        metric_line(x+21, top+41, i)
    # A small endpoint sits just outside the data disk, never enlarging a slice.
    end = polar(R+3, midpoints[i])
    draw.ellipse((xy(end[0]-2, end[1]-2), xy(end[0]+2, end[1]+2)), fill=CREAM)

# Sparse explanatory copy is outside the physical table and labels.
text(84, 315, 'A place for every', 31, '#d7dfcc', 'serif')
text(84, 352, 'gift-giving preference.', 31, '#d7dfcc', 'serif')
line([(82, 1329), (1918, 1329)], '#657c64', 0.8)
text(82, 1354, 'Read clockwise from 12 o’clock • Entire tabletop = 500 responses (100%).', 20, MUTED)
text(1250, 1354, 'Synthetic categories and participants; not a survey.', 20, MUTED)

final = canvas.convert('RGB').resize((W, H), Image.Resampling.LANCZOS)
final.save(ROOT / 'final.png')
geo_final = geometry.resize((W, H), Image.Resampling.NEAREST)
occ_final = occlusion.resize((W, H), Image.Resampling.LANCZOS)
geo_final.save(ART / 'sector_ids.png')
occ_final.save(ART / 'photographic_occlusion.png')

gid = np.asarray(geo_final)
occ = np.asarray(occ_final)
rows = []
for i, label in enumerate(data['categories']):
    mask = gid == i+1
    count = int(mask.sum())
    rows.append({'category':label, 'count':int(values[i]), 'percent':round(values[i]/total*100, 1),
                 'start_degrees_clockwise_from_top':round(float(starts[i]), 6),
                 'end_degrees_clockwise_from_top':round(float(ends[i]), 6),
                 'angle_degrees':round(float(ends[i]-starts[i]), 6),
                 'raster_area_px':count,
                 'raster_disk_share_percent':round(count / int((gid>0).sum()) * 100, 4),
                 'uncovered_by_photographic_parts_percent':round(float((mask & (occ < 20)).sum()) / count * 100, 2)})
evidence = {'canvas':[W,H], 'table_center':[CX,CY], 'table_radius_px':R,
            'mapping':'clockwise from 12 o’clock; angle = count / total * 360; no minimum size',
            'values_source':'data.json', 'total':total, 'category_checks':rows,
            'parts':placements, 'smallest_outer_arc_px':round(R * values.min()/total * 2*math.pi, 3)}
(ART / 'render_geometry.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(f'Rendered {W} × {H}: {len(values)} categories, {total} mutually exclusive simulated responses.')
