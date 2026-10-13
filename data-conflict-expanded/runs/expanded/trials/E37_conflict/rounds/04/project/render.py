"""Offline data-bound food infographic. Run with the supplied capture_render.py wrapper."""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy import ndimage

ROOT = Path(__file__).resolve().parent
inputs = json.loads((ROOT / 'input.json').read_text(encoding='utf-8'))
brief = json.loads((ROOT / inputs['brief']).read_text(encoding='utf-8'))
data = json.loads((ROOT / inputs['data']).read_text(encoding='utf-8'))
core = json.loads((ROOT / inputs['critical_metaphor_contract']).read_text(encoding='utf-8'))
assert (ROOT / inputs['reference']).exists()
assert data['synthetic'] and data['value_axis']['scale'] == 'linear'
assert [s['id'] for s in data['series']] == ['A', 'B']
assert data['title'] == core['visible_title']
W, H = brief['width'], brief['height']
assert (W, H) == (2000, 1320)
S = 2
SW, SH = W * S, H * S
CREAM = '#fff5da'
PALE = '#edd5b9'
GREEN = '#345923'
INK = '#423d25'
LABELS = []
BADGES = []

def font(size, kind='regular'):
    return ImageFont.truetype(str(ROOT / f'assets/fonts/{kind}.otf'), round(size * S))

def xy(point):
    return tuple(round(v * S) for v in point)

def rect(box):
    return tuple(round(v * S) for v in box)

def text_at(position, text, size, fill=CREAM, kind='regular', anchor='la', track=None):
    draw = ImageDraw.Draw(canvas)
    if track is None:
        f = font(size, kind)
        box = draw.textbbox(xy(position), text, font=f, anchor=anchor)
        draw.text(xy(position), text, fill=fill, font=f, anchor=anchor)
    else:
        f = font(size, kind)
        px, py = xy(position)
        for ch in text:
            draw.text((px, py), ch, fill=fill, font=f, anchor='la')
            px += draw.textlength(ch, font=f) + track * S
        box = (position[0] * S, position[1] * S, px, (position[1] + size * 1.2) * S)
    LABELS.append({'text': text, 'bbox': [round(v / S, 2) for v in box]})

def line(points, fill, width=1):
    ImageDraw.Draw(canvas).line([xy(p) for p in points], fill=fill, width=round(width * S), joint='curve')

def layer_shadow(alpha, x, y, opacity, blur, offset):
    local = Image.new('RGBA', alpha.size, (43, 12, 9, 0))
    local.putalpha(alpha.point(lambda v: round(v * opacity)))
    # Padding keeps the shadow continuous around the perimeter.
    pad = round(blur * 4 * S)
    padded = Image.new('RGBA', (local.width + pad * 2, local.height + pad * 2))
    padded.alpha_composite(local, (pad, pad))
    padded = padded.filter(ImageFilter.GaussianBlur(blur * S))
    canvas.alpha_composite(padded, (round((x + offset[0]) * S) - pad,
                                    round((y + offset[1]) * S) - pad))

# Deterministic red tabletop; no reference pixels or external resources are used.
rng = np.random.default_rng(37)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
fine = rng.normal(0, 1.35, (H, W))
coarse = ndimage.gaussian_filter(rng.normal(0, 1, (H, W)), 24)
coarse = coarse / (coarse.std() + 1e-8)
glow = 9 * np.exp(-((xx - 650) / 1300) ** 2 - ((yy - 390) / 1200) ** 2)
shade = 5 * (xx / W) + 4 * (yy / H)
grain = fine + coarse * 0.9
bg = np.stack([178 + glow - shade + grain,
               47 + glow * .45 - shade * .45 + grain * .55,
               47 + glow * .32 - shade * .38 + grain * .55], axis=2)
canvas = Image.fromarray(np.clip(bg, 0, 255).astype('uint8')).convert('RGBA').resize((SW, SH), Image.Resampling.BICUBIC)

# One mapping is used for every data-bound layer, label and verification mask.
x0, x1, y0, y1 = 300.0, 1500.0, 315.0, 1038.0
years = np.array(data['time_points'], dtype=float)
vmin, vmax = data['value_axis']['minimum'], data['value_axis']['maximum']
def X(year): return x0 + (year - years[0]) / (years[-1] - years[0]) * (x1 - x0)
def Y(value): return y1 - (value - vmin) / (vmax - vmin) * (y1 - y0)
xs = np.array([X(t) for t in years])
av = np.array(data['series'][0]['values'])
bv = np.array(data['series'][1]['values'])
ap = np.column_stack([xs, [Y(v) for v in av]])
bp = np.column_stack([xs, [Y(v) for v in bv]])

# Title and synthetic notice.
text_at((98, 56), 'POLITICAL PARTICIPATION', 23, PALE, kind='bold', track=3.4)
text_at((92, 90), 'Voting and non voting shares', 97, CREAM, kind='display')
text_at((100, 198), 'SPD and Greens vote share and non voters share, 1990–2009', 29, CREAM)
draw = ImageDraw.Draw(canvas)
draw.rounded_rectangle(rect((1608, 70, 1904, 129)), radius=9 * S, fill=CREAM)
text_at((1756, 100), data['visible_notice'], 25, '#9b292b', 'bold', anchor='mm')
text_at((1901, 158), 'ILLUSTRATIVE FIGURES', 18, PALE, 'bold', anchor='ra')
line([(99, 254), (1901, 254)], '#cd7570', 1)

# Quiet axis cues are drawn behind the ingredients.
text_at((99, 271), 'SHARE (%)', 20, PALE, 'bold', track=1.5)
for v in [15, 20, 25, 30, 35, 40, 42]:
    y = Y(v)
    text_at((194, y), str(v), 24, PALE, anchor='rm')
    line([(209, y), (237, y)], '#d2857a', 1.2)
    line([(279, y), (1527, y)], '#bd5b54' if v not in [15, 42] else '#c57065', .6)

# The generated bread and butter are the large physical base for both series.
bread = Image.open(ROOT / 'assets/bread_butter.png').convert('RGBA')
bread = bread.resize((1390 * S, 620 * S), Image.Resampling.LANCZOS)
bread_x, bread_y = 212, 450
layer_shadow(bread.getchannel('A'), bread_x, bread_y, .40, 17, (5, 16))
layer_shadow(bread.getchannel('A'), bread_x, bread_y, .24, 4, (1, 5))
canvas.alpha_composite(bread, xy((bread_x, bread_y)))

# Cheese face: its genuine physical lower silhouette is the B polyline.
# The top is a non-quantitative boundary at about 32.5%, above every B value.
cx, cy, cw, ch = int(x0), 552, int(x1 - x0), 415
gy, gx = np.mgrid[0:ch*S, 0:cw*S + 1].astype(np.float32)
world_x = gx / S + cx
world_y = gy / S + cy
bottom = np.interp(world_x, bp[:, 0], bp[:, 1])
top = 563 + 1.2 * np.sin((world_x - cx) / 71) + .6 * np.sin((world_x - cx) / 19)
inside = (world_y >= top) & (world_y <= bottom)
mask = (inside * 255).astype(np.uint8)
texture = Image.open(ROOT / 'assets/cheese_texture.png').convert('RGB')
texture = texture.resize((cw*S + 1, ch*S), Image.Resampling.LANCZOS)
cheese_rgb = np.asarray(texture).astype(np.float32)
# Neutralize the raw slice's deeper yellow slightly, retaining photographic detail.
cheese_rgb = cheese_rgb * .89 + np.array([255, 248, 206]) * .11
distance = ndimage.distance_transform_edt(inside) / S
bevel = np.exp(-distance / 1.6)
cheese_rgb *= (1 - .12 * bevel[:, :, None])
cheese_rgba = np.dstack([np.clip(cheese_rgb, 0, 255).astype(np.uint8), mask])
cheese = Image.fromarray(cheese_rgba)
layer_shadow(cheese.getchannel('A'), cx, cy, .31, 6, (2, 7))
layer_shadow(cheese.getchannel('A'), cx, cy, .19, 1.8, (0, 2))
canvas.alpha_composite(cheese, xy((cx, cy)))

# Chive: photographic texture is swept onto a constant-width tube whose
# centreline is exactly the piecewise-linear A path. No curve interpolation.
radius = 7.2
sx = int(ap[:, 0].min() - 13)
sy = int(ap[:, 1].min() - 13)
sw = int(ap[:, 0].max() - sx + 14)
sh = int(ap[:, 1].max() - sy + 14)
syy, sxx = np.mgrid[0:sh*S, 0:sw*S].astype(np.float32)
world_sx = sxx / S + sx
world_sy = syy / S + sy
mind2 = np.full(sxx.shape, np.inf, dtype=np.float32)
best_u = np.zeros(sxx.shape, dtype=np.float32)
best_v = np.zeros(sxx.shape, dtype=np.float32)
segment_lengths = np.linalg.norm(np.diff(ap, axis=0), axis=1)
total_length = segment_lengths.sum()
travel = 0.0
for i, seglen in enumerate(segment_lengths):
    p, q = ap[i], ap[i + 1]
    dx, dy = q - p
    t = np.clip(((world_sx - p[0]) * dx + (world_sy - p[1]) * dy) / (seglen * seglen), 0, 1)
    px, py = p[0] + t * dx, p[1] + t * dy
    d2 = (world_sx - px) ** 2 + (world_sy - py) ** 2
    take = d2 < mind2
    signed = ((world_sy - py) * dx - (world_sx - px) * dy) / seglen
    best_u[take] = (travel + t[take] * seglen) / total_length
    best_v[take] = signed[take] / radius
    mind2[take] = d2[take]
    travel += seglen
stem_alpha = np.clip((radius + .25 - np.sqrt(mind2)) * S, 0, 1)
stem_src = np.asarray(Image.open(ROOT / 'assets/chive_texture.png').convert('RGB')).astype(np.float32)
sample_x = best_u * (stem_src.shape[1] - 1)
sample_y = np.clip((best_v + 1) / 2, 0, 1) * (stem_src.shape[0] - 1)
stem_rgb = np.stack([ndimage.map_coordinates(stem_src[:, :, c], [sample_y, sample_x], order=1, mode='nearest') for c in range(3)], axis=2)
# Gentle cylindrical modelling is subordinate to the generated veins and pigment.
shade = .82 + .18 * np.sqrt(np.clip(1 - best_v ** 2, 0, 1))
stem_rgb *= shade[:, :, None]
stem = Image.fromarray(np.dstack([np.clip(stem_rgb, 0, 255).astype(np.uint8), (stem_alpha * 255).astype(np.uint8)]))
layer_shadow(stem.getchannel('A'), sx, sy, .39, 2.2, (1, 3))
canvas.alpha_composite(stem, xy((sx, sy)))

# Preserve a no-label diagnostic of the food region for direct final-pixel checks.
# This is a cropped diagnostic, not a second complete chart rendering.
canvas.crop(rect((270, 320, 1530, 975))).resize((1260, 655), Image.Resampling.LANCZOS).save(ROOT / 'artifacts/food_geometry_crop.png')

def value_badge(x, y, value, series):
    value_text = f'{value:.1f}'
    width = 82 if value < 10 else 91
    draw = ImageDraw.Draw(canvas)
    if series == 'A':
        center_x = x - 40 if abs(x-x0) < .01 else x
        lo, hi = max(x0, center_x-width/2), min(x1, center_x+width/2)
        samples = [lo, hi] + [float(px) for px in xs if lo < px < hi]
        local_top = float(np.interp(samples, ap[:,0], ap[:,1]).min())
        center_y = min(y - 43, local_top - radius - 8 - 23)
        box = (center_x-width/2, center_y-23, center_x+width/2, center_y+23)
        draw.rounded_rectangle(rect(box), radius=18*S, fill='#355c29')
        text_at((center_x, center_y + 1), value_text, 30, '#fff9e8', 'bold', anchor='mm')
        # The small node marker identifies the coordinate, without changing the tube.
        draw.ellipse(rect((x-2.4, y-2.4, x+2.4, y+2.4)), fill='#f4edce')
    else:
        lo, hi = max(x0, x-width/2), min(x1, x+width/2)
        samples = [lo, hi] + [float(px) for px in xs if lo < px < hi]
        local_top = float(np.interp(samples, bp[:,0], bp[:,1]).min())
        center_y = min(y - 28, local_top - 8 - 21)
        box = (x-width/2, center_y-21, x+width/2, center_y+21)
        draw.rounded_rectangle(rect(box), radius=17*S, fill='#fff7d9')
        text_at((x, center_y + 1), value_text, 29, INK, 'bold', anchor='mm')
        draw.ellipse(rect((x-2.0, y-2.0, x+2.0, y+2.0)), fill='#fff8d7', outline='#5c562f', width=S)
    BADGES.append({'series':series,'value':float(value),'node':[float(x),float(y)],'box':[float(v) for v in box]})

for p, value in zip(ap, av): value_badge(*p, value, 'A')
for p, value in zip(bp, bv): value_badge(*p, value, 'B')

# Direct semantic labels: leaders are neutral and do not extend the data paths.
line([(1514, ap[-1,1]), (1571, ap[-1,1]), (1610, 570)], '#f1d2b8', 1.3)
line([(1512, bp[-1,1]), (1575, bp[-1,1]), (1610, 795)], '#f1d2b8', 1.3)
draw = ImageDraw.Draw(canvas)
draw.rounded_rectangle(rect((1632, 480, 1734, 487)), radius=3*S, fill='#7b9d43')
text_at((1632, 510), 'SPD and Greens', 31, CREAM, 'bold')
text_at((1632, 548), 'vote share', 31, CREAM, 'bold')
text_at((1632, 602), 'Green stem', 24, CREAM)
text_at((1632, 633), 'Read the centreline', 22, PALE)
draw.rounded_rectangle(rect((1632, 722, 1734, 729)), radius=3*S, fill='#f3df97')
text_at((1632, 752), 'Non voters', 31, CREAM, 'bold')
text_at((1632, 790), 'share', 31, CREAM, 'bold')
text_at((1632, 844), 'Pale yellow cheese', 24, CREAM)
text_at((1632, 875), 'Read the lower edge', 22, PALE)

# Category labels use true elapsed-year spacing (2002–2005 is shorter).
for year, category, x in zip(years, data['categories'], xs):
    line([(x, 1084), (x, 1094)], '#e6b1a0', 1.2)
    text_at((x, 1121), category, 29, CREAM, 'regular', anchor='mm')
text_at((1580, 1121), 'Year', 22, PALE, anchor='lm')

# A concise reading note makes the non-data boundaries unambiguous.
line([(99, 1180), (1901, 1180)], '#cd7570', 1)
text_at((100, 1207), 'SAME SCALE, TWO EDGES', 20, CREAM, 'bold', track=1.5)
text_at((100, 1243), '15–42% linear scale  ·  Years spaced by elapsed time', 23, PALE)
text_at((1055, 1207), 'Synthetic figures; not historical election results.', 23, CREAM)
text_at((1055, 1243), 'Ingredient area and thickness do not encode values.', 23, PALE)

out = canvas.convert('RGB').resize((W, H), Image.Resampling.LANCZOS)
out.save(ROOT / 'final.png')

# Record actual masks and their measured silhouettes, not just target coordinates.
mask_full = Image.new('L', (SW, SH))
mask_full.paste(cheese.getchannel('A'), xy((cx, cy)))
mask_final = mask_full.resize((W, H), Image.Resampling.LANCZOS)
mask_final.save(ROOT / 'artifacts/cheese_mask.png')
stem_full = Image.new('L', (SW, SH))
stem_full.paste(stem.getchannel('A'), xy((sx, sy)))
stem_final = stem_full.resize((W, H), Image.Resampling.LANCZOS)
stem_final.save(ROOT / 'artifacts/stem_mask.png')
cheese_np = np.asarray(mask_final)
stem_np = np.asarray(stem_final)
measurements_b = []
measurements_a = []
for i, (x, ay, by) in enumerate(zip(xs, ap[:,1], bp[:,1])):
    ix = min(int(x1)-1, max(int(x0)+1, int(round(x))))
    pix = np.where(cheese_np[:,ix] >= 128)[0]
    observed_b = float(pix[-1]) + .5
    window = np.where(stem_np[max(0,int(ay)-20):int(ay)+21,ix] >= 128)[0] + max(0,int(ay)-20)
    observed_a = float((window[0] + window[-1]) / 2)
    measurements_b.append({'category':data['categories'][i], 'sample_x':ix, 'target_y':round(float(by),3), 'observed_lower_edge_y':observed_b, 'error_px':round(observed_b-float(np.interp(ix,bp[:,0],bp[:,1])),3)})
    measurements_a.append({'category':data['categories'][i], 'sample_x':ix, 'target_y':round(float(ay),3), 'observed_vertical_midpoint_y':observed_a, 'error_px':round(observed_a-float(np.interp(ix,ap[:,0],ap[:,1])),3)})
crossings=[]
delta=av-bv
for i in range(len(delta)-1):
    if delta[i]*delta[i+1]<0:
        frac=float(-delta[i]/(delta[i+1]-delta[i]))
        yr=float(years[i]+frac*(years[i+1]-years[i]))
        val=float(av[i]+frac*(av[i+1]-av[i]))
        crossings.append({'year':round(yr,6),'value_percent':round(val,6),'pixel':[round(X(yr),3),round(Y(val),3)]})
geometry = {
    'canvas':[W,H], 'plot_bounds':[x0,y0,x1,y1],
    'axis_range':[vmin,vmax], 'x_coordinates':xs.tolist(),
    'a_points':ap.tolist(), 'b_points':bp.tolist(),
    'a_measured_mask_midpoints':measurements_a,
    'b_measured_mask_edges':measurements_b,
    'crossings':crossings, 'stem_diameter_px':radius*2,
    'cheese_top_is_non_data':True,
    'cheese_minimum_thickness_px':float(np.min(bottom-top)),
    'labels':LABELS,
    'badge_boxes':BADGES,
}
(ROOT / 'artifacts/render_geometry.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Saved final.png (2000×1320); food boundaries and all labels are data-bound.')
