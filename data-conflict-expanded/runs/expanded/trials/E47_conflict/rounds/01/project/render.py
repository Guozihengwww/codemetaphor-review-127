#!/usr/bin/env python3
"""Offline, data-bound infographic. Run from this project with Python render.py."""
from pathlib import Path
import json
import math
import hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parent

def readj(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))

def savej(path, obj):
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

INPUT = readj('input.json')
BRIEF = readj(INPUT['brief'])
DATA = readj(INPUT['data'])
CORE = readj(INPUT['critical_metaphor_contract'])
assert (ROOT / INPUT['reference']).is_file()
assert DATA['synthetic'] is True
assert DATA['value_axis'] == {'minimum': 0, 'maximum': 150, 'scale': 'linear'}
assert len(DATA['series']) == 1
SERIES = DATA['series'][0]
CITIES, VALUES = DATA['categories'], SERIES['values']
assert len(CITIES) == len(VALUES)
assert all(0 < v <= DATA['value_axis']['maximum'] for v in VALUES)
W, H = BRIEF['width'], BRIEF['height']
S = 3
BG = '#F6F0E5'
INK = '#241D2C'
MUTED = '#6B615F'
ORANGE = '#D65321'
PURPLE = '#453569'
GRID = '#DDD2BD'
PAPER = '#FFFBF3'
FONT_ROOT = ROOT / 'assets/fonts'
FONTS = {
    'serif': 'NimbusRoman-Bold.otf',
    'sans': 'NimbusSans-Regular.otf',
    'bold': 'NimbusSans-Bold.otf',
    'mono': 'DejaVuSansMono.ttf',
}
canvas = Image.new('RGB', (W*S, H*S), BG)
d = ImageDraw.Draw(canvas)


def font(size, kind='sans'):
    return ImageFont.truetype(str(FONT_ROOT / FONTS[kind]), round(size*S))


def text(txt, x, y, size=24, fill=INK, kind='sans', anchor='lt'):
    d.text((round(x*S), round(y*S)), txt, font=font(size, kind), fill=fill, anchor=anchor)


def line(points, fill=GRID, width=1):
    d.line([(round(x*S), round(y*S)) for x, y in points], fill=fill, width=max(1, round(width*S)))


def rect(box, fill, outline=None, width=1, radius=0):
    box = tuple(round(v*S) for v in box)
    if radius:
        d.rounded_rectangle(box, radius=round(radius*S), fill=fill, outline=outline, width=round(width*S))
    else:
        d.rectangle(box, fill=fill, outline=outline, width=round(width*S))


def tracking(txt, x, y, size=18, spacing=2, fill=MUTED, kind='bold'):
    f = font(size, kind)
    for c in txt:
        d.text((round(x*S), round(y*S)), c, font=f, fill=fill, anchor='lt')
        x += d.textlength(c, font=f)/S + spacing


# Editorial header: generated object and city scene, with code-set typography.
rect((0, 0, W, 611), PAPER)
rect((72, 63, 351, 108), PURPLE, radius=3)
tracking(DATA['visible_notice'], 91, 77, size=19, spacing=2.0, fill=PAPER)
text('Annual cigarette', 70, 149, 86, kind='serif')
text('equivalent', 70, 244, 108, fill=ORANGE, kind='serif')
text('exposure', 70, 358, 108, kind='serif')
text('Indirect smoking in city air.', 75, 491, 28, kind='sans')
text(f'Per resident, per year  /  {len(CITIES)} synthetic cities', 75, 535, 22, fill=MUTED)
hero = Image.open(ROOT / 'assets/hero_01.png').convert('RGBA')
hero_h = 546
hero_w = round(hero.width * hero_h / hero.height)
hero = hero.resize((hero_w*S, hero_h*S), Image.Resampling.LANCZOS)
canvas.paste(hero, (round(903*S), round(29*S)), hero)
line([(72, 610), (1256, 610)], fill=INK, width=1.3)

# Shared linear scale. Cigarettes start at x0; all physical parts fit in that span.
x0 = 270.0
plot_width = 980.0
axis_max = DATA['value_axis']['maximum']
scale = plot_width / axis_max
first_y, row_step = 828, 80
row_y = [first_y + i*row_step for i in range(len(CITIES))]
mark_h = 38
plot_top, plot_bottom = 775, row_y[-1] + 50
text(SERIES['label'], 72, 650, 30, kind='bold')
text(SERIES['unit'], 72, 694, 23, fill=MUTED)
tracking('CITY', 73, 748, size=17, spacing=2)
text('0–150 · linear scale', 1250, 699, 18, fill=MUTED, anchor='rt')
for tick in range(0, int(axis_max)+1, 25):
    x = x0 + tick*scale
    text(str(tick), x, 748, 19, fill=MUTED, anchor='mt')
    if tick:
        line([(x, plot_top), (x, plot_bottom)], fill=GRID, width=0.65)
    else:
        # Leave the mark bands clear so tiny objects do not merge with the axis.
        last = plot_top
        for yy in row_y:
            line([(x, last), (x, yy - mark_h/2 - 9)], fill='#B7A98F', width=1.0)
            last = yy + mark_h/2 + 9
        line([(x, last), (x, plot_bottom)], fill='#B7A98F', width=1.0)

labels = []
for city, value, yy in zip(CITIES, VALUES, row_y):
    length = value*scale
    text(city, 73, yy-15, 30, kind='serif')
    label_x = max(x0 + length + 17, x0 + 92)
    label = str(value)
    text(label, label_x, yy-10, 25, kind='bold')
    labels.append({'city':city, 'value_text':label, 'x':label_x, 'y':yy-10})

# Data-derived annotation occupies open chart space without adding another scale.
min_i, max_i = int(np.argmin(VALUES)), int(np.argmax(VALUES))
ratio = VALUES[max_i]/VALUES[min_i]
call_x, call_y = 839, 1297
rect((call_x-23, call_y-27, 1239, call_y+300), BG)
line([(call_x, call_y), (call_x+61, call_y)], fill=ORANGE, width=3)
tracking('RANGE IN THIS DATASET', call_x, call_y+22, size=14, spacing=1.6)
text(f'{ratio:g}×', call_x-3, call_y+58, 77, kind='serif', fill=ORANGE)
text(f'{CITIES[max_i]} relative to {CITIES[min_i]}', call_x, call_y+150, 22, kind='bold')
text('Small values keep their true lengths.', call_x, call_y+202, 20, fill=MUTED)
text('No minimum length is imposed.', call_x, call_y+232, 20, fill=MUTED)
text('Read the labels for exact values.', call_x, call_y+262, 20, fill=MUTED)

# Footer explains the encoding and synthetic status; no real-world claim.
line([(72, 1893), (1256, 1893)], fill=INK, width=1)
tracking('HOW TO READ', 73, 1920, size=16, spacing=2, fill=ORANGE)
text('Full cigarette length—including the filter and tobacco end—encodes the annual equivalent.',
     73, 1955, 21, fill=INK)
text('All cities share the same zero and linear scale. Thickness carries no additional measure.',
     73, 1987, 21, fill=INK)
text('Synthetic data only; these are not measurements of real cities.', 73, 2031, 20, fill=MUTED)
text('SOURCE  /  supplied synthetic dataset', 1254, 2067, 15, fill=MUTED, anchor='rt')

# Downsample typography independently, then composite accurately covered data marks.
canvas = canvas.resize((W, H), Image.Resampling.LANCZOS).convert('RGBA')
source = np.asarray(Image.open(ROOT / 'assets/cigarette_01.png').convert('RGBA'), dtype=np.float32)


def cigarette(length, height):
    """Piecewise texture warp, with fractional-pixel end coverage and no min length.

    Width includes filter, paper and tobacco cap. The generated opaque centerline
    is mapped from source x=3..1504 so transparent source padding adds no length.
    """
    ww, hh = math.ceil(length), int(height)
    ssx, ssy = 16, 6
    xx = (np.arange(ww*ssx, dtype=np.float32)+0.5)/ssx
    yy = (np.arange(hh*ssy, dtype=np.float32)+0.5)/ssy
    # Exact fractional coverage at the data endpoint, rather than ceil/floor sizing.
    coverage = np.clip(length*ssx - np.arange(ww*ssx), 0, 1).astype(np.float32)
    x = np.minimum(xx, max(0, length-1e-6))
    f = min(152.0, length*0.27)
    cap = min(12.0, length*0.13)
    body = length-f-cap
    sx = np.where(x < f,
                  3 + x/f*(410-3),
                  np.where(x < length-cap,
                           410 + (x-f)/body*(1464-410),
                           1464 + (x-(length-cap))/cap*(1504-1464)))
    sy = 1 + yy/height*(source.shape[0]-3)
    xs, ys = np.meshgrid(sx, sy)
    channels = [map_coordinates(source[:,:,k], [ys, xs], order=1, mode='nearest') for k in range(4)]
    rgba = np.stack(channels, axis=-1)
    # Opaque generated material remains opaque; preserve silhouette antialiasing.
    alpha = np.clip(rgba[:,:,3]/245.0, 0, 1)*coverage[None,:]
    rgb_premul = rgba[:,:,:3]*alpha[:,:,None]
    a = alpha.reshape(hh, ssy, ww, ssx).mean(axis=(1,3))
    premul = rgb_premul.reshape(hh, ssy, ww, ssx, 3).mean(axis=(1,3))
    rgb = np.divide(premul, a[:,:,None], out=np.zeros_like(premul), where=a[:,:,None]>0)
    out = np.dstack((rgb, a*255)).round().clip(0,255).astype('uint8')
    return Image.fromarray(out, 'RGBA'), f, body, cap


metrics = []
mark_dir = ROOT / 'artifacts/mark_crops'
mark_dir.mkdir(parents=True, exist_ok=True)
for i, (city, value, yy) in enumerate(zip(CITIES, VALUES, row_y)):
    length = value*scale
    mark, f, body, cap = cigarette(length, mark_h)
    y = round(yy-mark_h/2)
    box = (round(x0), y, round(x0)+mark.width, y+mark.height)
    before = np.array(canvas.crop(box).convert('RGB'), dtype=np.int16)
    canvas.alpha_composite(mark, (round(x0), y))
    after = np.array(canvas.crop(box).convert('RGB'), dtype=np.int16)
    delta = np.max(np.abs(after-before), axis=2)
    alpha = np.asarray(mark)[:,:,3]/255
    center = alpha[mark_h//2]
    visible = np.any(delta >= 3, axis=0)
    occupied = np.where(visible)[0]
    mid50 = np.where(center >= 0.5)[0]
    metrics.append({
        'category':city, 'value':value, 'target_length_px':length,
        'rendered_center_alpha_integral_px':float(center.sum()),
        'alpha_integral_relative_error':float(abs(center.sum()-length)/length),
        'visible_color_difference_threshold':3,
        'visible_column_count':int(visible.sum()),
        'visible_pixel_bounds_x': [int(x0+occupied[0]), int(x0+occupied[-1]+1)] if occupied.size else None,
        'half_opacity_column_count':int(len(mid50)),
        'component_lengths_px':{'filter':f, 'paper':body, 'tobacco_end':cap},
        'start_x':x0, 'endpoint_x':x0+length, 'height_px':mark_h,
        'rgba_crop':f'artifacts/mark_crops/{i+1:02d}.png',
    })
    mark.save(mark_dir / f'{i+1:02d}.png')

canvas.convert('RGB').save(ROOT / 'final.png')
savej('artifacts/render_geometry.json', {
    'canvas_size':[W,H], 'scale':{'minimum':0,'maximum':axis_max,'plot_width_px':plot_width,
    'px_per_unit':scale,'start_x':x0,'type':'linear'},
    'source_parts':{'filter':[3,410],'paper':[410,1464],'tobacco':[1464,1504]},
    'rows':metrics, 'labels':labels,
    'derived_range_ratio':ratio,
    'core_dependency':INPUT['critical_metaphor_contract'],
    'data_sha256':hashlib.sha256((ROOT / INPUT['data']).read_bytes()).hexdigest(),
    'measurement_note_zh':'可见像素边界来自实际合成前后颜色差；覆盖长度来自实际输出香烟透明度中心线。小于像素的部件不能保证独立辨认。'
})
print(f'Rendered {W}×{H}; {len(metrics)} data-bound cigarettes; max alpha-integral relative error '
      f'{max(x["alpha_integral_relative_error"] for x in metrics):.6f}')
