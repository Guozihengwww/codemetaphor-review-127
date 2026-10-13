"""Offline infographic renderer. Run with Python render.py from this directory."""
from pathlib import Path
import hashlib
import json
import math
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
S = 2
PAPER = '#FAF5E9'
TEAL = '#40C4BA'
INK = '#123F40'
SOFT_INK = '#226665'
IVORY = '#FFF7E5'
GRID = '#8CDACF'
RED = '#A23426'


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


inputs = read_json('input.json')
brief = read_json(inputs['brief'])
data = read_json(inputs['data'])
core = read_json(inputs['critical_metaphor_contract'])
assert core['topic_zh'] == data['topic_zh']
assert data['value_axis'] == {'minimum': 0, 'maximum': 150, 'scale': 'linear'}
assert len(data['series']) == 1
categories = data['categories']
values = data['series'][0]['values']
assert len(categories) == len(values) == 20
assert abs(sum(values) - data['series_totals']['annual_consumption']) < 1e-8
W, H = brief['width'], brief['height']
canvas = Image.new('RGBA', (W * S, H * S), PAPER)
draw = ImageDraw.Draw(canvas)
labels = []
objects = []


def font(size, style='regular'):
    filename = {'regular': 'font_regular.otf', 'bold': 'font_bold.otf',
                'condensed': 'font_condensed.ttf', 'condensed_bold': 'font_condensed_bold.ttf'}[style]
    return ImageFont.truetype(str(ROOT / 'assets' / filename), round(size * S))


def text(x, y, value, size=24, fill=INK, style='regular', anchor='lt', role='annotation'):
    f = font(size, style)
    position = (round(x * S), round(y * S))
    draw.text(position, str(value), font=f, fill=fill, anchor=anchor)
    b = draw.textbbox(position, str(value), font=f, anchor=anchor)
    labels.append({'text': str(value), 'role': role, 'bbox_xyxy': [round(t / S, 2) for t in b]})


def rule(x1, y1, x2, y2, color=GRID, width=1):
    draw.line((round(x1 * S), round(y1 * S), round(x2 * S), round(y2 * S)), fill=color, width=max(1, round(width * S)))


def dashed(x1, x2, y, color, width=2.5, dash=13, gap=10):
    x = x1
    while x < x2:
        rule(x, y, min(x + dash, x2), y, color, width)
        x += dash + gap


def paste_asset(name, cx, top, width=None, height=None):
    asset = Image.open(ROOT / 'assets' / name).convert('RGBA')
    if height is None:
        height = width * asset.height / asset.width
    if width is None:
        width = height * asset.width / asset.height
    dest = asset.resize((round(width * S), round(height * S)), Image.Resampling.LANCZOS)
    left = round(cx * S - dest.width / 2)
    y = round(top * S)
    canvas.alpha_composite(dest, (left, y))
    return [left / S, y / S, (left + dest.width) / S, (y + dest.height) / S]


# Header. Synthetic status is deliberately prominent and distinct from the metric.
draw.rounded_rectangle((80 * S, 42 * S, 324 * S, 84 * S), radius=5 * S, fill=INK)
text(202, 54, data['visible_notice'], 22, PAPER, 'bold', 'mt', 'synthetic_notice')
text(348, 55, 'VOLUME / PERSON / YEAR', 21, SOFT_INK, 'regular')
text(80, 103, 'Annual beer consumption', 98, INK, 'bold', role='title')
text(84, 212, data['series'][0]['label'], 30, INK, role='series')
text(578, 214, data['series'][0]['unit'], 26, SOFT_INK, role='unit')
rule(1675, 44, 1675, 226, '#D2D8C9', 1.3)
text(1720, 64, '20', 88, INK, 'bold', role='category_count')
text(1724, 159, 'REGIONS', 22, INK, 'bold')
text(1724, 194, '+ 1 comparison', 23, SOFT_INK)

# One continuous turquoise field, with a common linear vertical scale.
draw.rectangle((40 * S, 272 * S, 1960 * S, 1350 * S), fill=TEAL)
baseline = 1225.0
plot_top = 325.0
plot_height = baseline - plot_top
unit_scale = plot_height / 150.0
plot_left, plot_right = 116, 1912
bar_width = 62.0
first_x, step = 152.0, 81.0
centers = {category: first_x + i * step for i, category in enumerate(categories)}

text(75, 287, 'L/person/year', 21, INK, 'bold', role='axis_unit')
for tick in range(0, 151, 25):
    yy = baseline - tick * unit_scale
    rule(plot_left, yy, plot_right, yy, GRID if tick else INK, 1.0 if tick else 1.6)
    text(99, yy - 11, str(tick), 22, INK, 'regular', 'rt', 'axis_tick')

# Explain the fused mark in the naturally open space above the shorter glasses.
text(1210, 368, 'THE GLASS IS THE BAR', 21, INK, 'bold')
text(1210, 405, "One person's annual consumption.", 28, INK)
text(1210, 445, 'Full height includes foam and glass base.', 24, SOFT_INK)

glass_source = Image.open(ROOT / 'assets/glass.png').convert('RGBA')
source_w, source_h = glass_source.size
source_cap = 103
source_base = 80


def make_glass(value):
    # A single mapping controls the full silhouette, including both end caps.
    hh = round(value * unit_scale * S)
    ww = round(bar_width * S)
    cap_h = min(round(25 * S), round(hh * 0.22))
    base_h = min(round(18 * S), round(hh * 0.16))
    body_h = hh - cap_h - base_h
    target = Image.new('RGBA', (ww, hh))
    parts = [
        (glass_source.crop((0, 0, source_w, source_cap)), cap_h, 0),
        (glass_source.crop((0, source_cap, source_w, source_h - source_base)), body_h, cap_h),
        (glass_source.crop((0, source_h - source_base, source_w, source_h)), base_h, cap_h + body_h),
    ]
    for part, ph, py in parts:
        if ph:
            target.alpha_composite(part.resize((ww, ph), Image.Resampling.LANCZOS), (0, py))
    return target, cap_h / S, base_h / S


def beer_column(category, value, cx, comparison=False, person_index=0):
    glass, cap_height, base_height = make_glass(value)
    px = round(cx * S - glass.width / 2)
    py = round(baseline * S) - glass.height
    canvas.alpha_composite(glass, (px, py))
    actual_a = np.asarray(glass.getchannel('A'))
    yy, xx = np.where(actual_a >= 128)
    bbox = [(px + int(xx.min())) / S, (py + int(yy.min())) / S,
            (px + int(xx.max()) + 1) / S, (py + int(yy.max()) + 1) / S]
    value_y = py / S - 37
    text(cx, value_y, f'{value:.1f}', 28 if comparison else 25, RED if comparison else INK,
         'condensed_bold', 'mt', 'comparison_value' if comparison else 'data_value')
    # Figures are non-quantitative foreground parts. The glass's outer base remains visible.
    person = ['person_coral.png', 'person_navy.png', 'person_ivory.png'][person_index % 3]
    person_box = paste_asset(person, cx, baseline - 63, width=76)
    prefix, suffix = category.rsplit(' ', 1)
    text(cx, 1250, prefix, 19, INK, 'regular', 'mt', 'category_parent')
    text(cx, 1279, suffix, 31, RED if comparison else INK, 'bold', 'mt', 'category_id')
    if comparison:
        text(cx, 1320, 'COMPARISON', 14, INK, 'bold', 'mt', 'comparison_role')
    objects.append({'category': category, 'value': value, 'comparison': comparison,
                    'x_center': cx, 'target_full_height_px': value * unit_scale,
                    'raster_box_xyxy': [px / S, py / S, (px + glass.width) / S, baseline],
                    'visible_alpha128_box_xyxy': bbox, 'foam_cap_height_px': cap_height,
                    'base_slice_height_px': base_height, 'person_box_xyxy': person_box})


for i, (category, value) in enumerate(zip(categories, values)):
    beer_column(category, value, centers[category], person_index=i)

# Bottles identify supplied annotations; their constant physical size encodes no value.
color_assets = {'green': 'bottle_green.png', 'near-black': 'bottle_black.png', 'brown': 'bottle_brown.png'}
bottle_records = []
for item in data['bottle_annotations']:
    category = 'Region ' + item['category_id']
    idx = categories.index(category)
    cx = centers[category]
    cup_top = baseline - values[idx] * unit_scale
    icon_top = cup_top - 220
    box = paste_asset(color_assets[item['glass_color']], cx, icon_top, height=136)
    text(cx, icon_top + 81, item['id'][1:], 17, INK, 'bold', 'mt', 'bottle_id')
    text(cx, cup_top - 74, item['label'], 18, INK, 'regular', 'mt', 'bottle_label')
    rule(cx, cup_top - 50, cx, cup_top - 41, SOFT_INK, 1)
    bottle_records.append({'id': item['id'], 'label': item['label'], 'category': category,
                           'glass_color': item['glass_color'], 'quantitative_encoding': False,
                           'bbox_xyxy': box, 'leader_to_category_x': cx})

# Independent comparison on the identical scale, visibly outside the 20-region sequence.
comparison = data['comparison']
comparison_x = 1854.0
rule(1766, 805, 1766, 1323, '#21968E', 1.1)
text(comparison_x, 817, 'OUTSIDE THE 20', 17, INK, 'bold', 'mt')
beer_column(comparison['label'], comparison['value'], comparison_x, comparison=True, person_index=0)
comparison_y = baseline - comparison['value'] * unit_scale
# Cross the full field at the top of the comparison glass, not at the liquid/foam interface.
dashed(plot_left, plot_right, comparison_y, RED, width=2.5, dash=13, gap=10)
draw.ellipse(((plot_left - 3) * S, (comparison_y - 3) * S,
              (plot_left + 3) * S, (comparison_y + 3) * S), fill=RED)

# Compact method and provenance notes. All are ordinary study annotations, not claims of real statistics.
text(80, 1385, '01 / READING THE SCALE', 18, SOFT_INK, 'bold')
text(80, 1418, 'Complete glass height = consumption.', 23, INK)
text(80, 1454, 'Linear scale from 0 to 150 L/person/year.', 22, SOFT_INK)
rule(662, 1385, 662, 1490, '#D2D8C9', 1.1)
text(707, 1385, '02 / OBJECT ANNOTATIONS', 18, SOFT_INK, 'bold')
text(707, 1418, 'Bottles and people do not encode quantity.', 23, INK)
text(707, 1454, 'Each bottle is aligned with its assigned region.', 22, SOFT_INK)
rule(1323, 1385, 1323, 1490, '#D2D8C9', 1.1)
text(1368, 1385, '03 / SYNTHETIC DATA', 18, SOFT_INK, 'bold')
text(1368, 1418, 'Fictional regions, supplied study values.', 23, INK)
text(1368, 1454, 'Dashed line: Region U, outside the 20.', 22, SOFT_INK)

final = canvas.convert('RGB').resize((W, H), Image.Resampling.LANCZOS)
final.save(ROOT / 'final.png')
(ROOT / 'artifacts').mkdir(exist_ok=True)
layout = {'canvas': [W, H], 'axis': {'min': 0, 'max': 150, 'baseline_y': baseline,
          'plot_top_y': plot_top, 'pixels_per_liter': unit_scale}, 'objects': objects,
          'bottles': bottle_records, 'comparison_line_y': comparison_y, 'labels': labels,
          'generated_glass_source': 'assets/glass.png', 'generated_raw_source': 'assets/raw_01.png',
          'data_sha256': hashlib.sha256((ROOT / inputs['data']).read_bytes()).hexdigest()}
(ROOT / 'artifacts/layout.json').write_text(json.dumps(layout, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
index_path = ROOT / 'rounds/index.json'
if index_path.exists():
    index = read_json('rounds/index.json')
    if index and index[-1]['status'] == 'running':
        shutil.copy2(ROOT / 'final.png', ROOT / f"artifacts/candidate_{index[-1]['number']:02d}.png")
print(json.dumps({'image': 'final.png', 'size': [W, H], 'cups': len(objects), 'bottles': len(bottle_records)}))
