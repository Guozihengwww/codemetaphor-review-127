"""Render the frozen pillow infographic from local data and local assets.

Run: PYTHON render.py
No model, web, or network calls. All geometry is computed from data.json.
The mid-height alpha silhouette supplies the cloth endpoints; 30% is decorative.
"""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'assets'
ARTIFACTS = ROOT / 'artifacts'
ARTIFACTS.mkdir(exist_ok=True)
DATA = json.loads((ROOT / 'data.json').read_text(encoding='utf-8'))
BRIEF = json.loads((ROOT / 'brief.json').read_text(encoding='utf-8'))
CW, CH = BRIEF['width'], BRIEF['height']
S = 2
COLORS = ['#50463F', '#008F83', '#086FAF', '#E9A829']
INK = '#292D2C'
MUTED = '#696D6B'
PAPER = '#F7F7F1'

def rgb(hex_color):
    return np.array([int(hex_color[k:k + 2], 16) for k in (1, 3, 5)], dtype=float)


def font(size, bold=False, cjk=False):
    name = 'NotoSansCJK-Regular.ttc' if cjk else ('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')
    return ImageFont.truetype(str(ASSETS / 'fonts' / name), round(size * S))


# A neutral studio surface. Its quiet gradient supports the pillow's volume.
yy, xx = np.mgrid[0:CH, 0:CW]
light = np.exp(-(((xx - 1050) / 1200) ** 2 + ((yy - 390) / 1050) ** 2))
base = np.zeros((CH, CW, 3), dtype=np.float32)
for ch, v in enumerate((228.0, 229.0, 226.0)):
    base[:, :, ch] = v + 11.0 * light
canvas = Image.fromarray(np.uint8(np.clip(base, 0, 255))).convert('RGBA').resize((CW * S, CH * S), Image.Resampling.BICUBIC)

# Keep the generated pillow's aspect ratio, including its intact protruding corners.
pillow = Image.open(ASSETS / 'pillow_clean.png').convert('RGBA')
pw = 1750
ph = round(pillow.height * pw / pillow.width)
pillow = pillow.resize((pw, ph), Image.Resampling.LANCZOS)
px, py = (CW - pw) // 2, 389
pa = np.array(pillow)
alpha = pa[:, :, 3]
mid_y = ph // 2
visible = np.flatnonzero(alpha[mid_y] >= 128)
left, right = float(visible[0]) - .5, float(visible[-1]) + .5
cloth_width = right - left
nondata_fraction = .30
data_left = left + cloth_width * nondata_fraction
data_right = right
data_width = data_right - data_left
values = np.array(DATA['series'][0]['values'], dtype=float)
categories = DATA['categories']
assert len(values) == len(categories) == len(COLORS)
assert abs(float(values.sum()) - 100) < 1e-9
assert DATA['value_axis'] == {'minimum': 0, 'maximum': 100, 'scale': 'linear'}
boundaries = data_left + np.r_[0., np.cumsum(values)] * data_width / 100.0

# Tint the actual photographed cloth. Each vertical boundary is data-driven and
# antialiased by pixel coverage. Luminance retains the generated cotton shading.
orig = pa[:, :, :3].astype(float)
lum = orig[:, :, 0] * .2126 + orig[:, :, 1] * .7152 + orig[:, :, 2] * .0722
shade = np.power(np.clip(lum / 228.0, .01, 1.3), 1.24)
xc = np.arange(pw, dtype=float)
white_cov = np.clip(data_left - (xc - .5), 0, 1)
colored = orig * white_cov[None, :, None]
for i, color in enumerate(COLORS):
    lo = boundaries[i]
    # Continue the final band to the silhouette at flared upper/lower corners.
    hi = boundaries[i + 1] if i + 1 < len(values) else float(pw + 1)
    cov = np.clip(np.minimum(xc + .5, hi) - np.maximum(xc - .5, lo), 0, 1)
    dyed = np.clip(shade[:, :, None] * rgb(color)[None, None, :], 0, 255)
    colored += dyed * cov[None, :, None]
pa[:, :, :3] = np.rint(colored).clip(0, 255).astype('uint8')
pillow_colored = Image.fromarray(pa, 'RGBA')

# Two soft, physically grounded floor shadows, behind the complete pillow.
shadow = Image.new('RGBA', (CW * S, CH * S))
sd = ImageDraw.Draw(shadow)
cy = py + ph - 12
sd.ellipse(tuple(round(z * S) for z in (px + 47, cy - 22, px + pw - 38, cy + 43)), fill=(49, 53, 49, 42))
shadow = shadow.filter(ImageFilter.GaussianBlur(23 * S))
canvas.alpha_composite(shadow)
contact = Image.new('RGBA', canvas.size)
cd = ImageDraw.Draw(contact)
cd.ellipse(tuple(round(z * S) for z in (px + 105, cy - 6, px + pw - 95, cy + 17)), fill=(37, 40, 36, 57))
contact = contact.filter(ImageFilter.GaussianBlur(9 * S))
canvas.alpha_composite(contact)
canvas.alpha_composite(pillow_colored.resize((pw * S, ph * S), Image.Resampling.LANCZOS), (px * S, py * S))
draw = ImageDraw.Draw(canvas)
label_records = []


def text(x, y, content, size, color=INK, bold=False, anchor='lt', cjk=False, role='text'):
    f = font(size, bold, cjk)
    at = (round(x * S), round(y * S))
    draw.text(at, content, fill=color, font=f, anchor=anchor)
    box = [float(v) / S for v in draw.textbbox(at, content, font=f, anchor=anchor)]
    label_records.append({'text': content, 'role': role, 'bbox': box})
    return box


def line(points, fill, width=1.4):
    draw.line([(round(x * S), round(y * S)) for x, y in points], fill=fill, width=round(width * S), joint='curve')


# Editorial heading; the frozen data title and notice remain explicit.
text(130, 76, DATA['title'], 25, MUTED, cjk=True, role='data_title')
text(125, 130, 'How many hours do you sleep?', 64, bold=True, role='headline')
text(131, 226, f"{DATA['series'][0]['label']} ({DATA['series'][0]['unit']})  /  Sleep duration in hours", 24, MUTED, role='series_and_unit')
# A quiet status mark is deliberately distant from the quantitative fabric.
line([(1557, 86), (1580, 86)], '#747976', 3)
text(1598, 75, DATA['visible_notice'], 21, INK, bold=True, role='synthetic_notice')

# Categories with narrow true widths receive external, directly attached labels.
# Broad categories have the exact number and category printed on the cloth.
labels_per_category = []
for i, (cat, val) in enumerate(zip(categories, values)):
    center = float((boundaries[i] + boundaries[i + 1]) / 2 + px)
    band_width = float(boundaries[i + 1] - boundaries[i])
    val_text = f'{val:g}{DATA["series"][0]["unit"]}'
    if val < 10:
        text(center, 298, val_text, 55, INK, bold=True, anchor='mt', role=f'value_{i}')
        text(center, 366, cat, 28, MUTED, anchor='mt', role=f'category_{i}')
        local_x = int(round(center - px))
        ys = np.flatnonzero(alpha[:, local_x] >= 128)
        end_y = float(py + ys[0] + 27)
        line([(center, 407), (center, end_y)], '#737A76', 1.7)
        r = 4.5
        draw.ellipse(tuple(round(z * S) for z in (center - r, end_y - r, center + r, end_y + r)), fill=PAPER, outline=COLORS[i], width=round(2 * S))
        labels_per_category.append({'category': cat, 'value': val_text, 'location': 'external', 'leader_endpoint': [center, end_y]})
    else:
        fsize = 96
        while draw.textlength(val_text, font=font(fsize, bold=True)) / S > band_width - 38:
            fsize -= 1
        text(center, 851 + (96 - fsize), val_text, fsize, PAPER, bold=True, anchor='mt', role=f'value_{i}')
        text(center, 965, cat, 35, PAPER, anchor='mt', role=f'category_{i}')
        labels_per_category.append({'category': cat, 'value': val_text, 'location': 'on_fabric', 'center_x': center})

white_center = px + (left + data_left) / 2
text(white_center, 875, 'WHITE FABRIC', 20, '#70736F', anchor='mt', role='decorative_label')
text(white_center, 911, 'Excluded from data', 20, '#7D807B', anchor='mt', role='decorative_explanation')

line([(130, 1324), (1870, 1324)], '#BFC3BE', 1)
text(130, 1350, '色带宽度表示占比；左侧白布为装饰，不计入 100%。', 23, '#5E655F', cjk=True, role='encoding_note')
text(1868, 1353, 'Synthetic data · Not survey results', 19, '#6F746E', anchor='rt', role='provenance_note')

final = canvas.convert('RGB').resize((CW, CH), Image.Resampling.LANCZOS)
final.save(ROOT / 'final.png')

# Verify actual final-image pixels at the encoding cross-section, away from text.
final_array = np.asarray(final).astype(float)
scan_y = py + mid_y
scan_left, scan_right = px + int(visible[0]), px + int(visible[-1])
scan = final_array[scan_y, scan_left:scan_right + 1]
prototypes = np.vstack([np.ones(3), *[rgb(c) for c in COLORS]])
prototypes /= np.linalg.norm(prototypes, axis=1)[:, None]
scan_norm = scan / np.maximum(np.linalg.norm(scan, axis=1)[:, None], 1e-9)
classes = np.argmin(np.sum((scan_norm[:, None, :] - prototypes[None, :, :]) ** 2, axis=2), axis=1)
runs = []
start = 0
for j in range(1, len(classes) + 1):
    if j == len(classes) or classes[j] != classes[start]:
        runs.append({'region_index': int(classes[start]), 'x_start': int(scan_left + start), 'x_end_inclusive': int(scan_left + j - 1), 'width_px': int(j - start)})
        start = j
measured = []
for i, value in enumerate(values):
    width = int(np.count_nonzero(classes == i + 1))
    pct = width / data_width * 100.0
    measured.append({'category': categories[i], 'input_value_percent': float(value), 'actual_pixel_width': width,
                     'actual_share_percent': round(pct, 4), 'absolute_error_percentage_points': round(abs(pct - value), 4)})

bounds_ok = all(0 <= b['bbox'][0] <= b['bbox'][2] <= CW and 0 <= b['bbox'][1] <= b['bbox'][3] <= CH for b in label_records)
geometry = {
    'canvas_size': [CW, CH],
    'cross_section_y': scan_y,
    'visible_cloth_midline_endpoints': [px + left, px + right],
    'visible_cloth_midline_width_px': cloth_width,
    'decorative_white_fraction': nondata_fraction,
    'data_interval': [px + data_left, px + data_right],
    'data_width_px': data_width,
    'vertical_boundaries_x': [round(float(x + px), 6) for x in boundaries],
    'target_band_widths_px': [round(float(x), 6) for x in np.diff(boundaries)],
    'actual_final_image_pixel_scan_runs': runs,
    'actual_final_image_band_measurements': measured,
    'all_text_within_canvas': bounds_ok,
    'data_labels': labels_per_category,
    'text_bounding_boxes': label_records,
    'final_png_sha256': hashlib.sha256((ROOT / 'final.png').read_bytes()).hexdigest(),
    'input_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in ['brief.json', 'data.json', 'reference.png']},
    'measurement_method_zh': '在最终PNG的枕头中段水平截面，对实际RGB像素按色彩方向分类，统计白布和四色的连续像素宽度；该行避开文字。'
}
(ARTIFACTS / 'render_measurements.json').write_text(json.dumps(geometry, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
# A small diagnostic strip is not a complete chart candidate.
final.crop((scan_left, scan_y - 8, scan_right + 1, scan_y + 9)).save(ARTIFACTS / 'midline_scan.png')
print(json.dumps({'canvas_size': [CW, CH], 'scan_y': scan_y, 'pixel_runs': runs, 'actual_band_measurements': measured, 'all_text_within_canvas': bounds_ok}, ensure_ascii=False))
