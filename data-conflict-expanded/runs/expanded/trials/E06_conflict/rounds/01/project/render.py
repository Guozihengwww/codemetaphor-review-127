"""Offline renderer. Run from this directory: PYTHON render.py.

The complete length from the shared left edge to the graphite tip encodes hours.
The generated tails and tips keep their aspect ratios; only the straight barrels
stretch. This script never generates images, accesses the network, or edits data.
"""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent
SCALE = 2
BG = (247, 245, 239)
INK = (38, 46, 43)
MUTED = (104, 112, 103)
ORIGIN_X = 246.0
FULL_LENGTH = 1510.0
FIRST_Y = 308.0
PITCH = 38.0
THICKNESS = 30.0


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def font(size, bold=False):
    name = 'NimbusSans-Bold.otf' if bold else 'NimbusSans-Regular.otf'
    return ImageFont.truetype(str(ROOT / 'assets/fonts' / name), round(size * SCALE))


def text(draw, xy, value, size, fill=INK, bold=False, anchor='lt'):
    pos = tuple(round(c * SCALE) for c in xy)
    draw.text(pos, value, font=font(size, bold), fill=fill, anchor=anchor)
    b = draw.textbbox(pos, value, font=font(size, bold), anchor=anchor)
    return [round(v / SCALE, 3) for v in b]


def line(draw, points, fill, width=1):
    draw.line([(round(x * SCALE), round(y * SCALE)) for x, y in points],
              fill=fill, width=max(1, round(width * SCALE)))


def build_pencil(source, meta, length):
    """One piecewise transform with no extra cap or arrow beyond the data tip."""
    total = round(length * SCALE)
    height = round(THICKNESS * SCALE)
    factor = height / source.height
    a, b = meta['tail_end_x'], meta['tip_start_x']
    tail_w = round(a * factor)
    tip_w = round((source.width - b) * factor)
    # No minimum data length. If needed, all parts shrink together.
    if total < tail_w + tip_w + 1:
        return source.resize((max(1, total), height), Image.Resampling.LANCZOS)
    body_w = total - tail_w - tip_w
    out = Image.new('RGBA', (total, height))
    for box, size, x in [((0, 0, a, source.height), (tail_w, height), 0),
                         ((a, 0, b, source.height), (body_w, height), tail_w),
                         ((b, 0, source.width, source.height), (tip_w, height), tail_w + body_w)]:
        segment = source.crop(box).resize(size, Image.Resampling.LANCZOS)
        out.alpha_composite(segment, (x, 0))
    return out


def main():
    input_files = read_json('input.json')
    brief = read_json(input_files['brief'])
    data = read_json(input_files['data'])
    core = read_json(input_files['critical_metaphor_contract'])
    assert (ROOT / input_files['reference']).is_file()
    assert core['visible_title'] == data['title']
    assert data['synthetic'] is True
    assert data['value_axis'] == {'minimum': 0, 'maximum': 2300, 'scale': 'linear'}
    assert len(data['series']) == 1
    categories = data['categories']
    series = data['series'][0]
    values = series['values']
    assert len(categories) == len(values) == 26
    assert all(0 < v <= data['value_axis']['maximum'] for v in values)
    assert math.isclose(sum(values), data['series_totals'][series['id']])
    width, height = brief['width'], brief['height']
    image = Image.new('RGBA', (width * SCALE, height * SCALE), BG + (255,))
    draw = ImageDraw.Draw(image)

    # Editorial heading and an explicit synthetic-data notice.
    draw.rounded_rectangle((64*SCALE, 38*SCALE, 290*SCALE, 71*SCALE),
                           radius=3*SCALE, fill=(41, 67, 56))
    text(draw, (177, 55), data['visible_notice'], 17, (255, 255, 249), True, 'mm')
    text(draw, (64, 95), 'Annual working hours', 78, INK, True)
    text(draw, (67, 185), 'Annual hours worked per person in fictional countries.', 25, MUTED)
    text(draw, (1756, 55), f'{len(categories):02d}  FICTIONAL COUNTRIES', 17, MUTED, False, 'rm')
    text(draw, (1756, 199), 'Annual hours  /  h/year', 21, INK, False, 'rm')
    line(draw, [(64, 225), (1756, 225)], (198, 203, 192), 1)

    text(draw, (194, 243), 'HOURS / YEAR', 15, MUTED, True, 'rm')
    text(draw, (305, 243), 'FICTIONAL COUNTRY', 15, MUTED, True, 'lm')
    max_value = data['value_axis']['maximum']
    pixels_per_hour = FULL_LENGTH / max_value
    ticks = [0, 500, 1000, 1500, 2000, 2300]
    for v in ticks:
        x = ORIGIN_X + v * pixels_per_hour
        line(draw, [(x, 282), (x, 1280)], (219, 223, 213) if v == 0 else (233, 234, 224), 0.7)
        line(draw, [(x, 273), (x, 282)], (162, 173, 157), 1)
        text(draw, (x, 266), f'{v:,}', 15, MUTED, False, 'mb')
    line(draw, [(ORIGIN_X, 275), (ORIGIN_X + FULL_LENGTH, 275)], (182, 191, 175), 1)

    # Shavings are a margin treatment. They never overlap a measured pencil.
    shavings = Image.open(ROOT / 'assets/shavings.png').convert('RGBA')
    shaving_h = round(1080 * SCALE)
    shaving_w = round(shavings.width / shavings.height * shaving_h)
    shavings = shavings.resize((shaving_w, shaving_h), Image.Resampling.LANCZOS)
    image.alpha_composite(shavings, (round(1787*SCALE), round(244*SCALE)))

    metadata = read_json('assets/pencil_parts.json')
    sources = {name: Image.open(ROOT / m['path']).convert('RGBA') for name, m in metadata.items()}
    colors = ['gold', 'teal', 'red', 'gold', 'red', 'green', 'gold', 'gold', 'teal',
              'turquoise', 'teal', 'turquoise', 'green', 'red', 'green', 'red', 'teal',
              'coral', 'turquoise', 'gold', 'teal', 'green', 'coral', 'red', 'turquoise', 'teal']
    records = []
    for i, (category, value, color) in enumerate(zip(categories, values, colors)):
        y = FIRST_Y + i * PITCH
        exact_length = value * pixels_per_hour
        pencil = build_pencil(sources[color], metadata[color], exact_length)
        image.alpha_composite(pencil, (round(ORIGIN_X*SCALE), round((y-THICKNESS/2)*SCALE)))
        draw = ImageDraw.Draw(image)
        label_color = (255, 253, 243) if color in ('teal', 'red') else (24, 48, 37)
        category_box = text(draw, (305, y+0.5), category, 21, label_color, True, 'lm')
        value_label = f'{value:,.0f}' if float(value).is_integer() else str(value)
        value_box = text(draw, (194, y+0.5), value_label, 25, INK, False, 'rm')
        records.append({'category': category, 'value': value, 'unit': series['unit'],
                        'row_center_y': y, 'asset': metadata[color]['path'],
                        'value_label': value_label, 'category_label_bbox': category_box,
                        'value_label_bbox': value_box,
                        'target_left_x': ORIGIN_X,
                        'target_tip_x': ORIGIN_X + exact_length,
                        'target_length_px': exact_length,
                        'rendered_image_width_px': pencil.width / SCALE})

    draw = ImageDraw.Draw(image)
    line(draw, [(64, 1310), (1756, 1310)], (198, 203, 192), 1)
    text(draw, (65, 1330), 'Full pencil length = annual hours, from the tail to the graphite tip.', 19, INK)
    text(draw, (65, 1360), 'Synthetic values and fictional countries.  Supplied country order.  Colour is decorative.', 17, MUTED)
    out = image.convert('RGB').resize((width, height), Image.Resampling.LANCZOS)
    out.save(ROOT / 'final.png', optimize=True)

    # Measure the actual final raster, not only the requested transform.
    # Grid contrast is under 45 (sum RGB distance); pencil ink exceeds 90.
    pixels = np.asarray(out).astype(np.int16)
    bg = np.array(BG, dtype=np.int16)
    for record in records:
        y = int(record['row_center_y'])
        left = int(ORIGIN_X) - 3
        right = int(math.ceil(record['target_tip_x'])) + 6
        band = pixels[y-15:y+16, left:right]
        visible = np.abs(band - bg).sum(axis=2) > 90
        ys, xs = np.where(visible)
        measured_left = left + int(xs.min())
        measured_right = left + int(xs.max())
        visible_span = measured_right - measured_left + 1
        record.update(visible_left_pixel=measured_left, visible_tip_pixel=measured_right,
                      visible_span_px=visible_span,
                      visible_span_implied_hours=round(visible_span/pixels_per_hour, 5),
                      measured_error_hours=round(visible_span/pixels_per_hour-record['value'], 5),
                      visible_tip_error_px=round(measured_right+1-record['target_tip_x'], 5))
    check = {'canvas': [width, height], 'row_count': len(records), 'series_count': 1,
             'axis': data['value_axis'], 'pixels_per_hour': pixels_per_hour,
             'measurement_zh': '在最终PNG逐行扫描与背景RGB总差大于90的像素，记录实体左右边缘；此阈值排除浅色网格。像素跨度含首尾像素。',
             'max_abs_measured_error_hours': max(abs(r['measured_error_hours']) for r in records),
             'max_abs_tip_error_px': max(abs(r['visible_tip_error_px']) for r in records),
             'rows': records}
    (ROOT / 'artifacts/geometry_check.json').write_text(json.dumps(check, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in check.items() if k not in ('rows', 'measurement_zh')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
