"""Regenerate final.png from the frozen local data and generated illustration.

Complete renders must be launched through capture_render.py.
Only Pillow and the Python standard library are needed at render time.
"""
from decimal import Decimal
from pathlib import Path
import hashlib
import json
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
TEXT_SS = 3
ARM_SS = 12
PAPER = '#F8F5ED'
INK = '#202A34'
MUTED = '#5D6469'
NAVY = '#17314F'
GRID = '#E4E1D9'
OCHRE = '#E8C57B'


def read_json(path, decimal=False):
    return json.loads((ROOT / path).read_text(encoding='utf-8'),
                      parse_float=Decimal if decimal else float)


def write_json(path, data):
    (ROOT / path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n',
                             encoding='utf-8')


def font(name, size):
    return ImageFont.truetype(str(ROOT / 'assets/fonts' / name), round(size * TEXT_SS))


def txt(draw, xy, content, size=30, family='Sans.otf', fill=INK, anchor='lt'):
    draw.text(tuple(round(v * TEXT_SS) for v in xy), content,
              font=font(family, size), fill=fill, anchor=anchor,
              stroke_width=0)


def line(draw, points, fill=GRID, width=1):
    draw.line([(round(x * TEXT_SS), round(y * TEXT_SS)) for x, y in points],
              fill=fill, width=max(1, round(width * TEXT_SS)))


def value_label(value, unit):
    return str(value) + unit


def make_arm(master, metadata, span):
    """One mapping for all nonzero values; no positive minimum span.

    Long sleeves stretch only before the button/cuff region. Below the compact
    pose width, the whole connected arm shrinks uniformly. Both regimes have
    exactly the requested total width and meet continuously at 450 px.
    """
    compact_width = 450.0
    normal_height = 135.0
    split = metadata['sleeve_slice_x_px']
    sw, sh = master.size
    shrink = min(1.0, span / compact_width)
    height = normal_height * shrink
    cap_width = (sw - split) * (normal_height / sh) * shrink
    sleeve_width = span - cap_width
    # Supersampling preserves fractional-pixel spans without a minimum size.
    target_w = max(1, round(span * ARM_SS))
    target_h = max(1, round(height * ARM_SS))
    cap_w = min(target_w - 1, max(1, round(cap_width * ARM_SS)))
    sleeve_w = target_w - cap_w
    rgba = Image.new('RGBA', (math.ceil(span) * ARM_SS,
                              math.ceil(height) * ARM_SS), (0, 0, 0, 0))
    sleeve = master.crop((0, 0, split, sh)).resize((sleeve_w, target_h), Image.Resampling.LANCZOS)
    cap = master.crop((split, 0, sw, sh)).resize((cap_w, target_h), Image.Resampling.LANCZOS)
    rgba.alpha_composite(sleeve, (0, 0))
    rgba.alpha_composite(cap, (sleeve_w, 0))
    rgba = rgba.resize((math.ceil(span), math.ceil(height)), Image.Resampling.BOX)
    return rgba, {
        'target_span_px': span,
        'nominal_height_px_non_data': height,
        'rasterization_grid_span_px': target_w / ARM_SS,
        'sleeve_span_px': sleeve_width,
        'connected_end_span_px': cap_width,
        'compact_pose_scale': shrink,
    }


def contrast_bbox(actual, before, threshold):
    diff = np.max(np.abs(actual.astype('int16') - before.astype('int16')), axis=2)
    yy, xx = np.where(diff >= threshold)
    if not len(xx):
        return None
    return [int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)]


def main():
    paths = read_json('input.json')
    brief = read_json(paths['brief'])
    data = read_json(paths['data'], decimal=True)
    core = read_json(paths['critical_metaphor_contract'])
    assert (ROOT / paths['reference']).is_file()
    assert core['semantic_roles_zh'] == ['Male nominees (%)']
    assert data['value_axis'] == {'minimum': 0, 'maximum': 100, 'scale': 'linear'}
    assert len(data['series']) == 1
    series = data['series'][0]
    values = series['values']
    categories = data['categories']
    assert len(categories) == len(values) == 6
    assert all(0 <= v <= 100 for v in values)
    width, height = brief['width'], brief['height']
    assert (width, height) == (1900, 2000)

    x0, plot_width = 480, 1320
    x1 = x0 + plot_width
    centers = [620 + i * 190 for i in range(len(categories))]
    axis_y = 1670
    img = Image.new('RGB', (width * TEXT_SS, height * TEXT_SS), PAPER)
    d = ImageDraw.Draw(img)

    # Clear synthetic notice, typographic headline, and the statistical meaning.
    d.rounded_rectangle((100 * TEXT_SS, 77 * TEXT_SS, 426 * TEXT_SS, 129 * TEXT_SS),
                        radius=4 * TEXT_SS, fill=OCHRE)
    txt(d, (121, 88), data['visible_notice'], 26, 'SansBold.otf')
    txt(d, (1800, 94), 'NOMINATIONS / DEPARTMENT COMPARISON', 21, fill=MUTED, anchor='rt')
    title = data['title'].removeprefix('SYNTHETIC ')
    first, second = title.split(' by ', 1)
    txt(d, (95, 169), first, 106, 'EditorialSerif.otf')
    txt(d, (95, 278), 'by ' + second, 106, 'EditorialSerif.otf')
    txt(d, (102, 407), 'Share of all nominees in each department who are male.', 31, fill=MUTED)
    line(d, [(100, 468), (1800, 468)], fill=INK, width=1.4)

    txt(d, (100, 500), 'DEPARTMENT / SHARE', 22, 'SansBold.otf', fill=MUTED)
    txt(d, (x0, 500), series['label'].upper() + ' (' + series['unit'] + ')', 22, 'SansBold.otf', fill=MUTED)
    txt(d, (x1, 500), '0–100%  ·  LINEAR SCALE', 22, fill=MUTED, anchor='rt')

    # A single shared linear ruler. The zero rule sits immediately left of the
    # quantitative objects so that subpixel marks are not overpainted by it.
    for tick in range(20, 101, 20):
        x = x0 + plot_width * tick / 100
        line(d, [(x, 547), (x, axis_y)], GRID, 1)
    line(d, [(x0 - 1, 547), (x0 - 1, axis_y)], '#7C8386', 1.5)
    line(d, [(x0 - 1, axis_y), (x1, axis_y)], '#7C8386', 1.5)
    for tick in range(0, 101, 20):
        x = x0 + plot_width * tick / 100
        line(d, [(x, axis_y), (x, axis_y + 9)], '#7C8386', 1.5)
        txt(d, (x, axis_y + 25), str(tick) + ('%' if tick == 100 else ''),
            29, anchor='mt', fill=MUTED)

    for category, value, cy in zip(categories, values, centers):
        txt(d, (100, cy - 38), category, 34, 'SansBold.otf')
        txt(d, (100, cy + 10), value_label(value, series['unit']), 34,
            'Numbers.ttf', fill=NAVY)

    # Notes are outside the plotting area, and do not add decorative marks to
    # zero-valued or tiny rows.
    line(d, [(100, 1790), (1800, 1790)], '#BDBDB5', 1)
    txt(d, (100, 1824), 'HOW TO READ', 22, 'SansBold.otf', fill=MUTED)
    txt(d, (100, 1865), 'Each arm runs from zero to its rightmost knuckle.', 27)
    txt(d, (100, 1904), 'Sleeve, cuff, watch and fist all count toward its span.', 27)
    txt(d, (990, 1824), 'ILLUSTRATIVE VALUES ONLY', 22, 'SansBold.otf', fill=MUTED)
    txt(d, (990, 1865), 'Synthetic departments; no real statistical claims.', 27)
    txt(d, (990, 1904), 'Tiny shares keep their true length; 0% has no arm.', 27)

    # Resolve typography first, then composite object pixels without a later
    # canvas rescale. This keeps small-arm coverage measurable in final.png.
    img = img.resize((width, height), Image.Resampling.LANCZOS).convert('RGBA')
    master = Image.open(ROOT / 'assets/arm_master.png').convert('RGBA')
    metadata = read_json('assets/arm_metadata.json')
    before_crops = []
    rows = []
    for i, (category, value, cy) in enumerate(zip(categories, values, centers)):
        sample_box = (x0 - 3, cy - 77, x1 + 3, cy + 77)
        before_crops.append((sample_box, np.array(img.crop(sample_box).convert('RGB'))))
        span = plot_width * float(value) / 100
        row = {'category': category, 'label': value_label(value, series['unit']),
               'value': float(value), 'zero_x_px': x0,
               'target_endpoint_x_px': x0 + span, 'target_span_px': span}
        if value > 0:
            arm, transform = make_arm(master, metadata, span)
            y = cy - arm.height // 2
            img.alpha_composite(arm, (x0, y))
            row.update(transform)
            row['object_pixel_box'] = [x0, y, x0 + arm.width, y + arm.height]
            row['alpha_sum_px2'] = round(float(np.array(arm)[:, :, 3].sum()) / 255, 5)
            row['alpha_max'] = int(np.array(arm)[:, :, 3].max())
        else:
            row['object_pixel_box'] = None
            row['alpha_sum_px2'] = 0
        rows.append(row)

    img.convert('RGB').save(ROOT / 'final.png', optimize=True)
    # Read the saved final raster and measure changed visible pixels against
    # each plot-row crop before the objects were composited.
    saved = Image.open(ROOT / 'final.png').convert('RGB')
    for row, (box, before) in zip(rows, before_crops):
        current = np.array(saved.crop(box))
        checks = {}
        for threshold in (8, 16, 32):
            bbox = contrast_bbox(current, before, threshold)
            if bbox:
                global_bbox = [bbox[0] + box[0], bbox[1] + box[1],
                               bbox[2] + box[0], bbox[3] + box[1]]
                visible_span = global_bbox[2] - x0
                checks[str(threshold)] = {
                    'changed_pixel_bbox': global_bbox,
                    'visible_span_to_pixel_edge_px': visible_span,
                    'endpoint_error_px': round(visible_span - row['target_span_px'], 6),
                    'endpoint_error_percentage_points': round(
                        100 * (visible_span - row['target_span_px']) / plot_width, 6),
                    'span_relative_error': round(
                        (visible_span - row['target_span_px']) / row['target_span_px'], 6)
                        if row['target_span_px'] else None,
                }
            else:
                checks[str(threshold)] = {'changed_pixel_bbox': None}
        row['saved_final_contrast_measurements'] = checks

    write_json('artifacts/geometry_checks.json', {
        'method_zh': '读取已保存final.png，与叠加手臂前同一区域逐像素比较；按RGB最大通道差8、16、32统计可见边界。',
        'canvas_px': [width, height],
        'axis': {'minimum': 0, 'maximum': 100, 'scale': 'linear',
                 'zero_x_px': x0, 'hundred_x_px': x1, 'pixels_per_percent': plot_width / 100},
        'positive_minimum_span': False,
        'rows': rows,
        'final_sha256': hashlib.sha256((ROOT / 'final.png').read_bytes()).hexdigest(),
    })
    print('Wrote final.png: 1900 x 2000; six categories, five connected arm marks, one empty zero row.')


if __name__ == '__main__':
    main()
