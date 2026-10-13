#!/usr/bin/env python3
"""Offline, data-bound rendering. Run through capture_render.py for each attempt."""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parent
SCALE = 2
BG = (248, 246, 239)
INK = (27, 42, 60)
MUTED = (102, 109, 117)
GRID = (223, 224, 219)
GOLD = (232, 205, 140)


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def save_json(path, value):
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    inputs = read_json('input.json')
    brief = read_json(inputs['brief'])
    data = read_json(inputs['data'])
    core = read_json(inputs['critical_metaphor_contract'])
    assert (ROOT / inputs['reference']).is_file()
    assert core['semantic_roles_zh'] == ['Male nominees (%)']
    assert data['synthetic'] is True
    assert len(data['series']) == 1
    series = data['series'][0]
    categories = data['categories']
    values = series['values']
    assert len(values) == len(categories)
    axis = data['value_axis']
    assert axis == {'minimum': 0, 'maximum': 100, 'scale': 'linear'}
    assert all(0 <= v <= axis['maximum'] for v in values)
    assert sum(values) == data['series_totals'][series['id']]

    w, h = brief['width'], brief['height']
    canvas = Image.new('RGB', (w * SCALE, h * SCALE), BG)
    draw = ImageDraw.Draw(canvas)
    fonts = {}

    def font(size, kind='regular'):
        file = {'regular': 'NimbusSans-Regular.otf', 'bold': 'NimbusSans-Bold.otf',
                'serif': 'NimbusRoman-Regular.otf'}[kind]
        key = (size, kind)
        if key not in fonts:
            fonts[key] = ImageFont.truetype(str(ROOT / 'assets/fonts' / file), round(size * SCALE))
        return fonts[key]

    def text(x, y, s, size, kind='regular', fill=INK, anchor='lt'):
        draw.text((round(x * SCALE), round(y * SCALE)), s,
                  font=font(size, kind), fill=fill, anchor=anchor)

    def tracked(x, y, s, size, spacing=2, fill=INK, right=False):
        f = font(size, 'bold')
        width = sum(draw.textlength(c, font=f) for c in s) / SCALE + spacing * (len(s) - 1)
        if right:
            x -= width
        for c in s:
            text(x, y, c, size, 'bold', fill)
            x += draw.textlength(c, font=f) / SCALE + spacing

    def line(points, fill, width=1):
        draw.line([(round(x * SCALE), round(y * SCALE)) for x, y in points],
                  fill=fill, width=max(1, round(width * SCALE)))

    # Title and topic. All statistical/category labels are drawn by code.
    draw.rounded_rectangle((104 * SCALE, 82 * SCALE, 443 * SCALE, 133 * SCALE),
                           radius=4 * SCALE, fill=GOLD)
    tracked(123, 98, data['visible_notice'], 25, 2.1)
    tracked(1796, 99, 'DEPARTMENT COMPARISON', 23, 2, MUTED, right=True)
    title = data['title'].removeprefix('SYNTHETIC ')
    first, second = title.split(' by ', 1)
    text(104, 177, first, 134, 'serif')
    text(104, 303, 'by ' + second, 134, 'serif')
    text(109, 451, 'Share of each department’s nominees who are male', 33, fill=MUTED)
    line([(104, 517), (1796, 517)], (184, 186, 181), 1.5)
    tracked(108, 549, 'DEPARTMENT', 22, 2)
    tracked(460, 549, series['label'].upper() + ' (' + series['unit'] + ')', 22, 2)

    # A single shared linear transform for all data-bearing arm spans.
    x0, x100 = 460.0, 1690.0
    full_span = x100 - x0
    px_per_unit = full_span / (axis['maximum'] - axis['minimum'])
    axis_y = 1776
    chart_top = 597
    row_centres = np.linspace(680, 1680, len(categories)).tolist()
    arm_height = 162.0
    for tick in range(20, 101, 20):
        tx = x0 + tick * px_per_unit
        line([(tx, chart_top), (tx, axis_y)], GRID, 1.15)
    line([(x0, chart_top), (x0, axis_y)], (77, 86, 95), 2.0)
    line([(x0, axis_y), (x100, axis_y)], (126, 132, 137), 1.6)
    for tick in range(0, 101, 20):
        tx = x0 + tick * px_per_unit
        line([(tx, axis_y), (tx, axis_y + 10)], (126, 132, 137), 1.5)
        label_text = str(tick) + ('%' if tick == 100 else '')
        text(tx, axis_y + 27, label_text, 30, fill=MUTED, anchor='mt')

    master_image = Image.open(ROOT / 'assets/arm_master.png').convert('RGBA')
    master = np.asarray(master_image, dtype=np.float32) / 255.0
    sh, sw = master.shape[:2]
    source_alpha = master[:, :, 3]
    premultiplied = master.copy()
    premultiplied[:, :, :3] *= source_alpha[:, :, None]
    split = float(read_json('artifacts/asset_processing.json')['sleeve_mapping_split_x'])

    def arm_for_span(span):
        # Suffix contains the sleeve buttons, cuff, watch, forearm and fist.
        # It retains its aspect ratio. Only the uninterrupted sleeve body stretches.
        natural_scale = arm_height / sh
        suffix_width = (sw - split) * natural_scale
        fixed_scale = min(1.0, span / (suffix_width + split * natural_scale * 0.10))
        suffix_width *= fixed_scale
        out_h = arm_height * fixed_scale
        body_width = span - suffix_width
        ow = max(1, round(span * SCALE))
        oh = max(1, round(out_h * SCALE))
        xx = (np.arange(ow, dtype=np.float32) + 0.5) * span / ow
        yy = (np.arange(oh, dtype=np.float32) + 0.5) * sh / oh - 0.5
        source_x = np.where(xx < body_width,
                            xx * split / body_width - 0.5,
                            split + (xx - body_width) * (sw - split) / suffix_width - 0.5)
        sy, sx = np.meshgrid(yy, source_x, indexing='ij')
        out = np.stack([map_coordinates(premultiplied[:, :, c], [sy, sx],
                                       order=1, mode='nearest', prefilter=False)
                        for c in range(4)], axis=2)
        out[:, :, :3] = np.divide(out[:, :, :3], out[:, :, 3:4],
                                 out=np.zeros_like(out[:, :, :3]),
                                 where=out[:, :, 3:4] > 1e-6)
        patch = Image.fromarray(np.rint(np.clip(out, 0, 1) * 255).astype(np.uint8), 'RGBA')
        return patch, out_h, suffix_width

    rows = []
    for category, value, cy in zip(categories, values, row_centres):
        text(107, cy, category, 42, anchor='lm')
        target_span = (value - axis['minimum']) * px_per_unit
        endpoint = x0 + target_span
        record = {'category': category, 'value': value, 'unit': series['unit'],
                  'row_centre_y': cy, 'target_endpoint_x': endpoint,
                  'target_span_px': target_span, 'asset_present': value > 0}
        if value > 0:
            patch, patch_h, suffix_w = arm_for_span(target_span)
            px = round(x0 * SCALE)
            py = round((cy - patch_h / 2) * SCALE)
            canvas.paste(patch, (px, py), patch)
            # Measure transformed raster alpha, not the target span alone.
            alpha_final = patch.getchannel('A').resize(
                (round(patch.width / SCALE), round(patch.height / SCALE)), Image.Resampling.LANCZOS)
            aa = np.asarray(alpha_final)
            yy, xx = np.where(aa >= 128)
            record['transformed_alpha_bbox_local_px'] = [int(xx.min()), int(yy.min()),
                                                         int(xx.max()) + 1, int(yy.max()) + 1]
            record['alpha_right_edge_x'] = x0 + int(xx.max()) + 1
            record['alpha_left_edge_x'] = x0 + int(xx.min())
            record['fixed_suffix_width_px'] = round(suffix_w, 4)
            record['display_height_px'] = patch_h
            record['alpha_pixel_count_ge_128'] = int((aa >= 128).sum())
        else:
            record['alpha_pixel_count_ge_128'] = 0
            record['display_height_px'] = 0
        label = f'{value:g}' + series['unit']
        label_x = endpoint + 25
        text(label_x, cy, label, 50, 'bold', anchor='lm')
        record['numeric_label'] = label
        record['numeric_label_x'] = label_x
        rows.append(record)

    line([(104, 1871), (1796, 1871)], (198, 199, 193), 1)
    text(108, 1901, 'Arm length measures the share, from the common zero to the fist tip.', 28, fill=MUTED)
    text(108, 1943, 'Illustrative departments. Synthetic figures, not real-world statistics.', 27, fill=MUTED)

    final = canvas.resize((w, h), Image.Resampling.LANCZOS)
    final.save(ROOT / 'final.png')

    # Inspect actual composited pixels for every positive endpoint. The mask
    # excludes the neutral grid/background and numerical labels (25 px away).
    pixels = np.asarray(final).astype(np.int16)
    for row in rows:
        cy = row['row_centre_y']
        ylo = max(0, math.floor(cy - arm_height / 2 - 2))
        yhi = min(h, math.ceil(cy + arm_height / 2 + 2))
        if row['value'] > 0:
            xlo = round(x0)
            xhi = math.ceil(row['target_endpoint_x']) + 8
            crop = pixels[ylo:yhi, xlo:xhi]
            chroma = crop.max(axis=2) - crop.min(axis=2)
            visible = (chroma > 24) | (crop.min(axis=2) < 140)
            support_columns = np.flatnonzero(visible.sum(axis=0) >= 2)
            observed_right = xlo + int(support_columns[-1]) + 1
            row['final_pixel_right_edge_x'] = observed_right
            row['observed_span_px'] = observed_right - x0
            row['observed_value_from_pixels'] = round((observed_right - x0) / px_per_unit, 5)
            row['absolute_error_percentage_points'] = round(abs((observed_right - x0) / px_per_unit - row['value']), 5)
        else:
            # The category and 0% label are allowed; the rest of the row must
            # contain no colored object. Neutral grid lines are not arm marks.
            crop = pixels[ylo:yhi, 590:round(x100) + 3]
            chroma = crop.max(axis=2) - crop.min(axis=2)
            visible = (chroma > 24) | (crop.min(axis=2) < 140)
            row['zero_row_colored_pixels_outside_label'] = int(visible.sum())
            row['observed_span_px'] = 0
            row['observed_value_from_pixels'] = 0
            row['absolute_error_percentage_points'] = 0
    save_json('artifacts/geometry_checks.json', {
        'canvas_size': [w, h], 'axis_origin_x': x0, 'axis_100_x': x100,
        'pixels_per_percentage_point': px_per_unit,
        'alpha_threshold': 128,
        'final_pixel_rule': 'In each arm-only crop: chroma > 24 or min channel < 140; require >= 2 pixels per column.',
        'rows': rows,
        'max_absolute_error_percentage_points': max(r['absolute_error_percentage_points'] for r in rows),
        'mapping': 'All rows use x = 460 + percentage * 12.3. Contiguous sleeve body alone stretches; complete suffix keeps its aspect ratio.',
    })
    print(f'Rendered {w} x {h}; {len(rows)} categories; max measured error '
          f'{max(r["absolute_error_percentage_points"] for r in rows):.5f} percentage points.')


if __name__ == '__main__':
    main()
