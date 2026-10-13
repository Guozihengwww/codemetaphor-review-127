#!/usr/bin/env python3
"""Offline, data-bound paper-stack infographic. Run with no arguments.

Paper front, side and top are derived from assets/raw_01.png (built-in image tool).
Only front height encodes pages. The shared oblique projection has no data role.
All local inputs and fonts remain available; no model, browser or network calls.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent
S = 4
BG = '#f2f3ee'
INK = '#233d3b'
MUTED = '#68766e'
RUST = '#a73824'
GRID = '#d6dcd5'
FLOOR = '#e8ece5'


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def save_json(name, obj):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    inputs = read_json('input.json')
    brief = read_json(inputs['brief'])
    data = read_json(inputs['data'])
    core = read_json(inputs['critical_metaphor_contract'])
    assert (ROOT / inputs['reference']).is_file()
    assert core['visible_title'] == data['title']
    assert data['value_axis'] == {'minimum': 0, 'maximum': 80000, 'scale': 'linear'}
    assert len(data['series']) == 1
    series = data['series'][0]
    values = series['values']
    categories = data['categories']
    assert len(values) == len(categories)
    assert series['unit'] == 'pages'
    assert sum(values) == data['series_totals'][series['id']]
    W, H = brief['width'], brief['height']
    assert (W, H) == (1395, 2000)
    image = Image.new('RGBA', (W * S, H * S), BG)
    draw = ImageDraw.Draw(image)
    fonts = {}
    text_inventory = []

    def font(size, role='text'):
        key = (size, role)
        if key not in fonts:
            fonts[key] = ImageFont.truetype(str(ROOT / f'assets/font_{role}.otf'), round(size * S))
        return fonts[key]

    def text(x, y, value, size, color=INK, role='text', anchor='ls'):
        value = str(value)
        f = font(size, role)
        draw.text((round(x * S), round(y * S)), value, font=f, fill=color, anchor=anchor)
        b = draw.textbbox((round(x * S), round(y * S)), value, font=f, anchor=anchor)
        text_inventory.append({'text': value, 'bbox_px': [round(c / S, 3) for c in b]})

    def line(points, fill, width=1):
        draw.line([(round(x * S), round(y * S)) for x, y in points], fill=fill, width=max(1, round(width * S)))

    def rect(box, fill):
        draw.rectangle(tuple(round(v * S) for v in box), fill=fill)

    # Heading hierarchy follows the reference's editorial typography.
    rect((78, 72, 379, 119), RUST)
    text(228.5, 105, data['visible_notice'], 25, '#ffffff', 'bold', 'ms')
    text(78, 254, data['title'].removeprefix('SYNTHETIC '), 120, INK, 'display')
    text(82, 309, 'Annual page count in tax-law documents', 34, MUTED)
    line([(80, 350), (1315, 350)], '#bfc9bf', 1.4)
    text(83, 399, f"{series['label']} ({series['unit']})", 29, INK, 'bold')
    text(1315, 399, 'Front-face height · linear scale', 27, MUTED, anchor='rs')

    baseline = 1660.0
    scale_height = 1200.0
    axis_max = data['value_axis']['maximum']
    px_per_page = scale_height / axis_max
    centers = np.linspace(244.0, 1244.0, len(values))
    fw, dx, dy = 148.0, 18.0, 12.0
    rect((80, baseline, 1340, 1770), FLOOR)
    for v in [0, 20000, 40000, 60000, 80000]:
        y = baseline - v * px_per_page
        line([(145, y), (1335, y)], '#99a79a' if v == 0 else GRID, 1.5 if v == 0 else 1.0)
        text(127, y + 10, f'{v:,}', 28, MUTED, anchor='rs')

    front = Image.open(ROOT / 'assets/paper_front.png').convert('RGBA')
    side = Image.open(ROOT / 'assets/paper_side.png').convert('RGBA')
    top = Image.open(ROOT / 'assets/paper_top.png').convert('RGBA')
    # A shared texture density keeps paper-edge detail consistent across values.
    texture_px_per_display_px = front.height / (max(values) * px_per_page)
    geometry = []
    mask_dir = ROOT / 'artifacts' / 'front_masks'
    mask_dir.mkdir(parents=True, exist_ok=True)

    def put_surface(texture, x, y, width, height, source_height=None):
        """Front face; y and height are already in supersampled pixel units."""
        if source_height is None:
            source_height = texture.height
        material = texture.transform(
            (width, height), Image.Transform.AFFINE,
            (texture.width / width, 0, 0,
             0, source_height / height, texture.height - source_height),
            Image.Resampling.BICUBIC,
        )
        image.alpha_composite(material, (x, y))
        return material

    for i, (cx, category, value) in enumerate(zip(centers, categories, values)):
        cx = float(cx)
        height = value * px_per_page
        ytop = baseline - height
        x = cx - fw / 2
        x_px, y_px = round(x * S), round(ytop * S)
        w_px, h_px = round(fw * S), round(baseline * S) - y_px
        used_height = h_px / S
        source_h = used_height * texture_px_per_display_px

        # A small contact shadow lies on the common plane, outside data height.
        shadow = Image.new('RGBA', (round((fw + 68) * S), 38 * S), (0, 0, 0, 0))
        sd = ImageDraw.Draw(shadow)
        sd.ellipse((14 * S, 11 * S, (fw + 49) * S, 22 * S), fill=(55, 69, 56, 37))
        shadow = shadow.filter(ImageFilter.GaussianBlur(4 * S))
        image.alpha_composite(shadow, (round((x - 13) * S), round((baseline - 12) * S)))

        # Right face: equal-height parallelogram shifted up by the common depth.
        sw, sy = round(dx * S), round(dy * S)
        side_rect = side.transform((sw, h_px), Image.Transform.AFFINE,
            (side.width / sw, 0, 0, 0, source_h / h_px, side.height - source_h),
            Image.Resampling.BICUBIC)
        # At local x=0, front-edge top is y=dy; at x=dx it is y=0.
        side_warp = side_rect.transform((sw, h_px + sy), Image.Transform.AFFINE,
            (1, 0, 0, sy / sw, 1, -sy), Image.Resampling.BICUBIC)
        image.alpha_composite(side_warp, (x_px + w_px - 1, y_px - sy))

        # The generated front itself supplies every stack's visible paper layers.
        face = put_surface(front, x_px, y_px, w_px, h_px, source_h)
        # Save the actual transformed front alpha, not an idealized target bar.
        alpha = face.getchannel('A')
        center_alpha = np.asarray(alpha)[:, w_px // 2].astype(np.float64)
        alpha.resize((round(fw), max(1, int(np.ceil(used_height)))), Image.Resampling.BOX).save(
            mask_dir / f'{category}.png')

        # Shallow top plane from the generated unprinted sheet, attached exactly
        # to the encoded front edge. It never changes front-face data height.
        tw, th = w_px - 8, sy
        top_rect = top.resize((tw, th), Image.Resampling.LANCZOS)
        top_warp = top_rect.transform((tw + sw, th), Image.Transform.AFFINE,
            (1, sw / th, -sw, 0, 1, 0), Image.Resampling.BICUBIC)
        image.alpha_composite(top_warp, (x_px + 8, y_px - sy))

        # Fine front creases give the very small stacks visible, measurable
        # top and bottom edges, placed inside rather than enlarging the faces.
        edge_x0, edge_x1 = x + 2.6, x + fw - 0.3
        actual_top = y_px / S
        line([(edge_x0, actual_top + 0.45), (edge_x1, actual_top + 0.45)], '#a2a6a0', 0.75)
        line([(edge_x0, baseline - 0.45), (edge_x1, baseline - 0.45)], '#9ea49d', 0.75)
        # Back edge of top sheet, kept much lighter than quantitative front edge.
        line([(edge_x0 + dx, actual_top - dy + 0.2),
              (edge_x1 + dx, actual_top - dy + 0.2)], '#ffffff', 0.6)

        text(cx + 7, actual_top - dy - 22, f'{value:,}', 50, RUST, 'display', 'ms')
        text(cx, 1729, category, 45, INK, 'display', 'ms')
        geometry.append({
            'category': category, 'value': value, 'unit': series['unit'],
            'front_x_center_px': cx, 'front_top_px': actual_top,
            'front_baseline_px': baseline, 'front_width_px': fw,
            'target_height_px': height,
            'transformed_front_center_alpha_height_px': float(center_alpha.sum() / 255.0 / S),
            'rasterized_height_px': used_height,
            'top_projection_dx_px': dx, 'top_projection_dy_px': dy,
            'mask_path': f'artifacts/front_masks/{category}.png',
        })

    text(84, 1729, 'Year', 28, MUTED)
    line([(80, 1810), (1315, 1810)], '#bdc8bc', 1.3)
    text(82, 1860, 'Height = page count. All stacks share the same 0–80,000 scale.', 28, INK)
    text(82, 1904, 'Years are equally spaced; paper layers and perspective are illustrative.', 27, MUTED)
    text(82, 1951, 'Source: supplied synthetic data. Values are not historical statistics.', 25, MUTED)

    # Exactly one full image is written per invocation; captures are wrapper-owned.
    final = image.convert('RGB').resize((W, H), Image.Resampling.LANCZOS)
    final.save(ROOT / 'final.png', optimize=True)
    save_json('artifacts/render_geometry.json', {
        'canvas_px': [W, H], 'supersampling': S, 'scale_px_per_page': px_per_page,
        'axis_range_pages': [0, axis_max], 'baseline_px': baseline,
        'stacks': geometry, 'text_inventory': text_inventory,
        'input_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                         for p in ['input.json', 'brief.json', 'data.json', 'core.json', 'reference.png']},
        'generated_surface_paths': ['assets/paper_front.png', 'assets/paper_side.png', 'assets/paper_top.png'],
    })
    print('Saved final.png: 1395 × 2000; 6 data-bound paper stacks.')


if __name__ == '__main__':
    main()
