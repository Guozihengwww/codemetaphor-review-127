#!/usr/bin/env python3
"""Offline, data-bound ocean infographic. Run with no arguments.

Water material and yacht are cropped from the retained built-in image output.
Pixel-center sampling and linear interpolation share one physical data mapping.
No minimum band thickness, overshooting splines, or model calls are used.
"""
from pathlib import Path
import json
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
S = 3
BG = '#f7faf8'
INK = '#173b48'
MUTED = '#526f79'
GRID = '#d7e2e1'
COLORS = ['#37b98a', '#258ade', '#253e82', '#9866bf']
# Low and high luminance endpoints keep the four sources distinct.
TINTS = [('#13694e', '#93f6cf'), ('#1554b0', '#71c5ff'),
         ('#121c49', '#546aaa'), ('#58377f', '#c6a1e6')]


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def write_json(path, obj):
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def rgb(hex_color):
    return tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))


def main():
    inputs = read_json('input.json')
    brief = read_json(inputs['brief'])
    data = read_json(inputs['data'])
    core = read_json(inputs['critical_metaphor_contract'])
    assert (ROOT / inputs['reference']).is_file()
    assert data['synthetic'] and data['visible_notice'] == 'SYNTHETIC DATA'
    assert data['value_axis'] == {'minimum': 0, 'maximum': 7200, 'scale': 'linear'}
    assert len(core['semantic_roles_zh']) == len(data['series']) == 4
    W, H = brief['width'], brief['height']
    categories = data['categories']
    time = np.array(data['time_points'], dtype=float)
    values = np.array([s['values'] for s in data['series']], dtype=float)
    assert values.shape == (4, len(categories)) and np.all(values >= 0)
    assert np.all(np.diff(time) > 0)
    for s in data['series']:
        assert sum(s['values']) == data['series_totals'][s['id']]
    cum = np.vstack([np.zeros(len(categories)), values.cumsum(axis=0)])
    totals = cum[-1]
    assert np.max(totals) <= 7200

    # One shared mapping. Increasing discharge points vertically downward.
    x0, x1, y0, plot_h = 180.0, 2110.0, 376.0, 720.0
    scale = plot_h / 7200.0
    y_max = y0 + plot_h
    xs = x0 + (time - time[0]) / (time[-1] - time[0]) * (x1 - x0)
    bounds = y0 + cum * scale
    canvas = Image.new('RGB', (W * S, H * S), BG)
    draw = ImageDraw.Draw(canvas)
    fonts = {}
    text_inventory = []

    def font(size, face='sans'):
        fn = {'sans': 'NimbusSans-Regular.otf', 'bold': 'NimbusSans-Bold.otf',
              'serif': 'NimbusRoman-Regular.otf', 'serif_bold': 'NimbusRoman-Bold.otf'}[face]
        key = (size, face)
        if key not in fonts:
            fonts[key] = ImageFont.truetype(str(ROOT / 'assets/fonts' / fn), round(size * S))
        return fonts[key]

    def text(x, y, value, size=25, fill=INK, face='sans', anchor='lt', role='text'):
        value = str(value)
        bbox = draw.textbbox((round(x * S), round(y * S)), value,
                             font=font(size, face), anchor=anchor)
        draw.text((round(x * S), round(y * S)), value, font=font(size, face),
                  fill=fill, anchor=anchor)
        text_inventory.append({'text': value, 'role': role,
                               'bbox': [round(v / S, 2) for v in bbox]})

    def line(points, fill=GRID, width=1):
        draw.line([(round(x * S), round(y * S)) for x, y in points],
                  fill=fill, width=max(1, round(width * S)))

    def rect(box, fill, radius=0, outline=None, width=1):
        coords = [round(v * S) for v in box]
        if radius:
            draw.rounded_rectangle(coords, round(radius * S), fill=fill,
                                   outline=outline, width=round(width * S))
        else:
            draw.rectangle(coords, fill=fill, outline=outline, width=round(width * S))

    # Editorial header and a dedicated legend leave the data surface unobstructed.
    rect((80, 42, 387, 84), INK, radius=5)
    text(99, 53, data['visible_notice'], 24, '#ffffff', 'bold', role='synthetic notice')
    text(2120, 54, '16 ordered periods  /  4 plastic sources', 24, MUTED,
         anchor='rt', role='scope')
    text(80, 109, 'Plastic discharge into the ocean', 88, INK, 'serif_bold', role='title')
    text(84, 212, 'Amount discharged in each period  ·  100 tonnes/period', 28, MUTED,
         role='metric and unit')
    for k, series in enumerate(data['series']):
        lx = 84 + 337 * k
        rect((lx, 274, lx + 32, 292), COLORS[k], radius=2)
        text(lx + 46, 270, series['label'], 27, INK, role='legend')

    # Tick/grid geometry is shared with the data material.
    for tick in range(0, 7201, 1200):
        yy = y0 + tick * scale
        if tick:
            line([(x0 - 8, yy), (x1 + 1, yy)], GRID, 0.8)
        text(x0 - 24, yy, f'{tick:,}', 23, MUTED, anchor='rm', role='value tick')
    text(83, 326, 'Discharge ↓', 22, MUTED, role='axis direction')
    for i, xx in enumerate(xs):
        line([(xx, y0 + 1), (xx, y_max)], '#e4ebe8', 0.65)
        line([(xx, y_max + 5), (xx, y_max + 12)], '#b5c6c8', 0.8)
        text(xx, y_max + 23, categories[i], 24, INK, anchor='mt', role='period tick')

    # Continuous generated water texture, with per-pixel code-controlled layer masks.
    pw, ph = round((x1 - x0) * S), round(plot_h * S)
    texture = Image.open(ROOT / 'assets/water_texture.png').convert('RGB')
    texture = texture.resize((pw, ph), Image.Resampling.LANCZOS)
    lum = np.asarray(texture, dtype=np.float32)
    lum = lum[:, :, 0] * .2126 + lum[:, :, 1] * .7152 + lum[:, :, 2] * .0722
    low, high = np.percentile(lum, [1, 99])
    lum = np.clip((lum - low) / (high - low), 0, 1) ** .88
    sample_x = x0 + (np.arange(pw) + .5) / S
    sample_y = (np.arange(ph) + .5) / S
    sampled_cum = np.array([np.interp(sample_x, xs, row) for row in cum]) * scale
    water = np.zeros((ph, pw, 4), dtype=np.uint8)
    area_by_layer = []
    for k in range(4):
        mask = ((sample_y[:, None] >= sampled_cum[k][None, :]) &
                (sample_y[:, None] < sampled_cum[k + 1][None, :]))
        lo, hi = np.array(rgb(TINTS[k][0])), np.array(rgb(TINTS[k][1]))
        for c in range(3):
            plane = lo[c] + (hi[c] - lo[c]) * lum
            water[:, :, c][mask] = np.round(plane[mask]).astype(np.uint8)
        water[:, :, 3][mask] = 255
        area_by_layer.append(int(np.count_nonzero(mask)) / S**2)
    water_image = Image.fromarray(water, 'RGBA')
    canvas.paste(water_image, (round(x0 * S), round(y0 * S)), water_image)
    del water, lum, texture

    # Restrained guides are clipped inside the material; they add no data thickness.
    guide = Image.new('RGBA', (W * S, H * S), (0, 0, 0, 0))
    gd = ImageDraw.Draw(guide)
    for i, xx in enumerate(xs):
        if i in (0, len(xs) - 1):
            continue
        gd.line([(round(xx * S), round((y0 + 1) * S)),
                 (round(xx * S), round(bounds[-1, i] * S) - 1)],
                fill=(255, 255, 255, 37), width=2)
    for tick in range(1200, 7201, 1200):
        yy = y0 + tick * scale
        # Draw into a local layer, then use the exact visible ocean alpha as a clip.
        grid_layer = Image.new('RGBA', (pw, ph), (0, 0, 0, 0))
        ld = ImageDraw.Draw(grid_layer)
        ld.line([(0, round((yy - y0) * S)), (pw, round((yy - y0) * S))],
                fill=(255, 255, 255, 36), width=2)
        clip = np.array(grid_layer.getchannel('A'))
        clip[np.asarray(water_image.getchannel('A')) == 0] = 0
        grid_layer.putalpha(Image.fromarray(clip))
        guide.alpha_composite(grid_layer, (round(x0 * S), round(y0 * S)))
    canvas = Image.alpha_composite(canvas.convert('RGBA'), guide).convert('RGB')
    draw = ImageDraw.Draw(canvas)

    # The zero line and all of the vessel remain at or above the measured surface.
    yacht = Image.open(ROOT / 'assets/yacht.png').convert('RGBA')
    yacht_w = 545
    yacht_h = yacht.height * yacht_w / yacht.width
    yacht = yacht.resize((round(yacht_w * S), round(yacht_h * S)), Image.Resampling.LANCZOS)
    yacht_pos = (1565, y0 - yacht_h)
    canvas.paste(yacht, (round(yacht_pos[0] * S), round(yacht_pos[1] * S)), yacht)
    line([(x0, y0 - .5), (x1, y0 - .5)], '#407776', .7)
    text(190, y0 - 24, 'Common water surface = 0', 21, MUTED, role='zero surface label')

    # Direct total annotation, placed entirely in the empty region below the taper.
    end_y = bounds[-1, -1]
    line([(x1, end_y), (x1, 554), (1760, 554)], '#8ca8ac', 1)
    text(1720, 580, 'P16 total discharge', 25, MUTED, role='final total annotation')
    text(1716, 623, f'{int(totals[-1]):,}', 106, INK, 'serif', role='final total number')
    text(1721, 738, '100 tonnes/period', 24, MUTED, role='final total unit')
    text(1721, 779, 'A + B + C + D', 23, MUTED, role='total definition')

    # A short reading guide occupies the naturally empty part of the section.
    rect((1373, 894, 1425, 899), INK)
    text(1373, 921, 'Read one water column at a time', 29, INK, 'bold', role='reading guide title')
    text(1373, 966, 'Band thickness = discharge from one plastic source.', 25, MUTED,
         role='encoding explanation')
    text(1373, 1005, 'Total depth = all four sources in the same period.', 25, MUTED,
         role='encoding explanation')
    text(1373, 1044, 'Periods are not accumulated over time.', 25, MUTED,
         role='encoding explanation')

    # Full numeric evidence, split into two aligned eight-period tables.
    line([(80, 1180), (2120, 1180)], '#aebfc2', 1)
    text(80, 1195, 'All period values', 30, INK, 'bold', role='table title')
    text(351, 1201, '100 tonnes/period', 24, MUTED, role='table unit')
    peak = int(np.argmax(totals))
    text(2120, 1200, f'Highest total: {categories[peak]} · {int(totals[peak]):,}', 24,
         MUTED, anchor='rt', role='peak total annotation')
    for start, left, width in [(0, 80, 992), (8, 1140, 980)]:
        top = 1241
        label_w, row_h = 225, 39
        col_w = (width - label_w) / 8
        rect((left, top, left + width, top + 38), '#e6efeb', radius=3)
        text(left + 17, top + 9, 'Period', 24, MUTED, 'bold', role='table header')
        for j in range(8):
            cx = left + label_w + (j + .5) * col_w
            text(cx, top + 9, categories[start + j], 24, INK, 'bold', anchor='mt',
                 role='table period header')
        for row in range(5):
            ry = top + 40 + row * row_h
            if row == 4:
                rect((left, ry - 1, left + width, ry + row_h - 3), '#e8eff1', radius=3)
            elif row in (1, 3):
                rect((left, ry - 1, left + width, ry + row_h - 3), '#f0f5f1')
            if row < 4:
                rect((left + 17, ry + 11, left + 29, ry + 24), COLORS[row], radius=1)
                label = data['series'][row]['label']
                numbers = values[row]
                face = 'sans'
            else:
                label, numbers, face = 'Total', totals, 'bold'
            text(left + 39, ry + 7, label, 24, INK, face, role='table row label')
            for j in range(8):
                cx = left + label_w + (j + .5) * col_w
                text(cx, ry + 7, f'{int(numbers[start + j]):,}', 24, INK, face,
                     anchor='mt', role='numeric table value')
    text(80, 1501, 'SYNTHETIC DATA · Periods P01–P16 are ordered synthetic categories; these are not real-world estimates.',
         22, MUTED, role='synthetic scope note')
    text(2120, 1501, 'Linear scale · 0–7,200', 22, MUTED, anchor='rt', role='scale note')

    # Native-size export is the only full candidate produced by this renderer.
    canvas = canvas.resize((W, H), Image.Resampling.LANCZOS)
    canvas.save(ROOT / 'final.png')
    write_json('artifacts/render_geometry.json', {
        'width': W, 'height': H, 'supersampling': S,
        'plot': {'left': x0, 'right': x1, 'zero_y': y0, 'max_y': y_max,
                 'min_value': 0, 'max_value': 7200, 'px_per_unit': scale},
        'interpolation': 'piecewise linear in actual time_points; no minimum thickness',
        'categories': categories, 'series_ids': [s['id'] for s in data['series']],
        'x': xs.tolist(), 'cumulative_values': cum.tolist(), 'boundary_y': bounds.tolist(),
        'thickness_px': (values * scale).tolist(), 'totals': totals.astype(int).tolist(),
        'rasterized_area_px': area_by_layer, 'tints': TINTS,
        'yacht_bbox': [yacht_pos[0], yacht_pos[1], yacht_pos[0] + yacht_w, y0],
        'font_substitution': 'Bundled Nimbus Roman and Nimbus Sans (installed URW fonts).',
    })
    outside = [v for v in text_inventory if v['bbox'][0] < 0 or v['bbox'][1] < 0
               or v['bbox'][2] > W or v['bbox'][3] > H]
    write_json('artifacts/text_inventory.json', {'items': text_inventory, 'outside_canvas': outside})
    print(json.dumps({'size': [W, H], 'periods': len(categories), 'sources': len(values),
                      'numeric_table_values': 80, 'text_outside_canvas': len(outside),
                      'min_band_px': float(values.min() * scale)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
