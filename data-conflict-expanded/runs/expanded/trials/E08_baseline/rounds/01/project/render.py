"""Offline data-bound egg infographic. Run via capture_render.py in this experiment."""
from pathlib import Path
import hashlib
import json
import math
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent
S = 2
INK = '#263C34'
MUTED = '#60756A'
BACKGROUND = (242, 247, 241)
PALETTE = np.array([
    (91, 111, 79),      # Vitamins, minerals and salt: olive
    (148, 94, 54),      # Polyunsaturated fat: brown
    (214, 133, 62),     # Saturated fat: amber
    (215, 188, 103),    # Monounsaturated fat: gold
    (208, 138, 109),    # Protein: terracotta
    (218, 179, 135),    # Water: eggshell ochre
], dtype=np.float32)


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def save_json(path, value):
    (ROOT / path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def font(size, bold=False):
    name = 'NimbusSans-Bold.otf' if bold else 'NimbusSans-Regular.otf'
    return ImageFont.truetype(str(ROOT / 'assets/fonts' / name), round(size * S))


def text(draw, xy, value, size, color=INK, bold=False, anchor='lt'):
    f = font(size, bold)
    pos = (round(xy[0] * S), round(xy[1] * S))
    draw.text(pos, value, font=f, fill=color, anchor=anchor, stroke_width=0)
    b = draw.textbbox(pos, value, font=f, anchor=anchor)
    return [round(v / S, 2) for v in b]


def line(draw, points, color='#71847A', width=1.8):
    draw.line([(round(x * S), round(y * S)) for x, y in points], fill=color,
              width=max(1, round(width * S)), joint='curve')


def wrap(draw, value, size, max_width):
    words = value.split()
    result = []
    row = ''
    f = font(size)
    for word in words:
        attempt = (row + ' ' + word).strip()
        if row and draw.textlength(attempt, font=f) > max_width * S:
            result.append(row)
            row = word
        else:
            row = attempt
    result.append(row)
    return result


def ray_boundary(alpha, p, theta):
    """Find the generated silhouette's actual outer boundary along a ray."""
    h, w = alpha.shape
    radii = np.arange(0, math.hypot(h, w), 0.25)
    xs = np.rint(p[0] + np.sin(theta) * radii).astype(int)
    ys = np.rint(p[1] - np.cos(theta) * radii).astype(int)
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    inside = np.zeros_like(valid)
    inside[valid] = alpha[ys[valid], xs[valid]] >= 0.5
    r = float(radii[np.flatnonzero(inside)[-1]])
    return np.array([p[0] + np.sin(theta) * r, p[1] - np.cos(theta) * r]), r


def build_egg(values):
    source = Image.open(ROOT / 'assets/egg_shell.png').convert('RGBA')
    eh = 1180
    ew = round(eh * source.width / source.height)
    ex = round(1040 - ew / 2)
    ey = 410
    shell = source.resize((ew * S, eh * S), Image.Resampling.LANCZOS)
    a = np.asarray(shell)
    alpha = a[:, :, 3].astype(np.float32) / 255
    h, w = alpha.shape
    # One shared internal point; all angles are determined from cumulative AREA.
    pivot = np.array([w * 0.505, h * 0.565])
    yy, xx = np.mgrid[:h, :w]
    start = -0.052
    theta = np.mod(np.arctan2(xx + 0.5 - pivot[0], -(yy + 0.5 - pivot[1])) - start, 2 * np.pi)
    use = alpha > 0
    angles = theta[use]
    weights = alpha[use].astype(np.float64)
    order = np.argsort(angles)
    cum_area = np.cumsum(weights[order])
    total_area = float(cum_area[-1])
    cut_targets = np.cumsum(values)[:-1] / sum(values) * total_area
    positions = np.searchsorted(cum_area, cut_targets)
    ordered_angles = angles[order]
    cuts = (ordered_angles[positions] + ordered_angles[np.maximum(positions - 1, 0)]) / 2
    region = np.searchsorted(cuts, theta, side='right').astype(np.uint8)
    bounds = np.r_[0, cuts, 2 * np.pi]

    # Generated luminance, including pores and speckles, supplies the full egg's material.
    lum = a[:, :, :3].astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    shade = 0.25 + 0.95 * lum / 255
    rgb = np.clip(PALETTE[region] * shade[:, :, None], 0, 255)
    egg = Image.fromarray(np.dstack([np.rint(rgb).astype(np.uint8), a[:, :, 3]]), 'RGBA')

    # Measure screen-resolution coverage after the SAME raster reduction as the visible egg.
    masks = []
    shares = []
    anchors = []
    arcs = []
    for i in range(len(values)):
        layer = Image.fromarray(np.rint(alpha * (region == i) * 255).astype(np.uint8))
        m = np.asarray(layer.resize((ew, eh), Image.Resampling.LANCZOS)).astype(np.float32) / 255
        masks.append(m)
        shares.append(float(m.sum()))
        mid = (bounds[i] + bounds[i + 1]) / 2 + start
        pt, rad = ray_boundary(alpha, pivot, mid)
        anchors.append((pt / S + [ex, ey]).tolist())
        # Approximate actual external contour length, not center-angle percentage.
        t = np.linspace(bounds[i] + start, bounds[i + 1] + start, 350 if i == 5 else 120)
        contour = np.array([ray_boundary(alpha, pivot, th)[0] / S for th in t])
        arcs.append(float(np.linalg.norm(np.diff(contour, axis=0), axis=1).sum()))

    masks = np.stack(masks)
    shares = np.asarray(shares) / sum(shares) * 100
    alpha_small = np.asarray(shell.getchannel('A').resize((ew, eh), Image.Resampling.LANCZOS)).astype(np.float32) / 255
    np.savez_compressed(ROOT / 'artifacts/geometry_masks.npz', alpha=alpha_small, regions=masks,
                        bbox=np.array([ex, ey, ew, eh]), palette=PALETTE)
    geometry = {
        'encoding': 'Visible screen area / full generated egg silhouette area',
        'canvas_bbox_xywh': [ex, ey, ew, eh],
        'shared_pivot_xy': (pivot / S + [ex, ey]).tolist(),
        'clockwise_category_indices': list(range(len(values))),
        'angle_boundaries_radians_clockwise_from_up': (bounds + start).tolist(),
        'screen_coverage_area_px2': float(sum(m.sum() for m in masks)),
        'target_percentages': [float(v) for v in values],
        'measured_screen_area_percentages': shares.tolist(),
        'absolute_error_percentage_points': np.abs(shares - values).tolist(),
        'outer_arc_lengths_px': arcs,
        'anchors_xy': anchors,
        'minimum_size_adjustment': False,
        'missing_pixels_above_half_coverage': int(np.count_nonzero((alpha_small > .5) & (masks.sum(axis=0) < .5))),
        'coverage_reduction_max_difference': float(np.max(np.abs(masks.sum(axis=0) - alpha_small))),
        'partition_method': 'Weighted quantiles of pixel polar angles inside the actual alpha mask; no direct angle-to-percentage mapping.',
    }
    return egg, geometry, anchors


def main():
    inputs = read_json('input.json')
    brief = read_json(inputs['brief'])
    data = read_json(inputs['data'])
    core = read_json(inputs['critical_metaphor_contract'])
    assert core['topic_zh'] == data['topic_zh']
    assert len(data['series']) == 1 and data['series'][0]['id'] == 'composition'
    values = np.asarray(data['series'][0]['values'], dtype=float)
    categories = data['categories']
    assert len(values) == len(categories) == len(PALETTE)
    assert np.all(values > 0) and abs(sum(values) - 100) < 1e-8
    width, height = brief['width'], brief['height']
    (ROOT / 'artifacts').mkdir(exist_ok=True)
    canvas = Image.new('RGB', (width * S, height * S), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    text_boxes = []

    # Editorial header; mandatory title and synthetic notice are code-rendered.
    text_boxes.append(text(draw, (125, 102), 'Egg nutrient composition', 87, bold=True))
    text_boxes.append(text(draw, (130, 218), 'Percent of total egg composition', 31, color=MUTED))
    pill = (1500 * S, 103 * S, 1875 * S, 175 * S)
    draw.rounded_rectangle(pill, radius=36 * S, fill=INK)
    text_boxes.append(text(draw, (1687.5, 139), data['visible_notice'], 27, '#FFFFFF', bold=True, anchor='mm'))
    line(draw, [(130, 287), (1875, 287)], color='#CBD7CD', width=1.1)

    egg, geometry, anchors = build_egg(values)
    ex, ey, ew, eh = geometry['canvas_bbox_xywh']
    shadow = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.ellipse(((ex + ew * .12) * S, (ey + eh - 13) * S,
                (ex + ew * .90) * S, (ey + eh + 42) * S), fill=(59, 81, 59, 42))
    shadow = shadow.filter(ImageFilter.GaussianBlur(22 * S))
    canvas.paste(shadow, (0, 0), shadow)
    canvas.paste(egg, (ex * S, ey * S), egg)
    draw = ImageDraw.Draw(canvas)

    # Leaders start just outside the actual silhouette, and never replace data areas.
    p = np.asarray(geometry['shared_pivot_xy'])
    outside = []
    for a in anchors:
        v = np.asarray(a) - p
        outside.append(np.asarray(a) + v / np.linalg.norm(v) * 8)
    # Water's callout uses its left side, which is inside the same (last) sector.
    water_anchor = np.array([ex + 10, ey + eh * .62])
    # Find the exact first occupied x at this row rather than assuming the bounding box edge.
    with np.load(ROOT / 'artifacts/geometry_masks.npz') as g:
        row = round(eh * .62)
        water_anchor[0] = ex + np.flatnonzero(g['alpha'][row] > .5)[0]
    water_outside = water_anchor + [-9, 0]

    a0, a1, a2, a3, a4, _ = outside
    leaders = [
        [a0.tolist(), [a0[0] - 48, 355], [635, 355], [535, 465]],
        [a1.tolist(), [a1[0] + 10, 347], [1295, 347], [1340, 387]],
        [a2.tolist(), [a2[0] + 60, a2[1] - 24], [1485, 532], [1540, 592]],
        [a3.tolist(), [a3[0] + 38, a3[1] - 10], [1510, 782], [1550, 840]],
        [a4.tolist(), [1515, a4[1] + 16], [1515, 1168], [1550, 1168]],
        [water_outside.tolist(), [557, water_outside[1]], [510, 1155]],
    ]
    for i, points in enumerate(leaders):
        line(draw, points, width=1.6)
        ax, ay = points[0]
        draw.ellipse(((ax - 3.5) * S, (ay - 3.5) * S, (ax + 3.5) * S, (ay + 3.5) * S),
                     fill=tuple(PALETTE[i].astype(int)))

    layout = [
        {'x': 132, 'y': 433, 'width': 365, 'size': 31, 'value_size': 68},
        {'x': 1370, 'y': 370, 'width': 470, 'size': 31, 'value_size': 68},
        {'x': 1570, 'y': 573, 'width': 310, 'size': 31, 'value_size': 68},
        {'x': 1580, 'y': 817, 'width': 310, 'size': 31, 'value_size': 68},
        {'x': 1580, 'y': 1148, 'width': 300, 'size': 34, 'value_size': 91},
        {'x': 130, 'y': 1068, 'width': 405, 'size': 38, 'value_size': 116},
    ]
    labels = []
    for i, spec in enumerate(layout):
        display = categories[i]
        if display == 'Vitamins minerals and salt':
            display = 'Vitamins, minerals and salt'
        rows = wrap(draw, display, spec['size'], spec['width'])
        y = spec['y']
        boxes = []
        for row in rows:
            boxes.append(text(draw, (spec['x'], y), row, spec['size']))
            y += spec['size'] * 1.17
        y += 14
        percent = f'{values[i]:.1f}%'
        # Dark ink keeps small values equally readable regardless of their fill lightness.
        boxes.append(text(draw, (spec['x'] - 2, y), percent, spec['value_size'], bold=True))
        text_boxes.extend(boxes)
        labels.append({'category': categories[i], 'display_lines': rows, 'numeric_label': percent,
                       'unit': data['series'][0]['unit'], 'boxes_xyxy': boxes})

    line(draw, [(130, 1680), (1875, 1680)], color='#CBD7CD', width=1.1)
    text_boxes.append(text(draw, (131, 1711), 'COLORED AREA = SHARE OF THE WHOLE EGG', 23, color=INK, bold=True))
    text_boxes.append(text(draw, (131, 1750), 'Illustrative synthetic composition. All six regions sum to 100%.', 24, color=MUTED))
    text_boxes.append(text(draw, (1873, 1728), f'TOTAL  {sum(values):.1f}%', 29, color=INK, bold=True, anchor='rt'))

    final = canvas.resize((width, height), Image.Resampling.LANCZOS)
    final.save(ROOT / 'final.png', optimize=True)
    geometry['labels'] = labels
    geometry['leaders_xy'] = leaders
    geometry['all_text_boxes_within_canvas'] = all(b[0] >= 0 and b[1] >= 0 and b[2] <= width and b[3] <= height for b in text_boxes)
    geometry['final_png_sha256'] = hashlib.sha256((ROOT / 'final.png').read_bytes()).hexdigest()
    save_json('artifacts/geometry.json', geometry)
    index_path = ROOT / 'rounds/index.json'
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding='utf-8'))
        if index and index[-1]['status'] == 'running':
            shutil.copy2(ROOT / 'final.png', ROOT / 'artifacts' / f"candidate_{index[-1]['number']:02d}.png")
    print(json.dumps({'size': final.size, 'area_percentages': geometry['measured_screen_area_percentages'],
                      'max_area_error_pp': max(geometry['absolute_error_percentage_points'])}))


if __name__ == '__main__':
    main()
