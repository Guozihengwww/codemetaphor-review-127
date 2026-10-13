"""Rebuild final.png offline from frozen data and local generated paper surfaces.

Run from this directory with the supplied Python interpreter. Experimental full
render attempts must be launched by capture_render.py, as specified by the user.
"""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent
S = 4
INK = '#252b2d'
MUTED = '#707878'
ACCENT = '#a8422d'
BG = '#f4f3ef'
GRID = '#d8dcda'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def p(v):
    return round(v * S)


def font(size, weight='regular'):
    name = {'regular': 'sans.otf', 'bold': 'sans_bold.otf', 'serif': 'serif.otf'}[weight]
    return ImageFont.truetype(str(ROOT / 'assets/fonts' / name), p(size))


def txt(draw, xy, value, size, fill=INK, weight='regular', anchor='lt'):
    draw.text((p(xy[0]), p(xy[1])), str(value), font=font(size, weight), fill=fill, anchor=anchor)


def track(draw, xy, value, size, spacing, fill, weight='bold'):
    x, y = xy
    f = font(size, weight)
    for ch in value:
        draw.text((p(x), p(y)), ch, font=f, fill=fill, anchor='lt')
        x += draw.textlength(ch, font=f) / S + spacing


def line(draw, points, fill, width=1):
    draw.line([(p(x), p(y)) for x, y in points], fill=fill, width=max(1, p(width)))


def make_side(texture, width, height, depth):
    """Extrude upward: the vertical dimension stays identical to the front."""
    w, h, d = p(width), height, p(depth)
    tex = texture.resize((w, h), Image.Resampling.LANCZOS).convert('RGBA')
    # Inverse affine: source_y = output_y + d*x/w - d.
    warped = tex.transform((w, h + d), Image.Transform.AFFINE,
                           (1, 0, 0, d / w, 1, -d),
                           Image.Resampling.BICUBIC, fillcolor=(0, 0, 0, 0))
    return warped


def make_top(texture, width, dx, dy):
    w, x, y = p(width), p(dx), p(dy)
    tex = texture.resize((w, y), Image.Resampling.LANCZOS).convert('RGBA')
    # Back edge is dx pixels right of the front edge.
    return tex.transform((w + x, y), Image.Transform.AFFINE,
                         (1, x / y, -x, 0, 1, 0),
                         Image.Resampling.BICUBIC, fillcolor=(0, 0, 0, 0))


def main():
    inputs = read_json(ROOT / 'input.json')
    brief = read_json(ROOT / inputs['brief'])
    data = read_json(ROOT / inputs['data'])
    core = read_json(ROOT / inputs['critical_metaphor_contract'])
    assert (ROOT / inputs['reference']).is_file()
    assert core['semantic_roles_zh'] == ['Page count (pages)']
    assert data['value_axis'] == {'minimum': 0, 'maximum': 80000, 'scale': 'linear'}
    assert len(data['series']) == 1
    labels = data['categories']
    series = data['series'][0]
    values = series['values']
    assert len(labels) == len(values) == 6
    assert data['synthetic'] and all(0 <= v <= 80000 for v in values)
    W, H = brief['width'], brief['height']
    canvas = Image.new('RGB', (p(W), p(H)), BG)
    draw = ImageDraw.Draw(canvas)

    # The hierarchy gives the synthetic-data notice its own prominent line.
    draw.rectangle((p(80), p(62), p(91), p(85)), fill=ACCENT)
    track(draw, (108, 61), data['visible_notice'], 25, 2.3, ACCENT)
    txt(draw, (1314, 63), 'ANNUAL DOCUMENT VOLUME', 20, MUTED, anchor='rt')
    display_title = data['title'].removeprefix('SYNTHETIC ')
    txt(draw, (76, 128), display_title, 117, INK, 'bold')
    txt(draw, (81, 271), 'Yearly page counts in tax-law documents', 31, MUTED)
    line(draw, [(80, 325), (1315, 325)], '#bfc6c2', 1)
    txt(draw, (80, 348), f"{series['label']} ({series['unit']})", 25, INK, 'bold')
    txt(draw, (1315, 348), '0–80,000  /  linear scale', 24, MUTED, anchor='rt')

    baseline = 1690.0
    full_height = 1280.0
    axis_max = data['value_axis']['maximum']
    scale = full_height / axis_max
    centers = np.linspace(251, 1221, len(labels)).tolist()
    face_width = 136.0
    side_dx, top_dy = 18.0, 14.0
    plot_left, plot_right = 167, 1314

    # Subtle shared floor, separate from the data-bearing front heights.
    draw.rectangle((p(plot_left), p(baseline-top_dy), p(plot_right), p(baseline+14)), fill='#eceee9')
    for value in [0, 20000, 40000, 60000, 80000]:
        y = baseline - value * scale
        txt(draw, (137, y), f'{value:,}', 23, MUTED, anchor='rm')
        line(draw, [(plot_left, y), (plot_right, y)], '#b9c1bc' if value == 0 else GRID, 1 if value == 0 else 0.7)

    # Material is generated; shape and text come exclusively from frozen data.
    front_source = Image.open(ROOT / 'assets/paper_front.png').convert('RGBA')
    side_source = Image.open(ROOT / 'assets/paper_side.png').convert('RGB')
    top_source = Image.open(ROOT / 'assets/paper_top.png').convert('RGB')
    template_h = p(full_height)
    front_template = front_source.resize((p(face_width), template_h), Image.Resampling.LANCZOS)
    side_template = side_source.resize((p(side_dx), template_h), Image.Resampling.LANCZOS)
    texture_top = make_top(top_source, face_width, side_dx, top_dy)
    geometry = []
    surfaces_dir = ROOT / 'artifacts/front_masks'
    surfaces_dir.mkdir(parents=True, exist_ok=True)

    # Contact shadows do not alter the clipped paper surfaces.
    for center in centers:
        x = center - face_width / 2
        shadow = Image.new('RGBA', (p(face_width + side_dx + 26), p(16)), (0, 0, 0, 0))
        sh = ImageDraw.Draw(shadow)
        sh.ellipse((p(9), p(4), p(face_width + side_dx + 17), p(10)), fill=(57, 61, 56, 22))
        shadow = shadow.filter(ImageFilter.GaussianBlur(p(2)))
        canvas.paste(shadow, (p(x-10), p(baseline-5)), shadow)

    for i, (year, value, center) in enumerate(zip(labels, values, centers)):
        x = center - face_width / 2
        height = value * scale
        top_y = baseline - height
        top_px, bottom_px = p(top_y), p(baseline)
        height_px = bottom_px - top_px
        # Fixed material scale, never a minimum visual height. Each front patch
        # is a direct-height crop from the long generated paper texture.
        available = template_h - height_px
        offset = round(available * [0.21, 0.63, 0.36, 0.81, 0.52, 0.0][i])
        front = front_template.crop((0, offset, p(face_width), offset + height_px))
        side_patch = side_template.crop((0, offset, p(side_dx), offset + height_px))
        side = make_side(side_patch, side_dx, height_px, top_dy)
        canvas.paste(side, (p(x + face_width), top_px - p(top_dy)), side)
        canvas.paste(front, (p(x), top_px), front)
        canvas.paste(texture_top, (p(x), top_px-p(top_dy)), texture_top)
        draw = ImageDraw.Draw(canvas)
        # Thin seams sit inside the numerical face; they add no paper height.
        line(draw, [(x+2, top_px/S+0.32), (x+face_width, top_px/S+0.32)], '#9ca29e', 0.55)
        line(draw, [(x+2, baseline-0.40), (x+face_width, baseline-0.40)], '#777f79', 0.55)
        line(draw, [(x+face_width, top_px/S+0.3), (x+face_width, baseline-0.35)], '#929893', 0.5)
        line(draw, [(x+face_width, top_px/S+0.2), (x+face_width+side_dx-0.3, top_px/S-top_dy+0.2)], '#929893', 0.5)
        line(draw, [(x+side_dx, top_px/S-top_dy+0.2), (x+face_width+side_dx-0.5, top_px/S-top_dy+0.2)], '#f5f4f1', 0.35)

        # Fully controlled values, centered over the corresponding object.
        label_y = top_px/S - top_dy - 56
        label = f'{value:,}'
        f = font(39, 'bold')
        label_width = draw.textlength(label, font=f)/S
        # Ground-color pad gives the value a clear space above faint gridlines.
        draw.rectangle((p(center+side_dx/2-label_width/2-7), p(label_y-5),
                        p(center+side_dx/2+label_width/2+7), p(label_y+43)), fill=BG)
        txt(draw, (center+side_dx/2, label_y), label, 39, ACCENT, 'bold', 'mt')
        txt(draw, (center+side_dx/2, 1725), year, 34, INK, 'bold', 'mt')
        line(draw, [(center+side_dx/2, 1705), (center+side_dx/2, 1713)], '#9ba49d', 1)

        # Small per-object masks document rasterized data-bearing geometry.
        alpha = front.getchannel('A')
        alpha.save(surfaces_dir / f'{year}.png')
        middle = np.asarray(alpha)[:, p(face_width/2)-4:p(face_width/2)+4]
        raster_coverage_height = float(middle.mean(axis=1).sum()/255/S)
        geometry.append({'category': year, 'value': value, 'unit': series['unit'],
                         'center_x': center, 'front_left': x, 'front_right': x+face_width,
                         'baseline_y': baseline, 'target_front_height_px': height,
                         'raster_front_top_y': top_px/S,
                         'raster_front_height_px': raster_coverage_height,
                         'height_error_px': raster_coverage_height-height,
                         'side_projection_dx_px': side_dx, 'top_projection_dy_px': top_dy,
                         'printed_value': label, 'value_label_top_y': label_y})

    # Concise reading key, with perspective explicitly outside the scale.
    line(draw, [(80, 1807), (1315, 1807)], '#bbc4bd', 1)
    track(draw, (80, 1840), 'READING THE STACKS', 20, 1.6, ACCENT)
    txt(draw, (420, 1838), 'Front height encodes page count; each stack starts at zero.', 25, INK)
    txt(draw, (420, 1879), 'Paper layers and top-face perspective are illustrative.', 25, MUTED)
    txt(draw, (80, 1942), 'Synthetic values · not historical tax statistics', 23, MUTED)
    txt(draw, (1315, 1942), 'YEAR  /  equally spaced categories', 23, MUTED, anchor='rt')

    canvas = canvas.resize((W, H), Image.Resampling.LANCZOS)
    canvas.save(ROOT / 'final.png')
    report = {'canvas': {'width': W, 'height': H}, 'series': series['label'],
              'unit': series['unit'], 'axis_minimum': 0, 'axis_maximum': axis_max,
              'scale_px_per_page': scale, 'full_scale_front_height_px': full_height,
              'supersampling': S, 'minimum_mark_height_applied': False,
              'equal_category_spacing_px': centers[1]-centers[0], 'marks': geometry}
    (ROOT / 'artifacts/geometry.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('Wrote final.png at', (W, H), 'from local generated paper assets and frozen data.')


if __name__ == '__main__':
    main()
