"""Offline pencil infographic. Run only through capture_render.py for trials."""
from pathlib import Path
from functools import lru_cache
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent
SCALE = 2
PAPER = '#F5F3ED'
INK = '#233D3F'
MUTED = '#697473'
GRID = '#DDDFD8'


def load_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def rgb(code):
    return np.array([int(code[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)


@lru_cache(None)
def font(size, bold=False):
    return ImageFont.truetype(str(ROOT / 'assets/fonts' / ('Bold.ttf' if bold else 'Regular.ttf')),
                              round(size * SCALE))


def text(draw, xy, content, size, fill=INK, bold=False, anchor='la'):
    draw.text(tuple(round(p * SCALE) for p in xy), content,
              font=font(size, bold), fill=fill, anchor=anchor,
              stroke_width=0)


def line(draw, points, fill, width=1):
    draw.line([(round(x * SCALE), round(y * SCALE)) for x, y in points],
              fill=fill, width=max(1, round(width * SCALE)))


def lettering(draw, xy, content, size=15, spacing=2.0, fill=INK, bold=True):
    x, y = xy
    for char in content:
        text(draw, (x, y), char, size, fill, bold)
        x += draw.textlength(char, font=font(size, bold)) / SCALE + spacing


def recolored_master(color):
    """Tint generated lacquer only, preserving generated cap, metal, wood and graphite."""
    source = np.array(Image.open(ROOT / 'assets/pencil.png').convert('RGBA'))
    arr = source.astype(float)
    h, w = arr.shape[:2]
    x = np.arange(w)[None, :]
    r, g, b = (arr[:, :, i] for i in range(3))
    # Raw pencil paint begins at x=119; its scalloped boundary is at x~1290.
    painted = ((x >= 119) & (x < 1288)) | (
        (x >= 1288) & (x <= 1309) & ((r - g) < 28) & ((r - b) < 73))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    shade = np.clip(luminance / 227.0, 0.30, 1.12) ** 0.95
    tint = rgb(color)[None, None, :] * shade[:, :, None]
    # Retain the small bright specular streak from the generated facets.
    highlight = np.clip((luminance - 237.0) / 18.0, 0, 1) * 0.16
    tint = tint * (1 - highlight[:, :, None]) + 255 * highlight[:, :, None]
    arr[:, :, :3] = np.where(painted[:, :, None], tint, arr[:, :, :3])
    return Image.fromarray(np.clip(arr, 0, 255).astype('uint8'))


def fit_pencil(master, length, height):
    """One data transform: full object length, with fixed-proportion identity end pieces."""
    width = round(length * SCALE)
    target_h = round(height * SCALE)
    source_w, source_h = master.size
    cut_left, cut_right = 160, 1230
    ratio = target_h / source_h
    left_w = round(cut_left * ratio)
    right_w = round((source_w - cut_right) * ratio)
    # No minimum data size. If a future value is tiny, the complete pencil scales down.
    if width <= left_w + right_w:
        return master.resize((max(1, width), target_h), Image.Resampling.LANCZOS)
    body_w = width - left_w - right_w
    result = Image.new('RGBA', (width, target_h))
    for box, position, new_w in [
        ((0, 0, cut_left, source_h), 0, left_w),
        ((cut_left, 0, cut_right, source_h), left_w, body_w),
        ((cut_right, 0, source_w, source_h), left_w + body_w, right_w),
    ]:
        patch = master.crop(box).resize((new_w, target_h), Image.Resampling.LANCZOS)
        result.alpha_composite(patch, (position, 0))
    return result


def paste_with_shadow(canvas, piece, xy, strength=22, blur=2.2, offset=(0, 2)):
    x, y = (round(n * SCALE) for n in xy)
    padding = round(blur * SCALE * 4)
    shadow = Image.new('RGBA', (piece.width + 2 * padding, piece.height + 2 * padding))
    alpha = Image.new('L', shadow.size)
    alpha.paste(piece.getchannel('A'), (padding, padding))
    alpha = alpha.filter(ImageFilter.GaussianBlur(blur * SCALE))
    alpha = alpha.point(lambda v: round(v * strength / 255))
    shadow.paste((22, 28, 24), (0, 0, *shadow.size), alpha)
    canvas.alpha_composite(shadow, (x - padding + round(offset[0] * SCALE),
                                    y - padding + round(offset[1] * SCALE)))
    canvas.alpha_composite(piece, (x, y))


def shaving(canvas, name, center, width, angle):
    source = Image.open(ROOT / f'assets/{name}.png').convert('RGBA')
    source = source.resize((round(width * SCALE),
                            round(width * SCALE * source.height / source.width)),
                           Image.Resampling.LANCZOS)
    source = source.rotate(angle, Image.Resampling.BICUBIC, expand=True)
    xy = (center[0] - source.width / (2 * SCALE), center[1] - source.height / (2 * SCALE))
    paste_with_shadow(canvas, source, xy, strength=25, blur=3, offset=(1, 3))


def main():
    inputs = load_json('input.json')
    brief = load_json(inputs['brief'])
    data = load_json(inputs['data'])
    core = load_json(inputs['critical_metaphor_contract'])
    assert core['semantic_roles_zh'] == ['Annual hours (h/year)']
    assert data['synthetic'] and data['value_axis']['minimum'] == 0
    assert data['value_axis']['scale'] == 'linear'
    assert len(data['series']) == 1
    series = data['series'][0]
    assert series['unit'] == 'h/year'
    labels, values = data['categories'], series['values']
    assert len(labels) == len(values)
    assert sum(values) == data['series_totals'][series['id']]
    width, height = brief['width'], brief['height']
    canvas = Image.new('RGBA', (width * SCALE, height * SCALE), PAPER)
    draw = ImageDraw.Draw(canvas)

    # Quiet editorial header; all words and data labels are code-set.
    draw.rounded_rectangle((74 * SCALE, 50 * SCALE, 326 * SCALE, 89 * SCALE),
                           radius=5 * SCALE, fill=INK)
    lettering(draw, (91, 58), data['visible_notice'], size=17, spacing=1.8, fill='#FFFFFF')
    lettering(draw, (351, 60), f'{len(labels)} FICTIONAL COUNTRIES', size=16, spacing=1.6,
              fill=MUTED, bold=False)
    text(draw, (72, 103), 'Annual working hours', 82, bold=True)
    text(draw, (77, 211), 'Hours worked per person, per year.', 26, MUTED)
    line(draw, [(75, 268), (1760, 268)], '#B8C5BE', 1)

    # Decorative wood curls stay outside the quantitative drawing region.
    shaving(canvas, 'shaving_fan', (1910, 127), 272, -19)
    shaving(canvas, 'shaving_ribbon', (1980, 466), 178, 17)
    shaving(canvas, 'shaving_spiral', (1960, 808), 267, -12)
    shaving(canvas, 'shaving_fan', (1952, 1125), 260, 22)
    shaving(canvas, 'shaving_ribbon', (1844, 1344), 106, -57)
    draw = ImageDraw.Draw(canvas)

    x0, span = 286.0, 1440.0
    maximum = data['value_axis']['maximum']
    first_y, last_y = 382.0, 1198.0
    pitch = (last_y - first_y) / (len(labels) - 1) if len(labels) > 1 else 0
    pencil_h = 34.0

    lettering(draw, (75, 307), 'H/YEAR', size=16, spacing=1.2, fill=MUTED)
    text(draw, (x0, 284), 'Annual hours (h/year)', 18, MUTED)
    text(draw, (1760, 284), 'Full pencil length, from butt to graphite tip', 18, MUTED, anchor='ra')
    for tick in [0, 500, 1000, 1500, 2000, maximum]:
        x = x0 + span * tick / maximum
        tick_text = f'{tick:,}'
        text(draw, (x, 316), tick_text, 17, MUTED, anchor='ma')
        line(draw, [(x, 343), (x, 1224)], GRID if tick else '#BBC7BE', 0.8 if tick else 1.3)
    line(draw, [(x0, 343), (x0 + span, 343)], '#CBD2CA', 1)

    colors = ['#D5A23B', '#357B9B', '#C96250', '#D7AE3F', '#B95748',
              '#86A566', '#D18437', '#C59B37', '#36658B', '#47999A',
              '#477F9A', '#2E8498', '#93A34D', '#B95667', '#3A9172',
              '#6E8494', '#426C8E']
    geometry = []
    for index, (category, value) in enumerate(zip(labels, values)):
        center_y = first_y + pitch * index
        length = span * value / maximum
        color = colors[index % len(colors)]
        pencil = fit_pencil(recolored_master(color), length, pencil_h)
        paste_with_shadow(canvas, pencil, (x0, center_y - pencil_h / 2),
                          strength=24, blur=1.7, offset=(0, 2))
        draw = ImageDraw.Draw(canvas)
        text(draw, (221, center_y - 1), f'{value:,}', 27, INK, anchor='rm')
        label_ink = '#263E3C' if index in [0, 3, 5, 6, 7, 12] else '#FFFFFF'
        text(draw, (348, center_y - 1), category, 20, label_ink, bold=True, anchor='lm')
        arr = np.array(pencil.getchannel('A'))
        rows, cols = np.where(arr >= 128)
        geometry.append({
            'category': category, 'annual_hours': value, 'unit': series['unit'],
            'center_y_px': center_y, 'target_start_x_px': x0,
            'target_tip_x_px': x0 + length, 'target_length_px': length,
            'alpha_50pct_start_x_px': x0 + float(cols.min()) / SCALE,
            'alpha_50pct_tip_x_px': x0 + float(cols.max() + 1) / SCALE,
            'color': color,
        })

    line(draw, [(75, 1263), (1760, 1263)], '#B8C5BE', 1)
    text(draw, (75, 1285), 'HOW TO READ', 16, INK, bold=True)
    text(draw, (222, 1281), 'Each pencil starts at zero. Its complete length represents annual working hours.',
         20, MUTED)
    text(draw, (75, 1332), 'SYNTHETIC DATA  ·  Fictional countries; not real-world statistics.', 18, MUTED)
    text(draw, (1760, 1332), 'Category order preserved', 17, MUTED, anchor='ra')

    output = canvas.convert('RGB').resize((width, height), Image.Resampling.LANCZOS)
    output.save(ROOT / 'final.png')
    (ROOT / 'artifacts').mkdir(exist_ok=True)
    (ROOT / 'artifacts/render_geometry.json').write_text(json.dumps({
        'canvas_px': [width, height], 'axis_minimum': 0, 'axis_maximum': maximum,
        'pixels_per_hour': span / maximum, 'pencil_height_px': pencil_h,
        'source_asset': 'assets/pencil.png', 'rows': geometry,
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
