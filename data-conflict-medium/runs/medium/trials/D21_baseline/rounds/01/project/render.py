"""Regenerate final.png from frozen local data and local image assets.

Every complete execution for this experiment must be launched by capture_render.py.
The renderer itself has no network or model calls and requires no arguments.
"""
from pathlib import Path
from decimal import Decimal
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent
S = 3
PAPER = (241, 240, 231)
INK = (37, 40, 33)
MUTED = (107, 107, 94)
GOLD = (139, 111, 62)
FONT_FILES = {
    'title': 'assets/fonts/title.otf',
    'sans': 'assets/fonts/sans.otf',
    'serif': 'assets/fonts/serif.ttf',
    'bold': 'assets/fonts/serif-bold.ttf',
}


def main():
    data = json.loads((ROOT / 'data.json').read_text(encoding='utf-8'))
    brief = json.loads((ROOT / 'brief.json').read_text(encoding='utf-8'))
    W, H = brief['width'], brief['height']
    categories = data['categories']
    values = [Decimal(str(v)) for v in data['series'][0]['values']]
    assert len(categories) == len(values) == 7
    assert sum(values) == Decimal('100')
    assert data['value_axis'] == {'minimum': 0, 'maximum': 100, 'scale': 'linear'}
    label_values = [f'{v}%' for v in values]

    # Low-amplitude paper grain, deterministic and subordinate to the objects.
    rng = np.random.default_rng(613)
    noise = rng.normal(0, 0.34, (H, W, 1))
    paper = np.clip(np.array(PAPER)[None, None, :] + noise, 0, 255).astype(np.uint8)
    canvas = Image.fromarray(paper).resize((W*S, H*S), Image.Resampling.BICUBIC)
    draw = ImageDraw.Draw(canvas)
    fonts = {}
    text_records = []

    def font(name, size):
        key = (name, size)
        if key not in fonts:
            fonts[key] = ImageFont.truetype(str(ROOT / FONT_FILES[name]), round(size*S))
        return fonts[key]

    def text(txt, x, y, size, face='sans', color=INK, anchor='mt', record=None):
        f = font(face, size)
        draw.text((x*S, y*S), txt, font=f, fill=color, anchor=anchor)
        if record:
            box = draw.textbbox((x*S, y*S), txt, font=f, anchor=anchor)
            text_records.append({'role': record, 'text': txt, 'bbox_xyxy': [round(q/S, 3) for q in box]})

    def line(points, color, width=1):
        draw.line([(round(x*S), round(y*S)) for x, y in points], fill=color, width=max(1, round(width*S)), joint='curve')

    def dot(x, y, radius, color):
        draw.ellipse(((x-radius)*S, (y-radius)*S, (x+radius)*S, (y+radius)*S), fill=color)

    def tracked(txt, x, y, size, spacing, color):
        f = font('sans', size)
        widths = [draw.textlength(c, font=f) / S for c in txt]
        total = sum(widths) + spacing*(len(txt)-1)
        pos = x-total/2
        for c, cw in zip(txt, widths):
            draw.text((pos*S, y*S), c, font=f, fill=color, anchor='lt')
            pos += cw + spacing

    # Editorial header and the explicit synthetic-data notice.
    tracked(data['visible_notice'], W/2, 47, 19, 3.7, GOLD)
    line([(692, 59), (783, 59)], (173, 160, 131), 0.8)
    line([(1217, 59), (1308, 59)], (173, 160, 131), 0.8)
    text(data['title'], W/2, 104, 80, 'title', record='title')
    text(data['series'][0]['label'] + '  /  percent (%)', W/2, 219, 30, 'serif', record='series_and_unit')
    text('Width of each spine represents its share of the collection.', W/2, 266, 22, 'sans', MUTED)

    # One single mapping drives all book faces, boundaries, labels and leaders.
    x0, y0, span, bh = 130, 352, 1740, 642
    x1, y1 = x0+span, y0+bh
    cumulative = [Decimal('0')]
    for v in values:
        cumulative.append(cumulative[-1] + v)
    edges = [Decimal(x0)+Decimal(span)*v/Decimal('100') for v in cumulative]
    pixel_edges = [round(float(v)*S) for v in edges]
    centers = [float((a+b)/2) for a, b in zip(edges[:-1], edges[1:])]

    # Span guide is separate from the book faces and adds no data mark.
    line([(x0, 329), (x1, 329)], (180, 177, 160), 0.8)
    line([(x0, 325), (x0, 337)], (152, 150, 135), 0.8)
    line([(x1, 325), (x1, 337)], (152, 150, 135), 0.8)
    text('0%', x0, 305, 17, 'sans', MUTED, 'lt')
    text('100%', x1, 305, 17, 'sans', MUTED, 'rt')
    tracked('TOTAL COLLECTION', W/2, 307, 15, 2.4, MUTED)

    # Continuous ground contact shadow; it is not an additional quantitative band.
    shadow = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.ellipse(((x0-21)*S, (y1-15)*S, (x1+21)*S, (y1+22)*S), fill=(27, 24, 15, 63))
    shadow = shadow.filter(ImageFilter.GaussianBlur(9*S))
    canvas = Image.alpha_composite(canvas.convert('RGBA'), shadow).convert('RGB')
    draw = ImageDraw.Draw(canvas)

    book_records = []
    for i, (left, right) in enumerate(zip(pixel_edges[:-1], pixel_edges[1:])):
        pw = right-left
        part = Image.open(ROOT / f'assets/processed/spine_{i+1:02d}.png').convert('RGB')
        part = part.resize((pw, bh*S), Image.Resampling.LANCZOS)
        a = np.asarray(part).astype(np.float32)
        # Every highlight and contact shadow stays INSIDE its own share interval.
        # Relative edge caps protect the 0.3% band's blue core without enlarging it.
        edge = min(6*S, pw*0.060)
        xx = np.arange(pw, dtype=np.float32)+0.5
        shade = 1 - 0.22*np.maximum(0, 1-xx/edge) - 0.28*np.maximum(0, 1-(pw-xx)/edge)
        a *= shade[None, :, None]
        part = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        canvas.paste(part, (left, y0*S))
        book_records.append({
            'category': categories[i], 'value_percent': float(values[i]),
            'asset': f'assets/processed/spine_{i+1:02d}.png',
            'ideal_left_px': float(edges[i]), 'ideal_right_px': float(edges[i+1]),
            'ideal_width_px': float(edges[i+1]-edges[i]),
            'supersampled_interval_px': [left, right],
            'rendered_width_px': pw/S,
            'center_x': centers[i], 'top_y': y0, 'bottom_y': y1,
            'effect_clip_interval': [left/S, right/S],
        })
    draw = ImageDraw.Draw(canvas)

    # Code-set spine names are optional internal repetitions. All required names
    # and exact values also appear externally; the true-width tiny spine is blank.
    lettering = [(225, 198, 125), (69, 51, 21), (233, 230, 206),
                 (233, 203, 134), (237, 234, 218), None, (237, 225, 197)]
    for i, cat in enumerate(categories):
        if i == 5:
            continue
        f = font('serif', 37 if i not in (1, 6) else 32)
        box = f.getbbox(cat)
        tw, th = box[2]-box[0], box[3]-box[1]
        layer = Image.new('RGBA', (tw+12*S, th+12*S), (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        origin = (6*S-box[0], 6*S-box[1])
        ld.text((origin[0]+0.5*S, origin[1]+0.5*S), cat, font=f, fill=(20, 17, 12, 145))
        ld.text(origin, cat, font=f, fill=lettering[i]+(255,))
        layer = layer.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
        bx = round(centers[i]*S-layer.width/2)
        by = round((y0+bh*0.50)*S-layer.height/2)
        assert pixel_edges[i] <= bx and bx+layer.width <= pixel_edges[i+1]
        canvas.paste(layer, (bx, by), layer)

    draw = ImageDraw.Draw(canvas)
    label_x = centers[:]
    label_x[5] = 1681.0
    rows = [1067, 1171, 1067, 1171, 1067, 1171, 1067]
    leaders = []
    for i, (cx, lx, ty) in enumerate(zip(centers, label_x, rows)):
        anchor_y = y1-9
        end_y = ty-17
        if i == 5:
            points = [(cx, anchor_y), (cx, 1040), (lx, 1135), (lx, end_y)]
        else:
            points = [(cx, anchor_y), (cx, end_y)]
        # Gold pinpoint inside each face, dark leader entirely outside below it.
        line(points, (70, 70, 59), 1.35)
        dot(cx, anchor_y, 1.0 if i == 5 else 2.0, (222, 210, 171))
        dot(lx, end_y, 1.8, (70, 70, 59))
        text(label_values[i], lx, ty, 44, 'bold', INK, record=f'value_{i}')
        text(categories[i], lx, ty+52, 28, 'serif', INK, record=f'category_{i}')
        leaders.append({'category': categories[i], 'points_xy': points, 'anchor_in_own_band': float(edges[i]) < cx < float(edges[i+1]) and y0 < anchor_y < y1})

    # Compact reading key, preserving the quiet ivory margin of the reference.
    line([(130, 1293), (1870, 1293)], (191, 188, 170), 0.8)
    text('书脊宽度对应份额；七类合计 100%。', 130, 1320, 22, 'sans', MUTED, 'lt', record='encoding_note')
    text('Source: supplied synthetic data', 1870, 1322, 21, 'serif', MUTED, 'rt', record='source_note')

    final = canvas.resize((W, H), Image.Resampling.LANCZOS)
    final.save(ROOT / 'final.png', optimize=True)
    artifacts = ROOT / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    geometry = {
        'canvas_px': [W, H], 'supersampling': S,
        'data_band': {'x': x0, 'y': y0, 'width': span, 'height': bh, 'linear_range_percent': [0, 100]},
        'sum_percent': float(sum(values)), 'minimum_width_override': False,
        'book_intervals': book_records, 'leader_paths': leaders,
        'text_boxes': text_records,
        'rendering_zh': '材质重采样到各自连续区间；所有边缘效果和内置文字均在本段内；最后统一抗锯齿缩小。',
    }
    (artifacts / 'render_geometry.json').write_text(json.dumps(geometry, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'output': 'final.png', 'canvas': [W, H], 'share_sum': float(sum(values)), 'widths_px': [b['rendered_width_px'] for b in book_records]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
