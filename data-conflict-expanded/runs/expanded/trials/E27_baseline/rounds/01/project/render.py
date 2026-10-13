"""Offline, deterministic renderer for the frozen energy-intensity composition.

Run: python render.py
During this experiment every complete invocation is launched by capture_render.py.
"""
from pathlib import Path
import json
import math
import hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
S = 2
INK = '#202924'
MUTED = '#596158'
GREEN = '#009e57'
CREAM = '#fcfaf4'


def load_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def main():
    inputs = load_json('input.json')
    brief = load_json(inputs['brief'])
    data = load_json(inputs['data'])
    core = load_json(inputs['critical_metaphor_contract'])
    # Keep the supplied visual-design source available without copying its pixels.
    with Image.open(ROOT / inputs['reference']) as reference:
        reference_size = list(reference.size)
    assert data['synthetic'] and len(data['series']) == 1
    assert data['value_axis'] == {'minimum': 0, 'maximum': 10, 'scale': 'linear'}
    series = data['series'][0]
    years = np.asarray(data['time_points'], dtype=float)
    values = np.asarray(series['values'], dtype=float)
    assert len(years) == len(values) == len(data['categories'])
    assert all(str(int(y)) == c for y, c in zip(years, data['categories']))
    W, H = brief['width'], brief['height']

    # A continuous, non-quantitative warm sky, echoing the supplied visual language.
    stops = [(0, (252,250,244)), (460, (252,250,244)),
             (830, (249,239,206)), (1220, (238,207,122)),
             (1450, (224,179,64)), (1765, (246,231,184))]
    yy = np.arange(H)
    row = np.stack([np.interp(yy, [s[0] for s in stops], [s[1][c] for s in stops])
                    for c in range(3)], axis=1).astype(np.uint8)
    bg = np.repeat(row[:, None, :], W, axis=1)
    canvas = Image.fromarray(bg).convert('RGBA').resize((W*S, H*S))
    d = ImageDraw.Draw(canvas)
    fonts = {}
    boxes = []

    def font(size, weight='regular'):
        key = (size, weight)
        if key not in fonts:
            filename = {'regular':'NimbusSans-Regular.otf', 'bold':'NimbusSans-Bold.otf',
                        'italic':'NimbusSans-Italic.otf'}[weight]
            fonts[key] = ImageFont.truetype(str(ROOT / 'assets/fonts' / filename), round(size*S))
        return fonts[key]

    def text_at(text, x, y, size=30, weight='regular', color=INK, anchor='lt', record=True):
        f = font(size, weight)
        pos = (round(x*S), round(y*S))
        d.text(pos, text, font=f, fill=color, anchor=anchor)
        if record:
            b = d.textbbox(pos, text, font=f, anchor=anchor)
            boxes.append({'text': text, 'bbox': [round(z/S, 2) for z in b]})

    def line(points, fill=INK, width=2):
        d.line([(round(x*S), round(y*S)) for x, y in points], fill=fill,
               width=max(1, round(width*S)), joint='curve')

    def circle(x, y, r, fill, outline=None, width=1):
        d.ellipse(((x-r)*S, (y-r)*S, (x+r)*S, (y+r)*S),
                  fill=fill, outline=outline, width=round(width*S))

    def bezier(points, fill, width=2):
        p = np.asarray(points, dtype=float)
        t = np.linspace(0, 1, 240)[:, None]
        curve = (1-t)**3*p[0]+3*(1-t)**2*t*p[1]+3*(1-t)*t*t*p[2]+t**3*p[3]
        line(curve, fill, width)

    # Editorial heading, metric definition, and explicit synthetic notice.
    text_at('SYNTHETIC', 110, 76, 33, 'bold', GREEN)
    text_at('Energy intensity', 103, 144, 119, 'bold')
    text_at('Energy consumed per dollar of economic output', 110, 289, 36)
    d.rounded_rectangle((1510*S, 76*S, 1890*S, 130*S), radius=27*S, fill='#155b3c')
    text_at(data['visible_notice'], 1700, 89, 27, 'bold', '#ffffff', 'mt')
    text_at('Lower intensity means less energy', 1890, 177, 27, anchor='rt', color=MUTED)
    text_at('per dollar of economic output.', 1890, 214, 27, anchor='rt', color=MUTED)
    line([(110, 366), (1890, 366)], '#c8c6b9', 1.5)

    line([(110, 421), (178, 421)], GREEN, 7)
    text_at(series['label'], 198, 404, 32, 'bold')
    text_at(series['unit'], 110, 463, 29, color=MUTED)
    text_at('Years positioned to scale', 1890, 416, 26, color=MUTED, anchor='rt')

    # ONE mapping is used by the path, markers, labels, bird perches, and connectors.
    X0, X1, YTOP, Y0 = 235.0, 1810.0, 520.0, 1450.0
    VMIN, VMAX = float(data['value_axis']['minimum']), float(data['value_axis']['maximum'])
    def px(year):
        return X0 + (year-years[0])/(years[-1]-years[0])*(X1-X0)
    def py(value):
        return Y0 - (value-VMIN)/(VMAX-VMIN)*(Y0-YTOP)
    points = [(px(y), py(v)) for y, v in zip(years, values)]
    def cable_y(x):
        return float(np.interp(x, [p[0] for p in points], [p[1] for p in points]))
    def cable_angle(x):
        xs = np.asarray([p[0] for p in points])
        i = min(max(int(np.searchsorted(xs, x))-1, 0), len(xs)-2)
        return math.atan2(points[i+1][1]-points[i][1], points[i+1][0]-points[i][0])

    # Low-contrast reference grid remains subordinate to the electrical scene.
    for tick in range(0, 11, 2):
        y = py(tick)
        line([(190, y), (1890, y)], '#cabd9a' if tick else '#817d64', 1.2 if tick else 2)
        text_at(str(tick), 158, y, 28, anchor='rm', color=MUTED)
        line([(177, y), (190, y)], '#8f947e', 1.5)
    line([(190, YTOP), (190, Y0)], '#969c88', 1.5)

    # Anchor the generated tower's upper-right insulator directly to the 1970 knot.
    pylon = Image.open(ROOT / 'assets/pylon.png').convert('RGBA')
    raw_box = next(a['raw_bbox'] for a in load_json('assets/component_regions.json')
                   if a['path'] == 'assets/pylon.png')
    raw_anchor = (700, 364)
    anchor_fraction = ((raw_anchor[0]-raw_box[0])/pylon.width,
                       (raw_anchor[1]-raw_box[1])/pylon.height)
    attachment = points[2]
    tower_bottom = Y0 - 1
    tower_h = (tower_bottom-attachment[1])/(1-anchor_fraction[1])
    tower_w = 586.0
    tower_left = attachment[0] - anchor_fraction[0]*tower_w
    tower_top = tower_bottom-tower_h
    def tower_xy(raw_x, raw_y):
        return (tower_left+(raw_x-raw_box[0])/pylon.width*tower_w,
                tower_top+(raw_y-raw_box[1])/pylon.height*tower_h)
    tl = tower_xy(164, 350)
    bl = tower_xy(133, 601)
    br = tower_xy(736, 608)

    # Dark non-data conductors are connected to real tower hardware.
    # The main conductor is black only outside the observed time range.
    dark_wire = '#3d4234'
    first_slope = (points[1][1]-points[0][1])/(points[1][0]-points[0][0])
    last_slope = (points[-1][1]-points[-2][1])/(points[-1][0]-points[-2][0])
    line([(0, points[0][1]-X0*first_slope), points[0]], dark_wire, 2.4)
    line([points[-1], (W, points[-1][1]+(W-X1)*last_slope)], dark_wire, 2.4)
    bezier([(-30, 666), (100, 732), (190, 804), tl], dark_wire, 2)
    bezier([tl, (625, 1010), (1330, 1408), (2005, 1384)], dark_wire, 2.1)
    bezier([(-30, 855), (105, 939), (187, 1006), bl], dark_wire, 2.2)
    bezier([bl, (710, 1255), (1410, 1504), (2005, 1430)], dark_wire, 2.2)
    bezier([br, (1160, 1323), (1660, 1494), (2005, 1445)], dark_wire, 2.0)

    tower_image = pylon.resize((round(tower_w*S), round(tower_h*S)), Image.Resampling.LANCZOS)
    canvas.alpha_composite(tower_image, (round(tower_left*S), round(tower_top*S)))
    d = ImageDraw.Draw(canvas)
    # Explicit hardware contact bridges any alpha-antialiasing gap at the terminal.
    contact_top = tower_xy(700, 357)
    line([contact_top, attachment], '#252d26', 3)

    # Quantitative geometry: exact polyline, with no smoothing or shape exaggeration.
    line(points, '#f6edcf', 13)
    line(points, GREEN, 8)
    for x, y in points:
        circle(x, y, 10, '#f7efd4')
        circle(x, y, 6.5, GREEN)

    # Perching feet are transformed onto the same data-controlled green conductor.
    birds = [
        ('bird_1', 350, 94, (177, 171)),
        ('bird_2', 889, 89, (80, 172)),
        ('bird_3', 1405, 96, (170, 196)),
        ('bird_4', 1662, 88, (95, 161)),
    ]
    bird_checks = []
    for name, x, width, anchor in birds:
        asset = Image.open(ROOT / f'assets/{name}.png').convert('RGBA')
        ratio = width*S/asset.width
        scaled = asset.resize((round(width*S), round(asset.height*ratio)), Image.Resampling.LANCZOS)
        theta = cable_angle(x)
        rotated = scaled.rotate(-math.degrees(theta), Image.Resampling.BICUBIC, expand=True)
        ax = anchor[0]*ratio-scaled.width/2
        ay = anchor[1]*ratio-scaled.height/2
        rx = math.cos(theta)*ax-math.sin(theta)*ay+rotated.width/2
        ry = math.sin(theta)*ax+math.cos(theta)*ay+rotated.height/2
        y = cable_y(x)
        left, top = round(x*S-rx), round(y*S-ry)
        canvas.alpha_composite(rotated, (left, top))
        bird_checks.append({'asset': f'assets/{name}.png', 'perch': [x, round(y, 3)],
                            'bbox': [left/S, top/S, (left+rotated.width)/S, (top+rotated.height)/S]})
    d = ImageDraw.Draw(canvas)

    # All supplied values are preserved. Dense endpoints get separate leader labels.
    for i, ((x, y), v) in enumerate(zip(points, values)):
        if i < 6:
            text_at(f'{v:.2f}', x, y-62, 41, 'bold', anchor='mt')
    x, y = points[6]
    line([(x-2, y+12), (x-30, y+57), (x-83, y+57)], '#596149', 1.6)
    text_at('2009', x-185, y+58, 25, color=MUTED)
    text_at(f'{values[6]:.2f}', x-102, y+51, 41, 'bold')
    x, y = points[7]
    line([(x+9, y-8), (x+45, y-51), (x+84, y-51)], '#596149', 1.6)
    text_at(f'{values[7]:.2f}', x+80, y-109, 41, 'bold', anchor='mt')
    text_at('2010', x+80, y-67, 25, color=MUTED, anchor='mt')

    # Calendar labels use true x positions; 2009 is directly labeled at its data point.
    for i, (cat, (x, _)) in enumerate(zip(data['categories'], points)):
        line([(x, Y0), (x, Y0+12)], '#696f56', 1.8)
        if i != 6:
            text_at(cat, x, Y0+32, 31, anchor='mt')
    text_at('Year', 1890, Y0+85, 27, color=MUTED, anchor='rt')

    # Interpretive footnote clearly distinguishes decorative objects from data.
    line([(110, 1600), (1890, 1600)], '#b9ac7f', 1.2)
    text_at('SYNTHETIC DATA', 110, 1640, 25, 'bold', GREEN)
    text_at('Illustrative values, not historical statistics.', 380, 1640, 25, color='#505948')
    text_at('Green line = energy intensity.  Tower, birds and dark cables are scene elements.',
            110, 1690, 25, color='#505948')
    text_at('1950–2010', 1890, 1640, 27, 'bold', anchor='rt')

    canvas.convert('RGB').resize((W,H), Image.Resampling.LANCZOS).save(ROOT / 'final.png')
    geometry = {
        'canvas': [W,H], 'supersampling': S,
        'mapping': {'x_min_year': float(years[0]), 'x_max_year': float(years[-1]),
                    'x_min_px': X0, 'x_max_px': X1, 'y_min_value': VMIN,
                    'y_max_value': VMAX, 'y_zero_px': Y0, 'y_top_px': YTOP},
        'points': [{'year': c, 'value': float(v), 'pixel': [round(x,6),round(y,6)]}
                   for c,v,(x,y) in zip(data['categories'],values,points)],
        'path_rule': 'piecewise linear; every supplied point; no spline or minimum mark length',
        'tower_attachment': list(attachment),
        'tower_bbox': [tower_left,tower_top,tower_left+tower_w,tower_bottom],
        'birds': bird_checks, 'text_boxes': boxes,
        'frozen_dependencies': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                for name in ['input.json',inputs['brief'],inputs['data'],
                                             inputs['critical_metaphor_contract'],inputs['reference']]},
        'reference_image_size': reference_size,
        'topic_zh': core['topic_zh'],
    }
    (ROOT/'artifacts').mkdir(exist_ok=True)
    (ROOT/'artifacts/geometry.json').write_text(json.dumps(geometry, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('Rendered final.png: exact calendar x positions, common linear 0–10 y scale, 8 points.')


if __name__ == '__main__':
    main()
