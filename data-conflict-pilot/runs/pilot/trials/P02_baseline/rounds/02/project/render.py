#!/usr/bin/env python3
"""Rebuild the infographic from the frozen data and local generated surfaces."""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageChops

ROOT = Path(__file__).resolve().parent
S = 2
INK = '#242220'
PAPER = '#f4efdf'
RED = '#a34c3d'


def load_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def xy(point):
    return tuple(int(round(v * S)) for v in point)


def polygon(canvas, points, fill, outline=None, width=2):
    d = ImageDraw.Draw(canvas)
    pts = [xy(p) for p in points]
    if fill is not None:
        d.polygon(pts, fill=fill)
    if outline:
        d.line(pts + [pts[0]], fill=outline, width=round(width * S), joint='curve')


def polyline(canvas, points, fill=INK, width=3):
    ImageDraw.Draw(canvas).line([xy(p) for p in points], fill=fill,
                               width=round(width * S), joint='curve')


def font(size, bold=False):
    return ImageFont.truetype(str(ROOT / 'assets' / ('NimbusSans-Bold.otf' if bold else 'NimbusSans-Regular.otf')), round(size * S))


def text(canvas, position, value, size, fill=INK, bold=False, anchor='la'):
    ImageDraw.Draw(canvas).text(xy(position), value, font=font(size, bold), fill=fill, anchor=anchor)


def tracked(canvas, position, value, size=20, spacing=2.6, fill=INK, bold=True):
    draw = ImageDraw.Draw(canvas)
    f = font(size, bold)
    x, y = xy(position)
    for letter in value:
        draw.text((x, y), letter, font=f, fill=fill)
        x += draw.textlength(letter, font=f) + spacing * S


def warp(canvas, image, corners, source=None):
    """Map a source quadrilateral to a canvas quadrilateral, without altering data coordinates."""
    pts = np.asarray(corners, dtype=float) * S
    lo = np.floor(pts.min(axis=0) - 2).astype(int)
    hi = np.ceil(pts.max(axis=0) + 3).astype(int)
    width, height = (hi - lo).tolist()
    dst = pts - lo
    iw, ih = image.size
    src = np.asarray(source if source is not None else [(0, 0), (iw - 1, 0), (iw - 1, ih - 1), (0, ih - 1)], dtype=float)
    a, b = [], []
    for (x, y), (u, v) in zip(dst, src):
        a.extend([[x, y, 1, 0, 0, 0, -u*x, -u*y],
                  [0, 0, 0, x, y, 1, -v*x, -v*y]])
        b.extend([u, v])
    coeff = np.linalg.solve(np.asarray(a), np.asarray(b))
    layer = image.convert('RGBA').transform((width, height), Image.Transform.PERSPECTIVE,
                                            coeff.tolist(), Image.Resampling.BICUBIC)
    mask = Image.new('L', (width, height))
    ImageDraw.Draw(mask).polygon([tuple(p) for p in dst], fill=255)
    layer.putalpha(ImageChops.multiply(mask, layer.getchannel('A')))
    canvas.alpha_composite(layer, tuple(lo))


def textured_polygon(canvas, image, points, outline=INK, width=3):
    lo = np.floor(np.min(points, axis=0) * S).astype(int)
    hi = np.ceil(np.max(points, axis=0) * S + 1).astype(int)
    size = tuple(hi - lo)
    layer = image.convert('RGBA').resize(size, Image.Resampling.LANCZOS)
    mask = Image.new('L', size)
    ImageDraw.Draw(mask).polygon([tuple(np.asarray(p)*S - lo) for p in points], fill=255)
    layer.putalpha(mask)
    canvas.alpha_composite(layer, tuple(lo))
    if outline:
        polygon(canvas, points, None, outline, width)


def strip(canvas, image, edge, vector):
    """One shared u coordinate binds all roof and wall segments to chronological x."""
    n = len(edge)
    w, h = image.size
    edge = np.asarray(edge, dtype=float)
    vector = np.asarray(vector, dtype=float)
    for i in range(n - 1):
        corners = [edge[i], edge[i+1], edge[i+1]+vector, edge[i]+vector]
        u0, u1 = i/(n-1)*(w-1), (i+1)/(n-1)*(w-1)
        warp(canvas, image, corners, [(u0, 0), (u1, 0), (u1, h-1), (u0, h-1)])


def paste_asset(canvas, path, pos, height):
    im = Image.open(ROOT / path).convert('RGBA')
    width = height * im.width / im.height
    im = im.resize((round(width*S), round(height*S)), Image.Resampling.LANCZOS)
    canvas.alpha_composite(im, xy(pos))
    return [pos[0], pos[1], pos[0]+width, pos[1]+height]


def halftone_polygon(canvas, points, base='#252322', dot='#827874', pitch=6, radius=.65):
    polygon(canvas, points, base)
    bbox = np.asarray(points)
    x0, y0 = bbox.min(axis=0).astype(int)
    x1, y1 = bbox.max(axis=0).astype(int)
    mask = Image.new('L', canvas.size)
    ImageDraw.Draw(mask).polygon([xy(p) for p in points], fill=255)
    dots = Image.new('RGBA', canvas.size)
    d = ImageDraw.Draw(dots)
    for row, y in enumerate(range(y0, y1+1, pitch)):
        for x in range(x0, x1+1, pitch):
            x += (pitch/2 if row % 2 else 0)
            d.ellipse([*xy((x-radius, y-radius)), *xy((x+radius, y+radius))], fill=dot)
    dots.putalpha(ImageChops.multiply(dots.getchannel('A'), mask))
    canvas.alpha_composite(dots)


def main():
    brief = load_json('brief.json')
    data = load_json('data.json')
    width, height = brief['width'], brief['height']
    categories = data['categories']
    values = np.asarray(data['series'][0]['values'], dtype=float)
    assert len(categories) == len(values)
    pmin, pmax = data['price_scale']['minimum'], data['price_scale']['maximum']
    assert data['price_scale']['type'] == 'linear'

    # This affine data map is used once, before any illustrative transformations.
    x0, x1 = 590.0, 1390.0
    y_zero, y_max = 1810.0, 310.0
    xs = np.linspace(x0, x1, len(values))
    ys = y_zero - (values-pmin)/(pmax-pmin)*(y_zero-y_max)
    ridge = np.column_stack([xs, ys])
    roof_vector = np.array([-100.0, 310.0])
    eave = ridge + roof_vector
    wall_top = eave + np.array([1.0, 9.0])
    wall_vector = np.array([12.0, 198.0])
    wall_base = wall_top + wall_vector

    canvas = Image.new('RGBA', (width*S, height*S), PAPER)
    draw = ImageDraw.Draw(canvas)

    # Restrained warm paper grain; deterministic and independent of chart values.
    rng = np.random.default_rng(404)
    noise = rng.normal(0, .60, (height, width, 1))
    paper = np.clip(np.array([244, 239, 223])[None, None, :] + noise, 0, 255).astype('uint8')
    canvas.alpha_composite(Image.fromarray(paper).resize(canvas.size, Image.Resampling.BILINEAR).convert('RGBA'))

    # Magazine frame and neutral heading.
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([*xy((54, 50)), *xy((1706, 1950))], outline=INK, width=3*S)
    tracked(canvas, (108, 92), 'ILLUSTRATIVE HOUSING SERIES', size=19, spacing=2.6)
    draw.rectangle([*xy((1353, 83)), *xy((1652, 120))], fill=RED)
    text(canvas, (1502.5, 102), data['visible_notice'], 19, '#fff8e9', True, 'mm')
    text(canvas, (106, 139), 'SYNTHETIC HOME PRICES', 81, INK, True)
    text(canvas, (109, 238), data['series'][0]['name']+'  /  USD', 30)
    polyline(canvas, [(110, 290), (1650, 290)], INK, 1.5)
    tracked(canvas, (111, 310), 'YEAR', 16, 1.6, '#736b61')
    tracked(canvas, (278, 310), 'PRICE (USD)', 16, 1.3, '#736b61')

    # The horizontal leaders share the exact y coordinate of each data point.
    for x, y in ridge:
        polyline(canvas, [(421, y), (x, y)], RED, 1.8)

    # Common ground and printed shadows.
    polygon(canvas, [(56, 1720), (1704, 1720), (1704, 1868), (56, 1868)], '#929d60')
    ground = Image.new('RGBA', canvas.size)
    gd = ImageDraw.Draw(ground)
    for _ in range(14500):
        x, y = rng.uniform(57, 1703), rng.uniform(1721, 1867)
        gd.ellipse([*xy((x, y)), *xy((x+.45, y+.45))], fill=(44, 53, 27, 65))
    canvas.alpha_composite(ground)
    halftone_polygon(canvas,
                      [(443, 1759), (792, 1765), (1328, 1808), (1467, 1821),
                       (1155, 1829), (685, 1803), (461, 1789)],
                      base='#3c432c', dot='#929d60', pitch=5, radius=.8)
    halftone_polygon(canvas, [(1288, 1810), (1630, 1810), (1674, 1823), (1353, 1822)],
                      base='#56603a', dot='#929d60', pitch=5, radius=.9)

    roof_im = Image.open(ROOT/'assets/roof_surface.png').convert('RGB')
    front_im = Image.open(ROOT/'assets/front_wall.png').convert('RGB')
    stucco_im = Image.open(ROOT/'assets/wall_stucco.png').convert('RGB')
    end_im = Image.open(ROOT/'assets/end_wall.png').convert('RGB')

    # Deep foundation under the entire warped facade; not a second chart.
    foundation = list(wall_base) + list((wall_base + np.array([28., 39.]))[::-1])
    halftone_polygon(canvas, foundation, base='#262422', dot='#aaa097', pitch=5, radius=.7)
    polygon(canvas, foundation, None, INK, 3)

    peak = ridge[-1]
    back_eave = peak + np.array([145., 320.])
    back_base = back_eave + np.array([0., 210.])
    side = [peak, back_eave, back_base, wall_base[-1], eave[-1]]
    textured_polygon(canvas, end_im, side, width=4)
    halftone_polygon(canvas, [wall_base[-1], back_base, back_base+[20, 28], wall_base[-1]+[28,39]],
                      base='#242221', dot='#918789', pitch=5, radius=.65)
    polyline(canvas, [peak, back_eave, back_eave+[5,3]], INK, 4)

    # Printed wall follows the silhouette; complete generated openings stay planar.
    strip(canvas, stucco_im, wall_top, wall_vector)
    polygon(canvas, list(wall_top) + list(wall_base[::-1]), None, INK, 3.4)

    def wall_at(u):
        t = u * (len(wall_top)-1)
        j = min(int(t), len(wall_top)-2)
        return wall_top[j]*(1-(t-j)) + wall_top[j+1]*(t-j)

    openings = [
        ('window_left.png', .12, .34, .16, .84),
        ('front_door.png', .45, .575, .12, 1.0),
        ('window_right.png', .605, .815, .14, .82),
    ]
    for asset, u0, u1, v0, v1 in openings:
        left, right = wall_at(u0), wall_at(u1)
        corners = [left+v0*wall_vector, right+v0*wall_vector,
                   right+v1*wall_vector, left+v1*wall_vector]
        warp(canvas, Image.open(ROOT/'assets'/asset), corners)

    # The visible red roof starts EXACTLY at the data polyline. No separate line chart.
    strip(canvas, roof_im, ridge, roof_vector)
    polygon(canvas, list(ridge)+list(eave[::-1]), None, INK, 4)
    fascia = list(eave) + list((eave+np.array([1.,9.]))[::-1])
    polygon(canvas, fascia, '#eee6d6', INK, 2.3)

    # Attached lean-to, connected to and occluding the end wall.
    annex_front = [(1368, 740), (1485, 805), (1485, 1028), (1368, 965)]
    annex_side = [(1485, 805), (1637, 845), (1637, 1006), (1485, 1028)]
    door_panel = front_im.crop((604, 0, 876, front_im.height))
    warp(canvas, door_panel, annex_front)
    polygon(canvas, annex_front, None, INK, 3.5)
    textured_polygon(canvas, end_im, annex_side, width=3.5)
    window = front_im.crop((1004, 34, 1306, 251))
    window_quad = [(1508, 867), (1615, 894), (1615, 961), (1508, 945)]
    warp(canvas, window, window_quad)
    polygon(canvas, window_quad, None, INK, 2)
    annex_roof = [(1363, 732), (1504, 766), (1643, 838), (1486, 800)]
    warp(canvas, roof_im, annex_roof)
    polygon(canvas, annex_roof, None, INK, 3.6)
    polyline(canvas, [(1363, 738), (1486, 807), (1643, 845)], '#e8dfd4', 4.5)
    polyline(canvas, [(1363, 733), (1486, 802), (1643, 841)], INK, 2.5)
    halftone_polygon(canvas, [(1368,965),(1485,1028),(1637,1006),(1606,1032),(1486,1056),(1352,994)],
                      base='#252321', dot='#9b9290', pitch=5, radius=.65)

    # Small foreground figures and vegetation, from the second generated sheet.
    shrubs_bbox = paste_asset(canvas, 'assets/shrubs.png', (245, 1635), 210)
    family_bbox = paste_asset(canvas, 'assets/family_dog.png', (1313, 1579), 231)

    # Code owns every mandatory numeric/category label and every node.
    label_boxes = []
    for category, value, (x, y) in zip(categories, values, ridge):
        text(canvas, (110, y), category, 31, INK, True, 'lm')
        label = '$'+format(int(value), ',')
        text(canvas, (394, y), label, 32, RED, True, 'rm')
        draw = ImageDraw.Draw(canvas)
        draw.ellipse([*xy((x-6.8, y-6.8)), *xy((x+6.8, y+6.8))], fill=INK)
        label_boxes.append({'category':category, 'price_text':label, 'center_y':float(y),
                            'year_bbox':[110, float(y-18), 238, float(y+18)],
                            'price_bbox':[264, float(y-18), 394, float(y+18)]})

    # A concise scale note makes the frozen linear range explicit without a second chart.
    text(canvas, (109, 1889), 'Roof edge: linear price scale, $0–$70,000. Each point is one equally spaced synthetic year.', 23)
    text(canvas, (109, 1921), 'Source: local synthetic data. Illustrative values; not market statistics.', 20, '#655d54')
    text(canvas, (1650, 1921), '11 YEARS  /  USD', 18, '#655d54', True, 'ra')

    # Save exactly once; external capture_render.py preserves every complete candidate.
    canvas.convert('RGB').resize((width, height), Image.Resampling.LANCZOS).save(ROOT/'final.png')
    metadata = {
        'canvas':[width,height], 'price_scale':[pmin,pmax],
        'data_mapping':{'x_first':x0,'x_last':x1,'x_step':float(xs[1]-xs[0]),
                        'y_at_minimum':y_zero,'y_at_maximum':y_max,
                        'pixels_per_usd':(y_zero-y_max)/(pmax-pmin)},
        'roof_ridge':[{'category':c,'value':int(v),'x':float(x),'y':float(y)}
                      for c,v,(x,y) in zip(categories,values,ridge)],
        'roof_eave':eave.tolist(), 'wall_base':wall_base.tolist(),
        'label_boxes':label_boxes,
        'generated_visible_regions':{'main_roof':(list(ridge)+list(eave[::-1])),
                                     'front_wall':(list(wall_top)+list(wall_base[::-1])),
                                     'side_wall':side, 'family_and_dog_bbox':family_bbox,
                                     'shrubs_bbox':shrubs_bbox},
        'main_roof_boundary_is_data':True,
    }
    def serial(value):
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, np.generic): return value.item()
        raise TypeError(type(value).__name__)
    (ROOT/'artifacts/geometry.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=serial)+'\n', encoding='utf-8')
    print('Wrote final.png at', width, 'x', height, 'with', len(values), 'data-bound roof nodes.')


if __name__ == '__main__':
    main()
