#!/usr/bin/env python3
"""Offline renderer: generated house materials, deterministic price geometry."""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

ROOT = Path(__file__).resolve().parent
S = 2
BG = '#F7F3E9'
INK = '#243E43'
MUTED = '#637578'
GRID = '#D7DCD3'
ACCENT = '#A23D2D'
EDGE = '#963B2F'
MARKER = '#EEBA5F'


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def save_json(name, data):
    p = ROOT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    inputs = read_json('input.json')
    brief = read_json(inputs['brief'])
    data = read_json(inputs['data'])
    core = read_json(inputs['critical_metaphor_contract'])
    assert (ROOT / inputs['reference']).is_file()
    assert core['visible_title'] == data['title']
    assert data['value_axis']['scale'] == 'linear'
    assert len(data['series']) == 1
    W, H = brief['width'], brief['height']
    series = data['series'][0]
    values = np.array(series['values'], dtype=float)
    times = np.array(data['time_points'], dtype=float)
    categories = data['categories']
    assert len(values) == len(times) == len(categories)
    assert np.all(np.diff(times) > 0)
    vmin, vmax = data['value_axis']['minimum'], data['value_axis']['maximum']
    x_left, x_right = 330.0, 1560.0
    zero_y, axis_top = 1470.0, 470.0
    scale = (zero_y - axis_top) / (vmax - vmin)
    xs = x_left + (times - times[0]) / (times[-1] - times[0]) * (x_right - x_left)
    ys = zero_y - (values - vmin) * scale
    P = np.column_stack([xs, ys])
    roof_depth = np.array([-105.0, 165.0])
    wall_height = 255.0
    Q = P + roof_depth
    B = Q + np.array([0.0, wall_height])
    sill_depth = 17.0
    floor = B + np.array([0.0, sill_depth])

    canvas = Image.new('RGB', (W*S, H*S), BG)
    d = ImageDraw.Draw(canvas)
    font_cache = {}
    text_records = []

    def font(size, weight='regular'):
        name = 'sans_bold.ttf' if weight == 'bold' else 'sans.ttf'
        key = (name, size)
        if key not in font_cache:
            font_cache[key] = ImageFont.truetype(str(ROOT / 'assets' / name), round(size*S))
        return font_cache[key]

    def text(x, y, value, size=26, fill=INK, weight='regular', anchor='la', record=True):
        f = font(size, weight)
        # Ascender anchor keeps all vertical layout deterministic.
        d.text((x*S, y*S), value, font=f, fill=fill, anchor=anchor)
        if record:
            box = d.textbbox((x*S, y*S), value, font=f, anchor=anchor)
            text_records.append({'text': value, 'bbox_px': [round(z/S, 2) for z in box]})

    def line(points, fill=INK, width=2):
        d.line([(float(x)*S, float(y)*S) for x, y in points], fill=fill, width=round(width*S), joint='curve')

    def polygon(points, fill, outline=None, width=1):
        pp = [(float(x)*S, float(y)*S) for x, y in points]
        d.polygon(pp, fill=fill)
        if outline:
            d.line(pp + [pp[0]], fill=outline, width=round(width*S), joint='curve')

    def rect(box, fill, radius=0):
        box = tuple(float(x)*S for x in box)
        if radius:
            d.rounded_rectangle(box, radius=radius*S, fill=fill)
        else:
            d.rectangle(box, fill=fill)

    def warp_panel(src, top, bottom):
        """Map a flat generated panel into affine strips; data coordinates are fixed."""
        src = src.convert('RGB')
        sw, sh = src.size
        for i in range(len(top)-1):
            dest = np.array([top[i], top[i+1], bottom[i+1], bottom[i]], dtype=float)*S
            left = math.floor(dest[:, 0].min())
            upper = math.floor(dest[:, 1].min())
            right = math.ceil(dest[:, 0].max()) + 1
            lower = math.ceil(dest[:, 1].max()) + 1
            size = (right-left, lower-upper)
            # u follows the true time spacing. v spans the non-data part.
            u0 = (times[i]-times[0])/(times[-1]-times[0])
            u1 = (times[i+1]-times[0])/(times[-1]-times[0])
            A = np.column_stack((dest[1]-dest[0], dest[3]-dest[0]))
            inv = np.linalg.inv(A)
            off = inv @ (np.array([left, upper])-dest[0])
            coeff = (
                sw*(u1-u0)*inv[0,0], sw*(u1-u0)*inv[0,1], sw*(u0+(u1-u0)*off[0]),
                sh*inv[1,0], sh*inv[1,1], sh*off[1],
            )
            patch = src.transform(size, Image.Transform.AFFINE, coeff, Image.Resampling.BICUBIC)
            mask = Image.new('L', size, 0)
            md = ImageDraw.Draw(mask)
            md.polygon([tuple(p-np.array([left, upper])) for p in dest], fill=255)
            canvas.paste(patch, (left, upper), mask)

    def warp_quad(src, corners):
        src = src.convert('RGB')
        dest = np.array(corners, dtype=float)*S
        left, upper = np.floor(dest.min(axis=0)).astype(int)
        right, lower = np.ceil(dest.max(axis=0)).astype(int)+1
        A = np.column_stack((dest[1]-dest[0], dest[3]-dest[0]))
        inv = np.linalg.inv(A)
        off = inv @ (np.array([left, upper])-dest[0])
        sw, sh = src.size
        coeff = (sw*inv[0,0], sw*inv[0,1], sw*off[0], sh*inv[1,0], sh*inv[1,1], sh*off[1])
        patch = src.transform((right-left, lower-upper), Image.Transform.AFFINE, coeff, Image.Resampling.BICUBIC)
        mask = Image.new('L', patch.size, 0)
        ImageDraw.Draw(mask).polygon([tuple(p-[left,upper]) for p in dest], fill=255)
        canvas.paste(patch, (left,upper), mask)

    # Editorial header: the synthetic status and statistical role are explicit.
    rect((100, 70, 377, 119), ACCENT, radius=4)
    text(120, 78, data['visible_notice'], 23, fill=BG, weight='bold')
    text(1660, 83, f'{len(values)} ANNUAL OBSERVATIONS', 21, fill=MUTED, anchor='ra')
    text(96, 148, data['title'], 74, weight='bold')
    text(100, 253, series['label'] + ' · ' + series['unit'], 32, fill=INK)
    line([(100, 320), (1660, 320)], fill=INK, width=1.5)
    text(100, 347, 'ANNUAL PRICE / USD', 22, weight='bold')
    text(1660, 347, 'The roof edge traces every annual price.', 24, fill=MUTED, anchor='ra')

    # A linear value frame remains visible behind the object. Only the upper
    # roof boundary carries quantitative values; walls provide object identity.
    tick_values = np.linspace(vmin, vmax, 5)
    for val in tick_values:
        y = zero_y-(val-vmin)*scale
        text(159, y-17, f'{int(val):,}', 24, fill=MUTED, anchor='ra')
        line([(177,y),(192,y)], fill=MUTED, width=1.2)
        for x in np.arange(200, 1645, 17):
            line([(x,y),(min(x+5,1645),y)], fill=GRID, width=1)
    line([(185,axis_top),(185,zero_y)], fill=GRID, width=1)

    # Ground and a soft cast shadow make the elevated house/family relation visible.
    ground_y = 1810.0
    polygon([(100,ground_y-18),(1660,ground_y-18),(1660,ground_y+28),(100,ground_y+28)], '#DDE2CE')
    line([(100,ground_y+28),(1660,ground_y+28)], '#A2AF98', 1.2)
    shadow = Image.new('RGBA', canvas.size, (0,0,0,0))
    sd = ImageDraw.Draw(shadow)
    sd.ellipse((470*S,1760*S,1590*S,1811*S), fill=(45,65,51,32))
    sd.ellipse((760*S,1770*S,1515*S,1807*S), fill=(45,65,51,32))
    shadow = shadow.filter(ImageFilter.GaussianBlur(13*S))
    canvas.paste(shadow, (0,0), shadow)
    d = ImageDraw.Draw(canvas)

    roof = Image.open(ROOT / 'assets/roof_texture.png')
    facade = Image.open(ROOT / 'assets/facade_texture.png')
    siding = Image.open(ROOT / 'assets/siding_texture.png').convert('RGB')
    siding = Image.blend(siding, Image.new('RGB', siding.size, '#537078'), .58)

    # The right end wall supplies perspective without moving a single data point.
    side_corners = [P[-1], Q[-1], B[-1], P[-1]+[0,wall_height]]
    warp_quad(siding, side_corners)
    line(side_corners+[side_corners[0]], INK, 3)
    warp_panel(facade, Q, B)
    polygon(np.vstack([B, floor[::-1]]), INK)
    line([Q[0],B[0],floor[0]], INK, 3)
    line(B, '#D5CDBB', 3)
    # Continuous terracotta roof material is clipped to the eleven-point contour.
    warp_panel(roof, P, Q)
    line(Q, INK, 5)
    line(Q+[0,5], '#E2C7A5', 3)
    line([P[0],Q[0]], INK, 3)
    line([P[-1],Q[-1]], INK, 3)
    line(P, EDGE, 6)
    # Small marker centres have a distinct colour for visible-pixel verification.
    for x,y in P:
        d.ellipse(((x-8)*S,(y-8)*S,(x+8)*S,(y+8)*S), fill=INK)
        d.ellipse(((x-4.5)*S,(y-4.5)*S,(x+4.5)*S,(y+4.5)*S), fill=MARKER)

    # All mandatory category/value labels are code-generated and bound to P.
    for category, value, (x,y) in zip(categories, values, P):
        text(x, y-100, category, 23, fill=MUTED, anchor='ma')
        text(x, y-67, f'{int(value):,}', 26, weight='bold', anchor='ma')
        line([(x,y-28),(x,y-12)], '#9BA6A2', 1)

    # Derived endpoint comparison, placed in the open space under the high end.
    change = (values[-1]/values[0]-1)*100
    delta = values[-1]-values[0]
    rect((255,1385,880,1697), BG)
    line([(282,1420),(349,1420)], ACCENT, 4)
    text(282, 1453, categories[0].upper()+' → '+categories[-1].upper(), 22, fill=MUTED, weight='bold')
    text(277, 1484, f'−{abs(change):.1f}%', 82, fill=ACCENT, weight='bold')
    text(282, 1593, f'{int(abs(delta)):,} USD lower', 29, weight='bold')
    text(282, 1641, 'Change in illustrative new-home price', 23, fill=MUTED)

    family = Image.open(ROOT / 'assets/family.png').convert('RGBA')
    family_height = 255
    fw = round(family.width/family.height*family_height)
    family = family.resize((fw*S, family_height*S), Image.Resampling.LANCZOS)
    family_x = 1175
    family_y = ground_y-family_height
    # Contact shadow is on the ground, separate from the hovering house.
    d.ellipse(((family_x-15)*S,(ground_y-9)*S,(family_x+fw+13)*S,(ground_y+6)*S), fill='#9BA78C')
    canvas.paste(family, (round(family_x*S),round(family_y*S)), family)
    d = ImageDraw.Draw(canvas)
    text(100, 1880, 'A house-shaped price history', 29, weight='bold')
    text(100, 1927, f'Synthetic annual values · Linear scale: {vmin:,}–{vmax:,} {series["unit"]} · No real-market claim', 23, fill=MUTED)
    text(1660, 1883, categories[0].upper()+' — '+categories[-1].upper(), 21, fill=MUTED, anchor='ra')

    final = canvas.resize((W,H), Image.Resampling.LANCZOS)
    final.save(ROOT / 'final.png')
    save_json('artifacts/geometry.json', {
        'canvas': [W,H],
        'mapping': {'x_min':x_left,'x_max':x_right,'zero_y':zero_y,'axis_top':axis_top,'pixels_per_usd':scale,'value_min':vmin,'value_max':vmax},
        'roof_points': [{'category':cat,'time':float(t),'value':int(v),'x':float(x),'y':float(y)} for cat,t,v,(x,y) in zip(categories,times,values,P)],
        'roof_front_offset':roof_depth.tolist(),
        'wall_height_px':wall_height,
        'family_bounds':[family_x,family_y,family_x+fw,ground_y],
        'text_records':text_records,
        'derived':{'endpoint_change_pct':float(change),'endpoint_change_usd':int(delta)},
        'generated_materials':['assets/roof_texture.png','assets/facade_texture.png','assets/siding_texture.png','assets/family.png'],
        'data_edge_color':EDGE,'marker_center_color':MARKER,
    })
    print(f'Saved final.png: {W} x {H}; {len(values)} roof points; {len(text_records)} text items.')


if __name__ == '__main__':
    main()
