#!/usr/bin/env python3
"""Offline, data-bound rendering. Run via capture_render.py for every full attempt."""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent
S = 2


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def rgb(value):
    return tuple(int(value[i:i+2], 16) for i in (1, 3, 5))


def main():
    inputs = read_json('input.json')
    brief = read_json(inputs['brief'])
    data = read_json(inputs['data'])
    core = read_json(inputs['critical_metaphor_contract'])
    # Keep the frozen visual reference and all contracts as local dependencies.
    with Image.open(ROOT / inputs['reference']) as ref:
        ref.verify()
    assert core['visible_title'] == data['title']
    assert data['synthetic'] and len(data['series']) == 1
    series = data['series'][0]
    names = data['categories']
    values = np.array(series['values'], dtype=float)
    total = float(values.sum())
    assert len(names) == len(values) == 18
    assert abs(total - data['series_totals'][series['id']]) < 1e-9
    assert data['value_axis']['minimum'] == 0
    assert abs(data['value_axis']['maximum'] - total) < 1e-9
    W, H = brief['width'], brief['height']
    assert (W, H) == (1440, 1991)

    bg = '#F8F0E3'
    ink = '#293A40'
    muted = '#687575'
    red = '#B63E31'
    blue = '#297E98'
    palette = ['#377AA3', '#4A856C', '#BD584A', '#6577A4', '#278C86',
               '#C39440', '#89823C', '#3C9DA1', '#967BAE', '#DA865C',
               '#609772', '#B95B7B', '#5186B4', '#AA9C42', '#7E609E',
               '#C56C3D', '#627C7F', '#A24F89']

    im = Image.new('RGB', (W*S, H*S), rgb(bg))
    draw = ImageDraw.Draw(im)
    fonts = {}
    boxes = []

    def font(size, weight='regular'):
        key = (size, weight)
        if key not in fonts:
            file = {'regular': 'NimbusSans-Regular.otf',
                    'bold': 'NimbusSans-Bold.otf',
                    'narrow': 'NimbusSansNarrow-Bold.otf'}[weight]
            fonts[key] = ImageFont.truetype(str(ROOT / 'assets/fonts' / file), round(size*S))
        return fonts[key]

    def text(x, y, value, size, color=ink, weight='regular', align='left', tag=None):
        f = font(size, weight)
        width = draw.textlength(value, font=f)
        px = x*S - (width/2 if align == 'center' else width if align == 'right' else 0)
        py = y*S
        draw.text((px, py), value, font=f, fill=color, anchor='lt')
        bbox = draw.textbbox((px, py), value, font=f, anchor='lt')
        boxes.append({'tag': tag or value, 'text': value,
                      'bbox': [round(v/S, 2) for v in bbox]})
        return width/S

    def line(points, fill, width=1):
        draw.line([(round(x*S), round(y*S)) for x, y in points], fill=fill, width=max(1, round(width*S)))

    def circle(x, y, radius, fill=None, outline=None, width=1):
        draw.ellipse(((x-radius)*S, (y-radius)*S, (x+radius)*S, (y+radius)*S),
                     fill=fill, outline=outline, width=round(width*S))

    # Header: clear synthetic status, topic, series, and total.
    text(86, 58, 'NATURAL GAS  /  REGIONAL RESERVES', 18, muted, 'bold')
    draw.rounded_rectangle((1100*S, 46*S, 1354*S, 92*S), radius=23*S, fill=red)
    text(1227, 59, data['visible_notice'], 19, '#FFFFFF', 'bold', align='center')
    text(80, 116, 'GAS RESERVES', 113, red, 'narrow')
    text(84, 222, 'BY REGION', 68, blue, 'narrow')
    text(1349, 132, f'{total:.1f}', 86, ink, 'bold', align='right', tag='total_value')
    text(1347, 225, f'{series["unit"]}  /  TOTAL', 23, muted, 'bold', align='right')
    text(86, 296, f'{series["label"]} ({series["unit"]})', 23, ink)
    text(1352, 298, f'{len(names)} synthetic regional categories', 20, muted, align='right')
    line([(86, 338), (1354, 338)], '#CEC6B9', 1.2)

    # A single common vertical coordinate system for all 18 data bands.
    body_x, body_y, body_w, body_h = 450.0, 565.0, 540.0, 1240.0
    edges = body_y + body_h * np.r_[0, np.cumsum(values)] / total
    mids = (edges[:-1] + edges[1:]) / 2
    body_tex = Image.open(ROOT / 'assets/body_metal.png').convert('RGBA')
    body_tex = body_tex.resize((round(body_w*S), round(body_h*S)), Image.Resampling.LANCZOS)
    arr = np.asarray(body_tex).astype(float)
    lum = arr[:, :, :3] @ np.array([.2126, .7152, .0722])
    shade = np.clip((lum / 210.0) ** .90, .38, 1.25)
    local_centers = body_y + (np.arange(round(body_h*S)) + .5) / S
    row_ids = np.searchsorted(edges[1:], local_centers, side='right').clip(0, len(values)-1)
    base = np.array([rgb(p) for p in palette], dtype=float)[row_ids]
    pixels = np.empty_like(arr, dtype=np.uint8)
    pixels[:, :, :3] = np.clip(base[:, None, :] * shade[:, :, None], 0, 255).astype(np.uint8)
    pixels[:, :, 3] = arr[:, :, 3].astype(np.uint8)
    body_layer = Image.fromarray(pixels, 'RGBA')

    # A quiet contact shadow does not alter or continue the encoded height.
    shadow = Image.new('RGBA', im.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.ellipse((455*S, 1798*S, 995*S, 1820*S), fill=(39, 39, 27, 30))
    shadow = shadow.filter(ImageFilter.GaussianBlur(12*S))
    im.paste(shadow, (0, 0), shadow)
    im.paste(body_layer, (round(body_x*S), round(body_y*S)), body_layer)

    # Generated guard, valve and neck: registered to the same source transform.
    source_body_height = 1438 - 251
    sy = body_h / source_body_height
    guard = Image.open(ROOT / 'assets/guard_valve.png').convert('RGBA')
    guard = guard.resize((round(body_w*S), round(270*sy*S)), Image.Resampling.LANCZOS)
    guard_y = body_y + (40-251)*sy
    im.paste(guard, (round(body_x*S), round(guard_y*S)), guard)
    draw = ImageDraw.Draw(im)

    # Compact reading notes balance the upper shoulder without data-like marks.
    text(86, 526, 'HOW TO READ', 18, red, 'bold')
    for k, t in enumerate(['Each colored band is', 'one region. Its height', 'shows its share of the', 'total gas reserves.']):
        text(86, 561 + 29*k, t, 23, ink)
    line([(86, 699), (287, 699)], '#CEC6B9', 1)
    text(86, 721, 'tcm', 29, blue, 'bold')
    text(86, 761, 'trillion cubic meters', 20, muted)
    text(86, 797, 'Illustrative values only.', 19, muted)
    text(86, 834, 'Height: shoulder to base.', 19, muted)

    text(1104, 531, 'ONE TOTAL', 18, red, 'bold')
    text(1101, 569, str(len(names)), 68, ink, 'bold')
    text(1104, 646, 'regional categories', 21, ink)
    line([(1104, 692), (1354, 692)], '#CEC6B9', 1)
    for k, t in enumerate(['Listed order, top to bottom.', 'Band height encodes value;', 'colored area does not.']):
        text(1104, 718+28*k, t, 18, muted)

    label_records = []

    # The six most spacious bands retain labels directly on the object.
    for i in [0, 1, 2, 3, 4, 17]:
        cy = float(mids[i])
        badge_x = 538
        name_x = 577
        badge = chr(65+i) if i < 17 else '+'
        circle(badge_x, cy, 23, outline='#FFFFFF', width=1.5)
        text(badge_x, cy-12, badge, 25, '#FFFFFF', 'bold', align='center')
        text(name_x, cy-29, names[i], 30, '#FFFFFF', 'bold', tag=f'category_{i}')
        value_text = f'{values[i]:.1f} {series["unit"]}'
        width = text(name_x, cy+10, value_text, 25, '#FFFFFF', 'bold', tag=f'value_{i}')
        text(name_x+width+16, cy+13, f'{values[i]/total*100:.1f}%', 21, '#F9F1E6', tag=f'share_{i}')
        label_records.append({'category': names[i], 'value': float(values[i]), 'placement': 'inside', 'y': cy})

    # Twelve true-proportion small bands fan out to six labels on either side.
    # Leader lane order prevents line crossings even when labels move upward.
    for side, ids in [('left', list(range(5, 17, 2))), ('right', list(range(6, 17, 2)))]:
        for j, i in enumerate(ids):
            cy = float(mids[i])
            label_y = 1149 + j*105
            color = palette[i]
            if side == 'left':
                edge_x = 452.5
                elbow = 432 - j*18
                dot_x = 320
                label_x, align = 291, 'right'
            else:
                edge_x = 988
                elbow = 1009 + j*18
                dot_x = 1118
                label_x, align = 1147, 'left'
            line([(edge_x, cy), (elbow, cy), (elbow, label_y), (dot_x, label_y)], color, 1.35)
            circle(edge_x, cy, 2.5, fill=color)
            circle(dot_x, label_y, 7, fill=color)
            # Inner white glint gives a small badge-like cue without adding data height.
            circle(dot_x-1.6, label_y-1.6, 1.6, fill='#F8F0E3')
            text(label_x, label_y-28, names[i], 27, ink, 'bold', align=align, tag=f'category_{i}')
            text(label_x, label_y+7, f'{values[i]:.1f} {series["unit"]}  /  {values[i]/total*100:.1f}%',
                 21, muted, align=align, tag=f'value_{i}')
            label_records.append({'category': names[i], 'value': float(values[i]), 'placement': side,
                                  'band_mid_y': cy, 'label_y': label_y,
                                  'leader': [[edge_x,cy],[elbow,cy],[elbow,label_y],[dot_x,label_y]]})

    # A modest base caption explicitly establishes height, not area, as the encoding.
    text(720, 1839, f'BAND HEIGHT  =  SHARE OF THE {total:.1f} {series["unit"]} TOTAL', 17, muted, 'bold', align='center')
    line([(86, 1890), (1354, 1890)], '#CEC6B9', 1.2)
    text(86, 1911, 'SYNTHETIC DATA', 16, red, 'bold')
    text(86, 1940, 'Source: supplied local data.json · illustrative gas reserves, not real regional estimates.', 17, muted)
    text(1354, 1911, 'Values in tcm  /  Shares rounded to 0.1%', 17, muted, align='right')
    text(1354, 1940, 'Guard and valve are outside the data scale.', 17, muted, align='right')

    # These are implementation measurements, not an independent visual assessment.
    geometry = {
        'canvas': [W, H], 'supersampling': S,
        'data_height': {'x': body_x, 'top': body_y, 'bottom': body_y+body_h, 'height': body_h,
                        'unit': series['unit'], 'total': total, 'pixels_per_tcm': body_h/total},
        'bands': [{'category': n, 'value': float(v), 'color': palette[i],
                   'top': float(edges[i]), 'bottom': float(edges[i+1]),
                   'height': float(edges[i+1]-edges[i]), 'center_y': float(mids[i])}
                  for i, (n, v) in enumerate(zip(names, values))],
        'labels': label_records, 'text_boxes': boxes,
        'frozen_dependencies': [inputs['brief'], inputs['data'], inputs['reference'], inputs['critical_metaphor_contract'], 'input.json'],
        'image_assets': ['assets/body_metal.png', 'assets/guard_valve.png'],
    }
    (ROOT / 'artifacts').mkdir(exist_ok=True)
    (ROOT / 'artifacts/geometry.json').write_text(json.dumps(geometry, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    im.resize((W, H), Image.Resampling.LANCZOS).save(ROOT / 'final.png', optimize=True)
    print(f'Rendered {W}x{H}; {len(values)} bands; total {total:.1f} {series["unit"]}.')


if __name__ == '__main__':
    main()
