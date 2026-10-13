#!/usr/bin/env python3
"""Regenerate the poster from the frozen JSON and local image assets only.

Complete rendering attempts must be invoked through capture_render.py.
All chart vertices share one time mapping and one 0–1000 linear value mapping.
"""
from pathlib import Path
import json
import math
import hashlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
SCALE = 2
COLORS = {
    'A': '#56BDEC', 'B': '#F17D6F', 'C': '#ECF0D9', 'D': '#EAC467',
    'E': '#6DD3AC', 'F': '#B7A2E6', 'G': '#91B3D4', 'H': '#DB91AD',
}
INK = '#E9EBD8'
MUTED = '#A6AF99'
SAGE = '#AFBC88'
BASE = (13, 21, 15)


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def rgb(color):
    if isinstance(color, str):
        return tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
    return tuple(color)


def save_json(path, value):
    (ROOT / path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    inp = read_json('input.json')
    brief = read_json(inp['brief'])
    data = read_json(inp['data'])
    contract = read_json(inp['critical_metaphor_contract'])
    # The reference is a retained design dependency, never a pixel source.
    assert (ROOT / inp['reference']).is_file()
    assert contract['visible_title'] == data['title']
    assert data['synthetic'] and data['value_axis'] == {'minimum': 0, 'maximum': 1000, 'scale': 'linear'}
    years = data['time_points']
    series = data['series']
    assert len(years) == len(data['categories']) == 21
    assert len(series) == 8
    assert all(len(s['values']) == len(years) for s in series)
    W, H = brief['width'], brief['height']
    assert (W, H) == (1600, 2000)
    (ROOT / 'artifacts').mkdir(exist_ok=True)

    # A repeatable, restrained atmospheric background. The generated image
    # supplies all aircraft, their connected parts, and their visible trails.
    yy, xx = np.mgrid[0:H, 0:W]
    glow = np.exp(-(((xx-865)/770)**2 + ((yy-1090)/820)**2))
    grain = np.random.default_rng(2503).normal(0, 0.37, (H, W))
    back = np.clip(np.array(BASE)[None, None, :] + glow[..., None]*np.array([4, 5, 3]) + grain[..., None], 0, 255).astype('uint8')
    canvas = Image.fromarray(back, 'RGB').resize((W*SCALE, H*SCALE), Image.Resampling.BICUBIC).convert('RGBA')
    asset = Image.open(ROOT / 'assets/raw_01.png').convert('RGB')
    aw, ah = 1450, round(asset.height * 1450 / asset.width)
    asset = asset.resize((aw*SCALE, ah*SCALE), Image.Resampling.LANCZOS).convert('RGBA')
    ay, ax = np.mgrid[0:ah*SCALE, 0:aw*SCALE]
    edge = np.minimum.reduce([ax/(125*SCALE), (aw*SCALE-1-ax)/(125*SCALE), ay/(92*SCALE), (ah*SCALE-1-ay)/(130*SCALE)])
    alpha = (np.clip(edge, 0, 1)*0.83*255).astype('uint8')
    asset.putalpha(Image.fromarray(alpha, 'L'))
    canvas.alpha_composite(asset, (-20*SCALE, 684*SCALE))
    # Store the processed asset separately, without changing the raw original.
    asset.resize((aw, ah), Image.Resampling.LANCZOS).save(ROOT / 'assets/aircraft_layer.png')
    draw = ImageDraw.Draw(canvas)
    font_cache = {}
    text_boxes = []

    def font(size, kind='regular'):
        key = (size, kind)
        if key not in font_cache:
            filename = {'regular': 'Text-Regular.ttf', 'bold': 'Text-Bold.ttf', 'display': 'Display-Bold.otf'}[kind]
            font_cache[key] = ImageFont.truetype(str(ROOT / 'assets/fonts' / filename), round(size*SCALE))
        return font_cache[key]

    def text(x, y, value, size, color=INK, kind='regular', anchor='lt', track=0, tag=None):
        value = str(value)
        f = font(size, kind)
        p = (round(x*SCALE), round(y*SCALE))
        if track:
            assert anchor == 'lt'
            cursor = p[0]
            for c in value:
                draw.text((cursor, p[1]), c, font=f, fill=color, anchor='lt')
                cursor += draw.textlength(c, font=f) + track*SCALE
            bbox = [x, y, cursor/SCALE, y+size]
        else:
            draw.text(p, value, font=f, fill=color, anchor=anchor)
            bbox = [v/SCALE for v in draw.textbbox(p, value, font=f, anchor=anchor)]
        text_boxes.append({'text': value, 'bbox_px': bbox, 'tag': tag})
        return bbox

    def line(points, color, width=1):
        pts = [(round(x*SCALE), round(y*SCALE)) for x, y in points]
        draw.line(pts, fill=color, width=max(1, round(width*SCALE)), joint='curve')

    def circle(x, y, r, fill, outline=None, width=1):
        draw.ellipse(((x-r)*SCALE, (y-r)*SCALE, (x+r)*SCALE, (y+r)*SCALE), fill=fill, outline=outline, width=max(1, round(width*SCALE)))

    def rect(box, fill, outline=None, width=1, radius=0):
        b = tuple(round(v*SCALE) for v in box)
        if radius:
            draw.rounded_rectangle(b, radius=radius*SCALE, fill=fill, outline=outline, width=round(width*SCALE))
        else:
            draw.rectangle(b, fill=fill, outline=outline, width=round(width*SCALE))

    # Editorial heading: metric, region role, time window and synthetic status.
    text(88, 64, f'{data["categories"][0]} — {data["categories"][-1]}', 24, SAGE, 'bold', track=2)
    rect((1176, 52, 1512, 103), '#AFBC88')
    text(1344, 78, data['visible_notice'], 22, '#172116', 'bold', anchor='mm', tag='synthetic_notice')
    text(83, 127, 'DEFENCE SPENDING', 132, INK, 'display', tag='title')
    text(85, 268, 'BY REGION', 97, SAGE, 'display', tag='title')
    text(1020, 274, 'All eight regions finish', 25, INK)
    text(1020, 309, 'below their 2003 level.', 25, INK)
    text(88, 386, 'Annual expenditure', 27, INK, 'bold')
    text(453, 389, 'USD billion (synthetic)', 25, MUTED, tag='unit')
    line([(88, 435), (1512, 435)], '#3B4935', 1.2)

    # One and only one data transform.
    x0, x1, y0, y1 = 144.0, 1188.0, 493.0, 1690.0
    vmin, vmax = data['value_axis']['minimum'], data['value_axis']['maximum']
    def X(year):
        return x0+(year-years[0])/(years[-1]-years[0])*(x1-x0)
    def Y(value):
        return y1-(value-vmin)/(vmax-vmin)*(y1-y0)
    # Horizontal rules preserve the common scale, including a true zero.
    for value in range(vmin, vmax+1, 100):
        y = Y(value)
        if value % 200 == 0:
            line([(x0, y), (x1, y)], '#374531' if value else '#64725A', 1 if value else 1.6)
            text(116, y, f'{value:,}', 22, MUTED, anchor='rm', tag='y_tick')
        else:
            line([(x0-9, y), (x0, y)], '#536048', 1)
    text(144, 461, 'COMMON LINEAR SCALE', 16, MUTED, 'bold', track=1.4)
    for i, (year, label) in enumerate(zip(years, data['categories'])):
        x = X(year)
        line([(x, y1+2), (x, y1+(11 if i % 2 == 0 else 7))], '#758268', 1)
        if i % 2 == 0:
            text(x, y1+24, label, 22, INK if i in (0,20) else MUTED, 'bold' if i in (0,20) else 'regular', anchor='mt', tag='x_tick')
    text((x0+x1)/2, y1+74, 'YEAR', 16, MUTED, 'bold', anchor='mt', tag='x_axis_title')

    # Narrative annotation is derived from data. It does not change geometry.
    a = series[0]
    change = (a['values'][-1]/a['values'][0]-1)*100
    line([(192, 1052), (192, 1163)], COLORS['A'], 3)
    text(218, 1052, 'REGION A · 2003–2023', 19, COLORS['A'], 'bold', track=0.4)
    text(214, 1081, f'−{abs(change):.1f}%', 66, INK, 'display', tag='derived_change')
    text(218, 1145, f'{a["values"][0]} → {a["values"][-1]} USD billion', 22, MUTED)

    vertices = {}
    # Paths connect all observations without smoothing or interpolation overshoot.
    # Thinner low-series strokes help preserve the densely clustered differences.
    for s in series:
        sid = s['id']
        points = [(X(t), Y(v)) for t, v in zip(years, s['values'])]
        vertices[sid] = [{'year': t, 'value': v, 'x': p[0], 'y': p[1]} for t,v,p in zip(years,s['values'],points)]
        width = 5.4 if sid == 'A' else 4.2 if sid == 'B' else 2.7
        if sid in ('A', 'B'):
            line(points, '#111B17', width+2)
        line(points, COLORS[sid], width)
        for x, y in points:
            circle(x, y, 2.2 if sid in ('A','B') else 1.45, COLORS[sid])
        # Endpoint position is never displaced for label layout.
        circle(*points[0], 4.1 if sid in ('A','B') else 1.6, COLORS[sid])
        circle(*points[-1], 4.1 if sid == 'A' else 1.1, COLORS[sid])

    # Start labels for the two separated series; a footer supplies all starts.
    text(x0+12, Y(series[0]['values'][0])-34, str(series[0]['values'][0]), 25, COLORS['A'], 'bold', tag='start_label_A')
    text(x0+12, Y(series[1]['values'][0])-35, str(series[1]['values'][0]), 25, COLORS['B'], 'bold', tag='start_label_B')

    # Direct endpoint labels. Thin leaders are annotation, not data continuations.
    label_layout = {}
    yA = Y(series[0]['values'][-1])
    line([(x1+6, yA), (1282, yA)], '#4A8DAE', 1.4)
    text(1302, yA-34, series[0]['label'], 27, COLORS['A'], 'bold', tag='end_category_A')
    text(1300, yA+4, str(series[0]['values'][-1]), 65, COLORS['A'], 'display', tag='end_value_A')
    text(1302, yA+68, 'USD bn · 2023', 18, MUTED)
    label_layout['A'] = {'endpoint_y':yA, 'label_y':yA, 'value':series[0]['values'][-1]}

    text(1302, 1339, 'REGION', 16, MUTED, 'bold', track=1)
    text(1512, 1339, '2023', 16, MUTED, 'bold', anchor='rt')
    ordered = sorted(series[1:], key=lambda s:s['values'][-1], reverse=True)
    for i, s in enumerate(ordered):
        sid, value = s['id'], s['values'][-1]
        y, label_y = Y(value), 1401+44.5*i
        lead_color = tuple(round(c*0.66) for c in rgb(COLORS[sid]))
        line([(x1+3, y), (x1+15, y), (1280, label_y), (1289, label_y)], lead_color, 1.15)
        text(1302, label_y, s['label'], 23, COLORS[sid], anchor='lm', tag=f'end_category_{sid}')
        text(1512, label_y, str(value), 28, COLORS[sid], 'bold', anchor='rm', tag=f'end_value_{sid}')
        label_layout[sid] = {'endpoint_y':y, 'label_y':label_y, 'value':value}

    # Compact, truthful endpoint register also makes all initial values explicit.
    line([(88, 1812), (1512, 1812)], '#3B4935', 1.2)
    text(88, 1837, '2003 → 2023', 18, SAGE, 'bold')
    text(88, 1870, 'USD billion', 17, MUTED)
    for i,s in enumerate(series):
        col, row = i % 4, i // 4
        x, y = 339+294*col, 1837+44*row
        line([(x,y+12),(x+21,y+12)], COLORS[s['id']], 3)
        text(x+31, y, s['label'], 17, MUTED)
        text(x+139, y, f'{s["values"][0]} → {s["values"][-1]}', 18, INK, 'bold', tag=f'endpoint_register_{s["id"]}')
    text(88, 1947, 'SYNTHETIC DATA  ·  Illustrative regions, not reported national budgets.', 17, MUTED, tag='provenance')
    text(1512, 1947, '21 annual points per region', 17, MUTED, anchor='rt')

    final = canvas.convert('RGB').resize((W,H), Image.Resampling.LANCZOS)
    final.save(ROOT / 'final.png')
    save_json('artifacts/geometry.json', {
        'canvas_px':[W,H], 'plot_box_px':[x0,y0,x1,y1],
        'transform':{'x_min_year':years[0], 'x_max_year':years[-1], 'y_min_value':vmin, 'y_max_value':vmax,
                     'x_px_per_year':(x1-x0)/(years[-1]-years[0]), 'y_px_per_unit':(y1-y0)/(vmax-vmin)},
        'points':vertices, 'colors':COLORS, 'endpoint_label_layout':label_layout,
        'text_boxes':text_boxes, 'asset_placement_px':[-20,684,aw,ah],
        'raw_asset_sha256':hashlib.sha256((ROOT/'assets/raw_01.png').read_bytes()).hexdigest(),
    })
    print(f'Wrote final.png: {W}×{H}; {len(series)} series; {sum(len(s["values"]) for s in series)} vertices.')


if __name__ == '__main__':
    main()
