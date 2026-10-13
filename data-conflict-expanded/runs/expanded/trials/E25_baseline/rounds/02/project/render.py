#!/usr/bin/env python3
"""Offline poster renderer. Run only through capture_render.py for full candidates."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

ROOT = Path(__file__).resolve().parent
S = 2
BG = '#0c1511'
INK = '#f0f0df'
MUTED = '#a9b4a7'
COLORS = {
    'A': '#48b6ec', 'B': '#ff7968', 'C': '#eff0dd', 'D': '#e5c668',
    'E': '#54c9ad', 'F': '#a8b5f1', 'G': '#dca16b', 'H': '#d590b8',
}
FONTS = {
    'display': ROOT/'assets/fonts/display.otf',
    'body': ROOT/'assets/fonts/body.ttf',
    'bold': ROOT/'assets/fonts/body_bold.ttf',
}

def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))

def save_json(name, value):
    (ROOT/name).parent.mkdir(parents=True, exist_ok=True)
    (ROOT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def rgb(color):
    return tuple(int(color[i:i+2], 16) for i in (1, 3, 5))

def main():
    brief, data, contract = read('brief.json'), read('data.json'), read('core.json')
    inputs = read('input.json')
    assert inputs['reference'] == 'reference.png' and (ROOT/inputs['reference']).is_file()
    assert data['synthetic'] is True and contract['topic_zh'] == data['topic_zh']
    W, H = brief['width'], brief['height']
    times = np.asarray(data['time_points'], dtype=float)
    assert len(times) == len(data['categories']) and np.all(np.diff(times) > 0)
    assert data['value_axis'] == {'minimum': 0, 'maximum': 1000, 'scale': 'linear'}
    for series in data['series']:
        assert len(series['values']) == len(times)
        assert all(0 <= v <= 1000 for v in series['values'])
        assert sum(series['values']) == data['series_totals'][series['id']]
    # One affine mapping for every year and every series, with an explicit zero.
    left, right, top, bottom = 146.0, 1220.0, 480.0, 1730.0
    def px(t): return left + (float(t)-times[0])/(times[-1]-times[0])*(right-left)
    def py(v): return bottom - float(v)/1000.0*(bottom-top)
    geometry = {s['id']: [[px(t), py(v)] for t, v in zip(times, s['values'])] for s in data['series']}

    # Full generated photographic plate, gently graded only for foreground legibility.
    raw = Image.open(ROOT/'assets/raw_01.png').convert('RGB').resize((W, H), Image.Resampling.LANCZOS)
    source = np.asarray(raw, dtype=np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    vertical = np.interp(yy[:, 0], [0, 360, 1330, 1610, 1740, H], [.88, .97, .97, .44, .15, .10])
    horizontal = np.interp(xx[0], [0, 100, 1240, W], [.80, 1, 1, .57])
    weight = (vertical[:, None] * horizontal[None, :])[:, :, None]
    base = np.asarray(rgb(BG), dtype=np.float32)
    plate = Image.fromarray(np.uint8(np.clip(source*weight + base*(1-weight), 0, 255)))
    plate.save(ROOT/'assets/flight_background.png')
    canvas = plate.resize((W*S, H*S), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(canvas)
    text_boxes = []
    fonts = {}
    def font(size, family='body'):
        key = (size, family)
        if key not in fonts:
            fonts[key] = ImageFont.truetype(str(FONTS[family]), round(size*S))
        return fonts[key]
    def text(x, y, value, size, fill=INK, family='body', anchor='lt', tracking=0, record=True):
        f = font(size, family)
        if tracking:
            assert anchor == 'lt'
            initial = x
            for char in value:
                draw.text((x*S, y*S), char, font=f, fill=fill, anchor='lt')
                x += (draw.textlength(char, font=f)/S) + tracking
            box = [initial, y, x-tracking, y+size]
        else:
            draw.text((x*S, y*S), value, font=f, fill=fill, anchor=anchor)
            box = [v/S for v in draw.textbbox((x*S, y*S), value, font=f, anchor=anchor)]
        if record:
            text_boxes.append({'text': value, 'bbox_px': box})
    def line(points, fill, width=1):
        draw.line([(round(x*S), round(y*S)) for x,y in points], fill=fill, width=max(1, round(width*S)), joint='curve')
    def circle(x, y, radius, fill, outline=None, width=1):
        draw.ellipse(((x-radius)*S,(y-radius)*S,(x+radius)*S,(y+radius)*S),fill=fill,outline=outline,width=round(width*S))
    def rect(box, fill, outline=None, width=1):
        draw.rectangle(tuple(v*S for v in box), fill=fill, outline=outline, width=round(width*S))

    # Poster typography is rendered as code, never as generative raster lettering.
    rect((90, 65, 416, 111), '#c6ce9c')
    text(107, 78, data['visible_notice'], 24, BG, 'bold', tracking=1.7)
    text(1510, 84, '2003 — 2023', 25, '#c6cebc', 'body', anchor='rm')
    text(86, 145, 'DEFENCE SPENDING', 168, INK, 'display')
    text(92, 304, 'BY REGION', 43, '#c2cb9e', 'bold', tracking=3.5)
    text(1510, 329, 'Annual expenditure', 28, MUTED, anchor='rm')
    line([(90, 386), (1510, 386)], '#536046', 1.25)
    text(left, 415, 'USD billion (synthetic)', 26, '#cad1c1')
    text(1321, 437, '2023', 22, '#c7d0bf', 'bold', tracking=1.5)
    text(1510, 437, 'USD bn', 20, MUTED, anchor='rt')

    # Grid uses the very same mapping as the colored quantitative marks.
    for tick in range(0, 1001, 200):
        y = py(tick)
        grid = '#455046' if tick == 0 else '#39433b'
        line([(left, y), (right, y)], grid, 1.25 if tick == 0 else .8)
        text(left-20, y, f'{tick:,}', 24, '#aebaaa', anchor='rm')
    for year in times:
        x = px(year)
        major = int(year) in (2003, 2008, 2013, 2018, 2023)
        line([(x, bottom), (x, bottom+(11 if major else 6))], '#83907d' if major else '#53614f', 1.15)
        if major:
            text(x, bottom+29, str(int(year)), 27, '#d4dacb', 'bold', anchor='mt')
    text(left, bottom+83, '21 annual observations  /  8 regions  /  shared linear scale', 23, MUTED)

    # All 168 points are joined by straight segments, without smoothing or value floors.
    reverse_series = list(reversed(data['series']))
    for s in reverse_series:
        width = 5.0 if s['id']=='A' else 4.5 if s['id']=='B' else 3.3
        line(geometry[s['id']], '#11201b', width+2.2)
    for s in reverse_series:
        color = COLORS[s['id']]
        width = 5.0 if s['id']=='A' else 4.5 if s['id']=='B' else 3.3
        line(geometry[s['id']], color, width)
        for x,y in geometry[s['id']]:
            circle(x, y, 2.7 if s['id'] in 'AB' else 1.9, color)
    # Terminal values stay at their exact coordinates; only text labels are displaced.
    label_y = {'A': py(data['series'][0]['values'][-1]),
               'B': py(data['series'][1]['values'][-1]),
               'C': 1492, 'D': 1542, 'E': 1592, 'F': 1642, 'G': 1692, 'H': 1742}
    labels = []
    for s in data['series']:
        sid, color = s['id'], COLORS[s['id']]
        x,y = geometry[sid][-1]
        ly = label_y[sid]
        line([(x+6, y), (x+21, y), (1294, ly), (1306, ly)], color, 1.25)
        circle(x, y, 5, color, BG, 1.5)
        big = sid in 'AB'
        if big:
            text(1321, ly-35, s['label'], 25, color, anchor='lt')
            text(1321, ly+4, str(s['values'][-1]), 46, color, 'bold', anchor='lt')
        else:
            text(1321, ly, s['label'], 24, color, anchor='lm')
            text(1510, ly, str(s['values'][-1]), 31, color, 'bold', anchor='rm')
        labels.append({'id':sid,'label':s['label'],'value':s['values'][-1],
                       'endpoint_px':[x,y],'label_y_px':ly,'color':color})
    # One starting value makes the leading series' initial level explicit without crowding the baseline.
    x,y = geometry['A'][0]
    text(x+11, y-31, str(data['series'][0]['values'][0]), 26, COLORS['A'], 'bold', anchor='lb')
    circle(x, y, 4.2, COLORS['A'], BG, 1)

    line([(90, 1858), (1510, 1858)], '#526044', 1.2)
    text(92, 1891, 'SYNTHETIC SERIES', 20, '#bec8a6', 'bold', tracking=2)
    text(92, 1927, 'Illustrative values supplied for this study; not real-world estimates.', 22, '#aab8a4')
    text(1510, 1905, 'End labels show 2023 values.', 22, '#c9d3be', anchor='rm')
    text(1510, 1943, 'Source: local synthetic dataset', 20, '#aab8a4', anchor='rm')

    output = canvas.resize((W,H), Image.Resampling.LANCZOS)
    output.save(ROOT/'final.png')
    save_json('artifacts/render_geometry.json', {
        'canvas':[W,H], 'plot':[left,top,right,bottom], 'x_domain':[float(times[0]),float(times[-1])],
        'y_domain':[0,1000], 'pixels_per_year':(right-left)/(times[-1]-times[0]),
        'pixels_per_unit':(bottom-top)/1000, 'paths':geometry, 'endpoint_labels':labels,
        'series_colors':COLORS,'text_boxes':text_boxes,
        'render_backend':'Pillow, 2x supersampling',
        'asset_transform':{'raw':'assets/raw_01.png','processed':'assets/flight_background.png',
                           'resize':[W,H],'crop':None,'flip':False},
    })
    # Data-native SVG is supplied for inspection and reuse of the exact encoded paths.
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
           '<title>Annual defence spending by region — synthetic data</title>',
           '<desc>All eight series use the same linear 0–1000 USD billion axis and 2003–2023 year axis.</desc>']
    for s in data['series']:
        pairs=' '.join(f'{x:.5f},{y:.5f}' for x,y in geometry[s['id']])
        svg.append(f'<polyline id="region-{s["id"]}" aria-label="{s["label"]}" points="{pairs}" fill="none" stroke="{COLORS[s["id"]]}" stroke-width="3.3"/>')
    svg.append('</svg>')
    (ROOT/'chart_paths.svg').write_text('\n'.join(svg)+'\n',encoding='utf-8')
    print(f'Rendered {W} x {H}; {len(data["series"])} series; {sum(len(s["values"]) for s in data["series"])} points.')

if __name__ == '__main__':
    main()
