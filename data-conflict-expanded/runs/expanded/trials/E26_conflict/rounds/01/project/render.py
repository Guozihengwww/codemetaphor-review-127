#!/usr/bin/env python3
"""Offline poster renderer: generated print material + exact quarterly GDP polygon.

Run from this directory with the supplied Python interpreter. For the experiment,
all complete rendering attempts must be invoked through capture_render.py.
"""
from pathlib import Path
import json
import math
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'assets'
ARTIFACTS = ROOT / 'artifacts'
S = 3


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def main():
    inputs = read_json('input.json')
    brief = read_json(inputs['brief'])
    data = read_json(inputs['data'])
    core = read_json(inputs['critical_metaphor_contract'])
    assert data['title'] == core['visible_title']
    assert data['synthetic'] and len(data['series']) == 1
    series = data['series'][0]
    times = data['time_points']
    values = series['values']
    categories = data['categories']
    assert len(times) == len(values) == len(categories)
    assert all(b > a for a, b in zip(times, times[1:]))
    assert data['value_axis'] == {'minimum': 0, 'maximum': 800, 'scale': 'linear'}
    W, H = brief['width'], brief['height']
    assert (W, H) == (1600, 2000)

    # One common linear mapping; there is no smoothing or minimum-size rule.
    left, right, top, baseline = 172.0, 1486.0, 700.0, 1710.0
    x = lambda t: left + (t - times[0]) / (times[-1] - times[0]) * (right - left)
    y = lambda v: baseline - v / 800.0 * (baseline - top)
    points = [(x(t), y(v)) for t, v in zip(times, values)]
    polygon = points + [(right, baseline), (left, baseline)]
    image = Image.open(ASSETS / 'paper_background.png').convert('RGB').resize((W*S, H*S), Image.Resampling.BICUBIC)
    draw = ImageDraw.Draw(image)
    ink = '#20282c'
    ivory = '#faf4e5'
    muted = '#34464c'
    label_boxes = []
    fonts = {}

    def font(size, style='sans'):
        key = (size, style)
        if key not in fonts:
            fonts[key] = ImageFont.truetype(str(ASSETS / 'fonts' / (style + '.otf')), round(size*S))
        return fonts[key]

    def text(value, at, size=30, style='sans', fill=ink, align='left', role='text'):
        f = font(size, style)
        tx, ty = at
        width = draw.textlength(value, font=f)/S
        if align == 'right': tx -= width
        if align == 'center': tx -= width/2
        bbox = draw.textbbox((tx*S, ty*S), value, font=f, anchor='lt')
        draw.text((tx*S, ty*S), value, font=f, fill=fill, anchor='lt')
        label_boxes.append({'text': value, 'role': role, 'bbox': [round(a/S, 3) for a in bbox]})

    def line(coords, fill=ivory, width=1.4):
        draw.line([(round(a*S), round(b*S)) for a,b in coords], fill=fill, width=max(1, round(width*S)), joint='curve')

    def rect(box, fill, outline=None, width=1):
        draw.rectangle(tuple(round(a*S) for a in box), fill=fill, outline=outline, width=round(width*S))

    def marker(index, radius=4.6):
        px, py = points[index]
        draw.ellipse(((px-radius)*S, (py-radius)*S, (px+radius)*S, (py+radius)*S), fill=ivory)

    def number(v):
        return format(v, 'g')

    # Editorial header; every letter and numeral is executable typography.
    text('GERMAN FLAG / QUARTERLY SERIES', (108, 61), 25, 'bold')
    rect((1180, 44, 1486, 94), ink)
    text(data['visible_notice'], (1333, 58), 26, 'bold', ivory, 'center', 'synthetic_notice')
    text('GERMANY', (98, 120), 319, 'display', ink, role='geographic_metaphor')
    title_strip = Image.open(ASSETS / 'red_print.png').resize((1380*S, 139*S), Image.Resampling.LANCZOS)
    image.paste(title_strip, (106*S, 366*S))
    text(data['title'], (130, 382), 147, 'display', ivory, role='title')
    text(series['label'], (108, 540), 35, 'bold', role='metric')
    text(series['unit'], (108, 586), 30, 'sans', muted, role='unit')
    text(f'{len(values)} QUARTERLY OBSERVATIONS', (1486, 544), 25, 'bold', ink, 'right')
    text(f'{categories[0]} — {categories[-1]}', (1486, 585), 28, 'sans', muted, 'right', 'period')

    # The generated raster material fills the exact data polygon. Color boundaries
    # are decorative horizontal flag stripes, not extra statistical series.
    mask = Image.new('L', image.size, 0)
    md = ImageDraw.Draw(mask)
    md.polygon([(round(px*S), round(py*S)) for px,py in polygon], fill=255)
    fabric = Image.new('RGB', image.size, (0,0,0))
    for name, low, high in [('black', 400, 800), ('red', 200, 400), ('gold', 0, 200)]:
        box = (round(left*S), round(y(high)*S), round(right*S), round(y(low)*S))
        patch = Image.open(ASSETS / f'{name}_print.png').convert('RGB').resize((box[2]-box[0]+1, box[3]-box[1]+1), Image.Resampling.LANCZOS)
        fabric.paste(patch, box[:2])
    image.paste(fabric, (0,0), mask)
    draw = ImageDraw.Draw(image)

    # Subtle grid, always in the same zero-based coordinate system.
    grid = Image.new('RGBA', image.size, (0,0,0,0))
    gd = ImageDraw.Draw(grid)
    for tick in range(0, 801, 100):
        yy = y(tick)
        alpha = 82 if tick % 200 == 0 else 34
        gd.line([(left*S, yy*S), (right*S, yy*S)], fill=(250,244,229,alpha), width=3 if tick%200==0 else 2)
    image = Image.alpha_composite(image.convert('RGBA'), grid).convert('RGB')
    draw = ImageDraw.Draw(image)
    for tick in range(0, 801, 200):
        text(str(tick), (148, y(tick)-12), 29, 'bold', ink, 'right', 'y_tick')
        line([(156, y(tick)), (left-3, y(tick))], muted, 1.3)
    line([(left, baseline), (right, baseline)], ink, 2.0)
    # A fine ivory edge separates black ink from the paper; exact centers follow
    # every supplied quarterly point. No smoothing of troughs or peaks.
    line(points, ivory, 1.8)

    for year in range(math.ceil(times[0]), math.floor(times[-1])+1):
        xx = x(year)
        line([(xx, baseline+1), (xx, baseline+8)], muted, 1.1)
    for year in [1991, 1995, 2000, 2005, 2010, 2015, 2020, 2024]:
        xx = x(year)
        line([(xx, baseline), (xx, baseline+14)], ink, 1.4)
        text(str(year), (xx, 1734), 31, 'bold', ink, 'center', 'x_tick')

    # Labels select quarters by their semantic identifiers, never by template index.
    def idx(category): return categories.index(category)
    i_first, i_peak = 0, values.index(max(values))
    i_dip, i_low, i_rebound, i_last = idx('1995-Q1'), idx('2006-Q2'), idx('2007-Q2'), len(values)-1
    marker(i_first)
    line([(points[i_first][0], points[i_first][1]-6), (points[i_first][0], 705)], ivory, 1.4)
    text(categories[i_first], (181, 630), 24, 'bold', role='quarter_label')
    text(number(values[i_first]), (180, 663), 55, 'bold', role='value_label')

    marker(i_peak, 3.6)
    line([points[i_peak], (362, 685), (404, 685)], ivory, 1.4)
    text('PEAK / ' + categories[i_peak], (421, 630), 23, 'bold', role='quarter_label')
    text(number(values[i_peak]), (420, 663), 51, 'bold', role='value_label')

    marker(i_dip)
    px, py = points[i_dip]
    line([(px, py+6), (px, 898), (286, 930)], ivory, 1.3)
    text(categories[i_dip], (205, 944), 25, 'sans', ivory, role='quarter_label')
    text(number(values[i_dip]), (204, 980), 57, 'bold', ivory, role='value_label')

    marker(i_low)
    px, py = points[i_low]
    line([(px, py+6), (px, 1009), (727, 1042)], ivory, 1.3)
    text(categories[i_low], (647, 1049), 25, 'sans', ivory, role='quarter_label')
    text(number(values[i_low]), (646, 1085), 57, 'bold', ivory, role='value_label')

    marker(i_rebound)
    px, py = points[i_rebound]
    line([(px, py-6), (835, 831), (854, 831)], ivory, 1.4)
    text(categories[i_rebound], (872, 770), 25, 'bold', role='quarter_label')
    text(number(values[i_rebound]), (870, 807), 55, 'bold', role='value_label')

    marker(i_last, 6.6)
    px, py = points[i_last]
    line([(px, py-8), (px, 1044), (1369, 1044)], ivory, 1.4)
    text(categories[i_last], (1318, 925), 29, 'bold', role='quarter_label')
    text(number(values[i_last]), (1316, 964), 88, 'bold', role='value_label')

    # This note lives within the printed flag and clarifies the single-series fill.
    text('ONE SERIES', (1448, 1136), 21, 'bold', ivory, 'right')
    text('Black, red and gold are one flag.', (1448, 1167), 25, 'sans', ivory, 'right')

    # A short reading key and truthful source treatment replace the old real-world claims.
    line([(108, 1821), (1486, 1821)], muted, 1.2)
    text('HOW TO READ', (108, 1851), 23, 'bold', ink)
    text('The flag’s upper edge is the GDP level; the area closes to zero.', (329, 1848), 28, 'sans', ink)
    text('Source: supplied synthetic dataset. These are not observed German GDP statistics.', (108, 1901), 27, 'sans', muted)
    text('Quarterly level, not a growth rate. No seasonal-adjustment claim is supplied.', (108, 1944), 25, 'sans', muted)

    ARTIFACTS.mkdir(exist_ok=True)
    image.resize((W, H), Image.Resampling.LANCZOS).save(ROOT/'final.png')
    mask.resize((W,H), Image.Resampling.LANCZOS).save(ARTIFACTS/'data_area_mask.png')
    geometry = {
        'canvas': [W,H], 'series_id': series['id'], 'unit': series['unit'],
        'plot': {'left': left, 'right': right, 'top': top, 'zero_y': baseline},
        'mapping': {'time_min': times[0], 'time_max': times[-1], 'value_min': 0, 'value_max': 800,
                    'px_per_value_unit': (baseline-top)/800, 'supersampling': S},
        'point_count': len(points), 'linear_segments': len(points)-1,
        'closure_points': [[right,baseline],[left,baseline]],
        'points': [{'category': c,'time': t,'value': v,'x': px,'y': py} for c,t,v,(px,py) in zip(categories,times,values,points)],
        'annotated_categories': [categories[i] for i in [i_first,i_peak,i_dip,i_low,i_rebound,i_last]],
        'label_boxes': label_boxes,
        'flag_boundaries_value_units': [0,200,400],
    }
    (ARTIFACTS/'geometry.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Rendered {W} x {H}; {len(points)} quarterly vertices; {len(points)-1} straight segments; one series.')


if __name__ == '__main__':
    main()
