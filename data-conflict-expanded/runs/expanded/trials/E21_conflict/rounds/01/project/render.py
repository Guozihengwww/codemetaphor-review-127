#!/usr/bin/env python3
"""Offline, data-bound rendering of the supplied annual forest-fire series.

Run through capture_render.py when producing a complete candidate. All inputs and
art assets are local. The fixed forest transform is independent of the data.
"""
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
SCALE = 2
BACKGROUND = '#eaf2e8'
INK = '#193e34'
MUTED = '#536f61'
LINE = '#c84b2d'
HALO = '#f6f6e9'
GRID = '#c3d4c5'
FONT_DIR = ROOT / 'assets' / 'fonts'


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def main():
    config = read_json('input.json')
    brief = read_json(config['brief'])
    data = read_json(config['data'])
    contract = read_json(config['critical_metaphor_contract'])
    assert contract['visible_title'] == data['title']
    assert data['synthetic'] is True
    assert len(data['series']) == 1
    series = data['series'][0]
    years = data['time_points']
    values = series['values']
    categories = data['categories']
    assert len(years) == len(values) == len(categories)
    assert all(str(int(year)) == label for year, label in zip(years, categories))
    assert all(a < b for a, b in zip(years, years[1:]))
    axis = data['value_axis']
    assert axis['scale'] == 'linear'
    assert axis['minimum'] == 0 and axis['maximum'] == 8_000_000
    assert all(axis['minimum'] <= value <= axis['maximum'] for value in values)
    assert sum(values) == data['series_totals'][series['id']]

    width, height = brief['width'], brief['height']
    assert (width, height) == (2000, 1600)
    canvas = Image.new('RGB', (width * SCALE, height * SCALE), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    font_cache = {}
    text_boxes = []

    def font(size, bold=False):
        key = (size, bold)
        if key not in font_cache:
            name = 'NimbusSans-Bold.otf' if bold else 'NimbusSans-Regular.otf'
            font_cache[key] = ImageFont.truetype(str(FONT_DIR / name), round(size * SCALE))
        return font_cache[key]

    def text(x, y, content, size=28, fill=INK, bold=False, align='left', tracking=0):
        f = font(size, bold)
        content = str(content)
        advance = draw.textlength(content, font=f) / SCALE
        if tracking:
            advance += max(0, len(content) - 1) * tracking
        left = x - (advance / 2 if align == 'center' else advance if align == 'right' else 0)
        if tracking:
            cursor = left
            for character in content:
                draw.text((round(cursor*SCALE), round(y*SCALE)), character, font=f, fill=fill, anchor='lt')
                cursor += draw.textlength(character, font=f) / SCALE + tracking
        else:
            draw.text((round(left*SCALE), round(y*SCALE)), content, font=f, fill=fill, anchor='lt')
        box = draw.textbbox((round(left*SCALE), round(y*SCALE)), content, font=f, anchor='lt')
        text_boxes.append({'text': content, 'bbox': [box[0]/SCALE, box[1]/SCALE,
                           left+advance, box[3]/SCALE]})

    def line(points, fill, stroke=1):
        points = [(round(x*SCALE), round(y*SCALE)) for x,y in points]
        draw.line(points, fill=fill, width=max(1, round(stroke*SCALE)), joint='curve')

    def circle(x, y, r, fill, outline=None, stroke=1):
        draw.ellipse(((x-r)*SCALE, (y-r)*SCALE, (x+r)*SCALE, (y+r)*SCALE),
                     fill=fill, outline=outline, width=round(stroke*SCALE))

    def rect(box, fill, radius=0, outline=None, stroke=1):
        box = tuple(round(v*SCALE) for v in box)
        draw.rounded_rectangle(box, radius=round(radius*SCALE), fill=fill,
                               outline=outline, width=round(stroke*SCALE))

    # Editorial header. The synthetic designation remains conspicuous.
    rect((90, 61, 372, 109), '#c84b2d', radius=4)
    text(231, 74, data['visible_notice'], 24, '#fff8ee', True, 'center', tracking=1.2)
    text(90, 147, data['title'].removeprefix('SYNTHETIC '), 88, INK, True)
    text(94, 250, 'Synthetic area burned, recorded annually in hectares.', 30, MUTED)
    text(1905, 157, f'{categories[0]}–{categories[-1]}', 40, INK, False, 'right')
    text(1905, 212, f'{len(values)} annual observations', 24, MUTED, False, 'right')
    line([(94, 307), (1905, 307)], '#acc5b1', 1.5)

    # Shared quantitative mapping. A year is an x-coordinate, not a rank.
    plot = {'left': 235.0, 'right': 1875.0, 'top': 408.0, 'bottom': 1240.0}
    year_min, year_max = min(years), max(years)
    x_scale = (plot['right'] - plot['left']) / (year_max-year_min)
    y_scale = (plot['bottom'] - plot['top']) / (axis['maximum']-axis['minimum'])

    def xy(year, value):
        return (plot['left']+(year-year_min)*x_scale,
                plot['bottom']-(value-axis['minimum'])*y_scale)

    points = [xy(year, value) for year, value in zip(years, values)]

    # Complete AI-created forest, extending through the plot. Its fixed transform
    # does not follow any peaks, categories, values, or segment lengths.
    art_box = (178, 353, 1756, 981)
    forest = Image.open(ROOT/'assets/forest_watercolor.png').convert('RGBA')
    forest = forest.resize((art_box[2]*SCALE, art_box[3]*SCALE), Image.Resampling.LANCZOS)
    canvas.paste(forest, (art_box[0]*SCALE, art_box[1]*SCALE), forest)
    draw = ImageDraw.Draw(canvas)

    text(plot['left'], 347, 'AREA BURNED', 23, MUTED, True, tracking=1.5)
    text(446, 347, '(hectares)', 25, MUTED)
    line([(1492, 358), (1543, 358)], LINE, 6.5)
    circle(1517.5, 358, 4, LINE)
    text(1560, 345, series['label'], 25, INK)

    for value in range(axis['minimum'], axis['maximum']+1, 1_000_000):
        _, y = xy(year_min, value)
        line([(plot['left'], y), (plot['right'], y)], GRID, 1.1 if value else 1.8)
        text(207, y-11, f'{value:,}', 25, MUTED, False, 'right')

    # Sparse labelled years plus an exact annual tick for every observation.
    major_years = [int(year_min)] + [year for year in range(1965, 2020, 5) if year < year_max] + [int(year_max)]
    for year in years:
        x, _ = xy(year, 0)
        major = int(year) in major_years
        line([(x, plot['bottom']), (x, plot['bottom']+(13 if major else 6))], '#849f8b', 1.3)
        if major:
            text(x, 1268, int(year), 26, INK, False, 'center')
    text(plot['right'], 1310, 'Year', 23, MUTED, False, 'right')

    # All 61 vertices are drawn in input order; no smoothing or value floors.
    line(points, HALO, 11)
    for x,y in points:
        circle(x,y,5.5,HALO)
    line(points, LINE, 6.5)
    for x,y in points:
        circle(x,y,3.5,LINE)

    # Three largest observations are annotated with exact numbers.
    ranked = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    peak_i = ranked[0]
    for i in ranked[:3]:
        x,y = points[i]
        circle(x,y,8,HALO)
        circle(x,y,5,LINE)
        if i == peak_i:
            # The leader terminates at the true data vertex, not the label box.
            bx, by = x+31, y-78
            rect((bx-11, by-10, bx+261, by+57), '#edf3e9', radius=5)
            line([(x+4,y-5),(bx-7,y-22)], LINE, 1.6)
            text(bx, by, f'PEAK · {categories[i]}', 21, LINE, True, tracking=0.7)
            text(bx, by+29, f'{values[i]:,} ha', 30, INK, True)
        else:
            bx,by=x,y-83
            rect((bx-106,by-10,bx+106,by+53),'#edf3e9',radius=5)
            text(bx,by,categories[i],22,MUTED,True,'center')
            text(bx,by+30,f'{values[i]:,} ha',28,INK,True,'center')
            line([(x,y-9),(x,by+64)],LINE,1.5)

    # The minimum remains at 230,000: 23.92 px above zero, without inflation.
    minimum_i = min(range(len(values)), key=lambda i: values[i])
    min_x,min_y = points[minimum_i]
    circle(min_x,min_y,6.8,HALO)
    circle(min_x,min_y,4.3,LINE)

    # Compact reading guide with exact derived statistics.
    line([(94, 1361), (1905, 1361)], '#acc5b1', 1.5)
    stats = [
        (94, f'HIGHEST ANNUAL AREA · {categories[peak_i]}', values[peak_i]),
        (729, f'LOWEST ANNUAL AREA · {categories[minimum_i]}', values[minimum_i]),
        (1336, f'PERIOD TOTAL · {len(values)} YEARS', sum(values)),
    ]
    for x,label,value in stats:
        text(x,1391,label,20,MUTED,True,tracking=0.6)
        text(x,1429,f'{value:,}',47,INK,True)
        number_width = draw.textlength(f'{value:,}',font=font(47,True))/SCALE
        text(x+number_width+14,1450,'hectares',23,MUTED)
    line([(675,1391),(675,1490)],'#c3d4c5',1)
    line([(1281,1391),(1281,1490)],'#c3d4c5',1)
    text(94,1542,'Synthetic study data · Not historical estimates. Forest texture is illustrative; only the line encodes area.',23,MUTED)

    output = canvas.resize((width,height),Image.Resampling.LANCZOS)
    output.save(ROOT/'final.png', optimize=True)

    # Reusable numerical audit data, expressed in final-image pixel coordinates.
    geometry = {
        'canvas': [width,height], 'plot': plot, 'axis': axis,
        'x_pixels_per_year': x_scale, 'y_pixels_per_hectare': y_scale,
        'line_color': LINE, 'line_width_px': 6.5,
        'points': [{'year':categories[i],'value':values[i],'x':p[0],'y':p[1]} for i,p in enumerate(points)],
        'annotated_years': [categories[i] for i in ranked[:3]],
        'major_x_ticks': major_years,
        'forest_box': list(art_box), 'forest_data_dependent': False,
        'minimum_value_zero_separation_px': plot['bottom']-min_y,
        'labels': text_boxes,
    }
    (ROOT/'artifacts').mkdir(exist_ok=True)
    (ROOT/'artifacts/geometry.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'width':width,'height':height,'annual_points':len(points),
                      'maximum_year':categories[peak_i],'minimum_year':categories[minimum_i],
                      'minimum_height_px':plot['bottom']-min_y},ensure_ascii=False))


if __name__ == '__main__':
    main()
