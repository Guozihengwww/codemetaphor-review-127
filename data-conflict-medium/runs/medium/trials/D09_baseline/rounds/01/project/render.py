"""Rebuild the infographic from frozen data and local generated textile assets.

Run through capture_render.py for every complete rendering attempt.
No model calls, network access, or random data are used.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parent
S = 2
INK = (13, 36, 59)
CREAM = (247, 241, 225)
MUTED = (160, 185, 204)
CORAL = (249, 108, 112)
GRID = (61, 89, 115)
X0, X1 = 148.0, 1460.0
YT, Y0 = 542.0, 1696.0
FONT_FILES = {
    'bold': 'assets/fonts/Display-Bold.otf',
    'regular': 'assets/fonts/Text-Regular.otf',
    'mono': 'assets/fonts/Numbers-Regular.ttf',
}


def write_json(rel, obj):
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def textile(name, size, tile_size=440):
    """Mirrored repeats keep woven fibers visible without introducing seams."""
    tile = Image.open(ROOT / f'assets/{name}.png').convert('RGB')
    tile = tile.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
    output = Image.new('RGB', size)
    for row, yy in enumerate(range(0, size[1], tile_size)):
        for col, xx in enumerate(range(0, size[0], tile_size)):
            part = ImageOps.mirror(tile) if col % 2 else tile
            part = ImageOps.flip(part) if row % 2 else part
            output.paste(part, (xx, yy))
    return output


def main():
    brief = json.loads((ROOT / 'brief.json').read_text(encoding='utf-8'))
    data = json.loads((ROOT / 'data.json').read_text(encoding='utf-8'))
    width, height = brief['width'], brief['height']
    assert (width, height) == (1600, 2000)
    years = np.asarray(data['time_points'], dtype=float)
    values = np.asarray(data['series'][0]['values'], dtype=float)
    assert len(years) == len(values) == len(data['categories']) == 136
    assert all(str(int(t)) == c for t, c in zip(years, data['categories']))
    assert np.all(np.diff(years) == 1)
    lo, hi = data['value_axis']['minimum'], data['value_axis']['maximum']
    assert (lo, hi, data['value_axis']['scale']) == (0, 130, 'linear')
    assert np.all((values >= lo) & (values <= hi))

    def x(year):
        return X0 + (year - years[0]) / (years[-1] - years[0]) * (X1 - X0)

    def y(value):
        return Y0 - (value - lo) / (hi - lo) * (Y0 - YT)

    points = [(x(t), y(v)) for t, v in zip(years, values)]
    by_year = {int(t): (float(v), (float(px), float(py))) for t, v, (px, py) in zip(years, values, points)}
    i_hist_end = int(np.where(years == 2024)[0][0])
    i_forecast = int(np.where(years == 2025)[0][0])
    history_peak = int(np.argmax(values[:i_forecast]))
    trough_70s = int(np.where((years >= 1970) & (years <= 1979))[0][np.argmin(values[(years >= 1970) & (years <= 1979)])])
    xf = x(2025)

    # The generated cloth contributes the actual material throughout the area.
    navy = np.asarray(textile('navy_weave', (width, height), 520), dtype=np.float32)
    lum = navy @ np.array([0.2126, 0.7152, 0.0722])
    centered = (lum - float(np.median(lum))) * 0.12
    vertical = np.linspace(1.5, -1.5, height, dtype=np.float32)[:, None]
    bg = np.array(INK, dtype=np.float32)[None, None, :] + centered[:, :, None] + vertical[:, :, None]
    canvas = Image.fromarray(np.clip(bg, 0, 255).astype('uint8')).resize((width*S, height*S), Image.Resampling.BICUBIC)
    draw = ImageDraw.Draw(canvas)
    fonts = {}
    text_boxes = []

    def font(size, kind='regular'):
        key = (size, kind)
        if key not in fonts:
            fonts[key] = ImageFont.truetype(str(ROOT / FONT_FILES[kind]), round(size*S))
        return fonts[key]

    def text(xx, yy, label, size, color=CREAM, kind='regular', anchor='lt', role='text'):
        label = str(label)
        f = font(size, kind)
        pos = (round(xx*S), round(yy*S))
        box = draw.textbbox(pos, label, font=f, anchor=anchor)
        draw.text(pos, label, fill=color, font=f, anchor=anchor, stroke_width=0)
        text_boxes.append({'text': label, 'role': role, 'bbox': [round(v/S, 2) for v in box]})

    def line(coords, fill, weight=1):
        draw.line([(round(a*S), round(b*S)) for a, b in coords], fill=fill, width=max(1, round(weight*S)), joint='curve')

    def rect(box, fill, outline=None, weight=1):
        draw.rectangle(tuple(round(v*S) for v in box), fill=fill, outline=outline, width=max(1, round(weight*S)))

    def circle(px, py, radius, fill, outline=None, weight=1):
        draw.ellipse(tuple(round(v*S) for v in (px-radius, py-radius, px+radius, py+radius)), fill=fill, outline=outline, width=max(1, round(weight*S)))

    def dashed(polyline, color, weight, pattern=(12.0, 8.0)):
        phase, remaining = 0, pattern[0]
        runs = []
        for a, b in zip(polyline[:-1], polyline[1:]):
            ax, ay = a
            bx, by = b
            length = math.hypot(bx-ax, by-ay)
            used = 0.0
            while used < length - 1e-9:
                step = min(remaining, length-used)
                p = (ax+(bx-ax)*used/length, ay+(by-ay)*used/length)
                q = (ax+(bx-ax)*(used+step)/length, ay+(by-ay)*(used+step)/length)
                if phase == 0:
                    line([p, q], color, weight)
                    runs.append([list(p), list(q)])
                used += step
                remaining -= step
                if remaining <= 1e-8:
                    phase = 1-phase
                    remaining = pattern[phase]
        return runs

    # Editorial hierarchy.
    rect((110, 86, 118, 109), CORAL)
    text(134, 84, data['visible_notice'], 27, CORAL, 'bold', role='synthetic_notice')
    text(1460, 86, f'{int(years[0])} — {int(years[-1])}', 27, MUTED, anchor='rt', role='time_range')
    text(108, 164, 'DEBT / OUTPUT', 160, CREAM, 'bold', role='title')
    text(113, 318, 'U.S. federal debt · an illustrative annual series', 33, CREAM, role='english_theme')

    line([(112, 409), (174, 409)], CORAL, 3.2)
    text(191, 395, 'Historical scenario · 1900–2024', 25, MUTED, role='legend_history')
    dashed([(647, 409), (708, 409)], CORAL, 3.2)
    text(725, 395, 'Projection · 2025–2035', 25, MUTED, role='legend_forecast')
    text(1460, 396, f'{len(values)} annual points', 24, MUTED, anchor='rt', role='point_count')
    line([(110, 453), (1490, 453)], GRID, 1)
    text(X0, 485, data['series'][0]['label'].upper(), 26, CREAM, 'bold', role='unit')

    # Forecast background and exact annual boundary.
    shade = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
    shade_draw = ImageDraw.Draw(shade)
    shade_draw.rectangle((round(xf*S), round(YT*S), round(X1*S), round(Y0*S)), fill=(233, 224, 211, 10))
    canvas = Image.alpha_composite(canvas.convert('RGBA'), shade).convert('RGB')
    draw = ImageDraw.Draw(canvas)

    ticks = [0, 20, 40, 60, 80, 100, 120, 130]
    for tick in ticks:
        py = y(tick)
        line([(X0-7, py), (X1, py)], GRID, 1.2 if tick in (0,100) else 0.8)
        text(X0-25, py, str(tick), 24, MUTED, anchor='rm', role='value_tick')

    # Generated print textures are bound to one shared area polygon, not warped data.
    plot_w, plot_h = round(X1-X0), round(Y0-YT)
    red = textile('red_weave', (plot_w, plot_h), 460)
    ivory = textile('ivory_weave', (plot_w, plot_h), 460)
    material = Image.new('RGB', (plot_w, plot_h))
    for stripe in range(13):
        top, bottom = round(stripe*plot_h/13), round((stripe+1)*plot_h/13)
        source = red if stripe % 2 == 0 else ivory
        material.paste(source.crop((0, top, plot_w, bottom)), (0, top))
    # A flag canton in the upper-left seven stripes; it is clipped by the same data mask.
    canton_w, canton_h = round(plot_w*0.43), round(plot_h*7/13)
    star = Image.open(ROOT / 'assets/star_weave.png').convert('RGB')
    star = star.resize((canton_w, canton_h), Image.Resampling.LANCZOS)
    material.paste(star, (0,0))
    forecast_cut = round(xf-X0)
    forecast_material = material.crop((forecast_cut, 0, plot_w, plot_h))
    forecast_material = Image.blend(forecast_material, Image.new('RGB', forecast_material.size, INK), 0.23)
    material.paste(forecast_material, (forecast_cut,0))
    material = material.resize((plot_w*S, plot_h*S), Image.Resampling.LANCZOS)
    mask = Image.new('L', canvas.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    polygon = [(X0,Y0)] + points + [(X1,Y0)]
    mask_draw.polygon([(round(px*S), round(py*S)) for px,py in polygon], fill=255)
    material_canvas = Image.new('RGB', canvas.size, INK)
    material_canvas.paste(material, (round(X0*S), round(YT*S)))
    canvas.paste(material_canvas, (0,0), mask)
    draw = ImageDraw.Draw(canvas)
    (ROOT / 'artifacts').mkdir(exist_ok=True)
    mask.resize((width,height), Image.Resampling.LANCZOS).save(ROOT / 'artifacts/area_mask.png')

    # Fine grid over the textile keeps the 0–130 linear scale readable.
    grid_layer = Image.new('RGBA', canvas.size, (0,0,0,0))
    gd = ImageDraw.Draw(grid_layer)
    for tick in ticks[1:-1]:
        py = round(y(tick)*S)
        gd.line([(round(X0*S),py),(round(X1*S),py)], fill=(8,31,53,45), width=S)
    grid_alpha = grid_layer.getchannel('A')
    grid_layer.putalpha(Image.fromarray((np.asarray(grid_alpha,dtype=np.uint16)*np.asarray(mask,dtype=np.uint16)//255).astype(np.uint8)))
    canvas = Image.alpha_composite(canvas.convert('RGBA'), grid_layer).convert('RGB')
    draw = ImageDraw.Draw(canvas)

    # All 136 supplied annual vertices survive. There is no smoothing or downsampling.
    history_points = points[:i_forecast]
    forecast_points = points[i_hist_end:]
    line(history_points, INK, 6.0)
    line(history_points, CORAL, 3.3)
    forecast_runs = dashed(forecast_points, CORAL, 4.0, (12,8))
    for px, py in points:
        circle(px,py,1.35,CORAL)
    dashed([(xf,YT),(xf,Y0)], (209,212,205), 1.3, (8,8))
    # A small cap identifies the first forecast year on the linear time axis.
    draw.polygon([(round((xf-5)*S),round((YT-10)*S)),(round((xf+5)*S),round((YT-10)*S)),(round(xf*S),round((YT-3)*S))],fill=MUTED)
    text(xf, 492, '2025', 21, MUTED, anchor='mt', role='forecast_boundary_year')

    # Baseline and yearly ticks; decade labels are sparse, vertices are not.
    line([(X0,Y0),(X1,Y0)], CREAM, 1.8)
    for year in years:
        major = int(year) % 10 == 0 or year == years[-1]
        line([(x(year),Y0),(x(year),Y0+(11 if major else 5))], MUTED, 1)
    tick_years = list(range(1900,2021,10)) + [2035]
    for year in tick_years:
        text(x(year), Y0+23, str(year), 25, CREAM, anchor='mt', role='time_tick')
    text(X1, Y0+70, 'YEAR', 21, MUTED, anchor='rt', role='axis_title')

    # Exact values are computed from the frozen input; each leader binds to its point.
    labels = []

    def mark(year, radius=6):
        value, (px,py) = by_year[year]
        circle(px,py,radius,CORAL,CREAM,1.3)
        return value,px,py

    v,px,py=mark(1900,5)
    line([(165,1605),(px,py-8)],MUTED,1.2)
    text(148,1509,'1900',26,CREAM,'bold',role='key_year')
    text(148,1547,f'{v:.1f}%',48,CORAL,'bold',role='key_value')
    labels.append({'year':1900,'value':v,'label':f'{v:.1f}%','point':[px,py]})

    peak_year=int(years[history_peak])
    v,px,py=mark(peak_year)
    line([(px,py-9),(px,714)],MUTED,1.2)
    text(510,626,f'{peak_year} · historical high',25,CREAM,role='key_year')
    text(510,665,f'{v:.1f}%',54,CORAL,'bold',role='key_value')
    labels.append({'year':peak_year,'value':v,'label':f'{v:.1f}%','point':[px,py]})

    valley_year=int(years[trough_70s])
    v,px,py=mark(valley_year,4.5)
    line([(px+7,py-7),(px+48,py-49)],MUTED,1.2)
    text(px+53,py-112,str(valley_year),25,CREAM,'bold',role='key_year')
    text(px+53,py-75,f'{v:.1f}%',44,CORAL,'bold',role='key_value')
    labels.append({'year':valley_year,'value':v,'label':f'{v:.1f}%','point':[px,py]})

    v,px,py=mark(2024)
    line([(1235,789),(1315,789),(px-4,py-8)],MUTED,1.2)
    text(1225,714,'2024',27,CREAM,'bold',anchor='rt',role='key_year')
    text(1225,754,f'{v:.1f}%',52,CORAL,'bold',anchor='rt',role='key_value')
    labels.append({'year':2024,'value':v,'label':f'{v:.1f}%','point':[px,py]})

    v,px,py=mark(2035,6.5)
    text(1460,533,'2035',27,CREAM,'bold',anchor='rt',role='key_year')
    text(1460,572,f'{v:.1f}%',54,CORAL,'bold',anchor='rt',role='key_value')
    labels.append({'year':2035,'value':v,'label':f'{v:.1f}%','point':[px,py]})

    # Breathing space inside the upper central region is used for a short reading cue.
    text(806,938,'ONE SERIES,',25,MUTED,'bold',role='reading_cue')
    text(806,976,'EVERY YEAR.',25,MUTED,'bold',role='reading_cue')
    text(806,1025,'The upper edge follows',27,CREAM,role='reading_cue')
    text(806,1063,'each annual observation.',27,CREAM,role='reading_cue')

    # The source, unit, and material metaphor are unambiguous.
    line([(110,1810),(1490,1810)],GRID,1)
    text(112,1840,'READING THE FLAG',22,CORAL,'bold',role='footer_heading')
    text(112,1881,'Equal year spacing · Linear 0–130% scale · Area closes at zero.',26,CREAM,role='method_note')
    text(112,1920,'Stars and stripes form the fabric, not additional data series.',26,MUTED,role='method_note')
    text(112,1970,'Source: supplied synthetic data · Percent of synthetic output',22,MUTED,role='source_unit')
    text(1488,1970,'ILLUSTRATIVE DATA',20,MUTED,'bold',anchor='rt',role='source_notice')

    final = canvas.resize((width,height),Image.Resampling.LANCZOS)
    final.save(ROOT / 'final.png', optimize=True)
    write_json('artifacts/geometry.json', {
        'canvas': [width,height],
        'plot': {'left':X0,'right':X1,'top':YT,'zero':Y0},
        'time_domain': [float(years[0]),float(years[-1])],
        'value_domain': [lo,hi],
        'pixels_per_year': (X1-X0)/(years[-1]-years[0]),
        'pixels_per_percentage_point': (Y0-YT)/(hi-lo),
        'interpolation': 'straight segments between every supplied annual vertex',
        'vertices': [{'year':int(t),'value':float(v),'x':float(px),'y':float(py),'period':'history' if t<=2024 else 'projection'} for t,v,(px,py) in zip(years,values,points)],
        'history_vertex_count': len(history_points),
        'projection_vertex_count': len(points[i_forecast:]),
        'forecast_boundary_x':xf,
        'dashed_connection': '2024 to 2025 connects the last historical observation to the first projected observation',
        'forecast_dash_runs': forecast_runs,
        'labels':labels,
        'canton':{'left':X0,'top':YT,'width':canton_w,'height':canton_h},
        'generated_material_pixels_in_mask': int(np.count_nonzero(np.asarray(mask.resize((width,height),Image.Resampling.LANCZOS))>127)),
    })
    write_json('artifacts/text_boxes.json',text_boxes)
    print(f'Rendered {width} × {height}; {len(points)} annual vertices; forecast begins at 2025.')


if __name__ == '__main__':
    main()
