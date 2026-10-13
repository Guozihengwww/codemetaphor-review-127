"""Offline renderer. Run through capture_render.py to preserve each full attempt."""
from pathlib import Path
import json
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
S = 2
INK = '#182125'
CREAM = '#fff8e9'
MUTED = '#26383f'


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def fmt(v):
    return str(int(v)) if float(v).is_integer() else str(v)


def main():
    inputs = read_json('input.json')
    brief = read_json(inputs['brief'])
    data = read_json(inputs['data'])
    core = read_json(inputs['critical_metaphor_contract'])
    assert data['synthetic'] is True
    assert data['title'] == core['visible_title']
    assert len(data['series']) == 1
    series = data['series'][0]
    categories = data['categories']
    times = np.asarray(data['time_points'], dtype=float)
    values = np.asarray(series['values'], dtype=float)
    assert len(categories) == len(times) == len(values)
    assert np.all(np.diff(times) > 0)
    assert np.isfinite(values).all()
    assert data['value_axis'] == {'minimum': 0, 'maximum': 800, 'scale': 'linear'}
    W, H = brief['width'], brief['height']
    assert (W, H) == (1600, 2000)
    x0, x1, ytop, y0 = 155., 1460., 520., 1740.
    vmin, vmax = data['value_axis']['minimum'], data['value_axis']['maximum']

    def X(t):
        return x0 + (float(t) - times[0]) / (times[-1] - times[0]) * (x1 - x0)

    def Y(v):
        return y0 - (float(v) - vmin) / (vmax - vmin) * (y0 - ytop)

    points = [(X(t), Y(v)) for t, v in zip(times, values)]
    def xy(p):
        return tuple(round(q * S) for q in p)

    im = Image.open(ROOT / 'assets/blue_paper.png').convert('RGB').resize((W*S, H*S), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(im)
    text_boxes = []
    font_paths = {
        'regular': ROOT / 'assets/fonts/DejaVuSans.ttf',
        'bold': ROOT / 'assets/fonts/DejaVuSans-Bold.ttf',
        'display': ROOT / 'assets/fonts/NimbusSansNarrow-Bold.otf',
    }
    fonts = {}

    def font(size, kind='regular'):
        key = (size, kind)
        if key not in fonts:
            fonts[key] = ImageFont.truetype(str(font_paths[kind]), round(size*S))
        return fonts[key]

    def text(content, x, top, size, kind='regular', fill=INK, align='left', role='label'):
        f = font(size, kind)
        box = f.getbbox(content)
        width = f.getlength(content)
        left = x*S - (width if align == 'right' else width/2 if align == 'center' else 0)
        pos = (left-box[0], top*S-box[1])
        draw.text(pos, content, font=f, fill=fill)
        actual = draw.textbbox(pos, content, font=f)
        text_boxes.append({'text':content, 'role':role, 'bbox':[round(n/S,3) for n in actual]})
        return width / S

    def line(coords, fill, width=1):
        draw.line([xy(p) for p in coords], fill=fill, width=max(1,round(width*S)), joint='curve')

    def dot(p, radius=5, fill=CREAM):
        x, y = p
        draw.ellipse(xy((x-radius,y-radius,x+radius,y+radius)), fill=fill)

    # Generated print material forms the entire data-bearing object.
    polygon = [(x0,y0)] + points + [(x1,y0)]
    area_mask = Image.new('L', (W*S,H*S), 0)
    mask_draw = ImageDraw.Draw(area_mask)
    mask_draw.polygon([xy(p) for p in polygon], fill=255)
    material = Image.new('RGB',(W*S,H*S))
    bands = [('black',ytop,Y(400)), ('red',Y(400),Y(200)), ('gold',Y(200),y0)]
    for name, top, bottom in bands:
        tile = Image.open(ROOT / f'assets/print_{name}.png').convert('RGB')
        left_i, right_i = round(x0*S), round(x1*S)+1
        top_i, bottom_i = round(top*S), round(bottom*S)+1
        tile = tile.resize((right_i-left_i,bottom_i-top_i),Image.Resampling.LANCZOS)
        material.paste(tile,(left_i,top_i))
    im.paste(material,(0,0),area_mask)
    draw = ImageDraw.Draw(im)

    # Restrained grid, with one scale for the entire three-colour silhouette.
    overlay = Image.new('RGBA', im.size, (0,0,0,0))
    od = ImageDraw.Draw(overlay)
    for v in range(100,801,100):
        yy = Y(v)
        # Draw the in-area and out-of-area portions with appropriate contrast.
        grid = Image.new('RGBA', im.size, (0,0,0,0))
        gd = ImageDraw.Draw(grid)
        gd.line([xy((x0,yy)),xy((x1,yy))], fill=(255,248,233,40 if v%200==0 else 20), width=2 if v%200==0 else 1)
        a = np.minimum(np.asarray(grid.getchannel('A')),np.asarray(area_mask))
        grid.putalpha(Image.fromarray(a))
        overlay = Image.alpha_composite(overlay,grid)
        if v % 200 == 0:
            outside = Image.new('RGBA', im.size, (0,0,0,0))
            xd = ImageDraw.Draw(outside)
            xd.line([xy((x0,yy)),xy((x1,yy))],fill=(24,33,37,30),width=2)
            aa = np.minimum(np.asarray(outside.getchannel('A')),255-np.asarray(area_mask))
            outside.putalpha(Image.fromarray(aa))
            overlay = Image.alpha_composite(overlay,outside)
    im = Image.alpha_composite(im.convert('RGBA'), overlay).convert('RGB')
    draw = ImageDraw.Draw(im)
    # Straight segments through every supplied quarterly point, without smoothing.
    line(points, CREAM, 2.1)
    line([(x0,Y(values[0])),(x0,y0),(x1,y0),(x1,Y(values[-1]))], '#233035', 1.3)

    # Poster masthead and prominent synthetic-data notice.
    text('ECONOMY / QUARTERLY SERIES', 110, 64, 21, 'bold', role='eyebrow')
    draw.rectangle(xy((1159,49,1490,103)), fill=INK)
    text(data['visible_notice'], 1324.5, 66, 25, 'bold', fill='#f2c443', align='center', role='synthetic_notice')
    text('GERMANY', 104, 126, 300, 'display', role='country_theme')
    title_rect = (110,365,1176,469)
    red = Image.open(ROOT/'assets/print_red.png').resize((round((title_rect[2]-title_rect[0])*S), round((title_rect[3]-title_rect[1])*S)),Image.Resampling.LANCZOS)
    im.paste(red,xy(title_rect[:2]))
    draw=ImageDraw.Draw(im)
    text(data['title'], 130, 387, 81, 'display', fill=CREAM, role='title')

    # Metric and units occupy the open area above the early observations.
    text('REAL GDP LEVEL', x0, 575, 34, 'display', role='metric')
    text(series['unit'], x0, 625, 27, role='unit')
    text('Quarterly observations · level, not growth rate', x0, 668, 23, fill=MUTED, role='statistical_scope')

    # Tick labels are outside the flag so colour changes cannot imply series.
    for v in [0,200,400,600,800]:
        yy=Y(v)
        text(fmt(v),x0-24,yy-12,26,'display',align='right',role='value_tick')
        line([(x0-10,yy),(x0,yy)],INK,1)
    for t in times:
        xx=X(t)
        line([(xx,y0),(xx,y0+5)],INK,.7)
    for yr in [1991,1995,2000,2005,2010,2015,2020,2024]:
        xx=X(yr)
        line([(xx,y0),(xx,y0+11)],INK,1.4)
        text(str(yr),xx,y0+27,29,'display',align='center',role='year_tick')

    # Key-point labels bind directly to their quarter and unrounded value.
    label_specs = []
    p=points[0]
    line([p,(p[0],896),(168,896)],INK,1.3)
    dot(p,5)
    text(categories[0],179,850,25,'bold',role='point_category')
    text(fmt(values[0]),177,891,70,'display',role='point_value')
    label_specs.append({'category':categories[0],'value':float(values[0]),'point':p})

    for cat,tx,ty,bend_x in [('2009-Q1',778,887,861.466),('2020-Q2',1320,780,1303.045)]:
        i=categories.index(cat)
        p=points[i]
        target_y=ty-18
        line([p,(p[0],target_y),(tx+14,target_y)],CREAM,1.3)
        dot(p,4.5)
        text(cat,tx,ty,24,'bold',fill=CREAM,role='point_category')
        text(fmt(values[i]),tx,ty+40,65,'display',fill=CREAM,role='point_value')
        label_specs.append({'category':cat,'value':float(values[i]),'point':p})

    p=points[-1]
    line([p,(p[0],490)],CREAM,1.5)
    dot(p,6)
    text(categories[-1],1460,365,28,'bold',align='right',role='point_category')
    text(fmt(values[-1]),1465,413,95,'display',align='right',role='point_value')
    label_specs.append({'category':categories[-1],'value':float(values[-1]),'point':p})

    # Explain the fused encoding, rather than introducing an additional series.
    text('The upper edge traces GDP.',1422,988,27,fill=CREAM,align='right',role='encoding_note')
    text('One flag. One level series.',1422,1033,27,fill=CREAM,align='right',role='encoding_note')

    # Explicit period, count, and synthetic provenance.
    text(f'{categories[0]} — {categories[-1]}',110,1832,26,'bold',role='period')
    text(f'{len(values)} quarters · {len(data["series"])} series',1490,1832,25,align='right',role='count')
    text('Black, red and gold form one flag motif; the colours are not separate data series.',110,1884,23,role='motif_note')
    line([(110,1926),(1490,1926)],'#607984',1)
    text('Series: '+series['label'],110,1950,20,role='series_label')
    text('Synthetic study data · not official German statistics',1490,1950,20,align='right',role='provenance')

    # Save inspectable geometry and the mask that was actually used in compositing.
    outdir = ROOT/'artifacts'
    outdir.mkdir(exist_ok=True)
    area_mask.resize((W,H),Image.Resampling.LANCZOS).save(outdir/'area_mask.png')
    geometry={
        'canvas':[W,H], 'supersampling':S, 'plot_bounds':[x0,ytop,x1,y0],
        'mapping':{'x':'linear time in calendar years','y':'linear GDP level, zero baseline',
                   'time_min':float(times[0]),'time_max':float(times[-1]),'value_min':vmin,'value_max':vmax},
        'series_id':series['id'], 'series_label':series['label'], 'unit':series['unit'],
        'points':[{'category':c,'time':float(t),'value':float(v),'x':p[0],'y':p[1]} for c,t,v,p in zip(categories,times,values,points)],
        'closed_polygon_xy':polygon, 'key_labels':label_specs,
        'flag_bands':[{'name':name,'top_px':top,'bottom_px':bottom,'role':'one non-quantitative flag pattern'} for name,top,bottom in bands],
        'text_boxes':text_boxes,
    }
    (outdir/'data_geometry.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    im.resize((W,H),Image.Resampling.LANCZOS).save(ROOT/'final.png')
    print(json.dumps({'canvas':[W,H],'quarterly_points':len(points),'series':1,'baseline':y0,'y_range':[vmin,vmax],'output':'final.png'}))


if __name__=='__main__':
    main()
