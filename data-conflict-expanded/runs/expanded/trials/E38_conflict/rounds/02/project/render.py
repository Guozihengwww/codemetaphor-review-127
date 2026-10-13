#!/usr/bin/env python3
"""Offline, data-bound photographic infographic. Run through capture_render.py."""
from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
S = 2
FONT_DIR = ROOT / 'assets/fonts'
IVORY = (246, 244, 234)
MUTED = (187, 200, 198)
YELLOW = (250, 218, 58)
GREEN = (107, 242, 169)
DARK = (14, 21, 23)


def load_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def main():
    inputs = load_json('input.json')
    brief = load_json(inputs['brief'])
    data = load_json(inputs['data'])
    core = load_json(inputs['critical_metaphor_contract'])
    assert (ROOT / inputs['reference']).is_file()
    assert core['visible_title'] == data['title']
    w, h = brief['width'], brief['height']
    assert (w, h) == (2000, 2000)
    series = data['series'][0]
    assert len(data['series']) == 1
    categories = data['categories']
    values = series['values']
    times = [date.fromisoformat(c).toordinal() for c in categories]
    assert times == data['time_points']
    assert len(categories) == len(values) == 60
    assert all(a < b for a, b in zip(times, times[1:]))
    vmin, vmax = data['value_axis']['minimum'], data['value_axis']['maximum']
    assert (vmin, vmax, data['value_axis']['scale']) == (-6, 30, 'linear')
    assert series['unit'] == 'percentage_points'

    # One mapping is shared by the path, markers, axes, leaders and evidence file.
    left, right, top, bottom = 182.0, 1818.0, 462.0, 1060.0
    def px(t):
        return left + (t-times[0])/(times[-1]-times[0])*(right-left)
    def py(v):
        return bottom - (v-vmin)/(vmax-vmin)*(bottom-top)
    points = [(px(t), py(v)) for t, v in zip(times, values)]
    zero_y = py(0)
    crossings = []
    for i, (a, b) in enumerate(zip(values, values[1:])):
        if a*b < 0:
            u = -a/(b-a)
            t = times[i] + u*(times[i+1]-times[i])
            crossings.append({'before_index': i, 'after_index': i+1,
                              'fraction': u, 'ordinal': t, 'x': px(t), 'y': zero_y})

    im = Image.open(ROOT/'assets/street_scene.png').convert('RGBA').resize((w*S,h*S),Image.Resampling.LANCZOS)
    # Smooth photographic veils improve contrast without creating a separate chart panel.
    yy = np.arange(h*S, dtype=float)/S
    alpha = np.interp(yy, [0, 350, 700, 980, 1120, 1220, 1820, 1880, 2000],
                     [100, 90, 66, 48, 9, 0, 0, 105, 240]).astype(np.uint8)
    veil = Image.new('RGBA', im.size, DARK+(0,))
    veil.putalpha(Image.fromarray(np.broadcast_to(alpha[:,None], (h*S,w*S)).copy()))
    im = Image.alpha_composite(im, veil)
    draw = ImageDraw.Draw(im)
    font_cache = {}
    text_records = []

    def font(size, bold=False):
        key = (size,bold)
        if key not in font_cache:
            fn = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
            font_cache[key] = ImageFont.truetype(str(FONT_DIR/fn), round(size*S))
        return font_cache[key]

    def text(x, y, s, size=28, fill=IVORY, bold=False, anchor='lt', stroke=0):
        xy=(round(x*S),round(y*S))
        f=font(size,bold)
        draw.text(xy,s,font=f,fill=fill,anchor=anchor,stroke_width=round(stroke*S),stroke_fill=DARK)
        box=draw.textbbox(xy,s,font=f,anchor=anchor,stroke_width=round(stroke*S))
        box=[n/S for n in box]
        assert box[0]>=0 and box[1]>=0 and box[2]<=w and box[3]<=h, (s,box)
        text_records.append({'text':s,'bbox':box})

    def line(coords, fill, width=1):
        draw.line([(round(x*S),round(y*S)) for x,y in coords], fill=fill, width=max(1,round(width*S)), joint='curve')

    def ellipse(x,y,r,fill,outline=None,width=1):
        draw.ellipse((round((x-r)*S),round((y-r)*S),round((x+r)*S),round((y+r)*S)),
                     fill=fill,outline=outline,width=round(width*S))

    def signed(v):
        return ('+' if v>0 else '−' if v<0 else '')+f'{abs(v):.1f}'

    # Title and period summary; the topic and metric are not encoded in the photo.
    draw.rounded_rectangle((110*S,67*S,425*S,115*S),radius=4*S,fill=YELLOW)
    text(127,79,data['visible_notice'],24,DARK,True)
    text(1890,81,'01 MAY 2017 — 01 JUN 2020',25,MUTED,anchor='rt')
    text(108,159,'Net support for',76,IVORY)
    text(104,248,'community street art',100,IVORY,True)
    text(1890,169,'PERIOD CHANGE',22,MUTED,bold=True,anchor='rt')
    delta=values[-1]-values[0]
    text(1890,217,f'{signed(delta)} pp',65,IVORY,anchor='rt')
    text(1890,299,'first to last observation',23,MUTED,anchor='rt')
    line([(110,366),(1890,366)],(119,139,142),0.8)
    text(110,405,'Net support  /  percentage points',29,IVORY)
    line([(1340,422),(1378,422)],YELLOW,6)
    text(1393,407,'Positive',25,IVORY)
    line([(1596,422),(1634,422)],GREEN,6)
    text(1649,407,'Negative',25,IVORY)

    # Quiet, linear grid over the photographic facade.
    grid=Image.new('RGBA',im.size,(0,0,0,0))
    gd=ImageDraw.Draw(grid)
    tick_values=[30,20,10,0,-6]
    for v in tick_values:
        y=py(v)
        color=(224,235,230,130 if v==0 else 42)
        if v==0:
            # Dashed equality reference; not an additional data series.
            for x in np.arange(left,right,18):
                gd.line([(round(x*S),round(y*S)),(round(min(x+8,right)*S),round(y*S))],fill=color,width=2*S)
        else:
            gd.line([(round(left*S),round(y*S)),(round(right*S),round(y*S))],fill=color,width=S)
    year_ticks=[date(y,1,1).toordinal() for y in (2018,2019,2020)]
    for t in year_ticks:
        x=px(t)
        gd.line([(round(x*S),round(top*S)),(round(x*S),round(bottom*S))],fill=(224,235,230,32),width=S)
    im=Image.alpha_composite(im,grid)
    draw=ImageDraw.Draw(im)
    for v in tick_values:
        label=f'+{v}' if v>0 else f'−{abs(v)}' if v<0 else '0'
        text(154,py(v)-14,label,27,IVORY if v==0 else MUTED,anchor='rt')
    text(201,zero_y-38,'Support = opposition',22,MUTED)

    # Every input vertex is present. Color is clipped at the exact y=0 boundary.
    stroke_layer=Image.new('RGBA',im.size,(0,0,0,0))
    sd=ImageDraw.Draw(stroke_layer)
    highpoints=[(round(x*S),round(y*S)) for x,y in points]
    sd.line(highpoints, fill=(3,10,12,205),width=12*S,joint='curve')
    im=Image.alpha_composite(im,stroke_layer)
    mask=Image.new('L',im.size,0)
    md=ImageDraw.Draw(mask)
    md.line(highpoints, fill=255,width=6*S,joint='curve')
    for x,y in points:
        r=3.2
        md.ellipse((round((x-r)*S),round((y-r)*S),round((x+r)*S),round((y+r)*S)),fill=255)
    ink=Image.new('RGBA',im.size,YELLOW+(255,))
    inkdraw=ImageDraw.Draw(ink)
    inkdraw.rectangle((0,round(zero_y*S),w*S,h*S),fill=GREEN+(255,))
    ink.putalpha(mask)
    im=Image.alpha_composite(im,ink)
    draw=ImageDraw.Draw(im)

    # Endpoint and first-negative annotations, derived from the data.
    p0=points[0]
    ellipse(*p0,6.8,YELLOW)
    line([p0,(244,p0[1])],IVORY,1.1)
    text(260,p0[1]-17,signed(values[0]),43,YELLOW,True)
    text(263,p0[1]+35,date.fromisoformat(categories[0]).strftime('%d %b %Y'),23,MUTED)

    first_negative=next(i for i,v in enumerate(values) if v<0)
    xf,yf=points[first_negative]
    line([(xf,yf-9),(xf,784),(xf-18,784)],(182,207,200),1.0)
    ellipse(xf,yf,5.8,GREEN)
    text(1245,702,'First negative reading',28,IVORY,True)
    negative_date=date.fromisoformat(categories[first_negative]).strftime('%d %b %Y')
    text(1245,744,f'{negative_date} · {signed(values[first_negative])} pp',25,GREEN)

    pe=points[-1]
    ellipse(*pe,6.8,GREEN)
    line([(pe[0]+9,pe[1]),(1866,pe[1]),(1866,925),(1850,925)],IVORY,1.1)
    text(1840,872,signed(values[-1]),43,GREEN,True,anchor='rt')
    text(1840,925,date.fromisoformat(categories[-1]).strftime('%d %b %Y'),23,MUTED,anchor='rt')

    # Calendar ticks use real ordinal distances, never equally spaced categories.
    for t in [times[0],*year_ticks,times[-1]]:
        x=px(t)
        line([(x,bottom+4),(x,bottom+14)],MUTED,1)
    text(left,1087,'01 MAY',21,MUTED,anchor='mt')
    text(left,1117,'2017',29,IVORY,bold=True,anchor='mt')
    for t in year_ticks:
        text(px(t),1090,str(date.fromordinal(t).year),30,IVORY,bold=True,anchor='mt')
    text(right,1087,'01 JUN',21,MUTED,anchor='mt')
    text(right,1117,'2020',29,IVORY,bold=True,anchor='mt')

    # Footer sits below the foreground brushes and explains the metric explicitly.
    text(1000,1901,'Net support = support (%) − opposition (%)',29,IVORY,anchor='mt',stroke=0.5)
    text(1000,1951,f'Community street-art plan  •  {len(values)} synthetic observations  •  Illustrative data',22,MUTED,anchor='mt',stroke=0.4)

    final=im.convert('RGB').resize((w,h),Image.Resampling.LANCZOS)
    final.save(ROOT/'final.png',compress_level=6)
    evidence={
        'canvas':[w,h],
        'plot':{'left':left,'right':right,'top':top,'bottom':bottom,'min':vmin,'max':vmax,'zero_y':zero_y,
                'pixels_per_day':(right-left)/(times[-1]-times[0]),'pixels_per_percentage_point':(bottom-top)/(vmax-vmin)},
        'series':series['id'],'unit':series['unit'],'line_width_px':6,
        'positive_rgb':YELLOW,'negative_rgb':GREEN,
        'vertices':[{'index':i,'date':d,'ordinal':t,'value':v,'x':p[0],'y':p[1]}
                    for i,(d,t,v,p) in enumerate(zip(categories,times,values,points))],
        'zero_crossings':crossings,'first_negative_index':first_negative,
        'period_change':round(delta,10),'rendered_text':text_records,
        'photo_asset':'assets/street_scene.png','all_vertices_in_input_order':True,
    }
    (ROOT/'artifacts').mkdir(exist_ok=True)
    (ROOT/'artifacts/geometry.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'image':'final.png','size':[w,h],'data_vertices':len(points),'zero_crossings':len(crossings)}))


if __name__=='__main__':
    main()
