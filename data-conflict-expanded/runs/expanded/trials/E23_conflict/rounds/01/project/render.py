#!/usr/bin/env python3
"""Offline, data-bound poster renderer. Run from the case directory.
Every complete experimental render must be invoked through capture_render.py.
"""
from pathlib import Path
import json
import math
import hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent
S = 2

def read_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))

def write_json(name, obj):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def font(size, face='regular'):
    names = {'regular':'Text-Regular.otf','bold':'Text-Bold.otf','display':'Display-Bold.otf'}
    return ImageFont.truetype(str(ROOT / 'assets' / 'fonts' / names[face]), round(size * S))

def main():
    inputs = read_json('input.json')
    brief = read_json(inputs['brief'])
    data = read_json(inputs['data'])
    core = read_json(inputs['critical_metaphor_contract'])
    assert (ROOT / inputs['reference']).is_file()
    assert data['synthetic'] is True and data['value_axis']['scale'] == 'linear'
    assert len(data['series']) == 1
    W, H = brief['width'], brief['height']
    assert (W, H) == (1600, 2000)
    cats = data['categories']
    times = data['time_points']
    series = data['series'][0]
    values = series['values']
    lo, hi = data['value_axis']['minimum'], data['value_axis']['maximum']
    assert len(cats) == len(values) == len(times)
    assert all(lo <= v <= hi for v in values)
    assert all(a < b for a,b in zip(times, times[1:]))
    assert sum(values) == data['series_totals'][series['id']]

    # One shared mapping for all annual values; no smoothing, data-size floor,
    # local rescaling, clipping, logarithm, or nonlinear picture deformation.
    left, right, top, baseline = 166.0, 1470.0, 770.0, 1740.0
    def x(t): return left + (t-times[0])/(times[-1]-times[0]) * (right-left)
    def y(v): return baseline - (v-lo)/(hi-lo) * (baseline-top)
    points = [(x(t), y(v)) for t,v in zip(times,values)]
    ink = (49, 32, 115, 255)
    paper = (255, 251, 225, 255)
    white = (255, 254, 246, 255)
    grid = (74, 45, 82, 61)

    # Deterministic, code-native paper field; never uses reference pixels.
    yy, xx = np.mgrid[0:H, 0:W]
    vertical = yy / H
    warm_top = np.array([255., 104., 19.])
    warm_bottom = np.array([255., 174., 58.])
    bg = warm_top[None,None,:] * (1-vertical[:,:,None]) + warm_bottom[None,None,:] * vertical[:,:,None]
    glow = np.exp(-(((xx-180)/1150)**2 + ((yy-1360)/1100)**2))
    bg[:,:,1] += 11 * glow
    bg[:,:,2] += 9 * glow
    noise = np.random.default_rng(23).normal(0, 0.44, (H,W,1))
    bg = np.uint8(np.clip(bg + noise, 0, 255))
    base = Image.fromarray(bg, 'RGB').convert('RGBA').resize((W*S,H*S), Image.Resampling.BICUBIC)

    # Large subject crop preserves the complete ball, hands, arms, and jersey,
    # matching the upper-body framing of the design reference. Its alpha is
    # preserved. The quantitative line will be laid directly over the jersey.
    athlete = Image.open(ROOT / 'assets/athlete_duotone.png').convert('RGBA')
    alpha = np.asarray(athlete.getchannel('A')).copy().astype(float)
    src_rows = np.arange(athlete.height)[:,None]
    # Fade only below the jersey hem, keeping the footer clear.
    fade = 1 - np.clip((src_rows-735)/110,0,1)
    alpha *= fade
    athlete.putalpha(Image.fromarray(np.uint8(np.round(alpha))))
    asset_scale = 2.0
    asset_xy = (100, 250)
    athlete = athlete.resize((round(athlete.width*asset_scale*S),round(athlete.height*asset_scale*S)),Image.Resampling.LANCZOS)
    base.alpha_composite(athlete,(asset_xy[0]*S,asset_xy[1]*S))

    layer = Image.new('RGBA',base.size,(0,0,0,0))
    d = ImageDraw.Draw(layer)
    text_records = []
    def line(coords, fill, width=1):
        d.line([(round(a*S),round(b*S)) for a,b in coords],fill=fill,width=round(width*S),joint='curve')
    def circle(cx,cy,r,fill,outline=None,width=1):
        d.ellipse(((cx-r)*S,(cy-r)*S,(cx+r)*S,(cy+r)*S),fill=fill,outline=outline,width=round(width*S))
    def text(txt,px,py,size,face='regular',fill=ink,anchor='lt',stroke=0,stroke_fill=None,role=None):
        f=font(size,face)
        d.text((round(px*S),round(py*S)),txt,font=f,fill=fill,anchor=anchor,
               stroke_width=round(stroke*S),stroke_fill=stroke_fill)
        bounds=d.textbbox((round(px*S),round(py*S)),txt,font=f,anchor=anchor,stroke_width=round(stroke*S))
        text_records.append({'text':txt,'role':role or 'editorial','bbox_px':[round(v/S,2) for v in bounds]})

    # Editorial hierarchy. All mandatory content is drawn with local fonts.
    d.rounded_rectangle((96*S,75*S,398*S,123*S),radius=4*S,fill=ink)
    text(data['visible_notice'],247,86,25,'bold',white,anchor='mt',role='synthetic_notice')
    text(f'{cats[0]}—{cats[-1]}',1504,86,26,'bold',ink,anchor='rt',role='time_span')
    text("WOMEN'S",90,174,160,'display',white,role='topic_title')
    text('BASKETBALL',90,326,160,'display',white,role='topic_title')
    text('VIEWERS',90,478,160,'display',ink,role='topic_title')
    text(series['label'],98,635,34,'bold',ink,role='series_label')
    text(series['unit'].capitalize(),98,685,27,'regular',ink,role='unit')

    # Same linear value axis from 0 through 1,500, displayed without a break.
    ticks = list(range(int(lo),int(hi)+1,300))
    for tick in ticks:
        gy=y(tick)
        line([(left,gy),(right,gy)],ink if tick==lo else grid,2.0 if tick==lo else 1.35)
        text(f'{tick:,}',139,gy,26,'regular',ink,anchor='rm',role='value_axis_tick')
    for px,cat in zip([p[0] for p in points],cats):
        line([(px,baseline),(px,baseline+16)],ink,2)
        text(cat,px,baseline+37,26,'regular',ink,anchor='mt',role='year')
    text('YEAR',98,baseline+91,20,'bold',ink,role='category_axis_title')

    # Subtle edge shadow follows exactly the same data vertices.
    shadow = Image.new('RGBA',base.size,(0,0,0,0))
    sd=ImageDraw.Draw(shadow)
    scaled=[(round(px*S),round(py*S)) for px,py in points]
    sd.line(scaled,fill=(119,43,40,145),width=16*S,joint='curve')
    for px,py in scaled:
        sd.ellipse((px-8*S,py-8*S,px+8*S,py+8*S),fill=(119,43,40,145))
    shadow=shadow.filter(ImageFilter.GaussianBlur(2*S))
    base=Image.alpha_composite(base,layer)
    base=Image.alpha_composite(base,shadow)
    layer=Image.new('RGBA',base.size,(0,0,0,0))
    d=ImageDraw.Draw(layer)
    line(points,white,9)
    for px,py in points:
        circle(px,py,4.5,white)
    for i,(px,py) in enumerate(points):
        r=13 if i==0 else (10 if i==len(points)-1 else 7.5)
        circle(px,py,r,ink,white,2.5)

    # Every exact value is visible; numeric positions are computed from the data.
    # Alternating label offsets avoid collisions without moving the marks.
    for i,(cat,val,(px,py)) in enumerate(zip(cats,values,points)):
        if i==0:
            text(f'{val:,}',px+47,py-43,77,'display',white,role='value:'+cat)
            text(f'{cat}  /  FIRST YEAR',px+48,py-75,21,'bold',ink,role='endpoint_annotation')
        elif i==len(points)-1:
            text(str(val),px,py-66,45,'display',ink,anchor='mt',stroke=1,stroke_fill=(255,181,65,255),role='value:'+cat)
        else:
            offset=-46 if i%2 else 24
            text(str(val),px,py+offset,28,'bold',ink,anchor='mt',stroke=1.2,stroke_fill=(255,182,66,255),role='value:'+cat)

    line([(98,1858),(1504,1858)],(69,42,104,108),1.2)
    text('SYNTHETIC DATA',98,1890,22,'bold',ink,role='footer_notice')
    text('Illustrative values, not observed audience statistics.',345,1890,24,'regular',ink,role='provenance_note')
    text(f'Source: supplied synthetic series  ·  {len(values)} annual observations  ·  Linear scale',98,1938,23,'regular',ink,role='source')

    base=Image.alpha_composite(base,layer).convert('RGB')
    final=base.resize((W,H),Image.Resampling.LANCZOS)
    final.save(ROOT/'final.png',compress_level=6)
    write_json('artifacts/geometry.json',{
        'canvas_px':[W,H], 'mapping':{'left':left,'right':right,'top':top,'baseline':baseline,
        'value_min':lo,'value_max':hi,'type':'linear','time_min':times[0],'time_max':times[-1]},
        'series_id':series['id'],'unit':series['unit'],
        'points':[{'category':c,'time':t,'value':v,'x':px,'y':py} for c,t,v,(px,py) in zip(cats,times,values,points)],
        'asset_transform':{'path':'assets/athlete_duotone.png','x':asset_xy[0],'y':asset_xy[1],
            'scale_x':asset_scale,'scale_y':asset_scale,'source_fade_below_y':735,'source_fade_end_y':845,
            'framing':'rear-view athlete with complete ball, arms and jersey; lower body outside poster crop'},
        'labels':text_records,
        'frozen_input_hashes':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['input.json','brief.json','data.json','core.json','reference.png']}
    })
    print(f'Wrote final.png ({W} × {H}); {len(points)} data vertices; one shared {lo}–{hi} linear axis.')

if __name__=='__main__':
    main()
