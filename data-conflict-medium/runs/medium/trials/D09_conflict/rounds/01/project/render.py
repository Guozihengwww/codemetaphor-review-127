#!/usr/bin/env python3
"""Rebuild the frozen synthetic flag infographic from local data and textile.
Run complete renders only via capture_render.py as required by this experiment.
"""
from pathlib import Path
from functools import lru_cache
import csv
import hashlib
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance

ROOT = Path(__file__).resolve().parent
S = 2
W, H = 1600, 2000
CREAM = '#f4eee0'
MUTED = '#a7bcc9'
CORAL = '#fa7564'
NAVY = '#102e48'
FONT_DIR = ROOT / 'assets/fonts'
TEXT_BOXES = []

@lru_cache(None)
def font(size, weight='regular'):
    name = {'regular':'Sans-Regular.otf', 'bold':'Sans-Bold.otf', 'narrow':'Narrow-Bold.otf'}[weight]
    return ImageFont.truetype(str(FONT_DIR/name), round(size*S))

def coord(p):
    return tuple(round(v*S) for v in p)

def line(draw, points, fill, width=1):
    draw.line([coord(p) for p in points], fill=fill, width=round(width*S), joint='curve')

def text(draw, x, y, value, size, color=CREAM, weight='regular', anchor='lt', tracking=None, record=True):
    if tracking is None:
        box = draw.textbbox(coord((x,y)), value, font=font(size,weight), anchor=anchor)
        draw.text(coord((x,y)), value, font=font(size,weight), fill=color, anchor=anchor)
    else:
        widths = [draw.textlength(ch,font=font(size,weight))/S for ch in value]
        total = sum(widths)+max(0,len(value)-1)*tracking
        if anchor.startswith('r'): x -= total
        elif anchor.startswith('m'): x -= total/2
        boxes=[]
        for ch,wd in zip(value,widths):
            boxes.append(draw.textbbox(coord((x,y)),ch,font=font(size,weight),anchor='lt'))
            draw.text(coord((x,y)),ch,font=font(size,weight),fill=color,anchor='lt')
            x += wd+tracking
        box=(min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
    if record:
        TEXT_BOXES.append({'text':value,'bbox_px':[round(a/S,2) for a in box]})
    return tuple(a/S for a in box)

def dashed(draw, points, fill, width=3, pattern=(9,6)):
    """Dash one continuous polyline using arc length, retaining each vertex."""
    k=0
    remaining=float(pattern[0])
    for a,b in zip(points,points[1:]):
        ax,ay=a; bx,by=b
        length=math.hypot(bx-ax,by-ay)
        if length==0: continue
        pos=0.0
        while pos < length-1e-9:
            span=min(remaining,length-pos)
            p=(ax+(bx-ax)*pos/length, ay+(by-ay)*pos/length)
            q=(ax+(bx-ax)*(pos+span)/length, ay+(by-ay)*(pos+span)/length)
            if k%2==0: line(draw,[p,q],fill,width)
            pos+=span; remaining-=span
            if remaining<1e-9:
                k=(k+1)%len(pattern); remaining=float(pattern[k])

def dot(draw, x,y, r=5, fill=CREAM, edge=CORAL, width=2):
    draw.ellipse(coord((x-r,y-r,x+r,y+r)), fill=fill, outline=edge, width=round(width*S))

def background():
    rng=np.random.default_rng(19002035)
    coarse=Image.fromarray(rng.integers(0,256,(70,56),dtype=np.uint8)).resize((W,H),Image.Resampling.BICUBIC)
    c=(np.asarray(coarse,dtype=np.float32)-127.5)/64
    yy,xx=np.mgrid[:H,:W]
    vignette=1.7*((xx-W*.52)/(W*.65))**2+1.4*((yy-H*.46)/(H*.7))**2
    weave=0.6*np.sin(xx*np.pi)+0.8*np.cos(yy*np.pi)
    fine=rng.normal(0,0.75,(H,W))
    noise=c*1.4+weave+fine-vignette
    arr=np.clip(np.array([15.,43.,69.])[None,None,:]+noise[:,:,None],0,255).astype(np.uint8)
    return Image.fromarray(arr).resize((W*S,H*S),Image.Resampling.BICUBIC).convert('RGBA')

def main():
    brief=json.loads((ROOT/'brief.json').read_text(encoding='utf-8'))
    data=json.loads((ROOT/'data.json').read_text(encoding='utf-8'))
    assert (brief['width'],brief['height'])==(W,H)
    years=np.array(data['time_points'],dtype=float)
    values=np.array(data['series'][0]['values'],dtype=float)
    assert len(years)==len(values)==len(data['categories'])==136
    assert all(str(int(y))==cat for y,cat in zip(years,data['categories']))
    assert np.all(np.diff(years)>0)
    assert data['value_axis']=={'minimum':0,'maximum':130,'scale':'linear'}
    x0,x1,y0,y1=150.,1480.,550.,1665.
    x=lambda year: x0+(float(year)-years[0])/(years[-1]-years[0])*(x1-x0)
    y=lambda value: y1-float(value)/130*(y1-y0)
    points=[(x(a),y(b)) for a,b in zip(years,values)]
    hist_index=int(np.where(years==2024)[0][0])
    forecast_index=int(np.where(years==2025)[0][0])
    xp=x(2025)
    get=lambda yr: float(values[int(np.where(years==yr)[0][0])])
    percent=lambda value:f'{float(value):.1f}%'
    im=background()
    d=ImageDraw.Draw(im)

    # Editorial header; all letters are set by code.
    line(d,[(110,86),(160,86)],CORAL,5)
    text(d,179,73,'A SYNTHETIC TIME SERIES',23,weight='bold',tracking=2.0)
    d.rounded_rectangle(coord((1189,61,1490,113)),radius=5*S,outline='#7591a6',width=1*S)
    text(d,1339.5,76,data['visible_notice'],21,weight='bold',anchor='mt',tracking=1.6)
    text(d,104,168,'Debt / Output',160,weight='bold')
    text(d,111,342,'1900–2035',52,weight='regular')
    text(d,498,350,'Debt relative to synthetic economic output.',29,color=MUTED)
    line(d,[(110,427),(1490,427)],'#58748a',1)

    text(d,110,464,'DEBT / OUTPUT (%)',24,weight='bold',tracking=1.0)
    text(d,110,501,'Percent of synthetic output',23,color=MUTED)
    line(d,[(744,480),(809,480)],CORAL,3.8)
    text(d,824,468,'1900–2024  Historical',24,color=CREAM)
    dashed(d,[(744,521),(809,521)],CORAL,3.8,(9,6))
    text(d,824,509,'2025–2035  Projected',24,color=CREAM)

    # Projection band is independent of values and begins at the exact 2025 x coordinate.
    overlay=Image.new('RGBA',im.size,(0,0,0,0)); od=ImageDraw.Draw(overlay)
    od.rectangle(coord((xp,y0,x1,y1)),fill=(247,215,186,15))
    im=Image.alpha_composite(im,overlay); d=ImageDraw.Draw(im)
    text(d,1480,471,'PROJECTION',20,color='#e8baa6',weight='bold',anchor='rt',tracking=1.5)
    text(d,1480,508,'2025–2035',27,color=CREAM,anchor='rt')

    ticks=[0,20,40,60,80,100,120,130]
    for val in ticks:
        gy=y(val)
        line(d,[(x0,gy),(x1,gy)],'#34546e' if val not in (0,130) else '#527088',1)
        text(d,x0-23,gy,str(val),24,color=MUTED,anchor='rm')

    # One exact polygon controls the generated textile's visible area.
    polygon=points+[(x1,y1),(x0,y1)]
    mask=Image.new('L',im.size,0); md=ImageDraw.Draw(mask)
    md.polygon([coord(p) for p in polygon],fill=255)
    textile=Image.open(ROOT/'assets/flag_textile.png').convert('RGB')
    textile=ImageEnhance.Color(textile).enhance(0.91)
    textile=ImageEnhance.Brightness(textile).enhance(1.04)
    textile=textile.resize((round((x1-x0)*S)+1,round((y1-y0)*S)+1),Image.Resampling.LANCZOS)
    material=Image.new('RGB',im.size,(0,0,0)); material.paste(textile,coord((x0,y0)))
    im.paste(material,(0,0),mask)
    # Slightly mute the projected cloth without changing its mask or height.
    fade=Image.new('RGBA',im.size,(0,0,0,0)); fd=ImageDraw.Draw(fade)
    fd.rectangle(coord((xp,y0,x1,y1)),fill=(15,43,69,45))
    fa=np.array(fade)
    ma=np.asarray(mask)
    fa[:,:,3]=(fa[:,:,3].astype(np.uint16)*(ma>0)).astype(np.uint8)
    im=Image.alpha_composite(im,Image.fromarray(fa))
    d=ImageDraw.Draw(im)
    # Faint continuous reading rules also cross the woven body.
    grid=Image.new('RGBA',im.size,(0,0,0,0)); gd=ImageDraw.Draw(grid)
    for val in [20,40,60,80,100,120]:
        line(gd,[(x0,y(val)),(x1,y(val))],(118,155,180,35),0.7)
    im=Image.alpha_composite(im,grid); d=ImageDraw.Draw(im)
    dashed(d,[(xp,y0),(xp,y1)],'#a7a49c',1.1,(5,7))

    # No smoothing, resampling or aggregation of the supplied annual series.
    line(d,points[:hist_index+1],'#572f36',7)
    line(d,points[:hist_index+1],CORAL,3.7)
    dashed(d,points[hist_index:],CORAL,3.7,(8,5))
    line(d,[(x0,y1),(x1,y1)],CREAM,1.5)
    for year in years:
        height=15 if int(year)%10==0 else 9 if int(year)%5==0 else 5
        line(d,[(x(year),y1),(x(year),y1+height)],'#afc1ca',0.8)
    for yr in list(range(1900,2021,10))+[2035]:
        text(d,x(yr),y1+26,str(yr),22,color=CREAM,anchor='mt')

    # The void above the changing silhouette carries the explanatory copy.
    text(d,544,768,'One flag. One annual series.',34,weight='bold')
    text(d,544,820,'The upper edge traces debt / output.',27,color=MUTED)
    text(d,544,860,'The stripes are a single continuous fill.',27,color=MUTED)

    # Key labels are derived from data, and anchored to data coordinates.
    text(d,187,565,'1900',28,weight='bold')
    text(d,183,606,percent(get(1900)),61,weight='bold')
    line(d,[(174,658),points[0]],'#ddc5b2',1.2)
    dot(d,*points[0],r=5.4)

    p=(x(1989),y(get(1989)))
    text(d,1060,608,'1989',28,weight='bold')
    text(d,1056,650,percent(get(1989)),61,weight='bold')
    line(d,[(1069,720),p],'#ddc5b2',1.2)
    dot(d,*p,r=5.4)

    p=(x(1941),y(get(1941)))
    text(d,572,1128,'1941',26,weight='bold')
    text(d,569,1168,percent(get(1941)),49,weight='bold')
    line(d,[(582,1227),p],'#ddc5b2',1.2)
    dot(d,*p,r=4.6)

    p=(x(2024),y(get(2024)))
    text(d,1330,1480,'2024',25,weight='bold')
    text(d,1327,1518,percent(get(2024)),45,weight='bold')
    line(d,[(p[0],1574),(p[0],p[1]-8)],'#c4c8c2',1.2)
    dot(d,*p,r=4.8)

    p=(x(2035),y(get(2035)))
    text(d,1480,1327,'2035',27,weight='bold',anchor='rt')
    text(d,1483,1366,percent(get(2035)),56,weight='bold',anchor='rt')
    line(d,[(p[0],1433),(p[0],p[1]-9)],'#c4c8c2',1.2)
    dot(d,*p,r=5.4)

    # Below-chart facts do not rescale or exaggerate the low-valued observations.
    line(d,[(110,1756),(1490,1756)],'#58748a',1)
    min_i=int(np.argmin(values)); max_i=int(np.argmax(values))
    columns=[
        (110,'SERIES HIGH',percent(values[max_i]),f'{int(years[max_i])} · first observation'),
        (594,'SERIES LOW',percent(values[min_i]),f'{int(years[min_i])} · minimum'),
        (1047,'PROJECTION',f'{percent(get(2025))} → {percent(get(2035))}','2025 → 2035'),
    ]
    for col,title,value,caption in columns:
        text(d,col,1787,title,20,color=MUTED,weight='bold',tracking=1.8)
        text(d,col-2,1822,value,47,weight='bold')
        text(d,col,1881,caption,23,color=MUTED)
    line(d,[(544,1790),(544,1906)],'#385873',1)
    line(d,[(998,1790),(998,1906)],'#385873',1)
    text(d,110,1940,'Source: supplied synthetic data · 136 annual observations',23,color=MUTED)
    text(d,1490,1940,'Illustrative series; not actual U.S. statistics.',23,color=MUTED,anchor='rt')

    out=im.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
    out.save(ROOT/'final.png', optimize=True)
    out_mask=mask.resize((W,H),Image.Resampling.LANCZOS)
    out_mask.save(ROOT/'artifacts/area_mask.png')
    np.save(ROOT/'artifacts/area_mask_2x.npy',np.asarray(mask))
    rows=[]
    for i,(yr,val,pt) in enumerate(zip(years,values,points)):
        rows.append({'year':int(yr),'value':float(val),'phase':'historical' if yr<=2024 else 'projected','x_px':pt[0],'y_px':pt[1]})
    (ROOT/'artifacts/data_geometry.json').write_text(json.dumps({'plot_box_px':[x0,y0,x1,y1],'axis_range':[0,130],'projection_boundary_px':xp,'history_points':hist_index+1,'projection_points':len(years)-forecast_index,'all_vertices':rows,'text_boxes':TEXT_BOXES},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (ROOT/'artifacts/annual_values.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f); writer.writerow(['year','debt_output_percent','phase','x_px','y_px'])
        for row in rows: writer.writerow(list(row.values()))
    print(json.dumps({'canvas':[W,H],'annual_vertices':len(points),'history_points':hist_index+1,'projection_points':len(years)-forecast_index,'final_sha256':hashlib.sha256((ROOT/'final.png').read_bytes()).hexdigest()}))

if __name__=='__main__':
    main()
