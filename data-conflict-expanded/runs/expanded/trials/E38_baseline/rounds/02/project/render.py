#!/usr/bin/env python3
"""Offline photographic infographic. Execute complete renders via capture_render.py."""
from __future__ import annotations
from pathlib import Path
from datetime import date
import json
import math
import hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
S = 2
GREEN = '#79EEAB'
YELLOW = '#FFDD35'
WHITE = '#FFFDF4'
MUTED = '#CDCFC8'
INK = '#101B19'
FONT = ROOT / 'assets/fonts/Sans-Regular.otf'
BOLD = ROOT / 'assets/fonts/Sans-Bold.otf'
LABEL = ROOT / 'assets/fonts/Labels.ttf'

def read_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))

def hex_rgb(h):
    return tuple(int(h[i:i+2],16) for i in (1,3,5))

def main():
    # Keep the frozen topic and input contract local and available to this renderer.
    inputs = read_json('input.json')
    brief = read_json(inputs['brief'])
    data = read_json(inputs['data'])
    core = read_json(inputs['critical_metaphor_contract'])
    assert core['visible_title'] == data['title']
    assert brief['width'] == brief['height'] == 2000
    assert data['synthetic'] and data['visible_notice'] == 'SYNTHETIC DATA'
    assert (ROOT / inputs['reference']).is_file()  # Visual reference is never sampled.
    W, H = brief['width'], brief['height']
    series = data['series'][0]
    assert len(data['series']) == 1 and series['id'] == 'net_support'
    assert series['unit'] == 'percentage_points'
    dates = [date.fromisoformat(v) for v in data['categories']]
    t = np.array([d.toordinal() for d in dates], dtype=float)
    vals = np.array(series['values'],dtype=float)
    assert len(t) == len(vals) == len(data['time_points'])
    assert np.array_equal(t,np.array(data['time_points']))
    assert all(t[i+1] > t[i] for i in range(len(t)-1))
    axis = data['value_axis']
    assert axis == {'minimum':-6,'maximum':30,'scale':'linear'}
    lo, hi = axis['minimum'],axis['maximum']
    X0, X1, Y0, Y1 = 200.0, 1790.0, 280.0, 846.0
    def xx(v): return X0 + (v-t[0])/(t[-1]-t[0])*(X1-X0)
    def yy(v): return Y1 - (v-lo)/(hi-lo)*(Y1-Y0)
    pts = [(xx(a),yy(b)) for a,b in zip(t,vals)]

    photo = Image.open(ROOT/'assets/raw_02.png').convert('RGB').resize((W*S,H*S),Image.Resampling.LANCZOS)
    # Local, deterministic darkening protects data and type while preserving the scene.
    arr = np.asarray(photo,dtype=np.float32)
    y = np.arange(H*S,dtype=np.float32)/S
    opacity = np.interp(y,[0,240,520,750,880,1000,1500,1570,1650,1810,2000],
                          [.58,.43,.32,.19,.09,.01,0,.11,.47,.78,.89])
    tint = np.array([9,18,17],dtype=np.float32)
    arr = arr*(1-opacity[:,None,None])+tint[None,None,:]*opacity[:,None,None]
    im = Image.fromarray(np.clip(arr,0,255).astype('uint8'),'RGB').convert('RGBA')
    overlay = Image.new('RGBA',im.size,(0,0,0,0))
    dr = ImageDraw.Draw(overlay)
    boxes=[]
    def font(size,bold=False,label=False):
        return ImageFont.truetype(str(LABEL if label else BOLD if bold else FONT),round(size*S))
    def text(s,x,y,size=28,fill=WHITE,bold=False,anchor='lt',label=False,stroke=0,tag=None):
        f=font(size,bold,label)
        p=(round(x*S),round(y*S))
        dr.text(p,s,font=f,fill=fill,anchor=anchor,stroke_width=round(stroke*S),stroke_fill=(12,22,20,235))
        bb=dr.textbbox(p,s,font=f,anchor=anchor,stroke_width=round(stroke*S))
        boxes.append({'text':s,'box':[v/S for v in bb],'role':tag or 'label'})
    def line(points,fill,width=1):
        dr.line([(round(x*S),round(y*S)) for x,y in points],fill=fill,width=round(width*S),joint='curve')
    def dot(x,y,r,fill,outline=None,ow=1):
        dr.ellipse(((x-r)*S,(y-r)*S,(x+r)*S,(y+r)*S),fill=fill,outline=outline,width=round(ow*S))
    def dash(a,b,fill,width=1,d=10,g=9):
        length=math.dist(a,b)
        if not length:return
        for start in np.arange(0,length,d+g):
            end=min(start+d,length)
            line([(a[0]+(b[0]-a[0])*start/length,a[1]+(b[1]-a[1])*start/length),
                  (a[0]+(b[0]-a[0])*end/length,a[1]+(b[1]-a[1])*end/length)],fill,width)

    # Editorial header and semantic legend.
    dr.rounded_rectangle((108*S,91*S,440*S,145*S),radius=5*S,fill=YELLOW)
    text(data['visible_notice'],274,118,28,INK,True,anchor='mm',tag='synthetic_notice')
    text(f'MAY 2017 — JUNE 2020  /  {len(vals)} OBSERVATIONS',1890,113,24,MUTED,anchor='rt',label=True)
    text(series['label'],110,182,36,WHITE,True,tag='metric')
    text('·  percentage points',348,191,25,MUTED,tag='unit')
    line([(1075,202),(1121,202)],GREEN,8)
    text('More opposition',1138,184,28,WHITE)
    line([(1485,202),(1531,202)],YELLOW,8)
    text('More support',1548,184,28,WHITE)

    # Axes share one linear y scale; zero is the only emphasized rule.
    for v in [30,20,10,0,-6]:
        gy=yy(v)
        if v==0:
            dash((X0,gy),(1840,gy),(239,246,233,138),1.5,9,9)
        else:
            line([(X0,gy),(1840,gy)],(242,246,235,44),1)
        labelstr='0' if v==0 else ('−6' if v==-6 else f'+{v}')
        text(labelstr,155,gy,28,WHITE if v==0 else MUTED,anchor='rm',label=True,stroke=.6,tag='value_tick')
    text('ZERO',1850,yy(0),18,MUTED,anchor='lm',label=True,stroke=.8)

    # Every observation is joined linearly. A mixed-sign segment is split at
    # its interpolated zero in DATA space before applying the same transform.
    segments=[]
    crossings=[]
    for i in range(len(vals)-1):
        a,b=float(vals[i]),float(vals[i+1])
        if (a<0<b) or (b<0<a):
            f=-a/(b-a)
            ct=t[i]+f*(t[i+1]-t[i])
            cp=(xx(ct),yy(0))
            col1=GREEN if a<0 else YELLOW
            col2=GREEN if b<0 else YELLOW
            segments.extend([(pts[i],cp,col1,i),(cp,pts[i+1],col2,i)])
            crossings.append({'between':[data['categories'][i],data['categories'][i+1]],
                              'fraction':f,'ordinal':ct,'x':cp[0],'y':cp[1]})
        else:
            segments.append((pts[i],pts[i+1],GREEN if (a+b)/2<0 else YELLOW,i))
    # A narrow neutral halo separates the actual colored stroke from the photo.
    line(pts,(7,22,19,175),15)
    for a,b,c,i in segments:
        line([a,b],c,10)
    for (px,py),v in zip(pts,vals):
        dot(px,py,5,GREEN if v<0 else YELLOW)

    # Labels: selected values are tied to the actual observation by markers/leaders.
    def marker(index):
        px,py=pts[index]
        dot(px,py,8, GREEN if vals[index]<0 else YELLOW,INK,2)
    def value_label(v):
        return ('+' if v>0 else '−' if v<0 else '') + f'{abs(float(v)):.1f}'
    marker(0)
    text(value_label(vals[0]),200,859,35,GREEN,True,anchor='mt',label=True,stroke=1.3,tag='value_label')
    low=int(np.argmin(vals));marker(low)
    lx,ly=pts[low]
    line([(lx,ly+10),(lx+18,871)],(121,238,171,185),1.3)
    text(value_label(vals[low]),lx+29,864,30,GREEN,True,label=True,stroke=1.2,tag='value_label')

    # The zero crossing is a mathematically interpolated location, not a sample.
    for cr in crossings:
        cx,cy=cr['x'],cr['y']
        line([(cx,cy-10),(cx,704),(cx-18,686)],(245,247,231,150),1.5)
        text('Crosses zero',cx-30,641,26,WHITE,True,anchor='rt',stroke=1,tag='crossing_label')
        a,b=[date.fromisoformat(v) for v in cr['between']]
        crossing_caption=f'Between {a.day} {a:%b} & {b.day} {b:%b %Y}'
        text(crossing_caption,cx-30,682,21,MUTED,anchor='rt',stroke=1)
        # No extra colored dot is added: the exact green/yellow transition stays visible.
    for ds in ['2019-01-01','2020-01-01']:
        i=data['categories'].index(ds);marker(i)
        px,py=pts[i]
        line([(px,py-10),(px,py-33)],(255,221,53,170),1.3)
        text(value_label(vals[i]),px,py-73,35,YELLOW,True,anchor='mt',label=True,stroke=1,tag='value_label')

    marker(len(vals)-1)
    ex,ey=pts[-1]
    line([(ex-13,ey),(1720,ey)],(255,221,53,185),1.5)
    text(value_label(vals[-1]),1700,259,58,YELLOW,True,anchor='rt',label=True,stroke=1,tag='value_label')
    text(f'{dates[-1].day} {dates[-1]:%b %Y}'.upper(),1700,330,23,WHITE,anchor='rt',label=True,stroke=1)

    # True-date tick locations; no equal-spacing substitution.
    ticks=[('2017-05-01','MAY 2017'),('2018-01-01','JAN 2018'),('2019-01-01','JAN 2019'),
           ('2020-01-01','JAN 2020'),('2020-06-01','JUN 2020')]
    for ds,lab in ticks:
        tx=xx(date.fromisoformat(ds).toordinal())
        line([(tx,Y1+4),(tx,Y1+16)],(231,239,224,150),1.3)
        text(lab,tx,922,25,WHITE,anchor='mt',label=True,stroke=1.2,tag='date_tick')
    # Discrete dates are represented by low-contrast rug ticks on the bottom boundary.
    for px,_ in pts:
        line([(px,Y1),(px,Y1+5)],(223,232,216,75),1)

    # Main subject title, matching the frozen topic and metric.
    text('Net support for',108,1664,87,WHITE,True,tag='title')
    text('community street art',103,1761,112,WHITE,True,tag='title')
    line([(110,1900),(1890,1900)],(241,242,230,81),1)
    text('Net support = share in favour (%) − share opposed (%).',110,1924,29,WHITE,tag='definition')
    text(f'Synthetic series · {len(vals)} dated observations · Not a real-world survey',110,1969,23,MUTED,tag='provenance')
    text('pp',1888,1928,27,MUTED,True,anchor='rt',tag='unit')

    im=Image.alpha_composite(im,overlay).convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
    im.save(ROOT/'final.png',format='PNG',compress_level=6)
    out=ROOT/'artifacts'
    out.mkdir(exist_ok=True)
    metadata={
      'canvas':[W,H], 'plot':{'left':X0,'right':X1,'top':Y0,'bottom':Y1,'minimum':lo,'maximum':hi},
      'x_mapping':'linear Gregorian ordinal dates','y_mapping':'linear percentage points',
      'stroke_width_px':10,'point_joint_radius_px':5,
      'negative_color':GREEN,'positive_color':YELLOW,'zero_crossings':crossings,
      'observations':[{'date':ds,'value':float(v),'ordinal':int(ordinal),'x':float(px),'y':float(py)}
                      for ds,v,ordinal,(px,py) in zip(data['categories'],vals,t,pts)],
      'colored_segments':[{'from':list(a),'to':list(b),'color':c,'source_interval':i} for a,b,c,i in segments],
      'labels':boxes,
      'frozen_input_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                             for name in ['input.json','brief.json','data.json','core.json','reference.png']}
    }
    (out/'geometry.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Rendered {W}x{H}; {len(pts)} observations; {len(segments)} sign-specific segments.')

if __name__=='__main__':
    main()
