#!/usr/bin/env python3
"""Offline rice-market infographic. Run only through capture_render.py for a complete render."""
from __future__ import annotations
from pathlib import Path
import json
import math
import hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent
SS = 2
BG = '#FAFAF6'
INK = '#253326'
MUTED = '#657063'


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def save_json(name, obj):
    p=ROOT/name; p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def font(size, bold=False, chinese=False):
    name='NotoSansCJK-Regular.ttc' if chinese else ('NimbusSans-Bold.otf' if bold else 'NimbusSans-Regular.otf')
    return ImageFont.truetype(str(ROOT/'assets/fonts'/name), round(size*SS))


def pt(p):
    return tuple(round(float(v)*SS) for v in p)


def main():
    inp=read_json('input.json')
    brief=read_json(inp['brief'])
    data=read_json(inp['data'])
    core=read_json(inp['critical_metaphor_contract'])
    # Keep every frozen dependency in the offline project. The reference supplies no image pixels.
    assert (ROOT/inp['reference']).is_file()
    assert data['topic_zh']==core['topic_zh']
    vals=np.asarray(data['series'][0]['values'],dtype=float)
    cats=data['categories']
    assert len(cats)==len(vals)==4 and np.all(vals>0)
    total=float(vals.sum())
    assert total == data['series_totals'][data['series'][0]['id']]
    assert data['series'][0]['unit']=='%'
    W,H=brief['width'],brief['height']
    canvas=Image.new('RGB',(W*SS,H*SS),BG)
    draw=ImageDraw.Draw(canvas)
    bboxes=[]

    def text(x,y,s,size=32,color=INK,bold=False,chinese=False,anchor='lt',tag=None):
        f=font(size,bold,chinese)
        xy=pt((x,y))
        draw.text(xy,s,font=f,fill=color,anchor=anchor,stroke_width=0)
        bb=draw.textbbox(xy,s,font=f,anchor=anchor)
        bboxes.append({'tag':tag or s,'text':s,'bbox':[round(v/SS,2) for v in bb]})
        return (bb[2]-bb[0])/SS

    def line(points,fill,width=2.2):
        draw.line([pt(p) for p in points],fill=fill,width=round(width*SS),joint='curve')

    def dot(x,y,color,r=5):
        draw.ellipse(pt((x-r,y-r,x+r,y+r)),fill=color)

    # Editorial header. All semantic and quantitative text is bound in code.
    line([(110,86),(1890,86)],INK,2)
    draw.rounded_rectangle(pt((110,120,467,179)),radius=7*SS,fill=INK)
    text(133,135,data['visible_notice'],30,'#FFFFFF',True,tag='synthetic_notice')
    text(1890,138,f'{len(cats)} producers  /  {total:g}% total',30,MUTED,anchor='rt',tag='summary')
    title=data['title'].removeprefix('SYNTHETIC ')
    text(105,228,title,150,INK,True,tag='title')
    text(113,390,data['topic_zh'],32,MUTED,chinese=True,tag='metric_zh')
    text(358,393,'/  Share of total market (%)',33,MUTED,tag='metric_en')

    # One linear angular mapping, clockwise from twelve o'clock.
    cx,cy,R=1010.,1110.,610.
    size=round(2*R*SS)
    source=Image.open(ROOT/'assets/raw_01.png').convert('RGB').resize((size,size),Image.Resampling.LANCZOS)
    spoon=Image.open(ROOT/'assets/spoon_mask.png').convert('L').resize((size,size),Image.Resampling.LANCZOS)
    raw=np.asarray(source,dtype=np.float32)
    sy,sx=np.mgrid[:size,:size].astype(np.float32)
    dx=(sx+.5)/SS-R; dy=(sy+.5)/SS-R
    radius=np.hypot(dx,dy)
    theta=(np.degrees(np.arctan2(dy,dx))+90.)%360.
    cumulative=np.concatenate(([0.],np.cumsum(vals)/total*360.))
    sector_ids=np.searchsorted(cumulative[1:],theta,side='right').clip(0,len(vals)-1)
    # The rice photograph occupies every data sector. Hue is applied without changing geometry.
    tints=np.array([[151,211,222],[207,173,221],[250,203,150],[189,215,161]],dtype=np.float32)
    colors=['#247B90','#85509A','#B0702F','#607D3D']
    lum=(raw[:,:,0]*.2126+raw[:,:,1]*.7152+raw[:,:,2]*.0722)/255.
    shade=np.clip(lum/.94,0,1.08)
    colored=np.clip(tints[sector_ids]*shade[:,:,None],0,255)
    material=Image.fromarray(colored.astype(np.uint8),'RGB')
    md=ImageDraw.Draw(material)
    local_center=(R*SS,R*SS)
    # These same boundary angles generate the material masks, seams, and leaders.
    for degrees in cumulative[:-1]-90.:
        a=math.radians(degrees)
        end=((R+2)*math.cos(a)*SS+R*SS,(R+2)*math.sin(a)*SS+R*SS)
        md.line([local_center,end],fill=BG,width=round(5*SS))
    # The original spoon is an occluding, non-quantitative part of the same photograph.
    material.paste(source,(0,0),spoon)
    circle=Image.fromarray(np.uint8(radius<=R)*255,'L')
    px,py=round((cx-R)*SS),round((cy-R)*SS)
    shadow_layer=Image.new('RGBA',canvas.size,(0,0,0,0))
    shadow_draw=ImageDraw.Draw(shadow_layer)
    shadow_draw.ellipse(pt((cx-R+2,cy-R+11,cx+R+2,cy+R+11)),fill=(32,38,23,33))
    shadow_layer=shadow_layer.filter(ImageFilter.GaussianBlur(13*SS))
    canvas=Image.alpha_composite(canvas.convert('RGBA'),shadow_layer).convert('RGB')
    canvas.paste(material,(px,py),circle)
    draw=ImageDraw.Draw(canvas)

    # Four category callouts; their first point always follows the data-derived angular midpoint.
    layouts=[
        {'label':(1590,606),'value':(1584,661),'route':[(1460,572),(1830,572),(1830,594)]},
        {'label':(1660,1005),'value':(1653,1061),'route':[(1634,1092)]},
        {'label':(1590,1476),'value':(1584,1531),'route':[(1530,1519),(1562,1519)]},
        {'label':(110,870),'value':(104,925),'route':[(329,1262),(221,1262),(221,1085)]},
    ]
    geometry=[]
    for i,(v,cat,layout,col) in enumerate(zip(vals,cats,layouts,colors)):
        a0=float(cumulative[i]-90)
        a1=float(cumulative[i+1]-90)
        mid=math.radians((a0+a1)/2)
        anchor=(cx+R*.905*math.cos(mid),cy+R*.905*math.sin(mid))
        points=[anchor]+layout['route']
        line(points,col,2.6)
        dot(*anchor,col,r=5)
        dot(*points[-1],col,r=4)
        lx,ly=layout['label']; vx,vy=layout['value']
        text(lx,ly,cat,34,INK,tag=f'category_{i+1}')
        number=f'{v:g}'
        nw=text(vx,vy,number,116,col,True,tag=f'value_{i+1}')
        text(vx+nw+7,vy+45,data['series'][0]['unit'],56,col,tag=f'unit_{i+1}')
        geometry.append({'category':cat,'value':float(v),'unit':'%','center':[cx,cy],
                         'radius_px':R,'start_deg_clockwise_from_top':float(cumulative[i]),
                         'end_deg_clockwise_from_top':float(cumulative[i+1]),
                         'angle_deg':float(cumulative[i+1]-cumulative[i]),
                         'midpoint_deg_from_x':(a0+a1)/2,
                         'leader_anchor':[round(x,4) for x in anchor],
                         'tint_rgb':tints[i].astype(int).tolist()})

    line([(110,1802),(1890,1802)],'#D6DCD1',1.5)
    text(110,1845,'Sector angle = market share',30,INK,tag='encoding_note')
    text(1890,1845,f'TOTAL  {total:g}%',30,INK,True,anchor='rt',tag='total')
    text(110,1900,'Rice grains and spoon are illustrative. Producers and values are synthetic.',27,MUTED,tag='illustration_note')
    canvas=canvas.resize((W,H),Image.Resampling.LANCZOS)
    canvas.save(ROOT/'final.png',format='PNG',optimize=True)
    save_json('artifacts/render_geometry.json',{
        'canvas':[W,H],'circle_center':[cx,cy],'circle_radius':R,
        'start_angle':'12 o’clock','direction':'clockwise','total':total,
        'sectors':geometry,'text_bboxes':bboxes,'spoon_role':'occluding decoration; not a quantitative area',
        'material_source':'assets/raw_01.png','spoon_mask':'assets/spoon_mask.png',
        'source_data_sha256':hashlib.sha256((ROOT/inp['data']).read_bytes()).hexdigest(),
        'reference_pixels_used':False
    })
    print(json.dumps({'output':'final.png','size':[W,H],'angles':[g['angle_deg'] for g in geometry]},ensure_ascii=False))


if __name__=='__main__':
    main()
