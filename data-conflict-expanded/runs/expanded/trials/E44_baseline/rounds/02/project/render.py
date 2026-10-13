#!/usr/bin/env python3
"""Offline compositor: generated garden materials + data-bound string-net geometry.
Run from this project with the supplied Python interpreter. No model/network calls.
"""
from __future__ import annotations
import json, math
from pathlib import Path
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

ROOT = Path(__file__).resolve().parent
S = 2

def read_json(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8'))

inputs=read_json('input.json')
brief=read_json(inputs['brief'])
data=read_json(inputs['data'])
core=read_json(inputs['critical_metaphor_contract'])
assert (ROOT/inputs['reference']).is_file()  # Kept as design-source dependency; no pixels copied.
W,H=brief['width'],brief['height']
series=data['series'][0]
assert len(data['series'])==1 and series['unit']=='kg'
assert data['value_axis']=={'minimum':0,'maximum':90,'scale':'linear'}
assert len(data['categories'])==len(series['values'])==6
assert sum(series['values'])==data['series_totals']['yield']
parts=read_json('assets/processed/parts.json')

INK='#233b31'
MUTED='#5b6b5e'
PAPER='#f4f1e8'
WHITE='#fffbee'
COLORS=['#8fe2de','#63c7f2','#589eec','#b6a1ed','#e797bd','#f18b77']
BASELINE=1044.0
AXIS_TOP=324.0
SCALE=(BASELINE-AXIS_TOP)/(data['value_axis']['maximum']-data['value_axis']['minimum'])
NET_W=152.0
NET_LEFTS=[410.0+255*i for i in range(6)]

canvas=Image.new('RGBA',(W*S,H*S),PAPER)

def rgba(hexstr,alpha=255):
    h=hexstr.lstrip('#')
    return tuple(int(h[i:i+2],16) for i in (0,2,4))+(alpha,)

def coords(box): return tuple(int(round(v*S)) for v in box)

@lru_cache(None)
def font(size,bold=False):
    return ImageFont.truetype(str(ROOT/'assets/fonts'/('bold.otf' if bold else 'sans.otf')),round(size*S))

def text(x,y,value,size=24,bold=False,fill=INK,anchor='lt',image=None):
    im=canvas if image is None else image
    ImageDraw.Draw(im).text((round(x*S),round(y*S)),str(value),font=font(size,bold),fill=fill,anchor=anchor)

def line(points,fill,width=1,image=None):
    ImageDraw.Draw(canvas if image is None else image).line([(round(x*S),round(y*S)) for x,y in points],fill=fill,width=max(1,round(width*S)))

def rect(box,fill,radius=0,outline=None,width=1,image=None):
    d=ImageDraw.Draw(canvas if image is None else image)
    if radius:
        d.rounded_rectangle(coords(box),round(radius*S),fill=fill,outline=outline,width=round(width*S))
    else: d.rectangle(coords(box),fill=fill,outline=outline,width=round(width*S))

def ellipse(box,fill,image=None):
    ImageDraw.Draw(canvas if image is None else image).ellipse(coords(box),fill=fill)

def contact_shadow(layer,offset=(2.5,4),blur=2.4,alpha=135):
    a=layer.getchannel('A').filter(ImageFilter.GaussianBlur(blur*S))
    a=a.point(lambda p: round(p*alpha/255))
    shadow=Image.new('RGBA',layer.size,(21,17,10,0))
    shadow.putalpha(a)
    canvas.alpha_composite(shadow,(round(offset[0]*S),round(offset[1]*S)))

# Calm editorial title and an explicit synthetic-data notice.
text(80,42,'GARDEN TRIALS  /  HARVEST STUDY',18,True,MUTED)
text(78,80,data['title'],75,True,INK)
text(81,172,f"{series['label']} ({series['unit']}) across six garden trial plots.",25,False,MUTED)
rect((1700,43,2020,88),INK,radius=4)
text(1860,66,data['visible_notice'],21,True,WHITE,anchor='mm')
text(2020,124,f"{sum(series['values'])} kg",49,True,INK,anchor='rt')
text(2020,184,'TOTAL HARVEST  /  6 PLOTS',17,True,MUTED,anchor='rt')
line([(80,228),(2020,228)],'#c6cbbf',1)

# AI-created soil and surrounding grass fill the actual quantitative scene.
photo=Image.open(ROOT/'assets/raw_01.png').convert('RGB')
photo=photo.resize((1980*S,886*S),Image.Resampling.LANCZOS)
photo=ImageEnhance.Brightness(photo).enhance(.85)
canvas.paste(photo,(60*S,254*S))

# Quiet survey strings carry a single shared, linear scale.
guides=Image.new('RGBA',canvas.size)
for value in range(0,91,15):
    yy=BASELINE-value*SCALE
    if value==0:
        line([(339,yy),(1880,yy)],rgba('#f3ead1',185),1.7,guides)
    else:
        for xx in range(339,1880,24):
            line([(xx,yy),(min(xx+10,1880),yy)],rgba('#e7dfc7',65),.9,guides)
    line([(329,yy),(345,yy)],rgba('#f6efdf',205),1.6,guides)
    text(313,yy,str(value),22,True,rgba('#fff9e8'),anchor='rm',image=guides)
line([(339,AXIS_TOP),(339,BASELINE)],rgba('#f1e8d2',100),1,guides)
canvas.alpha_composite(guides)
text(303,294,'kg',20,True,WHITE,anchor='rm')

# Load isolated overhead material assets. Anchors record contact points in source pixels.
asset_cache={}
def stamp(name, width, point, anchor=None, target=None, rotation=0):
    if name not in asset_cache:
        asset_cache[name]=Image.open(ROOT/parts[name]['path']).convert('RGBA')
    src=asset_cache[name]
    factor=width/src.width
    h=src.height*factor
    im=src.resize((round(width*S),round(h*S)),Image.Resampling.LANCZOS)
    if rotation:
        im=im.rotate(rotation,resample=Image.Resampling.BICUBIC,expand=True)
    if anchor is None:
        ox,oy=0,0
    elif anchor=='source':
        ox,oy=parts[name]['anchor']
        ox*=factor; oy*=factor
    else: ox,oy=anchor
    xy=(round((point[0]-ox)*S),round((point[1]-oy)*S))
    (canvas if target is None else target).alpha_composite(im,xy)
    return [xy[0]/S,xy[1]/S,im.width/S,im.height/S]

# Tool props remain off the measured net rectangles.
props=Image.new('RGBA',canvas.size)
stamp('wheelbarrow',159,(106,390),target=props,rotation=-17)
stamp('tools',171,(113,740),target=props,rotation=78)
contact_shadow(props,offset=(3,5),blur=2.5,alpha=145)
canvas.alpha_composite(props)

# Generated twine is colorized and mapped along every string; code fixes all endpoints.
rope_src=Image.open(ROOT/parts['twine']['path']).convert('RGBA').crop((22,1,485,60))

@lru_cache(None)
def rope_tile(color,thickness):
    new_h=max(2,round(thickness*S))
    new_w=round(rope_src.width/rope_src.height*new_h)
    tile=rope_src.resize((new_w,new_h),Image.Resampling.LANCZOS)
    arr=np.asarray(tile).astype(float)
    lum=arr[:,:,:3].mean(axis=2)
    # Material brightness, photographed fibers, and actual alpha are all retained.
    mod=np.clip(lum/205.0,.45,1.12)
    c=np.array(rgba(color)[:3])
    arr[:,:,:3]=np.clip(mod[:,:,None]*c,0,255)
    return Image.fromarray(arr.astype('uint8'))

def rope_segment(target,p0,p1,color,thickness=2.15):
    dx=(p1[0]-p0[0])*S; dy=(p1[1]-p0[1])*S
    length=math.hypot(dx,dy)
    if length<.1: return
    tile=rope_tile(color,thickness)
    strip=Image.new('RGBA',(max(1,round(length)),tile.height))
    for xx in range(0,strip.width,tile.width):strip.alpha_composite(tile,(xx,0))
    angle=math.degrees(math.atan2(dy,dx))
    strip=strip.rotate(-angle,resample=Image.Resampling.BICUBIC,expand=True)
    mx=(p0[0]+p1[0])*S/2; my=(p0[1]+p1[1])*S/2
    target.alpha_composite(strip,(round(mx-strip.width/2),round(my-strip.height/2)))

stakes=Image.new('RGBA',canvas.size)
mesh=Image.new('RGBA',canvas.size)
geometry=[]
for index,(category,value,x0,color) in enumerate(zip(data['categories'],series['values'],NET_LEFTS,COLORS)):
    x1=x0+NET_W
    y1=BASELINE
    y0=BASELINE-float(value)*SCALE
    # Six equal-size pegs per rectangle. Peg count and width carry no data.
    peg_pts=[(x0,y0),(x1,y0),(x0,(y0+y1)/2),(x1,(y0+y1)/2),(x0,y1),(x1,y1)]
    for xx,yy in peg_pts:stamp('wood_peg',22,(xx,yy),anchor='source',target=stakes)
    # Local clipped mesh, with constant pitch and constant width for every plot.
    pad=8
    local=Image.new('RGBA',(round((NET_W+2*pad)*S),round((y1-y0+2*pad)*S)))
    local_h=y1-y0
    pitch=28.0
    for slope in (-.82,.82):
        low=min(0,-slope*NET_W)-local_h
        high=max(local_h,local_h-slope*NET_W)+local_h
        for intercept in np.arange(math.floor(low/pitch)*pitch,high+pitch,pitch):
            pts=[]
            for xx in [0.0,NET_W]:
                yy=slope*xx+intercept
                if -.01<=yy<=local_h+.01:pts.append((xx,yy))
            for yy in [0.0,local_h]:
                xx=(yy-intercept)/slope
                if -.01<=xx<=NET_W+.01:pts.append((xx,yy))
            unique=[]
            for p in pts:
                if not any(abs(p[0]-q[0])+abs(p[1]-q[1])<.1 for q in unique):unique.append(p)
            if len(unique)>=2:
                rope_segment(local,(unique[0][0]+pad,unique[0][1]+pad),(unique[1][0]+pad,unique[1][1]+pad),color,2.15)
    # Clip diagonal strand texture to the quantitative rectangle; no fake minimum length.
    clip=Image.new('L',local.size)
    ImageDraw.Draw(clip).rectangle(coords((pad,pad,NET_W+pad,local_h+pad)),fill=255)
    al=np.array(local.getchannel('A')).astype('uint16')
    local.putalpha(Image.fromarray((al*np.asarray(clip)//255).astype('uint8')))
    mesh.alpha_composite(local,(round((x0-pad)*S),round((y0-pad)*S)))
    for p0,p1 in [((x0,y0),(x1,y0)),((x0,y1),(x1,y1)),((x0,y0),(x0,y1)),((x1,y0),(x1,y1))]:
        rope_segment(mesh,p0,p1,color,3.8)
    # Tiny real contact wraps on the peg caps.
    for xx,yy in peg_pts:
        ellipse((xx-3.5,yy-2.6,xx+3.5,yy+2.6),rgba(color),mesh)
        line([(xx-3,yy-1),(xx+3,yy+.8)],rgba('#fff6dd',175),.8,mesh)
    geometry.append({'category':category,'value':value,'unit':series['unit'],'color':color,'left_x':x0,'right_x':x1,'top_y':y0,'baseline_y':y1,'length_px':y1-y0,'width_px':NET_W,'peg_count':len(peg_pts),'mesh_pitch_px':pitch})
contact_shadow(stakes,offset=(3,4.5),blur=2,alpha=160)
canvas.alpha_composite(stakes)
contact_shadow(mesh,offset=(1.7,2.5),blur=1.1,alpha=190)
canvas.alpha_composite(mesh)

# Generated gardeners physically overlap the side strings and the wooden anchor pegs.
workers=Image.new('RGBA',canvas.size)
worker_boxes=[]
# The right glove touches Plot C's upper-left wooden peg.
worker_boxes.append(stamp('gardener_yellow',172,(NET_LEFTS[2]+1,geometry[2]['top_y']+2),anchor='source',target=workers))
# The left glove covers the middle side peg of Plot F, leaving both data endpoints clear.
worker_boxes.append(stamp('gardener_blue',170,(NET_LEFTS[5]+NET_W-1,(geometry[5]['top_y']+BASELINE)/2+1),anchor='source',target=workers))
contact_shadow(workers,offset=(3.5,6),blur=3,alpha=148)
canvas.alpha_composite(workers)

# Precisely bound numeric plaques and category planting tags, all rendered in code.
label_boxes=[]
for g in geometry:
    cx=(g['left_x']+g['right_x'])/2
    top=g['top_y']
    color=g['color']
    box=(cx-55,top-65,cx+55,top-19)
    line([(cx,top-18),(cx,top-5)],rgba('#ede5d0',215),1.3)
    rect(tuple(v+(3 if i%2 else 2) for i,v in enumerate(box)),(24,20,13,100),radius=4)
    rect(box,rgba('#fffae9'),radius=4)
    text(cx-7,top-41.5,str(g['value']),31,True,INK,anchor='mm')
    text(cx+28,top-37.5,g['unit'],17,False,MUTED,anchor='mm')
    label_boxes.append({'category':g['category'],'value_box':list(box)})
    # An off-white garden identification tag below the common 0 kg baseline.
    tag=(cx-76,1077,cx+76,1118)
    rect((tag[0]+2,tag[1]+3,tag[2]+2,tag[3]+3),(25,20,12,110),radius=2)
    rect(tag,rgba('#f7f1df'),radius=2)
    ellipse((cx-65,1094,cx-59,1100),'#756d53')
    text(cx+5,1099,g['category'],26,True,INK,anchor='mm')

# Footer gives the visual decoding rule without changing any data semantics.
line([(80,1171),(2020,1171)],'#c6cbbf',1)
text(80,1194,'READ THE NETS',18,True,INK)
text(292,1191,'Net length from the shared baseline = harvest yield (kg).',24,False,INK)
text(292,1227,'Linear scale 0–90 kg.  Net width, string density and peg count do not encode yield.',20,False,MUTED)
text(2020,1247,'ILLUSTRATIVE VALUES',16,True,MUTED,anchor='rb')

out=canvas.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
out.save(ROOT/'final.png',compress_level=7)
report={'canvas':[W,H],'metric':series['label'],'unit':series['unit'],'scale_px_per_kg':SCALE,'axis_minimum':0,'axis_maximum':90,'axis_top_y':AXIS_TOP,'baseline_y':BASELINE,'geometry':geometry,'label_boxes':label_boxes,'worker_boxes':worker_boxes,'frozen_dependencies':inputs,'reference_pixels_copied':False,'generated_materials':['soil and grass','wood pegs','twine mapped to all measured nets','two contacting gardeners','wheelbarrow, rake and spade']}
(ROOT/'artifacts').mkdir(exist_ok=True)
(ROOT/'artifacts/geometry.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(f'Rendered {W} x {H}; scale {SCALE:g} px/kg; values {series["values"]}.')
