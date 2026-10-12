#!/usr/bin/env python3
"""Regenerate the infographic locally from frozen data and generated face parts.
No model calls or network access. Run through capture_render.py for an experiment.
"""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT/'data.json').read_text(encoding='utf-8'))
BRIEF = json.loads((ROOT/'brief.json').read_text(encoding='utf-8'))
W,H = BRIEF['width'], BRIEF['height']
S = 2
CX,CY,R = 900.0,1120.0,560.0
INK = '#2c2927'
RUST = '#a94330'
PAPER = '#f5f0e5'
PALETTE = ['#f1d3c2','#cf9a73','#963f36','#dca292',
           '#75484c','#b26350','#e9bb96','#e6bfb4']
FONT_BOLD = '/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf'
FONT_REG = '/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf'
FONT_CJK = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'

values = DATA['series'][0]['values']
categories = DATA['categories']
assert len(values)==len(categories)==len(PALETTE)
assert abs(sum(values)-100)<1e-9
assert DATA['value_axis']=={'minimum':0,'maximum':100,'scale':'linear'}
angles=[0.0]
for v in values:
    angles.append(angles[-1]+360*float(v)/100)

im=Image.new('RGB',(W*S,H*S),PAPER)
draw=ImageDraw.Draw(im)

def pt(x,y): return (round(x*S),round(y*S))
def radial(a,r):
    rad=math.radians(a)
    return (CX+r*math.sin(rad),CY-r*math.cos(rad))
def line(points,fill=INK,width=2):
    draw.line([pt(*p) for p in points],fill=fill,width=round(width*S),joint='curve')
def font(size,bold=False,cjk=False):
    return ImageFont.truetype(FONT_CJK if cjk else FONT_BOLD if bold else FONT_REG,round(size*S))
text_boxes=[]
def text(x,y,t,size,fill=INK,bold=False,cjk=False,anchor='lt',role=None):
    f=font(size,bold,cjk)
    draw.text(pt(x,y),t,font=f,fill=fill,anchor=anchor)
    if role:
        b=draw.textbbox(pt(x,y),t,font=f,anchor=anchor)
        text_boxes.append({'role':role,'text':t,'bbox':[round(q/S,2) for q in b]})

def circle(x,y,r,fill,outline=None,width=1):
    draw.ellipse([pt(x-r,y-r),pt(x+r,y+r)],fill=fill,outline=outline,width=round(width*S))

def asset(name,x,y,w=None,h=None):
    a=Image.open(ROOT/'assets'/name).convert('RGBA')
    if w is None and h is None: w,h=a.size
    elif w is None: w=a.width*h/a.height
    elif h is None: h=a.height*w/a.width
    a=a.resize((round(w*S),round(h*S)),Image.Resampling.LANCZOS)
    im.paste(a,pt(x,y),a)
    return {'path':'assets/'+name,'bbox':[x,y,x+w,y+h]}

# Background and border: quiet printed-page framing.
draw.rectangle([pt(44,44),pt(W-44,H-44)],outline='#797067',width=2*S)
text(102,89,DATA['visible_notice'],31,RUST,bold=True,role='synthetic_notice')
text(1694,91,f'{len(values)} GROUPS  /  {sum(values):g}%',27,'#655e57',anchor='rt',role='total')
text(98,154,'Synthetic',104,bold=True,role='title_latin')
text(581,145,'人脸扇区',99,bold=True,cjk=True,role='title_chinese')
text(102,288,DATA['series'][0]['label'],37,role='series')
text(1694,290,'Unit: '+DATA['series'][0]['unit'],31,anchor='rt',role='unit')
line([(102,352),(1694,352)],'#81766a',1.5)

# A true circle is the quantitative denominator. Skin regions carry every share.
for i,(a0,a1) in enumerate(zip(angles[:-1],angles[1:])):
    n=max(3,math.ceil((a1-a0)/0.075))
    poly=[(CX,CY)]+[radial(a,R) for a in np.linspace(a0,a1,n+1)]+[(CX,CY)]
    draw.polygon([pt(*p) for p in poly],fill=PALETTE[i])
# Restrained deterministic print texture, clipped to the disk, never changing geometry.
bbox=(round((CX-R-2)*S),round((CY-R-2)*S),round((CX+R+2)*S),round((CY+R+2)*S))
a=np.array(im.crop(bbox)).astype(np.int16)
yy,xx=np.mgrid[:a.shape[0],:a.shape[1]]
rr=((xx+bbox[0])/S-CX)**2+((yy+bbox[1])/S-CY)**2
rng=np.random.default_rng(3108)
noise=rng.normal(0,0.9,(a.shape[0]//2+1,a.shape[1]//2+1))
noise=np.repeat(np.repeat(noise,2,axis=0),2,axis=1)[:a.shape[0],:a.shape[1]]
noise=np.where(rr<R*R,noise,0)
a=np.clip(a+noise[:,:,None],0,255).astype('uint8')
im.paste(Image.fromarray(a),(bbox[0],bbox[1]))
draw=ImageDraw.Draw(im)
circle(CX,CY,R,None,INK,3)

# Generated raster illustration is physically integrated into the data-bearing face.
placements=[]
placements.append(asset('ear.png',286,1026,w=135,h=246))
placements.append(asset('hair_fitted.png',CX-800,CY-800,w=1600,h=1600))
placements.append(asset('eyes.png',559,787,w=682,h=209.5))
# Nose-tip source registration is (88,299) within the source crop.
placements.append(asset('nose.png',CX-88*.92,CY-299*.92,w=175*.92,h=318*.92))
placements.append(asset('lips.png',CX-154*1.1,1350,w=308*1.1,h=135*1.1))
draw=ImageDraw.Draw(im)

# Boundaries are drawn over overprinting features, preserving traceable small wedges.
for angle in angles[:-1]:
    line([(CX,CY),radial(angle,R)],INK,2.6)
circle(CX,CY,4,INK)

# Outside labels in the input's clockwise order. Every leader begins within its wedge.
# Each coordinate refers to layout only; anchors and data angles are computed above.
slots=[
    {'x':1483,'y':559,'elbow_r':713,'end':(1467,632)},
    {'x':1530,'y':1194,'elbow_r':610,'end':(1515,1290.2)},
    {'x':1499,'y':1379,'elbow_r':660,'end':(1483,1464.8)},
    {'x':1450,'y':1558,'elbow_r':688,'end':(1434,1643.2)},
    {'x':1176,'y':1755,'elbow_r':666,'via':(1135,1830),'end':(1160,1830)},
    {'x':881,'y':1780,'elbow_r':618,'end':(880.6,1756),'center':True},
    {'x':350,'y':1719,'elbow_r':660,'end':(452,1698)},
    {'x':96,'y':724,'elbow_r':678,'end':(289,821)}
]
records=[]
for i,(cat,val,slot) in enumerate(zip(categories,values,slots)):
    mid=(angles[i]+angles[i+1])/2
    anchor=radial(mid,R*.88 if i in (0,7) else R-18)
    elbow=radial(mid,slot['elbow_r'])
    path=[anchor,elbow]
    if 'via' in slot:path.append(slot['via'])
    path.append(slot['end'])
    # A narrow paper-colored halo keeps the two leaders legible across hair.
    if i in (0,7):line(path,PAPER,6)
    line(path,INK,2.3)
    circle(*anchor,3.7,PAPER,INK,1.7)
    anch='mt' if slot.get('center') else 'lt'
    text(slot['x'],slot['y'],cat,37,bold=True,anchor=anch,role=f'category_{i}')
    text(slot['x']-1,slot['y']+47,f'{val:g}%',69,RUST,bold=True,anchor=anch,role=f'value_{i}')
    records.append({'category':cat,'value':val,'color':PALETTE[i],
                    'start_degrees':angles[i],'end_degrees':angles[i+1],
                    'midpoint_degrees':mid,'leader':path})

# Compact reading note makes the denominator and orientation explicit.
line([(102,1902),(1694,1902)],'#81766a',1.5)
text(102,1932,'READ THE PORTRAIT',23,'#71665d',bold=True)
text(443,1923,'12 点起顺时针 · 扇角 = 360° × 份额 / 100',25,cjk=True,role='encoding_note')
text(443,1959,'鼻部为圆心；头发与耳朵不计入比例。',22,'#71665d',cjk=True,role='denominator_note')

im=im.resize((W,H),Image.Resampling.LANCZOS)
im.save(ROOT/'final.png',compress_level=9)
(ROOT/'artifacts').mkdir(exist_ok=True)
geometry={'canvas':[W,H],'center':[CX,CY],'radius':R,'start':'12 oclock','direction':'clockwise',
          'angle_mapping':'degrees = 360 * value / 100','sectors':records,
          'text_boxes':text_boxes,'generated_asset_placements':placements,
          'layer_order':['paper','data-colored face disk','generated ear and hair','generated facial features','data rays','labels']}
(ROOT/'artifacts'/'render_geometry.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Rendered final.png:',W,H,'; angles:',angles)
