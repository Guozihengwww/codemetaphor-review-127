#!/usr/bin/env python3
"""Reproduce final.png locally. Invoke complete renders through capture_render.py."""
from pathlib import Path
import json, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parent
DATA=json.loads((ROOT/'data.json').read_text(encoding='utf-8'))
BRIEF=json.loads((ROOT/'brief.json').read_text(encoding='utf-8'))
W,H=BRIEF['width'],BRIEF['height']
S=2
CX,CY,R=845.0,1065.0,500.0
BG='#F5F1E9'
INK='#272624'
MUTED='#67635E'
RUST='#B13F29'
COLORS=['#EFCFB7','#CBD8D7','#F1D8A3','#C6CBD6','#E8B3A1','#ABC8C2','#F2D5B8','#C9BCD0','#E7BD91','#B4C8CB','#E6D3C4','#D3C09F','#BFBCB3','#EAC7BD']
FONTS={
 'regular':'/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
 'bold':'/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
 'cjk':'/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
 'cjkbold':'/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
}
values=DATA['series'][0]['values']
categories=DATA['categories']
assert len(values)==len(categories)==14
assert abs(sum(values)-100)<1e-9
angles=np.r_[0,np.cumsum(np.asarray(values))*3.6]
canvas=Image.new('RGB',(W*S,H*S),BG)
d=ImageDraw.Draw(canvas)
texts=[]

def p(x,y):return (round(x*S),round(y*S))
def box(a):return tuple(round(z*S) for z in a)
def font(size,style='regular'):return ImageFont.truetype(FONTS[style],round(size*S))
def text_at(x,y,txt,size,fill=INK,style='regular'):
    f=font(size,style)
    # Coordinates denote the visible ink's top-left, rather than font ascent.
    bb=d.textbbox((0,0),txt,font=f)
    loc=(round(x*S)-bb[0],round(y*S)-bb[1])
    d.text(loc,txt,font=f,fill=fill)
    rect=[x,y,x+(bb[2]-bb[0])/S,y+(bb[3]-bb[1])/S]
    texts.append({'text':txt,'bbox':rect})
    return rect

def line(points,fill=INK,width=2):
    d.line([p(x,y) for x,y in points],fill=fill,width=max(1,round(width*S)),joint='curve')
def xy(angle,r):
    t=math.radians(angle)
    return CX+r*math.sin(t),CY-r*math.cos(t)

def leader(points):
    # A restrained light underprint keeps lines trackable across the hairstyles.
    line(points,BG,5.1)
    line(points,INK,1.8)

# Editorial frame and hierarchy.
d.rectangle(box((43,43,W-43,H-43)),outline=INK,width=2*S)
text_at(100,88,DATA['visible_notice'],25,RUST,'bold')
text_at(100,144,DATA['title'],82,INK,'cjkbold')
text_at(103,257,DATA['series'][0]['label']+' (%)',34,INK)
line([(100,324),(1700,324)],INK,1.7)
text_at(1513,94,'14 GROUPS',20,MUTED,'bold')
text_at(1513,129,'TOTAL 100%',20,MUTED,'bold')

# Each sector's angular width comes from the input share; the disk has no
# minimum-slice exaggeration, sorting, explosion or irregular denominator.
for i,(a,b) in enumerate(zip(angles[:-1],angles[1:])):
    steps=max(2,int(math.ceil((b-a)/.12)))
    pts=[p(CX,CY)]+[p(*xy(float(t),R)) for t in np.linspace(a,b,steps+1)]
    d.polygon(pts,fill=COLORS[i])

# The prepared raster contains generated face details, different hairstyles,
# and the left ear. Its origin and circular scale use the same chart mapping.
asset=Image.open(ROOT/'assets/portrait_circle.png').convert('RGBA')
scale=R/560
nw=round(asset.width*scale*S)
nh=round(asset.height*scale*S)
asset=asset.resize((nw,nh),Image.Resampling.LANCZOS)
left=round((CX-800*scale)*S)
top=round((CY-800*scale)*S)
canvas.paste(asset,(left,top),asset)
d=ImageDraw.Draw(canvas)

# Exact round data boundary stays distinct from hair/ear silhouette.
d.ellipse(box((CX-R,CY-R,CX+R,CY+R)),outline=INK,width=3*S)
# Draw all quantitative rays over the facial layer, so even the narrowest
# sector remains traceable through the nose and lips.
for a in angles[:-1]:
    line([(CX,CY),xy(float(a),R)],'#F6EDE4',3.9)
    line([(CX,CY),xy(float(a),R)],INK,1.65)
# Small origin marker belongs to the nose, not to an extra data category.
d.ellipse(box((CX-3.4,CY-3.4,CX+3.4,CY+3.4)),fill=INK)

# Label anchors are the angular midpoints of their own sector. Positions only
# affect annotation routing, never data geometry. Counterclockwise label
# relocation is not used to reorder the categories.
layouts=[
 (1210,380,(1187,448)),
 (1470,536,(1447,600)),
 (1535,735,(1512,798)),
 (1535,943,(1512,1006)),
 (1535,1184,(1512,1247)),
 (1468,1363,(1445,1426)),
 (1336,1520,(1313,1583)),
 (1196,1655,(1270,1634)),
 (986,1757,(1058,1736)),
 (743,1812,(820,1791)),
 (487,1757,(564,1736)),
 (276,1620,(353,1599)),
 (105,1428,(285,1491)),
 (105,751,(292,825)),
]
geometry=[]
for i,(a,b) in enumerate(zip(angles[:-1],angles[1:])):
    mid=float((a+b)/2)
    anchor=xy(mid,R+1)
    elbow=xy(mid,R+43)
    tx,ty,end=layouts[i]
    leader([anchor,elbow,end])
    d.ellipse(box((anchor[0]-3.3,anchor[1]-3.3,anchor[0]+3.3,anchor[1]+3.3)),fill=INK)
    text_at(tx,ty,categories[i],29,INK,'bold')
    value_label=str(values[i])+'%'
    fs=53 if i!=13 else 59
    text_at(tx,ty+44,value_label,fs,RUST,'bold')
    geometry.append({'category':categories[i],'value':values[i],'label':value_label,
      'start_angle_clockwise_from_12':float(a),'end_angle_clockwise_from_12':float(b),
      'mid_angle':mid,'fill':COLORS[i], 'anchor':list(anchor),'leader':[list(anchor),list(elbow),list(end)],
      'label_position':[tx,ty]})

# Compact reading key is explicit about the denominator and clockwise order.
line([(100,1930),(1700,1930)],INK,1.3)
text_at(101,1951,'从 12 点顺时针 · 扇角 = 份额 × 360° · 仅圆盘计量，发型和耳朵不计入',23,MUTED,'cjk')

final=canvas.resize((W,H),Image.Resampling.LANCZOS)
final.save(ROOT/'final.png')
(ROOT/'artifacts').mkdir(exist_ok=True)
(ROOT/'artifacts/geometry.json').write_text(json.dumps({
 'canvas':[W,H], 'disk':{'center':[CX,CY],'radius':R,'direction':'clockwise','zero':'12 oclock'},
 'angles_degrees':angles.tolist(),'sectors':geometry,'text_bounds':texts,
 'generated_asset_affine':{'scale':scale,'left':left/S,'top':top/S},
},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Rendered final.png',W,H,'categories:',len(categories),'sum:',sum(values))
