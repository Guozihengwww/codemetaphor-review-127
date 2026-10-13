#!/usr/bin/env python3
"""Offline renderer: generated banquet photography + code-controlled data sectors."""
from pathlib import Path
import json, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT=Path(__file__).resolve().parent

def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))

INPUT=read('input.json')
BRIEF=read(INPUT['brief']); DATA=read(INPUT['data']); CORE=read(INPUT['critical_metaphor_contract'])
assert (ROOT/INPUT['reference']).is_file()
assert DATA['synthetic'] and DATA['response_model'].startswith('one mutually exclusive')
W,H=BRIEF['width'],BRIEF['height']
assert (W,H)==(2000,1412)
VALUES=np.array(DATA['series'][0]['values'],dtype=np.float64)
CATS=DATA['categories']; TOTAL=int(VALUES.sum())
assert len(CATS)==6 and TOTAL==DATA['total_count']==DATA['series_totals']['preference']
PCTS=VALUES/TOTAL*100
ANGLES=VALUES/TOTAL*360
START=-130.0
BOUNDS=np.r_[START,START+np.cumsum(ANGLES)]
PALETTE=[(222,210,184),(128,48,61),(152,173,170),(39,61,84),(240,108,54),(195,151,65)]
CREAM='#f6efdc'; MUTED='#d2d7c8'; LINE=(233,230,204,196)
CX,CY,R=1060.,765.,430.
S=2
GEO=read('assets/source_geometry.json')
scx,scy=GEO['center']; rx,ry=GEO['radii']
sx,sy=R/rx,R/ry
source=Image.open(ROOT/'assets/raw_02.png').convert('RGB')
source_mask=Image.open(ROOT/'assets/foreground_mask.png').convert('L')
# Reflect only empty floor at the outer asset edges.
pad=96
padded=Image.fromarray(np.pad(np.array(source),((pad,pad),(pad,pad),(0,0)),mode='reflect'))

def photo_region(x,y,w,h,scale=1):
    return padded.transform((w*scale,h*scale),Image.Transform.AFFINE,
        (1/(sx*scale),0,scx+(x-CX)/sx+pad,
         0,1/(sy*scale),scy+(y-CY)/sy+pad),
        resample=Image.Resampling.BICUBIC)

base=photo_region(0,0,W,H)
a=np.asarray(base).astype(np.float32)
yy,xx=np.mgrid[:H,:W]
# Subtle edge grading on the carpet provides a quiet reading field.
rr=np.hypot(xx-CX,yy-CY)
top=np.clip((260-yy)/230,0,1)
left=np.clip((520-xx)/420,0,1)
right=np.clip((xx-1520)/460,0,1)
floor=(rr>R+16).astype(np.float32)
grade=1-floor*(.25*top+.12*left+.10*right)
a*=grade[:,:,None]
base=Image.fromarray(np.uint8(np.clip(a,0,255))).convert('RGBA')
canvas=base.resize((W*S,H*S),Image.Resampling.LANCZOS)

# All quantitative sectors use this one polar mapping. No minimum wedge size.
margin=11
x0=int(CX-R-margin);y0=int(CY-R-margin)
N=int(2*(R+margin))
photo=photo_region(x0,y0,N,N,S)
photo_a=np.asarray(photo).astype(np.float32)
Y,X=np.mgrid[:N*S,:N*S]
X=(X+.5)/S+x0-CX;Y=(Y+.5)/S+y0-CY
rad=np.hypot(X,Y)
deg=(np.degrees(np.arctan2(Y,X))-START)%360
index=np.searchsorted(np.cumsum(ANGLES),deg,side='right')
index=np.minimum(index,5)
shade=(photo_a[:,:,0]*.299+photo_a[:,:,1]*.587+photo_a[:,:,2]*.114)/219.
color=np.array(PALETTE,dtype=np.float32)[index]*shade[:,:,None]
# Rebuild a narrow non-data wood rim around the exact circular data surface.
ring_shade=.93+.11*np.cos(np.arctan2(Y,X)+2.0)
ring_rgb=np.array([136,91,49])[None,None,:]*ring_shade[:,:,None]
rim_high=np.exp(-((rad-R-1.6)/1.3)**2)*25
ring_rgb=np.clip(ring_rgb+rim_high[:,:,None],0,255)
inside=(rad<=R)
composed=np.where(inside[:,:,None],color,ring_rgb)
alpha=np.clip((R+7.5-rad)*S+.5,0,1)
# Photographed contact objects use the exact same affine transformation.
foreground=source_mask.transform((N*S,N*S),Image.Transform.AFFINE,
    (1/(sx*S),0,scx+(x0-CX)/sx,0,1/(sy*S),scy+(y0-CY)/sy),
    resample=Image.Resampling.BICUBIC)
fga=np.asarray(foreground).astype(np.float32)/255
composed=composed*(1-fga[:,:,None])+photo_a*fga[:,:,None]
rgba=np.dstack([np.clip(composed,0,255).astype(np.uint8),np.uint8(alpha*255)])
canvas.alpha_composite(Image.fromarray(rgba,'RGBA'),(x0*S,y0*S))
# Text and leaders are code-native and remain independent of the image generator.
overlay=Image.new('RGBA',canvas.size)
draw=ImageDraw.Draw(overlay)
font_root=ROOT/'assets/fonts'
FONTS={'sans':'Sans-Regular.ttf','bold':'Sans-Bold.ttf','serif':'Serif-Regular.otf','cjk':'CJK-Regular.ttc'}
text_checks=[]
def font(size,kind='sans'):
    return ImageFont.truetype(str(font_root/FONTS[kind]),round(size*S),index=2 if kind=='cjk' else 0)
def txt(x,y,text,size=24,kind='sans',fill=CREAM):
    f=font(size,kind);pos=(round(x*S),round(y*S))
    draw.text(pos,text,font=f,fill=fill,anchor='lt')
    b=draw.textbbox(pos,text,font=f,anchor='lt')
    text_checks.append({'text':text,'bbox':[round(v/S,2) for v in b]})
    return (b[2]-b[0])/S

def rule(points,color=LINE,width=1.2):
    draw.line([(round(x*S),round(y*S)) for x,y in points],fill=color,width=max(1,round(width*S)),joint='curve')

def swatch(x,y,i,size=11):
    draw.rectangle((x*S,y*S,(x+size)*S,(y+size)*S),fill=tuple(PALETTE[i])+(255,))

def fmt(v):return f'{v:.1f}%'

# Header: the topic and synthetic status are part of the primary hierarchy.
txt(84,42,DATA['topic_zh'],23,'cjk',MUTED)
txt(82,85,'Gift budget preferences',76,'serif')
txt(87,170,'Holiday giving  /  Six mutually exclusive budget choices',23,'sans',MUTED)
draw.rounded_rectangle((1583*S,51*S,1916*S,108*S),radius=3*S,outline='#d7be80',width=2*S,fill=(17,46,33,170))
txt(1608,64,DATA['visible_notice'],27,'bold','#efdfb2')
# Compact sample-size block in clear floor, separate from all data sectors.
txt(88,342,str(TOTAL),82,'serif')
txt(92,436,DATA['series'][0]['unit'],25,'sans')
txt(92,476,'One participant. One choice.',21,'sans',MUTED)
rule([(92,527),(338,527)],(216,211,183,132),1)

# Direct labels retain every supplied category, count, percentage and unit.
# Fractions only select a readable point WITHIN each sector; angles still come
# exclusively from the numerical mapping above.
fractions=[.70,.89,.50,.50,.50,.62]
leader_paths=[[(1420,413),(1887,413)],[(1510,1125),(1887,1125)],
              [(1400,1320),(1887,1320)],[(1130,1221),(1360,1221)],
              [(965,1221),(884,1221)],[(568,785),(94,785)]]
label_positions=[(1490,274,64),(1555,968,60),(1530,1171,54),
                 (1170,1240,42),(887,1240,42),(94,624,71)]
leaders=[]
for i,(cat,v,pct) in enumerate(zip(CATS,VALUES,PCTS)):
    theta=math.radians(BOUNDS[i]+fractions[i]*ANGLES[i])
    start=(CX+(R+9)*math.cos(theta),CY+(R+9)*math.sin(theta))
    points=[start]+leader_paths[i]
    rule(points,LINE,1.15)
    leaders.append({'category':cat,'angle_deg':math.degrees(theta),'points':points})
    x,y,sz=label_positions[i]
    small=i in (3,4)
    swatch(x,y+7,i,10 if small else 12)
    txt(x+(20 if small else 25),y,cat,23 if small else 29,'sans')
    txt(x,y+(30 if small else 43),fmt(pct),sz,'serif')
    count_y=y+(75 if small else 43+sz+8)
    txt(x,count_y,f'{int(v)} {DATA["series"][0]["unit"]}',17 if small else 22,'sans',MUTED)

# Footer clarifies the supplied synthetic category semantics without inventing
# currency thresholds, real participants, survey provenance, or population claims.
rule([(84,1358),(1916,1358)],(221,223,197,105),1)
txt(86,1378,'One response per simulated participant. Shares sum to 100%.',18,'sans',MUTED)
txt(1065,1378,'Budget A–E are synthetic options; no currency ranges are specified.',18,'sans',MUTED)
canvas=Image.alpha_composite(canvas,overlay)
final=canvas.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
final.save(ROOT/'final.png',optimize=True)
(ROOT/'artifacts').mkdir(exist_ok=True)
metrics={
 'canvas':[W,H],'center':[CX,CY],'data_radius_px':R,'start_angle_deg_clockwise_from_right':START,
 'source_to_output':{'source_center':[scx,scy],'x_scale':sx,'y_scale':sy,'output_center':[CX,CY]},
 'total':TOTAL,'unit':DATA['series'][0]['unit'],
 'sectors':[{'category':CATS[i],'count':int(VALUES[i]),'percentage':float(PCTS[i]),
  'start_deg':float(BOUNDS[i]),'end_deg':float(BOUNDS[i+1]),'span_deg':float(ANGLES[i]),
  'palette_rgb':PALETTE[i]} for i in range(6)],
 'leaders':leaders,'text':text_checks,
 'foreground_mask':'assets/foreground_mask.png'
}
(ROOT/'artifacts/render_metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'canvas':[W,H],'total':TOTAL,'percentages':PCTS.tolist(),'angle_sum':float(ANGLES.sum())}))
