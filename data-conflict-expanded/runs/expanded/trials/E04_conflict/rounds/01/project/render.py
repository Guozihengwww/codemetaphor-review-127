#!/usr/bin/env python3
"""Offline renderer. Run through capture_render.py to preserve full attempts."""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent

def read_json(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))

INPUT = read_json('input.json')
BRIEF = read_json(INPUT['brief'])
DATA = read_json(INPUT['data'])
CORE = read_json(INPUT['critical_metaphor_contract'])
assert (ROOT/INPUT['reference']).is_file(), 'Keep the frozen visual reference available.'
W,H = BRIEF['width'],BRIEF['height']
S = 2
INK = '#272A28'
MUTED = '#656660'
RED = '#B84232'
PAPER = '#F6F1E5'
C = (810.,1150.)
R = 500.
PALETTE = ['#ECC7B6','#E5B5A0','#DBA38C','#D09278',
           '#E1C98F','#4B7773','#87A299','#385E65',
           '#5E8384','#A9B9AA','#E2C292','#F0DBB9']
VALUES=DATA['series'][0]['values']
assert len(VALUES)==len(DATA['categories'])==len(PALETTE)==12
assert math.isclose(sum(VALUES),100.0,abs_tol=1e-8)
assert DATA['series'][0]['unit']=='%'
EDGES=np.r_[0,np.cumsum(VALUES)]*3.6

im=Image.new('RGBA',(W*S,H*S),PAPER)
d=ImageDraw.Draw(im)
fonts={}
texts=[]
def font(name,size):
    key=(name,size)
    if key not in fonts: fonts[key]=ImageFont.truetype(str(ROOT/'assets'/'fonts'/name),round(size*S))
    return fonts[key]
def text(x,y,t,size=30,kind='sans.otf',fill=INK,anchor='lt'):
    f=font(kind,size)
    d.text((round(x*S),round(y*S)),t,font=f,fill=fill,anchor=anchor)
    box=d.textbbox((round(x*S),round(y*S)),t,font=f,anchor=anchor)
    texts.append({'text':t,'bbox_px':[round(z/S,2) for z in box]})
    return box

def line(points,fill=INK,width=2):
    d.line([(round(x*S),round(y*S)) for x,y in points],fill=fill,width=max(1,round(width*S)),joint='curve')

def dot(x,y,r,fill=INK):
    d.ellipse(((x-r)*S,(y-r)*S,(x+r)*S,(y+r)*S),fill=fill)

def xy(angle,r=R):
    a=math.radians(angle)
    return (C[0]+r*math.sin(a),C[1]-r*math.cos(a))

# Editorial heading. Every quantitative/category label below comes from data.json.
d.rounded_rectangle((90*S,79*S,369*S,123*S),radius=4*S,fill=RED)
text(109,91,DATA['visible_notice'],25,'sans_bold.otf',PAPER)
text(1710,95,'AGE  /  GENDER  /  RACE',23,'sans.otf',MUTED,anchor='rt')
text(84,168,'Portrait of',103,'serif.otf')
text(82,265,'unemployment',121,'serif.otf')
text(1135,283,'Share of unemployed people',28,'sans.otf')
text(1135,321,'by age, gender and race (%)',28,'sans.otf')
text(1135,365,'12 categories  ·  100% total',24,'sans_bold.otf',RED)
line([(90,418),(1710,418)],fill='#A5A49B',width=1.4)

# The generated frame is non-data. Its ear and hair surround the exact code-made disk.
asset_scale=R/465.0
asset_size=round(1254*asset_scale*S)
asset_x=round((C[0]-625*asset_scale)*S)
asset_y=round((C[1]-705*asset_scale)*S)
frame=Image.open(ROOT/'assets'/'portrait_frame.png').convert('RGBA').resize((asset_size,asset_size),Image.Resampling.LANCZOS)
im.alpha_composite(frame,(asset_x,asset_y))

# Linear angular encoding, computed directly at supersampled pixel centers.
N=round(2*R*S)
yy,xx=np.mgrid[0:N,0:N].astype('float32')
xx=(xx+.5)/S-R; yy=(yy+.5)/S-R
angle=np.degrees(np.arctan2(xx,-yy))%360
idx=np.searchsorted(EDGES[1:],angle,side='right').clip(0,len(VALUES)-1)
pal=np.array([tuple(bytes.fromhex(c[1:])) for c in PALETTE],dtype='float32')
base=pal[idx]
tex=np.asarray(Image.open(ROOT/'assets'/'face_texture.png'),dtype='float32')/230.
th,tw=tex.shape
texture=tex[(np.arange(N)//S)%th][:,(np.arange(N)//S)%tw]
base=np.clip(base*texture[:,:,None],0,255).astype('uint8')
alpha=np.where(xx*xx+yy*yy<=R*R,255,0).astype('uint8')
disc=Image.fromarray(np.dstack((base,alpha)),'RGBA')
im.alpha_composite(disc,(round((C[0]-R)*S),round((C[1]-R)*S)))
d=ImageDraw.Draw(im)
for a in EDGES[:-1]:
    line([C,xy(float(a))],fill='#353A36',width=2.15)
d.ellipse(((C[0]-R)*S,(C[1]-R)*S,(C[0]+R)*S,(C[1]+R)*S),outline=INK,width=4*S)
# All features are original generated raster material. The 50-source-pixel lift
# aligns the nose bulb with the logical disk center, without changing the disk.
features=Image.open(ROOT/'assets'/'portrait_features.png').convert('RGBA').resize((asset_size,asset_size),Image.Resampling.LANCZOS)
im.alpha_composite(features,(asset_x,asset_y-round(50*asset_scale*S)))
d=ImageDraw.Draw(im)

# Direct perimeter callouts follow the supplied order. No legend substitutes names.
LABELS=[
 {'x':1270,'y':520,'width':440,'size':30,'lead':[(927,584),(1245,565)]},
 {'x':1470,'y':705,'width':260,'size':29,'lead':[(1134,686),(1445,745)]},
 {'x':1490,'y':905,'width':235,'size':29,'lead':[(1298,881),(1465,945)]},
 {'x':1510,'y':1100,'width':220,'size':29,'lead':[(1400,1105),(1485,1140)]},
 {'x':1450,'y':1325,'width':280,'size':30,'lead':[(1370,1311),(1425,1365)]},
 {'x':1340,'y':1485,'width':380,'size':30,'lead':[(1293,1446),(1315,1525)]},
 {'x':1130,'y':1660,'width':570,'size':30,'lead':[(1173,1575),(1105,1700)]},
 {'x':1000,'y':1760,'width':340,'size':28,'lead':[(1007,1665),(1060,1736)]},
 {'x':680,'y':1770,'width':310,'size':28,'lead':[(877,1696),(807,1744)]},
 {'x':355,'y':1725,'width':320,'size':28,'lead':[(699,1688),(505,1701)]},
 {'x':85,'y':1500,'width':280,'size':32,'lead':[(454,1616),(370,1560)]},
 {'x':85,'y':785,'width':232,'size':32,'lead':[(314,907),(310,863)]},
]

def wrap(t,maxw,f):
    words=t.split(); rows=[]; row=''
    for word in words:
        test=(row+' '+word).strip()
        if row and d.textlength(test,font=f)>maxw*S:
            rows.append(row); row=word
        else: row=test
    if row: rows.append(row)
    return rows

sectors=[]
for i,(cat,val,p) in enumerate(zip(DATA['categories'],VALUES,LABELS)):
    mid=(EDGES[i]+EDGES[i+1])/2
    start=xy(float(mid))
    pts=[start]+p['lead']
    # The paper halo keeps callouts readable across the non-data hair frame.
    line(pts,fill=PAPER,width=6.3)
    line(pts,fill=INK,width=1.6)
    dot(*start,4.3,fill=PAPER);dot(*start,2.8,fill=INK)
    x,y=p['x'],p['y']; sz=p['size']
    parent=DATA['category_parent'][i]
    rows=wrap(parent,p['width'],font('sans_bold.otf',sz))
    for row in rows:
        text(x,y,row,sz,'sans_bold.otf');y+=sz+5
    if ' / ' in cat:
        suffix=cat.split(' / ',1)[1]
        text(x,y+1,suffix,24,'sans.otf',MUTED);y+=31
    value_string=str(val)+'%'
    numsize=60 if i==11 else 54 if i==10 else 46
    text(x-1,y+6,value_string,numsize,'sans_bold.otf',RED)
    sectors.append({'index':i,'category':cat,'value':val,'unit':'%',
                    'start_deg':round(float(EDGES[i]),8),'end_deg':round(float(EDGES[i+1]),8),
                    'mid_deg':round(float(mid),8),'color':PALETTE[i],
                    'leader_anchor_px':[round(z,3) for z in start],
                    'label_parent_lines':rows,'numeric_label':value_string})

# Reading rule and synthetic scope. These are descriptive, not additional data.
line([(90,1904),(1710,1904)],fill='#A5A49B',width=1.4)
text(90,1930,'HOW TO READ',22,'sans_bold.otf',RED)
text(304,1929,'Clockwise from 12 o’clock. Sector angle = 3.6° × share (%).',23,'sans.otf')
text(304,1963,'The nose is the center; hair and ear are outside the data circle.',23,'sans.otf')
text(90,2008,'Synthetic categories, not a real statistical survey.',20,'sans.otf',MUTED)
text(1710,2008,'Subcategories are mutually exclusive parts of the named parent.',20,'sans.otf',MUTED,anchor='rt')

im.convert('RGB').resize((W,H),Image.Resampling.LANCZOS).save(ROOT/'final.png',optimize=True)
metadata={'canvas':[W,H],'center_px':list(C),'radius_px':R,'angular_origin':'12 o’clock',
          'direction':'clockwise','sectors':sectors,'text_objects':texts,
          'portrait_source_center_px':[625,705],'portrait_source_radius_px':465,
          'features_source_translation_px':[0,-50],'asset_scale':asset_scale,
          'frozen_dependencies':[INPUT[k] for k in ('brief','data','reference','critical_metaphor_contract')],
          'render_route':'local Pillow/numpy, generated raster components, no model or network calls'}
(ROOT/'artifacts'/'render_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Rendered final.png:',W,H,'with',len(VALUES),'exact angular sectors.')

if __name__ == '__main__':
    pass
