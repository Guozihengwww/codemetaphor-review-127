#!/usr/bin/env python3
"""Offline renderer: generated photographs + a single data-driven cylindrical mapping."""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parent
inputs = json.loads((ROOT / 'input.json').read_text(encoding='utf-8'))
brief = json.loads((ROOT / inputs['brief']).read_text(encoding='utf-8'))
data = json.loads((ROOT / inputs['data']).read_text(encoding='utf-8'))
core = json.loads((ROOT / inputs['critical_metaphor_contract']).read_text(encoding='utf-8'))
assert (ROOT / inputs['reference']).is_file()  # Design source is preserved, never sampled.
assert data['synthetic'] and core['topic_zh'] == data['topic_zh']
values = np.array(data['series'][0]['values'], dtype=float)
categories = data['categories']
assert len(values) == len(categories) == len(data['category_use_illustrations'])
assert abs(float(values.sum()) - 100) < 1e-8
W, HH = brief['width'], brief['height']
S = 2
canvas = Image.new('RGB', (W*S, HH*S), 'white')
INK = '#202B32'
MUTED = '#627078'
LIGHT = '#DCE2E5'
FONT = ROOT / 'assets' / 'font_regular.ttf'
BOLD = ROOT / 'assets' / 'font_bold.ttf'

def font(size, bold=False):
    return ImageFont.truetype(str(BOLD if bold else FONT), round(size*S))

def text(x,y,label,size=28,fill=INK,bold=False,anchor='la',stroke=0,stroke_fill=None):
    d = ImageDraw.Draw(canvas)
    d.text((round(x*S),round(y*S)), label, font=font(size,bold), fill=fill, anchor=anchor,
           stroke_width=round(stroke*S), stroke_fill=stroke_fill)

def line(points,fill=LIGHT,width=2):
    ImageDraw.Draw(canvas).line([(round(x*S),round(y*S)) for x,y in points],fill=fill,width=max(1,round(width*S)),joint='curve')

def ellipse(box,fill=None,outline=None,width=1):
    ImageDraw.Draw(canvas).ellipse(tuple(round(p*S) for p in box),fill=fill,outline=outline,width=max(1,round(width*S)))

def rounded(box,r,fill,outline=None,width=1):
    ImageDraw.Draw(canvas).rounded_rectangle(tuple(round(p*S) for p in box),radius=round(r*S),fill=fill,outline=outline,width=round(width*S))

def fit_asset(name, box):
    im=Image.open(ROOT/'assets'/f'use_{name}.png').convert('RGB')
    x,y,w,h=box
    # A plain white mat keeps the generated object shadows intact.
    scale=min(w*S/im.width,h*S/im.height)
    im=im.resize((round(im.width*scale),round(im.height*scale)),Image.Resampling.LANCZOS)
    canvas.paste(im,(round((x+w/2)*S-im.width/2),round((y+h/2)*S-im.height/2)))

# Title and measure hierarchy.
rounded((72,59,371,108),9,INK)
text(94,70,data['visible_notice'],25,'white',True)
text(72,129,data['title'].removeprefix('SYNTHETIC '),77,INK,True)
text(75,231,data['series'][0]['label']+' ('+data['series'][0]['unit']+')',29,MUTED)
line([(74,292),(1626,292)],'#CDD5D9',1.5)

# One model controls all interfaces, annotations, and numeric measurements.
X0, BW, Y0, H, A = 500.0, 700.0, 470.0, 1180.0, 60.0
bounds = np.r_[0, np.cumsum(values)] * H / 100.0
centers = (bounds[:-1] + bounds[1:]) / 2
palette = ['#9955AF','#259AB5','#D0DED6','#EB8A3B','#9A705D','#586CBA',
           '#624988','#C94D59','#E9B943','#439C8D','#88AB4F']
colors = np.array([tuple(bytes.fromhex(c[1:])) for c in palette],dtype=np.float32)

def curve(x):
    return A * math.sqrt(max(0.0,1.0-((x-(X0+BW/2))/(BW/2))**2))

def layer_y(i,x):
    return Y0 + curve(x) + centers[i]

def sample_rgb(source,sy,sx):
    return np.stack([map_coordinates(source[:,:,k], [sy,sx],order=1,mode='constant',cval=255) for k in range(3)],axis=-1)

raw = np.array(Image.open(ROOT/'assets'/'raw_01.png').convert('RGB'),dtype=np.float32)
# Physical contact shadow, underneath the generated bottom rim.
shadow=Image.new('RGBA',canvas.size,(0,0,0,0))
sd=ImageDraw.Draw(shadow)
sd.ellipse((round((X0-48)*S),round((Y0+H+25)*S),round((X0+BW+58)*S),round((Y0+H+100)*S)),fill=(29,39,44,50))
shadow=shadow.filter(ImageFilter.GaussianBlur(16*S))
canvas=Image.alpha_composite(canvas.convert('RGBA'),shadow).convert('RGB')

# The cap is a separate non-data surface. Its generated pixels are resampled, not re-created.
box=(X0-20,Y0-175,X0+BW+20,Y0+A+2)
bx,by,br,bb=box
xx=(np.arange(round((br-bx)*S),dtype=np.float32)+.5)/S+bx
yy=(np.arange(round((bb-by)*S),dtype=np.float32)+.5)/S+by
xx,yy=np.meshgrid(xx,yy)
u=(xx-X0)/BW
v=np.sqrt(np.maximum(0,1-(2*u-1)**2))
sx=130+690*u
sy=266+(yy-Y0)*(36/60)
cap=sample_rgb(raw,sy,sx)
mask=(yy <= Y0+A*v).astype('uint8')*255
cap=np.dstack([np.clip(cap,0,255).astype('uint8'),mask])
canvas.paste(Image.fromarray(cap,'RGBA'),(round(bx*S),round(by*S)),Image.fromarray(mask,'L'))

# The sidewall: y(x,t) = Y0 + A*sqrt(1-u^2) + t, t in [0,H].
# Every boundary is the SAME curve translated by H*cumulative_share/100.
bx,by=X0,Y0
ww,hh=round(BW*S),round((H+A+1)*S)
xx=(np.arange(ww,dtype=np.float32)+.5)/S+X0
yy=(np.arange(hh,dtype=np.float32)+.5)/S+Y0
xx,yy=np.meshgrid(xx,yy)
u=(xx-X0)/BW
v=np.sqrt(np.maximum(0,1-(2*u-1)**2))
t=yy-Y0-A*v
valid=(t>=0)&(t<H)
source_top=266+36*v
source_bottom=1370+97*v
sx=130+690*u
sy=source_top+np.clip(t/H,0,1)*(source_bottom-source_top)
lum_source=np.array(Image.open(ROOT/'assets'/'drum_luminance.png'),dtype=np.float32)
lum=map_coordinates(lum_source,[sy,sx],order=1,mode='nearest')
ids=np.clip(np.searchsorted(bounds,t,side='right')-1,0,len(values)-1)
shade=.40+.72*(lum/255.0)
wall=np.clip(colors[ids]*shade[:,:,None],0,255).astype('uint8')
wall=np.dstack([wall,valid.astype('uint8')*255])
canvas.paste(Image.fromarray(wall,'RGBA'),(round(bx*S),round(by*S)),Image.fromarray(valid.astype('uint8')*255,'L'))

# Generated base rim and its photographic contact shadow sit below the data wall.
bx,by=X0-45,Y0+H
ww,hh=round((BW+90)*S),round((A+116)*S)
xx=(np.arange(ww,dtype=np.float32)+.5)/S+bx
yy=(np.arange(hh,dtype=np.float32)+.5)/S+by
xx,yy=np.meshgrid(xx,yy)
u=(xx-X0)/BW
v=np.sqrt(np.maximum(0,1-(2*u-1)**2))
dy=yy-(Y0+H+A*v)
sx=130+690*u
sy=1370+97*v+dy*.96
base=sample_rgb(raw,sy,sx)
valid=(dy>=0)
# Feather only the photographic matte outside the rim, removing rectangular shadow edges.
x_fade=np.clip(np.minimum((xx-bx)/40,(bx+BW+90-xx)/40),0,1)
y_fade=np.clip((116-dy)/55,0,1)
base_alpha=(255*valid*x_fade*y_fade).astype('uint8')
base=np.dstack([np.clip(base,0,255).astype('uint8'),base_alpha])
canvas.paste(Image.fromarray(base,'RGBA'),(round(bx*S),round(by*S)),Image.fromarray(base_alpha,'L'))

# Subtle outer data-wall edge; no artificial minimum-height bands or separators.
line([(X0,Y0),(X0,Y0+H)],'#58616A',.7)
line([(X0+BW,Y0),(X0+BW,Y0+H)],'#58616A',.7)

# Numeric labels for the six larger layers remain part of the object.
cx=X0+BW/2
for i in range(6):
    cy=layer_y(i,cx)
    if i in (0,1):
        val_size=86 if i==0 else 72
        text(cx+1,cy-56+1,f'{values[i]:.1f}%',val_size,'#302837',True,'ma')
        text(cx,cy-56,f'{values[i]:.1f}%',val_size,'white',True,'ma')
        text(cx,cy+49,categories[i],43 if i==0 else 40,'white',True,'ma')
    elif i==2:
        text(cx,cy-49,f'{values[i]:.1f}%',59,INK,True,'ma')
        text(cx,cy+23,categories[i],34,INK,True,'ma')
    else:
        size={3:39,4:33,5:28}[i]
        text(cx,cy-size*.57,f'{values[i]:.1f}%  {categories[i]}',size,'white',True,'ma')

# Quiet explanatory block in the same position as the reference's editorial note.
text(1273,410,'ONE DAILY TOTAL',22,MUTED,True)
text(1267,447,f'{float(values.sum()):.1f}%',68,INK,True)
text(1272,535,f'{len(categories)} product categories',26,INK)
line([(1274,592),(1614,592)],LIGHT,1.5)
text(1273,623,'From transport fuels',27,INK)
text(1273,664,'to everyday materials.',27,INK)
text(1273,730,'Each layer represents its',23,MUTED)
text(1273,766,'share of the synthetic',23,MUTED)
text(1273,802,'daily total.',23,MUTED)

# Leaders are intentionally drawn before illustrations.
def leader(i,points):
    line(points,'#9EA9AF',2.7)
    x,y=points[0]
    r=4.0 if values[i]>=4 else 3.2
    ellipse((x-r,y-r,x+r,y+r),fill=palette[i],outline='white',width=1)

leader(0,[(X0+12,layer_y(0,X0+12)),(448,693),(425,693)])
leader(1,[(X0+12,layer_y(1,X0+12)),(454,1080),(430,1080)])
leader(2,[(X0+12,layer_y(2,X0+12)),(457,1408),(430,1408)])
leader(3,[(X0+BW-12,layer_y(3,X0+BW-12)),(1232,layer_y(3,X0+BW-12)),(1232,1145),(1270,1145)])
leader(4,[(X0+BW-12,layer_y(4,X0+BW-12)),(1235,1470),(1270,1470)])
leader(5,[(X0+BW-12,layer_y(5,X0+BW-12)),(1232,1610),(1232,1741),(1270,1741)])

# The eleven illustrations encode category semantics, not magnitude.
ellipse((62,493,424,853),'white',LIGHT,1.8)
fit_asset('gasoline',(138,516,212,276))
text(243,802,'Fuel pump',24,MUTED,False,'ma')
ellipse((50,891,430,1241),'white',LIGHT,1.8)
fit_asset('diesel',(95,923,292,233))
text(240,1164,'Tractor · truck',22,MUTED,False,'ma')
text(240,1195,'Locomotive · backhoe',22,MUTED,False,'ma')
ellipse((39,1263,431,1616),'white',LIGHT,1.8)
fit_asset('hgl',(93,1291,284,249))
text(235,1555,'Materials & manufactured goods',20,MUTED,False,'ma')
ellipse((1268,979,1618,1310),'white',LIGHT,1.8)
fit_asset('jet_fuel',(1310,1015,266,239))
text(1443,1262,'Passenger & smaller aircraft',20,MUTED,False,'ma')
ellipse((1268,1322,1618,1621),'white',LIGHT,1.8)
fit_asset('other_products',(1319,1340,248,226))
text(1443,1570,'Everyday products',22,MUTED,False,'ma')
ellipse((1268,1640,1618,1843),'white',LIGHT,1.8)
fit_asset('propane',(1318,1648,250,152))
text(1443,1804,'Tank & barbecue grill',21,MUTED,False,'ma')
# Restore uninterrupted circular frames after compositing the white-matted photos.
for box in [(62,493,424,853),(50,891,430,1241),(39,1263,431,1616),
            (1268,979,1618,1310),(1268,1322,1618,1621),(1268,1640,1618,1843)]:
    ellipse(box,outline=LIGHT,width=1.8)

# Five small shares keep their exact heights. The lower fan provides room for names.
card_centers=[205,525,845,1165,1485]
leader(6,[(X0+14,layer_y(6,X0+14)),(466,1629),(466,1772),(205,1813),(205,1850)])
leader(7,[(X0+38,layer_y(7,X0+38)),(480,1695),(480,1790),(525,1830),(525,1850)])
leader(8,[(818,layer_y(8,818)),(818,1776),(845,1819),(845,1850)])
leader(9,[(972,layer_y(9,972)),(1024,1788),(1165,1820),(1165,1850)])
leader(10,[(1102,layer_y(10,1102)),(1180,1788),(1200,1848),(1485,1856)])
use_names=['asphalt','residual_fuel_oil','propylene','petroleum_coke','lubricants']
name_lines=[['Asphalt'],['Residual Fuel Oil'],['Propylene'],['Petroleum Coke'],['Lubricants']]
use_captions=['Road surfacing','Powered cargo vessel','Molded plastic items','Steel-making end use','Lubricant packages']
for i,cx in enumerate(card_centers,6):
    ellipse((cx-6,1848,cx+6,1860),fill=palette[i])
    text(cx,1868,f'{values[i]:.1f}%',43,INK,True,'ma')
    text(cx,1924,categories[i],26,INK,True,'ma')
    fit_asset(use_names[i-6],(cx-133,1970,266,136))
    text(cx,2116,use_captions[i-6],19,MUTED,False,'ma')

line([(74,2165),(1626,2165)],LIGHT,1.2)
text(74,2174,'Layer height ∝ share · Metal lid excluded · Illustrations show uses, not additional quantities',18,MUTED)
# Downsampling is the only final-image operation; candidate capture is external.
canvas=canvas.resize((W,HH),Image.Resampling.LANCZOS)
canvas.save(ROOT/'final.png',optimize=True)
(ROOT/'artifacts').mkdir(exist_ok=True)
geometry={
    'canvas':[W,HH], 'sidewall':{'x':X0,'width':BW,'top_at_edge':Y0,'height':H,'curve_amplitude':A},
    'mapping':'y = Y0 + A*sqrt(1-((x-center)/(width/2))**2) + H*cumulative_percent/100',
    'lid_in_data':False, 'minimum_height_applied':False,
    'categories':[{'category':c,'value_percent':float(v),'height_px':float(v*H/100),'top_t_px':float(bounds[i]),'bottom_t_px':float(bounds[i+1]),'color':palette[i]} for i,(c,v) in enumerate(zip(categories,values))]
}
(ROOT/'artifacts'/'geometry.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Rendered final.png at',W,HH,'with',len(values),'data-driven layers.')
