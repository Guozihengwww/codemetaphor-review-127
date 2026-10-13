#!/usr/bin/env python3
"""Offline renderer. Data geometry is a translated elliptical sidewall.
Run a complete render only via capture_render.py in this experiment.
"""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.ndimage import map_coordinates

ROOT=Path(__file__).resolve().parent
DATA=json.loads((ROOT/'data.json').read_text(encoding='utf-8'))
BRIEF=json.loads((ROOT/'brief.json').read_text(encoding='utf-8'))
CORE=json.loads((ROOT/'core.json').read_text(encoding='utf-8'))
W,H=BRIEF['width'],BRIEF['height']
S=2
INK='#172f3b'; MUTED='#617078'; LINE='#cad1d4'
COLORS=['#803d87','#168da7','#347a92','#61afbb','#196476',
        '#becbbb','#96b6a6','#d4c9aa','#e79c47','#c7783c',
        '#826052','#527aab','#525361','#b55b65','#bfac42','#718899','#29927b']
VALUES=DATA['series'][0]['values']
CATS=DATA['categories']
assert len(VALUES)==len(CATS)==len(COLORS)
assert abs(sum(VALUES)-100)<1e-8
X0,X1=540.,1160.
CENTER=(X0+X1)/2
TOP=430.
SIDE_H=1360.
CURVE=60.
CUM=np.r_[0,np.cumsum(VALUES)]
img=Image.new('RGB',(W*S,H*S),'white')
draw=ImageDraw.Draw(img)
label_audit=[]

def rgb(h):return tuple(int(h[i:i+2],16) for i in (1,3,5))
def fnt(size,bold=False,cond=False):
 name=('Condensed' if cond else 'Sans')+('-Bold' if bold else '')+'.ttf'
 return ImageFont.truetype(str(ROOT/'assets'/name),round(size*S))
def text(x,y,value,size=26,fill=INK,bold=False,anchor='lt',cond=False,stroke=0):
 font=fnt(size,bold,cond)
 draw.text((round(x*S),round(y*S)),value,font=font,fill=fill,anchor=anchor,stroke_width=round(stroke*S),stroke_fill='white')
 return tuple(v/S for v in draw.textbbox((x*S,y*S),value,font=font,anchor=anchor))
def line(points,fill=LINE,width=1.5):
 draw.line([(round(x*S),round(y*S)) for x,y in points],fill=fill,width=round(width*S),joint='curve')
def dot(x,y,color,r=4):
 draw.ellipse(((x-r)*S,(y-r)*S,(x+r)*S,(y+r)*S),fill=color)
def illustration(key,box):
 a=Image.open(ROOT/'assets'/f'use_{key}.png').convert('RGB')
 bx,by,bw,bh=box
 ratio=min(bw/a.width,bh/a.height)
 nw,nh=round(a.width*ratio*S),round(a.height*ratio*S)
 a=a.resize((nw,nh),Image.Resampling.LANCZOS)
 px=round((bx+(bw-nw/S)/2)*S);py=round((by+(bh-nh/S)/2)*S)
 img.paste(a,(px,py))
def fmt(v):
 # Preserve the supplied decimal precision (e.g. 42.0, 3.15).
 return str(v)+'%'
def boundary_y(x,pct):
 u=(x-CENTER)/((X1-X0)/2)
 return TOP+CURVE*math.sqrt(max(0,1-u*u))+SIDE_H*pct/100

def connect(i,side,ey,ex=None,lane=None):
 color=COLORS[i]
 if side=='left':
  ax=X0+2; end=490 if ex is None else ex; bend=513 if lane is None else lane
 else:
  ax=X1-2; end=1210 if ex is None else ex; bend=1187 if lane is None else lane
 ay=boundary_y(ax,(CUM[i]+CUM[i+1])/2)
 line([(ax,ay),(bend,ay),(end,ey)],fill=color,width=1.7)
 dot(ax,ay,color,4.3);dot(end,ey,color,2.7)
 return {'index':i,'category':CATS[i],'anchor':[ax,ay],'label_endpoint':[end,ey]}
connectors=[]

# Header.
draw.rounded_rectangle((75*S,57*S,401*S,106*S),radius=9*S,fill=INK)
text(238,82,'SYNTHETIC DATA',22,'white',True,anchor='mm')
text(1625,82,'ENERGY / MATERIALS',20,MUTED,anchor='rm')
text(71,137,'Petroleum product use',83,INK,True,cond=True)
text(76,240,'Share of synthetic daily total (%)',31,INK)
text(1623,251,'17 CATEGORIES  /  TOTAL 100.0%',21,MUTED,anchor='rm',bold=True)
line([(75,302),(1625,302)],'#d6dfe2',1.4)

# Soft grounded contact shadow; does not encode data.
shadow=Image.new('RGBA',img.size,(0,0,0,0))
sd=ImageDraw.Draw(shadow)
sd.ellipse((525*S,1793*S,1192*S,1880*S),fill=(18,32,40,35))
shadow=shadow.filter(ImageFilter.GaussianBlur(22*S))
img=Image.alpha_composite(img.convert('RGBA'),shadow).convert('RGB')
draw=ImageDraw.Draw(img)

# Map generated metal texture onto exact analytical sidewall coordinates.
raw=np.asarray(Image.open(ROOT/'assets/raw_01.png').convert('RGBA')).astype(np.float32)
lw=round((X1-X0)*S)
lh=round((SIDE_H+CURVE+2)*S)
xx=(np.arange(lw)+.5)/S+X0
yy=(np.arange(lh)+.5)/S+TOP
u=(xx-CENTER)/((X1-X0)/2)
q=np.sqrt(np.clip(1-u*u,0,1))
base=TOP+CURVE*q
v=(yy[:,None]-base[None,:])/SIDE_H
sx=111.5+(xx-X0)/(X1-X0)*746.
src_top=166+64*q
src_bottom=1390+118*q
sy=src_top[None,:]+np.clip(v,0,1)*(src_bottom-src_top)[None,:]
sx2=np.broadcast_to(sx[None,:],sy.shape)
gray=raw[:,:,:3].mean(axis=2)
lum=map_coordinates(gray,[sy,sx2],order=1,mode='nearest')/180.
# Multiplicative color retains the actual generated reflections and metal grain.
metal_gain=np.clip(.20+.80*lum,.55,1.30)
ids=np.clip(np.searchsorted(CUM[1:],v*100,side='right'),0,len(VALUES)-1)
palette=np.array([rgb(c) for c in COLORS],np.float32)
color=palette[ids]*metal_gain[:,:,None]
alpha=((v>=0)&(v<=1))*255
body=np.dstack([np.clip(color,0,255).astype(np.uint8),alpha.astype(np.uint8)])
img.paste(Image.fromarray(body),(round(X0*S),round(TOP*S)),Image.fromarray(body[:,:,3]))

# Generated cap mapped separately: entirely outside the quantitative sidewall.
cap_x0,cap_y0=525,326
cap_w,cap_h=650,167
cy,cx=np.mgrid[0:cap_h*S,0:cap_w*S].astype(np.float32)
cx=(cx+.5)/S+cap_x0;cy=(cy+.5)/S+cap_y0
rsx=111.5+(cx-X0)/(X1-X0)*746.
rsy=166+(cy-TOP)/(CURVE/64.)
cu=(cx-CENTER)/((X1-X0)/2)
cb=TOP+CURVE*np.sqrt(np.clip(1-cu*cu,0,1))
cap=np.stack([map_coordinates(raw[:,:,k],[rsy,rsx],order=1,mode='constant',cval=0) for k in range(4)],axis=2)
cap[:,:,3]*=(cy<=cb+.3)
cap[:,:,3][cap[:,:,3]<24]=0
cap=Image.fromarray(np.uint8(np.clip(cap,0,255)))
img.paste(cap,(cap_x0*S,cap_y0*S),cap)

# Generated steel base rim, translated below 100%; no layer is covered.
bw=640;bh=83;bx=530;by=round(TOP+SIDE_H)
ry,rx=np.mgrid[0:bh*S,0:bw*S].astype(np.float32)
rx=(rx+.5)/S+bx;ry=(ry+.5)/S+by
ru=(rx-CENTER)/((X1-X0)/2)
rq=np.sqrt(np.clip(1-ru*ru,0,1))
rbase=TOP+SIDE_H+CURVE*rq
rsx=111.5+(rx-X0)/(X1-X0)*746.
rsy=1390+118*rq+(ry-rbase)*1.65
rim=np.stack([map_coordinates(raw[:,:,k],[rsy,rsx],order=1,mode='constant',cval=0) for k in range(4)],axis=2)
rim[:,:,3]*=(ry>=rbase)&(ry<=rbase+18)&(np.abs(ru)<1.025)
rim[:,:,3][rim[:,:,3]<24]=0
rim=Image.fromarray(np.uint8(np.clip(rim,0,255)))
img.paste(rim,(bx*S,by*S),rim)
draw=ImageDraw.Draw(img)

# Fine true interfaces; every curve is a vertical translation of the same curve.
for b in CUM[1:-1]:
 pts=[(x,boundary_y(x,float(b))) for x in np.linspace(X0+.3,X1-.3,300)]
 line(pts,'#ffffff',.7)
# A single dominant value on the drum, with exact label and unit.
text(CENTER,737,fmt(VALUES[0]),97,'white',True,anchor='mm',cond=True)
text(CENTER,818,'Gasoline',41,'white',True,anchor='mm')
text(CENTER,865,'of the synthetic daily total',20,'#f7e9f7',anchor='mm')
# Values remain legible in the middle-sized layers; full category labels are outside.
for i in range(1,12):
 y=boundary_y(CENTER,float((CUM[i]+CUM[i+1])/2))
 tc='#243b39' if i in (5,6,7,8) else 'white'
 text(CENTER,y,fmt(VALUES[i]),28 if i<=4 or i>=10 else 24,tc,True,anchor='mm',cond=True)

# Peripheral use illustrations and explicitly bound category labels.
# Gasoline.
illustration('gasoline',(145,452,270,345))
text(76,833,'Gasoline',32,INK,True)
text(473,834,fmt(VALUES[0]),33,COLORS[0],True,anchor='rt',cond=True)
line([(76,886),(490,886)],'#e5e9ea',1.2)
connectors.append(connect(0,'left',865))
label_audit.append({'category':CATS[0],'value':fmt(VALUES[0]),'rect':[76,830,473,886]})

# Diesel: four mutually exclusive rows retain the full parent name.
illustration('diesel',(1225,570,400,350))
text(1225,941,'TRANSPORT & MACHINERY',18,MUTED,True)
for j,i in enumerate(range(1,5)):
 y=1000+j*70
 text(1225,y,CATS[i],24,INK,False,cond=True)
 text(1624,y+29,fmt(VALUES[i]),28,COLORS[i],True,anchor='rt',cond=True)
 line([(1225,y+59),(1624,y+59)],'#e6ebed',1)
 connectors.append(connect(i,'right',y+19))
 label_audit.append({'category':CATS[i],'value':fmt(VALUES[i]),'rect':[1225,y,1624,y+59]})

# HGL: labels are fully expanded for each subcategory.
illustration('hgl',(76,971,400,223))
for j,i in enumerate(range(5,8)):
 y=1226+j*73
 text(76,y,'Hydrocarbon Gas Liquids',25,INK,True,cond=True)
 text(76,y+30,f'subcategory {j+1}',22,MUTED)
 text(473,y+27,fmt(VALUES[i]),29,'#405e55',True,anchor='rt',cond=True)
 line([(76,y+62),(490,y+62)],'#e6ebed',1)
 connectors.append(connect(i,'left',y+34))
 label_audit.append({'category':CATS[i],'value':fmt(VALUES[i]),'rect':[76,y,490,y+62]})

# Jet fuel.
illustration('jet',(1270,1300,310,157))
for j,i in enumerate(range(8,10)):
 y=1477+j*62
 text(1225,y,CATS[i],24,INK,False,cond=True)
 text(1624,y+28,fmt(VALUES[i]),28,COLORS[i],True,anchor='rt',cond=True)
 line([(1225,y+56),(1624,y+56)],'#e6ebed',1)
 connectors.append(connect(i,'right',y+17))
 label_audit.append({'category':CATS[i],'value':fmt(VALUES[i]),'rect':[1225,y,1624,y+56]})

# Remaining individual categories. Tiny data layers are not enlarged.
# Side, asset, illustration box, text origin, connector end.
items=[
 (10,'left','other',(76,1505,199,171),(291,1543),1598,490,514),
 (11,'right','propane',(1225,1628,190,119),(1435,1647),1671,1210,1187),
 (12,'left','asphalt',(76,1728,182,139),(289,1760),1790,490,510),
 (13,'right','residual',(1225,1774,188,124),(1435,1790),1824,1210,1182),
 (14,'left','propylene',(76,1918,190,153),(289,1950),1984,490,523),
 (15,'right','coke',(1225,1912,180,125),(1435,1934),1968,1210,1194),
 (16,'right','lubricants',(1245,2040,143,117),(1435,2057),2091,1210,1203),
]
for i,side,key,box,tx,ey,ex,lane in items:
 illustration(key,box)
 x,y=tx
 names={10:['Other','Products'],11:['Propane'],12:['Asphalt'],13:['Residual','Fuel Oil'],14:['Propylene'],15:['Petroleum','Coke'],16:['Lubricants']}[i]
 for n,row in enumerate(names):text(x,y+29*n,row,26,INK,True,cond=True)
 vy=y+len(names)*29+8
 text(x,vy,fmt(VALUES[i]),33,COLORS[i],True,cond=True)
 connectors.append(connect(i,side,ey,ex,lane))
 label_audit.append({'category':CATS[i],'value':fmt(VALUES[i]),'rect':[x,y,1630 if side=='right' else 486,vy+35]})

# Reading key beneath the central object.
text(850,1930,'READING THE DRUM',19,MUTED,True,anchor='mt')
text(850,1970,'Layer height represents share.',25,INK,anchor='mt',cond=True)
text(850,2005,'The lid and base rim are excluded.',23,MUTED,anchor='mt',cond=True)
text(850,2057,'Use illustrations are examples,',22,MUTED,anchor='mt',cond=True)
text(850,2088,'not additional quantities.',22,MUTED,anchor='mt',cond=True)

# Footer states the synthetic scope, without borrowing reference statistics.
line([(75,2171),(1625,2171)],'#d6dfe2',1.3)
text(75,2184,'Synthetic subcategories partition their named parent. This is not a real-world survey.',17,MUTED)
text(1625,2184,'100.0%  /  17',17,MUTED,True,anchor='rt')

# Preserve the captured output. This program never calls a model or network.
final=img.resize((W,H),Image.Resampling.LANCZOS)
final.save(ROOT/'final.png')
(ROOT/'artifacts').mkdir(exist_ok=True)
geom={'canvas':[W,H],'series':DATA['series'][0]['label'],'unit':'%',
      'body':{'x0':X0,'x1':X1,'top_at_edge':TOP,'sidewall_height':SIDE_H,'curve_depth':CURVE,'lid_excluded':True,'base_rim_excluded':True},
      'mapping':'y(x,p) = TOP + CURVE * sqrt(1-((x-CENTER)/(width/2))**2) + SIDE_H * p / 100',
      'categories':[{'index':i,'label':c,'value':v,'cumulative_start':float(CUM[i]),'cumulative_end':float(CUM[i+1]),'axial_height_px':SIDE_H*v/100,'color':COLORS[i]} for i,(c,v) in enumerate(zip(CATS,VALUES))],
      'connectors':connectors,'labels':label_audit}
(ROOT/'artifacts'/'geometry.json').write_text(json.dumps(geom,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'output':'final.png','size':[W,H],'categories':len(CATS),'total':sum(VALUES),'min_layer_height_px':min(VALUES)*SIDE_H/100}))
