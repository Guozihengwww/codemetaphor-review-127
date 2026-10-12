#!/usr/bin/env python3
"""Rebuild the infographic from frozen local data and generated local assets."""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageChops
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts'
OUT.mkdir(exist_ok=True)
data = json.loads((ROOT/'data.json').read_text(encoding='utf-8'))
brief = json.loads((ROOT/'brief.json').read_text(encoding='utf-8'))
W,H = brief['width'],brief['height']
S = 2
INK = '#232323'
RUST = '#973e33'
PAPER = '#f6f1e7'
values = np.asarray(data['series'][0]['values'],dtype=float)
categories = data['categories']
n = len(values)
scale = data['price_scale']
assert scale['type'] == 'linear' and len(categories) == n
X0,X1 = 620.,1460.
Y_ZERO,Y_MAX = 1870.,370.
xs = np.linspace(X0,X1,n)
ys = Y_ZERO-(values-scale['minimum'])/(scale['maximum']-scale['minimum'])*(Y_ZERO-Y_MAX)
P = np.column_stack([xs,ys])
roof_shift = np.array([-135.,210.])
Q = P + roof_shift
wall_top = Q + np.array([0.,8.])
wall_bottom = wall_top + np.array([0.,255.])

rng = np.random.default_rng(402)
base = np.empty((H*S,W*S,3),dtype=np.uint8)
noise = rng.normal(0,0.55,(H*S,W*S)).astype(np.float32)
for k,v in enumerate((246,241,231)):
 base[:,:,k] = np.clip(v+noise,0,255).astype(np.uint8)
canvas = Image.fromarray(base,'RGB').convert('RGBA')
del base,noise
house_mask = Image.new('L',canvas.size,0)
roof_mask = Image.new('L',canvas.size,0)
D = ImageDraw.Draw(canvas)

def pts(seq): return [(round(float(x)*S),round(float(y)*S)) for x,y in seq]
def line(seq, fill=INK, width=2):
 D.line(pts(seq),fill=fill,width=max(1,round(width*S)),joint='curve')
def polygon(seq,fill,outline=None,width=2):
 p=pts(seq);D.polygon(p,fill=fill)
 if outline: D.line(p+[p[0]],fill=outline,width=round(width*S),joint='curve')
def font(size,style='regular'):
 name={'regular':'NimbusSans-Regular.otf','bold':'NimbusSans-Bold.otf','condensed':'NimbusSansNarrow-Bold.otf'}[style]
 return ImageFont.truetype(str(ROOT/'assets/fonts'/name),round(size*S))
def text(x,y,s,size=26,fill=INK,style='regular',anchor='lt'):
 D.text((round(x*S),round(y*S)),s,font=font(size,style),fill=fill,anchor=anchor)
def tracked(x,y,s,size=21,spacing=2,fill=INK,style='bold'):
 f=font(size,style)
 for c in s:
  D.text((round(x*S),round(y*S)),c,font=f,fill=fill,anchor='lt')
  x += D.textlength(c,font=f)/S+spacing

def paste_asset(name,x,y,width=None,height=None):
 global D
 im=Image.open(ROOT/'assets'/name).convert('RGBA')
 if height is not None: width=height*im.width/im.height
 else: height=width*im.height/im.width
 im=im.resize((round(width*S),round(height*S)),Image.Resampling.LANCZOS)
 canvas.alpha_composite(im,(round(x*S),round(y*S)))
 D=ImageDraw.Draw(canvas)
 return [x,y,x+width,y+height]

def fill_texture_polygon(name,poly,mask_acc=None):
 global D
 tex=Image.open(ROOT/'assets'/name).convert('RGBA')
 box=(int(min(x for x,y in poly)*S)-2,int(min(y for x,y in poly)*S)-2,
      int(max(x for x,y in poly)*S)+3,int(max(y for x,y in poly)*S)+3)
 mask=Image.new('L',(box[2]-box[0],box[3]-box[1]),0)
 ImageDraw.Draw(mask).polygon([(round(x*S)-box[0],round(y*S)-box[1]) for x,y in poly],fill=255)
 tex=tex.resize(mask.size,Image.Resampling.BICUBIC)
 canvas.paste(tex,(box[0],box[1]),mask)
 if mask_acc is not None: mask_acc.paste(255,(box[0],box[1],box[2],box[3]),mask)
 D=ImageDraw.Draw(canvas)

def warp_strip(name,top,offset,mask_acc,extra_mask=None):
 """Piecewise affine strip: common node coordinates, with no data-edge perspective."""
 global D
 src=np.asarray(Image.open(ROOT/'assets'/name).convert('RGB'))
 sh,sw=src.shape[:2]
 for i in range(len(top)-1):
  a,b=top[i],top[i+1]
  poly=np.array([a,b,b+offset,a+offset])
  box=(max(0,int(np.floor(poly[:,0].min()*S))-1),max(0,int(np.floor(poly[:,1].min()*S))-1),
       min(W*S,int(np.ceil(poly[:,0].max()*S))+2),min(H*S,int(np.ceil(poly[:,1].max()*S))+2))
  yy,xx=np.mgrid[box[1]:box[3],box[0]:box[2]]/S
  inv=np.linalg.inv(np.column_stack([b-a,offset]))
  local=np.einsum('ij,jkl->ikl',inv,np.stack([xx-a[0],yy-a[1]]))
  u=(i+np.clip(local[0],0,1))/(len(top)-1)*(sw-1)
  v=np.clip(local[1],0,1)*(sh-1)
  sampled=np.empty((*u.shape,3),dtype=np.uint8)
  for c in range(3):
   sampled[:,:,c]=map_coordinates(src[:,:,c],[v,u],order=1,mode='nearest')
  m=Image.new('L',(box[2]-box[0],box[3]-box[1]),0)
  ImageDraw.Draw(m).polygon([(round(x*S)-box[0],round(y*S)-box[1]) for x,y in poly],fill=255)
  canvas.paste(Image.fromarray(sampled),(box[0],box[1]),m)
  mask_acc.paste(255,box,m)
  if extra_mask is not None: extra_mask.paste(255,box,m)
 D=ImageDraw.Draw(canvas)

# Page typography is entirely code-rendered.
tracked(106,108,'AN ILLUSTRATIVE HOUSING SERIES',21,1.5)
D.rounded_rectangle((1410*S,99*S,1650*S,139*S),radius=2*S,outline=RUST,width=2*S)
text(1530,120,data['visible_notice'],20,RUST,'bold','mm')
text(103,168,brief['title_zh'].upper(),91,INK,'condensed')
text(106,275,'New-home prices, across eleven equally spaced years',30)
line([(106,331),(1650,331)],'#afa69a',1)
tracked(106,369,'YEAR',19,1.5)
tracked(272,369,'PRICE / USD',19,1.4)

# Green ground and an elongated stippled ground shadow.
ground_top=1730
D.rectangle((73*S,ground_top*S,1687*S,1865*S),fill='#a5ad62')
for _ in range(13000):
 x=rng.uniform(74,1686);y=rng.uniform(ground_top+2,1865)
 col=('#939c54' if rng.random()<0.55 else '#b9bf7d')
 r=rng.choice([.4,.6,.8]);D.ellipse(((x-r)*S,(y-r)*S,(x+r)*S,(y+r)*S),fill=col)
shadow_poly=[(337,1807),(1370,1744),(1655,1777),(1608,1820),(700,1854),(350,1835)]
polygon(shadow_poly,'#6e7549')
for _ in range(7500):
 x=rng.uniform(360,1610);y=rng.uniform(1772,1840)
 # Long organic shadow, denser beneath the low end of the house.
 if ((x-1030)/685)**2+((y-1807)/39)**2 <1:
  r=rng.uniform(.5,1.1);D.ellipse(((x-r)*S,(y-r)*S,(x+r)*S,(y+r)*S),fill='#454c36')

# End wall and dark under-floor plane join the same house geometry.
last=P[-1]; qlast=Q[-1]; back=last+np.array([170.,190.])
endpoly=[last,back,back+[0.,255.],wall_bottom[-1],qlast]
fill_texture_polygon('endwall_texture.png',endpoly,house_mask)
line(endpoly+[endpoly[0]],INK,3.6)
floor_lower=wall_bottom+np.array([12.,24.])
floorpoly=list(wall_bottom)+list(floor_lower[::-1])
polygon(floorpoly,'#353439',INK,2.5)
ImageDraw.Draw(house_mask).polygon(pts(floorpoly),fill=255)
for i in range(n-1):
 a,b=wall_bottom[i],wall_bottom[i+1]
 for t in np.linspace(0,1,13):
  c=a+(b-a)*t
  for v in [7,15,22]:
   x,y=c+[v*.5,v];D.ellipse(((x-.8)*S,(y-.8)*S,(x+.8)*S,(y+.8)*S),fill='#928d92')

# Generated facade and roof are warped as connected piecewise-affine surfaces.
warp_strip('front_facade.png',wall_top,np.array([0.,255.]),house_mask)
line(list(wall_bottom),INK,3.4)
line([wall_top[0],wall_bottom[0]],INK,3.4)
line([wall_top[-1],wall_bottom[-1]],INK,3.4)
warp_strip('roof_texture.png',P,roof_shift,house_mask,roof_mask)
line([P[0],Q[0]],INK,3.8)
line(list(Q),'#29272a',5)
line(list(Q+[0.,5.5]),'#f4eddf',4)
line([last,qlast],INK,3.5)

# One small annex is connected in front of the end wall.
annex_box=paste_asset('annex.png',1442,1589,width=228)
# A simple attic louver gives the shaded gable a readable architectural scale.
polygon([(1474,1429),(1533,1489),(1533,1538),(1474,1478)],'#615e68',INK,2)
for yy in [1439,1450,1461,1472]:
 line([(1482,yy),(1525,yy+44)],'#b8b0b8',2)

# Guides use the same y coordinate as the data boundary; thin ivory casing is
# clipped to the building so red lines remain visible over generated materials.
casing=Image.new('RGBA',canvas.size,(0,0,0,0));cd=ImageDraw.Draw(casing)
for x,y in P:
 cd.line(pts([(435,y),(x,y)]),fill=(249,243,232,220),width=round(4.3*S))
casing.putalpha(ImageChops.multiply(casing.getchannel('A'),house_mask))
canvas.alpha_composite(casing);D=ImageDraw.Draw(canvas)
for category,value,(x,y) in zip(categories,values,P):
 line([(435,y),(x,y)],RUST,1.4)
 text(106,y,category,31,INK,'bold','lm')
 label=data['number_format']['prefix']+f'{value:,.0f}'
 text(272,y,label,34,RUST,'bold','lm')
# The actual roof edge is one continuous data polyline, overprinted in black.
line(list(P),INK,4.6)
for x,y in P:
 r=6.4;D.ellipse(((x-r)*S,(y-r)*S,(x+r)*S,(y+r)*S),fill=INK)

# Short reading key occupies the blank space below the left annotation column.
tracked(106,1447,'THE ROOF IS THE LINE',22,1.2)
text(106,1495,'Each black node marks one year.',25)
text(106,1531,'Price uses a linear $0–$70,000 scale.',25)
text(106,1567,'Years are equally spaced from left to right.',25)

# Three people and one dog share the ground with two shrub clusters.
family_box=paste_asset('family_dog.png',873,1643,height=190)
shrub_left_box=paste_asset('shrub_wide.png',99,1720,width=320)
shrub_right_box=paste_asset('shrub_tall.png',1443,1738,width=214)

# Border and compact source note finish the printed page.
D.rectangle((72*S,66*S,1688*S,1866*S),outline=INK,width=round(2.6*S))
text(106,1902,'SOURCE: SUPPLIED SYNTHETIC DATASET',20,INK,'bold')
text(106,1935,'Illustrative values only. Roof geometry and labels follow the supplied data.',22,'#5e5b54')
text(1650,1902,'USD',20,INK,'bold','rt')

final=canvas.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
final.save(ROOT/'final.png',compress_level=7)
roof_mask.resize((W,H),Image.Resampling.LANCZOS).save(OUT/'roof_mask.png')
house_mask.resize((W,H),Image.Resampling.LANCZOS).save(OUT/'house_mask.png')
geometry={
 'canvas':[W,H], 'mapping':{'x_start':X0,'x_end':X1,'equal_step':float(xs[1]-xs[0]),
 'y_at_scale_minimum':Y_ZERO,'y_at_scale_maximum':Y_MAX,'scale_minimum':scale['minimum'],'scale_maximum':scale['maximum']},
 'nodes':[{'category':cat,'value_usd':int(v),'x':float(x),'y':float(y),'leader_start_x':435.,'label':f'${v:,.0f}'} for cat,v,(x,y) in zip(categories,values,P)],
 'roof_near_edge':Q.tolist(),'roof_fill_polygon':P.tolist()+Q[::-1].tolist(),
 'facade_top':wall_top.tolist(),'facade_bottom':wall_bottom.tolist(),
 'asset_bounding_boxes':{'family_and_dog':family_box,'annex':annex_box,'left_shrub':shrub_left_box,'right_shrub':shrub_right_box},
 'minimum_label_row_spacing_px':float(np.min(np.diff(ys))),
 'counts':{'data_nodes':n,'human_figures':3,'dogs':1,'main_windows':3,'main_doors':1,'annexes':1},
 'generated_surface_assets':['assets/roof_texture.png','assets/front_facade.png','assets/endwall_texture.png']}
(OUT/'geometry.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(f'Saved final.png: {W} x {H}; {n} data-bound roof nodes.')
