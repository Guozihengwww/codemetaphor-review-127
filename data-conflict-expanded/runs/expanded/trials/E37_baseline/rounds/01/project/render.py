#!/usr/bin/env python3
"""Offline photographic food chart. Run through capture_render.py for recorded renders."""
from __future__ import annotations
from pathlib import Path
from functools import lru_cache
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.ndimage import gaussian_filter, map_coordinates, distance_transform_edt

ROOT = Path(__file__).resolve().parent
INPUT = json.loads((ROOT/'input.json').read_text(encoding='utf-8'))
BRIEF = json.loads((ROOT/INPUT['brief']).read_text(encoding='utf-8'))
DATA = json.loads((ROOT/INPUT['data']).read_text(encoding='utf-8'))
CORE = json.loads((ROOT/INPUT['critical_metaphor_contract']).read_text(encoding='utf-8'))
assert (ROOT/INPUT['reference']).is_file()  # Frozen design reference retained; never sampled.
W, H = BRIEF['width'], BRIEF['height']
S = 2
ASSETS = ROOT/'assets'
ART = ROOT/'artifacts'
ART.mkdir(exist_ok=True)
C = (255, 245, 219, 255)
GREEN = (48, 77, 24, 255)
INK = (72, 62, 34, 255)
TIMES = np.asarray(DATA['time_points'], dtype=float)
MIN_V = float(DATA['value_axis']['minimum'])
MAX_V = float(DATA['value_axis']['maximum'])
X0, X1, Y0, Y1 = 330., 1740., 410., 1058.

def to_x(t): return X0+(np.asarray(t)-TIMES[0])/(TIMES[-1]-TIMES[0])*(X1-X0)
def to_y(v): return Y1-(np.asarray(v)-MIN_V)/(MAX_V-MIN_V)*(Y1-Y0)
xs = to_x(TIMES)
series = {v['id']:v for v in DATA['series']}
points = {k:np.c_[xs,to_y(v['values'])] for k,v in series.items()}
labels = []

@lru_cache(None)
def font(size, name='sans_regular'):
    return ImageFont.truetype(str(ASSETS/'fonts'/f'{name}.otf'), round(size*S))

def coords(box): return tuple(round(v*S) for v in box)
def line(d, xy, fill, width=1):
    d.line([(round(x*S),round(y*S)) for x,y in xy], fill=fill, width=max(1,round(width*S)))
def text(d, xy, value, size, fill=C, name='sans_regular', anchor='lt'):
    d.text((round(xy[0]*S),round(xy[1]*S)),value,font=font(size,name),fill=fill,anchor=anchor)

def measured_text(d, xy, value, size, fill=C, name='sans_regular', anchor='lt', role=None):
    bbox=d.textbbox((xy[0]*S,xy[1]*S),value,font=font(size,name),anchor=anchor)
    text(d,xy,value,size,fill,name,anchor)
    if role:
        labels.append({'text':value,'role':role,'bbox':[round(z/S,2) for z in bbox]})

def alpha_shadow(alpha, color, blur, offset):
    sh=Image.new('RGBA',(W*S,H*S),(0,0,0,0))
    local=Image.new('RGBA',alpha.size,color)
    local.putalpha(alpha.point(lambda p:round(p*color[3]/255)))
    sh.alpha_composite(local,(round(offset[0]*S),round(offset[1]*S)))
    return sh.filter(ImageFilter.GaussianBlur(blur*S))

# A reproducible matte red tabletop: gentle illumination, fine grain, soft wear.
rng=np.random.default_rng(37)
yy,xx=np.mgrid[0:H,0:W]
light=np.exp(-(((xx-630)/1450)**2+((yy-300)/1050)**2))
coarse=gaussian_filter(rng.normal(size=(H,W)),25)
coarse/=coarse.std()
fine=rng.normal(0,.9,(H,W))
base=np.stack([178+18*light,43+17*light,51+13*light],axis=2)
base+=coarse[:,:,None]*.55+fine[:,:,None]
bg=Image.fromarray(np.uint8(np.clip(base,0,255)),'RGB').convert('RGBA')
canvas=bg.resize((W*S,H*S),Image.Resampling.BICUBIC)
d=ImageDraw.Draw(canvas)

# Headline and series key, all typography is executable and data-bound.
d.rounded_rectangle(coords((120,57,390,102)),radius=22*S,fill=C)
text(d,(139,68),DATA['visible_notice'],22,(163,43,49,255),'sans_bold')
text(d,(1880,69),'1990—2009',25,C,'sans_regular','rt')
measured_text(d,(116,137),'Voting and non-voting shares',102,C,'sans_bold',role='topic_title')
text(d,(120,257),'Party vote share and the population not voting',31)
text(d,(1880,263),'SHARE IN PERCENT',22,C,'sans_bold','rt')
# Small material keys repeat the actual central encodings.
chive_key=Image.open(ASSETS/'chive_surface.png').convert('RGB').rotate(90,expand=True)
chive_key=chive_key.resize((70*S,11*S),Image.Resampling.LANCZOS).convert('RGBA')
canvas.alpha_composite(chive_key,(120*S,325*S))
text(d,(208,317),series['A']['label'],27)
cheese_key=Image.open(ASSETS/'cheese_surface.png').convert('RGBA').resize((62*S,23*S),Image.Resampling.LANCZOS)
canvas.alpha_composite(cheese_key,(1090*S,320*S))
text(d,(1170,317),series['B']['label'],27)
line(d,[(120,373),(1880,373)],(255,238,214,100),1)

# Shared value axis. Grid sits underneath the actual food, not above it.
axis=Image.new('RGBA',canvas.size)
da=ImageDraw.Draw(axis)
text(da,(120,408),'%',26,C,'sans_bold')
line(da,[(203,Y0),(203,Y1)],(255,238,214,92),1.2)
for value in [42,40,35,30,25,20,15]:
    y=float(to_y(value))
    text(da,(181,y),str(value),25,C,'sans_regular','rm')
    line(da,[(195,y),(214,y)],(255,241,214,148),1.2)
    line(da,[(226,y),(1855,y)],(255,239,211,29),.7)
canvas=Image.alpha_composite(canvas,axis)

# Buttered rustic bread is the base of the fused chart.
bread=Image.open(ASSETS/'bread_butter.png').convert('RGBA')
bread=bread.resize((1690*S,698*S),Image.Resampling.LANCZOS)
bread_full=Image.new('RGBA',canvas.size)
bread_full.alpha_composite(bread,(210*S,403*S))
canvas=Image.alpha_composite(canvas,alpha_shadow(bread_full.getchannel('A'),(63,18,23,117),13,(6,13)))
canvas=Image.alpha_composite(canvas,bread_full)

# The cheese's TRUE lower boundary is B. Its top is explicitly non-quantitative.
CHEESE_TOP=628.
b=points['B']
polygon=[(X0,CHEESE_TOP),(X1,CHEESE_TOP)]+[tuple(p) for p in b[::-1]]
cheese_mask=Image.new('L',canvas.size,0)
dm=ImageDraw.Draw(cheese_mask)
dm.polygon([(round(x*S),round(y*S)) for x,y in polygon],fill=255)
canvas=Image.alpha_composite(canvas,alpha_shadow(cheese_mask,(74,48,22,93),5.2,(1,7)))
# Reuse generated photographic marbling and pores, only clipping geometry by code.
texture=Image.open(ASSETS/'cheese_surface.png').convert('RGB')
rect_w=int(round((X1-X0)*S))+1
rect_h=int(round((max(b[:,1])-CHEESE_TOP)*S))+1
texture=texture.resize((rect_w,rect_h),Image.Resampling.LANCZOS)
arr=np.asarray(texture,dtype=np.float32).copy()
ry,rx=np.mgrid[0:rect_h,0:rect_w]
# Gentle face lighting remains subordinate to the source surface texture.
shade=1.005-.025*(ry/max(1,rect_h-1))+.009*np.cos(rx/rect_w*np.pi)
arr=np.uint8(np.clip(arr*shade[:,:,None],0,255))
cheese=Image.new('RGBA',canvas.size)
cheese.paste(Image.fromarray(arr).convert('RGBA'),(round(X0*S),round(CHEESE_TOP*S)))
cheese.putalpha(cheese_mask)
# A narrow internal cut-face edge; nothing extends the data-bearing contour.
edge=Image.new('RGBA',canvas.size)
de=ImageDraw.Draw(edge)
line(de,[tuple(p) for p in b],(135,111,47,153),2.4)
line(de,[(X0,CHEESE_TOP),(X1,CHEESE_TOP)],(255,249,196,208),1.7)
edge.putalpha(Image.fromarray(np.minimum(np.array(edge.getchannel('A')),np.array(cheese_mask))))
cheese=Image.alpha_composite(cheese,edge)
canvas=Image.alpha_composite(canvas,cheese)
cheese_mask.resize((W,H),Image.Resampling.LANCZOS).save(ART/'cheese_geometry_mask.png')

# Map the generated chive's fiber and transverse lighting along all five segments.
# Nearest-segment coordinates produce one continuous, fixed-width tubular leaf.
a=points['A']
margin=18
bx0=int(np.floor(a[:,0].min()-margin)); by0=int(np.floor(a[:,1].min()-margin))
bx1=int(np.ceil(a[:,0].max()+margin)); by1=int(np.ceil(a[:,1].max()+margin))
gy,gx=np.mgrid[by0*S:by1*S,bx0*S:bx1*S]/S
best=np.full(gx.shape,np.inf,dtype=np.float32)
trans=np.zeros(gx.shape,dtype=np.float32)
along=np.zeros(gx.shape,dtype=np.float32)
accum=0.
for p,q in zip(a[:-1],a[1:]):
    v=q-p; length=np.linalg.norm(v)
    t=((gx-p[0])*v[0]+(gy-p[1])*v[1])/length**2
    tc=np.clip(t,0,1)
    dx=gx-(p[0]+tc*v[0]); dy=gy-(p[1]+tc*v[1])
    dist=np.sqrt(dx*dx+dy*dy)
    take=dist<best
    best[take]=dist[take]
    signed=((gx-p[0])*(-v[1])+(gy-p[1])*v[0])/length
    trans[take]=signed[take]
    along[take]=accum+tc[take]*length
    accum+=length
radius=7.
alpha=np.clip((radius-best)*S+.5,0,1)
source=np.array(Image.open(ASSETS/'chive_surface.png').convert('RGB'),dtype=np.float32)
sh,sw=source.shape[:2]
u=np.clip((trans+radius)/(radius*2)*(sw-1),0,sw-1)
v=np.mod(along/accum*sh*2.2,sh-1)
chive_rgb=np.stack([map_coordinates(source[:,:,c],[v,u],order=1,mode='nearest') for c in range(3)],axis=2)
chive_rgba=np.dstack([np.clip(chive_rgb*1.08,0,255),alpha*255]).astype('uint8')
chive=Image.new('RGBA',canvas.size)
chive.alpha_composite(Image.fromarray(chive_rgba),(bx0*S,by0*S))
canvas=Image.alpha_composite(canvas,alpha_shadow(chive.getchannel('A'),(34,40,13,150),2.2,(1.2,3.5)))
canvas=Image.alpha_composite(canvas,chive)
chive.getchannel('A').resize((W,H),Image.Resampling.LANCZOS).save(ART/'chive_geometry_mask.png')

# Value callouts use leaders to bind every number to its actual food contour.
d=ImageDraw.Draw(canvas)
def value_tag(point, value, kind, shift=0):
    x,y=map(float,point)
    label=f'{value:.1f}%'
    cx=x+shift
    cy=y-(49 if kind=='A' else 41)
    f=font(32,'sans_bold')
    width=d.textlength(label,font=f)/S+24
    height=46
    box=(cx-width/2,cy-height/2,cx+width/2,cy+height/2)
    leader=(244,246,211,228) if kind=='A' else (104,86,38,180)
    line(d,[(cx,box[3]+1),(x,y-5)],leader,1.2)
    if kind=='A':
        d.rounded_rectangle(coords(box),radius=22*S,fill=GREEN)
        measured_text(d,(cx,cy+1),label,32,C,'sans_bold','mm',role=f'A_value_{value}')
        d.ellipse(coords((x-3.6,y-3.6,x+3.6,y+3.6)),fill=(217,230,156,255),outline=GREEN,width=2*S)
    else:
        d.rounded_rectangle(coords(box),radius=22*S,fill=(255,250,218,245))
        measured_text(d,(cx,cy+1),label,32,INK,'sans_bold','mm',role=f'B_value_{value}')
        d.ellipse(coords((x-2.3,y-2.3,x+2.3,y+2.3)),fill=(92,76,30,255))
    labels.append({'role':f'{kind}_callout','text':label,'bbox':[round(v,2) for v in box],'point':[round(x,3),round(y,3)]})
for p,v in zip(a,series['A']['values']): value_tag(p,v,'A')
for i,(p,v) in enumerate(zip(b,series['B']['values'])):
    shift=53 if i==0 else -57 if i==len(b)-1 else 0
    value_tag(p,v,'B',shift)

# Time positions use the actual 4, 4, 4, 3 and 4 year intervals.
for x,category in zip(xs,DATA['categories']):
    line(d,[(x,1103),(x,1116)],(255,240,214,155),1.5)
    measured_text(d,(float(x),1132),category,32,C,'sans_regular','mt',role='year')
text(d,(120,1137),'YEAR',19,C,'sans_bold')
line(d,[(120,1200),(1880,1200)],(255,237,215,110),1)
text(d,(120,1225),'Chive centreline = votes',25,C,'sans_bold')
text(d,(690,1225),'Cheese lower edge = non-voters',25,C,'sans_bold')
text(d,(1880,1227),'SHARED LINEAR SCALE: 15–42%',22,C,'sans_bold','rt')
text(d,(120,1271),'Years are spaced to time. Cheese area, thickness and upper edge are not data.',22)
text(d,(1880,1271),'Synthetic figures • not observed election results',22,C,'sans_regular','rt')

final=canvas.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
final.save(ROOT/'final.png')
geometry={
 'size':[W,H], 'scale':{'x_year_range':[float(TIMES[0]),float(TIMES[-1])], 'x_pixel_range':[X0,X1],
 'y_value_range':[MIN_V,MAX_V], 'y_pixel_range':[Y1,Y0], 'pixels_per_percentage_point':(Y1-Y0)/(MAX_V-MIN_V)},
 'data_points':{k:[{'category':category,'value':float(value),'x':round(float(p[0]),6),'y':round(float(p[1]),6)} for category,value,p in zip(DATA['categories'],series[k]['values'],ps)] for k,ps in points.items()},
 'cheese_top_y':CHEESE_TOP, 'chive_radius_px':radius,'labels':labels,
 'data_binding':'A: centreline of textured chive; B: real lower boundary of textured cheese; straight interpolation; true time spacing.',
 'source_images':['assets/bread_butter.png','assets/cheese_surface.png','assets/chive_surface.png']
}
(ART/'render_geometry.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Saved final.png (2000 x 1320); 6 years, 12 value labels, 2 data-shaped food components.')

if __name__=='__main__':
    pass
