#!/usr/bin/env python3
"""Deterministic food-metaphor chart. Run from this directory; no network/model calls."""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.ndimage import map_coordinates, gaussian_filter

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT/'data.json').read_text(encoding='utf-8'))
BRIEF = json.loads((ROOT/'brief.json').read_text(encoding='utf-8'))
W,H = BRIEF['width'],BRIEF['height']
S = 2
SIZE = (W*S,H*S)
X0,X1,YTOP,YBOTTOM = 300.,1580.,330.,1060.
VMIN,VMAX = DATA['value_axis']['minimum'],DATA['value_axis']['maximum']
times = np.asarray(DATA['time_points'],dtype=float)
xx = X0 + (times-times[0])/(times[-1]-times[0])*(X1-X0)
series = {s['id']:s for s in DATA['series']}
def yval(v): return YBOTTOM-(np.asarray(v)-VMIN)/(VMAX-VMIN)*(YBOTTOM-YTOP)
ya,yb = yval(series['A']['values']), yval(series['B']['values'])
PA=list(zip(xx,ya)); PB=list(zip(xx,yb))
CJK = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
CJK_BOLD = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
LAT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
LAT_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
IVORY=(255,247,224,255)

def font(n,bold=False,cjk=False):
    return ImageFont.truetype((CJK_BOLD if bold else CJK) if cjk else (LAT_BOLD if bold else LAT),round(n*S))
def box(coords): return tuple(round(v*S) for v in coords)
def line(draw,points,fill,width=1): draw.line([(round(x*S),round(y*S)) for x,y in points],fill=fill,width=max(1,round(width*S)))
def text(draw,xy,txt,size=28,fill=IVORY,bold=False,cjk=False,anchor='lt'):
    draw.text((round(xy[0]*S),round(xy[1]*S)),txt,font=font(size,bold,cjk),fill=fill,anchor=anchor)
def shadow_of(mask,offset=(0,9),blur=10,opacity=85):
    sh=Image.new('RGBA',SIZE,(40,9,2,0)); a=Image.new('L',SIZE,0)
    a.paste(mask,(round(offset[0]*S),round(offset[1]*S)))
    a=a.filter(ImageFilter.GaussianBlur(blur*S)).point(lambda q:round(q*opacity/255))
    sh.putalpha(a); return sh

# Quiet red tabletop with fine natural-looking grain, generated deterministically.
rng=np.random.default_rng(2718)
grain=rng.normal(0,1.25,(H,W)).astype(np.float32)
cloud=gaussian_filter(rng.normal(0,1,(H,W)).astype(np.float32),40)
cloud=cloud/(cloud.std()+1e-8)
ygrid,xgrid=np.mgrid[0:H,0:W]
light=1.9*np.exp(-((xgrid-680)**2/(W*.64)**2+(ygrid-350)**2/(H*.9)**2))
base=np.array([194,52,48],dtype=np.float32)
bg=np.clip(base+grain[:,:,None]+cloud[:,:,None]*.60+light[:,:,None]*np.array([3.5,2.0,1.9]),0,255).astype('uint8')
canvas=Image.fromarray(bg).convert('RGBA').resize(SIZE,Image.Resampling.BICUBIC)
d=ImageDraw.Draw(canvas)

# Hierarchy and all visible titles are code-set.
text(d,(100,61),'FOOD / DATA',20,bold=True)
text(d,(100,105),DATA['title'],64,bold=True,cjk=True)
text(d,(104,203),'1990—2009  ·  两系列百分比走势',27,cjk=True)
d.rounded_rectangle(box((1560,63,1900,116)),radius=26*S,outline=(255,235,209,220),width=2*S)
text(d,(1730,90),DATA['visible_notice'],22,bold=True,anchor='mm')
text(d,(1900,143),'合成数据 · 食物隐喻',20,fill=(255,230,205,230),cjk=True,anchor='rt')
line(d,[(100,265),(1900,265)],(255,225,206,75),1)

# A shared linear value axis; subtle guides lie behind the food.
axisx=190
text(d,(axisx,289),'百分比 / %',23,cjk=True,anchor='mt')
line(d,[(axisx,YTOP),(axisx,YBOTTOM)],(255,235,211,170),1.4)
for v in [15,20,25,30,35,40,42]:
    yy=float(yval(v))
    line(d,[(axisx-8,yy),(axisx+10,yy)],(255,240,220,200),1.5)
    text(d,(axisx-23,yy),str(v),23,fill=(255,238,216,235),anchor='rm')
    if v in [15,20,25,30,35,40]:
        for start in np.arange(axisx+22,X1+1,17):
            line(d,[(start,yy),(min(start+4,X1),yy)],(255,239,218,28),.8)

# Photographic bread and butter are a substantial physical support layer.
bread=Image.open(ROOT/'assets/bread_butter.png').convert('RGBA').resize((1450*S,600*S),Image.Resampling.LANCZOS)
bread_layer=Image.new('RGBA',SIZE,(0,0,0,0)); bread_layer.alpha_composite(bread,(230*S,480*S))
canvas=Image.alpha_composite(canvas,shadow_of(bread_layer.getchannel('A'),(3,16),15,108))
canvas=Image.alpha_composite(canvas,bread_layer)

# The cheese's physical lower boundary is the straight-segment B polyline.
# Its top edge is an explicitly nondata boundary above every B value.
cheese_top=600.
cheese_poly=[(X0,cheese_top),(X1,cheese_top)]+PB[::-1]
cmask=Image.new('L',SIZE,0); md=ImageDraw.Draw(cmask)
md.polygon([(round(x*S),round(y*S)) for x,y in cheese_poly],fill=255)
ctex=Image.open(ROOT/'assets/cheese_surface.png').convert('RGB').resize((round((X1-X0)*S)+1,round((float(max(yb))-cheese_top)*S)+3),Image.Resampling.LANCZOS)
cheese=Image.new('RGBA',SIZE,(0,0,0,0)); cheese.paste(ctex,(round(X0*S),round(cheese_top*S)))
cheese.putalpha(cmask)
# A shallow bevel is entirely inside the data boundary, so the outer cut is exact.
arr=np.array(cheese)
xs=np.arange(round(X0*S),round(X1*S)+1)/S
edge=np.interp(xs,xx,yb)
ys=np.arange(round(cheese_top*S),round(max(yb)*S)+2)/S
inside_depth=edge[None,:]-ys[:,None]
bevel=np.clip(1-inside_depth/6.,0,1)
shade=1-.19*bevel
sl=arr[round(cheese_top*S):round(cheese_top*S)+len(ys),round(X0*S):round(X0*S)+len(xs),:3]
sl[:]=np.clip(sl.astype(np.float32)*shade[:,:,None],0,255).astype(np.uint8)
cheese=Image.fromarray(arr)
cd=ImageDraw.Draw(cheese)
line(cd,[(X0+2,cheese_top+1.1),(X1-2,cheese_top+1.1)],(255,251,215,150),1.6)
canvas=Image.alpha_composite(canvas,shadow_of(cmask,(0,7),5,112))
canvas=Image.alpha_composite(canvas,cheese)

# Normalize the generated stem's cross section, then bend its photographic texture
# along the exact straight-segment A path. Radius does not encode a value.
chive=np.array(Image.open(ROOT/'assets/chive.png').convert('RGBA'))
sh,sw=chive.shape[:2]
tex=np.zeros((64,sw,3),dtype=np.uint8)
for j in range(sw):
    good=np.flatnonzero(chive[:,j,3]>160)
    if len(good)<2:
        tex[:,j,:]=(62,105,15)
    else:
        ys0=np.linspace(good[0],good[-1],64)
        for c in range(3): tex[:,j,c]=np.interp(ys0,np.arange(sh),chive[:,j,c])
# Work only in the stem's tightly bounded local rectangle.
radius=8.2
left=int(min(xx)-radius-3); right=int(max(xx)+radius+4)
top=int(min(ya)-radius-3); bottom=int(max(ya)+radius+4)
py,px=np.mgrid[top*S:bottom*S,left*S:right*S].astype(np.float32)/S
best=np.full(px.shape,np.inf,dtype=np.float32)
ubest=np.zeros_like(px); vbest=np.zeros_like(px)
lengths=np.hypot(np.diff(xx),np.diff(ya)); total=lengths.sum(); startdist=0.
for i,L in enumerate(lengths):
    ax,ay=PA[i]; bx,by=PA[i+1]; dx=bx-ax; dy=by-ay
    t=np.clip(((px-ax)*dx+(py-ay)*dy)/(L*L),0,1)
    qx=ax+t*dx; qy=ay+t*dy
    distance=np.hypot(px-qx,py-qy)
    signed=((py-qy)*dx-(px-qx)*dy)/L
    use=distance<best
    ubest[use]=(startdist+t[use]*L)/total
    vbest[use]=signed[use]
    best[use]=distance[use]
    startdist+=L
coords=np.array([np.clip((vbest/radius*.5+.5)*63,0,63),np.clip(ubest*(sw-21)+10,0,sw-1)])
color=np.stack([map_coordinates(tex[:,:,c].astype(float),coords,order=1,mode='nearest') for c in range(3)],axis=2)
# Gentle edge darkening, retaining the generated longitudinal highlights.
edge_shade=1-.08*np.clip(np.abs(vbest)/radius-.65,0,.35)/.35
color=np.clip(color*edge_shade[:,:,None],0,255).astype(np.uint8)
a=(np.clip((radius+.25-best)*S,0,1)*255).astype(np.uint8)
stem=Image.fromarray(np.dstack([color,a]),'RGBA')
stemlayer=Image.new('RGBA',SIZE,(0,0,0,0)); stemlayer.alpha_composite(stem,(left*S,top*S))
canvas=Image.alpha_composite(canvas,shadow_of(stemlayer.getchannel('A'),(0,4.0),2.5,118))
canvas=Image.alpha_composite(canvas,stemlayer)

# Code-set labels attach directly to their material boundaries.
d=ImageDraw.Draw(canvas)
def value_tag(x,y,value,kind):
    cy=y-45 if kind=='A' else y-38
    tw=94; th=42
    fill=(87,111,42,245) if kind=='A' else (255,249,223,246)
    ink=IVORY if kind=='A' else (51,49,33,255)
    d.rounded_rectangle(box((x-tw/2,cy-th/2,x+tw/2,cy+th/2)),radius=11*S,fill=fill)
    text(d,(x,cy),f'{value:.1f}',29,fill=ink,anchor='mm')
    line(d,[(x,cy+th/2),(x,y-(10 if kind=='A' else 1))],fill,1.5)
for i in range(len(xx)-1):
    value_tag(float(xx[i]),float(ya[i]),series['A']['values'][i],'A')
    value_tag(float(xx[i]),float(yb[i]),series['B']['values'][i],'B')

# Direct endpoint labels remain aligned with the final data points.
for sid,yy,desc in [('B',float(yb[-1]),'奶酪下边缘'),('A',float(ya[-1]),'葱茎中心线')]:
    col=(255,242,196,255) if sid=='B' else (183,207,105,255)
    for lx in range(round(X1)+18,1655,13):
        line(d,[(lx,yy),(lx+6,yy)],col,2)
    text(d,(1682,yy-43),series[sid]['label'],28,bold=True)
    text(d,(1680,yy-2),f"{series[sid]['values'][-1]:.1f}%",42,bold=True)
    text(d,(1683,yy+51),desc,23,fill=(255,232,209,240),cjk=True)

# True year spacing, rather than categorical equal spacing.
for t,x,cat in zip(times,xx,DATA['categories']):
    line(d,[(x,1095),(x,1108)],(255,236,216,210),1.5)
    text(d,(x,1128),cat,30,anchor='mt')
text(d,(1682,1133),'年份',23,cjk=True)
line(d,[(100,1200),(1900,1200)],(255,228,206,85),1)
text(d,(100,1230),'读取方式',22,bold=True,cjk=True)
text(d,(255,1230),'A · 葱茎中心线    B · 奶酪下边缘',24,cjk=True)
text(d,(100,1270),'共用 15–42% 线性轴；年份按真实间距排列。',20,fill=(255,229,206,235),cjk=True)
text(d,(1900,1230),'食材面积与厚度不编码',22,cjk=True,anchor='rt')
text(d,(1900,1270),'数据来源：本案 data.json（合成）',20,fill=(255,229,206,235),cjk=True,anchor='rt')

# The final image is produced only by the frozen wrapper's invocation of this file.
canvas.convert('RGB').resize((W,H),Image.Resampling.LANCZOS).save(ROOT/'final.png')
# Diagnostic masks are not complete chart previews; they support geometric checks.
(ROOT/'artifacts').mkdir(exist_ok=True)
cmask.resize((W,H),Image.Resampling.LANCZOS).save(ROOT/'artifacts/cheese_mask.png')
stemlayer.getchannel('A').resize((W,H),Image.Resampling.LANCZOS).save(ROOT/'artifacts/stem_mask.png')
geometry={'canvas':[W,H],'plot':{'x0':X0,'x1':X1,'y_top':YTOP,'y_bottom':YBOTTOM,'v_min':VMIN,'v_max':VMAX},
          'years':times.tolist(),'points_A':[list(p) for p in PA],'points_B':[list(p) for p in PB],
          'cheese_top_y':cheese_top,'stem_radius_px':radius,'layers':['red tabletop','bread and butter','cheese','green chive','labels'],
          'labels':{'A':[f'{v:.1f}' for v in series['A']['values']], 'B':[f'{v:.1f}' for v in series['B']['values']]}}
(ROOT/'artifacts/geometry.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Wrote final.png at',W,'x',H)
