#!/usr/bin/env python3
"""Offline, data-bound nose infographic. Run from this directory with Python.
The generated atlas supplies the body material and anatomical details. All chart
geometry, scales, observation markers and typography are computed here.
"""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parent
BRIEF = json.loads((ROOT/'brief.json').read_text(encoding='utf-8'))
DATA = json.loads((ROOT/'data.json').read_text(encoding='utf-8'))
CORE = json.loads((ROOT/'core.json').read_text(encoding='utf-8'))
W,H = BRIEF['width'],BRIEF['height']
S = 2
INK = '#292e29'
MUTED = '#626a60'
GREEN = '#245d49'
PAPER = '#f8f5eb'
X0,X1 = 350.,1290.
YTOP,YBOTTOM = 550.,1490.
MINV,MAXV = DATA['value_axis']['minimum'],DATA['value_axis']['maximum']
TIMES = np.array(DATA['time_points'],dtype=float)
VALUES = np.array(DATA['series'][0]['values'],dtype=float)
CATS = DATA['categories']
assert len(TIMES)==len(VALUES)==len(CATS)==5
assert DATA['value_axis']['scale']=='linear'
assert np.all(np.diff(TIMES)>0)

def xmap(value): return X0+(float(value)-MINV)/(MAXV-MINV)*(X1-X0)
def ymap(year): return YBOTTOM-(float(year)-TIMES.min())/(TIMES.max()-TIMES.min())*(YBOTTOM-YTOP)
PTS=[(xmap(v),ymap(t)) for t,v in zip(TIMES,VALUES)]
ordered=sorted(PTS,key=lambda p:p[1])
interpolation=PchipInterpolator([p[1] for p in ordered],[p[0] for p in ordered])

def font(size,bold=False,serif=False,italic=False):
    name = 'P052-Italic.otf' if italic else ('P052-Roman.otf' if serif else ('NotoSans-Bold.ttf' if bold else 'NotoSans-Regular.ttf'))
    return ImageFont.truetype(str(ROOT/'assets'/name),round(size*S))

def line(draw,points,fill,width=1,joint='curve'):
    draw.line([(round(x*S),round(y*S)) for x,y in points],fill=fill,width=max(1,round(width*S)),joint=joint)

def txt(draw,pos,text,size,fill=INK,bold=False,serif=False,italic=False,anchor='lt'):
    draw.text((pos[0]*S,pos[1]*S),str(text),font=font(size,bold,serif,italic),fill=fill,anchor=anchor,stroke_width=0)

def paste_asset(layer,path,box):
    x,y,w,h=box
    im=Image.open(ROOT/path).convert('RGBA').resize((round(w*S),round(h*S)),Image.Resampling.LANCZOS)
    layer.alpha_composite(im,(round(x*S),round(y*S)))

# A deterministic, quiet paper ground; generated skin grain remains the dominant texture.
rng=np.random.default_rng(30)
noise=rng.normal(0,0.52,(H,W,1))
bg=np.clip(np.array([248,245,235],float)[None,None,:]+noise,0,255).astype('uint8')
canvas=Image.fromarray(bg).resize((W*S,H*S),Image.Resampling.BICUBIC).convert('RGBA')
d=ImageDraw.Draw(canvas)

# Editorial hierarchy.
d.rounded_rectangle((110*S,94*S,433*S,148*S),radius=8*S,fill=GREEN)
txt(d,(271.5,121),DATA['visible_notice'],25,fill='#fffdf4',bold=True,anchor='mm')
txt(d,(1488,99),'1995–2015',28,fill=GREEN,bold=True,anchor='rt')
txt(d,(1488,143),'FIVE INDICATED YEARS',18,fill=MUTED,anchor='rt')
txt(d,(108,194),DATA['series'][0]['label'],113,serif=True)
txt(d,(114,329),'Annual counts at five indicated years',29,fill=MUTED)
line(d,[(113,399),(1487,399)],'#c3c8bb',1.4)

# Near-frontal graph paper, with true square cells in the plotted data region.
panel=(110.,445.,1490.,1675.)
grid=Image.new('RGBA',canvas.size,(0,0,0,0)); gd=ImageDraw.Draw(grid)
for k in range(-5,25):
    xx=X0+k*47
    if panel[0]<=xx<=panel[2]:
        line(gd,[(xx,panel[1]),(xx,panel[3])],(104,139,146,32 if k%5 else 52),1 if k%5 else 1.25)
for k in range(-3,26):
    yy=YTOP+k*47
    if panel[1]<=yy<=panel[3]:
        line(gd,[(panel[0],yy),(panel[2],yy)],(104,139,146,32 if k%5 else 52),1 if k%5 else 1.25)
canvas.alpha_composite(grid)

# A single quantitative silhouette, independent from all decorative overlays.
ycoords=np.arange(round(YTOP*S),round(YBOTTOM*S)+1,dtype=float)/S
edge_x=interpolation(ycoords)
outline=[(X0,YTOP)]+list(zip(edge_x,ycoords))+[(X0,YBOTTOM)]
mask=Image.new('L',canvas.size,0); md=ImageDraw.Draw(mask)
md.polygon([(round(x*S),round(y*S)) for x,y in outline],fill=255)

# Row-wise material coordinates keep the generated texture inside the exact width.
skin=Image.open(ROOT/'assets/skin_pencil.png').convert('RGB')
skin_np=np.asarray(skin)
sw,sh=skin.size
body=Image.new('RGBA',canvas.size,(0,0,0,0))
body_np=np.zeros((round((YBOTTOM-YTOP)*S)+1,round((X1-X0)*S)+1,4),dtype=np.uint8)
for iy,(yy,right) in enumerate(zip(ycoords,edge_x)):
    span=max(0,round((right-X0)*S))
    if span==0: continue
    row=np.clip(round(iy/(len(ycoords)-1)*(sh-1)),0,sh-1)
    u=np.linspace(0,sw-1,span+1)
    lo=np.floor(u).astype(int); hi=np.minimum(lo+1,sw-1); frac=u-lo
    colors=skin_np[row,lo]*(1-frac[:,None])+skin_np[row,hi]*frac[:,None]
    body_np[iy,:span+1,:3]=np.rint(colors).astype('uint8')
    body_np[iy,:span+1,3]=255
body.alpha_composite(Image.fromarray(body_np),(round(X0*S),round(YTOP*S)))
body.putalpha(mask)
canvas.alpha_composite(body)

# Anatomical components are clipped to the same body. Their white/green/dark marks
# are color overlays and do not remove any logical quantitative area.
anatomy=Image.new('RGBA',canvas.size,(0,0,0,0))
paste_asset(anatomy,'assets/eye.png',(389,576,270,182))
paste_asset(anatomy,'assets/nose_tip.png',(942,631,332,357))
# Multiply the overlay alpha by the quantitative mask; no anatomy changes bounds.
aa=np.asarray(anatomy).copy()
aa[:,:,3]=np.rint(aa[:,:,3].astype(float)*np.asarray(mask)/255.).astype('uint8')
anatomy=Image.fromarray(aa)
canvas.alpha_composite(anatomy)

# Show the graph paper lightly through the warm skin; retain the eye's clarity.
inside_grid=Image.new('RGBA',canvas.size,(0,0,0,0)); ig=ImageDraw.Draw(inside_grid)
for k in range(21):
    xx=X0+k*47
    line(ig,[(xx,YTOP),(xx,YBOTTOM)],(86,100,99,25 if k%5 else 42),0.85 if k%5 else 1.2)
for k in range(21):
    yy=YTOP+k*47
    line(ig,[(X0,yy),(X1,yy)],(86,100,99,25 if k%5 else 42),0.85 if k%5 else 1.2)
ia=np.asarray(inside_grid).copy(); ia[:,:,3]=np.rint(ia[:,:,3].astype(float)*np.asarray(mask)/255.).astype('uint8')
canvas.alpha_composite(Image.fromarray(ia))
d=ImageDraw.Draw(canvas)

# The data edge is drawn from exactly the same interpolant used by the fill.
line(d,[(X0,YTOP),(ordered[0][0],YTOP)],INK,3.3)
line(d,list(zip(edge_x,ycoords)),INK,4.2)
line(d,[(X0,YBOTTOM),(ordered[-1][0],YBOTTOM)],INK,3.3)
# Axes and readable observation rows.
line(d,[(X0,YTOP-14),(X0,YBOTTOM),(X1,YBOTTOM)],INK,3.3)
txt(d,(X0-35,474),'YEAR',22,fill=GREEN,bold=True,anchor='rt')
txt(d,(X1,474),'NOSE PROCEDURES',22,fill=GREEN,bold=True,anchor='rt')
for cat,(xx,yy),value in zip(CATS,PTS,VALUES):
    line(d,[(X0-14,yy),(X0,yy)],INK,2.5)
    txt(d,(X0-31,yy),cat,32,fill=INK,bold=True,anchor='rm')
    # Exact boundary center; the uniform marker is not a quantitative minimum.
    r=8
    d.ellipse(((xx-r)*S,(yy-r)*S,(xx+r)*S,(yy+r)*S),fill=PAPER,outline=GREEN,width=3*S)
    r2=3
    d.ellipse(((xx-r2)*S,(yy-r2)*S,(xx+r2)*S,(yy+r2)*S),fill=GREEN)
    # A compact paper-backed annotation preserves the texture yet separates type.
    label=f'{value:.2f}'
    ff=font(33,True)
    bbox=d.textbbox((0,0),label,font=ff)
    tw=(bbox[2]-bbox[0])/S
    left=xx+25
    d.rounded_rectangle(((left-7)*S,(yy-24)*S,(left+tw+8)*S,(yy+25)*S),radius=6*S,fill=PAPER)
    txt(d,(left,yy),label,33,fill=GREEN,bold=True,anchor='lm')
for v in np.arange(MINV,MAXV+0.01,1):
    xx=xmap(v)
    line(d,[(xx,YBOTTOM),(xx,YBOTTOM+14)],INK,2.5)
    txt(d,(xx,YBOTTOM+34),f'{v:g}',29,fill=INK,anchor='mt')
txt(d,((X0+X1)/2,1590),'Million procedures per indicated year',28,fill=INK,anchor='mt')

# Short decoding note and explicit provenance, with no claim of real-world counts.
line(d,[(112,1732),(1488,1732)],'#bbc2b5',1.4)
txt(d,(112,1780),'READ THE OUTLINE',22,fill=GREEN,bold=True)
txt(d,(112,1825),'At each year, horizontal distance from 0 to the nose edge',25,fill=INK)
txt(d,(112,1868),'gives the count. Dots are observations; the line interpolates.',25,fill=INK)
txt(d,(112,1920),'Area and facial details do not encode additional quantities.',22,fill=MUTED)
line(d,[(1100,1780),(1100,1943)],'#c8cdbc',1.2)
txt(d,(1140,1782),'SYNTHETIC STUDY',20,fill=GREEN,bold=True)
txt(d,(1140,1828),'Illustrative values.',23,fill=INK)
txt(d,(1140,1867),'Not real-world',23,fill=INK)
txt(d,(1140,1906),'medical statistics.',23,fill=INK)

# Deterministic outputs. No model/network calls occur in this program.
canvas.convert('RGB').resize((W,H),Image.Resampling.LANCZOS).save(ROOT/'final.png')
(ROOT/'artifacts').mkdir(exist_ok=True)
mask.resize((W,H),Image.Resampling.LANCZOS).save(ROOT/'artifacts/geometry_mask.png')
geometry={
  'canvas':[W,H], 'axis':{'x0':X0,'x4':X1,'y1995':YBOTTOM,'y2015':YTOP},
  'mapping':{'px_per_million':(X1-X0)/(MAXV-MINV),'px_per_year':(YBOTTOM-YTOP)/(TIMES.max()-TIMES.min()),'interpolation':'shape-preserving piecewise cubic, no extra observations'},
  'observations':[{'year':cat,'value':float(value),'x':xx,'y':yy,'width_px':xx-X0} for cat,value,(xx,yy) in zip(CATS,VALUES,PTS)],
  'decorations_modify_quantitative_mask':False,
  'input_files':['brief.json','data.json','core.json'],
}
(ROOT/'artifacts/geometry.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Rendered final.png at',W,H)
