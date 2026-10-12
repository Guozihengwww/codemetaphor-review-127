#!/usr/bin/env python3
"""Render the frozen local data with one generated envelope template.

Run from this directory with the supplied interpreter. No network/model calls.
Every complete experimental render must be invoked through capture_render.py.
"""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
S = 3  # supersampling; output remains exactly the brief's dimensions

def read_json(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))

def save_json(name, value):
    path=ROOT/name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

brief=read_json('brief.json')
data=read_json('data.json')
W,H=brief['width'],brief['height']
series={s['id']:s for s in data['series']}
ceo=series['ceo_pay']['values']
workers=series['worker_median_pay']['values']
categories=data['categories']
assert len(categories)==len(ceo)==len(workers)==5
assert data['worker_axis']=={'minimum':0,'maximum':80000}
assert data['value_axis']['minimum']==0
assert data['value_axis']['maximum']==data['ceo_area_reference']

# One unbroken zero ground, one linear mapping for every basket center.
GROUND_Y=1724.0
HEIGHT_SPAN=840.0
WORKER_MAX=float(data['worker_axis']['maximum'])
REF_PAY=float(data['ceo_area_reference'])
REF_ENVELOPE_H=398.0
X=[390.0,730.0,1070.0,1410.0,1750.0]
BASKET_W=62.0
BASKET_H=40.0
NECK_TO_BASKET_CENTER=78.0
COLORS=['#E97B59','#E8BC37','#62A9CE','#84BA51','#E79B40']
INK='#193B46'
MUTED='#57717B'
GRID='#CDDEE0'
TEAL='#638F98'
FONT_CACHE={}


def font(size,bold=False,cjk=False):
    key=(size,bold,cjk)
    if key not in FONT_CACHE:
        name='NotoSansCJK-Medium.ttc' if cjk else ('NotoSans-Bold.ttf' if bold else 'NotoSans-Regular.ttf')
        FONT_CACHE[key]=ImageFont.truetype(str(ROOT/'fonts'/name),round(size*S))
    return FONT_CACHE[key]

# A restrained sky gradient, generated object material remains the visual focus.
top=np.array([244,250,251],float)
bottom=np.array([254,254,249],float)
y=np.linspace(0,1,H*S)[:,None,None]
bg=top[None,None,:]*(1-y)+bottom[None,None,:]*y
bg=np.broadcast_to(bg,(H*S,W*S,3)).astype(np.uint8).copy()
canvas=Image.fromarray(bg,'RGB').convert('RGBA')
draw=ImageDraw.Draw(canvas)


def xybox(box): return tuple(round(v*S) for v in box)

def line(points,fill,width=1):
    draw.line([(round(x*S),round(y*S)) for x,y in points],fill=fill,width=max(1,round(width*S)))

def ellipse(box,fill,outline=None,width=1):
    draw.ellipse(xybox(box),fill=fill,outline=outline,width=max(1,round(width*S)))

def rounded(box,radius,fill,outline=None,width=1):
    draw.rounded_rectangle(xybox(box),radius=round(radius*S),fill=fill,outline=outline,width=max(1,round(width*S)))

def text(x,y,txt,size=28,color=INK,bold=False,anchor='la',cjk=False):
    draw.text((round(x*S),round(y*S)),txt,font=font(size,bold,cjk),fill=color,anchor=anchor)

def smallcaps(x,y,txt,size=18,color=MUTED,spacing=2.0,anchor='left'):
    f=font(size,True)
    widths=[draw.textlength(c,font=f)/S for c in txt]
    total=sum(widths)+spacing*(len(txt)-1)
    if anchor=='center': x-=total/2
    for c,w in zip(txt,widths):
        text(x,y,c,size,color,True)
        x+=w+spacing


def cloud(cx,cy,w,h):
    # Quiet graphic clouds from the supplied reference's visual language.
    color='#E9F0F0'
    ellipse([cx-w*.5,cy-h*.04,cx+w*.50,cy+h*.36],color)
    ellipse([cx-w*.37,cy-h*.31,cx-w*.01,cy+h*.26],color)
    ellipse([cx-w*.15,cy-h*.49,cx+w*.22,cy+h*.26],color)
    ellipse([cx+w*.09,cy-h*.22,cx+w*.38,cy+h*.28],color)

# Header and brief-derived title.
smallcaps(90,64,'PAY LANDSCAPE  /  05 COMPANIES',18)
text(86,105,data['title'],70,INK,cjk=True)
text(90,218,'CEO compensation and median worker salary',31,MUTED)
rounded([1517,91,1890,148],28,'#E3EEF0')
smallcaps(1703.5,106,data['visible_notice'],20,INK,1.8,'center')
text(1888,216,'USD / year',27,MUTED,anchor='ra')
line([(90,288),(1890,288)],'#D8E6E6',1.5)
text(90,322,'Bigger envelope = higher CEO compensation',25,INK)
text(1070,322,'Higher basket = higher worker salary',25,INK)

# Shared scene: clouds are background only, with no data semantics.
cloud(573,1190,180,84)
cloud(1180,1428,187,82)
cloud(1810,780,157,70)
cloud(833,531,148,64)
cloud(310,1543,142,66)

# Worker axis: linearly positioned basket-center ticks and a common zero.
text(78,774,'WORKER PAY',20,INK,True)
text(78,808,'USD / year',21,MUTED)
axis_x=215.0
line([(axis_x,GROUND_Y-HEIGHT_SPAN),(axis_x,GROUND_Y)],'#93AFB4',1.4)
for v in range(0,80001,20000):
    yy=GROUND_Y-HEIGHT_SPAN*v/WORKER_MAX
    if v>0:
        # Interrupted very light grid so envelope areas remain easy to see.
        for xx in range(229,1900,18):
            line([(xx,yy),(min(xx+6,1900),yy)],GRID,1.2)
    line([(axis_x-8,yy),(axis_x+6,yy)],'#77999F',1.5)
    text(191,yy,f'${v:,.0f}',25,INK,anchor='rm')

# The pasture starts on the same flat, non-data zero for all observations.
path=[(215,GROUND_Y),(1900,GROUND_Y),(1900,1778),(1780,1792),(1600,1800),
      (1420,1805),(1160,1811),(925,1813),(680,1807),(470,1795),(300,1783),(215,1768)]
draw.polygon([(round(x*S),round(y*S)) for x,y in path],fill='#97B45B')
# Broad, quiet contours contained below the zero plane.
draw.polygon([tuple(round(v*S) for v in p) for p in [(215,1760),(615,1745),(1120,1782),(1670,1751),(1900,1752),(1900,1778),(1780,1792),(1160,1811),(680,1807),(300,1783),(215,1768)]],fill='#809E4A')
line([(215,GROUND_Y),(1900,GROUND_Y)],'#5F8040',2.5)
text(249,1736,'COMMON ZERO GROUND',16,'#F8FBED',True)

# Small fixed scenery, drawn below the data objects.
def tree(cx,base,scale=1):
    def e(box,col): ellipse([cx+box[0]*scale,base+box[1]*scale,cx+box[2]*scale,base+box[3]*scale],col)
    e([-20,-67,22,-19],'#BBD46A')
    e([-27,-49,1,-15],'#A9C459')
    e([2,-52,28,-16],'#C1D974')
    line([(cx,base),(cx,base-52*scale)],'#6C7842',3*scale)
    line([(cx,base-20*scale),(cx-13*scale,base-32*scale)],'#6C7842',2*scale)
    line([(cx,base-29*scale),(cx+12*scale,base-42*scale)],'#6C7842',2*scale)

tree(943,GROUND_Y,1.05)
tree(1876,GROUND_Y,.83)

source=Image.open(ROOT/'assets/envelope_template.png').convert('RGBA')
source_arr=np.array(source)
source_alpha=source_arr[:,:,3]
source_luma=(.2126*source_arr[:,:,0]+.7152*source_arr[:,:,1]+.0722*source_arr[:,:,2])/255.
# Tonal material mapping keeps generated fabric and seams visible in each hue.
def tinted(hexcolor):
    base=np.array([int(hexcolor[k:k+2],16) for k in (1,3,5)],dtype=float)
    t=np.clip((source_luma-.30)/.67,0,1)
    shade=.51+.49*t
    rgb=base[None,None,:]*shade[:,:,None]
    highlight=np.clip((t-.76)/.24,0,1)*.12
    rgb=rgb*(1-highlight[:,:,None])+255*highlight[:,:,None]
    a=np.dstack([np.clip(rgb,0,255).astype(np.uint8),source_alpha])
    return Image.fromarray(a,'RGBA')

base_scale=REF_ENVELOPE_H/source.height
ref_area=float((source_alpha/255.).sum()*base_scale**2)
geometry=[]
maskdir=ROOT/'artifacts/masks'; maskdir.mkdir(parents=True,exist_ok=True)

for i,(category,cv,wv,cx,color) in enumerate(zip(categories,ceo,workers,X,COLORS)):
    cy=GROUND_Y-HEIGHT_SPAN*float(wv)/WORKER_MAX
    scale=base_scale*math.sqrt(float(cv)/REF_PAY)
    ew=source.width*scale
    eh=source.height*scale
    bottom=cy-NECK_TO_BASKET_CENTER
    left=cx-ew/2
    topy=bottom-eh
    # All ropes are behind the generated envelope and terminate on this basket.
    for side in (-1,1):
        line([(cx+side*.335*ew,topy+.59*eh),(cx+side*25,cy-20)],'#8C7860',1.65)
        line([(cx+side*.185*ew,topy+.80*eh),(cx+side*17,cy-20)],'#9A876F',1.1)
    # Local affine map uses one scalar, not independently fitted width/height.
    # A padded crop accommodates subpixel positioning with no silhouette cutoff.
    ox=math.floor(left)-3; oy=math.floor(topy)-3
    ow=math.ceil(ew)+7; oh=math.ceil(eh)+7
    tile=tinted(color).transform((ow*S,oh*S),Image.Transform.AFFINE,
        (1/(scale*S),0,(ox-left)/scale,0,1/(scale*S),(oy-topy)/scale),
        Image.Resampling.BICUBIC)
    canvas.alpha_composite(tile,(ox*S,oy*S))
    draw=ImageDraw.Draw(canvas)
    ma=tile.getchannel('A').resize((ow,oh),Image.Resampling.LANCZOS)
    ma.save(maskdir/f'envelope_{i+1:02d}.png')
    alpha=np.array(ma,dtype=float)/255
    measured_area=float(alpha.sum())
    # Clear direct binding: category above, annual CEO amount on the envelope.
    text(cx,topy-29,category,31,INK,True,anchor='ms')
    text(cx,topy+.29*eh,'CEO',18,'#263E43',True,anchor='mm')
    text(cx,topy+.405*eh,f'${cv:,}',31,'#132E37',True,anchor='mm')
    # The drop line and tiny side ticks meet the geometrical basket center.
    line([(cx,cy),(cx,GROUND_Y)],'#9FC7CB',1.4)
    line([(cx-43,cy),(cx+43,cy)],'#73999F',1.6)
    # Fixed burner frame and fixed-size woven basket.
    line([(cx-9,cy-42),(cx-9,cy-24)],'#665D4B',2)
    line([(cx+9,cy-42),(cx+9,cy-24)],'#665D4B',2)
    rounded([cx-10,cy-45,cx+10,cy-40],1.5,'#7F6B48')
    basket_polygon=[(cx-29,cy-18),(cx+29,cy-18),(cx+26,cy+20),(cx-26,cy+20)]
    draw.polygon([(round(x*S),round(y*S)) for x,y in basket_polygon],fill='#A07A48')
    for dy in (-9,-1,7,15):
        line([(cx-26,cy+dy),(cx+26,cy+dy)],'#B9955E',1.5)
    for dx in range(-22,24,7):
        line([(cx+dx,cy-13),(cx+dx-1,cy+18)],'#866641',.9)
    rounded([cx-31,cy-20,cx+31,cy-11],3,'#655A3C')
    line([(cx-25,cy+20),(cx+25,cy+20)],'#745B3B',1.4)
    # Keep the guide legible while giving numeric labels a clean sky backing.
    rounded([cx-98,cy+31,cx+98,cy+79],10,'#F8FBF8')
    text(cx,cy+55,f'${wv:,}',31,INK,True,anchor='mm')
    text(cx,cy+91,'median worker',19,MUTED,anchor='mm')
    # A fixed-size landing dash, not a second size encoding.
    ellipse([cx-17,GROUND_Y-3,cx+17,GROUND_Y+3],'#C5D493')
    geometry.append({
        'category':category,'ceo_pay':cv,'worker_median_pay':wv,
        'template_scale':scale,'area_ratio_target':cv/REF_PAY,
        'actual_envelope_alpha_area_px2':measured_area,
        'actual_area_ratio':measured_area/ref_area,
        'envelope_bbox_float':[left,topy,left+ew,bottom],
        'mask_path':f'artifacts/masks/envelope_{i+1:02d}.png',
        'mask_canvas_origin':[ox,oy],
        'basket_center':[cx,cy],'basket_bbox':[cx-31,cy-20,cx+31,cy+20],
        'rope_endpoints':[[cx-25,cy-20],[cx+25,cy-20]],
        'all_parts_within_canvas':bool(left>=0 and topy>=0 and cx+ew/2<W and cy+101<H)
    })

# Compact encoding contract, kept below the shared scene.
line([(90,1855),(1890,1855)],'#D8E4DE',1.5)
smallcaps(90,1881,'AREA  /  CEO COMPENSATION',18,INK,1.5)
text(90,1919,'Same envelope; area scales linearly from $0.',22,MUTED)
text(90,1950,f'Area reference: ${int(REF_PAY):,} per year.',22,MUTED)
smallcaps(1070,1881,'HEIGHT  /  MEDIAN WORKER SALARY',18,INK,1.3)
text(1070,1919,'Basket center on the shared $0–$80,000 axis.',22,MUTED)
text(1070,1950,'Ropes and fixed-size baskets do not encode CEO pay.',22,MUTED)

final=canvas.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
final.save(ROOT/'final.png')
save_json('artifacts/render_geometry.json',{
    'canvas':[W,H], 'area_reference_usd_per_year':REF_PAY,
    'area_reference_px2':ref_area,'template_size':list(source.size),
    'ground_y_px':GROUND_Y,'worker_axis_span_px':HEIGHT_SPAN,
    'worker_axis_range':[0,WORKER_MAX],'basket_size_px':[BASKET_W,BASKET_H],
    'uniform_scale_formula':'(reference_envelope_height / template_height) * sqrt(ceo_pay / ceo_area_reference)',
    'basket_center_formula':'ground_y - worker_axis_span * worker_salary / worker_axis_max',
    'objects':geometry})
print('Rendered final.png at',W,'x',H,'with',len(geometry),'data-bound balloons.')
