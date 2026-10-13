#!/usr/bin/env python3
"""Offline infographic renderer. Run only through capture_render.py for this trial."""
from pathlib import Path
from decimal import Decimal
from functools import lru_cache
import json
import math
import hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'assets'
OUT = ROOT / 'artifacts'
S = 3
COLORS = {'Others':'#aebdc5','Net A':'#ef9989','Net B':'#71d5c4','Net C':'#8abafb','Net D':'#eac17d'}
INK = '#edf2ee'
MUTED = '#9fafba'
MINT = '#a3e0cf'
TEXT_BOXES = []

@lru_cache(None)
def font(size, face='regular'):
    names = {'regular':'sans_regular.otf','bold':'sans_bold.otf','numeric':'numeric_bold.ttf'}
    return ImageFont.truetype(str(ASSETS/'fonts'/names[face]),round(size*S))

def rgba(hexcolor,alpha=255):
    h=hexcolor.lstrip('#')
    return tuple(int(h[i:i+2],16) for i in (0,2,4))+(alpha,)

def text(im, xy, content, size, fill=INK, face='regular', anchor='lt', record=True):
    d=ImageDraw.Draw(im)
    pos=(xy[0]*S,xy[1]*S)
    d.text(pos,content,font=font(size,face),fill=fill,anchor=anchor,stroke_width=0)
    if record:
        b=d.textbbox(pos,content,font=font(size,face),anchor=anchor)
        TEXT_BOXES.append({'text':content,'box':[round(v/S,2) for v in b]})

def line(im,pts,fill,width=1):
    ImageDraw.Draw(im).line([(round(x*S),round(y*S)) for x,y in pts],fill=fill,width=max(1,round(width*S)))

def rect(im,box,fill,outline=None,width=1,radius=0):
    b=tuple(round(v*S) for v in box)
    d=ImageDraw.Draw(im)
    if radius:
        d.rounded_rectangle(b,radius=round(radius*S),fill=fill,outline=outline,width=max(1,round(width*S)))
    else: d.rectangle(b,fill=fill,outline=outline,width=max(1,round(width*S)))

def ellipse(im,box,fill=None,outline=None,width=1):
    ImageDraw.Draw(im).ellipse(tuple(round(v*S) for v in box),fill=fill,outline=outline,width=max(1,round(width*S)))

def paste(im,part,x,y,w,h):
    p=part.resize((max(1,round(w*S)),max(1,round(h*S))),Image.Resampling.LANCZOS)
    im.alpha_composite(p,(round(x*S),round(y*S)))

MEMBER = Image.open(ASSETS/'metal_member.png').convert('RGBA')
CROSSBAR = Image.open(ASSETS/'metal_crossbar.png').convert('RGBA')
SIGN = Image.open(ASSETS/'category_sign.png').convert('RGBA')
FOUNDATION = Image.open(ASSETS/'foundation.png').convert('RGBA')

def beam(im, a, b, thickness=5.5, bright=1.0):
    """Place generated metal surface along a code-controlled structural member."""
    dx,dy=b[0]-a[0],b[1]-a[1]
    length=math.hypot(dx,dy)
    if length<=0: return
    patch=MEMBER.resize((max(1,round(length*S)),max(1,round(thickness*S))),Image.Resampling.LANCZOS)
    if bright!=1: patch=ImageEnhance.Brightness(patch).enhance(bright)
    angle=-math.degrees(math.atan2(dy,dx))
    patch=patch.rotate(angle,resample=Image.Resampling.BICUBIC,expand=True)
    cx=(a[0]+b[0])*S/2; cy=(a[1]+b[1])*S/2
    im.alpha_composite(patch,(round(cx-patch.width/2),round(cy-patch.height/2)))

def support(im,cx,y0):
    """Generated splayed foundation, row-warped to connect at the common zero line."""
    w,h=138,166
    src=np.asarray(FOUNDATION)
    # Use a mesh to independently control the support's top and bottom connection widths.
    dst=Image.new('RGBA',(w*S,h*S))
    mesh=[]
    sw,sh=FOUNDATION.size
    source_center=sw/2
    for j in range(h*S):
        t0=j/(h*S); t1=(j+1)/(h*S)
        # Raw support silhouette spans approx. 0.46 to 1.00 of its overall width.
        # Mapping connects its upper 0.46 width to the 106px upper-truss base.
        def source_half(t): return (119+139*t)
        def target_half(t): return (53+16*t)*S
        f0=source_half(t0)/target_half(t0); f1=source_half(t1)/target_half(t1)
        l0=source_center-w*S/2*f0; r0=source_center+w*S/2*f0
        l1=source_center-w*S/2*f1; r1=source_center+w*S/2*f1
        mesh.append(((0,j,w*S,j+1),(l0,t0*sh,l1,t1*sh,r1,t1*sh,r0,t0*sh)))
    dst=FOUNDATION.transform(dst.size,Image.Transform.MESH,mesh,Image.Resampling.BICUBIC)
    dst=ImageEnhance.Brightness(dst).enhance(.66)
    im.alpha_composite(dst,(round((cx-w/2)*S),round(y0*S)))

def tower(im,cx,y0,ytop,color):
    """All data-bearing metal remains inside [ytop, y0]; no minimum height."""
    h=y0-ytop
    # Work in a local clipping rectangle so thick members cannot inflate data height.
    width=120
    height_px=max(1,math.ceil(h*S))
    layer=Image.new('RGBA',(width*S,height_px))
    c=width/2
    # Clipping with a fractional final row preserves the positive 0.3% height.
    def hw(t): return 22+31*t
    t0=min(2.2,h*.18)
    beam(layer,(c-hw(0),t0),(c-hw(1),h),6.0)
    beam(layer,(c+hw(0),t0),(c+hw(1),h),6.0)
    beam(layer,(c,t0),(c,h),3.8,0.77)
    bays=max(1,round(h/96))
    for k in range(bays):
        q0=k/bays; q1=(k+1)/bays
        ya=t0+(h-t0)*q0; yb=t0+(h-t0)*q1
        wa=hw(q0); wb=hw(q1)
        beam(layer,(c-wa,ya),(c+wb,yb),4.7,.93)
        beam(layer,(c+wa,ya),(c-wb,yb),4.7,1.05)
        if k>0:
            beam(layer,(c-wa,ya),(c+wa,ya),4.5)
    # Generated metal crossbar; its exact upper edge is the measurement endpoint.
    cap_h=min(4.5,h*.30)
    paste(layer,CROSSBAR,c-42,0,84,cap_h)
    rect(layer,(c-42,0,c+42,min(1.0,cap_h)),color)
    # Install at the nearest supersample boundary (<0.167px); bottom is clipped to zero.
    ypx=round(ytop*S)
    bottom=round(y0*S)
    layer=layer.crop((0,0,width*S,bottom-ypx))
    im.alpha_composite(layer,(round((cx-width/2)*S),ypx))
    return {'bays':bays,'cap_width_px':84,'cap_thickness_px':round(cap_h,4),'visible_clip_px':[ypx/S,bottom/S]}

def label(im,cx,ytop,parent,category,value,color):
    cy=ytop-110
    diam=130
    # Physical connection to the tower; the sign is explicitly outside the height scale.
    beam(im,(cx,cy+diam/2-1),(cx,ytop-2),5.2,.82)
    paste(im,SIGN,cx-diam/2,cy-diam/2,diam,diam)
    ellipse(im,(cx-65,cy-65,cx+65,cy+65),outline=color,width=1.6)
    if '/' in category:
        suffix=category.split('subcategory ',1)[1]
        text(im,(cx,cy-27),parent,28,'#142b3b','bold','mt')
        text(im,(cx,cy+6),'subcategory',17,'#344c5b','regular','mt')
        text(im,(cx,cy+28),suffix,23,'#142b3b','bold','mt')
    else:
        text(im,(cx,cy-14),category,29 if category!='Others' else 27,'#142b3b','bold','mt')
    value_text=format(value,'g')+'%'
    # A dark recess makes the value legible without hiding the data endpoint.
    f=font(36,'numeric')
    width=ImageDraw.Draw(im).textlength(value_text,font=f)/S
    rect(im,(cx-width/2-6,ytop-43,cx+width/2+6,ytop-7),'#0b1e2b',radius=7)
    text(im,(cx,ytop-40),value_text,36,color,'numeric','mt')
    return {'category':category,'category_parent':parent,'value_text':value_text,'badge_center_px':[cx,round(cy,6)],'badge_diameter_px':diam}


def main():
    OUT.mkdir(exist_ok=True)
    data=json.loads((ROOT/'data.json').read_text(encoding='utf-8'))
    brief=json.loads((ROOT/'brief.json').read_text(encoding='utf-8'))
    core=json.loads((ROOT/'core.json').read_text(encoding='utf-8'))
    inputs=json.loads((ROOT/'input.json').read_text(encoding='utf-8'))
    assert inputs['critical_metaphor_contract']=='core.json'
    assert core['semantic_roles_zh']==['Wireless subscriber share (percent)']
    assert data['value_axis']=={'minimum':0,'maximum':40,'scale':'linear'}
    assert data['series'][0]['unit']=='percent'
    W,H=brief['width'],brief['height']
    assert (W,H)==(1600,2000)
    values=data['series'][0]['values']
    cats=data['categories']; parents=data['category_parent']
    assert len(values)==len(cats)==len(parents)==8
    assert sum(Decimal(str(v)) for v in values)==Decimal('100')
    # Restrained procedural background; all physical tower surfaces use generated pixels.
    yy,xx=np.mgrid[0:H,0:W]
    glow=np.exp(-((xx-720)**2/(1250**2)+(yy-1040)**2/(1250**2)))
    base=np.zeros((H,W,3),dtype=np.uint8)
    for i,(v,g) in enumerate([(5,7),(16,14),(25,16)]):base[:,:,i]=v+g*glow
    im=Image.fromarray(base).convert('RGBA').resize((W*S,H*S),Image.Resampling.BICUBIC)
    # Header.
    text(im,(86,72),'NETWORK COMPOSITION',20,MUTED,'bold')
    rect(im,(1222,60,1514,105),MINT,radius=5)
    text(im,(1368,72),data['visible_notice'],22,'#102630','bold','mt')
    text(im,(80,144),'Wireless subscriber',94,INK,'bold')
    text(im,(80,238),'shares',94,INK,'bold')
    text(im,(85,349),'Share of total wireless subscribers (%)',29,'#b5c3cd')
    line(im,[(86,401),(1514,401)],'#3c505e',1)
    text(im,(206,439),'HEIGHT ABOVE ZERO = SHARE',19,MINT,'bold')
    text(im,(206,470),'Measure to the top crossbar; signs and lower supports are excluded.',22,MUTED)
    # The frozen axis is uniformly linear: 24.25 pixels per percentage point.
    y0=1490.0
    axis_top=520.0
    scale=(y0-axis_top)/40.0
    left=166.0; right=1540.0
    # Lower support zone clearly separated from the quantitative chart region.
    rect(im,(165,y0,1540,1680),rgba('#091721',180))
    for tick in [40,30,20,10]:
        y=y0-tick*scale
        for x in np.arange(left,right,13):line(im,[(x,y),(min(x+4,right),y)],'#334754',.9)
        text(im,(136,y-11),str(tick)+'%',23,MUTED,'regular','rt')
    # Zero remains visible around all eight physical connection points.
    line(im,[(left,axis_top),(left,y0)],'#435966',1)
    centers=np.linspace(244,1468,len(cats))
    geometry=[]
    for idx,(cx,category,parent,value) in enumerate(zip(centers,cats,parents,values)):
        ytop=y0-float(Decimal(str(value))*Decimal(str(scale)))
        color=COLORS[parent]
        support(im,float(cx),y0)
        geom=tower(im,float(cx),y0,ytop,color)
        bound=label(im,float(cx),ytop,parent,category,value,color)
        geometry.append({'index':idx,'category':category,'value':value,'x_px':float(cx),'baseline_px':y0,'top_px':ytop,'height_px':y0-ytop,'color':color,**geom,**bound})
    # Draw a continuous common zero reference after the tower/support joins.
    line(im,[(left,y0),(right,y0)],MINT,1.6)
    text(im,(136,y0-14),'0%',28,MINT,'bold','rt')
    text(im,(172,1506),'COMMON ZERO',15,MINT,'bold')
    # Footnotes and reading key.
    line(im,[(86,1711),(1514,1711)],'#3c505e',1)
    line(im,[(88,1755),(139,1755)],MINT,3)
    text(im,(157,1743),'HOW TO READ',21,MINT,'bold')
    text(im,(86,1789),'Only the upper lattice height encodes subscriber share.',28,INK,'bold')
    text(im,(86,1832),'Round signs and the equally tall lower supports are non-data components.',24,'#b5c3cd')
    text(im,(86,1880),'Net B and Net C subcategories are mutually exclusive parts of their named networks.',22,MUTED)
    text(im,(86,1916),'Synthetic values and category names; not a real statistical survey.',22,MUTED)
    text(im,(86,1964),'8 CATEGORIES  /  TOTAL 100%  /  INPUT ORDER PRESERVED',17,'#8297a6','bold')
    # Frozen title is expressed by the prominent synthetic badge + topic title.
    final=im.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
    final.save(ROOT/'final.png',format='PNG',optimize=True)
    payload={'canvas':[W,H],'axis':{'minimum':0,'maximum':40,'linear':True,'pixels_per_percentage_point':scale,'top_px':axis_top,'baseline_px':y0},'marks':geometry,'text_boxes':TEXT_BOXES,'generated_surface_assets':['assets/foundation.png','assets/category_sign.png','assets/metal_crossbar.png','assets/metal_member.png'],'input_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['input.json','brief.json','data.json','core.json','reference.png']}}
    (OUT/'geometry.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Rendered final.png at 1600 × 2000 using generated tower parts and a 0–40% linear scale.')

if __name__=='__main__':main()
