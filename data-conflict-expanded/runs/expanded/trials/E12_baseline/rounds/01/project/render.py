#!/usr/bin/env python3
"""Offline renderer. Generated fork parts + data-controlled tines and typography.
Run only through capture_render.py during the frozen experiment.
"""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parent
DATA=json.loads((ROOT/'data.json').read_text(encoding='utf-8'))
BRIEF=json.loads((ROOT/'brief.json').read_text(encoding='utf-8'))
CORE=json.loads((ROOT/'core.json').read_text(encoding='utf-8'))
assert DATA['value_axis']=={'minimum':0,'maximum':20,'scale':'linear'}
assert len(DATA['series'])==2
W,H=BRIEF['width'],BRIEF['height']
S=2
INK='#162431'; MUTED='#64717A'; LIGHT='#DBE1E5'
BLUE='#034FBA'; RED='#D00B26'
ANGLE=-38.0
C=math.cos(math.radians(ANGLE)); SN=math.sin(math.radians(ANGLE))
PX_PER_PERCENT=18.0
TINE_WIDTH=38.0
CENTERS=[-164,-82,0,82,164]
BODY_W,BODY_H=400,770
LOCAL_X,LOCAL_Y=350,620
LOCAL_W,LOCAL_H=700,1430
SERIES_STYLE=[
    {'name':'blue','color':BLUE,'origin':(515,1090),'heading':(110,456),'name_label':'Table service'},
    {'name':'red','color':RED,'origin':(1300,1050),'heading':(1435,454),'name_label':'Quick service'},
]

def font(size,bold=False):
    return ImageFont.truetype(str(ROOT/'assets/fonts'/('NimbusSans-Bold.otf' if bold else 'NimbusSans-Regular.otf')),round(size*S))

def pos(x,y): return (round(x*S),round(y*S))
def rgba(hexcolor,alpha=255):
    return tuple(int(hexcolor[i:i+2],16) for i in (1,3,5))+(alpha,)
def text(draw,xy,value,size,color=INK,bold=False,anchor='lt'):
    draw.text(pos(*xy),value,font=font(size,bold),fill=color,anchor=anchor)
def line(draw,xy,fill,width=1):
    draw.line([pos(x,y) for x,y in xy],fill=fill,width=max(1,round(width*S)))
def tracking(draw,xy,value,size,color,space=2):
    x,y=xy; f=font(size,True)
    for ch in value:
        draw.text(pos(x,y),ch,font=f,fill=color,anchor='lt')
        x+=draw.textlength(ch,font=f)/S+space

def local_xy(u,v): return (round((u+LOCAL_X)*S),round((v+LOCAL_Y)*S))
def project(origin,u,v):
    return [origin[0]+C*u-SN*v,origin[1]+SN*u+C*v]

canvas=Image.new('RGBA',(W*S,H*S),'white')
d=ImageDraw.Draw(canvas)
# A deliberately quiet editorial header on a white field.
d.rounded_rectangle([pos(110,70),pos(402,117)],radius=8*S,fill=INK)
tracking(d,(128,83),DATA['visible_notice'],22,'#FFFFFF',1.2)
tracking(d,(1444,84),'DINING / COMPARISON',20,MUTED,1.5)
text(d,(105,153),'Restaurant spending change',105,INK,True)
text(d,(110,287),'Two service types, measured by the length of each fork tine.',31,MUTED)
line(d,[(110,357),(1890,357)],LIGHT,1.3)

for i,(series,style) in enumerate(zip(DATA['series'],SERIES_STYLE)):
    hx,hy=style['heading']; col=style['color']
    d.rounded_rectangle([pos(hx,hy),pos(hx+47,hy+7)],radius=3*S,fill=col)
    text(d,(hx,hy+27),style['name_label'],41,INK,True)
    text(d,(hx,hy+81),'Spending change (%)',28,MUTED)
    tracking(d,(hx,hy+124),'5 SYNTHETIC CATEGORIES',18,col,1.1)

# The legend states what is actually encoded; it never reports a derived total.
text(d,(1473,726),'ONE SHARED SCALE',21,INK,True)
text(d,(1473,768),'0–20%',60,INK,True)
text(d,(1476,844),'Tine root = 0%',26,MUTED)
text(d,(1476,884),'Length follows the local tine axis.',23,MUTED)

geometry=[]
for series,style in zip(DATA['series'],SERIES_STYLE):
    layer=Image.new('RGBA',(LOCAL_W*S,LOCAL_H*S),(0,0,0,0))
    ld=ImageDraw.Draw(layer)
    body=Image.open(ROOT/f"assets/{style['name']}_body.png").convert('RGBA').resize((BODY_W*S,BODY_H*S),Image.Resampling.LANCZOS)
    material=Image.open(ROOT/f"assets/{style['name']}_tine_material.png").convert('RGB')
    bodyarr=np.array(body)
    # Tine masks are the quantitative silhouette; materials never alter its length.
    for u,value,label in zip(CENTERS,series['values'],series['category_labels']):
        length=PX_PER_PERCENT*float(value)
        if value>0:
            tw=round(TINE_WIDTH*S); th=round(length*S)
            tex=material.resize((tw,th),Image.Resampling.BICUBIC)
            a=np.array(tex).astype(float)
            # Match the generated head at the joint without changing the mask.
            bx0=round((u-TINE_WIDTH/2+BODY_W/2)*S)
            joint=bodyarr[2,bx0:bx0+tw,:3].astype(float)
            blend_rows=min(th,round(26*S))
            t=np.linspace(0,1,blend_rows)[:,None,None]
            a[-blend_rows:]=a[-blend_rows:]*(1-t)+joint[None,:,:]*t
            tine=Image.fromarray(np.uint8(np.clip(a,0,255))).convert('RGBA')
            mask=Image.new('L',(tw,th),255)
            # Square true endpoints, matching the bar-like prongs in the reference.
            tine.putalpha(mask)
            layer.alpha_composite(tine,local_xy(u-TINE_WIDTH/2,-length))
        # Labels follow the tine direction, all typeset by code from data.json.
        cat_font=font(32); val_font=font(35,True)
        cat_text=label+' '
        val_text=f'{value:g}%'
        temp=Image.new('RGBA',(700*S,55*S),(0,0,0,0)); td=ImageDraw.Draw(temp)
        catw=td.textlength(cat_text,font=cat_font)
        valw=td.textlength(val_text,font=val_font)
        td.text((2*S,6*S),cat_text,font=cat_font,fill='#4D5962',anchor='lt')
        td.text((2*S+catw,4*S),val_text,font=val_font,fill=style['color'],anchor='lt')
        label_w=math.ceil(catw+valw+5*S)
        label_patch=temp.crop((0,0,label_w,45*S)).transpose(Image.Transpose.ROTATE_270)
        # End of each text label is 15 px away from its true endpoint, including zero.
        label_x=(u+LOCAL_X)*S-label_patch.width/2
        label_y=(-length+LOCAL_Y-15)*S-label_patch.height
        layer.alpha_composite(label_patch,(round(label_x),round(label_y)))
        geometry.append({
            'series':series['id'],'category':label,'value':value,'unit':series['unit'],
            'local_center_u':u,'root_xy':project(style['origin'],u,0),
            'tip_xy':project(style['origin'],u,-length),
            'local_axis_length_px':length,'axis_rotation_degrees':ANGLE,
        })
    # The generated body is a visible, continuous part of the quantitative object.
    layer.alpha_composite(body,local_xy(-BODY_W/2,0))
    ld=ImageDraw.Draw(layer)
    # A restrained line identifies the common root, with no extension for zero.
    ld.line([local_xy(-183,1),local_xy(183,1)],fill=(255,255,255,150),width=2*S)
    for u,value in zip(CENTERS,series['values']):
        if value==0:
            ld.line([local_xy(u-11,4),local_xy(u+11,4)],fill=(255,255,255,240),width=2*S)
    ox,oy=style['origin']
    inverse=(C,SN,(LOCAL_X-C*ox-SN*oy)*S,
             -SN,C,(LOCAL_Y+SN*ox-C*oy)*S)
    rotated=layer.transform((W*S,H*S),Image.Transform.AFFINE,inverse,Image.Resampling.BICUBIC)
    canvas.alpha_composite(rotated)

# The caption preserves the synthetic status and clarifies the special zero.
d=ImageDraw.Draw(canvas)
line(d,[(110,1733),(1890,1733)],LIGHT,1.3)
text(d,(110,1760),'HOW TO READ',19,INK,True)
text(d,(308,1759),'Equal lengths mean equal changes. Table D = 0%: no projecting tine.',25,MUTED)
text(d,(110,1798),'Synthetic categories and illustrative values; not observed restaurant statistics.',21,MUTED)

out=canvas.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
out.save(ROOT/'final.png',compress_level=6)
# This is a coordinate ledger, not a claim of pixel verification.
(ROOT/'artifacts').mkdir(exist_ok=True)
(ROOT/'artifacts/geometry.json').write_text(json.dumps({
    'canvas':[W,H],'scale_px_per_percentage_point':PX_PER_PERCENT,
    'rotation_degrees':ANGLE,'tine_width_px':TINE_WIDTH,'marks':geometry,
    'frozen_dependencies':['brief.json','data.json','core.json'],
},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(f'Rendered {W} x {H}; {len(geometry)} values; one common scale = {PX_PER_PERCENT:g} px/percentage point.')
