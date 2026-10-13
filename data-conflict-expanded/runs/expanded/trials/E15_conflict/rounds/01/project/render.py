#!/usr/bin/env python3
"""Offline, deterministic object-integrated chart. Run through capture_render.py.

Data encodes vertical height only. The generated glass is warped with one shared
source-to-canvas mapping. Its decorative width and base have no quantitative role.
"""
from pathlib import Path
import json
import math
import hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parent
SCALE = 2

def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))

def save_json(name, obj):
    (ROOT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

brief = read('brief.json')
data = read('data.json')
core = read('core.json')
inputs = read('input.json')
# The frozen inputs remain available to this renderer. The reference is not sampled.
assert all((ROOT / inputs[k]).is_file() for k in ('brief','data','reference','critical_metaphor_contract'))
assert core['semantic_roles_zh'] == ['Synthetic share of indoor household water use (%)']
assert data['value_axis'] == {'minimum': 0, 'maximum': 100, 'scale': 'linear'}
W, H = brief['width'], brief['height']
values = np.asarray(data['series'][0]['values'], dtype=float)
categories = data['categories']
assert len(categories) == len(values) == len(data['category_parent'])
assert np.all(values > 0) and np.isclose(values.sum(), 100)
assert data['series'][0]['unit'] == '%'

INK = '#143F50'
MUTED = '#546F7B'
LINE = '#79949F'
COLORS = ['#59AD9C', '#D8A03A', '#E9C76F', '#AD4168', '#D77E94',
          '#299ABB', '#7FC3D6', '#756291', '#ACA0C2', '#DE864E',
          '#9CCDA6', '#647F91']
assert len(COLORS) == len(values)

# A single source-coordinate-to-canvas transform for the generated glass.
# The original full asset was cropped at (145,125); this maps the crop only.
GLASS_X = 735.0
GLASS_W = 800.0
SRC_Y = np.array([0., 58., 710., 863.])
DST_Y = np.array([552., 650., 1730., 1818.])
DATA_TOP, DATA_BOTTOM = DST_Y[1:3]
DATA_HEIGHT = DATA_BOTTOM - DATA_TOP
# Inner wall positions measured in source-pixel coordinates (within the crop).
INNER_X_TOP = (34., 510.)
INNER_X_BOTTOM = (108., 437.)
FX = GLASS_W / 550.

def inner_edges(y):
    t = (np.asarray(y) - DATA_TOP) / DATA_HEIGHT
    left = GLASS_X + (INNER_X_TOP[0] * (1-t) + INNER_X_BOTTOM[0]*t) * FX
    right = GLASS_X + (INNER_X_TOP[1] * (1-t) + INNER_X_BOTTOM[1]*t) * FX
    return left, right

bounds = DATA_TOP + np.r_[0, np.cumsum(values)] / 100. * DATA_HEIGHT
centers = (bounds[:-1] + bounds[1:]) / 2

def hexrgb(h):
    return tuple(int(h[i:i+2],16) for i in (1,3,5))

# A quiet editorial background, independent of the data.
yy, xx = np.mgrid[0:H, 0:W]
halo = np.exp(-(((xx-1150)/700)**2 + ((yy-1170)/1050)**2))
bg0 = np.array([247.,249.,247.])
bg1 = np.array([229.,240.,243.])
bg = (bg0[None,None,:] * (1-.63*halo[:,:,None]) + bg1[None,None,:]*(.63*halo[:,:,None])).astype('uint8')
canvas = Image.fromarray(bg).convert('RGBA').resize((W*SCALE,H*SCALE), Image.Resampling.BICUBIC)
draw = ImageDraw.Draw(canvas)
fonts = {}

def font(size, bold=False):
    key = (size,bold)
    if key not in fonts:
        name = 'NimbusSans-Bold.otf' if bold else 'NimbusSans-Regular.otf'
        fonts[key] = ImageFont.truetype(str(ROOT/'assets'/name), round(size*SCALE))
    return fonts[key]

text_records = []
def text_at(s, x, y, size, fill=INK, bold=False, align='left', track=None):
    # Coordinates use a typographic ascender line and are tracked for clipping checks.
    f = font(size,bold)
    if track is None:
        width = draw.textlength(s,font=f)/SCALE
    else:
        width = sum(draw.textlength(c,font=f)/SCALE for c in s) + max(0,len(s)-1)*track
    if align=='right': x -= width
    elif align=='center': x -= width/2
    if track is None:
        draw.text((round(x*SCALE),round(y*SCALE)),s,font=f,fill=fill,anchor='lt')
        box = draw.textbbox((round(x*SCALE),round(y*SCALE)),s,font=f,anchor='lt')
    else:
        xp=x
        for c in s:
            draw.text((round(xp*SCALE),round(y*SCALE)),c,font=f,fill=fill,anchor='lt')
            xp += draw.textlength(c,font=f)/SCALE + track
        box = (x*SCALE,y*SCALE,(x+width)*SCALE,(y+size)*SCALE)
    text_records.append({'text':s,'bbox':[round(v/SCALE,2) for v in box]})
    return width

def line(points, fill, width=1):
    draw.line([(round(x*SCALE),round(y*SCALE)) for x,y in points],fill=fill,width=max(1,round(width*SCALE)),joint='curve')

def circle(x,y,r,fill,outline=None,width=1):
    draw.ellipse([(round((x-r)*SCALE),round((y-r)*SCALE)),(round((x+r)*SCALE),round((y+r)*SCALE))],fill=fill,outline=outline,width=round(width*SCALE))

def paste_asset(name, box):
    x,y,w,h=box
    im=Image.open(ROOT/'assets'/name).convert('RGBA')
    im=im.resize((round(w*SCALE),round(h*SCALE)),Image.Resampling.LANCZOS)
    canvas.alpha_composite(im,(round(x*SCALE),round(y*SCALE)))

# Header: title, metric and unmistakable synthetic-data notice.
draw.rounded_rectangle((100*SCALE,76*SCALE,401*SCALE,120*SCALE),radius=22*SCALE,fill=INK)
text_at(data['visible_notice'],250.5,87,22,fill='#FFFFFF',bold=True,align='center',track=2.0)
text_at('Indoor household',96,164,100,bold=True)
text_at('water use',96,274,100,bold=True)
text_at(data['series'][0]['label'] + ' (%)',100,405,28,fill=MUTED)
line([(100,470),(972,470)],'#B8CBD1',1.4)

# A generated faucet and water droplets connect physically to the cup opening.
paste_asset('faucet.png',(1035,98,542,425.5))
paste_asset('drop_01.png',(1073,538,40,60.4))
paste_asset('drop_02.png',(1081,608,24,35.7))

# Subtle glass contact shadow; decoration only.
shadow = Image.new('RGBA',canvas.size,(0,0,0,0))
sd = ImageDraw.Draw(shadow)
sd.ellipse((859*SCALE,1778*SCALE,1416*SCALE,1831*SCALE),fill=(72,105,115,28))
shadow=shadow.filter(ImageFilter.GaussianBlur(18*SCALE))
canvas.alpha_composite(shadow)

# Liquid as a smooth cylindrical color field, then clipped to the true inside walls.
# Every interface has y = top + cumulative percent * common height / 100.
xlo, xhi = math.floor(GLASS_X), math.ceil(GLASS_X+GLASS_W)
ylo, yhi = int(DATA_TOP), int(DATA_BOTTOM)
y = ylo + (np.arange((yhi-ylo)*SCALE)+.5)/SCALE
x = xlo + (np.arange((xhi-xlo)*SCALE)+.5)/SCALE
l,r=inner_edges(y)
u=(x[None,:]-l[:,None])/(r-l)[:,None]
inside=(u>=0)&(u<=1)
row_ids=np.searchsorted(bounds[1:],y,side='right').clip(0,len(values)-1)
base=np.asarray([hexrgb(c) for c in COLORS],dtype=float)[row_ids]
# Refraction-like horizontal light is not data; boundaries remain perfectly flat.
shade=(1.0 - .09*np.exp(-u/.07) - .095*np.exp(-(1-u)/.07))
light=.11*np.exp(-((u-.17)/.115)**2)+.025*np.exp(-((u-.8)/.18)**2)
rgb=base[:,None,:]*shade[:,:,None]
rgb=rgb*(1-light[:,:,None])+255*light[:,:,None]
liq=np.empty((len(y),len(x),4),dtype='uint8')
liq[:,:,:3]=np.clip(rgb,0,255).astype('uint8')
liq[:,:,3]=inside.astype('uint8')*255
liquid = Image.fromarray(liq,'RGBA')
canvas.alpha_composite(liquid,(xlo*SCALE,ylo*SCALE))

# Generated glass material is warped with the same vertical mapping as the liquid.
glass=np.asarray(Image.open(ROOT/'assets/glass.png').convert('RGBA'),dtype=np.float32)
gh=round((DST_Y[-1]-DST_Y[0])*SCALE)
gw=round(GLASS_W*SCALE)
out_y=DST_Y[0]+(np.arange(gh)+.5)/SCALE
out_x=(np.arange(gw)+.5)/SCALE
src_y=np.interp(out_y,DST_Y,SRC_Y)-.5
src_x=out_x/FX-.5
Y,X=np.meshgrid(src_y,src_x,indexing='ij')
coords=[Y,X]
# Premultiplied interpolation protects transparent cutout edges from dark halos.
a=glass[:,:,3:4]/255.
premul=np.concatenate([glass[:,:,:3]*a,glass[:,:,3:4]],axis=2)
warped=np.stack([map_coordinates(premul[:,:,c],coords,order=1,mode='constant',cval=0) for c in range(4)],axis=2)
aout=warped[:,:,3:4]/255.
warped[:,:,:3]=np.divide(warped[:,:,:3],aout,out=np.zeros_like(warped[:,:,:3]),where=aout>0)
# Preserve source wall/rim/base; attenuate only the glass's interior tint so all
# twelve independently encoded layers remain legible through the clear glass.
fy=out_y[:,None]
fl,fr=inner_edges(out_y)
fx=GLASS_X+out_x[None,:]
interior=(fy>=DATA_TOP)&(fy<DATA_BOTTOM)&(fx>fl[:,None]+30)&(fx<fr[:,None]-30)
warped[:,:,3]=np.where(interior,np.minimum(warped[:,:,3],42),warped[:,:,3])
warped=np.clip(warped,0,255).astype('uint8')
canvas.alpha_composite(Image.fromarray(warped,'RGBA'),(round(GLASS_X*SCALE),round(DST_Y[0]*SCALE)))
# Reattach draw after raster compositing.
draw=ImageDraw.Draw(canvas)

# Non-data bracket: a linear 0–100% height ruler, with no area/volume claim.
ruler_x=1550
line([(ruler_x,DATA_TOP),(ruler_x,DATA_BOTTOM)],'#9DB3BC',1.4)
for p in (100,75,50,25,0):
    ry=DATA_BOTTOM-p/100.*DATA_HEIGHT
    line([(ruler_x-10,ry),(ruler_x+9,ry)],'#6F8F9C',1.6)
    text_at(str(p),ruler_x+18,ry-11,23,fill=MUTED)
text_at('%',ruler_x+18,DATA_TOP-49,24,fill=MUTED)

# Semantic labels in input order. Small layers fan out without changing heights.
text_at('END USE',128,643,20,fill=MUTED,bold=True,track=2.8)
text_at('SHARE',610,643,20,fill=MUTED,bold=True,align='right',track=2.8)
label_y = centers.copy()
# Make space for the final short layers; all leader endpoints keep true midpoints.
label_y[8:] = [1595.,1675.,1741.,1805.]
label_records=[]
for i,(cat,val,cy,ly,color) in enumerate(zip(categories,values,centers,label_y,COLORS)):
    parts=cat.split(' / ')
    parent=parts[0]
    subtitle=parts[1] if len(parts)>1 else None
    circle(108,ly,6.5,color)
    if subtitle:
        text_at(parent,128,ly-25,32,bold=True)
        text_at(subtitle,128,ly+12,23,fill=MUTED)
    else:
        text_at(parent,128,ly-15,32,bold=True)
    number=f'{val:g}%'
    text_at(number,610,ly-16,36,bold=True,align='right')
    left,right=inner_edges(cy)
    anchor=float(left+60)
    # White keyline keeps a connector distinguishable over glass reflections.
    pts=[(636,ly),(706,ly),(float(left-30),float(cy)),(anchor,float(cy))]
    line(pts,'#F5F8F7',4.8)
    line(pts,LINE,1.65)
    circle(anchor,float(cy),4.1,INK,outline='#EAF4F4',width=.8)
    label_records.append({'category':cat,'label_y_px':float(ly),'value_text':number,
                          'anchor_xy_px':[anchor,float(cy)],'layer_top_px':float(bounds[i]),
                          'layer_bottom_px':float(bounds[i+1]),'color':color})

# Encoding and provenance are visible, concise, and directly tied to the diagram.
line([(100,1870),(1580,1870)],'#B8CBD1',1.35)
text_at('HOW TO READ',100,1897,19,bold=True,fill=INK,track=2.0)
text_at('Layer height = share of indoor water use. Glass width and hardware are illustrative.',330,1893,25,fill=MUTED)
text_at('Subcategories are mutually exclusive synthetic parts of their named use. This is not a survey.',100,1940,24,fill=MUTED)

# Final downsampling preserves the exact common-height mapping up to raster precision.
final=canvas.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
final.save(ROOT/'final.png',format='PNG',optimize=True)
(ROOT/'artifacts').mkdir(exist_ok=True)
# Diagnostic geometry is used with final-image pixel measurements; it is not an
# independent judgment or a substitute for checking the actual rendered image.
save_json('artifacts/layout_geometry.json',{
    'canvas_px':[W,H], 'data_top_px':float(DATA_TOP), 'data_bottom_px':float(DATA_BOTTOM),
    'common_data_height_px':float(DATA_HEIGHT), 'encoding':'vertical height only; linear',
    'category_order':'top to bottom, as listed in data.json',
    'layers':label_records, 'text_boxes':text_records,
    'generated_glass_transform':{'source_y':SRC_Y.tolist(),'canvas_y':DST_Y.tolist(),
                                'source_crop_origin':[145,125], 'canvas_x':GLASS_X,'width_px':GLASS_W},
    'frozen_input_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['input.json','brief.json','core.json','data.json','reference.png']}
})
print(f'Saved final.png: {W} x {H}; {len(categories)} proportional layers; sum={values.sum():g}%.')
