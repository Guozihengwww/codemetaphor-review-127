#!/usr/bin/env python3
"""Deterministic local rendering. Run through capture_render.py for experiments."""
from pathlib import Path
from functools import lru_cache
from decimal import Decimal
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / 'data.json').read_text(encoding='utf-8'))
BRIEF = json.loads((ROOT / 'brief.json').read_text(encoding='utf-8'))
W, H = BRIEF['width'], BRIEF['height']
S = 3
BG = (229, 178, 43)
CREAM = (255, 246, 215)
INK = (78, 51, 29)
SHELF = (132, 70, 35)
BASELINE = 1120.0
PLOT_TOP = 140.0
BOTTOM = 1766.0
X_START = 200.0
BOOK_W = 152.0
GAP = 18.0
AXIS_X = 168.0
RIGHT = X_START + 10 * BOOK_W + 9 * GAP
AXIS = DATA['value_axis']
SCALE = (BASELINE - PLOT_TOP) / (AXIS['maximum'] - AXIS['minimum'])
assert AXIS['minimum'] == 0

@lru_cache(None)
def font(name, size):
    return ImageFont.truetype(str(ROOT / 'assets/fonts' / name), round(size * S))

F_SANS = 'NimbusSans-Regular.otf'
F_BOLD = 'NimbusSans-Bold.otf'
F_NARROW = 'NimbusSansNarrow-Regular.otf'
F_NARROW_BOLD = 'NimbusSansNarrow-Bold.otf'
F_SERIF = 'NimbusRoman-Regular.otf'
F_SERIF_BOLD = 'NimbusRoman-Bold.otf'
F_ITALIC = 'NimbusRoman-Italic.otf'

rng = np.random.default_rng(137)
noise = rng.normal(0, 0.9, (H, W, 1))
paper = np.clip(np.asarray(BG)[None,None,:] + noise, 0, 255).astype('uint8')
im = Image.fromarray(paper, 'RGB').resize((W*S, H*S), Image.Resampling.BICUBIC).convert('RGBA')
draw = ImageDraw.Draw(im)
text_records = []
geometry = []

def xy(box):
    return tuple(round(v*S) for v in box)

def line(points, fill=CREAM, width=1):
    draw.line([(round(x*S),round(y*S)) for x,y in points], fill=fill, width=max(1,round(width*S)))

def rect(box, fill=None, outline=None, width=1):
    draw.rectangle(xy(box), fill=fill, outline=outline, width=max(1,round(width*S)))

def text(x,y,value,size=30,face=F_SANS,color=INK,align='left', tracking=None,record=True):
    f=font(face,size)
    if tracking is None:
        box=draw.textbbox((0,0),value,font=f)
        tw=box[2]-box[0]; th=box[3]-box[1]
        px=x*S if align=='left' else x*S-tw if align=='right' else x*S-tw/2
        py=y*S
        draw.text((round(px-box[0]),round(py-box[1])),value,font=f,fill=color)
    else:
        widths=[draw.textlength(c,font=f) for c in value]
        tw=sum(widths)+tracking*S*max(0,len(value)-1)
        box=draw.textbbox((0,0),value,font=f); th=box[3]-box[1]
        px=x*S if align=='left' else x*S-tw if align=='right' else x*S-tw/2
        py=y*S
        cx=px
        for c,cw in zip(value,widths):
            draw.text((round(cx),round(py-box[1])),c,font=f,fill=color)
            cx+=cw+tracking*S
    if record:
        text_records.append({'text':value,'box':[round(px/S,3),round(py/S,3),round((px+tw)/S,3),round((py+th)/S,3)],'font_size_px':size,'rotation':0})
    return (tw/S,th/S)

def fit_size(value,face,size,limit):
    while font(face,size).getlength(value)/S > limit:
        size-=0.5
    return size

def rotated_title(x,y,value,maxlen,size=43,face=F_SERIF,color=CREAM):
    size=fit_size(value,face,size,maxlen)
    f=font(face,size)
    bb=f.getbbox(value)
    layer=Image.new('RGBA',(bb[2]-bb[0]+10*S,bb[3]-bb[1]+10*S),(0,0,0,0))
    ld=ImageDraw.Draw(layer)
    ld.text((5*S-bb[0],5*S-bb[1]),value,font=f,fill=color)
    layer=layer.rotate(-90,expand=True,resample=Image.Resampling.BICUBIC)
    left=round(x*S-layer.width/2)
    top=round(y*S)
    im.alpha_composite(layer,(left,top))
    text_records.append({'text':value,'box':[left/S,top/S,(left+layer.width)/S,(top+layer.height)/S],'font_size_px':size,'rotation':90})
    return layer.height/S

def ornament(cx,cy,span=43,color=CREAM):
    line([(cx-span,cy),(cx-10,cy)],color,0.8)
    line([(cx+10,cy),(cx+span,cy)],color,0.8)
    draw.polygon(xy([cx,cy-5,cx+5,cy,cx,cy+5,cx-5,cy]),fill=color)

# Left scale: ticks always come from data.json. A thin grid stays behind the books.
for tick in AXIS['ticks']:
    y=BASELINE-float(tick)*SCALE
    if tick > 0:
        # Short, quiet ticks preserve the reference's open yellow field.
        line([(AXIS_X-18,y),(AXIS_X,y)],(249,230,162),1.5)
    text(AXIS_X-32,y-10,str(tick),25,F_SANS,INK,align='right')
text(95,1160,'million',23,F_SANS,INK,align='center')
text(95,1190,'copies',23,F_SANS,INK,align='center')
text(95,1240,'0–40',23,F_SANS,INK,align='center')

# Upper-right poster title and data disclosure.
rect((1406,86,RIGHT,145),fill=INK)
text((1406+RIGHT)/2,101,DATA['data_status'],27,F_BOLD,CREAM,align='center',tracking=2.1)
text(RIGHT,224,'SYNTHETIC',136,F_BOLD,CREAM,align='right')
text(RIGHT,377,'BOOK SALES',155,F_BOLD,CREAM,align='right')
line([(985,563),(RIGHT,563)],INK,1.2)
text(RIGHT,598,DATA['units'],43,F_SERIF,INK,align='right')
text(RIGHT,660,'Illustrative cumulative copies sold',31,F_SANS,INK,align='right')
text(RIGHT,702,f"during a fictional {DATA['time_window_years']}-year period.",31,F_SANS,INK,align='right')
text(RIGHT,802,'Height above zero = copies sold.',26,F_SANS,INK,align='right')
text(RIGHT,840,'Equal lower sections carry titles and authors.',26,F_SANS,INK,align='right')

# A front-facing shelf: its top is below every common non-data base.
rect((0,BOTTOM,W,H),fill=SHELF)
rect((0,BOTTOM,W,BOTTOM+10),fill=(103,54,28))
rect((0,BOTTOM+10,W,BOTTOM+15),fill=(160,89,45))
line([(0,H-7),(W,H-7)],(104,54,27),2)

cats = {c['id']:c for c in DATA['categories']}
values = dict(zip(DATA['category_order'],DATA['series'][0]['values']))
for index,category_id in enumerate(DATA['category_order']):
    cat=cats[category_id]
    value=values[category_id]
    x=X_START+index*(BOOK_W+GAP)
    cx=x+BOOK_W/2
    top=BASELINE-float(value)*SCALE
    # One rectangle, one shared transform: material fills both quantitative and base zones.
    top_i=round(top*S); bottom_i=round(BOTTOM*S)
    left_i=round(x*S); right_i=round((x+BOOK_W)*S)
    material=Image.open(ROOT/f'assets/spine_{index+1:02d}.png').convert('RGBA')
    material=material.resize((right_i-left_i,bottom_i-top_i),Image.Resampling.LANCZOS)
    im.alpha_composite(material,(left_i,top_i))
    # Head rule is fully inset. It never adds to the encoded height.
    line([(x+1,top+1),(x+BOOK_W-1,top+1)],(206,181,126),0.7)
    # The data/background edge is created above, not inferred from the raster asset.
    geom={'id':category_id,'value':value,'display_value':format(Decimal(str(value)),'f'),
          'x_left':x,'x_right':x+BOOK_W,'y_top_target':top,'y_top_raster':top_i/S,
          'y_zero':BASELINE,'y_bottom':BOTTOM,'data_height_target':float(value)*SCALE,
          'data_height_raster':BASELINE-top_i/S,'source_asset':f'assets/spine_{index+1:02d}.png',
          'spine_color':cat['spine_color'],'title_layout':cat['title_layout']}
    geometry.append(geom)
    # All book identity typography and decorative rules are below zero.
    text(cx,1155,cat['label'],21,F_SANS,CREAM,align='center',tracking=0.5)
    line([(x+16,1191),(x+BOOK_W-16,1191)],CREAM,0.75)
    if cat['title_layout']=='rotated_90_degrees':
        rotated_title(cx,1221,cat['spine_title'],407,size=47 if index==2 else 43,
                      face=F_SERIF if index in (1,2,4,5,9) else F_SERIF_BOLD)
    elif cat['title_layout']=='stacked_in_frames':
        rect((x+10,1212,x+BOOK_W-10,1286),outline=CREAM,width=1.4)
        rect((x+15,1217,x+BOOK_W-15,1281),outline=CREAM,width=0.5)
        text(cx,1235,'THE',30,F_SERIF,CREAM,align='center')
        rect((x+10,1299,x+BOOK_W-10,1467),outline=CREAM,width=1.4)
        rect((x+15,1304,x+BOOK_W-15,1462),outline=CREAM,width=0.5)
        text(cx,1336,'SILVER',32,F_SERIF,CREAM,align='center')
        text(cx,1385,'PATH',32,F_SERIF,CREAM,align='center')
        rect((x+10,1480,x+BOOK_W-10,1605),outline=CREAM,width=1.4)
        ornament(cx,1542,42)
    else:
        words=cat['spine_title'].split()
        if index==0:
            for k,word in enumerate(words):
                fs=fit_size(word,F_SERIF_BOLD,36,BOOK_W-25)
                text(cx,1252+k*48,word,fs,F_SERIF_BOLD,CREAM,align='center')
            ornament(cx,1415,42)
        elif index==7:
            for k,word in enumerate(words):
                fs=fit_size(word,F_NARROW_BOLD,43,BOOK_W-26)
                text(cx,1234+k*68,word,fs,F_NARROW_BOLD,CREAM,align='center')
            ornament(cx,1534,42)
        else:
            for k,word in enumerate(words):
                fs=fit_size(word,F_NARROW_BOLD,42,BOOK_W-28)
                text(cx,1234+k*68,word,fs,F_NARROW_BOLD,CREAM,align='center')
            ornament(cx,1534,42)
    line([(x+10,1672),(x+BOOK_W-10,1672)],CREAM,1.1)
    line([(x+10,1678),(x+BOOK_W-10,1678)],CREAM,0.55)
    author_size=fit_size(cat['spine_byline'],F_SERIF_BOLD,22,BOOK_W-25)
    text(cx,1706,cat['spine_byline'],author_size,F_SERIF_BOLD,CREAM,align='center')
    # Fixed source precision, not rounded to a shared arbitrary number of decimals.
    label=geom['display_value']
    if value <= 1.5:
        label_y=1033
        text(cx,label_y,label,42,F_BOLD,INK,align='center')
        line([(cx,1078),(cx,top-5)],INK,1.1)
        line([(cx-5,top-5),(cx+5,top-5)],INK,1.1)
    else:
        text(cx,top-56,label,44,F_BOLD,INK,align='center')

# This is the sole chart zero line. It crosses gaps as well as the books.
line([(AXIS_X-18,BASELINE),(RIGHT+10,BASELINE)],CREAM,2.1)
# Footer copy stays on the shelf, with no real-world sales claim.
text(W/2,1810,'TEN FICTIONAL TITLES  /  ONE SHARED LINEAR SCALE',24,F_BOLD,CREAM,align='center',tracking=1.4)

im=im.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
im.save(ROOT/'final.png',optimize=True)
art=ROOT/'artifacts'
art.mkdir(exist_ok=True)
(art/'geometry.json').write_text(json.dumps({'canvas':[W,H],'baseline_y':BASELINE,'plot_top_y':PLOT_TOP,
    'scale_pixels_per_million':SCALE,'decoration_height':BOTTOM-BASELINE,
    'axis':AXIS,'books':geometry,'text':text_records},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(f'Rendered {W}x{H}; {len(geometry)} books; shared scale {SCALE:g} px / million copies.')
