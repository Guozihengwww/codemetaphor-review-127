"""Offline infographic renderer. Run only through capture_render.py during this study."""
from __future__ import annotations
from pathlib import Path
import json
import math
from decimal import Decimal
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT/'data.json').read_text(encoding='utf-8'), parse_float=Decimal)
BRIEF = json.loads((ROOT/'brief.json').read_text(encoding='utf-8'))
CORE = json.loads((ROOT/'core.json').read_text(encoding='utf-8'))
INPUT = json.loads((ROOT/'input.json').read_text(encoding='utf-8'))
W, H = BRIEF['width'], BRIEF['height']
SS = 4
PAPER = '#f7f2e6'
TEAL = '#43c9c3'
DARK = '#123d3e'
RED = '#b73c31'
GRID = '#8bded8'
BASE = 1100.0
PLOT_TOP = 300.0
MAXIMUM = float(DATA['value_axis']['maximum'])
SCALE = (BASE-PLOT_TOP)/MAXIMUM
CENTERS = [194.0 + 86.0*i for i in range(len(DATA['categories']))]
GLASS_W = 56.0
assert (W,H)==(2000,1540)
assert DATA['value_axis']['scale']=='linear' and DATA['value_axis']['minimum']==0
assert len(CENTERS)==20
assert CORE['topic_zh']==DATA['topic_zh']
assert INPUT['critical_metaphor_contract']=='core.json'
# The source reference remains an offline input dependency; no source pixels are copied.
assert (ROOT/INPUT['reference']).is_file()

canvas = Image.new('RGBA', (W*SS,H*SS), PAPER)
draw = ImageDraw.Draw(canvas)
ART = ROOT/'artifacts'
(ART/'geometry').mkdir(parents=True,exist_ok=True)

@lru_cache(None)
def font(size, bold=False, narrow=False):
    name = 'NimbusSans' + ('Narrow' if narrow else '') + ('-Bold.otf' if bold else '-Regular.otf')
    return ImageFont.truetype(str(ROOT/'assets/fonts'/name), int(round(size*SS)))

def line(points, fill, width=1):
    draw.line([(round(x*SS),round(y*SS)) for x,y in points],fill=fill,width=max(1,round(width*SS)))

def rect(box, fill, radius=0, outline=None, width=1):
    box = tuple(round(x*SS) for x in box)
    if radius:
        draw.rounded_rectangle(box, radius=round(radius*SS),fill=fill,outline=outline,width=round(width*SS))
    else:
        draw.rectangle(box,fill=fill,outline=outline,width=round(width*SS))

text_records=[]
def text(s, x,y, size=20, fill=DARK, bold=False,narrow=False,align='left', tracking=0):
    s=str(s); f=font(size,bold,narrow)
    box=f.getbbox(s)
    tw=f.getlength(s)+tracking*SS*max(0,len(s)-1)
    xx=x*SS-(tw/2 if align=='center' else tw if align=='right' else 0)
    yy=y*SS-box[1]
    if tracking:
        for c in s:
            draw.text((round(xx),round(yy)),c,font=f,fill=fill)
            xx+=f.getlength(c)+tracking*SS
    else:
        draw.text((round(xx),round(yy)),s,font=f,fill=fill)
    text_records.append({'text':s,'x':x,'y':y,'font_size':size,'width':float(tw/SS)})

def vertical_value(s,cx,y):
    f=font(27,True,True)
    b=f.getbbox(s)
    tile=Image.new('RGBA',(math.ceil(f.getlength(s))+12*SS,b[3]-b[1]+8*SS),(0,0,0,0))
    td=ImageDraw.Draw(tile)
    td.text((6*SS,4*SS-b[1]),s,font=f,fill=DARK)
    tile=tile.transpose(Image.Transpose.ROTATE_270)
    canvas.alpha_composite(tile,(round(cx*SS-tile.width/2),round(y*SS)))
    text_records.append({'text':s,'center_x':cx,'y':y,'font_size':27,'rotation_degrees':90})

@lru_cache(None)
def asset(name):
    return Image.open(ROOT/'assets'/name).convert('RGBA')

def place_asset(name,cx,top,height):
    im=asset(name)
    width=height*im.width/im.height
    im=im.resize((round(width*SS),round(height*SS)),Image.Resampling.LANCZOS)
    canvas.alpha_composite(im,(round((cx-width/2)*SS),round(top*SS)))
    return [cx-width/2,top,cx+width/2,top+height]

source=np.asarray(asset('glass.png'),dtype=np.float32)/255.
source_pm=source.copy()
source_pm[:,:,:3]*=source_pm[:,:,3:4]
sh,sw=source.shape[:2]

def glass_image(height,top,cx):
    """Map full generated glass to the data height. No minimum size is imposed."""
    # Denser sampling only improves integration for tiny marks; it does not enlarge them.
    ss=16 if height<20 else SS
    ix0=math.floor((cx-GLASS_W/2)*ss)
    ix1=math.ceil((cx+GLASS_W/2)*ss)
    iy0=math.floor(top*ss)
    iy1=math.ceil(BASE*ss)
    row_top=np.arange(iy0,iy1,dtype=np.float64)/ss
    coverage=np.clip((np.minimum(row_top+1/ss,BASE)-np.maximum(row_top,top))*ss,0,1)
    local=np.clip(row_top+0.5/ss-top,0,height)
    foam=min(18.,height*.20)
    base=min(9.,height*.12)
    liquid=height-foam-base
    ys=np.where(local<foam,local/foam*150,
        np.where(local<height-base,150+(local-foam)/liquid*(815-150),
                 815+(local-(height-base))/base*((sh-1)-815)))
    xs=((np.arange(ix0,ix1)+.5)/ss-(cx-GLASS_W/2))/GLASS_W*(sw-1)
    xx,yy=np.meshgrid(xs,ys)
    sampled=np.stack([map_coordinates(source_pm[:,:,k],[yy,xx],order=1,mode='nearest') for k in range(4)],axis=2)
    sampled*=coverage[:,None,None]
    a=sampled[:,:,3:4]
    sampled[:,:,:3]=np.divide(sampled[:,:,:3],a,out=np.zeros_like(sampled[:,:,:3]),where=a>1e-8)
    im=Image.fromarray(np.round(np.clip(sampled,0,1)*255).astype(np.uint8),'RGBA')
    # Align the integer output-pixel crop before reducing; preserves subpixel origin.
    box=(math.floor(cx-GLASS_W/2),math.floor(top),math.ceil(cx+GLASS_W/2),math.ceil(BASE))
    tile=Image.new('RGBA',((box[2]-box[0])*ss,(box[3]-box[1])*ss),(0,0,0,0))
    tile.alpha_composite(im,(ix0-box[0]*ss,iy0-box[1]*ss))
    if ss!=SS:
        tile=tile.resize(((box[2]-box[0])*SS,(box[3]-box[1])*SS),Image.Resampling.LANCZOS)
    return tile,box,foam,base

# Editorial heading.
rect((62,38,297,76),RED,8)
text(DATA['visible_notice'],179.5,49,19,PAPER,True,align='center',tracking=1.25)
text('LOCAL STUDY  /  SYNTHETIC REGIONS',1938,50,17,DARK,align='right',tracking=1.4)
text('Annual beer consumption',60,98,88,DARK,True)
text(DATA['series'][0]['label']+'  ·  '+DATA['series'][0]['unit'],64,200,26,DARK)
text('20',1829,95,94,DARK,True,True,align='center')
text('REGIONS',1829,191,22,DARK,True,align='center',tracking=3)
line([(1456,107),(1456,213)],'#cecfbf',1.25)
text('A full glass,',1490,117,28,DARK,True)
text('a year of volume.',1490,154,28,DARK,True)
text('One shared linear scale.',1490,199,18,DARK)

# Turquoise chart field and unified scale.
rect((40,246,1960,1440),TEAL)
text('L/person/year',77,271,18,DARK,True)
for tick in range(0,151,25):
    yy=BASE-tick*SCALE
    line([(144,yy),(1896,yy)], GRID if tick else '#209b98',1 if tick else .7)
    line([(137,yy),(144,yy)],DARK,.9)
    text(str(tick),122,yy-8,19,DARK,align='right')

# Reading note, positioned in the open area above the falling glass series.
text('HOW TO READ',432,334,17,DARK,True,tracking=2.6)
text('The whole glass',432,375,33,DARK,True)
text('carries the number.',432,416,33,DARK,True)
for k,s in enumerate(['Foam + golden liquid + glass base', 'all fit inside the data height.', 'Small values stay small.']):
    text(s,432,474+k*31,21,DARK)
line([(432,593),(740,593)],'#1c9e9b',1)
text(DATA['categories'][0],432,622,21,DARK)
text(str(DATA['series'][0]['values'][0]),432,660,64,DARK,True,True)
text('L/person/year',432,733,18,DARK)

# Non-quantitative bottle annotations, with exact category binding.
text('BOTTLE NOTES',874,288,19,DARK,True,tracking=2.5)
text('Illustrative bottles · sizes do not encode consumption',874,322,18,DARK)
bottle_records=[]
colors={'green':'bottle_green.png','near-black':'bottle_black.png','brown':'bottle_brown.png'}
for i,b in enumerate(DATA['bottle_annotations']):
    if i<6:
        cx=917+i*169; top=374
    else:
        cx=1001+(i-6)*169; top=611
    bbox=place_asset(colors[b['glass_color']],cx,top,135)
    text(b['label'],cx,top+150,19,DARK,True,align='center')
    text('Region '+b['category_id'],cx,top+179,20,DARK,align='center')
    bottle_records.append({'id':b['id'],'category':'Region '+b['category_id'],'glass_color':b['glass_color'],'bbox':bbox,'quantitative_encoding':False})

# The comparison is the same data-to-y function as every glass.
comp=DATA['comparison']
comp_y=BASE-float(comp['value'])*SCALE
for x in np.arange(144,1896,24):
    line([(float(x),comp_y),(min(float(x)+14,1896),comp_y)],RED,2.8)
rect((1321,863,1894,954),PAPER,10)
text(comp['label']+'  ·  '+str(comp['value'])+' '+comp['unit'],1342,881,26,RED,True)
text('Comparison outside the 20 regions',1342,921,19,DARK)
line([(1871,comp_y),(1871,863)],RED,1.5)

# Bars are whole generated glasses, adapted by a single numerical transform.
glasses=[]
for i,(cat,val,cx) in enumerate(zip(DATA['categories'],DATA['series'][0]['values'],CENTERS)):
    height=float(val)*SCALE
    top=BASE-height
    tile,box,foam,base=glass_image(height,top,cx)
    # Background crop retained for actual final-image difference measurement.
    checkbox=(box[0]-3,box[1]-3,box[2]+3,box[3]+3)
    before=canvas.crop(tuple(v*SS for v in checkbox)).resize((checkbox[2]-checkbox[0],checkbox[3]-checkbox[1]),Image.Resampling.LANCZOS).convert('RGB')
    before.save(ART/'geometry'/f'background_{i+1:02d}.png')
    canvas.alpha_composite(tile,(box[0]*SS,box[1]*SS))
    tile.getchannel('A').resize((box[2]-box[0],box[3]-box[1]),Image.Resampling.LANCZOS).save(ART/'geometry'/f'alpha_{i+1:02d}.png')
    glasses.append({'category':cat,'value':str(val),'center_x':cx,'baseline_y':BASE,'top_y':top,'height_px':height,'foam_height_px':foam,'base_height_px':base,'bbox':list(box),'check_bbox':list(checkbox)})

# Values and category names remain readable even when the data mark is tiny.
line([(144,1119),(1896,1119)],'#2aafaa',.8)
for i,(cat,val,cx) in enumerate(zip(DATA['categories'],DATA['series'][0]['values'],CENTERS)):
    parent, name = cat.rsplit(' ', 1)
    text(parent.upper(),cx,1134,14,DARK,align='center',tracking=.8)
    text(name,cx,1155,24,DARK,True,align='center')
    vertical_value(str(val),cx,1187)
    place_asset('person.png',cx+1,1334,88)
text('VALUES',88,1211,14,DARK,True,align='center',tracking=.5)
text('L/person/year',88,1240,12,DARK,align='center')

# Footer: explicit statistical scope and treatment of non-data elements.
text('0–150 linear scale · foam and base included · bottle and person sizes are illustrative',62,1470,19,DARK)
text('All regional figures are synthetic.',1938,1470,19,RED,True,align='right')
text('Exact values appear below each glass; sub-pixel details in the smallest glasses may not resolve.',62,1504,17,'#536967')
text('A–T / supplied order',1938,1504,17,'#536967',align='right')

final=canvas.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
final.save(ROOT/'final.png')
layout={'canvas':[W,H],'scale_px_per_liter':SCALE,'axis_minimum':0,'axis_maximum':MAXIMUM,'baseline_y':BASE,'plot_top_y':PLOT_TOP,'glasses':glasses,'comparison':{'category':comp['label'],'value':str(comp['value']),'line_y':comp_y,'same_axis':True},'bottles':bottle_records,'person_count':len(CENTERS),'texts':text_records,'asset_geometry':{'foam_source_end':150,'base_source_start':815,'source_height':sh,'supersampling':SS,'small_glass_sampling':16}}
(ART/'render_geometry.json').write_text(json.dumps(layout,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Rendered final.png at',W,H,'with',len(glasses),'data glasses and',len(bottle_records),'bottle annotations.')
