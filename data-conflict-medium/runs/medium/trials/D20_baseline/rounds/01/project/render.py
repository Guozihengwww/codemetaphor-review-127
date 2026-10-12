#!/usr/bin/env python3
"""Regenerate final.png from local data and generated assets, without network/model calls.

The nose uses one common linear mapping; each aircraft is transformed uniformly
by sqrt(value / reference_value). Visible masks are measured after rasterization.
Run the complete renderer through capture_render.py for the frozen experiment.
"""
from pathlib import Path
from decimal import Decimal
import hashlib
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / 'data.json').read_text(encoding='utf-8'))
BRIEF = json.loads((ROOT / 'brief.json').read_text(encoding='utf-8'))
GEOM = json.loads((ROOT / 'assets/aircraft_geometry.json').read_text(encoding='utf-8'))
W, H = BRIEF['width'], BRIEF['height']
S = 2
X0, X1 = 470.0, 1765.0
AXIS_Y = 423.0
ROW_Y = [560.0 + 188.0*i for i in range(len(DATA['categories']))]
V_MIN, V_MAX = DATA['value_axis']['minimum'], DATA['value_axis']['maximum']
VALUES = DATA['series'][0]['values']
MAX_VALUE = max(VALUES)
BASE_WINGSPAN = 220.0
COL = {
    'navy': (10, 24, 42),
    'ink': (20, 40, 59),
    'paper': (240, 242, 237),
    'white': (245, 247, 245),
    'gold': (247, 221, 161),
    'muted': (162, 185, 203),
    'grid': (52, 78, 101),
}
FONTS = {
    'regular': ROOT / 'assets/fonts/NimbusSans-Regular.otf',
    'bold': ROOT / 'assets/fonts/NimbusSans-Bold.otf',
    'cjk': ROOT / 'assets/fonts/NotoSansCJK-Regular.ttc',
    'cjk_bold': ROOT / 'assets/fonts/NotoSansCJK-Bold.ttc',
}


def font(size, face='regular'):
    return ImageFont.truetype(str(FONTS[face]), round(size*S))


def label(text, x, y, size, color, face='regular', anchor='lm'):
    draw.text((round(x*S),round(y*S)), text, font=font(size,face), fill=color,
              anchor=anchor, stroke_width=0)


def line(coords, fill, width=1):
    draw.line([round(q*S) for q in coords], fill=fill, width=max(1,round(width*S)))


def rectangle(coords, fill, outline=None, width=1, radius=0):
    box=[round(q*S) for q in coords]
    if radius:
        draw.rounded_rectangle(box,round(radius*S),fill,outline,width=round(width*S))
    else:
        draw.rectangle(box,fill,outline,width=round(width*S))


def tracked(text, x, y, size, fill, tracking=2.5, face='bold'):
    f=font(size,face)
    for ch in text:
        draw.text((round(x*S),round(y*S)),ch,font=f,fill=fill,anchor='lm')
        x += draw.textlength(ch,font=f)/S+tracking
    return x


def formatted(v):
    # Preserve the input's explicit one-decimal values, without rounding them.
    return str(v)


def endpoint(v):
    return X0 + (v - V_MIN)/(V_MAX - V_MIN)*(X1-X0)


def dash_vertical(x, y0, y1, color, dash=4, gap=10):
    for y in np.arange(y0,y1,dash+gap):
        line((x,y,x,min(y+dash,y1)),color,0.7)


# Background plate remains visible as a map, with restrained texture.
bg = Image.open(ROOT / 'assets/raw_02.png').convert('RGB').resize((W*S,H*S),Image.Resampling.LANCZOS)
a = np.asarray(bg,dtype=np.float32)
y = np.arange(H*S,dtype=np.float32)[:,None,None]/S
shade = 0.28 + 0.39*np.exp(-((y-120)/260)**2) + 0.30*np.exp(-((y-1920)/200)**2)
a = a*(1-shade) + np.array(COL['navy'],dtype=np.float32)[None,None,:]*shade
im = Image.fromarray(np.uint8(np.clip(a,0,255))).convert('RGBA')
del a,bg
draw = ImageDraw.Draw(im)

# Header: the synthetic notice and the supplied title are explicit.
rectangle((96,72,107,95),COL['gold'])
tracked(DATA['visible_notice'],124,85,25,COL['gold'],2.6)
right_header='AVIATION / MARKET CAPITALIZATION'
label(right_header,1904,85,24,COL['muted'],anchor='rm')
line((96,126,1904,126),(65,87,105),1)
label(DATA['title'],96,222,84,COL['white'],'cjk_bold')
label(DATA['series'][0]['label'],99,315,38,COL['muted'])
label(DATA['series'][0]['unit'],1904,315,31,COL['gold'],'bold',anchor='rm')

# A common quantitative axis, visible on the top, with quiet vertical guides.
tracked('AIRLINE',96,AXIS_Y-20,24,COL['muted'],2.8)
for tick in range(int(V_MIN),int(V_MAX)+1,5):
    x=endpoint(tick)
    label(str(tick),x,AXIS_Y-29,28,COL['muted'],anchor='mm')
    line((x,AXIS_Y-4,x,AXIS_Y+7),(105,131,150),1)
    if tick==0:
        line((x,AXIS_Y+16,x,1780),(118,142,159),1.3)
    else:
        dash_vertical(x,AXIS_Y+17,1780,COL['grid'])
line((X0,AXIS_Y,X1,AXIS_Y),(79,108,131),1)

plane = Image.open(ROOT/'assets/aircraft.png').convert('RGBA')
plane_np=np.asarray(plane)
base_scale=BASE_WINGSPAN/plane.height
records=[]
label_boxes=[]
mask_dir=ROOT/'artifacts/masks'
mask_dir.mkdir(parents=True,exist_ok=True)

for i,(category,v,cy) in enumerate(zip(DATA['categories'],VALUES,ROW_Y)):
    nose=endpoint(v)
    uniform_scale=base_scale*math.sqrt(v/MAX_VALUE)
    left=nose-GEOM['nose_x']*uniform_scale
    top=cy-GEOM['nose_y']*uniform_scale
    body_end=nose-plane.width*uniform_scale*0.13

    # Fade the broad white wake into the tail/body, behind the plane.
    # This wake is excluded from area measurements.
    wake_left=X0
    wake_right=body_end
    wake_h=82.0
    box=(round(wake_left*S),round((cy-wake_h/2)*S),round(wake_right*S),round((cy+wake_h/2)*S))
    ww,hh=box[2]-box[0],box[3]-box[1]
    q=np.linspace(0,1,ww,dtype=np.float32)[None,:]
    vertical=np.minimum(np.linspace(0,1,hh)[:,None],np.linspace(1,0,hh)[:,None])
    soft=np.clip(vertical*hh/3.5,0,1)
    alpha=224*(1-q)**1.26*soft
    wake=np.zeros((hh,ww,4),dtype=np.uint8)
    wake[...,:3]=(242,246,247)
    wake[...,3]=alpha.astype(np.uint8)
    im.alpha_composite(Image.fromarray(wake),dest=box[:2])
    # Twin restrained vapor streaks make the wake read as flight, not a separate icon.
    st=Image.new('RGBA',im.size,(0,0,0,0))
    sd=ImageDraw.Draw(st)
    streak_end=left+plane.width*uniform_scale*0.59
    for offset in (-11,11):
        sd.line((round((X0+6)*S),round((cy+offset)*S),round(streak_end*S),round((cy+offset)*S)),
                fill=(239,248,251,35),width=round(1.1*S))
    im.alpha_composite(st)
    del st,sd

    # Exact affine scaling uses the SAME scalar in x/y; rounding only bounds the patch.
    bx0=math.floor(left)-5; by0=math.floor(top)-5
    bx1=math.ceil(left+plane.width*uniform_scale)+5
    by1=math.ceil(top+plane.height*uniform_scale)+5
    patch_size=((bx1-bx0)*S,(by1-by0)*S)
    inv=1/(uniform_scale*S)
    matrix=(inv,0,(bx0-left)/uniform_scale,0,inv,(by0-top)/uniform_scale)
    patch=plane.transform(patch_size,Image.Transform.AFFINE,matrix,
                          resample=Image.Resampling.BICUBIC)
    # No detached plane shadow: the entity mask corresponds only to the aircraft.
    im.alpha_composite(patch,dest=(bx0*S,by0*S))

    # Save the actual final-resolution entity alpha for measurable visible geometry.
    mask=patch.getchannel('A').resize((bx1-bx0,by1-by0),Image.Resampling.LANCZOS)
    mask_path=f'artifacts/masks/aircraft_{i+1:02d}.png'
    mask.save(ROOT/mask_path)
    ma=np.asarray(mask)
    yy,xx=np.where(ma>=128)
    maxx=int(xx.max());nose_y_pixels=yy[xx==maxx]
    measured_x=bx0+maxx+0.5
    measured_y=by0+float(nose_y_pixels.mean())+0.5
    area=float(ma.sum()/255)
    body_area=int((ma>=128).sum())
    bounds=[bx0+int(xx.min()),by0+int(yy.min()),bx0+int(xx.max())+1,by0+int(yy.max())+1]

    draw=ImageDraw.Draw(im)
    # One category identification block touches the common zero / wake origin.
    rectangle((96,cy-42,X0,cy+42),COL['paper'])
    rectangle((96,cy-42,177,cy+42),COL['gold'])
    code=category.rsplit(' ',1)[-1]
    label(code,136.5,cy+1,38,COL['ink'],'bold',anchor='mm')
    label(category,200,cy+1,39,COL['ink'],'bold')

    # Mandatory amount labels stay clear of the measured nose.
    text=f'${formatted(v)}'
    tx=nose+28
    f=font(58,'bold')
    label(text,tx,cy+2,58,COL['gold'],'bold')
    tw=draw.textlength(text,font=f)/S
    label('Bn',tx+tw+11,cy+7,31,COL['gold'],'bold')
    total_width=tw+11+draw.textlength('Bn',font=font(31,'bold'))/S
    label_boxes.append([tx,cy-38,tx+total_width,cy+40])
    records.append({
        'category':category, 'value':v,
        'intended_nose_x':nose,'measured_nose_x':measured_x,'measured_nose_y':measured_y,
        'intended_row_y':cy,'nose_error_px':measured_x-nose,
        'recovered_value_from_visible_nose':(measured_x-X0)/(X1-X0)*(V_MAX-V_MIN)+V_MIN,
        'uniform_scale':uniform_scale,
        'alpha_area_px2':area,'opaque_area_px2_threshold128':body_area,
        'visible_bounds':bounds,'mask_path':mask_path,'mask_origin':[bx0,by0],
        'amount_label':text+' Bn','amount_label_bounds':label_boxes[-1],
        'amount_label_nose_gap_px':tx-measured_x,
    })

# Compact decoding note: both channels represent the same single supplied series.
draw=ImageDraw.Draw(im)
line((96,1830,1904,1830),(65,88,107),1)
tracked('HOW TO READ',96,1877,23,COL['gold'],2.7)
label('机鼻距零点的水平距离 ∝ 市值；飞机实体面积 ∝ 市值。',365,1875,29,COL['paper'],'cjk')
label('Linear scale: 0–35  /  One shared aircraft template  /  Input order preserved',96,1930,26,COL['muted'])
label('7 AIRLINES',1904,1930,25,COL['muted'],'bold',anchor='rm')

# Save once. The wrapper is the sole full-render entry point used in this experiment.
final=im.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
final.save(ROOT/'final.png',optimize=True)

base_area=records[0]['alpha_area_px2']
base_binary_area=records[0]['opaque_area_px2_threshold128']
for row in records:
    expected=row['value']/VALUES[0]
    row['expected_area_ratio']=expected
    row['measured_alpha_area_ratio']=row['alpha_area_px2']/base_area
    row['measured_threshold_area_ratio']=row['opaque_area_px2_threshold128']/base_binary_area
    row['relative_alpha_ratio_error']=row['measured_alpha_area_ratio']/expected-1
    row['relative_threshold_ratio_error']=row['measured_threshold_area_ratio']/expected-1

geometry={
    'canvas':[W,H], 'axis':{'zero_x':X0,'maximum_x':X1,'minimum':V_MIN,'maximum':V_MAX,'px_per_unit':(X1-X0)/(V_MAX-V_MIN)},
    'template':{'path':'assets/aircraft.png','raw_path':'assets/raw_01.png','size':[plane.width,plane.height],
                'opaque_area_px2':GEOM['opaque_area_px'],'nose_anchor':[GEOM['nose_x'],GEOM['nose_y']]},
    'area_excludes':['wake','background','category_tile','amount_label'],
    'measurements':records,
    'max_endpoint_error_full_scale_fraction':max(abs(r['nose_error_px'])/(X1-X0) for r in records),
    'max_relative_alpha_area_error':max(abs(r['relative_alpha_ratio_error']) for r in records),
    'max_relative_threshold_area_error':max(abs(r['relative_threshold_ratio_error']) for r in records),
    'all_amount_labels_in_canvas':all(r['amount_label_bounds'][2]<W-60 for r in records),
    'all_amount_labels_clear_of_nose':all(r['amount_label_nose_gap_px']>25 for r in records),
    'data_sha256':hashlib.sha256((ROOT/'data.json').read_bytes()).hexdigest(),
    'final_sha256':hashlib.sha256((ROOT/'final.png').read_bytes()).hexdigest(),
}
(ROOT/'artifacts/geometry_checks.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:geometry[k] for k in ('canvas','max_endpoint_error_full_scale_fraction','max_relative_alpha_area_error',
      'max_relative_threshold_area_error','all_amount_labels_in_canvas','all_amount_labels_clear_of_nose')},ensure_ascii=False))
