#!/usr/bin/env python3
"""Offline train infographic. Run only through capture_render.py in this trial."""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
INPUT = json.loads((ROOT / 'input.json').read_text(encoding='utf-8'))
DATA = json.loads((ROOT / INPUT['data']).read_text(encoding='utf-8'))
BRIEF = json.loads((ROOT / INPUT['brief']).read_text(encoding='utf-8'))
CORE = json.loads((ROOT / INPUT['critical_metaphor_contract']).read_text(encoding='utf-8'))
# Keep every frozen dependency available; reference pixels are never composited.
assert (ROOT / INPUT['reference']).is_file()
META = json.loads((ROOT / 'assets/train_metadata.json').read_text(encoding='utf-8'))
assert CORE['semantic_roles_zh'] == ['Synthetic peak speed (km/h)']
assert DATA['series'][0]['unit'] == 'km/h'
assert DATA['value_axis'] == {'minimum': 0, 'maximum': 600, 'scale': 'linear'}
assert len(DATA['categories']) == len(DATA['series'][0]['values']) == len(META)
assert sum(DATA['series'][0]['values']) == DATA['series_totals']['synthetic_peak_speed']

W, H = BRIEF['width'], BRIEF['height']
S = 2
PAPER = '#F5F4EE'
INK = '#173A3D'
MUTED = '#667574'
GRID = '#D9DFD8'
ACCENT = '#205F62'
X0, X1 = 180.0, 1870.0
TOP_AXIS, BOTTOM_AXIS = 422, 1670
ROW0, PITCH, BODY_HEIGHT = 450, 149, 80
VALUES = DATA['series'][0]['values']
CONVERSION = DATA['conversion']['km_per_mile']
HOURS = DATA['conversion']['distance_comparison_hours']

def x_for_value(v):
    a = DATA['value_axis']
    return X0 + (v - a['minimum']) / (a['maximum'] - a['minimum']) * (X1-X0)

canvas = Image.new('RGB', (W*S, H*S), PAPER)
draw = ImageDraw.Draw(canvas)
fonts = {}
def font(size, weight='regular'):
    key=(size,weight)
    if key not in fonts:
        fonts[key]=ImageFont.truetype(str(ROOT / f'assets/fonts/{weight}.ttf'), round(size*S))
    return fonts[key]

def text(x,y,label,size=28,fill=INK,weight='regular',anchor='lt'):
    draw.text((round(x*S),round(y*S)),str(label),font=font(size,weight),fill=fill,anchor=anchor)

def line(xy,fill=GRID,width=1):
    draw.line([(round(x*S),round(y*S)) for x,y in xy],fill=fill,width=max(1,round(width*S)))

def rect(box,fill,radius=0,outline=None,width=1):
    b=tuple(round(v*S) for v in box)
    if radius:
        draw.rounded_rectangle(b,radius=round(radius*S),fill=fill,outline=outline,width=round(width*S))
    else:
        draw.rectangle(b,fill=fill,outline=outline,width=round(width*S))

def label_width(parts):
    return sum(draw.textlength(t,font=font(sz,wt))/S for t,sz,wt,col in parts)

def rich_right(right,top,parts):
    x=right-label_width(parts)
    for string,size,weight,color in parts:
        # A shared baseline keeps unlike font sizes aligned.
        draw.text((round(x*S),round((top+41)*S)),string,font=font(size,weight),fill=color,anchor='ls')
        x+=draw.textlength(string,font=font(size,weight))/S

# Editorial header.
rect((104,64,429,112),INK,radius=8)
text(125,76,DATA['visible_notice'],24,'#FFFFFF','bold')
text(1870,77,'PEAK SPEED  /  1 HOUR',23,MUTED,'regular','rt')
text(100,147,'Train speed & one-hour distance',100,INK,'title')
text(104,266,'Synthetic peak speed · one-hour distance at a constant peak speed',29,MUTED)
line([(104,327),(1896,327)],'#B8C6BE',1.3)

# One function controls both speed axes, all object widths, and the distance axis.
text(137,351,'km/h',23,INK,'bold','rt')
text(137,389,'mph',22,MUTED,'regular','rt')
for v in range(0,601,100):
    x=x_for_value(v)
    line([(x,TOP_AXIS),(x,BOTTOM_AXIS)], '#B8C6BE' if v==0 else GRID, 1.2 if v==0 else 0.85)
    text(x,350,str(v),28,INK,'bold','mt')
    text(x,389,f'{v/CONVERSION:.1f}',23,MUTED,'regular','mt')
    line([(x,TOP_AXIS-9),(x,TOP_AXIS+8)],INK,1.6)
    line([(x,BOTTOM_AXIS-8),(x,BOTTOM_AXIS+7)],INK,1.6)
    text(x,1693,str(v*HOURS),27,INK,'bold','mt')
for v in range(0,601,20):
    if v%100:
        x=x_for_value(v)
        line([(x,TOP_AXIS-5),(x,TOP_AXIS)],'#8C9C96',0.8)
        line([(x,BOTTOM_AXIS),(x,BOTTOM_AXIS+5)],'#8C9C96',0.8)
line([(X0,TOP_AXIS),(X1,TOP_AXIS)],INK,1.5)
line([(X0,BOTTOM_AXIS),(X1,BOTTOM_AXIS)],INK,1.5)
text(137,1696,'km',23,INK,'bold','rt')

artifacts = ROOT / 'artifacts'
artifacts.mkdir(exist_ok=True)
(artifacts / 'masks').mkdir(exist_ok=True)
geometry=[]
label_records=[]

for i,(category,value,meta) in enumerate(zip(DATA['categories'],VALUES,META)):
    assert meta['category']==category
    rowtop=ROW0+i*PITCH
    wheel_bottom=rowtop+135
    endpoint=x_for_value(value)
    total_width=round((endpoint-X0)*S)
    sprite=Image.open(ROOT / meta['path']).convert('RGBA')
    body_height=meta['body_bottom']-meta['body_top']+1
    sy=BODY_HEIGHT/body_height
    target_height=round(sprite.height*sy*S)
    split=meta['nose_start']
    nose_width=round((sprite.width-split)*sy*S)
    assert total_width>nose_width
    # Two joined image pieces: nonuniform body length, uniformly scaled nose.
    # The supplied data set is long enough to preserve each complete cab.
    rear=sprite.crop((0,0,split,sprite.height)).resize((total_width-nose_width,target_height),Image.Resampling.LANCZOS)
    nose=sprite.crop((split,0,sprite.width,sprite.height)).resize((nose_width,target_height),Image.Resampling.LANCZOS)
    train=Image.new('RGBA',(total_width,target_height),(0,0,0,0))
    train.alpha_composite(rear,(0,0))
    train.alpha_composite(nose,(total_width-nose_width,0))
    left=round(X0*S)
    top=round(wheel_bottom*S)-target_height
    alpha=np.asarray(train)[:,:,3]
    # Measure the transformed visible raster instead of repeating the width formula.
    yy,xx=np.where(alpha>=128)
    actual_left=(left+int(xx.min()))/S
    actual_right=(left+int(xx.max())+1)/S
    occupied=(alpha>=128).any(axis=0)
    nose_ys=yy[xx==xx.max()]
    tip_y=(top+float(np.median(nose_ys)))/S
    # A short hairline connects the right-aligned data label to the nose.
    line([(endpoint,rowtop+48),(endpoint,tip_y-2)],'#9FADA5',0.8)
    text(104,rowtop+7,category,30,INK,'bold')
    parts=[(str(value),43,'bold',INK),(' km/h',25,'regular',INK),
           ('  /  ',25,'regular','#9CA7A1'),(f'{value/CONVERSION:.1f}',29,'bold',MUTED),
           (' mph',24,'regular',MUTED)]
    rich_right(endpoint,rowtop-1,parts)
    before=canvas.crop((left,top,left+total_width,top+target_height))
    canvas.paste(train,(left,top),train)
    # Difference against the exact pre-composite background quantifies visible pixels.
    after=canvas.crop((left,top,left+total_width,top+target_height))
    delta=np.max(np.abs(np.asarray(after).astype(np.int16)-np.asarray(before).astype(np.int16)),axis=2)
    vy,vx=np.where(delta>24)
    visible_left=(left+int(vx.min()))/S
    visible_right=(left+int(vx.max())+1)/S
    Image.fromarray(alpha).save(artifacts / 'masks' / f'train_{i+1:02d}.png')
    geometry.append({'category':category,'value_kmh':value,'value_mph':f'{value/CONVERSION:.1f}',
                     'one_hour_km':value*HOURS,'target_tip_x_px':endpoint,
                     'visible_alpha_bbox_x_px':[actual_left,actual_right],
                     'actual_composite_contrast_bbox_x_px':[visible_left,visible_right],
                     'contrast_tip_error_px':visible_right-endpoint,
                     'contrast_tip_error_kmh':(visible_right-endpoint)/(X1-X0)*600,
                     'continuous_occupied_columns':bool(occupied.all()),
                     'empty_column_count':int((~occupied).sum()),
                     'tip_y_px':tip_y,'wheel_bottom_px':wheel_bottom,
                     'body_height_px':BODY_HEIGHT,'nose_uniform_scale':sy,
                     'source':meta['path'],'raster_width_px':total_width/S,
                     'body_length_scale_x':(total_width-nose_width)/S/split})
    label_records.append({'category':category,'kmh':str(value),'mph':f'{value/CONVERSION:.1f}',
                          'category_box':[104,rowtop+7,104+draw.textlength(category,font=font(30,'bold'))/S,rowtop+42],
                          'value_box':[endpoint-label_width(parts),rowtop,endpoint,rowtop+43]})

# A bottom ruler expresses the same lengths as distance over one hour.
text(104,1765,'DISTANCE IN ONE HOUR',29,INK,'bold')
text(104,1821,'1 km/h sustained for 1 hour = 1 km travelled.',30,INK)
text(104,1870,'Read from the common zero to each physical nose tip. Length encodes speed.',25,MUTED)
text(104,1908,'Train height, windows, colours and equipment are illustrative, not additional measures.',24,MUTED)
text(1896,1957,'SYNTHETIC VALUES · ILLUSTRATIVE TRAIN DESIGNS',19,MUTED,'regular','rt')

final=canvas.resize((W,H),Image.Resampling.LANCZOS)
final.save(ROOT / 'final.png')
(artifacts / 'geometry_checks.json').write_text(json.dumps({
    'canvas':[W,H],'common_zero_x_px':X0,'full_scale_x_px':X1,
    'scale_px_per_kmh':(X1-X0)/600,'top_speed_axis_y_px':TOP_AXIS,
    'bottom_distance_axis_y_px':BOTTOM_AXIS,'records':geometry,
    'label_records':label_records,'total_kmh':sum(VALUES),
    'measurement':'Rendered train alpha >=128 and actual pre/post-composite RGB difference >24; measured at 2x, reported in final-image pixels.'
},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Wrote final.png:',W,H,'; measured',len(geometry),'train bodies.')
