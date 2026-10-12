"""Rebuild final.png from frozen local data and generated railcar parts.

No network or model calls. All train nose positions, axes, units, and labels
use one linear mapping. Invoke through capture_render.py in the experiment.
"""
from pathlib import Path
from functools import lru_cache
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT/'data.json').read_text(encoding='utf-8'))
BRIEF = json.loads((ROOT/'brief.json').read_text(encoding='utf-8'))
PARTS = json.loads((ROOT/'assets/train_parts.json').read_text(encoding='utf-8'))
W, H = BRIEF['width'], BRIEF['height']
S = 2
X0, X1 = 240.0, 1880.0
V0, V1 = DATA['value_axis']['minimum'], DATA['value_axis']['maximum']
SCALE = (X1-X0)/(V1-V0)
KM_PER_MILE = DATA['conversion']['km_per_mile']
HOURS = DATA['conversion']['distance_comparison_hours']
BG = '#f6f4ee'
INK = '#182e3d'
MUTED = '#64716f'
GRID = '#d8ddd6'
ACCENT = '#d5573f'
COLORS = ['#1f58aa','#235a79','#ce564e','#853448','#168589','#203e65','#bd4540','#d33e29']
RESAMPLE = Image.Resampling.LANCZOS


def px(v):
    return int(round(v*S))


def x_of(v):
    return X0+(v-V0)*SCALE


def mph(v):
    return f'{v/KM_PER_MILE:.1f}'


@lru_cache(None)
def font(size, bold=False, cjk=False):
    fn = ('CJK-Bold.ttc' if bold else 'CJK.ttc') if cjk else ('Sans-Bold.ttf' if bold else 'Sans.ttf')
    return ImageFont.truetype(str(ROOT/'assets/fonts'/fn),px(size),index=0)


canvas = Image.new('RGB',(px(W),px(H)), BG)
draw = ImageDraw.Draw(canvas)
text_boxes = []
train_mask = Image.new('L',(px(W),px(H)),0)


def text(s, x, y, size=28, fill=INK, bold=False, cjk=False, align='left', track=False):
    f=font(size,bold,cjk)
    # y is the actual top ink edge, independent of Latin/CJK font ascent.
    bb=draw.textbbox((0,0),s,font=f)
    tw=bb[2]-bb[0]
    left=px(x)-(tw if align=='right' else tw/2 if align=='center' else 0)
    top=px(y)
    draw.text((left-bb[0],top-bb[1]),s,font=f,fill=fill)
    bbox=[left/S,top/S,(left+tw)/S,(top+bb[3]-bb[1])/S]
    text_boxes.append({'text':s,'bbox':bbox})
    return bbox


def line(points, fill=GRID, width=1):
    draw.line([(px(x),px(y)) for x,y in points],fill=fill,width=max(1,px(width)))


def rect(box, fill, outline=None, width=1, radius=0):
    box=tuple(px(v) for v in box)
    if radius:
        draw.rounded_rectangle(box,radius=px(radius),fill=fill,outline=outline,width=px(width))
    else:
        draw.rectangle(box,fill=fill,outline=outline,width=px(width))


def make_train(i,length):
    """Construct one continuous railcar consist, ending in a generated cab.

    Long trains repeat real generated cabin sections, with a terminal cab.
    Small trains omit the central passenger stretch, retaining rear/door/
    pantograph and front cab/wheels. No minimum length is imposed.
    """
    meta=PARTS[i]
    source=Image.open(ROOT/meta['path']).convert('RGBA')
    sw,sh=source.size
    body_height=50.0
    sy=body_height/(meta['body_bottom']-meta['body_top'])
    th=round(sh*sy*S)
    target=round(length*S)
    result=Image.new('RGBA',(target,th),(0,0,0,0))
    natural=sw*sy*S
    pieces=[]
    if target < natural*.56:
        # Keep rear door, a few windows, pantograph, two wheel groups, and nose.
        split=round(target*.47)
        pieces=[(0,round(sw*.265),0,split),
                (round(sw*.755),sw,split,target)]
    elif target <= natural*1.12:
        pieces=[(0,sw,0,target)]
    else:
        cab_start=round(sw*.72)
        cab_w=round((sw-cab_start)*sy*S)
        cabin_w=target-cab_w
        count=max(1,round(cabin_w/(cab_start*sy*S)))
        starts=[round(cabin_w*j/count) for j in range(count+1)]
        for a,b in zip(starts,starts[1:]):
            pieces.append((0,cab_start,a,b))
        pieces.append((cab_start,sw,cabin_w,target))
    for src0,src1,dst0,dst1 in pieces:
        tile=source.crop((src0,0,src1,sh)).resize((dst1-dst0,th),RESAMPLE)
        result.alpha_composite(tile,(dst0,0))
    if target > natural*1.12:
        # Reuse the generated rear bogie at the far end of each trailer.
        # Each extended coach then has two wheel groups, like the terminal car.
        by=max(0,meta['body_bottom']-23)
        bogie=source.crop((round(sw*.065),by,round(sw*.215),sh))
        bh=round((sh-by)*sy*S)
        bw=round(bogie.width*sy*S)
        bogie=bogie.resize((bw,bh),RESAMPLE)
        for src0,src1,dst0,dst1 in pieces[:-2]:
            if dst1-dst0 > bw*2.8:
                result.alpha_composite(bogie,(dst1-bw-px(16),th-bh))
    # No shadow, smoke or track: the visible right boundary is the nose.
    alpha=np.array(result.getchannel('A'))
    yy,xx=np.where(alpha>=100)
    left,right=int(xx.min()),int(xx.max())+1
    # Trim only antialiasing padding and resample to the one true data length.
    if left!=0 or right!=target:
        result=result.crop((left,0,right,th)).resize((target,th),RESAMPLE)
    return result,sy,meta


# Editorial header.
rect((0,0,W,14),INK)
text('Synthetic',80,68,31,bold=True,fill=ACCENT)
rect((1515,62,1880,116),INK,radius=8)
text(DATA['visible_notice'],1697.5,80,25,bold=True,fill='#ffffff',align='center')
text('列车长度与换算',75,133,106,bold=True,cjk=True)
text('Synthetic peak speed',82,264,32,fill=INK)
text('八辆列车，共用一个零点。',532,263,31,fill=MUTED,cjk=True)
text('0–600',1880,164,60,bold=True,align='right')
text('km/h 线性尺度',1880,240,26,cjk=True,fill=MUTED,align='right')
line([(80,326),(1880,326)],INK,1.3)
text('车身从零点连续延伸，实体鼻尖对应速度。',80,347,25,cjk=True,fill=MUTED)

# Shared grid and dual-unit speed scale.
AXIS_Y=455
GRID_BOTTOM=1580
for tick in range(int(V0),int(V1)+1,100):
    x=x_of(tick)
    line([(x,AXIS_Y),(x,GRID_BOTTOM)],INK if tick==0 else GRID,1.4 if tick==0 else .8)
    text(mph(tick),x,391,23,fill=MUTED,align='center')
    text(str(tick),x,425,26,bold=True,align='center')
for tick in range(int(V0),int(V1)+1,25):
    x=x_of(tick)
    line([(x,AXIS_Y),(x,AXIS_Y+(12 if tick%100==0 else 6))],INK,.9)
line([(X0,AXIS_Y),(X1,AXIS_Y)],INK,1.3)
text('mph',80,391,24,fill=MUTED)
text('km/h',80,425,26,bold=True)

geometry=[]
for i,(name,value) in enumerate(zip(DATA['categories'],DATA['series'][0]['values'])):
    baseline=600+i*134
    endpoint=x_of(value)
    length=endpoint-X0
    train,sy,meta=make_train(i,length)
    top=baseline-(meta['body_bottom']*sy)
    # Train sprites are alpha-composited directly; their geometry is the bar.
    position=(px(X0),px(top))
    canvas.paste(train,position,train)
    train_mask.paste(train.getchannel('A'),position)
    draw=ImageDraw.Draw(canvas)
    rect((80,baseline-47,86,baseline-16),COLORS[i],radius=2)
    text(name,100,baseline-42,27,bold=True)
    if i==0:
        # Full-scale endpoint: keep the value block inside the right margin.
        text(f'{value} km/h',endpoint,473,39,bold=True,align='right')
        text(f'{mph(value)} mph',endpoint,519,24,fill=MUTED,align='right')
    else:
        text(f'{value} km/h',endpoint+24,baseline-52,39,bold=True)
        text(f'{mph(value)} mph',endpoint+26,baseline-5,24,fill=MUTED)
    geometry.append({'category':name,'km_h':value,'mph':mph(value),
        'one_hour_km':value*HOURS,'one_hour_miles':f'{value*HOURS/KM_PER_MILE:.1f}',
        'target_tip_x':endpoint,'body_height_px':50,
        'sprite_paste_xy':[position[0]/S,position[1]/S],
        'sprite_final_size':[train.width/S,train.height/S],
        'body_mid_y':top+(meta['body_top']+meta['body_bottom'])/2*sy})

# One-hour distance ruler, using the exact same 0 and horizontal scale.
text('行驶 1 小时',80,1600,29,bold=True,cjk=True)
text('相同水平尺度',1880,1603,23,cjk=True,fill=MUTED,align='right')
BAND_Y=1691
for j,tick in enumerate(range(int(V0),int(V1),100)):
    rect((x_of(tick),BAND_Y,x_of(tick+100),BAND_Y+15),INK if j%2==0 else '#94a3a4')
for tick in range(int(V0),int(V1)+1,100):
    x=x_of(tick)
    line([(x,BAND_Y-13),(x,BAND_Y+15)],INK,1.2)
    text(str(tick*HOURS),x,BAND_Y-49,25,bold=True,align='center')
    text(f'{tick*HOURS/KM_PER_MILE:.1f}',x,BAND_Y+34,23,fill=MUTED,align='center')
text('km',80,BAND_Y-49,25,bold=True)
text('mi',80,BAND_Y+34,24,fill=MUTED)
text('以各自峰值速度匀速行驶 1 小时，公里距离与 km/h 速度数值相同。',80,1808,27,cjk=True)
line([(80,1870),(1880,1870)],GRID,1)
text('mph = km/h ÷ 1.609344',80,1902,23,fill=MUTED)
text('合成数据 · 不代表真实列车性能',1880,1901,23,cjk=True,fill=MUTED,align='right')

# Downsample exactly once, and do not modify this image after capture.
final=canvas.resize((W,H),RESAMPLE)
final.save(ROOT/'final.png')
art=ROOT/'artifacts';art.mkdir(exist_ok=True)
# Measure the delivered pixel geometry using the same global placement,
# antialiasing phase, and downsampling as final.png (not nominal target widths).
delivered_alpha=np.array(train_mask.resize((W,H),RESAMPLE))
for g in geometry:
    top=math.floor(g['sprite_paste_xy'][1])-3
    bottom=math.ceil(g['sprite_paste_xy'][1]+g['sprite_final_size'][1])+3
    a=delivered_alpha[top:bottom,:]
    yy,xx=np.where(a>=100)
    left,right=int(xx.min()),int(xx.max())+1
    rows=[]
    for y in range(round(g['body_mid_y'])-3,round(g['body_mid_y'])+4):
        on=np.where(delivered_alpha[y]>=100)[0]
        if len(on):
            rows.append({'y':int(y),'from':int(on.min()),'to':int(on.max()),
                         'interior_holes':int((delivered_alpha[y,on.min():on.max()+1]<100).sum())})
    g.update({'visible_left_x':left,'visible_tip_x':right,
              'visible_length_px':right-left,
              'visible_speed_from_length':(right-left)/SCALE,
              'error_percent_full_scale':abs((right-left)-(g['target_tip_x']-X0))/(X1-X0)*100,
              'body_continuity_rows':rows,
              'occupied_columns_missing':int((a[:,left:right].max(axis=0)<100).sum())})
(art/'geometry_checks.json').write_text(json.dumps({
    'canvas':[W,H],'mapping':{'zero_x':X0,'maximum_x':X1,'minimum':V0,'maximum':V1,'pixels_per_km_h':SCALE},
    'alpha_threshold':100,'measurements':geometry,
    'axis_ticks':[{'km_h':v,'mph':mph(v),'x':x_of(v),'one_hour_km':v*HOURS} for v in range(int(V0),int(V1)+1,100)],
    'text_boxes':text_boxes,
    'all_text_in_canvas':all(b['bbox'][0]>=0 and b['bbox'][1]>=0 and b['bbox'][2]<=W and b['bbox'][3]<=H for b in text_boxes)
},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Saved final.png:',W,'x',H)
print('Max visible endpoint error (% full scale):',max(g['error_percent_full_scale'] for g in geometry))
