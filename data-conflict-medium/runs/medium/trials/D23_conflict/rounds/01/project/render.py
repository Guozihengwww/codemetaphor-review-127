"""Rebuild final.png locally. Run complete renders through capture_render.py.

All five envelopes share one generated silhouette and one isotropic transform.
The physical alpha area encodes CEO pay. Basket bounding-box centers encode
worker pay on a common linear scale. Text comes from the frozen data.json.
"""
from pathlib import Path
from functools import lru_cache
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / 'data.json').read_text(encoding='utf-8'))
BRIEF = json.loads((ROOT / 'brief.json').read_text(encoding='utf-8'))
W, H = BRIEF['width'], BRIEF['height']
SS = 3
INK = '#233d47'
MUTED = '#647a80'
TEAL = '#68aeb4'
COLORS = ['#df563d', '#e8bd39', '#479cc7', '#78ae48', '#e59037']
CENTERS = [520., 1040., 1350., 1580., 1810.]
GROUND_Y = 1770.
WORKER_SPAN = 760.
AXIS_X = 179.
RIGHT = 1892.
REFERENCE_WIDTH = 640.
RIGGING_GAP = 42.
BASKET_WIDTH = 34.
CEO = next(s for s in DATA['series'] if s['id'] == 'ceo_pay')
WORKER = next(s for s in DATA['series'] if s['id'] == 'worker_median_pay')
AX_MIN, AX_MAX = DATA['worker_axis']['minimum'], DATA['worker_axis']['maximum']


def rgb(color):
    return tuple(int(color[i:i+2], 16) for i in (1, 3, 5))


@lru_cache(None)
def font(size, bold=False, cjk=False):
    file = 'NotoSansCJK-Bold.ttc' if cjk else ('NotoSans-Bold.ttf' if bold else 'NotoSans-Regular.ttf')
    return ImageFont.truetype(str(ROOT / 'assets/fonts' / file), round(size * SS), index=2 if cjk else 0)


def text(x, y, content, size=28, fill=INK, bold=False, align='left', cjk=False):
    f = font(size, bold, cjk)
    width = draw.textlength(content, font=f)
    if align == 'center':
        x -= width / SS / 2
    elif align == 'right':
        x -= width / SS
    draw.text((round(x*SS), round(y*SS)), content, fill=fill, font=f, anchor='lt')
    return width/SS


def line(points, fill, width=1):
    draw.line([(round(x*SS), round(y*SS)) for x, y in points], fill=fill, width=max(1,round(width*SS)))


def ellipse(box, fill, outline=None, width=1):
    draw.ellipse(tuple(round(v*SS) for v in box), fill=fill, outline=outline, width=max(1,round(width*SS)))


def rounded(box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(tuple(round(v*SS) for v in box), round(radius*SS), fill=fill, outline=outline, width=round(width*SS))


def dollar(n):
    return '${:,}'.format(int(n))


def y_worker(value):
    return GROUND_Y - (value-AX_MIN)/(AX_MAX-AX_MIN)*WORKER_SPAN


def cloud(cx, cy, factor):
    for dx,dy,rx,ry in [(-58,7,39,26),(-21,-15,42,40),(28,-12,42,35),(68,10,32,22),(2,15,73,24)]:
        ellipse([cx+(dx-rx)*factor, cy+(dy-ry)*factor, cx+(dx+rx)*factor, cy+(dy+ry)*factor], '#ffffff')


def tree(x, base, scale):
    # Non-data scenery is kept below the salary plot except for small tree crowns.
    ellipse([x-18*scale,base-5*scale,x+23*scale,base+3*scale], '#bacb8e')
    ellipse([x-18*scale,base-55*scale,x+18*scale,base-16*scale], '#aabd64')
    ellipse([x-11*scale,base-69*scale,x+14*scale,base-26*scale], '#b9ca78')
    line([(x,base),(x,base-50*scale)], '#7a8154',1.8)
    line([(x,base-20*scale),(x-11*scale,base-35*scale)], '#7a8154',1.4)
    line([(x,base-31*scale),(x+9*scale,base-43*scale)], '#7a8154',1.4)


def recolor_envelope(source, color):
    a = np.asarray(source).astype(np.float32)
    luminance = .2126*a[:,:,0]+.7152*a[:,:,1]+.0722*a[:,:,2]
    shade = np.clip(luminance / 224., .60, 1.12)
    hue = np.asarray(rgb(color), dtype=np.float32)
    a[:,:,:3] = np.clip(hue[None,None,:]*shade[:,:,None],0,255)
    return Image.fromarray(a.astype(np.uint8), 'RGBA')


def place_isotropic(sprite, cx, bottom, scale, record_alpha=False, part=''):
    """Place with a single continuous scale; no per-axis rounding or minimum size."""
    sw, sh = sprite.size
    x0, y0 = cx-sw*scale/2, bottom-sh*scale
    left, top = math.floor(x0*SS), math.floor(y0*SS)
    right, low = math.ceil((x0+sw*scale)*SS), math.ceil(bottom*SS)
    affine = (1/(scale*SS),0,(left/SS-x0)/scale,
              0,1/(scale*SS),(top/SS-y0)/scale)
    transformed = sprite.transform((right-left,low-top),Image.Transform.AFFINE,affine,Image.Resampling.BICUBIC)
    image.alpha_composite(transformed,(left,top))
    alpha = np.asarray(transformed.getchannel('A'))
    yy,xx = np.nonzero(alpha >= 128)
    alpha_path = None
    if record_alpha:
        alpha_path = f'artifacts/{part}_alpha.png'
        transformed.getchannel('A').save(ROOT / alpha_path)
    return {
        'continuous_bbox_xyxy':[x0,y0,x0+sw*scale,bottom],
        'visible_bbox_xyxy':[(left+int(xx.min()))/SS,(top+int(yy.min()))/SS,
                             (left+int(xx.max())+1)/SS,(top+int(yy.max())+1)/SS],
        'alpha_area_px2':float(alpha.sum(dtype=np.float64)/255/SS**2),
        'threshold_area_px2':float((alpha>=128).sum()/SS**2),
        'alpha_crop_path':alpha_path,
        'alpha_crop_origin_output_px':[left/SS,top/SS]
    }


# Pale sky, changing only in lightness. No reference pixels are used.
top_color=np.array([247,250,249.],dtype=np.float32)
bottom_color=np.array([240,247,244.],dtype=np.float32)
t=np.linspace(0,1,H*SS,dtype=np.float32)[:,None,None]
gradient=np.tile(top_color[None,None,:]*(1-t)+bottom_color[None,None,:]*t,(1,W*SS,1))
image=Image.fromarray(np.uint8(gradient),'RGB').convert('RGBA')
del gradient
draw=ImageDraw.Draw(image)

# Header and a small explicit synthetic-data notice.
text(76,60,DATA['title'],66,INK,True,cjk=True)
text(80,153,'CEO compensation and median worker salary',32,MUTED)
text(80,200,'Two quantities. One balloon. All pay in USD/year.',26,MUTED)
rounded([1507,70,1903,124],27,'#e3ece7')
text(1705,86,DATA['visible_notice'],23,'#466254',True,'center')
line([(80,252),(1900,252)],'#dce7e2',1)

# Faint shared-sky elements sit behind all data marks.
cloud(922,638,.83)
cloud(1767,916,.89)
cloud(750,1460,.70)
cloud(346,1519,.44)

# A typographic key makes both channels explicit, without adding another balloon.
text(1085,313,'HOW TO READ THE BALLOONS',21,MUTED,True)
rounded([1085,363,1139,417],16,'#e8efea')
text(1112,377,'01',23,'#52735d',True,'center')
text(1160,364,'Envelope area',34,INK,True)
text(1160,410,'CEO compensation · USD/year',27,MUTED)
text(1160,451,'Area grows in direct proportion to pay.',24,MUTED)
rounded([1085,514,1139,568],16,'#e4eff0')
text(1112,528,'02',23,'#54878e',True,'center')
text(1160,514,'Basket-center height',34,INK,True)
text(1160,560,'Median worker salary · USD/year',27,MUTED)
text(1160,601,'Higher basket = higher worker salary.',24,MUTED)

# Common worker axis and unobtrusive grid. The zero is a single horizontal ground.
for value in range(int(AX_MIN),int(AX_MAX)+1,20000):
    yy=y_worker(value)
    if value != 0:
        for x in np.arange(AXIS_X,RIGHT,17):
            line([(x,yy),(min(x+5,RIGHT),yy)],'#dbe7e3',1)
    line([(AXIS_X-9,yy),(AXIS_X,yy)],'#9bb2ad',1.3)
    text(AXIS_X-20,yy-14,dollar(value),25,MUTED,False,'right')
line([(AXIS_X,y_worker(AX_MAX)),(AXIS_X,GROUND_Y)],'#b1c5be',1.3)
vertical_label=Image.new('RGBA',(660*SS,44*SS),(0,0,0,0))
vd=ImageDraw.Draw(vertical_label)
vd.text((0,0),'MEDIAN WORKER SALARY · USD/year',font=font(22,True),fill=MUTED,anchor='lt')
vertical_label=vertical_label.rotate(90,expand=True)
image.alpha_composite(vertical_label,(24*SS,1100*SS))

# Green ground remains independent of both numerical channels.
draw.polygon([(round(x*SS),round(y*SS)) for x,y in
              [(AXIS_X,GROUND_Y),(RIGHT,GROUND_Y),(1920,1797),(1880,1816),(450,1816),(145,1808)]],fill='#dbe6c8')
line([(AXIS_X,GROUND_Y),(RIGHT,GROUND_Y)],'#99b277',2)
tree(275,GROUND_Y,0.75)
tree(865,GROUND_Y,0.64)
tree(1684,GROUND_Y,0.62)

envelope=Image.open(ROOT/'assets/envelope.png').convert('RGBA')
basket=Image.open(ROOT/'assets/basket.png').convert('RGBA')
source_mask=np.asarray(envelope.getchannel('A'))>=128
source_area=int(source_mask.sum())
reference_scale=REFERENCE_WIDTH/envelope.width
reference_area=source_area*reference_scale**2
reference_value=DATA['ceo_area_reference']
measurements=[]

for i,(category,ceo_value,worker_value,cx,color) in enumerate(zip(DATA['categories'],CEO['values'],WORKER['values'],CENTERS,COLORS)):
    cy=y_worker(worker_value)
    scale=reference_scale*math.sqrt(ceo_value/reference_value)
    ew,eh=envelope.width*scale,envelope.height*scale
    bottom=cy-RIGGING_GAP
    top=bottom-eh
    line([(cx,cy),(cx,GROUND_Y)],TEAL,1.5)
    ellipse([cx-27,GROUND_Y+4,cx+27,GROUND_Y+12],'#b9ce91')

    # Ropes attach to the lower sides and converge on this category's basket.
    anchor_row=round(envelope.height*.66)
    visible_x=np.nonzero(source_mask[anchor_row])[0]
    xl=(int(visible_x.min())-envelope.width/2)*scale
    xr=(int(visible_x.max())-envelope.width/2)*scale
    for ax,endx in [(cx+xl,cx-12),(cx+xr,cx+12)]:
        line([(ax,top+anchor_row*scale),(endx,cy-9)],'#8e8060',1.6)
    for dx in [-.16,.16]:
        line([(cx+ew*dx,top+eh*.74),(cx+(-7 if dx<0 else 7),cy-10)],'#ac9978',1.15)

    envelope_record=place_isotropic(recolor_envelope(envelope,color),cx,bottom,scale,True,f'envelope_{i+1:02d}')
    bscale=BASKET_WIDTH/basket.width
    bh=basket.height*bscale
    basket_record=place_isotropic(basket,cx,cy+bh/2,bscale,True,f'basket_{i+1:02d}')
    line([(cx-28,cy),(cx-20,cy)],'#5d949b',1.8)
    line([(cx+20,cy),(cx+28,cy)],'#5d949b',1.8)

    # Exact CEO labels, internally for large envelopes and externally for small ones.
    if i == 0:
        text(cx,top+eh*.34,category,39,'#fffdf4',True,'center')
        text(cx,top+eh*.34+57,'CEO COMPENSATION',18,'#ffeadb',True,'center')
        text(cx,top+eh*.34+91,dollar(ceo_value),53,'#fffdf4',True,'center')
    elif i == 1:
        text(cx,top+eh*.32,category,29,'#493e23',True,'center')
        text(cx,top+eh*.32+45,'CEO COMPENSATION',15,'#5c4b25',True,'center')
        text(cx,top+eh*.32+75,dollar(ceo_value),35,'#493e23',True,'center')
    else:
        label_top=top-118 if i < 4 else top-129
        text(cx,label_top,category,29,INK,True,'center')
        text(cx,label_top+41,'CEO compensation',19,MUTED,False,'center')
        text(cx,label_top+72,dollar(ceo_value),32,INK,True,'center')
        line([(cx,label_top+110),(cx,top-7)],'#afc0b7',1.1)

    # Worker labels always share the same relation to their basket centers.
    rounded([cx-105,cy+24,cx+105,cy+101],13,'#f3f8f5')
    text(cx,cy+29,'WORKER',19,MUTED,True,'center')
    text(cx,cy+57,dollar(worker_value),33,INK,True,'center')

    measurements.append({
        'category':category,'ceo_pay':ceo_value,'worker_pay':worker_value,
        'color':color,'center_x':cx,'isotropic_scale_from_source':scale,
        'sqrt_ratio_to_ceo_reference':math.sqrt(ceo_value/reference_value),
        'expected_envelope_area_px2':reference_area*ceo_value/reference_value,
        'envelope':envelope_record,
        'basket':basket_record,'basket_center_y_target':cy,
        'basket_center_y_visible':(basket_record['visible_bbox_xyxy'][1]+basket_record['visible_bbox_xyxy'][3])/2,
        'basket_width_nominal_px':BASKET_WIDTH,'labels':[category,dollar(ceo_value),dollar(worker_value)]
    })

text(1890,1830,'SHARED ZERO GROUND',19,'#718564',True,'right')
line([(80,1872),(1900,1872)],'#d5e1d8',1)
text(80,1900,'AREA REFERENCE',19,MUTED,True)
text(80,1935,dollar(reference_value),31,INK,True)
text(491,1900,'Only the fabric envelope encodes CEO pay; ropes and baskets are excluded.',23,MUTED)
text(491,1938,'Source: supplied synthetic data. Fixed-size baskets; linear height scale from $0 to $80,000.',23,MUTED)

output=image.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
output.save(ROOT/'final.png')
geometry={
    'canvas':[W,H], 'supersampling':SS,
    'mapping':{'worker_zero_y_px':GROUND_Y,'worker_max_y_px':y_worker(AX_MAX),
               'worker_minimum':AX_MIN,'worker_maximum':AX_MAX,'worker_axis_scale':'linear',
               'ceo_reference_value':reference_value,'ceo_reference_width_px':REFERENCE_WIDTH,
               'source_envelope_area_px2':source_area,'ceo_reference_area_px2':reference_area,
               'ceo_area_rule':'same source mask; one isotropic scale = (640/source_width)*sqrt(value/reference)'},
    'objects':measurements,
    'visible_notice':DATA['visible_notice'],
    'all_mandatory_labels':[v for r in measurements for v in r['labels']]
}
(ROOT/'artifacts/geometry.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Rendered final.png at',W,H,'using',len(measurements),'data-bound generated envelopes.')
