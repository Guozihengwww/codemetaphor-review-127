#!/usr/bin/env python3
"""Render the frozen synthetic book-sales chart, using only local assets/data.

Regenerate with: PYTHON render.py
During the experiment, invoke through capture_render.py to preserve every attempt.
"""
from pathlib import Path
import json
import math
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import numpy as np

ROOT = Path(__file__).resolve().parent
S = 2
BRIEF = json.loads((ROOT / 'brief.json').read_text(encoding='utf-8'))
DATA = json.loads((ROOT / 'data.json').read_text(encoding='utf-8'))
W, H = BRIEF['width'], BRIEF['height']
OUT = ROOT / 'artifacts'
OUT.mkdir(exist_ok=True)

# One linear transform, used by spines, ticks and leaders alike.
ZERO_Y = 1100.0
SCALE_TOP_Y = 172.0
BOOK_BOTTOM = 1767.0
BOOK_LEFT = 220.0
BOOK_WIDTH = 153.0
BOOK_STEP = 171.0
AXIS_X = 177.0
axis = DATA['value_axis']
PX_PER_UNIT = (ZERO_Y - SCALE_TOP_Y) / (axis['maximum'] - axis['minimum'])
def value_y(value):
    return ZERO_Y - (value - axis['minimum']) * PX_PER_UNIT

def px(x): return int(round(x * S))
PAPER = '#f7eed5'
INK = '#4b321e'
MUTED = '#684b2b'
GOLD = '#eab52d'

# Quiet paper grain; all book faces come from the generated raw material.
rng = np.random.default_rng(23)
grain = rng.normal(0, 0.8, (H*S, W*S, 1)).astype(np.float32)
base = np.array([234,181,45], dtype=np.float32)[None,None,:]
bg = np.clip(base + grain, 0, 255).astype('uint8')
im = Image.fromarray(bg, 'RGB').convert('RGBA')
d = ImageDraw.Draw(im)

font_names = {
    'display': 'NimbusSansNarrow-Bold.otf',
    'sans': 'NimbusSans-Regular.otf',
    'bold': 'NimbusSans-Bold.otf',
    'serif': 'NimbusRoman-Regular.otf',
    'serif_bold': 'NimbusRoman-Bold.otf',
    'book': 'P052-Roman.otf',
}
def font(size, kind='sans'):
    return ImageFont.truetype(str(ROOT/'assets/fonts'/font_names[kind]), px(size))

text_records = []
def text(s, x, y, size, kind='sans', fill=INK, align='left', role=None):
    f = font(size, kind)
    box = f.getbbox(s)
    width = f.getlength(s)
    xx = px(x) - (width if align == 'right' else width/2 if align == 'center' else 0)
    yy = px(y) - box[1]
    d.text((xx, yy), s, font=f, fill=fill)
    actual = [xx/S, y, (xx+width)/S, y+(box[3]-box[1])/S]
    if role: text_records.append({'role':role, 'text':s, 'bounds':actual})
    return actual

def line(coords, fill, width=1):
    d.line(tuple(px(v) for v in coords), fill=fill, width=max(1,px(width)))

def rect(bounds, fill=None, outline=None, width=1):
    d.rectangle(tuple(px(v) for v in bounds), fill=fill, outline=outline, width=max(1,px(width)))

# The upper-right editorial masthead leaves the scale and tallest book unobscured.
text('TEN FICTIONAL TITLES  /  50 YEARS',1910,151,27,'bold',INK,'right','scope')
line([901,202,1910,202],INK,1.4)
text('SYNTHETIC',1912,243,156,'bold',PAPER,'right','title')
text('BOOK SALES',1912,405,175,'bold',PAPER,'right','title')

text('Illustrative cumulative copies sold',1910,635,34,'sans',INK,'right','metric')
text('during a fictional 50-year period.',1910,681,34,'sans',INK,'right','metric')
text(DATA['units'],1910,748,31,'bold',INK,'right','units')

# Clearly separate the study's synthetic status from the illustrated book labels.
status = DATA['data_status']
f = font(25,'bold')
status_width = f.getlength(status)/S + 38
rect([1910-status_width,809,1910,854],fill=INK)
text(status,1910-status_width/2,820,25,'bold',PAPER,'center','synthetic_status')
text('Only the extension above 0 encodes sales.',1910,892,25,'sans',INK,'right','encoding_note')

# Left-hand linear scale: the input ticks are authoritative.
text('COPIES SOLD',65,76,23,'bold',INK,'left','axis_heading')
text('(million copies)',65,110,22,'sans',INK,'left','axis_units')
for tick in axis['ticks']:
    yy = value_y(tick)
    text(f'{tick:,}',150,yy-12.5,25,'sans',INK,'right','tick')
    line([AXIS_X-14,yy,AXIS_X+16,yy],INK,1.3)

categories = {c['id']:c for c in DATA['categories']}
values = DATA['series'][0]['values']
assert len(DATA['category_order']) == len(values) == 10
geometry = []

# Paint one uninterrupted generated cloth face from its data-defined top to the shelf.
# Raster bounds are determined here, never by image generation.
for i, (cid, value) in enumerate(zip(DATA['category_order'], values)):
    c = categories[cid]
    x = BOOK_LEFT + i*BOOK_STEP
    yy = value_y(value)
    face = Image.open(ROOT/f'assets/spine_{i+1:02d}.png').convert('RGB')
    face = face.resize((px(BOOK_WIDTH),px(BOOK_BOTTOM)-px(yy)),Image.Resampling.LANCZOS)
    # Retain the generated binding seams and linen texture; a small saturation lift
    # restores the vivid editorial colors after printing-like downsampling.
    face = ImageEnhance.Color(face).enhance(1.08)
    im.paste(face,(px(x),px(yy)))
    # Inner rules stay inside the precise data-bearing rectangle.
    line([x+0.8,yy+0.5,x+BOOK_WIDTH-0.8,yy+0.5], '#503521',0.65)
    line([x+BOOK_WIDTH-1.1,yy+1,x+BOOK_WIDTH-1.1,BOOK_BOTTOM], '#332d23',0.6)
    geometry.append({'id':cid,'category_label':c['label'], 'value':value,
                     'spine_color':c['spine_color'], 'title_layout':c['title_layout'],
                     'x_left':x,'x_right':x+BOOK_WIDTH,'top_y':yy,'zero_y':ZERO_Y,
                     'data_height_px':ZERO_Y-yy,'bottom_y':BOOK_BOTTOM,
                     'asset':f'assets/spine_{i+1:02d}.png'})

# A single thin shared zero line. Decorative binding rules begin farther below it.
line([AXIS_X-14,ZERO_Y,1915,ZERO_Y],PAPER,2)
text('0',150,ZERO_Y-12.5,25,'bold',INK,'right')

# Every exact number is independently typeset. Short leaders terminate at the true tops.
for g in geometry:
    mid = (g['x_left']+g['x_right'])/2
    yy = g['top_y']
    text(f"{g['value']:,}",mid,yy-60,44,'bold',INK,'center','value_'+g['id'])
    if g['value'] <= 108:
        line([mid,yy-20,mid,yy-3],MUTED,1.1)

# Reusable book typography. Title fields and rotation are taken directly from data.
def fit_size(s, limit, initial, kind):
    size = initial
    while font(size,kind).getlength(s)/S > limit:
        size -= 0.5
    return size

def stack_title(words, center_x, top, width, initial, kind, fill, role, line_gap=9):
    size = min(fit_size(w,width,initial,kind) for w in words)
    y = top
    for w in words:
        box = text(w,center_x,y,size,kind,fill,'center',role)
        y = box[3] + line_gap
    return y

def rotated_title(s, center_x, center_y, max_length, initial, kind, fill, role):
    size = fit_size(s,max_length,initial,kind)
    f = font(size,kind)
    b = f.getbbox(s)
    layer = Image.new('RGBA',(int(math.ceil(f.getlength(s)))+px(8),b[3]-b[1]+px(8)),(0,0,0,0))
    ld = ImageDraw.Draw(layer)
    ld.text((px(4),px(4)-b[1]),s,font=f,fill=fill)
    layer = layer.transpose(Image.Transpose.ROTATE_270)
    dest = (px(center_x)-layer.width//2,px(center_y)-layer.height//2)
    im.alpha_composite(layer,dest)
    text_records.append({'role':role,'text':s,'rotation_clockwise_degrees':90,
                         'bounds':[dest[0]/S,dest[1]/S,(dest[0]+layer.width)/S,(dest[1]+layer.height)/S]})

def diamond(cx,cy,r,fill):
    d.polygon([(px(cx),px(cy-r)),(px(cx+r),px(cy)),(px(cx),px(cy+r)),(px(cx-r),px(cy))],fill=fill)

for i,g in enumerate(geometry):
    c = categories[g['id']]
    x = g['x_left']; right=g['x_right']; mid=(x+right)/2
    cream = '#f6e9ca'
    col = '#382a1c' if c['spine_color']=='orange' else cream
    # Fine binding marks are strictly in the common non-data zone.
    line([x+12,1141,right-12,1141],col,1.0)
    line([x+12,1147,right-12,1147],col,0.65)
    line([x+12,1647,right-12,1647],col,1.1)
    line([x+12,1653,right-12,1653],col,0.65)
    title_role = 'spine_title_'+c['id']
    if c['title_layout']=='rotated_90_degrees':
        kind = 'serif' if i in (1,4,5,9) else 'book'
        rotated_title(c['spine_title'],mid,1396,458,53,kind,col,title_role)
    elif c['title_layout']=='stacked_in_frames':
        rect([x+10,1165,right-10,1380],outline=col,width=1.3)
        rect([x+15,1170,right-15,1375],outline=col,width=0.65)
        stack_title(c['spine_title'].split(),mid,1210,BOOK_WIDTH-37,43,'serif_bold',col,title_role,11)
        rect([x+10,1393,right-10,1627],outline=col,width=1.3)
        rect([x+15,1398,right-15,1622],outline=col,width=0.65)
        diamond(mid,1509,7,col)
    elif i==0:
        stack_title(c['spine_title'].split(),mid,1195,BOOK_WIDTH-30,48,'serif_bold',col,title_role,16)
        line([x+15,1339,right-15,1339],col,1.0)
        line([mid,1432,mid,1530],col,1)
        line([mid-30,1481,mid+30,1481],col,1)
        diamond(mid,1481,8,col)
        d.ellipse([px(mid-20),px(1461),px(mid+20),px(1501)],outline=col,width=px(1))
    else:
        stack_title(c['spine_title'].split(),mid,1195,BOOK_WIDTH-28,54,'display',col,title_role,20)
        line([mid-20,1505,mid+20,1505],col,0.9)
        diamond(mid,1505,5,col)
    bysize=fit_size(c['spine_byline'],BOOK_WIDTH-18,25,'serif_bold')
    text(c['spine_byline'],mid,1698,bysize,'serif_bold',col,'center','spine_byline_'+c['id'])

# A full-width warm wood shelf supports every spine at exactly the same level.
rect([0,1767,W,1788],fill='#aa6130')
line([0,1768,W,1768],'#d4954d',2)
rect([0,1788,W,H],fill='#864527')
line([0,1789,W,1789],'#62381f',2)
for i,g in enumerate(geometry):
    text(g['category_label'],(g['x_left']+g['x_right'])/2,1816,22,'sans',PAPER,'center','category_'+g['id'])
text('FICTIONAL',45,1817,17,'bold',PAPER,'left','footer_status')

im=im.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
im.save(ROOT/'final.png')
(OUT/'layout.json').write_text(json.dumps({
    'canvas':[W,H],
    'linear_mapping':{'minimum':axis['minimum'],'maximum':axis['maximum'],'zero_y':ZERO_Y,
                      'maximum_y':SCALE_TOP_Y,'pixels_per_unit':PX_PER_UNIT},
    'books':geometry,'text':text_records,
},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(f'Rendered {W}×{H}; {len(geometry)} data-defined continuous book spines.')
