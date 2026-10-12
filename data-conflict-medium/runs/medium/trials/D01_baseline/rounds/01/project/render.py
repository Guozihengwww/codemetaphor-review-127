"""Rebuild final.png from frozen local data and generated local materials.

Usage: PYTHON render.py
For experimental captures, invoke only through capture_render.py.
No network or model calls. All dimensions below are final-image pixels.
"""
from pathlib import Path
from decimal import Decimal
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
S = 2
BRIEF = json.loads((ROOT / 'brief.json').read_text(encoding='utf-8'))
DATA = json.loads((ROOT / 'data.json').read_text(encoding='utf-8'))
W, H = BRIEF['width'], BRIEF['height']
series = DATA['series'][0]
values = [Decimal(str(v)) for v in series['values']]
names = DATA['categories']
total = sum(values)
assert len(values) == len(names) == 18
assert total == Decimal('192.0')
BG = '#F8EFDF'
INK = '#303C3D'
MUTED = '#68716F'
RED = '#B34433'
TEAL = '#427F86'
PALETTE = ['#4287B3','#52936F','#C55D53','#6A7FB7','#3C9B92',
           '#C19334','#59A16B','#C87F50','#708B94','#BD6574',
           '#44A4B3','#9B983C','#7773B4','#32947E','#BF8F9F',
           '#9B755B','#4B819D','#A064A8']
FONTS = {
    'regular': '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    'medium': '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    'bold': '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    'condensed': '/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf',
    'cjk': '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
}
# One common mapping defines ALL bands, labels, leaders and diagnostics.
BX0, BX1, BT, BB = 442.0, 998.0, 600.0, 1770.0
BW, BH = BX1 - BX0, BB - BT
RADIUS_X, RADIUS_Y = 112.0, 160.0
bounds = [BT]
for value in values:
    bounds.append(BT + BH * float(sum(values[:len(bounds)])) / float(total))
mids = [(bounds[i] + bounds[i+1]) / 2 for i in range(len(values))]

im = Image.new('RGB', (W*S, H*S), BG)
d = ImageDraw.Draw(im)
text_boxes = []

def font(size, weight='regular'):
    return ImageFont.truetype(FONTS[weight], round(size*S))

def text(x, y, value, size=26, fill=INK, weight='regular', anchor='lt', record=True):
    value = str(value)
    f = font(size, weight)
    d.text((round(x*S), round(y*S)), value, font=f, fill=fill, anchor=anchor, stroke_width=0)
    if record:
        box = d.textbbox((round(x*S), round(y*S)), value, font=f, anchor=anchor)
        text_boxes.append({'text': value, 'bbox': [round(v/S,2) for v in box]})

def line(points, fill, width=1):
    d.line([(round(x*S),round(y*S)) for x,y in points], fill=fill, width=round(width*S), joint='curve')

def ellipse(box, fill, outline=None, width=1):
    d.ellipse(tuple(round(v*S) for v in box), fill=fill, outline=outline, width=round(width*S))

def rect(box, fill, outline=None, width=1, radius=None):
    q = tuple(round(v*S) for v in box)
    if radius:
        d.rounded_rectangle(q, radius=round(radius*S), fill=fill, outline=outline, width=round(width*S))
    else:
        d.rectangle(q, fill=fill, outline=outline, width=round(width*S))

def rgb(h):
    return np.array([int(h[i:i+2],16) for i in (1,3,5)],dtype=np.float32)

def darker(h, factor=.8):
    return tuple(np.rint(rgb(h)*factor).astype(int))

def badge(x, y, idx, radius=25, inside=False):
    col = PALETTE[idx]
    fill = tuple(np.rint(rgb(col)*.83).astype(int)) if inside else col
    ellipse((x-radius,y-radius,x+radius,y+radius), fill, '#F8EFDF' if inside else None, 1.2)
    letter = chr(65+idx) if idx < 17 else '+'
    text(x, y-1, letter, 24 if radius>=23 else 22, '#FFFFFF', 'bold', 'mm')

# Header: exact data title, synthetic status, series and unit.
text(72, 60, 'ILLUSTRATIVE GAS RESERVES', 23, TEAL, 'bold')
rect((1089,48,1368,91), '#EEE0CC', radius=8)
text(1228.5,69, DATA['visible_notice'], 20, RED, 'bold', 'mm')
text(68, 111, DATA['title'], 99, RED, 'cjk')
text(74, 250, 'Illustrative gas reserves  /  18 categories  /  Unit: tcm', 27, MUTED)
line([(72,312),(1368,312)], '#CABDA9', 1.4)

# Object reading instructions and total sit in the clear space above the shoulder.
text(74, 373, 'READING THE BOTTLE', 21, TEAL, 'bold')
for j, s in enumerate(['Band height shows its', 'share of the total volume.', 'The shoulder is included;', 'the guard and valve are not.']):
    text(74, 415+j*35, s, 23, INK)
text(1061, 373, 'TOTAL VOLUME', 21, TEAL, 'bold')
text(1053, 412, f'{total:.1f}', 71, INK, 'condensed')
text(1060, 497, 'tcm', 31, INK, 'bold')
text(1060, 543, '100% of the stack', 22, MUTED)

# Generated red guard and valve. Body is layered in front at the shoulder.
guard = Image.open(ROOT / 'assets/red_guard_valve.png').convert('RGBA')
gscale = .925
guard = guard.resize((round(guard.width*gscale*S), round(guard.height*gscale*S)),Image.Resampling.LANCZOS)
gleft = 720 + (300-512)*gscale
gtop = BT + (20-292)*gscale
im.paste(guard, (round(gleft*S),round(gtop*S)), guard)

# Exact data-bearing body, from shoulder crown BT to flat base BB.
# Quarter-ellipse shoulders; no top/bottom pads, no minimum band height.
bw, bh = round(BW*S), round(BH*S)
y = (np.arange(bh,dtype=np.float32)+.5)/S
x = (np.arange(bw,dtype=np.float32)+.5)/S
ry = np.clip(y/RADIUS_Y, 0, 1)
left = np.where(y<RADIUS_Y, RADIUS_X*(1-np.sqrt(np.maximum(0,1-(1-ry)**2))), 0)
right = BW-left
mask = (x[None,:] >= left[:,None]) & (x[None,:] < right[:,None])
# The generated material is reprojected across each row of the curved silhouette.
# All body hue comes from code-defined band colors, while generated surface
# light, shadow and fine texture remain visible throughout the same body.
material = np.asarray(Image.open(ROOT / 'assets/body_surface.png').convert('L'),dtype=np.float32)
from scipy.ndimage import map_coordinates
src_y = y/BH*(material.shape[0]-1)
u = np.clip((x[None,:]-left[:,None])/(right-left)[:,None],0,1)
coords = np.array([np.broadcast_to(src_y[:,None],u.shape), u*(material.shape[1]-1)])
surface = map_coordinates(material, coords, order=1, mode='nearest')
light = np.clip(surface/213.0, .52, 1.13)
# A slight right-face facet reinforces the reference's volume, without moving
# or curving any horizontal data boundary.
facet = 1 - .06*np.clip((u-.865)/.02,0,1)
light *= facet
edges = np.array(bounds[1:-1], dtype=np.float64)-BT
ids = np.searchsorted(edges, y, side='right')
palette = np.array([rgb(c) for c in PALETTE])
body_rgb = np.clip(palette[ids,None,:]*light[...,None],0,255)
body = np.dstack([body_rgb, mask.astype(np.uint8)*255]).astype(np.uint8)
body_image = Image.fromarray(body,'RGBA')
im.paste(body_image,(round(BX0*S),round(BT*S)),body_image)
# Save native high-resolution semantic mask for reproducible numerical checks.
# It is a diagnostic body mask, not another complete chart.
semantic = np.where(mask,ids[:,None]+1,0).astype(np.uint8)
Image.fromarray(semantic,'L').save(ROOT / 'artifacts/body_band_ids.png')

d = ImageDraw.Draw(im)
# The six larger sections carry readable labels inside the bottle.
for i in [0,1,2,3,4,17]:
    cy = mids[i]
    badge(537,cy,i,25,inside=True)
    name_size = 33 if i in [3,4,17] else 36
    value_size = 25 if i in [3,4] else 27
    text(584, cy-33, names[i], name_size, '#FFFFFF', 'bold')
    text(584, cy+11, f'{values[i]:.1f} tcm  |  {float(values[i]/total*100):.1f}%',value_size,'#FFFFFF')

# Narrow bands: twelve externally spaced labels, in the same supplied order.
# Each badge and endpoint is directly tied to its own stripe center.
leaders = []
left_ids = [5,7,9,11,13,15]
right_ids = [6,8,10,12,14,16]
for side, indices in [('left',left_ids),('right',right_ids)]:
    for row, i in enumerate(indices):
        ly = (1110 if side=='left' else 1139) + row*116
        sy = mids[i]
        rail = (422-row*11) if side=='left' else (1018+row*11)
        if side=='left':
            edge = BX0
            end = 353
            pts = [(edge,sy),(rail,sy),(rail,ly),(end,ly)]
            tx, anchor, bx = 298, 'rt', 331
        else:
            edge = BX1
            end = 1087
            pts = [(edge,sy),(rail,sy),(rail,ly),(end,ly)]
            tx, anchor, bx = 1142, 'lt', 1109
        line(pts,darker(PALETTE[i],.86),1.8)
        ellipse((edge-3.6,sy-3.6,edge+3.6,sy+3.6),PALETTE[i])
        badge(bx,ly,i,22)
        text(tx,ly-29,names[i],30,INK,'bold',anchor)
        text(tx,ly+12,f'{values[i]:.1f} tcm  |  {float(values[i]/total*100):.1f}%',23,MUTED,'regular',anchor)
        leaders.append({'category':names[i],'band_index':i,'anchor_y':sy,'label_y':ly,'points':pts})

# Clear shoulder-to-base span, without treating the guard as part of the data.
line([(BX0,1790),(BX1,1790)], '#B8AD9D',1)
line([(BX0,1786),(BX0,1794)], '#B8AD9D',1)
line([(BX1,1786),(BX1,1794)], '#B8AD9D',1)
text(720,1812,'ALL 18 CATEGORIES  ·  192.0 tcm',20,MUTED,'bold','mt')

# Footer makes encoding and provenance explicit, not a claim of real reserves.
line([(72,1871),(1368,1871)], '#CABDA9',1.4)
text(73,1899,'HEIGHT = SHARE',20,TEAL,'bold')
text(334,1897,'Band height = value ÷ 192.0 × shoulder-to-base height.',23,INK)
text(73,1942,'Synthetic study data  •  tcm = trillion cubic metres  •  Input order retained  •  Area is not encoded',20,MUTED)

im.resize((W,H),Image.Resampling.LANCZOS).save(ROOT / 'final.png')
# Evidence metadata records geometry and public layout facts, not judgments.
records = []
for i,v in enumerate(values):
    records.append({'index':i,'category':names[i],'value':str(v),'share':float(v/total),
                    'y_top':bounds[i],'y_bottom':bounds[i+1],'height_px':bounds[i+1]-bounds[i],
                    'color':PALETTE[i],'label':f'{v:.1f} tcm','label_location':'inside' if i in [0,1,2,3,4,17] else 'external'})
(ROOT / 'artifacts/layout_geometry.json').write_text(json.dumps({
    'canvas':[W,H],'body_bbox':[BX0,BT,BX1,BB],'body_height':BH,'mapping':'y = 600 + 1170 × cumulative_value / 192.0',
    'bands':records,'leaders':leaders,'text_boxes':text_boxes,'total':str(total),'supersampling':S,
    'generated_surface':'assets/body_surface.png','generated_guard':'assets/red_guard_valve.png'
},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Rendered final.png at',W,H,'with',len(values),'proportional bands; total',total)
