#!/usr/bin/env python3
"""Offline rendering of the frozen rice market-share infographic.

Run using the frozen capture_render.py wrapper. Angular geometry and every
category/value label are determined from data.json. The photographic source is
local built-in image-generation material. No network/model calls are made.
"""
from pathlib import Path
import json
import math
import hashlib
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent

def read_json(rel):
    return json.loads((ROOT / rel).read_text(encoding='utf-8'))

INPUT = read_json('input.json')
BRIEF = read_json(INPUT['brief'])
DATA = read_json(INPUT['data'])
CORE = read_json(INPUT['critical_metaphor_contract'])
assert (ROOT / INPUT['reference']).is_file()  # frozen reference stays available
assert DATA['synthetic'] and DATA['series'][0]['unit'] == '%'
assert CORE['topic_zh'] == DATA['topic_zh']
assert len(DATA['series']) == 1
VALUES = np.array(DATA['series'][0]['values'], dtype=float)
CATEGORIES = DATA['categories']
PARENTS = DATA['category_parent']
TOTAL = float(VALUES.sum())
assert len(VALUES) == len(CATEGORIES) == len(PARENTS) == 6
assert np.all(VALUES > 0) and abs(TOTAL - DATA['series_totals']['S1']) < 1e-9

W, H = BRIEF['width'], BRIEF['height']
S = 2
CX, CY, R = 960., 1100., 610.
START = -90.
SWEEPS = VALUES / TOTAL * 360
BOUNDS = np.concatenate([[START], START + np.cumsum(SWEEPS)])
MIDPOINTS = (BOUNDS[:-1] + BOUNDS[1:])/2
COLORS = ['#73cbd4','#5aa5cd','#af8fbd','#e0b268','#dc945e','#a8c789']
INK = '#1d3029'
MUTED = '#68756c'
BG = '#f8f7f1'
RULE = '#d7dcd1'
RED = '#a9422d'

@lru_cache(None)
def font(size, bold=False, cjk=False):
    name = 'NotoSansCJK-Regular.ttc' if cjk else ('NimbusSans-Bold.otf' if bold else 'NimbusSans-Regular.otf')
    return ImageFont.truetype(str(ROOT/'assets'/name), round(size*S))

image = Image.new('RGB', (W*S,H*S), BG)
draw = ImageDraw.Draw(image)
text_boxes = []

def xy(point):
    return tuple(round(float(v)*S) for v in point)

def text(pos, value, size, color=INK, bold=False, cjk=False, anchor='lt', role=None):
    value = str(value)
    f = font(size,bold,cjk)
    box = draw.textbbox(xy(pos),value,font=f,anchor=anchor)
    draw.text(xy(pos),value,font=f,fill=color,anchor=anchor)
    if role:
        text_boxes.append({'role':role,'text':value,'bbox':[round(v/S,2) for v in box]})
    return draw.textlength(value,font=f)/S

def line(points, color, width=2):
    draw.line([xy(p) for p in points], fill=color, width=max(1,round(width*S)), joint='curve')

def dot(point,r,color):
    x,y=point
    draw.ellipse(xy((x-r,y-r,x+r,y+r)),fill=color)

def value_string(v):
    return format(float(v), '.15g') + DATA['series'][0]['unit']

def pt(angle, radius=R):
    theta = math.radians(float(angle))
    return (CX+radius*math.cos(theta), CY+radius*math.sin(theta))

def color_rgb(color):
    return np.array(tuple(bytes.fromhex(color[1:])),dtype=np.float32)

# Typographic header. The notice and title together express the frozen title.
draw.rounded_rectangle(xy((120,100,460,157)),radius=9*S,fill=RED)
text((143,115),DATA['visible_notice'],29,'#ffffff',True,role='synthetic_notice')
text((118,203),'Rice market shares',144,INK,True,role='topic_title')
text((123,363),DATA['series'][0]['label']+'（%）',31,MUTED,cjk=True,role='metric_and_unit')
text((457,367),'Share of the total rice market',31,MUTED,role='metric_explanation')
text((1875,103),'TOTAL MARKET',25,MUTED,True,anchor='rt',role='total_heading')
text((1875,151),value_string(TOTAL),70,INK,True,anchor='rt',role='total_value')
line([(120,426),(1880,426)],RULE,2)

# Soft physical shadow is non-data decoration. Data geometry remains a circle.
shadow = Image.new('RGBA', image.size, (0,0,0,0))
sd = ImageDraw.Draw(shadow)
sd.ellipse(xy((CX-R+8,CY-R+21,CX+R+8,CY+R+21)),fill=(55,60,40,42))
shadow = shadow.filter(ImageFilter.GaussianBlur(20*S))
image = Image.alpha_composite(image.convert('RGBA'),shadow).convert('RGB')
draw = ImageDraw.Draw(image)

# Single polar transform for all sector colors, separators, leaders, and audit.
size = round(2*R*S)
photo = np.asarray(Image.open(ROOT/'assets/rice_spoon_centered.png').convert('RGB').resize((size,size),Image.Resampling.LANCZOS),dtype=np.float32)
wood = np.asarray(Image.open(ROOT/'assets/wood_mask.png').convert('L').resize((size,size),Image.Resampling.LANCZOS),dtype=np.float32)/255
local_y, local_x = np.mgrid[:size,:size]
dx = (local_x+.5)/S - R
dy = (local_y+.5)/S - R
rho = np.hypot(dx,dy)
angles = (np.degrees(np.arctan2(dy,dx)) - START) % 360
ends = np.cumsum(SWEEPS)
sector_id = np.searchsorted(ends, angles, side='right').clip(0,len(VALUES)-1)
palette = np.array([color_rgb(c) for c in COLORS],dtype=np.float32)
# Dye the photographic rice while retaining the generated grain contrast.
luminance = photo @ np.array([.2126,.7152,.0722],dtype=np.float32)
colored = palette[sector_id] + (luminance[...,None]-202)*.86
colored = np.clip(colored,0,255)
# Restore original walnut wood; rice in the bowl still follows the sectors.
keep_wood = (wood*.97)[...,None]
material = np.uint8(np.clip(colored*(1-keep_wood)+photo*keep_wood,0,255))
alpha = np.uint8(np.clip((R-rho)*S+.5,0,1)*255)
material_rgba = Image.fromarray(material).convert('RGBA')
material_rgba.putalpha(Image.fromarray(alpha))
image.paste(material_rgba,xy((CX-R,CY-R)),material_rgba)
draw = ImageDraw.Draw(image)

# Straight equal-width dividers all meet the same center; no exploded wedges.
for angle in BOUNDS[:-1]:
    line([(CX,CY),pt(angle)],'#fffef8',5)
draw.ellipse(xy((CX-R,CY-R,CX+R,CY+R)),outline='#fffef8',width=3*S)
dot((CX,CY),3.2,'#fffef8')

# Label positions are editorial layout; leader anchors come from data midangles.
# Every subcategory is explicitly attached to its semantic parent.
label_specs = [
    {'pos':(1360,459),'route':[(1245,486),(1328,486)]},
    {'pos':(1640,708),'route':[(1608,745)]},
    {'pos':(1675,1000),'route':[(1621,1079),(1643,1024)]},
    {'pos':(1640,1255),'route':[(1608,1368),(1608,1282)]},
    {'pos':(1435,1575),'route':[(1403,1603)]},
    {'pos':(96,940),'route':[(268,1250),(112,1250),(112,1150)]},
]
leader_records=[]
for i,(category,parent,v,mid,color,spec) in enumerate(zip(CATEGORIES,PARENTS,VALUES,MIDPOINTS,COLORS,label_specs)):
    # Darker matching line colors remain legible on the warm paper.
    leader_color = tuple(np.uint8(color_rgb(color)*.63))
    points=[pt(mid,R-4),pt(mid,R+36)]+spec['route']
    line(points,leader_color,2.7)
    dot(points[-1],5,leader_color)
    x,y=spec['pos']
    text((x,y),parent,32,INK,True,role=f'category_{i+1}_parent')
    if ' / ' in category:
        sub = category.split(' / ',1)[1]
        text((x,y+43),sub,29,MUTED,role=f'category_{i+1}_subcategory')
        val_y=y+90
    else:
        val_y=y+49
    text((x-2,val_y),value_string(v),108 if i==5 else 74,INK,True,role=f'value_{i+1}')
    if i==5:
        text((x+2,y+170),'of the total market',27,MUTED,role='largest_share_detail')
    leader_records.append({'category':category,'value':float(v),'midpoint_deg':float(mid),'points':points})

# Compact explanation of synthetic scope and encoding.
line([(120,1818),(1880,1818)],RULE,2)
text((120,1853),'SYNTHETIC SCOPE',24,RED,True,role='scope_label')
text((413,1850),'Subcategories are mutually exclusive parts of their named rice producer.',27,MUTED,role='subcategory_scope')
text((120,1902),'READING THE CHART',24,INK,True,role='encoding_label')
text((413,1899),'Angle shows market share. Rice grains and spoon size do not encode values.',27,MUTED,role='encoding_explanation')

# Final export. Intermediate diagnostic masks are not additional full renders.
image = image.resize((W,H),Image.Resampling.LANCZOS)
image.save(ROOT/'final.png',format='PNG',optimize=True)
art = ROOT/'artifacts'
art.mkdir(exist_ok=True)
geometry = {
    'canvas':[W,H],'center':[CX,CY],'radius':R,'clockwise':True,
    'start_degrees_from_positive_x':START,'total':TOTAL,
    'categories':CATEGORIES,'values':VALUES.tolist(),'colors':COLORS,
    'boundary_angles_degrees':BOUNDS.tolist(),'sector_sweeps_degrees':SWEEPS.tolist(),
    'separator_width_px':5,'leaders':leader_records,'text_boxes':text_boxes,
    'raw_asset_sha256':hashlib.sha256((ROOT/'assets/raw_01.png').read_bytes()).hexdigest(),
    'final_sha256':hashlib.sha256((ROOT/'final.png').read_bytes()).hexdigest(),
}
(art/'render_geometry.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(f'Rendered {W} x {H}: {len(VALUES)} sectors; total {TOTAL:g}%.')
