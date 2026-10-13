#!/usr/bin/env python3
"""Offline, data-bound infographic. Run through capture_render.py for full renders."""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageChops

ROOT = Path(__file__).resolve().parent

def read_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))

inputs = read_json('input.json')
brief = read_json(inputs['brief'])
data = read_json(inputs['data'])
core = read_json(inputs['critical_metaphor_contract'])
assert data['synthetic'] is True
assert core['visible_title'] == data['title']
assert len(data['series']) == 1
series = data['series'][0]
years = np.array(data['time_points'], dtype=float)
values = np.array(series['values'], dtype=float)
assert len(years) == len(values) == len(data['categories'])
assert all(str(int(y)) == c for y, c in zip(years, data['categories']))
assert np.all(np.diff(years) > 0)
lo, hi = data['value_axis']['minimum'], data['value_axis']['maximum']
assert data['value_axis']['scale'] == 'linear'
W, H = brief['width'], brief['height']
S = 2
INK = '#303834'
MUTED = '#586965'
RED = '#b7352d'
BLUE = '#759aab'
PAPER = '#f6f2e9'
FONT_DIR = ROOT / 'assets/fonts'
font_cache = {}

def font(size, face='sans'):
    key = (size,face)
    if key not in font_cache:
        font_cache[key] = ImageFont.truetype(str(FONT_DIR / (face+'.otf')), round(size*S))
    return font_cache[key]

canvas = Image.new('RGBA', (W*S,H*S), PAPER)
d = ImageDraw.Draw(canvas)

def xy(p): return tuple(round(v*S) for v in p)
def line(points, fill=INK, width=1):
    d.line([xy(p) for p in points], fill=fill, width=max(1,round(width*S)), joint='curve')
def text(x,y,txt,size=28,face='sans',fill=INK,anchor='lt'):
    d.text(xy((x,y)),str(txt),font=font(size,face),fill=fill,anchor=anchor)
def tracking(x,y,txt,size=20,spacing=2.0,fill=MUTED):
    for ch in txt:
        text(x,y,ch,size,'sans_bold',fill)
        x += d.textlength(ch,font=font(size,'sans_bold'))/S + spacing
    return x

def paragraph(x,y,lines,size=32,leading=42,face='serif_italic',fill=INK):
    for ln in lines:
        text(x,y,ln,size,face,fill)
        y+=leading

# A book-page structure: editorial explanation at left, illustrated figure at right.
tracking(100,66,'POPULATION / VISUAL STUDY',19,1.8)
text(1980,65,f"{data['categories'][0]}–{data['categories'][-1]}  /  ANNUAL SERIES",20,'sans',MUTED,'rt')
line([(100,108),(1980,108)],'#b8bdb4',1.3)
d.rounded_rectangle(xy((100,150,403,194)),radius=4*S,fill=RED)
tracking(119,161,data['visible_notice'],22,1.7,'#fffaf2')
text(95,228,'Annual',116,'serif')
text(95,344,'birth rate',116,'serif')
text(100,476,f"{data['categories'][0]}–{data['categories'][-1]}",41,'serif_italic',MUTED)
paragraph(100,554,['Births per 1,000 women','per year'],31,39,'sans')
line([(100,671),(588,671)],'#a5aea7',1.2)
tracking(100,706,'THE PICTURE & THE MEASURE',18,1.1)
paragraph(100,759,[
    'The diaper becomes a page',
    'for a long view of birth rates.',
    'Each bend in the red line is',
    "one year’s observation; every",
    'year in this synthetic series',
    'is connected in time order.'
],34,42,'serif_italic')
line([(100,1058),(588,1058)],'#a5aea7',1.2)
min_i=int(np.argmin(values)); max_i=int(np.argmax(values))
tracking(100,1090,f"LOWEST / {data['categories'][min_i]}",17,1.0)
tracking(350,1090,f"HIGHEST / {data['categories'][max_i]}",17,1.0)
text(100,1121,f'{values[min_i]:.1f}',64,'serif',RED)
text(350,1121,f'{values[max_i]:.1f}',64,'serif',RED)
paragraph(100,1224,[
    f'Read on a {lo}–{hi} linear scale.',
    f'The vertical axis starts at {lo}.',
    'Values are simulated, not historical',
    'population statistics.'
],25,34,'sans',MUTED)

# Blue bookplate. The generated figure crosses its frame, as in the design reference.
d.rectangle(xy((712,243,1980,1392)),fill='#dce8eb',outline='#a0b3b9',width=2*S)
# All asset and mask coordinates use this one affine transform.
asset_rect = (790,130,1920,1484)
ax,ay,ar,ab = asset_rect
aw,ah=ar-ax,ab-ay
baby = Image.open(ROOT/'assets/baby_clean.png').convert('RGBA')
source_size=baby.size
scaled=baby.resize((aw*S,ah*S),Image.Resampling.LANCZOS)
canvas.alpha_composite(scaled,xy((ax,ay)))
diaper=Image.open(ROOT/'assets/diaper_mask.png').convert('L').resize((aw*S,ah*S),Image.Resampling.LANCZOS)
grid_mask=Image.open(ROOT/'assets/grid_mask.png').convert('L').resize((aw*S,ah*S),Image.Resampling.LANCZOS)
# Cartesian data mapping. Neither fabric contours nor selected labels warp data.
plot = {'left':925.0,'right':1815.0,'top':855.0,'bottom':1180.0,
        'x_min':float(years[0]),'x_max':float(years[-1]),'y_min':float(lo),'y_max':float(hi)}
def X(t): return plot['left']+(t-plot['x_min'])/(plot['x_max']-plot['x_min'])*(plot['right']-plot['left'])
def Y(v): return plot['bottom']-(v-lo)/(hi-lo)*(plot['bottom']-plot['top'])
pts=[(X(t),Y(v)) for t,v in zip(years,values)]

# Grid is clipped to the generated white fabric. It has no opaque chart backdrop.
grid=Image.new('RGBA',(aw*S,ah*S),(0,0,0,0))
gd=ImageDraw.Draw(grid)
def gl(points,rgba,width):
    gd.line([xy((x-ax,y-ay)) for x,y in points],fill=rgba,width=max(1,round(width*S)))
for val in range(int(lo),int(hi)+1,10):
    major=val%20==0
    gl([(ax,Y(val)),(ar,Y(val))],(103,148,169,132 if major else 84),1.15 if major else 0.8)
for year in range(int(years[0]),int(years[-1])+1,5):
    major=year%10==0
    gl([(X(year),plot['top']),(X(year),plot['bottom'])],(103,148,169,126 if major else 83),1.15 if major else 0.8)
grid.putalpha(ImageChops.multiply(grid.getchannel('A'),grid_mask))
canvas.alpha_composite(grid,xy((ax,ay)))
d=ImageDraw.Draw(canvas)

# Shared numeric axis, including the explicitly non-zero lower endpoint.
text(777,810,'Rate',24,'sans',MUTED,'rt')
line([(790,plot['top']),(790,plot['bottom'])],'#88a4ae',1.1)
for val in range(int(lo),int(hi)+1,20):
    y=Y(val)
    text(773,y,str(val),25,'sans',MUTED,'rm')
    line([(784,y),(799,y)],'#7596a4',1.2)
# Legend sits on the diaper, above the quantitative plotting area.
line([(1122,813),(1167,813)],RED,5.8)
text(1183,800,series['label'],25,'sans',INK)

# Unsmoothed, ordered polyline uses all annual observations.
line(pts,RED,6.2)
for x,y in pts:
    d.ellipse(xy((x-3.1,y-3.1,x+3.1,y+3.1)),fill=RED)

# Exact current-data annotations, not historical causal claims.
annotations=[]
def mark(i,label_x,label_y,anchor='mt',below=False):
    x,y=pts[i]
    r=7.0
    d.ellipse(xy((x-r,y-r,x+r,y+r)),fill='#fffaf3',outline=RED,width=3*S)
    label=f"{data['categories'][i]} · {values[i]:.1f}"
    # Short, neutral leader; red is reserved for data marks.
    if below:
        mid_y=label_y-9
        line([(x,y+r+3),(x,mid_y),(label_x,mid_y)],'#697e83',1.15)
    else:
        mid_y=label_y+27
        line([(x,y-r-3),(x,mid_y),(label_x,mid_y)],'#697e83',1.15)
    text(label_x,label_y,label,25,'sans_bold',INK,anchor)
    annotations.append({'category':data['categories'][i],'value':float(values[i]),'text':label,'point':[x,y],'label_anchor':[label_x,label_y]})
mark(min_i,1021,1121,'mt',True)
mark(max_i,1267,863,'mt',False)
mid_i=data['categories'].index('1964')
mark(mid_i,1595,1113,'mt',True)
mark(len(values)-1,1754,865,'mt',False)

# Decade ticks keep the annual semantic roles readable without 61 overlapping labels.
for yr in range(int(years[0]),int(years[-1])+1,10):
    x=X(yr)
    text(x,1203,str(yr),25,'sans',INK,'mt')
text((plot['left']+plot['right'])/2,1245,'Year',24,'sans',MUTED,'mt')

# Compact, truthful book-page footer.
line([(100,1515),(1980,1515)],'#b8bdb4',1.3)
text(100,1540,'SOURCE  Local synthetic study data · Not historical statistics',20,'sans',MUTED)
text(1980,1540,f'{len(values)} annual samples  ·  {len(data["series"])} series',20,'sans',MUTED,'rt')

out=canvas.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
out.save(ROOT/'final.png',compress_level=7)
artifact=ROOT/'artifacts'; artifact.mkdir(exist_ok=True)
geometry={
  'canvas':[W,H],
  'plot':plot,
  'asset_transform':{'source_crop_xyxy':[176,0,1080,1254],'processed_size':list(source_size),'destination_xyxy':list(asset_rect),'scale_x':aw/source_size[0],'scale_y':ah/source_size[1]},
  'line_color':RED,'line_width_px':6.2,'annual_count':len(values),
  'axis_ticks':list(range(int(lo),int(hi)+1,20)),
  'year_ticks':list(range(int(years[0]),int(years[-1])+1,10)),
  'series_label':series['label'],'unit':series['unit'],
  'points':[{'category':c,'time':float(t),'value':float(v),'x':x,'y':y} for c,t,v,(x,y) in zip(data['categories'],years,values,pts)],
  'annotations':annotations,
  'font_assets':[str(p.relative_to(ROOT)) for p in sorted(FONT_DIR.glob('*.otf'))]
}
(artifact/'geometry.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(f'Wrote final.png ({W} × {H}); {len(pts)} ordered annual vertices; axis {lo}–{hi}.')
