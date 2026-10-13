#!/usr/bin/env python3
"""Offline infographic renderer. Run from this directory with the supplied Python.

The globe and perspective plinth are illustrative. Both quantitative polylines
use one untransformed Cartesian mapping from the frozen data.json. All labels
and coordinates are computed here, with no network or model calls.
"""
from pathlib import Path
import json, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT/'data.json').read_text(encoding='utf-8'))
BRIEF = json.loads((ROOT/'brief.json').read_text(encoding='utf-8'))
CORE = json.loads((ROOT/'core.json').read_text(encoding='utf-8'))
INPUT = json.loads((ROOT/'input.json').read_text(encoding='utf-8'))
# Keep the reference and all frozen input dependencies available offline.
assert (ROOT/INPUT['reference']).is_file()
assert len(DATA['categories']) == len(DATA['time_points'])
assert all(len(s['values']) == len(DATA['categories']) for s in DATA['series'])
assert [s['label']+' ('+s['unit']+')' for s in DATA['series']] == CORE['semantic_roles_zh']
W,H = BRIEF['width'], BRIEF['height']
S = 2
PAPER = '#f3f0e9'
INK = '#181b1b'
MUTED = '#656b68'
RED = '#c31d27'
BLACK = '#141718'
BOARD = '#2a3030'
CREAM = '#fff9ec'
COLORS = {'price_inflation': BLACK, 'output_growth': RED}
FONTDIR = ROOT/'assets/fonts'
FONT_FILES = {'regular':'NimbusSans-Regular.otf','bold':'NimbusSans-Bold.otf'}
fonts={}

def font(size, weight='regular'):
    key=(size,weight)
    if key not in fonts: fonts[key]=ImageFont.truetype(str(FONTDIR/FONT_FILES[weight]),round(size*S))
    return fonts[key]

def xy(p): return tuple(round(v*S) for v in p)
def box(b): return tuple(round(v*S) for v in b)
canvas = Image.new('RGBA',(W*S,H*S),PAPER)
draw = ImageDraw.Draw(canvas)
text_records=[]

def text(x,y,value,size=24,color=INK,weight='regular',anchor='lt',spacing=6):
    if '\n' in str(value):
        for i,part in enumerate(str(value).split('\n')):
            text(x,y+i*(size+spacing),part,size,color,weight,anchor,spacing)
        return
    f=font(size,weight)
    draw.multiline_text(xy((x,y)),str(value),font=f,fill=color,anchor=anchor,spacing=round(spacing*S))
    bb=draw.multiline_textbbox(xy((x,y)),str(value),font=f,anchor=anchor,spacing=round(spacing*S))
    text_records.append({'text':str(value),'bbox':[round(v/S,2) for v in bb]})

def line(points,fill,width=1):
    draw.line([xy(p) for p in points],fill=fill,width=max(1,round(width*S)),joint='curve')

def rect(b,fill,outline=None,width=1,radius=0):
    if radius: draw.rounded_rectangle(box(b),round(radius*S),fill,outline,round(width*S))
    else: draw.rectangle(box(b),fill,outline,round(width*S))

def ellipse(b,fill,outline=None,width=1): draw.ellipse(box(b),fill,outline,round(width*S))

def dashed(p,q,color,width=2,dash=9,gap=7):
    dx,dy=q[0]-p[0],q[1]-p[1]; length=math.hypot(dx,dy)
    if not length:return
    pos=0
    while pos<length:
        stop=min(pos+dash,length)
        line([(p[0]+dx*pos/length,p[1]+dy*pos/length),(p[0]+dx*stop/length,p[1]+dy*stop/length)],color,width)
        pos+=dash+gap

def pill(x,y,label,color=INK,size=24,fill=CREAM,pad=9):
    f=font(size,'bold'); b=f.getbbox(label); w=f.getlength(label)/S
    rect((x-pad,y-5,x+w+pad,y+(b[3]-b[1])/S+8),fill,radius=5)
    text(x,y,label,size,color,'bold')
    return w

# Page title, exact semantic roles and clearly visible synthetic notice.
text(80,64,DATA['title'].removeprefix('SYNTHETIC '),66,INK,'bold')
text(83,152,f"Annual change from the previous year (%)  ·  {DATA['categories'][0]}–{DATA['categories'][-1]}",28,MUTED)
rect((1572,67,1917,123),INK,radius=3)
text(1744.5,82,DATA['visible_notice'],27,CREAM,'bold','mt')
text(1917,150,'Illustrative economic series',22,MUTED,anchor='rt')
line([(82,207),(1918,207)],'#bdbfb6',1.4)

# Recessed dark backboard and shallow perspective plinth.
rect((525,238,1513,914),'#1c2222')
rect((528,240,1509,911),BOARD)
# A restrained screenprint grain is generated deterministically, not by a model.
rng=np.random.default_rng(39)
noise=rng.normal(0,1.45,(674,985,1))
b=np.clip(np.array([42,48,48],dtype=np.float32)[None,None,:]+noise,0,255).astype('uint8')
board_im=Image.fromarray(b,'RGB').resize((985*S,674*S),Image.Resampling.NEAREST).convert('RGBA')
canvas.alpha_composite(board_im,xy((528,240)))
# Floor shadow is independent of quantitative geometry.
shadow=Image.new('RGBA',canvas.size,(0,0,0,0)); sd=ImageDraw.Draw(shadow)
sd.polygon([xy(p) for p in [(160,1017),(1845,1017),(1858,1094),(160,1094)]],fill=(30,35,32,35))
shadow=shadow.filter(ImageFilter.GaussianBlur(14*S));canvas.alpha_composite(shadow)
draw=ImageDraw.Draw(canvas)
draw.polygon([xy(p) for p in [(525,911),(1513,911),(1840,1001),(159,1001)]],fill='#e1e0d6',outline='#828983',width=2*S)
draw.polygon([xy(p) for p in [(159,1001),(1840,1001),(1840,1086),(159,1086)]],fill='#d9cdc6',outline='#828983',width=2*S)
line([(162,1004),(1837,1004)],'#f3e9df',2)

# Continuous generated globe, pool and drips. The sphere is not chart geometry.
# Only the lower 404 source pixels (pool and drips) are compressed vertically.
# This preserves a circular upper globe and keeps all drips within the canvas.
asset=Image.open(ROOT/'assets/globe_soft.png').convert('RGBA')
aw,ah=asset.size
cut=620
bottom_h=230
warped=Image.new('RGBA',(aw,cut+bottom_h),(0,0,0,0))
warped.alpha_composite(asset.crop((0,0,aw,cut)),(0,0))
warped.alpha_composite(asset.crop((0,cut,aw,ah)).resize((aw,bottom_h),Image.Resampling.LANCZOS),(0,cut))
asset_x,asset_y=232,238
warped=warped.resize((aw*S,(cut+bottom_h)*S),Image.Resampling.LANCZOS)
canvas.alpha_composite(warped,xy((asset_x,asset_y)))
draw=ImageDraw.Draw(canvas)

# Shared, flat data coordinate system. No perspective or smoothing is applied.
x0,x1=650.,1350.
ytop,ybottom=290.,870.
vmin,vmax=map(float,(DATA['value_axis']['minimum'],DATA['value_axis']['maximum']))
times=np.asarray(DATA['time_points'],dtype=float)
xmin,xmax=float(times.min()),float(times.max())
def X(t):return x0+(float(t)-xmin)/(xmax-xmin)*(x1-x0)
def Y(v):return ybottom-(float(v)-vmin)/(vmax-vmin)*(ybottom-ytop)
base_y=Y(0)
axis_x=1420
# The globe's curved graticule is distinct from sparse horizontal scale guides.
grid=Image.new('RGBA',canvas.size,(0,0,0,0));gd=ImageDraw.Draw(grid)
for v in (15,10,5):gd.line([xy((628,Y(v))),xy((axis_x,Y(v)))],fill=(255,249,236,70),width=1*S)
canvas.alpha_composite(grid);draw=ImageDraw.Draw(canvas)
line([(axis_x,ytop),(axis_x,ybottom)],'#bbbfb6',1)
for v in (15,10,5,0,-1):
    line([(axis_x-6,Y(v)),(axis_x+7,Y(v))],CREAM,1.6)
    text(axis_x+16,Y(v),('−1' if v==-1 else str(v)),21,CREAM,'regular','lm')
text(axis_x+17,258,'%',23,CREAM,'bold')
# Zero is explicit and a true negative value lies below it.
dashed((628,base_y),(axis_x,base_y),'#fff8e9',2.0,8,6)
text(1469,base_y,'zero',17,'#d9ded6',anchor='lm')

# Forecast identity comes only from the frozen year boundary.
forecast=float(DATA['forecast_boundary'])
fi=next(i for i,t in enumerate(times) if t>=forecast)
fx=X(forecast)
dashed((fx,ytop+1),(fx,ybottom),'#f8f2e8',1.2,4,7)
text(fx,268,str(DATA['categories'][fi]),18,CREAM,'bold','mt')

# Draw both polylines, then all markers. Halo improves contrast over the map.
geometry=[]
for series in DATA['series']:
    color=COLORS[series['id']]
    pts=[(X(t),Y(v)) for t,v in zip(times,series['values'])]
    history=pts[:fi]
    line(history,CREAM,10.3)
    line(history,color,6.6)
    # Forecast segment reaches the final actual coordinate, with a dashed stroke.
    for a,b in zip(pts[fi-1:],pts[fi:]):
        dashed(a,b,CREAM,10.3,8,6)
        dashed(a,b,color,6.6,8,6)
    geometry.append({'series':series['id'],'label':series['label'],'color':color,'unit':series['unit'],
                     'points':[{'year':str(year),'value':float(v),'x':p[0],'y':p[1],'projected':bool(t>=forecast)}
                               for year,t,v,p in zip(DATA['categories'],times,series['values'],pts)]})
for series,g in zip(DATA['series'],geometry):
    color=COLORS[series['id']]
    for p in g['points']:
        r=5.2 if p['projected'] else 4.0
        ellipse((p['x']-r-1.8,p['y']-r-1.8,p['x']+r+1.8,p['y']+r+1.8),CREAM)
        ellipse((p['x']-r,p['y']-r,p['x']+r,p['y']+r),CREAM if p['projected'] else color,color,2.2)

# Key labels are bound to the actual input points.
price=DATA['series'][0];output=DATA['series'][1]
peak_i=int(np.argmax(price['values']));low_i=int(np.argmin(output['values']))
peak=(X(times[peak_i]),Y(price['values'][peak_i]))
line([peak,(peak[0]+15,peak[1]-31)],BLACK,1.5)
pill(peak[0]-16,peak[1]-57,f"{price['values'][peak_i]:.1f}%",BLACK,24)
low=(X(times[low_i]),Y(output['values'][low_i]))
line([low,(low[0]+5,low[1]+13)],RED,1.5)
pill(low[0]-26,low[1]+17,f"{output['values'][low_i]:.1f}%".replace('-', '−'),RED,23)
for series in DATA['series']:
    p=(X(times[-1]),Y(series['values'][-1]))
    line([(p[0]+7,p[1]),(p[0]+18,p[1])],COLORS[series['id']],1.6)
    pill(p[0]+24,p[1]-12,f"{series['values'][-1]:.1f}",COLORS[series['id']],24)

# All 23 annual positions are ticked; alternate labels avoid crowding.
line([(x0,900),(x1,900)],'#49514d',1.1)
for i,(t,year) in enumerate(zip(times,DATA['categories'])):
    xx=X(t)
    line([(xx,900),(xx,908 if i%2==0 else 904)],'#3c433f',1.2)
    if i%2==0 or i==len(times)-1:
        f=font(19,'bold' if i==len(times)-1 else 'regular');tw=f.getlength(year)/S
        rect((xx-tw/2-5,914,xx+tw/2+5,940),PAPER,radius=2)
        text(xx,918,year,19,INK,'bold' if i==len(times)-1 else 'regular','mt')
text(609,925,'YEAR',15,'#424b46','bold','rm')

# Semantic legend and reading context stay outside the data-bearing surface.
text(84,279,'TWO ANNUAL MEASURES',17,MUTED,'bold')
line([(84,340),(144,340)],BLACK,6.6)
ellipse((110,335.6,118.8,344.4),BLACK)
text(163,324,price['label'],27,INK,'bold')
text(163,365,'Percent change (%)',22,MUTED)
line([(84,429),(144,429)],RED,6.6)
ellipse((110,424.6,118.8,433.4),RED)
text(163,413,output['label'],27,RED,'bold')
text(163,454,'Real output · percent change',21,MUTED)
line([(85,518),(421,518)],'#b8beb5',1)
text(84,547,'ONE SHARED SCALE',17,MUTED,'bold')
text(82,585,'−1% to 15%',42,INK,'bold')
close_i=int(np.argmin(np.abs(np.asarray(price['values'])-np.asarray(output['values']))))
text(84,651,f"{DATA['categories'][close_i]} / CLOSE VALUES",17,MUTED,'bold')
text(83,687,f"{price['values'][close_i]:.1f}%",32,INK,'bold')
text(259,687,f"{output['values'][close_i]:.1f}%",32,RED,'bold')
text(84,731,'Price inflation',18,MUTED)
text(260,731,'Output growth',18,MUTED)
text(84,788,f'{int(forecast)} is projected.',24,INK,'bold')
dashed((84,839),(142,839),BLACK,4.5,9,7)
text(160,826,'Dashed final segment',21,MUTED)

# Derived key facts use the current synthetic values, never reference values.
rx=1590
text(rx,280,'PRICE INFLATION PEAK',17,MUTED,'bold')
text(rx-3,314,f"{price['values'][peak_i]:.1f}%",74,INK,'bold')
text(rx,400,DATA['categories'][peak_i],26,MUTED)
line([(rx,454),(1916,454)],'#b8beb5',1)
text(rx,487,'OUTPUT CONTRACTION',17,MUTED,'bold')
text(rx-3,522,f"{output['values'][low_i]:.1f}%".replace('-', '−'),74,RED,'bold')
text(rx,608,DATA['categories'][low_i]+' · below zero',26,MUTED)
line([(rx,663),(1916,663)],'#b8beb5',1)
text(rx,696,f'{int(forecast)} / PROJECTED',17,MUTED,'bold')
text(rx,734,f"{price['values'][-1]:.1f}%",40,INK,'bold')
text(rx+127,746,'Price inflation',21,MUTED)
text(rx,792,f"{output['values'][-1]:.1f}%",40,RED,'bold')
text(rx+127,804,'Output growth',21,MUTED)

# Plinth caption and provenance note; no real-country statistical claims.
text(206,1029,'SYNTHETIC DATA',20,INK,'bold')
text(206,1060,'A melting-world illustration of two annual economic series',19,'#535b56')
text(1793,1030,f"{len(times)} years",26,INK,'bold','rt')
text(1793,1066,f"{DATA['categories'][0]}–{DATA['categories'][-1]}",19,'#535b56',anchor='rt')
text(82,1111,'Synthetic study data · illustrative values, not observed economic statistics.',18,MUTED)
text(1918,1111,'Percent (%)  /  Annual change',18,MUTED,anchor='rt')

# Raster output and an explicit geometry record support independent pixel checks.
out=canvas.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
out.save(ROOT/'final.png',compress_level=7)
art=ROOT/'artifacts';art.mkdir(exist_ok=True)
record={'canvas':[W,H],'shared_axis':{'minimum':vmin,'maximum':vmax,'scale':'linear','x0':x0,'x1':x1,'y_top':ytop,'y_bottom':ybottom,'zero_y':base_y,'pixels_per_percent':(ybottom-ytop)/(vmax-vmin)},
        'series':geometry,'category_tick_count':len(times),'labeled_years':[str(y) for i,y in enumerate(DATA['categories']) if i%2==0 or i==len(times)-1],
        'forecast_boundary':forecast,'forecast_encoding':'Final segment dashed; endpoint hollow; 1982 explicitly projected',
        'asset_transform':{'source':'assets/globe_soft.png','position':[asset_x,asset_y],'cut_y':cut,'upper_scale':[1,1],'lower_scale':[1,bottom_h/(ah-cut)],'chart_transform':'none'},
        'text_boxes':text_records}
(art/'geometry.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'output':'final.png','size':[W,H],'points_per_series':len(times),'series':len(geometry),'forecast_year':forecast}))

if __name__=='__main__':
    pass
