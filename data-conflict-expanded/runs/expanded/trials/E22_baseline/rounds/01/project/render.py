#!/usr/bin/env python3
"""Offline, deterministic composition of the supplied synthetic debt series.

Run only through capture_render.py in this experiment. The image model is not
called here. All quantitative positions share the x()/y() transforms below.
"""
from pathlib import Path
import json
import math
import hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

ROOT = Path(__file__).resolve().parent

def read_json(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))

inputs = read_json('input.json')
brief = read_json(inputs['brief'])
data = read_json(inputs['data'])
core = read_json(inputs['critical_metaphor_contract'])
W, H = brief['width'], brief['height']
S = 2
L, R, T, B = 148.0, 1480.0, 540.0, 1768.0
series = data['series'][0]
years = np.asarray(data['time_points'], dtype=float)
values = np.asarray(series['values'], dtype=float)
assert len(years) == len(values) == len(data['categories'])
assert all(float(c) == t for c,t in zip(data['categories'],years))
assert data['value_axis'] == {'minimum':0,'maximum':130,'scale':'linear'}
assert data['synthetic'] and core['visible_title'] == data['title']
assert np.all(np.diff(years) > 0)
vmin, vmax = data['value_axis']['minimum'], data['value_axis']['maximum']
forecast = data['forecast_boundary']
cut = int(np.searchsorted(years,forecast))

def x(t):
    return L+(float(t)-years[0])/(years[-1]-years[0])*(R-L)

def y(v):
    return B-(float(v)-vmin)/(vmax-vmin)*(B-T)

points = [(x(t),y(v)) for t,v in zip(years,values)]
points_px = [(int(round(px*S)),int(round(py*S))) for px,py in points]
cream = '#f7f0dd'
muted = '#9bb3c9'
soft = '#698dab'
red = '#ff726b'
navy = '#102d4c'

# Fine generated textile luminance, with a restrained navy tonal field.
tile = np.asarray(Image.open(ROOT/'assets/weave_luma.png'),dtype=np.float32)
yt, xt = np.indices((H,W))
weave = tile[yt % tile.shape[0],xt % tile.shape[1]]-128.0
vignette = np.clip(1.0-((xt-W*.48)/(W*.92))**2-((yt-H*.48)/(H*.86))**2,0,1)
base = np.zeros((H,W,3),dtype=np.float32)
for ch,(b,glow) in enumerate(zip([12,34,57],[4,11,17])):
    base[:,:,ch] = b + vignette*glow + weave*0.072
base = np.uint8(np.clip(base,0,255))
canvas = Image.fromarray(base).resize((W*S,H*S),Image.Resampling.BICUBIC)
draw = ImageDraw.Draw(canvas)
fonts={}
font_files={
 'regular':'NimbusSans-Regular.otf',
 'bold':'NimbusSans-Bold.otf',
 'display':'NimbusSansNarrow-Bold.otf',
 'italic':'NimbusSans-Italic.otf'
}
text_records=[]

def font(size,face='regular'):
    key=(size,face)
    if key not in fonts:
        fonts[key]=ImageFont.truetype(str(ROOT/'assets'/font_files[face]),round(size*S))
    return fonts[key]

def text(txt,px,py,size=26,fill=cream,face='regular',align='left',tracking=0,record=True):
    f=font(size,face)
    if tracking:
        width=sum(f.getlength(c) for c in txt)+(len(txt)-1)*tracking*S
    else:
        width=f.getlength(txt)
    bx=px*S
    if align=='right': bx-=width
    elif align=='center': bx-=width/2
    if tracking:
        start=bx
        for c in txt:
            draw.text((bx,py*S),c,font=f,fill=fill,anchor='lt')
            bx+=f.getlength(c)+tracking*S
        bbox=[start/S,py,(start+width)/S,py+f.getbbox(txt,anchor='lt')[3]/S]
    else:
        bb=draw.textbbox((bx,py*S),txt,font=f,anchor='lt')
        draw.text((bx,py*S),txt,font=f,fill=fill,anchor='lt')
        bbox=[z/S for z in bb]
    if record: text_records.append({'text':txt,'bbox':bbox})
    return bbox

def line(coords,fill,width=1):
    draw.line([(round(a*S),round(b*S)) for a,b in coords],fill=fill,width=max(1,round(width*S)),joint='curve')

def dashed(coords,fill,width=3,on=10,off=8):
    # Dash phase follows the actual multi-segment polyline, not a smoothed curve.
    phase=0.0
    period=on+off
    for p,q in zip(coords[:-1],coords[1:]):
        dx,dy=q[0]-p[0],q[1]-p[1]
        length=math.hypot(dx,dy)
        if length==0: continue
        dist=0.0
        while dist < length-1e-9:
            at=phase % period
            ink=at < on
            remaining=(on-at) if ink else (period-at)
            step=min(remaining,length-dist)
            if ink and step>1e-9:
                a=(p[0]+dx*dist/length,p[1]+dy*dist/length)
                b=(p[0]+dx*(dist+step)/length,p[1]+dy*(dist+step)/length)
                line([a,b],fill,width)
            dist+=step
            phase=(phase+step)%period
            if step<1e-9: phase+=1e-7

def dot(pt,r=5,fill=red,outline=None,width=1):
    px,py=pt
    draw.ellipse(((px-r)*S,(py-r)*S,(px+r)*S,(py+r)*S),fill=fill,outline=outline,width=round(width*S))

# Header: a conspicuous synthetic notice accompanies the semantic title.
draw.rectangle((108*S,62*S,399*S,104*S),fill='#ce4345')
text(data['visible_notice'],126,74,22,cream,'bold',tracking=1.4)
text('ECONOMY / ANNUAL SERIES',1480,77,19,muted,align='right',tracking=1.4)

title=data['title'].removeprefix('SYNTHETIC ')
first, second = title.split(' to ',1)
text(first,106,129,164,cream,'display')
text('to '+second,106,263,164,cream,'display')

# Legend and scope are positioned as a compact editorial block.
line([(1040,167),(1040,365)],'#45627b',1)
text(f'{int(years[0])}–{int(years[-1])}',1082,169,48,cream,'display')
text(f'{len(years)} annual observations',1084,230,24,muted)
line([(1085,294),(1140,294)],red,4.5)
text(f'Historical  {int(years[0])}–{forecast-1}',1158,283,24,cream)
dashed([(1085,344),(1140,344)],red,4.5,12,8)
text(f'Forecast  {forecast}–{int(years[-1])}',1158,333,24,cream)
text(series['label'],148,453,28,cream,'bold')
text(series['unit'].capitalize(),148,494,22,muted)

# The shared 0–130 scale. Grids are behind the flag, so they cannot break
# its uninterrupted textile object identity.
for tick in [0,20,40,60,80,100,120,130]:
    py=y(tick)
    col='#4b6c87' if tick==100 else '#304f6a'
    line([(L,py),(R,py)],col,1.3 if tick==100 else 0.85)
    text(f'{tick}%',L-24,py-10,22,muted,'regular','right')

# One closed annual polygon; the model-generated textile is visible only
# within it. No smoothing, minimum-height adjustment, data warp, or stacked data.
mask=Image.new('L',(W*S,H*S),0)
md=ImageDraw.Draw(mask)
md.polygon(points_px+[(round(R*S),round(B*S)),(round(L*S),round(B*S))],fill=255)
material=Image.open(ROOT/'assets/flag_material.png').convert('RGB')
material=material.resize((round((R-L)*S),round((B-T)*S)),Image.Resampling.LANCZOS)
flag_layer=Image.new('RGB',canvas.size,navy)
flag_layer.paste(material,(round(L*S),round(T*S)))
canvas.paste(flag_layer,(0,0),mask)
draw=ImageDraw.Draw(canvas)

# Forecast is a continuation of the same area and flag. A restrained dark
# veil identifies the forecast side without changing quantitative geometry.
forecast_mask=mask.copy()
fm=ImageDraw.Draw(forecast_mask)
fm.rectangle((0,0,round(x(forecast)*S)-1,H*S),fill=0)
forecast_mask=forecast_mask.point(lambda a:round(a*0.16))
canvas.paste(Image.new('RGB',canvas.size,'#123557'),(0,0),forecast_mask)
draw=ImageDraw.Draw(canvas)

# Exact historical vertices and a dashed connection to the first forecast.
line(points[:cut],red,4.3)
dashed(points[cut-1:],red,5.2,13,10)
line([(L,B),(R,B)],'#adc0cf',1.5)
# The boundary is attached to the correct year, distinct from the data line.
dashed([(x(forecast),672),(x(forecast),B+38)],'#abc3d5',1.5,7,8)
for yr in range(int(years[0]),int(years[-1])+1):
    xx=x(yr)
    if yr%5==0 or yr==int(years[-1]):
        line([(xx,B),(xx,B+8 if yr%20 else B+13)],'#9fb2c4',1)
for yr in [1900,1920,1940,1960,1980,2000,2020,2035]:
    if years[0]<=yr<=years[-1]:
        text(str(yr),x(yr),B+27,25,cream,align='center')
text(str(forecast),x(forecast),B+70,21,muted,align='center')
text('YEAR',L,B+78,18,muted,tracking=2)

# Data-bound key labels. The peak is computed over historical observations.
hist_max=int(np.argmax(values[:cut]))
start=0
last_hist=cut-1
last=len(values)-1
highlight_indices=[start,hist_max,last_hist,last]

def point_value(i):
    return f'{values[i]:.1f}%'

# Starting value.
text(data['categories'][start],L,1603,27,cream,'bold')
text(point_value(start),L,1641,62,red,'display')
line([(L,1691),(L,points[start][1]-8)],'#c8d0d3',1.4)
dot(points[start],5.1)
# Historical high.
pp=points[hist_max]
text(data['categories'][hist_max],pp[0],627,28,cream,'bold','center')
text(point_value(hist_max),pp[0],666,67,red,'display','center')
line([(pp[0],730),(pp[0],pp[1]-9)],'#c8d0d3',1.4)
dot(pp,5.3)
# Last historical year, set away from the narrow forecast strip.
pp=points[last_hist]
text(data['categories'][last_hist],1273,698,28,cream,'bold','right')
text(point_value(last_hist),1273,738,72,red,'display','right')
line([(1285,797),(1338,797),(pp[0]-5,pp[1]-8)],'#c8d0d3',1.4)
dot(pp,5.3)
# Forecast endpoint.
pp=points[last]
text(data['categories'][last]+' FORECAST',R,510,25,cream,'bold','right')
text(point_value(last),R,552,76,red,'display','right')
line([(R,622),(R,pp[1]-8)],'#c8d0d3',1.4)
dot(pp,6.2)
# First forecast observation is hollow. Its quantitative value is unmodified.
dot(points[cut],4.5,navy,red,2)

# Explanation occupies the natural empty region above the data valley.
text('READING THE MEASURE',824,1012,19,muted,tracking=1.7)
line([(824,1050),(872,1050)],red,3)
text('100% means debt equals',824,1073,30,cream)
text('one year of synthetic output.',824,1114,30,cream)
text('The upper edge tracks the ratio;',824,1180,23,muted)
text('the flag pattern is illustrative.',824,1214,23,muted)

# Provenance does not imply an official real-world statistical source.
line([(108,1892),(1490,1892)],'#45627b',1)
text('SOURCE',108,1920,18,muted,'bold',tracking=1.5)
text('Supplied synthetic data · 1900–2035',216,1918,23,cream)
text('Not official statistics or projections.',1490,1918,22,muted,'italic','right')

# Artifacts document the rendered geometry, not independent evaluation.
artifacts=ROOT/'artifacts'
artifacts.mkdir(exist_ok=True)
mask.resize((W,H),Image.Resampling.LANCZOS).save(artifacts/'area_mask.png')
geometry={
 'canvas':[W,H], 'plot_bbox':[L,T,R,B],
 'axis':data['value_axis'], 'forecast_boundary':forecast,
 'historical_point_count':cut,'forecast_point_count':len(years)-cut,
 'x_pixels_per_year':(R-L)/(years[-1]-years[0]),
 'y_pixels_per_percentage_point':(B-T)/(vmax-vmin),
 'vertices':[{'year':int(t),'value':float(v),'x':p[0],'y':p[1],'phase':'historical' if i<cut else 'forecast'} for i,(t,v,p) in enumerate(zip(years,values,points))],
 'data_labels':[{'year':int(years[i]),'value':float(values[i]),'text':point_value(i)} for i in highlight_indices],
 'data_sha256':hashlib.sha256((ROOT/'data.json').read_bytes()).hexdigest(),
 'text_boxes':text_records
}
(artifacts/'geometry.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
metadata=PngImagePlugin.PngInfo()
metadata.add_text('Title',data['title'])
metadata.add_text('Description','Synthetic annual Debt / Output (%); 1900–2024 historical; 2025–2035 forecast. All 136 supplied vertices are used.')
metadata.add_text('Data SHA256',geometry['data_sha256'])
canvas.resize((W,H),Image.Resampling.LANCZOS).save(ROOT/'final.png',pnginfo=metadata)
print(json.dumps({'size':[W,H],'points':len(points),'historical_points':cut,'forecast_points':len(points)-cut,'historical_max':geometry['vertices'][hist_max],'output':'final.png'}))
