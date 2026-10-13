#!/usr/bin/env python3
"""Offline poster renderer. Run only through capture_render.py during the study."""
from pathlib import Path
import json
import math
from PIL import Image, ImageDraw, ImageFont
import numpy as np

ROOT = Path(__file__).resolve().parent
load = lambda name: json.loads((ROOT/name).read_text(encoding='utf-8'))
inputs=load('input.json')
brief=load(inputs['brief'])
data=load(inputs['data'])
core=load(inputs['critical_metaphor_contract'])
assert (ROOT/inputs['reference']).is_file(), 'Keep the frozen visual reference available.'
assert data['synthetic'] and data['value_axis']['scale']=='linear'
assert core['topic_zh']==data['topic_zh']
W,H = brief['width'],brief['height']
assert (W,H)==(1600,2000)
S=2
INK=(44,35,112,255)
PAPER=(255,249,226,255)
FONT_DIR=ROOT/'assets/fonts'
FONTS={'display':'display.otf','body':'body.ttf','bold':'bold.ttf'}
fonts={}
def font(kind,size):
    key=(kind,size)
    if key not in fonts:
        fonts[key]=ImageFont.truetype(str(FONT_DIR/FONTS[kind]),round(size*S))
    return fonts[key]

# Deterministic amber-to-orange atmosphere, close to the reference's visual language.
y,x=np.mgrid[0:H,0:W].astype(np.float32)
u=x/(W-1);v=y/(H-1)
glow=np.exp(-((u-.08)/.78)**2-((v-.88)/.77)**2)
base=np.array([255,94,8],dtype=np.float32)
light=np.array([255,166,46],dtype=np.float32)
background=np.uint8(np.clip(base+(light-base)*glow[:,:,None],0,255))
im=Image.fromarray(background,'RGB').convert('RGBA').resize((W*S,H*S),Image.Resampling.BICUBIC)
# The generated connected athlete is the central visual scene, directly under the chart.
player=Image.open(ROOT/'assets/player_duotone.png').convert('RGBA')
player_h=1760
player_w=round(player.width*player_h/player.height)
player_x=round(860-player_w/2)
player_y=32
player=player.resize((player_w*S,player_h*S),Image.Resampling.LANCZOS)
im.alpha_composite(player,(player_x*S,player_y*S))
d=ImageDraw.Draw(im)
text_records=[]

def text(txt,x,y,size=28,kind='body',fill=INK,anchor='lt',stroke=0,stroke_fill=None,role='text'):
    f=font(kind,size)
    pos=(round(x*S),round(y*S))
    kwargs=dict(font=f,fill=fill,anchor=anchor,stroke_width=round(stroke*S))
    if stroke_fill is not None: kwargs['stroke_fill']=stroke_fill
    d.text(pos,str(txt),**kwargs)
    bb=d.textbbox(pos,str(txt),font=f,anchor=anchor,stroke_width=round(stroke*S))
    text_records.append({'text':str(txt),'role':role,'bbox':[round(z/S,2) for z in bb]})

def line(points,fill,width):
    d.line([(round(a*S),round(b*S)) for a,b in points],fill=fill,width=round(width*S),joint='curve')

def circle(x,y,r,fill):
    d.ellipse((round((x-r)*S),round((y-r)*S),round((x+r)*S),round((y+r)*S)),fill=fill)

# Header and topic. Type is code-rendered; the raster contains no text.
d.rounded_rectangle((90*S,70*S,426*S,122*S),radius=6*S,fill=INK)
text(data['visible_notice'],112,83,26,'bold',PAPER,role='synthetic_notice')
text('2010—2024  /  ANNUAL SERIES',1506,85,25,'bold',anchor='rt',role='time_range')
text("WOMEN'S",86,177,118,'display',PAPER,role='title')
text('BASKETBALL',83,294,176,'display',PAPER,role='title')
text('VIEWERS',87,456,166,'display',INK,role='title')
text(data['series'][0]['label'],92,640,32,'bold',role='metric')
text(data['series'][0]['unit'],92,685,27,'body',role='unit')

# One explicit affine coordinate transform, shared by every data mark.
plot={'left':150.0,'right':1450.0,'top':760.0,'bottom':1760.0}
times=np.array(data['time_points'],dtype=float)
values=np.array(data['series'][0]['values'],dtype=float)
categories=data['categories']
assert len(categories)==len(values)==len(times)
vmin=float(data['value_axis']['minimum']);vmax=float(data['value_axis']['maximum'])
tmin=float(times[0]);tmax=float(times[-1])
def to_xy(t,value):
    return (plot['left']+(t-tmin)/(tmax-tmin)*(plot['right']-plot['left']),
            plot['bottom']-(value-vmin)/(vmax-vmin)*(plot['bottom']-plot['top']))
points=[to_xy(t,z) for t,z in zip(times,values)]

# Grid uses a separate translucent layer; it does not obscure the athlete.
grid=Image.new('RGBA',im.size,(0,0,0,0));gd=ImageDraw.Draw(grid)
for value in (0,500,1000,1500):
    yy=to_xy(tmin,value)[1]
    end=1160 if value==1500 else plot['right']
    gd.line((plot['left']*S,round(yy*S),end*S,round(yy*S)),fill=(*INK[:3],55 if value else 220),width=(2 if value else 3)*S)
im.alpha_composite(grid)
d=ImageDraw.Draw(im)
for value in (0,500,1000,1500):
    yy=to_xy(tmin,value)[1]
    text(f'{value:,}',120,yy,25,'bold',anchor='rm',role='value_axis_tick')

# Subtle warm shadow followed by exact unsmoothed annual polyline.
# Stroke and marker sizes are presentation attributes, not extra encoded quantities.
shadow_points=[(px+3,py+7) for px,py in points]
line(shadow_points,(216,80,19,255),14)
line(points,PAPER,10)
for i,(px,py) in enumerate(points):
    circle(px,py,8 if i<len(points)-1 else 22,INK)
    circle(px,py,3.8 if i<len(points)-1 else 7.5,PAPER)

# Every year is a tick and every supplied numerical value is displayed.
for i,((px,py),cat,value) in enumerate(zip(points,categories,values)):
    line([(px,plot['bottom']),(px,plot['bottom']+16)],INK,2)
    text(cat,px,1798,25,'bold',anchor='mt',role='year')
    if i==len(points)-1:
        continue
    above = i%2==0
    label_y=py-33 if above else py+22
    text(f'{int(value):,}',px,label_y,27,'bold',anchor='mb' if above else 'mt',
         stroke=1.5,stroke_fill=(255,157,41,255),role='value_label')
# A directly bound endpoint callout; no historical claim or inherited percentage.
last=points[-1]
text(f'{int(values[-1]):,}',1490,688,106,'display',INK,anchor='rt',role='endpoint_value')
text(categories[-1],1486,795,28,'bold',INK,anchor='rt',role='endpoint_year')
text('YEAR',1504,1853,19,'bold',INK,anchor='rt',role='category_axis_title')

# Minimal provenance, clearly distinct from the statistical series.
line([(90,1892),(1510,1892)],INK,2)
text('SYNTHETIC DATA  /  15 annual observations',90,1921,24,'bold',role='footer_notice')
text('0–1,500 · LINEAR SCALE',1510,1921,24,'bold',anchor='rt',role='scale_note')
text('Source: supplied synthetic dataset. Values are not real audience statistics.',90,1960,20,'body',role='provenance')

# Save exact data bindings for reproducibility and post-render pixel checks.
geometry={
    'canvas':[W,H], 'supersampling':S, 'plot':plot,
    'axis':{'minimum':vmin,'maximum':vmax,'scale':'linear','unit':data['series'][0]['unit']},
    'points':[{'category':cat,'time':float(t),'value':float(val),'x':round(px,5),'y':round(py,5)}
              for cat,t,val,(px,py) in zip(categories,times,values,points)],
    'player':{'path':'assets/player_duotone.png','bbox':[player_x,player_y,player_x+player_w,player_y+player_h],
              'all_body_parts_inside_canvas':player_y>=0 and player_y+player_h<=H},
    'colors':{'ink':list(INK[:3]),'line':list(PAPER[:3])},
    'line_width_px':10, 'normal_marker_outer_radius_px':8,
    'labels':text_records,
    'rendered_from':['input.json','brief.json','data.json','core.json','reference.png','assets/player_duotone.png',
                     'assets/fonts/display.otf','assets/fonts/body.ttf','assets/fonts/bold.ttf']
}
(ROOT/'artifacts').mkdir(exist_ok=True)
(ROOT/'artifacts/geometry.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
im.convert('RGB').resize((W,H),Image.Resampling.LANCZOS).save(ROOT/'final.png')
print('Saved final.png (1600 × 2000); 15 data points on one 0–1500 linear scale.')
