"""Offline, data-bound icebreaker infographic. Run: PYTHON render.py.

Complete experiment renders must be launched through capture_render.py.
Generated imagery supplies the hull/deck and floes. Data geometry and all type
are controlled here. The first datum includes the ship, ending at its bow.
"""
from pathlib import Path
import json
import math
import hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
S = 2

def read_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))

inputs = read_json('input.json')
brief = read_json(inputs['brief'])
data = read_json(inputs['data'])
core = read_json(inputs['critical_metaphor_contract'])
assert core['data_carrying_relationship_zh'] == brief['encoding_definition_zh']
assert (ROOT / inputs['reference']).is_file(), 'Keep the frozen reference available.'
series = data['series'][0]
values = series['values']
categories = data['categories']
assert len(categories) == len(values)
assert data['value_axis']['scale'] == 'linear'
W, H = brief['width'], brief['height']
assert (W, H) == (2000, 2000)
assert sum(values) == data['series_totals'][series['id']]

BG = '#f3f8fa'
INK = '#122f3f'
MUTED = '#59717e'
GRID = '#dce7ec'
BLUE = '#64afe6'
im = Image.new('RGBA', (W*S, H*S), BG)
draw = ImageDraw.Draw(im)
font_paths = {False: ROOT/'assets/fonts/NimbusSans-Regular.otf',
              True: ROOT/'assets/fonts/NimbusSans-Bold.otf'}
font_cache = {}
text_records = []

def font(size, bold=False):
    key = (size, bold)
    if key not in font_cache:
        font_cache[key] = ImageFont.truetype(str(font_paths[bold]), round(size*S))
    return font_cache[key]

def text(x, y, message, size, fill=INK, bold=False, align='left', valign='top', tracking=0):
    f = font(size, bold)
    if tracking:
        length = sum(draw.textlength(c, font=f) for c in message) + max(0, len(message)-1)*tracking*S
    else:
        length = draw.textlength(message, font=f)
    box = draw.textbbox((0, 0), message, font=f)
    height = box[3] - box[1]
    xx, yy = x*S, y*S
    if align == 'right': xx -= length
    if align == 'center': xx -= length/2
    if valign == 'middle': yy -= height/2
    if valign == 'bottom': yy -= height
    yy -= box[1]
    start_x = xx
    if tracking:
        for char in message:
            draw.text((xx, yy), char, font=f, fill=fill)
            xx += draw.textlength(char, font=f) + tracking*S
    else:
        draw.text((xx, yy), message, font=f, fill=fill)
    text_records.append({'text': message, 'ink_box': [round(start_x/S,2), round((yy+box[1])/S,2),
        round((start_x+length)/S,2), round((yy+box[3])/S,2)]})

def line(coords, fill, width=1):
    draw.line([round(v*S) for v in coords], fill=fill, width=max(1,round(width*S)))

def rect(coords, fill, outline=None, width=1):
    draw.rectangle([round(v*S) for v in coords], fill=fill, outline=outline, width=round(width*S))

# Editorial header. The synthetic notice is visible and separate from the topic.
draw.rounded_rectangle((120*S,86*S,458*S,141*S),radius=8*S,fill='#d9eef9')
text(141, 102, data['visible_notice'], 27, '#24698d', True, tracking=1.5)
title = data['title'].removeprefix('SYNTHETIC ')
text(120, 197, title, 124, bold=True)
text(125, 350, series['label'], 40, MUTED)
line((1575,194,1575,322),'#c7dce6',2)
text(1855,193,str(sum(values)),98,bold=True,align='right')
text(1854,301,'SHIPS TOTAL',23,MUTED,True,align='right',tracking=1.8)

# A single common mapping is used by every row and the complete ship + lane union.
x0, x1 = 430.0, 1830.0
vmin, vmax = data['value_axis']['minimum'], data['value_axis']['maximum']
scale = (x1-x0)/(vmax-vmin)
def X(value): return x0 + (value-vmin)*scale
ys = np.linspace(650,1665,len(values)).tolist()
bar_h = 76
axis_y, chart_bottom = 550, 1730
text(120,456,'FLEET',25,MUTED,True,tracking=2)
text(x0,456,'NUMBER OF SHIPS',25,MUTED,True,tracking=1.8)
for value in range(int(vmin),int(vmax)+1,10):
    xx=X(value)
    text(xx,505,str(value),27,MUTED,align='center')
    line((xx,axis_y,xx,chart_bottom),'#c1d7e2' if value==0 else GRID,1.8 if value==0 else 1)
    line((xx,axis_y-6,xx,axis_y), '#8cabbc',1.5)
line((x0,axis_y,x1,axis_y),'#adc9d7',1.4)

# Water lanes have exact half-open raster bounds; no minimum length is imposed.
def water_lane(left,right,cy):
    xx0,xx1=round(left*S),round(right*S)
    yy0,yy1=round((cy-bar_h/2)*S),round((cy+bar_h/2)*S)
    ww,hh=xx1-xx0,yy1-yy0
    if ww<=0: return
    top=np.array([94,171,229],float)
    bottom=np.array([108,184,237],float)
    t=np.linspace(0,1,hh)[:,None,None]
    rgb=np.repeat((top[None,None,:]*(1-t)+bottom[None,None,:]*t),ww,axis=1)
    layer=Image.fromarray(np.uint8(np.round(rgb)),'RGB').convert('RGBA')
    im.alpha_composite(layer,(xx0,yy0))
    # Two subtle inset current streaks, contained by the real data rectangle.
    if ww>150*S:
        ld=ImageDraw.Draw(im)
        for frac,lfrac,rfrac in [(0.24,.03,.44),(0.73,.23,.89)]:
            yy=yy0+round(hh*frac)
            ld.line((xx0+round(ww*lfrac),yy,xx0+round(ww*rfrac),yy),fill=(153,209,244,155),width=1*S)

asset_placements=[]
def floe(name,cx,cy,width,angle=0):
    ice=Image.open(ROOT/'assets'/f'{name}.png').convert('RGBA')
    ice=ice.resize((round(width*S),round(width*S*ice.height/ice.width)),Image.Resampling.LANCZOS)
    if angle:
        ice=ice.rotate(angle,Image.Resampling.BICUBIC,expand=True)
    pos=(round(cx*S-ice.width/2),round(cy*S-ice.height/2))
    im.alpha_composite(ice,pos)
    asset_placements.append({'asset':f'assets/{name}.png','box':[pos[0]/S,pos[1]/S,(pos[0]+ice.width)/S,(pos[1]+ice.height)/S], 'role':'nonquantitative ice'})

ship=Image.open(ROOT/'assets/ship.png').convert('RGBA')
ship_width=min(550.0,(X(values[0])-x0)*.48)
ship=ship.resize((round(ship_width*S),round(ship_width*S*ship.height/ship.width)),Image.Resampling.LANCZOS)
sa=np.asarray(ship)[:,:,3]
solid=sa>=96
sy,sx=np.nonzero(solid)
solid_right=int(sx.max())+1
solid_left=int(sx.min())
bow_y=float(np.median(np.nonzero(solid[:,max(0,solid_right-3):solid_right])[0]))
ship_x=round(X(values[0])*S)-solid_right
ship_y=round(ys[0]*S-bow_y)
stern_x=(ship_x+solid_left)/S
channel_end=stern_x+15
water_lane(x0,channel_end,ys[0])

# Detached ice is under/alongside the vessel; nothing extends its counted bow.
cy=ys[0]
for args in [
 ('ice_01',stern_x+88,cy-33,140,12),
 ('ice_02',stern_x+245,cy-35,154,-12),
 ('ice_03',stern_x+383,cy-24,86,-28),
 ('ice_04',stern_x+401,cy+44,78,18),
 ('ice_05',stern_x+231,cy+54,151,-9),
 ('ice_06',stern_x+46,cy+42,64,24),
 ('ice_06',stern_x-17,cy-16,23,19),
 ('ice_04',stern_x-40,cy+13,17,-30),
 ('ice_02',stern_x-60,cy-17,14,40),
]: floe(*args)
im.alpha_composite(ship,(ship_x,ship_y))
asset_placements.append({'asset':'assets/ship.png','box':[ship_x/S,ship_y/S,(ship_x+ship.width)/S,(ship_y+ship.height)/S], 'role':'data-bearing ship; bow is endpoint'})

row_records=[]
for i,(category,value,cy) in enumerate(zip(categories,values,ys)):
    end=X(value)
    if i>0: water_lane(x0,end,cy)
    text(120,cy,category,45,INK,False,valign='middle')
    text(end+22,cy,str(value),50 if i==0 else 47,INK,True,valign='middle')
    row_records.append({'category':category,'value':value,'unit':series['unit'],
        'center_y':cy,'data_left':x0,'data_right':end,'expected_span':end-x0,
        'visible_bar_height':bar_h,'endpoint':'bow' if i==0 else 'bar right edge'})

# Quiet, explicit reading guidance without changing any numeric geometry.
line((120,1794,1880,1794),'#c6dbe5',1.5)
text(120,1832,'READING THE LANES',23,MUTED,True,tracking=1.7)
text(120,1876,'The first lane ends at the bow; the ship is included in its total length.',29,INK)
text(120,1923,'Floating ice and deck details do not encode values. Fleet labels are synthetic.',26,MUTED)

# Save one final image, with numeric labels provided entirely by executable code.
output=im.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
output.save(ROOT/'final.png',format='PNG',optimize=True)
artifacts=ROOT/'artifacts';artifacts.mkdir(exist_ok=True)
geometry={'canvas':[W,H],'axis':{'min':vmin,'max':vmax,'scale':'linear','x0':x0,'x1':x1,'px_per_ship':scale},
          'rows':row_records,'asset_placements':asset_placements,
          'ship_transform':{'solid_alpha_threshold':96,'solid_source_right_exclusive':solid_right,
              'paste_xy':[ship_x/S,ship_y/S],'bow_center_y':(ship_y+bow_y)/S,
              'solid_bow_right':(ship_x+solid_right)/S,'stern_x':stern_x,'channel_end':channel_end,
              'channel_hull_overlap_px':channel_end-stern_x},
          'labels':text_records,
          'frozen_input_hashes':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in [inputs['brief'],inputs['data'],inputs['critical_metaphor_contract'],inputs['reference'],'input.json']}}
(artifacts/'render_geometry.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'output':'final.png','canvas':[W,H],'rows':len(values),'total':sum(values),
                  'ship_bow_x':geometry['ship_transform']['solid_bow_right'],'px_per_ship':scale},ensure_ascii=False))
