"""Offline infographic renderer. Run with the supplied capture_render.py wrapper."""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent
load = lambda name: json.loads((ROOT / name).read_text(encoding='utf-8'))
inputs = load('input.json')
brief = load(inputs['brief'])
data = load(inputs['data'])
core = load(inputs['critical_metaphor_contract'])
W, H = brief['width'], brief['height']
assert (W, H) == (1680, 2000)
values = data['series'][0]['values']
categories = data['categories']
assert len(values) == len(categories) == 8
assert math.isclose(sum(values), 100)
assert data['value_axis'] == {'minimum': 0, 'maximum': 100, 'scale': 'linear'}
assert data['synthetic'] is True

S = 2
INK = '#203E50'
MUTED = '#617783'
PAPER = '#F7F5EF'
COLORS = ['#89BC94', '#F2C45B', '#B64164', '#A992B4',
          '#32A9BC', '#ED8855', '#355E75', '#B7C7CC']
FONTS = ROOT / 'assets/fonts'
font_cache = {}
def font(size, bold=False):
    key = (size, bold)
    if key not in font_cache:
        font_cache[key] = ImageFont.truetype(str(FONTS / ('Sans-Bold.otf' if bold else 'Sans-Regular.otf')), round(size*S))
    return font_cache[key]

# A gentle blue ambient light behind the transparent glass.
y, x = np.mgrid[0:H, 0:W]
ambient = np.exp(-(((x-1150)/530)**2 + ((y-1170)/910)**2)*1.5)
base = np.array([247,245,239], float)
glow = np.array([228,240,241], float)
rgb = np.clip(base + ambient[...,None]*(glow-base),0,255).astype('uint8')
im = Image.fromarray(rgb).convert('RGBA').resize((W*S,H*S), Image.Resampling.BICUBIC)
d = ImageDraw.Draw(im)
text_boxes = []
def text(x, y, txt, size, fill=INK, bold=False, anchor='la', track=0, record=True):
    f = font(size,bold)
    if track:
        xx = x*S
        for ch in txt:
            d.text((xx,y*S),ch,font=f,fill=fill,anchor=anchor)
            xx += d.textlength(ch,font=f)+track*S
        box = (x*S, y*S, xx, (y+size*1.2)*S)
    else:
        d.text((x*S,y*S),txt,font=f,fill=fill,anchor=anchor)
        box = d.textbbox((x*S,y*S),txt,font=f,anchor=anchor)
    if record:
        text_boxes.append({'text':txt,'bbox_px':[round(v/S,2) for v in box]})

def line(points, fill, width=2):
    d.line([(round(a*S),round(b*S)) for a,b in points], fill=fill,width=round(width*S),joint='curve')
def paste_asset(name, box):
    asset = Image.open(ROOT / f'assets/{name}.png').convert('RGBA')
    ax, ay, aw, ah = box
    asset = asset.resize((round(aw*S),round(ah*S)),Image.Resampling.LANCZOS)
    im.alpha_composite(asset,(round(ax*S),round(ay*S)))

# Header: the notice remains separate and prominent at full reading size.
d.rounded_rectangle((96*S,83*S,446*S,134*S),radius=25*S,fill='#DCEBE9')
text(121,95,data['visible_notice'],26,fill='#296A72',bold=True,track=2)
text(95,163,'Indoor household',90,bold=True)
text(95,256,'water use',90,bold=True)
text(100,387,'Synthetic share of indoor household water use (%)',31,fill=MUTED)
line([(100,455),(874,455)],'#CBD5D4',2)
text(100,515,'HOUSEHOLD USE',23,bold=True,fill=MUTED,track=2.4)
text(651,515,'SHARE',23,bold=True,fill=MUTED,anchor='ra',track=0)

# The generated faucet pours toward the open mouth, not toward a side icon.
paste_asset('faucet',(1005,153,575,421))
paste_asset('drop_small',(1048,591,16,25))
paste_asset('drop_medium',(1042,628,25,39))

# Ground the glass with a soft contact shadow, entirely outside the data field.
shadow=Image.new('RGBA',im.size,(0,0,0,0))
sd=ImageDraw.Draw(shadow)
sd.ellipse((865*S,1790*S,1400*S,1828*S),fill=(35,77,93,27))
shadow=shadow.filter(ImageFilter.GaussianBlur(15*S))
im.alpha_composite(shadow)
d=ImageDraw.Draw(im)

# One mapping controls the liquid, generated glass transform and leader anchors.
TOP, BOTTOM = 700.0, 1700.0
HEIGHT = BOTTOM-TOP
GX, GS = 786.0, 1.25
def inner_edges(y):
    t = (y-TOP)/HEIGHT
    raw_left, raw_right = 146+82*t, 652-82*t
    return GX+(raw_left-121)*GS, GX+(raw_right-121)*GS

# Generated surface light stays above these code-defined quantitative bands.
liquid=Image.new('RGBA',im.size,(0,0,0,0))
ld=ImageDraw.Draw(liquid)
boundaries=[TOP]
geometry=[]
cum=0.0
for i,(cat,val,color) in enumerate(zip(categories,values,COLORS)):
    ya=TOP+cum/100*HEIGHT
    cum+=val
    yb=TOP+cum/100*HEIGHT
    la,ra=inner_edges(ya); lb,rb=inner_edges(yb)
    # Pillow polygons include their bottom edge; the next layer overwrites the
    # common boundary. No gaps, fake minimum sizes, or white separator strokes.
    ld.polygon([(round(la*S),round(ya*S)),(round(ra*S),round(ya*S)),
                (round(rb*S),round(yb*S)),(round(lb*S),round(yb*S))],fill=color)
    geometry.append({'category':cat,'value':val,'unit':'%', 'top_y':ya,
                     'bottom_y':yb,'height_px':yb-ya,'color':color})
    boundaries.append(yb)
im.alpha_composite(liquid)

# Preserve the generated rim, side walls and base. Only the non-data rim and
# base use separate vertical scales; the whole liquid interval has one scale.
shell=Image.open(ROOT / 'assets/glass_shell.png').convert('RGBA')
for (sa,sb),(ta,tb) in zip([(0,46),(46,794),(794,966)],[(650,700),(700,1700),(1700,1810)]):
    strip=shell.crop((0,sa,556,sb)).resize((round(556*GS*S),round((tb-ta)*S)),Image.Resampling.LANCZOS)
    im.alpha_composite(strip,(round(GX*S),round(ta*S)))
d=ImageDraw.Draw(im)

# Labels stay in input order; the bottom leaders fan out for the 1.5% / 2.5%
# layers without changing those layers' heights.
label_y=[810,1010,1200,1385,1535,1637,1719,1787]
for i,(g,ly) in enumerate(zip(geometry,label_y)):
    mid=(g['top_y']+g['bottom_y'])/2
    edge=inner_edges(mid)[0]
    tip=edge+59
    points=[(693,ly),(754,ly),(842,mid),(tip,mid)]
    line(points,'#637A86',2)
    d.ellipse(((tip-4)*S,(mid-4)*S,(tip+4)*S,(mid+4)*S),fill=INK)
    d.rounded_rectangle((99*S,(ly-17)*S,107*S,(ly+17)*S),radius=4*S,fill=COLORS[i])
    text(124,ly,categories[i],37,bold=True,anchor='lm')
    text(651,ly,f'{values[i]:g}%',46,bold=True,anchor='rm')
    g.update(label_center_y=ly,leader_endpoint=[tip,mid])

# Explicit common data height; it is not a volume or area scale.
line([(1512,TOP),(1530,TOP),(1530,BOTTOM),(1512,BOTTOM)],'#8DA5AE',2)
note='100% TOTAL'
nf=font(25,True)
bb=nf.getbbox(note)
rot=Image.new('RGBA',(int(nf.getlength(note))+20*S,44*S),(0,0,0,0))
ImageDraw.Draw(rot).text((10*S,4*S),note,font=nf,fill='#617F8B')
rot=rot.rotate(90,expand=True,resample=Image.Resampling.BICUBIC)
im.alpha_composite(rot,(1543*S,round(1200*S-rot.height/2)))
d=ImageDraw.Draw(im)

# The footer states the sole quantitative encoding and the synthetic scope.
line([(100,1858),(1580,1858)],'#C7D2D1',2)
text(100,1888,'READING THE GLASS',22,bold=True,fill=MUTED,track=1.3)
text(487,1883,'Layer height = share of total indoor water use.',28,fill=INK)
text(487,1921,'Glass width, volume and droplets do not encode values.',25,fill=MUTED)
text(100,1961,'Synthetic illustration · not measured household data.',22,fill=MUTED)

final=im.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
final.save(ROOT / 'final.png',optimize=True)
(ROOT/'artifacts').mkdir(exist_ok=True)
(ROOT/'artifacts/render_geometry.json').write_text(json.dumps({
    'canvas':[W,H], 'data_field':{'top':TOP,'bottom':BOTTOM,'height':HEIGHT},
    'layers':geometry, 'boundaries_y':boundaries,
    'glass_mapping':{'raw_x_origin':121,'target_x_origin':GX,'x_scale':GS,
                     'raw_y_breaks':[29,75,823,995],'target_y_breaks':[650,700,1700,1810]},
    'text_boxes':text_boxes,
    'input_files':[inputs['brief'],inputs['data'],inputs['critical_metaphor_contract'],inputs['reference']],
},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Rendered final.png: 1680 × 2000; all 8 shares bound to a 1000 px liquid height.')
