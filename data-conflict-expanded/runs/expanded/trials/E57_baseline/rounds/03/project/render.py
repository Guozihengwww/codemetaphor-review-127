"""Offline, data-bound bank-facade infographic. Run through capture_render.py.

All data-bearing rectangles share one linear transform. Generated pixels supply
the actual architectural facade and fluted material, never the quantitative shape.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent
def load(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8'))

inputs = load('input.json')
brief = load(inputs['brief'])
data = load(inputs['data'])
core = load(inputs['critical_metaphor_contract'])
assert (ROOT/inputs['reference']).is_file()  # Kept as design provenance; pixels are not copied.
W, H = brief['width'], brief['height']
assert (W,H) == (2160,1876)
assert data['synthetic'] and data['value_axis']['scale'] == 'linear'
series = data['series'][0]
cats, values = data['categories'], series['values']
assert len(cats) == len(values) == 10 and len(data['series']) == 1
assert core['semantic_roles_zh'] == [f"{series['label']} ({series['unit']})"]
assert abs(sum(values)-data['series_totals'][series['id']]) < 1e-8

# Supersampling preserves the fractional height of even the smallest mark.
S = 2
def px(v): return int(round(v*S))
TOP, BASE, FULL_H = 806.0, 1516.0, 710.0
X_CENTERS = np.linspace(360,1800,len(cats))
BAR_W = 112.0
VMIN, VMAX = data['value_axis']['minimum'], data['value_axis']['maximum']
def y_of(v): return BASE - (v-VMIN)/(VMAX-VMIN)*FULL_H

# A fixed, quiet paper-like background; not a quantitative layer.
yy = np.linspace(0,1,H)[:,None,None]
c0, c1 = np.array([197,219,224.]), np.array([233,241,235.])
bg = np.broadcast_to(c0*(1-yy)+c1*yy, (H,W,3)).copy()
noise = np.random.default_rng(57).normal(0,.46,(H,W,1))
bg = np.clip(bg+noise,0,255).astype('uint8')
im = Image.fromarray(bg).convert('RGBA').resize((px(W),px(H)), Image.Resampling.LANCZOS)

def asset(path, box, opacity=1):
    x,y,w,h = box
    a = Image.open(ROOT/path).convert('RGBA').resize((px(w),px(h)),Image.Resampling.LANCZOS)
    if opacity != 1:
        a.putalpha(a.getchannel('A').point(lambda q: round(q*opacity)))
    im.alpha_composite(a,(px(x),px(y)))

asset('assets/vault.png',(1374,40,806,772),.19)
d = ImageDraw.Draw(im)
INK = '#153f4b'
MUTED = '#446570'
BRASS = '#ab9261'
FONT_DIR = ROOT/'assets/fonts'
def font(size,bold=False):
    return ImageFont.truetype(str(FONT_DIR/('NimbusSans-Bold.otf' if bold else 'NimbusSans-Regular.otf')),px(size))
text_boxes=[]
def txt(text,x,y,size=32,fill=INK,bold=False,anchor='lt',role='annotation'):
    f=font(size,bold)
    bbox=d.textbbox((px(x),px(y)),text,font=f,anchor=anchor)
    d.text((px(x),px(y)),text,font=f,fill=fill,anchor=anchor)
    text_boxes.append({'text':text,'role':role,'bbox_px':[round(v/S,2) for v in bbox]})
def line(coords, fill, width=1):
    d.line([px(q) for q in coords],fill=fill,width=px(width))

# Header hierarchy retains the topic, synthetic status, series role and unit.
d.rounded_rectangle((px(128),px(68),px(460),px(117)),radius=px(6),fill=INK)
txt(data['visible_notice'],294,80,27,'#f4f7ef',True,'mt','synthetic_notice')
txt('Central bank assets',128,145,104,INK,True,role='title')
txt(f"{series['label']}  /  {series['unit']}",132,270,37,MUTED,False,role='metric_unit')
txt('10 fictional central banks',134,326,28,MUTED,False,role='semantic_parent')

txt('TOTAL ILLUSTRATIVE ASSETS',2028,100,22,MUTED,True,'rt','total_label')
txt(f"{data['series_totals'][series['id']]:.2f}",2028,143,70,INK,True,'rt','total_value')
txt(series['unit'],2028,225,26,MUTED,False,'rt','total_unit')

# Guides are behind the supporting shafts, so the filled net heights remain clear.
for tick in [0,2,4,6,VMAX]:
    y=y_of(tick)
    line((277,y,1885,y),'#b9ced0',1)
    line((265,y,277,y),'#839fa5',1.3)
    # The roof occupies the maximum-level label area. Its value is already
    # printed in Bank A and the explicit scale definition below the facade.
    if tick != VMAX:
        txt(str(tick),248,y,25,MUTED,False,'rm','axis_tick')

# Full-height pale supports carry the roof; only the dark portions encode assets.
SUPPORT_TOP = TOP - 18
support = Image.open(ROOT/'assets/support_texture.png').convert('RGBA').resize((px(BAR_W),px(BASE-SUPPORT_TOP)),Image.Resampling.LANCZOS)
for x in X_CENTERS:
    im.alpha_composite(support,(px(x-BAR_W/2),px(SUPPORT_TOP)))

asset('assets/pediment.png',(90,398,1980,410))
d=ImageDraw.Draw(im)

# Synthetic round bank badges occupy the generated entablature's blank face.
for x,cat in zip(X_CENTERS,cats):
    cy,r=744,24
    d.ellipse((px(x-r),px(cy-r),px(x+r),px(cy+r)),fill='#eaf0ea',outline=BRASS,width=px(1.8))
    txt(cat.split()[-1],x,cy,28,INK,True,'mm','bank_badge')

# Clip a full-height generated shaft texture to the single data transform.
texture = Image.open(ROOT/'assets/asset_texture.png').convert('RGBA').resize((px(BAR_W),px(FULL_H)),Image.Resampling.LANCZOS)
geometry=[]
for x,cat,v in zip(X_CENTERS,cats,values):
    top=y_of(v)
    left=px(x-BAR_W/2)
    top_i=px(top)
    bottom_i=px(BASE)
    h=bottom_i-top_i
    a=texture.crop((0,px(FULL_H)-h,px(BAR_W),px(FULL_H)))
    # Material occupies the entire rectangle; no minimum data size is imposed.
    im.alpha_composite(a,(left,top_i))
    d=ImageDraw.Draw(im)
    txt(f'{v:.2f}',x,top+17,37,'#f8f8ed',True,'mt','value')
    geometry.append({'category':cat,'value':v,'x_center':float(x),'left':x-BAR_W/2,'right':x+BAR_W/2,
                     'top':top,'baseline':BASE,'net_height':BASE-top,'supersampled_visible_rect':[left,top_i,left+px(BAR_W),bottom_i]})

# Generated steps meet the common zero baseline. The narrow ledge closes any
# diagonal top-surface gap; it is wholly below the data-bearing rectangle.
d=ImageDraw.Draw(im)
d.rectangle((px(282),px(BASE),px(1878),px(BASE+10)),fill='#93aaa9')
asset('assets/plinth.png',(80,BASE,2000,169))
d=ImageDraw.Draw(im)
line((291,BASE,1868,BASE), '#697f81', 1)
for x,cat in zip(X_CENTERS,cats):
    txt(cat,x,1604,30,INK,True,'mt','category')

# Short guide distinguishes architecture from measured height.
line((128,1721,2032,1721),'#9db5b8',1)
d.rectangle((px(132),px(1760),px(156),px(1784)),fill='#145364')
txt('Filled height = illustrative assets',172,1757,28,INK,True,role='encoding_legend')
d.rectangle((px(865),px(1760),px(889),px(1784)),fill='#dce7e4',outline='#9cb3b6',width=px(1))
txt('Pale shafts = non-data supports',905,1757,28,MUTED,False,role='support_legend')
txt('Linear scale: 0–7.60 trillion USD. Roof, foundation and shadows are excluded.',132,1810,27,MUTED,role='scale_definition')
txt('SYNTHETIC • NOT REAL BANK STATISTICS',2032,1811,20,MUTED,True,'rt','synthetic_footnote')

final=im.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
final.save(ROOT/'final.png',compress_level=6)
(ROOT/'artifacts').mkdir(exist_ok=True)
evidence={'canvas':[W,H],'data_transform':{'minimum':VMIN,'maximum':VMAX,'top':TOP,'baseline':BASE,'full_height_px':FULL_H,'scale':'linear','no_minimum_size':True},
          'geometry':geometry,'text_boxes':text_boxes,
          'frozen_input_sha256':{str(p):hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['brief.json','data.json','core.json','input.json','reference.png']}}
(ROOT/'artifacts/render_geometry.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'final':'final.png','size':[W,H],'data_values':values,'height_scale_px_per_trillion':FULL_H/(VMAX-VMIN)},ensure_ascii=False))

