#!/usr/bin/env python3
"""Offline, deterministic rendering of the supplied synthetic bank-asset data.
Run complete renders only through the supplied capture_render.py wrapper.
"""
from pathlib import Path
import json
from decimal import Decimal
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / 'data.json').read_text(encoding='utf-8'))
BRIEF = json.loads((ROOT / 'brief.json').read_text(encoding='utf-8'))
CORE = json.loads((ROOT / 'core.json').read_text(encoding='utf-8'))
INPUT = json.loads((ROOT / 'input.json').read_text(encoding='utf-8'))
# Retain and verify all frozen dependencies without using reference pixels as output.
assert INPUT['reference'] == 'reference.png' and (ROOT / INPUT['reference']).is_file()
assert CORE['visible_title'] == DATA['title']
assert DATA['synthetic'] and DATA['value_axis']['scale'] == 'linear'
assert len(DATA['series']) == 1 and DATA['series'][0]['id'] == 'assets'
W, H = BRIEF['width'], BRIEF['height']
S = 3
A = ROOT / 'assets'
OUT = ROOT / 'artifacts'
OUT.mkdir(exist_ok=True)
INK = '#15344B'
MUTED = '#526F80'
COPPER = '#B74C2F'

def font(size, face='sans'):
    suffix = 'ttf' if face == 'mono' else 'otf'
    return ImageFont.truetype(str(A / 'fonts' / (face + '.' + suffix)), round(size * S))

def box(coords):
    return tuple(round(v * S) for v in coords)

def text(x, y, label, size, color=INK, face='sans', anchor='lt'):
    dr.text((round(x*S), round(y*S)), label, font=font(size, face), fill=color, anchor=anchor,
            stroke_width=0)

def tracked(x, y, label, size, spacing=2.2, color=MUTED, centered=False):
    ff = font(size, 'sans_bold')
    widths = [dr.textlength(ch, font=ff) / S for ch in label]
    if centered:
        x -= (sum(widths) + spacing * (len(label)-1)) / 2
    for ch, width in zip(label, widths):
        dr.text((round(x*S), round(y*S)), ch, font=ff, fill=color, anchor='lt')
        x += width + spacing

def paste_rgba(im, x, y, w, h, opacity=1.0):
    im = im.convert('RGBA').resize((round(w*S), round(h*S)), Image.Resampling.LANCZOS)
    if opacity != 1:
        im.putalpha(im.getchannel('A').point(lambda v: round(v*opacity)))
    canvas.paste(im, (round(x*S), round(y*S)), im)

def rect(coords, fill, outline=None, width=1):
    dr.rectangle(box(coords), fill=fill, outline=outline, width=round(width*S))

def line(coords, fill, width=1):
    dr.line(box(coords), fill=fill, width=max(1, round(width*S)))

# A quiet blue paper ground; seeded texture is deterministic.
y, x = np.mgrid[0:H, 0:W]
grad = (y/H * -3.0 + (x/W) * -1.0)[..., None]
noise = np.random.default_rng(5701).normal(0, 0.42, (H, W, 1))
bg = np.clip(np.array([234, 243, 244])[None,None,:] + grad + noise, 0, 255).astype('uint8')
canvas = Image.fromarray(bg).resize((W*S, H*S), Image.Resampling.BICUBIC)
dr = ImageDraw.Draw(canvas)

# The generated vault remains behind, and partly occluded by, the bank facade.
vault = Image.open(A / 'vault.png')
paste_rgba(vault, 1252, 243, 1050, 1050, opacity=0.135)
dr = ImageDraw.Draw(canvas)

# Header and exact semantics.
dr.rounded_rectangle(box((105, 66, 399, 111)), radius=8*S, fill=COPPER)
tracked(127, 80, DATA['visible_notice'], 22, spacing=1.8, color='white')
text(108, 153, DATA['title'].removeprefix('SYNTHETIC '), 116, face='sans_bold')
text(112, 289, DATA['series'][0]['label'] + '  /  ' + DATA['series'][0]['unit'], 33)
text(2037, 78, f"{len(DATA['categories'])} FICTITIOUS CENTRAL BANKS", 23, color=MUTED, face='sans_bold', anchor='rt')
text(2037, 166, str(DATA['series_totals']['assets']), 55, face='mono', anchor='rt')
text(2037, 233, 'trillion USD in total', 25, color=MUTED, anchor='rt')

CENTERS = np.linspace(302.0, 1858.0, len(DATA['categories']))
BW = 106.0
TOP = 680.0
BASELINE = 1540.0
PLOT_H = BASELINE-TOP
MAXV = Decimal(str(DATA['value_axis']['maximum']))
assert DATA['value_axis']['minimum'] == 0
values = [Decimal(str(v)) for v in DATA['series'][0]['values']]
assert all(0 <= v <= MAXV for v in values)

# Grid lines are drawn behind the full-height building supports.
for tick in (0, 1.9, 3.8, 5.7, 7.6):
    ty = BASELINE - float(Decimal(str(tick))/MAXV) * PLOT_H
    for xx in np.arange(206, 1960, 16):
        line((xx,ty,min(xx+7,1960),ty), '#B7CAD1', 0.8)
    line((162,ty,194,ty), '#8DA7B4', 1.2)
    text(110, ty, '0' if tick == 0 else f'{tick:.2f}', 25, color=MUTED, face='mono', anchor='rm')
line((182,TOP,182,BASELINE), '#A1B8C3', 1)

# Cast shadow belongs to the building, and is entirely below the data baseline.
shadow = Image.new('RGBA', (W*S,H*S))
sd = ImageDraw.Draw(shadow)
sd.ellipse(box((124,1718,2036,1770)), fill=(29,64,82,22))
shadow = shadow.filter(ImageFilter.GaussianBlur(15*S))
canvas.paste(shadow,(0,0),shadow)
dr = ImageDraw.Draw(canvas)

# One generated shaft texture supplies both the non-data support and the data body.
shaft_source = Image.open(A / 'shaft.png').convert('RGB')
shaft = shaft_source.resize((round(BW*S),round(PLOT_H*S)), Image.Resampling.LANCZOS)
sa = np.asarray(shaft, dtype=np.float32)
lum = sa @ np.array([0.2126,0.7152,0.0722], dtype=np.float32)
f = np.clip((lum-80)/165,0,1)[...,None]
pale = (np.array([185,207,216]) + f*np.array([44,31,26])).astype(np.uint8)
dark = (np.array([13,53,80]) + f*np.array([23,44,41])).astype(np.uint8)
pale_im, dark_im = Image.fromarray(pale), Image.fromarray(dark)
geometry = []
# Save small diagnostic color profiles, not an additional chart rendering.
probe_bg, probe_ink = [], []
for idx, (cx, v, cat) in enumerate(zip(CENTERS, values, DATA['categories'])):
    left = round((cx-BW/2)*S)
    top = round(TOP*S)
    canvas.paste(pale_im, (left,top))
    exact_h = float(v/MAXV) * PLOT_H
    exact_top = BASELINE-exact_h
    # Analytic fractional-pixel coverage; no positive value is inflated to a minimum.
    row0 = TOP + np.arange(round(PLOT_H*S))/S
    coverage = np.clip((row0 + 1/S - exact_top)*S,0,1)
    alpha = np.repeat(np.rint(coverage[:,None]*255).astype(np.uint8),round(BW*S),axis=1)
    canvas.paste(dark_im,(left,top),Image.fromarray(alpha))
    geometry.append({'category':cat,'value_text':str(v),'value':float(v),
                     'left_px':left/S,'right_px':(left+BW*S)/S,
                     'top_px':exact_top,'baseline_px':BASELINE,
                     'net_height_px':exact_h,'support_height_px':PLOT_H})
    # Reference color profiles use the same final resampling as final.png.
    p = pale_im.resize((round(BW),round(PLOT_H)),Image.Resampling.LANCZOS)
    q = dark_im.resize((round(BW),round(PLOT_H)),Image.Resampling.LANCZOS)
    probe_bg.append(np.asarray(p)[:,35:71].mean(axis=1))
    probe_ink.append(np.asarray(q)[:,35:71].mean(axis=1))

# Generated pediment and entablature. Stretch only the blank fascia to accept seals.
roof = Image.open(A / 'roof.png').convert('RGBA')
rw, rh = roof.size
# Raw crop starts at y=27: slope and upper moulding end near row 251.
upper = roof.crop((0,0,rw,251))
fascia = roof.crop((0,251,rw,276))
lower = roof.crop((0,276,rw,rh))
paste_rgba(upper,80,350,2000,244)
paste_rgba(fascia,80,594,2000,73)
paste_rgba(lower,80,667,2000,13)
dr = ImageDraw.Draw(canvas)
tracked(1080, 530, 'FICTITIOUS CENTRAL BANKS', 23, 3.5, INK, centered=True)

# Ten neutral circular seals preserve the role of the reference's round emblems.
for cx, cat in zip(CENTERS, DATA['categories']):
    cy, radius = 631, 29
    dr.ellipse(box((cx-radius,cy-radius+2,cx+radius,cy+radius+2)), fill='#55788D')
    dr.ellipse(box((cx-radius,cy-radius,cx+radius,cy+radius)),fill='#EBF3F3',outline='#41677D',width=2*S)
    dr.ellipse(box((cx-radius+5,cy-radius+5,cx+radius-5,cy+radius-5)),outline='#A5BBC5',width=S)
    text(cx,cy+1,cat.split()[-1],27,INK,'sans_bold','mm')

# Generated plinth, with a taller upper riser containing the exact label/value pairs.
base = Image.open(A / 'base.png').convert('RGBA')
bw,bh=base.size
strips=[(0,24,26),(24,56,86),(56,98,47),(98,bh,50)]
by=BASELINE
for sy,ey,dh in strips:
    paste_rgba(base.crop((0,sy,bw,ey)),80,by,2000,dh)
    by+=dh
# A translucent wash on the first stone riser keeps small exact numbers legible.
wash=Image.new('RGBA',(W*S,H*S))
wd=ImageDraw.Draw(wash)
wd.rectangle(box((224,1567,1937,1650)),fill=(235,244,245,80))
canvas.paste(wash,(0,0),wash)
dr = ImageDraw.Draw(canvas)
for i,(cx,cat,v) in enumerate(zip(CENTERS,DATA['categories'],values)):
    text(cx,1590,cat,28,INK,'sans_bold','mm')
    text(cx,1627,str(v),25,INK,'mono','mm')
    if i < len(CENTERS)-1:
        sep=(cx+CENTERS[i+1])/2
        line((sep,1580,sep,1639),'#7F9BAB',0.8)

# Reading key distinguishes every non-data component from the quantitative body.
rect((110,1786,137,1813),'#205B78')
text(151,1788,'Solid blue height = assets',26)
rect((697,1786,724,1813),'#CDDFE5',outline='#93AFBC')
text(738,1788,'Pale shaft = structural support',26)
text(2045,1788,'LINEAR SCALE  0–7.60',24,MUTED,'sans_bold','rt')
text(110,1831,'Synthetic figures for fictitious banks. All ten values are shown exactly; tiny bars keep their true height.',24,MUTED)

final=canvas.resize((W,H),Image.Resampling.LANCZOS)
final.save(ROOT / 'final.png', optimize=True)
(OUT / 'geometry.json').write_text(json.dumps({
    'canvas':[W,H], 'unit':DATA['series'][0]['unit'],
    'plot_top_px':TOP,'baseline_px':BASELINE,'full_scale_height_px':PLOT_H,
    'mapping':'net_height_px = value / 7.6 * 860; no minimum height',
    'supersampling':S,'marks':geometry,
},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
np.savez_compressed(OUT / 'pixel_probe_reference.npz',
    pale=np.asarray(probe_bg),dark=np.asarray(probe_ink))
print('Rendered final.png:',W,H,'; bars:',len(geometry),'; smallest height:',geometry[-1]['net_height_px'])
