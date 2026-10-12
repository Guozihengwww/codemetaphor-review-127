#!/usr/bin/env python3
"""Rebuild the local, data-driven infographic. No model or network calls."""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / 'data.json').read_text(encoding='utf-8'))
BRIEF = json.loads((ROOT / 'brief.json').read_text(encoding='utf-8'))
W, H = BRIEF['width'], BRIEF['height']
S = 2
Y0, SCALE_HEIGHT = 585.0, 1100.0
HEAD_W = 164.0
HEAD_H = HEAD_W * 527.0 / 825.0
HEAD_X0, PITCH = 170.0, 176.0
PAPER_OFFSET = (301.0 - 102.0) / 825.0 * HEAD_W
PAPER_W = (873.0 - 301.0) / 825.0 * HEAD_W
TEETH, TOOTH_AMPLITUDE = 8, 4.0
INK, MUTED = '#293735', '#65736F'
COLORS = ['#377B90', '#BC6950', '#699467', '#B29439', '#786293', '#3D9295', '#AC6580', '#56749B', '#9A784A', '#799450']
FONTS = {'regular': 'NotoSans-Regular.ttf', 'bold': 'NotoSans-Bold.ttf', 'cjk': 'NotoSansCJK-Medium.ttc'}
canvas = Image.new('RGB', (W*S, H*S), 'white')
draw = ImageDraw.Draw(canvas)


def font(size, weight='regular'):
    return ImageFont.truetype(str(ROOT / 'assets/fonts' / FONTS[weight]), round(size*S))


def text(x, y, value, size, color=INK, weight='regular', anchor='lt'):
    draw.text((round(x*S), round(y*S)), value, font=font(size, weight), fill=color, anchor=anchor)


def line(points, fill, width=1):
    draw.line([(round(x*S),round(y*S)) for x,y in points], fill=fill, width=max(1,round(width*S)), joint='curve')


# Editorial header: preserve the exact supplied title and the synthetic notice.
draw.rounded_rectangle((96*S, 78*S, 393*S, 125*S), radius=10*S, fill='#E7F0EC')
text(116, 88, DATA['visible_notice'], 25, '#356252', 'bold')
text(1902, 91, f"{len(DATA['categories']):02d} GROUPS   /   0–100", 24, MUTED, 'regular', 'rt')
text(94, 161, DATA['title'], 82, INK, 'cjk')
series = DATA['series'][0]
text(98, 291, series['label'], 34, INK, 'bold')
text(302, 295, '·  ' + series['unit'], 29, MUTED)
text(1900, 298, '纸带越长，年用量越大', 27, MUTED, 'cjk', 'rt')
line([(98, 366), (1900, 366)], '#D9E0DC', 1.3)

# One common linear scale, increasing downwards from the head's lower edge.
lo = float(DATA['value_axis']['minimum'])
hi = float(DATA['value_axis']['maximum'])
assert lo == 0 and hi == 100 and DATA['value_axis']['scale'] == 'linear'
text(127, 544, '用量', 23, MUTED, 'cjk', 'mm')
for tick in range(0, 101, 20):
    yy = Y0 + (tick-lo)/(hi-lo) * SCALE_HEIGHT
    line([(145, yy), (1920, yy)], '#A9B9B1' if tick == 0 else '#E9EDE9', 1.7 if tick == 0 else 1.0)
    line([(132, yy), (146, yy)], '#A9B9B1', 1.4)
    text(116, yy, str(tick), 26 if tick == 0 else 24, '#3D6052' if tick == 0 else MUTED, 'bold' if tick == 0 else 'regular', 'rm')
line([(145,Y0),(145,Y0+SCALE_HEIGHT)], '#B7C3BC', 1.3)

# The generated paper material is sampled at the same physical scale for all groups.
texture = Image.open(ROOT / 'assets/processed/paper_texture.png').convert('RGB')
tile_w = round(PAPER_W*S)
tile = texture.resize((tile_w,tile_w), Image.Resampling.LANCZOS)
strip_h = math.ceil((SCALE_HEIGHT + TOOTH_AMPLITUDE + 8)*S)
strip_surface = Image.new('RGB',(tile_w,strip_h),'white')
for k, yy in enumerate(range(0,strip_h,tile_w)):
    piece = ImageOps.flip(tile) if k % 2 else tile
    strip_surface.paste(piece,(0,yy))

geometry = []
for i, (category, value) in enumerate(zip(DATA['categories'], series['values'])):
    hx = HEAD_X0 + i*PITCH
    x0 = hx + PAPER_OFFSET
    x1 = x0 + PAPER_W
    mid = (x0+x1)/2
    yend = Y0 + (float(value)-lo)/(hi-lo) * SCALE_HEIGHT
    label = str(value)
    text(mid, 426, category, 28, INK, 'bold', 'mm')
    # The zigzag has zero integral offset: each rising/falling segment averages to yend.
    torn = [(x0 + j*PAPER_W/(2*TEETH), yend + (TOOTH_AMPLITUDE if j%2 else -TOOTH_AMPLITUDE)) for j in range(2*TEETH+1)]
    polygon = [(x0,Y0),(x1,Y0)] + list(reversed(torn))
    mask = Image.new('L', canvas.size, 0)
    md = ImageDraw.Draw(mask)
    md.polygon([(round(x*S),round(y*S)) for x,y in polygon], fill=255)
    material = Image.new('RGB',canvas.size,'white')
    material.paste(strip_surface,(round(x0*S),round(Y0*S)))
    canvas.paste(material,(0,0),mask)
    draw = ImageDraw.Draw(canvas)
    # A fine edge and a symmetric tear make the data-carrying contour explicit.
    line([(x0,Y0),torn[0]], '#B5B2AA', 1.3)
    line([(x1,Y0),torn[-1]], '#B5B2AA', 1.3)
    line(torn, '#8D9690', 1.8)
    # Decorative sheet perforations have a fixed pitch, independent of the values.
    number_y = yend - 48
    yy = Y0 + 77
    perforations = []
    while yy < yend-15:
        if abs(yy-number_y)>42:
            xx = x0+8
            while xx < x1-7:
                line([(xx,yy),(min(xx+5.2,x1-7),yy)], '#B0B3AB', 1.6)
                xx += 10.4
            perforations.append(yy)
        yy += 100
    # Generated head is shared in geometry, but has a distinct code-tinted end pattern.
    head = Image.open(ROOT / f'assets/processed/roll_head_{i+1:02d}.png').convert('RGBA')
    head = head.resize((round(HEAD_W*S),round(HEAD_H*S)),Image.Resampling.LANCZOS)
    canvas.paste(head,(round(hx*S),round((Y0-HEAD_H)*S)),head)
    draw = ImageDraw.Draw(canvas)
    text(mid, number_y, label, 56, INK, 'bold', 'mm')
    geometry.append({'category':category,'value':value,'label':label,'head_box':[hx,Y0-HEAD_H,hx+HEAD_W,Y0],
                     'paper_x0':x0,'paper_x1':x1,'zero_y':Y0,'endpoint_mean_y':yend,
                     'paper_length_px':yend-Y0,'torn_vertices':torn,'number_center':[mid,number_y],
                     'perforation_y':perforations})

# Explanatory footer stays outside the data marks.
line([(98,1776),(1900,1776)], '#D9E0DC', 1.3)
text(98,1813,'数据说明',24,'#3D6052','cjk')
text(98,1859,'本例合成数据，按输入顺序展示。',25,MUTED,'cjk')
text(98,1900,'Annual use  ·  rolls/person/year',24,MUTED)
text(990,1813,'阅读方式',24,'#3D6052','cjk')
text(990,1859,'卷头下缘为零；撕口的平均高度对应数值。',25,MUTED,'cjk')
text(990,1900,'卷头大小、纹样与穿孔数量不表示数值。',25,MUTED,'cjk')

canvas = canvas.resize((W,H), Image.Resampling.LANCZOS)
canvas.save(ROOT / 'final.png')
(ROOT / 'artifacts').mkdir(exist_ok=True)
(ROOT / 'artifacts/geometry.json').write_text(json.dumps({'canvas':[W,H], 'axis':[lo,hi], 'zero_y':Y0,
    'full_scale_px':SCALE_HEIGHT,'serration_amplitude_px':TOOTH_AMPLITUDE,'groups':geometry},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(f'Rendered {W}×{H}: {len(geometry)} source-ordered groups, one shared linear scale.')
