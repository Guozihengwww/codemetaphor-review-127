"""Offline, data-bound telecommunications infographic.

Run from this directory with the supplied Python interpreter. For the experiment,
every complete execution must be launched through capture_render.py.
"""
from pathlib import Path
import json
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter


ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / 'data.json').read_text(encoding='utf-8'))
BRIEF = json.loads((ROOT / 'brief.json').read_text(encoding='utf-8'))
CORE = json.loads((ROOT / 'core.json').read_text(encoding='utf-8'))
INPUT = json.loads((ROOT / 'input.json').read_text(encoding='utf-8'))
W, H = BRIEF['width'], BRIEF['height']
S = 2
BASELINE = 1555.0
AXIS_TOP = 595.0
AXIS = DATA['value_axis']
PIXELS_PER_UNIT = (BASELINE - AXIS_TOP) / (AXIS['maximum'] - AXIS['minimum'])
N = len(DATA['categories'])
CENTERS = np.linspace(320.0, 1380.0, N).tolist()
MAST_WIDTH = 174.0
SIGN_DIAMETER = 178.0
BASE_HEIGHT = 155.0
INK = '#f3f2e9'
MUTED = '#a2b5c1'
CORAL = '#ed947e'
PALE_BLUE = '#96c9d3'
DARK_INK = '#132b3b'


def xy(v):
    return int(round(v * S))


def font(size, bold=False):
    name = 'font_bold.otf' if bold else 'font_regular.otf'
    return ImageFont.truetype(str(ROOT / 'assets' / name), xy(size))


def text(draw, x, y, value, size, color=INK, bold=False, anchor='lt'):
    draw.text((xy(x), xy(y)), str(value), font=font(size, bold), fill=color, anchor=anchor)


def line(draw, coords, fill, width=1):
    draw.line(tuple(xy(v) for v in coords), fill=fill, width=max(1, xy(width)))


def composite_box(canvas, asset, box):
    x0, y0, x1, y1 = box
    x, y = xy(x0), xy(y0)
    width, height = xy(x1) - x, xy(y1) - y
    if width <= 0 or height <= 0:
        return
    resized = asset.resize((width, height), Image.Resampling.LANCZOS)
    canvas.alpha_composite(resized, (x, y))


def background():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    glow = np.exp(-(((xx-740)/1090)**2 + ((yy-1170)/1230)**2))
    top_glow = np.exp(-(((xx-200)/1100)**2 + ((yy-0)/770)**2))
    noise = np.random.default_rng(31).normal(0, 0.28, (H,W))
    channels = []
    for start, gain, t in [(6,9,1), (20,17,3), (31,23,4)]:
        channels.append(np.clip(start + glow*gain + top_glow*t + noise, 0,255))
    rgb = np.stack(channels, axis=-1).astype('uint8')
    canvas = Image.fromarray(rgb).convert('RGBA').resize((W*S,H*S), Image.Resampling.BICUBIC)
    city = Image.new('RGBA', canvas.size)
    cd = ImageDraw.Draw(city)
    rng = np.random.default_rng(11)
    x = -20
    while x < W:
        width = int(rng.integers(44,100))
        top = int(rng.integers(1370,1620))
        cd.rectangle((xy(x),xy(top),xy(x+width),xy(1730)), fill=(3,16,25,110))
        for wx in range(x+12,x+width-8,18):
            for wy in range(top+19,1710,26):
                if rng.random() > .58:
                    cd.rectangle((xy(wx),xy(wy),xy(wx+5),xy(wy+9)), fill=(58,86,100,45))
        x += width+int(rng.integers(7,18))
    canvas.alpha_composite(city)
    return canvas


def support_asset():
    """Warp the generated pedestal to connect to the 174 px mast foot.

    The top of the source has a 410 px crossbeam in a 673 px image. Horizontal
    scale varies continuously to make this beam 174 px, and the feet 232 px.
    This support is fixed and never contributes to the encoded height.
    """
    source = Image.open(ROOT/'assets/support.png').convert('RGBA')
    width, height = xy(290), xy(BASE_HEIGHT)
    out = Image.new('RGBA',(width,height))
    sw, sh = source.size
    for row in range(height):
        t = row/(height-1)
        source_y = min(sh-1, int(round(t*(sh-1))))
        full_width = (174.0*673.0/410.0)*(1-t) + 232.0*t
        strip_width = xy(full_width)
        strip = source.crop((0,source_y,sw,source_y+1)).resize((strip_width,1),Image.Resampling.BICUBIC)
        out.alpha_composite(strip,((width-strip_width)//2,row))
    return out


def draw_header(canvas):
    d = ImageDraw.Draw(canvas)
    d.rounded_rectangle((xy(100),xy(86),xy(346),xy(132)),radius=xy(23),fill=CORAL)
    text(d,223,110,DATA['visible_notice'],21,DARK_INK,True,'mm')
    text(d,1495,109,'WIRELESS NETWORKS  /  % OF TOTAL',20,MUTED,False,'rm')
    text(d,98,170,'Wireless',117,INK,True)
    text(d,98,290,'subscriber shares',106,INK,True)
    text(d,103,413,'Illustrative subscriber mix across five network groups',28,MUTED)
    text(d,102,503,'Wireless subscriber share (%)',28,PALE_BLUE,True)
    text(d,103,545,'Highlighted crossbeam marks the measured height.',22,MUTED)


def draw_scale(canvas):
    d = ImageDraw.Draw(canvas)
    for value in range(int(AXIS['minimum']), int(AXIS['maximum'])+1, 10):
        y = BASELINE - (value-AXIS['minimum'])*PIXELS_PER_UNIT
        if value == 0:
            line(d,(192,y,1490,y),'#9bacb7',1.6)
        else:
            for x in range(193,1490,20):
                line(d,(x,y,min(x+7,1490),y),'#395260',1)
        text(d,163,y,f'{value}%',24,'#bbcad2' if value==0 else '#8199a8',value==0,'rm')
    # This bracket labels only the common, non-data support band.
    line(d,(177,BASELINE+8,177,BASELINE+BASE_HEIGHT),'#567180',1)
    line(d,(177,BASELINE+8,185,BASELINE+8),'#567180',1)
    line(d,(177,BASELINE+BASE_HEIGHT,185,BASELINE+BASE_HEIGHT),'#567180',1)
    text(d,160,1611,'Fixed',20,MUTED,False,'rt')
    text(d,160,1639,'supports',20,MUTED,False,'rt')


def tower(canvas, x, category, value, mast, cap, support, sign):
    top = BASELINE - (value-AXIS['minimum'])*PIXELS_PER_UNIT
    height = BASELINE-top
    # Same physical mapping for all categories: no minimum height is imposed.
    # A fixed, centered top crossbeam is the measurement reference.
    composite_box(canvas,mast,(x-MAST_WIDTH/2,top+2,x+MAST_WIDTH/2,BASELINE))
    canvas.alpha_composite(support,(xy(x-145),xy(BASELINE)))
    composite_box(canvas,cap,(x-35,top-3.5,x+35,top+3.5))
    d = ImageDraw.Draw(canvas)
    line(d,(x-31,top,x+31,top),CORAL,2.5)
    # Short collar connects sign and crossbeam without encoding data.
    line(d,(x,top-16,x,top-4),'#bdc7ca',7)
    line(d,(x-1,top-16,x-1,top-4),'#eff3ef',2)
    bottom = top-13
    center_y = bottom-SIGN_DIAMETER/2
    composite_box(canvas,sign,(x-SIGN_DIAMETER/2,bottom-SIGN_DIAMETER,
                               x+SIGN_DIAMETER/2,bottom))
    d = ImageDraw.Draw(canvas)
    text(d,x,center_y-44,'NETWORK GROUP',13,'#566b76',True,'mm')
    text(d,x,center_y-15,category,29,DARK_INK,True,'mm')
    line(d,(x-24,center_y+7,x+24,center_y+7),'#d78370',2)
    text(d,x,center_y+37,f'{value:.1f}%',37,DARK_INK,True,'mm')
    return {
        'category':category,'value_percent':value,
        'crossbeam_center_xy':[x,top],
        'zero_y':BASELINE,'encoded_height_px':height,
        'upper_mast_bbox':[x-MAST_WIDTH/2,top+2,x+MAST_WIDTH/2,BASELINE],
        'fixed_support_height_px':BASE_HEIGHT,
        'sign_center_xy':[x,center_y],
        'crossbeam_measurement_roi':[x-29,top-6,x+29,top+6],
    }


def draw_footer(canvas):
    d = ImageDraw.Draw(canvas)
    line(d,(100,1740,1500,1740),'#1f3b4a',1)
    text(d,800,1770,'One shared linear scale  ·  0–40%',24,PALE_BLUE,False,'mm')
    line(d,(100,1820,1500,1820),'#34505e',1)
    text(d,102,1855,'READING THE TOWERS',18,CORAL,True)
    text(d,102,1890,'Only height above 0% encodes share; signs and supports are not scaled.',22,INK)
    text(d,102,1925,'The 0.3% segment is shown at its true height.',21,MUTED)
    total = DATA['series_totals'][DATA['series'][0]['id']]
    text(d,1498,1855,f'SYNTHETIC TOTAL  {total:.0f}%',18,PALE_BLUE,True,'rt')
    text(d,1498,1890,'Illustrative network labels',22,MUTED,False,'rt')
    text(d,1498,1925,'Source: supplied data.json',21,MUTED,False,'rt')


def main():
    assert (W,H)==(1600,2000)
    assert AXIS['scale']=='linear'
    assert len(DATA['series'])==1
    assert DATA['series'][0]['unit']=='percent'
    assert N==len(DATA['series'][0]['values'])
    # Required frozen inputs remain local and the metaphor contract is loaded.
    for field in ('brief','data','reference','critical_metaphor_contract'):
        assert (ROOT/INPUT[field]).is_file()
    assert CORE['topic_zh']==DATA['topic_zh']
    full_mast=Image.open(ROOT/'assets/mast.png').convert('RGBA')
    mast=full_mast.crop((0,16,275,990))
    cap=full_mast.crop((83,0,193,16))
    sign=Image.open(ROOT/'assets/sign.png').convert('RGBA')
    support=support_asset()
    canvas=background()
    draw_header(canvas)
    draw_scale(canvas)
    geometry=[]
    for x,category,value in zip(CENTERS,DATA['categories'],DATA['series'][0]['values']):
        geometry.append(tower(canvas,x,category,value,mast,cap,support,sign))
    draw_footer(canvas)
    final=canvas.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
    final.save(ROOT/'final.png',optimize=True)
    (ROOT/'artifacts').mkdir(exist_ok=True)
    (ROOT/'artifacts/render_geometry.json').write_text(json.dumps({
        'canvas':[W,H],'axis_range':[AXIS['minimum'],AXIS['maximum']],
        'pixels_per_percentage_point':PIXELS_PER_UNIT,
        'measurement':'centerline of highlighted top crossbeam',
        'baseline_y':BASELINE,'towers':geometry,
    },ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Wrote final.png: 1600 x 2000; all five data-bound towers.')


if __name__=='__main__':
    main()
