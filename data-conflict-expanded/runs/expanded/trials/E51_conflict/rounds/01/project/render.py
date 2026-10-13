"""Offline, data-bound shipbuilding pictograph. Run through capture_render.py."""
from __future__ import annotations
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
S = 4
PAPER = '#F8F5EE'
INK = '#202D36'
BLUE = '#145D9B'
MUTED = '#586A72'
RULE = '#D9D9D1'
RED = '#A9493F'
PINK = '#F3E3DF'

def load(name, decimals=False):
    return json.loads((ROOT / name).read_text(encoding='utf-8'), parse_float=Decimal if decimals else float)

def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()

def rgb(color):
    return tuple(int(color[i:i+2], 16) for i in (1, 3, 5))

def main():
    input_spec = load('input.json')
    brief = load(input_spec['brief'])
    data = load(input_spec['data'], decimals=True)
    core = load(input_spec['critical_metaphor_contract'])
    assert data['synthetic'] and len(data['series']) == 1
    assert data['value_axis'] == {'minimum':0, 'maximum':1300000, 'scale':'linear'}
    assert data['pictogram']['fraction_method'] == 'tight width left to right clipping'
    assert core['visible_title'] == data['title']
    width, height = brief['width'], brief['height']
    assert (width, height) == (2000, 1430)
    values = data['series'][0]['values']
    years = data['categories']
    assert len(values) == len(years)
    unit = Decimal(str(data['pictogram']['value']))
    assert unit == 45000
    assert all(Decimal(0) <= v <= data['value_axis']['maximum'] for v in values)
    canvas = Image.new('RGB', (width*S, height*S), PAPER)
    draw = ImageDraw.Draw(canvas)
    f_cache = {}
    def font(size, weight='regular'):
        key = (size, weight)
        if key not in f_cache:
            file = {'regular':'NimbusSans-Regular.otf', 'bold':'NimbusSans-Bold.otf', 'mono':'NimbusMonoPS-Regular.otf'}[weight]
            f_cache[key] = ImageFont.truetype(str(ROOT/'assets/fonts'/file), round(size*S))
        return f_cache[key]
    text_boxes = []
    def text(x, y, string, size, fill=INK, weight='regular', anchor='lt', role=None):
        f = font(size, weight)
        draw.text((round(x*S),round(y*S)), string, font=f, fill=fill, anchor=anchor)
        if role:
            bbox = draw.textbbox((round(x*S),round(y*S)), string, font=f, anchor=anchor)
            text_boxes.append({'role':role, 'text':string, 'bbox_px':[round(a/S,3) for a in bbox]})
    def tracking(x, y, string, size=18, spacing=1.5, fill=MUTED, weight='bold', anchor='left'):
        f = font(size,weight)
        widths = [draw.textlength(c,font=f)/S for c in string]
        length = sum(widths)+spacing*(len(string)-1)
        if anchor == 'right':
            x -= length
        for c,w in zip(string,widths):
            text(x,y,c,size,fill,weight)
            x += w+spacing
    def line(x1,y1,x2,y2,fill=RULE,stroke=1):
        draw.line((round(x1*S),round(y1*S),round(x2*S),round(y2*S)),fill=fill,width=round(stroke*S))
    def rectangle(box,fill):
        draw.rectangle(tuple(round(a*S) for a in box),fill=fill)

    glyph_source = Image.open(ROOT/'assets/shipyard.png').convert('RGBA')
    icon_w = 44
    icon_h = 49
    icon_gap = 6
    pitch = icon_w+icon_gap
    glyph = glyph_source.resize((icon_w*S,icon_h*S),Image.Resampling.LANCZOS)
    def paste_glyph(x,y,fraction=Decimal(1),tile=None):
        tile = glyph if tile is None else tile
        if fraction < 1:
            a = np.array(tile)
            cutoff = float(fraction)*a.shape[1]
            coverage = np.clip(cutoff-np.arange(a.shape[1]),0,1)
            a[:,:,3] = np.rint(a[:,:,3].astype(float)*coverage[None,:]).astype('uint8')
            tile = Image.fromarray(a)
        canvas.paste(tile,(round(x*S),round(y*S)),tile)

    # Masthead and explicitly synthetic metric title.
    tracking(76,58,'SHIPBUILDING  /  OBSERVATION YEARS',size=18,spacing=1.8)
    rectangle((1590,48,1924,94),RED)
    tracking(1611,62,data['visible_notice'],size=21,spacing=1.5,fill=PAPER)
    text(76,115,data['title'],64,weight='bold',role='title')
    text(78,206,'Shipbuilding output by observation year · net tons',27,fill=MUTED,role='topic_and_unit')
    legend_glyph = glyph_source.resize((58*S,65*S),Image.Resampling.LANCZOS)
    paste_glyph(1510,154,tile=legend_glyph)
    text(1589,158,'= 45,000',34,weight='bold',fill=BLUE,role='legend_value')
    text(1621,199,'net tons per full symbol',22,fill=MUTED,role='legend_unit')
    line(76,263,1924,263,INK,1.6)
    tracking(88,289,'YEAR',size=18,spacing=1.7)
    tracking(256,289,'OUTPUT IN SHIPYARD SYMBOLS',size=18,spacing=1.7)
    tracking(1924,289,'NET TONS BUILT',size=18,spacing=1.7,anchor='right')

    x0 = 256
    first_cy = 358
    row_step = 78
    row_info = []
    for i,(year,value) in enumerate(zip(years,values)):
        cy = first_cy+i*row_step
        y0 = cy-icon_h/2
        highlighted = year == '1915'
        if highlighted:
            rectangle((76,cy-37,1924,cy+37),PINK)
            # Subtle red print-dot strip, like the source's emphasis band.
            for yy in range(round(cy-33),round(cy+34),7):
                for xx in range(87+(yy%2)*3,1920,7):
                    r = 0.45*S
                    draw.ellipse((xx*S-r,yy*S-r,xx*S+r,yy*S+r),fill='#DDB9B1')
            rectangle((76,cy-37,81,cy+37),RED)
        if i != len(years)-1 and not highlighted and years[i+1] != '1915':
            line(86,cy+39,1924,cy+39,fill='#E6E3DA',stroke=0.5)
        text(88,cy,year,34,fill=RED if highlighted else INK,weight='bold',anchor='lm',role=f'year_{year}')
        n = int((value/unit).to_integral_value(rounding=ROUND_FLOOR))
        fraction = value/unit-n
        for j in range(n):
            paste_glyph(x0+j*pitch,y0)
        if fraction > 0:
            paste_glyph(x0+n*pitch,y0,fraction)
        label = format(value,',f')
        text(1924,cy,label,29,fill=RED if highlighted else INK,weight='mono',anchor='rm',role=f'value_{year}')
        whole_boxes = [[x0+j*pitch,y0,x0+j*pitch+icon_w,y0+icon_h] for j in range(n)]
        fractional_box = [x0+n*pitch,y0,x0+n*pitch+float(fraction)*icon_w,y0+icon_h] if fraction else None
        row_info.append({
            'year':year,'value':str(value),'label':label,'unit_value':str(unit),
            'full_icons':n,'fraction':str(fraction),'fraction_width_px':float(fraction)*icon_w,
            'symbol_width_px':icon_w,'symbol_height_px':icon_h,'gap_px':icon_gap,
            'x_start_px':x0,'y_top_px':y0,'y_center_px':cy,
            'whole_boxes_px':whole_boxes,'fractional_box_px':fractional_box,
            'highlighted':highlighted,'background':PINK if highlighted else PAPER
        })

    line(76,1205,1924,1205,INK,1.6)
    tracking(78,1235,'HOW TO READ',size=18,spacing=1.8,fill=BLUE)
    text(78,1277,'A partial symbol shows the same fraction of 45,000 net tons.',26,weight='bold',role='fraction_explanation')
    text(78,1316,'Read each partial symbol from the left; every full symbol has the same size.',23,fill=MUTED,role='clip_explanation')
    text(78,1352,'Years are equally spaced. The rows do not encode elapsed time.',23,fill=MUTED,role='spacing_explanation')
    line(1430,1236,1430,1380,fill=RULE,stroke=1)
    tracking(1470,1237,'ABOUT THE DATA',size=18,spacing=1.4,fill=RED)
    text(1470,1279,'11 observation years · one metric',24,fill=INK,role='scope')
    text(1470,1317,'Synthetic values, not historical statistics.',22,fill=MUTED,role='synthetic_context')
    text(1470,1353,'Exact values are listed at right.',22,fill=MUTED,role='precision_note')

    # One final downsample preserves fractional edge coverage without a minimum mark size.
    final = canvas.resize((width,height),Image.Resampling.LANCZOS)
    final.save(ROOT/'final.png', optimize=True)
    geometry = {
        'canvas_px':[width,height], 'supersampling':S, 'chart_x_start_px':x0,
        'linear_value_range':{'minimum':0,'maximum':1300000},
        'encoding':'fixed-size repeated generated glyphs; fractional terminal glyph clipped left to right across its tight width',
        'visible_unit_value':str(unit), 'icon_width_px':icon_w,'icon_height_px':icon_h,
        'rows':row_info,'text_boxes':text_boxes,
        'input_sha256':{name:sha(name) for name in ('input.json','brief.json','data.json','core.json','reference.png')},
        'processed_asset_sha256':sha('assets/shipyard.png')
    }
    (ROOT/'artifacts').mkdir(exist_ok=True)
    (ROOT/'artifacts/render_geometry.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Wrote final.png: 2000 x 1430; 11 rows; one 45,000-net-ton generated pictogram; exact decimal labels.')

if __name__ == '__main__':
    main()
