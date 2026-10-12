"""Render the frozen data as faucet-connected water lengths, entirely offline.

Run via the supplied capture_render.py wrapper for all complete renders.
The only quantitative transform is y = OUTLET_Y + value / 9000 * AXIS_HEIGHT.
Generated water and splash pixels are both clipped at the same data endpoint.
"""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / 'data.json').read_text(encoding='utf-8'))
BRIEF = json.loads((ROOT / 'brief.json').read_text(encoding='utf-8'))
W, H = BRIEF['width'], BRIEF['height']
S = 3
OUTLET_Y = 640.0
AXIS_HEIGHT = 1330.0
AXIS_X = 159.0
FLOW_X = [425.0, 720.0, 1015.0, 1310.0, 1605.0]
FLOW_WIDTH = 30.0
FAUCET_WIDTH = 218.0
INK = '#172124'
MUTED = '#617479'
TEAL = '#117e94'
PALE = '#e3f7fa'
RULE = '#d8e5e8'
bg = Image.new('RGB', (W*S, H*S), 'white')
d = ImageDraw.Draw(bg)
text_records = []


def unit(x):
    return int(round(x*S))


def font(size, bold=False, cjk=False):
    name = ('NotoSansCJK-' if cjk else 'NotoSans-') + ('Bold' if bold else 'Regular')
    return ImageFont.truetype(str(ROOT / 'assets/fonts' / (name + ('.ttc' if cjk else '.ttf'))), unit(size))


def text(x, y, content, size, color=INK, bold=False, cjk=False, align='left'):
    f = font(size, bold, cjk)
    bbox = d.textbbox((0, 0), content, font=f)
    tw = d.textlength(content, font=f)
    px = unit(x) - (tw/2 if align == 'center' else tw if align == 'right' else 0)
    py = unit(y) - bbox[1]
    d.text((px, py), content, font=f, fill=color)
    actual = d.textbbox((px, py), content, font=f)
    text_records.append({'text': content, 'bbox': [round(v/S, 3) for v in actual]})


def line(points, fill=RULE, width=1):
    d.line([(unit(x),unit(y)) for x,y in points], fill=fill, width=max(1,unit(width)))


def dashed(y, x0, x1, dash=8, gap=9, color=RULE):
    x = x0
    while x < x1:
        line([(x,y),(min(x+dash,x1),y)], color, 1.1)
        x += dash + gap


def alpha_paste(im, x, y):
    bg.paste(im, (unit(x),unit(y)), im)


def compose_flow(cx, length):
    """Local supersampled water layer; its physical bottom is the data endpoint."""
    stream = Image.open(ROOT / 'assets/water_texture.png').convert('RGBA')
    splash = Image.open(ROOT / 'assets/splash.png').convert('RGBA')
    # Splash adapts within the available length; it never adds to that length.
    splash_h = min(65.0, length * 0.90)
    splash_w = FLOW_WIDTH + 100.0 * (splash_h / 65.0)**0.35
    if length < 10:
        # At a sub-four-pixel length, retain the generated base fan rather than
        # compressing the long, thin central crest into the same few pixel rows.
        splash = splash.crop((0,splash.height-160,splash.width,splash.height))
    local_w = unit(160)
    local_h = max(1, unit(length))
    layer = Image.new('RGBA', (local_w,local_h), (0,0,0,0))
    rw = unit(FLOW_WIDTH)
    ribbon = stream.resize((rw,local_h), Image.Resampling.LANCZOS)
    # Ensure a connected column throughout; the material has full opacity.
    ribbon.putalpha(255)
    layer.alpha_composite(ribbon, ((local_w-rw)//2,0))
    sw, sh = unit(splash_w), max(1,unit(splash_h))
    splash = splash.resize((sw,sh), Image.Resampling.LANCZOS)
    layer.alpha_composite(splash, ((local_w-sw)//2,local_h-sh))
    alpha_paste(layer, cx - 80, OUTLET_Y)
    return {
        'length_px': length,
        'endpoint_y': OUTLET_Y + length,
        'raster_clip_bottom_y': OUTLET_Y + local_h/S,
        'splash_height_px': sh/S,
        'splash_width_px': sw/S,
        'stream_width_px': FLOW_WIDTH,
    }


def main():
    assert len(DATA['categories']) == len(FLOW_X) == 5
    series = DATA['series'][0]
    assert len(series['values']) == 5
    axis = DATA['value_axis']
    assert axis['scale'] == 'linear' and axis['minimum'] == 0

    # Clear typographic hierarchy, with the source notice visible above the title.
    d.rounded_rectangle((unit(112),unit(73),unit(447),unit(121)),
                        radius=unit(24), fill=PALE)
    d.ellipse((unit(132),unit(91),unit(144),unit(103)),fill=TEAL)
    text(159,85,DATA['visible_notice'],23,TEAL,bold=True)
    text(110,161,DATA['title'],76,INK,bold=True,cjk=True)
    text(112,277,series['label'],38,INK)
    text(112,344,'水流长度表示年可再生水资源量',29,MUTED,cjk=True)
    text(112,389,'Gm³/year = 十亿立方米/年',25,MUTED,cjk=True)
    text(1690,389,'0–9000 · LINEAR SCALE',23,MUTED,align='right')
    line([(112,440),(1690,440)],RULE,1.4)

    # A restrained ruler reveals the common zero and exact, unbroken linear scale.
    line([(AXIS_X,OUTLET_Y),(AXIS_X,OUTLET_Y+AXIS_HEIGHT)],'#c9dadd',1.5)
    for value in range(0,9001,1000):
        y = OUTLET_Y + (value-axis['minimum'])/(axis['maximum']-axis['minimum'])*AXIS_HEIGHT
        line([(AXIS_X-8,y),(AXIS_X+9,y)],'#afc6cd',1.5)
        text(AXIS_X-23,y-13,str(value),23,MUTED,align='right')
    dashed(OUTLET_Y,AXIS_X+13,1712,color='#b7cdd2')
    line([(AXIS_X,OUTLET_Y+AXIS_HEIGHT),(1712,OUTLET_Y+AXIS_HEIGHT)],'#e3ecee',1)

    faucet = Image.open(ROOT / 'assets/faucet.png').convert('RGBA')
    fscale = FAUCET_WIDTH/faucet.width
    fw, fh = unit(FAUCET_WIDTH), unit(faucet.height*fscale)
    faucet = faucet.resize((fw,fh),Image.Resampling.LANCZOS)
    # Raw outlet spans x=605..726: its center relative to the cropped faucet is 597.5.
    outlet_local_x = (665.5-68)*fscale
    records = []
    for i,(cat,value,cx) in enumerate(zip(DATA['categories'],series['values'],FLOW_X)):
        fl = cx-outlet_local_x
        text(fl+FAUCET_WIDTH/2,461,cat,29,INK,bold=True,align='center')
        length = (value-axis['minimum'])/(axis['maximum']-axis['minimum'])*AXIS_HEIGHT
        record = compose_flow(cx,length)
        # Faucets sit above the ribbon. All five use exactly the same generated sprite.
        alpha_paste(faucet,fl,OUTLET_Y-fh/S)
        record.update(category=cat,value=value,flow_center_x=cx,
                      faucet_bbox=[fl,OUTLET_Y-fh/S,fl+FAUCET_WIDTH,OUTLET_Y],
                      outlet_y=OUTLET_Y)
        end = OUTLET_Y+length
        # Exact, external labels do not affect the water geometry.
        text(cx,end+30,str(value),45,TEAL,bold=True,align='center')
        records.append(record)

    # A data-derived numeric callout occupies the natural negative space.
    ratio = series['values'][0]/series['values'][-1]
    line([(953,1259),(1011,1259)],'#8edce9',6)
    text(949,1310,f'{ratio:g}×',128,TEAL,bold=True)
    text(954,1480,'Region A / Region E',33,INK,bold=True)
    text(954,1540,'8100 ÷ 20 = 405',29,MUTED)
    text(954,1611,'同一量尺，保留每一条短水流。',28,MUTED,cjk=True)

    # Explanatory caption and source status.
    line([(112,2044),(1690,2044)],RULE,1.4)
    text(112,2077,'读图',25,INK,bold=True,cjk=True)
    text(189,2077,'出水口下缘至飞溅底部的竖直距离，按数值线性缩放。',25,MUTED,cjk=True)
    text(112,2125,'合成数据 · Region A–E 为虚拟类别',23,MUTED,cjk=True)
    text(1690,2125,'Gm³/year',23,MUTED,align='right')

    final = bg.resize((W,H),Image.Resampling.LANCZOS)
    final.save(ROOT / 'final.png', optimize=True)
    (ROOT / 'artifacts/geometry.json').write_text(json.dumps({
        'canvas': [W,H], 'supersampling': S,
        'mapping': {'outlet_y':OUTLET_Y,'axis_height':AXIS_HEIGHT,
                    'axis_min':axis['minimum'],'axis_max':axis['maximum'],
                    'pixels_per_unit':AXIS_HEIGHT/(axis['maximum']-axis['minimum'])},
        'marks':records,'text_boxes':text_records,
        'numeric_callout':{'value':ratio,'formula':'8100 / 20'},
    }, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
