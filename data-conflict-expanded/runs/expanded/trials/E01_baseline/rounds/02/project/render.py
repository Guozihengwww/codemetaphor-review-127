"""Offline rendering: local data + generated coal bodies + code-owned geometry.

Run through capture_render.py for every complete rendering attempt.
The body has one height scale: 10 image pixels per percentage point.
Its full alpha bounds, excluding independent ground shadows, define height.
"""
from pathlib import Path
from decimal import Decimal
import hashlib
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent
S = 3
INK = '#153F56'
MUTED = '#4B6F7B'


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def font(size, bold=False):
    name = 'sans_bold.otf' if bold else 'sans_regular.otf'
    return ImageFont.truetype(str(ROOT / 'assets/fonts' / name), round(size * S))


def draw_text(draw, xy, text, size, fill=INK, bold=False, anchor='lt'):
    draw.text(tuple(round(t*S) for t in xy), text, font=font(size, bold),
              fill=fill, anchor=anchor, stroke_width=0)


def line(draw, points, fill, width=1):
    draw.line([(round(x*S), round(y*S)) for x,y in points],
              fill=fill, width=max(1, round(width*S)))


def main():
    brief = read_json('brief.json')
    data = read_json('data.json')
    core = read_json('core.json')
    assert core['visible_title'] == data['title']
    assert len(data['series']) == 1
    series = data['series'][0]
    categories, values = data['categories'], series['values']
    assert len(categories) == len(values) == 6
    assert data['value_axis'] == {'minimum': 0, 'maximum': 30, 'scale': 'linear'}
    assert sum(Decimal(str(v)) for v in values) == Decimal('100.0')
    W, H = brief['width'], brief['height']
    assert (W, H) == (2000, 1148)

    # Smooth icy-blue studio ground, with restrained reproducible paper grain.
    yy, xx = np.mgrid[0:H, 0:W]
    t = yy / (H-1)
    upper = np.array([234., 246., 246.])
    lower = np.array([193., 221., 223.])
    bg = upper[None,None,:] * (1-t[:,:,None]) + lower[None,None,:] * t[:,:,None]
    glow = np.exp(-(((xx-810)/1190)**2 + ((yy-625)/620)**2))
    bg += glow[:,:,None] * np.array([5., 5., 4.])
    grain = np.random.default_rng(20261013).normal(0, .34, (H,W,1))
    bg = np.uint8(np.clip(bg + grain, 0, 255))
    canvas = Image.fromarray(bg, 'RGB').resize((W*S,H*S), Image.Resampling.BICUBIC).convert('RGBA')
    d = ImageDraw.Draw(canvas)

    # The exact SYNTHETIC prefix is expressed through the prominent notice.
    d.rounded_rectangle((100*S, 68*S, 414*S, 115*S), radius=5*S,
                        fill='#A54728')
    draw_text(d, (121, 81), data['visible_notice'], 24, '#FFFFFF', True)
    title = data['title'].removeprefix('SYNTHETIC ')
    draw_text(d, (96, 156), title, 86, bold=True)
    draw_text(d, (101, 267), f"{series['label']} ({series['unit']})", 31)
    line(d, [(100,326),(1900,326)], '#91B4BD', 1)
    draw_text(d, (1900, 284), 'BY IMPORT PARTNER', 21, MUTED, True, 'rt')

    baseline = 823
    px_per_pct = 10.0
    maximum = float(data['value_axis']['maximum'])
    chart_top = baseline - maximum * px_per_pct
    coal = [Image.open(ROOT / f'assets/coal_{i+1:02d}.png').convert('RGBA')
            for i in range(len(values))]
    # Preserve the natural aspect of every complete body. Horizontal placement
    # only allocates space for the objects; no horizontal scale is presented.
    heights = [float(v) * px_per_pct for v in values]
    widths = [im.width / im.height * h for im,h in zip(coal,heights)]
    gap = 16.0
    content_width = sum(widths) + gap*(len(values)-1)
    left = 104.0 + (1818.0-content_width)/2
    boxes = []
    for width, height in zip(widths, heights):
        boxes.append((left, baseline-height, left+width, baseline))
        left += width + gap
    assert boxes[0][0] >= 100 and boxes[-1][2] <= 1930

    # A quiet height ruler and one physical ground plane.
    draw_text(d, (66, chart_top-44), '%', 21, MUTED, anchor='mt')
    line(d, [(78,chart_top),(78,baseline)], '#729DAA', 1.1)
    for tick in (0,10,20,30):
        y = baseline-tick*px_per_pct
        line(d, [(78,y),(90,y)], '#729DAA', 1.1)
        draw_text(d, (65,y), str(tick), 22, MUTED, anchor='rm')
    line(d, [(91,baseline),(1920,baseline)], '#A5C5C9', .8)

    # All shadows are separate from the data-bearing body and share one recipe.
    shadow = Image.new('RGBA',canvas.size)
    sd = ImageDraw.Draw(shadow)
    for x0,y0,x1,y1 in boxes:
        cx = (x0+x1)/2
        # Identical ambient shadow dimensions: no numeric mapping in shadows.
        sd.ellipse(((cx-137)*S,(baseline-8)*S,(cx+137)*S,(baseline+23)*S),
                   fill=(43,72,76,32))
    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(11*S)))
    contact = Image.new('RGBA',canvas.size)
    cd = ImageDraw.Draw(contact)
    for x0,y0,x1,y1 in boxes:
        # Support shadow follows the footprint only, never height or values.
        cd.ellipse(((x0+12)*S,(baseline-3)*S,(x1-8)*S,(baseline+9)*S),
                   fill=(32,58,64,46))
    canvas.alpha_composite(contact.filter(ImageFilter.GaussianBlur(4*S)))

    records = []
    for i,(im,box,v,category) in enumerate(zip(coal,boxes,values,categories)):
        x0,y0,x1,y1=box
        x, y = round(x0*S), round(y0*S)
        w, h = round((x1-x0)*S), round((y1-y0)*S)
        body = im.resize((w,h), Image.Resampling.LANCZOS)
        canvas.alpha_composite(body, (x,y))
        # Alpha bounds are logged separately from final-raster measurements.
        aa = np.asarray(body.getchannel('A'))
        ay,ax = np.where(aa>=128)
        alpha_bbox = [(x+ax.min())/S,(y+ay.min())/S,
                      (x+ax.max()+1)/S,(y+ay.max()+1)/S]
        records.append({'category':category,'value':v,'unit':series['unit'],
                        'asset':f'assets/coal_{i+1:02d}.png',
                        'target_bbox':list(box),'target_height_px':h/S,
                        'alpha_128_bbox':alpha_bbox,
                        'alpha_128_height_px':float((ay.max()-ay.min()+1)/S)})

    d = ImageDraw.Draw(canvas)
    for i,(box,v,category) in enumerate(zip(boxes,values,categories)):
        x0,y0,x1,y1 = box
        cx = (x0+x1)/2
        # Category binding and every mandatory numeric glyph are code-rendered.
        label = f'{v:.1f}%'
        draw_text(d, (cx,y0-97), label, 57, bold=True, anchor='mt')
        line(d, [(cx,y0-40),(cx,y0-15)], '#2D5B70', 1.5)
        draw_text(d, (cx,879), category, 31, bold=True, anchor='mt')

    # A compact key makes height, topic and the synthetic semantic scope explicit.
    line(d, [(100,997),(1900,997)], '#92B4BD', 1)
    draw_text(d, (100,1033), 'PILE HEIGHT = IMPORT SHARE', 22, bold=True)
    draw_text(d, (100,1069), 'Shared linear scale: 0–30%. Footprints, lump counts and shadows carry no values.', 22, MUTED)
    total = data['series_totals'][series['id']]
    draw_text(d, (1900,1033), f'Total  {total:.1f}%', 25, bold=True, anchor='rt')
    draw_text(d, (1900,1071), 'Synthetic partners · Illustrative data', 21, MUTED, anchor='rt')

    final = canvas.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
    final.save(ROOT / 'final.png', optimize=True)
    layout = {'canvas':[W,H], 'baseline_y':baseline,
              'axis_range':[0,maximum],'pixels_per_percentage_point':px_per_pct,
              'coal_bodies':records,
              'frozen_input_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                                     for p in ('input.json','brief.json','data.json','core.json','reference.png')
                                     if (ROOT/p).is_file()}}
    (ROOT/'artifacts').mkdir(exist_ok=True)
    (ROOT/'artifacts/geometry.json').write_text(
        json.dumps(layout,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'written':'final.png','size':[W,H],
                      'body_heights_px':heights,'baseline':baseline}))


if __name__ == '__main__':
    main()
