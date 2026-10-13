#!/usr/bin/env python3
"""Offline compositor. Run complete renders only through capture_render.py.

The generated photograph supplies the physical magnifier, connected handle,
ledger and shadows. Every sector and all editorial/data text are drawn here.
One angular mapping is used for sector paths, label placement and diagnostics.
"""
from pathlib import Path
from decimal import Decimal
import json
import math
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
S = 3
CX, CY = 780.0, 1005.0
R = 560.0
HOLE_RATIO = 0.45
YSCALE = 0.98
START_DEG = -90.0
NAVY = '#102C43'
MUTED = '#466274'
PALETTE = ['#E7A18B', '#F6D885', '#EDB95E', '#D9923B',
           '#9ACCBF', '#58AB9B', '#87B3D9', '#193C59']
FONTS = {
    'regular': 'assets/fonts/Sans-Regular.ttf',
    'bold': 'assets/fonts/Sans-Bold.ttf',
    'display': 'assets/fonts/Display-Bold.ttf',
}


def load_inputs():
    # Keep all frozen inputs local and available; do not modify them.
    inp = json.loads((ROOT / 'input.json').read_text(encoding='utf-8'))
    brief = json.loads((ROOT / inp['brief']).read_text(encoding='utf-8'))
    data = json.loads((ROOT / inp['data']).read_text(encoding='utf-8'))
    core = json.loads((ROOT / inp['critical_metaphor_contract']).read_text(encoding='utf-8'))
    assert (ROOT / inp['reference']).is_file()
    assert len(data['series']) == 1
    assert len(data['categories']) == len(data['series'][0]['values']) == len(PALETTE)
    assert core['visible_title'] == data['title']
    return brief, data


def font(size, weight='regular', scale=S):
    return ImageFont.truetype(str(ROOT / FONTS[weight]), round(size * scale))


def fmt(v):
    d = Decimal(str(v))
    return f'{d:,.0f}' if d == d.to_integral() else f'{d:,.1f}'


def percent(value, total):
    return format(Decimal(str(value)) / total * 100, 'f').rstrip('0').rstrip('.') + '%'


def point(radius, deg):
    a = math.radians(deg)
    return CX + radius * math.cos(a), CY + YSCALE * radius * math.sin(a)


def sectors(data):
    vals = data['series'][0]['values']
    total = sum(Decimal(str(v)) for v in vals)
    assert total == Decimal(str(data['series_totals']['S1']))
    assert total == Decimal(str(data['value_axis']['maximum']))
    out = []
    a = Decimal(str(START_DEG))
    for i, (category, value, parent) in enumerate(zip(data['categories'], vals, data['category_parent'])):
        sweep = Decimal(360) * Decimal(str(value)) / total
        out.append(dict(index=i, category=category, parent=parent, value=value,
                        percentage=percent(value, total), start=float(a), end=float(a+sweep),
                        sweep=float(sweep), mid=float(a+sweep/2), color=PALETTE[i]))
        a += sweep
    assert a == Decimal(str(START_DEG)) + 360
    return total, out


def text(draw, xy, content, size, weight='regular', fill=NAVY, anchor='lt'):
    draw.text(tuple(round(v*S) for v in xy), str(content), font=font(size, weight),
              fill=fill, anchor=anchor)


def tracked(draw, xy, content, size, spacing=2.5, fill=NAVY, weight='bold'):
    f = font(size, weight)
    x, y = (v*S for v in xy)
    for ch in content:
        draw.text((round(x), round(y)), ch, font=f, fill=fill, anchor='lt')
        x += draw.textlength(ch, font=f) + spacing*S
    return x/S


def make_block(sec):
    i = sec['index']
    sub = sec['category'].split(' / ')[1] if ' / ' in sec['category'] else None
    if i == 0:
        sizes, gap = [40, 65, 31], 13
    elif i in (1, 2):
        sizes, gap = [25, 19, 35, 22], 6
    elif sub:
        sizes, gap = [31, 22, 48, 27], 9
    else:
        sizes, gap = [36, 58, 30], 11
    lines = [(sec['parent'], sizes[0], 'bold')]
    if sub:
        lines.append((sub, sizes[1], 'regular'))
    lines += [(fmt(sec['value']), sizes[-2], 'bold'),
              (sec['percentage'], sizes[-1], 'regular')]
    return lines, gap


def block_dimensions(lines, gap):
    dims = []
    for content, size, weight in lines:
        f = font(size, weight, scale=1)
        b = f.getbbox(content)
        dims.append((f.getlength(content), b[3]-b[1]))
    return max(w for w, h in dims), sum(h for w, h in dims)+gap*(len(lines)-1), dims


def point_inside_sector(x, y, sec, margin=6):
    xx, yy = x-CX, (y-CY)/YSCALE
    rad = math.hypot(xx, yy)
    a = math.degrees(math.atan2(yy, xx))
    while a < sec['start']:
        a += 360
    # Geometric clearance from both radial boundaries, as well as the circles.
    lo, hi = a-sec['start'], sec['end']-a
    if lo < 0 or hi < 0:
        return False
    return (R*HOLE_RATIO+margin <= rad <= R-margin and
            rad*math.sin(math.radians(min(lo, 90))) >= margin and
            rad*math.sin(math.radians(min(hi, 90))) >= margin)


def label_layout(sec):
    lines, gap = make_block(sec)
    preferred = 0.75 if sec['index'] not in (1, 2) else 0.82
    w, h, dims = block_dimensions(lines, gap)
    candidates = []
    for fraction in np.linspace(.66, .87, 421):
        x, y = point(R*fraction, sec['mid'])
        corners = [(x+dx*w/2, y+dy*h/2) for dx in (-1,1) for dy in (-1,1)]
        if all(point_inside_sector(xx, yy, sec, 7) for xx,yy in corners):
            candidates.append((abs(fraction-preferred), fraction, x, y))
    if not candidates:
        raise ValueError('Direct label cannot fit: '+sec['category'])
    _, frac, x, y = min(candidates)
    return dict(lines=lines, gap=gap, width=w, height=h, dimensions=dims,
                x=x, y=y, radius_fraction=frac,
                bbox=[x-w/2, y-h/2, x+w/2, y+h/2])


def base_photo(w, h):
    raw = Image.open(ROOT / 'assets/raw_01.png').convert('RGB')
    # Measured aperture: left/right ~161/911, top/bottom ~199/942.
    # Map its axes to the one quantitative ellipse used by all data marks.
    src_cx, src_cy, src_rx, src_ry = 536.0, 570.5, 375.0, 371.5
    sx, sy = R/src_rx, R*YSCALE/src_ry
    tx, ty = CX-sx*src_cx, CY-sy*src_cy
    im = raw.transform((w*S, h*S), Image.Transform.AFFINE,
                       (1/(S*sx), 0, -tx/sx, 0, 1/(S*sy), -ty/sy),
                       Image.Resampling.BICUBIC, fillcolor=(226,237,241)).convert('RGBA')
    # Quiet the header without covering the generated lens, rim or handle.
    yy = np.arange(h*S, dtype=np.float32)/S
    alpha = (np.clip((410-yy)/245, 0, 1)*250).astype(np.uint8)
    am = Image.fromarray(np.repeat(alpha[:, None], w*S, axis=1), 'L')
    wash = Image.new('RGBA', im.size, (232,241,244,0))
    wash.putalpha(am)
    im = Image.alpha_composite(im, wash)
    return im, dict(source_center=[src_cx,src_cy], source_aperture_radii=[src_rx,src_ry],
                    scale=[sx,sy], translation=[tx,ty])


def annulus_layer(size, secs):
    layer = Image.new('RGBA', size, (0,0,0,0))
    d = ImageDraw.Draw(layer)
    for sec in secs:
        angles = np.linspace(sec['start'], sec['end'], max(3, math.ceil(sec['sweep']/.10)+1))
        outer = [point(R, a) for a in angles]
        inner = [point(R*HOLE_RATIO, a) for a in angles[::-1]]
        d.polygon([(round(x*S),round(y*S)) for x,y in outer+inner], fill=sec['color'])
    return layer


def main():
    brief, data = load_inputs()
    w,h = brief['width'],brief['height']
    total, secs = sectors(data)
    layouts = {sec['index']: label_layout(sec) for sec in secs[:-1]}
    im, transform = base_photo(w,h)
    im = Image.alpha_composite(im, annulus_layer(im.size, secs))

    # Clear glass still shows the generated ledger in the hole. This disk is
    # non-data; its radius stays exactly 0.45 times the data's outer radius.
    frost = Image.new('RGBA', im.size, (0,0,0,0))
    fd = ImageDraw.Draw(frost)
    ri = R*HOLE_RATIO
    fd.ellipse(((CX-ri)*S,(CY-ri*YSCALE)*S,(CX+ri)*S,(CY+ri*YSCALE)*S),
               fill=(235,244,246,212))
    im = Image.alpha_composite(im, frost)
    draw = ImageDraw.Draw(im)

    # The editorial title deliberately retains the metric and synthetic status.
    draw.rounded_rectangle((90*S,66*S,408*S,116*S), radius=7*S, fill=NAVY)
    tracked(draw,(111,81),data['visible_notice'],20,spacing=1.55,fill='#FFFFFF')
    text(draw,(86,143),'Audit fee',102,'display')
    text(draw,(86,249),'market shares',102,'display')
    text(draw,(92,366),data['series'][0]['label']+'  ·  '+data['series'][0]['unit'],28,
         fill=MUTED)

    for sec in secs[:-1]:
        lay=layouts[sec['index']]
        yy=lay['y']-lay['height']/2
        for (content,size,weight), (_,line_h) in zip(lay['lines'],lay['dimensions']):
            text(draw,(lay['x'],yy),content,size,weight,anchor='mt')
            yy+=line_h+lay['gap']

    # A small real slice receives a leader, never a minimum-size wedge.
    other = secs[-1]
    ox,oy=point(R+1,other['mid'])
    leader=[(ox,oy),(ox,413),(917,413)]
    draw.line([(round(x*S),round(y*S)) for x,y in leader],fill='#ECF3F6',width=6*S)
    draw.line([(round(x*S),round(y*S)) for x,y in leader],fill=NAVY,width=2*S)
    text(draw,(935,387),other['category'],26,'bold')
    text(draw,(935,422),fmt(other['value'])+'  ·  '+other['percentage'],24,fill=MUTED)

    # Central total is derived from the same data used by the sectors.
    ctitle='TOTAL AUDIT FEES'
    tf=font(20,'bold'); tw=sum(draw.textlength(ch,font=tf) for ch in ctitle)/S+(len(ctitle)-1)*2
    tracked(draw,(CX-tw/2,CY-109),ctitle,20,2,fill=MUTED)
    text(draw,(CX,CY-66),fmt(total),83,'display',anchor='mt')
    text(draw,(CX,CY+29),data['series'][0]['unit'],27,anchor='mt',fill=MUTED)
    draw.line(((CX-66)*S,(CY+82)*S,(CX+66)*S,(CY+82)*S),fill='#A6BAC4',width=2*S)
    text(draw,(CX,CY+103),'100% of the fee pool',21,anchor='mt',fill=MUTED)

    # Footer stays clear of the physical handle; it explains statistical scope.
    draw.line((92*S,1657*S,1032*S,1657*S),fill='#859EAC',width=2*S)
    tracked(draw,(92,1680),'READING THE LENS',18,2.5,fill=MUTED)
    text(draw,(91,1725),'Eight categories. One fee pool.',37,'display')
    text(draw,(92,1783),'Ring angles show each category’s share of total audit fees.',23,fill=MUTED)
    text(draw,(92,1823),'Firm B and Firm C subcategories are mutually exclusive.',23,fill=MUTED)
    text(draw,(92,1863),'Illustrative values; this is not a statistical survey.',23,fill=MUTED)
    text(draw,(92,1944),'Source: supplied synthetic dataset',18,fill=MUTED)

    final = im.convert('RGB').resize((w,h),Image.Resampling.LANCZOS)
    final.save(ROOT/'final.png')
    art=ROOT/'artifacts'; art.mkdir(exist_ok=True)
    layout_report=[]
    for sec in secs:
        row={k:v for k,v in sec.items()}
        if sec['index'] in layouts:
            row['text_bbox_px']=layouts[sec['index']]['bbox']
            row['label_radius_fraction']=layouts[sec['index']]['radius_fraction']
        layout_report.append(row)
    (art/'geometry.json').write_text(json.dumps(dict(
        canvas=[w,h], center=[CX,CY], outer_radius=R, inner_radius=ri,
        vertical_scale=YSCALE, angle_direction='clockwise', start_degrees=START_DEG,
        total=float(total), series=data['series'][0]['label'], unit=data['series'][0]['unit'],
        photographic_transform=transform, sectors=layout_report),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    index=json.loads((ROOT/'rounds/index.json').read_text(encoding='utf-8'))
    shutil.copy2(ROOT/'final.png',art/f"candidate_{index[-1]['number']:02d}.png")
    print(json.dumps({'size':[w,h],'total':float(total),'categories':len(secs),
                      'other_angle_degrees':secs[-1]['sweep']},ensure_ascii=False))


if __name__=='__main__':
    main()
