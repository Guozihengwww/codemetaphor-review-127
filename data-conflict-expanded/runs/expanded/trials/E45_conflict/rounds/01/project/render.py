#!/usr/bin/env python3
"""Offline infographic. All quantitative marks are controlled by local data.json.

Run complete candidates only through the supplied capture_render.py wrapper.
The bird photograph is non-quantitative; the identical small silhouettes count
followers in units of 10 million, with tight-width fractional clipping.
"""
from __future__ import annotations

import io
import json
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path

import cairosvg
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent
S = 4
BG = '#EEEEE6'
INK = '#21353E'
MUTED = '#687A80'
BLUE = '#12A7D6'
RULE = '#E6EAE7'


def read_json(name, exact=False):
    return json.loads((ROOT / name).read_text(encoding='utf-8'),
                      parse_float=Decimal if exact else float)


def save_json(name, data):
    p = ROOT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def font(size, bold=False):
    return ImageFont.truetype(str(ROOT / 'assets' / ('NimbusSans-Bold.otf' if bold else 'NimbusSans-Regular.otf')), round(size*S))


def plain_decimal(v):
    return format(Decimal(str(v)), 'f')


def symbol_mask(width):
    """Trim to the silhouette's actual nonzero-alpha box BEFORE sizing/clipping."""
    png = cairosvg.svg2png(url=str(ROOT / 'assets/bird_symbol.svg'), output_width=2400, output_height=2400)
    alpha = Image.open(io.BytesIO(png)).convert('RGBA').getchannel('A')
    bbox = alpha.getbbox()
    tight = alpha.crop(bbox)
    h = round(width * tight.height / tight.width)
    mask = tight.resize((width, h), Image.Resampling.LANCZOS)
    assert mask.getbbox()[0] == 0 and mask.getbbox()[2] == width
    return mask, bbox


def fractional_mask(mask, fraction):
    """Integrate a vertical clipping boundary across each raster column.

    This is a width fraction, independent of the amount of blue bird ink.
    No squeezing, area normalization, or minimum mark width is used.
    """
    cutoff = float(fraction) * mask.width
    coverage = np.clip(cutoff - np.arange(mask.width, dtype=np.float64), 0.0, 1.0)
    clipped = np.rint(np.asarray(mask, dtype=np.float64) * coverage[None, :]).astype(np.uint8)
    return Image.fromarray(clipped, 'L')


def main():
    inputs = read_json('input.json')
    brief = read_json(inputs['brief'])
    data = read_json(inputs['data'], exact=True)
    core = read_json(inputs['critical_metaphor_contract'])
    # Frozen inputs are retained and used; the visual reference is never sampled.
    assert (ROOT / inputs['reference']).is_file()
    assert core['visible_title'] == data['title']
    assert data['synthetic'] and data['pictogram']['fraction_method'] == 'tight width left to right clipping'
    assert data['value_axis']['scale'] == 'linear' and data['value_axis']['minimum'] == 0
    series = data['series'][0]
    values = series['values']
    unit_value = Decimal(str(data['pictogram']['value']))
    assert len(data['categories']) == len(values) == 5
    assert sum(values) == data['series_totals'][series['id']]
    assert series['unit'] == 'million followers'
    assert max(values) <= data['value_axis']['maximum']

    W, H = brief['width'], brief['height']
    canvas = Image.new('RGB', (W*S, H*S), BG)
    draw = ImageDraw.Draw(canvas)
    text_records = []

    def rect(box, fill, outline=None, width=1, radius=0):
        box = tuple(round(v*S) for v in box)
        if radius:
            draw.rounded_rectangle(box, radius=round(radius*S), fill=fill, outline=outline, width=round(width*S))
        else:
            draw.rectangle(box, fill=fill, outline=outline, width=round(width*S))

    def line(points, fill=RULE, width=1):
        draw.line([(round(x*S), round(y*S)) for x,y in points], fill=fill, width=max(1,round(width*S)))

    def text(x, y, value, size, color=INK, bold=False, anchor='lt'):
        f = font(size, bold)
        xy = (round(x*S), round(y*S))
        draw.text(xy, value, fill=color, font=f, anchor=anchor)
        box = draw.textbbox(xy, value, font=f, anchor=anchor)
        text_records.append({'text': value, 'box_px': [round(b/S,3) for b in box]})
        return draw.textlength(value, font=f)/S

    def tracked(x, y, value, size, spacing=2.0, color=MUTED, bold=True):
        f = font(size, bold)
        xx = x
        for c in value:
            draw.text((round(xx*S),round(y*S)), c, font=f, fill=color, anchor='lt')
            xx += draw.textlength(c,font=f)/S + spacing
        text_records.append({'text':value,'box_px':[x,y,xx,y+size]})
        return xx-x

    def put_mask(mask, x, y, color=BLUE):
        canvas.paste(color, (round(x*S),round(y*S)), mask)

    # A quiet warm background and an editorial speech panel echo the reference.
    draw.polygon([(110*S,105*S),(1330*S,105*S),(1288*S,147*S),(1288*S,505*S),(110*S,505*S)], fill='white')
    rect((160,143,470,187), '#DFF3F8', radius=8)
    tracked(180,154,data['visible_notice'],24,2.0, '#097FA8')
    text(157,218,'Social account',108,BLUE,True)
    text(152,324,'followers',132,BLUE,True)
    text(163,452,'A portrait of five synthetic social accounts',28,MUTED)

    # Main data board: the photographic bird will visibly overlap its upper edge.
    rect((110,552,2050,1620),'white')
    text(160,604,'Followers by account',38,INK,True)
    text(160,662,'Same unit. Very different audiences.',27,MUTED)

    icon_width = 80
    mask, source_bbox = symbol_mask(icon_width*S)
    icon_h = mask.height/S
    legend = mask.resize((48*S, round(icon_h*0.6*S)), Image.Resampling.LANCZOS)
    put_mask(legend,832,606)
    text(898,612,'= 10 million followers',30,INK,True)
    text(833,663,'A partial bird uses a fraction of its width.',25,MUTED)

    tracked(160,747,'ACCOUNT / FOLLOWERS',21,1.8)
    tracked(832,747,'ONE SHARED PICTOGRAM SCALE',21,1.5)
    line([(160,780),(2000,780)], RULE,1.5)

    row_top = 786
    row_h = 160
    x0 = 850
    gap = 14
    pitch = icon_width+gap
    capacity = int(Decimal(str(data['value_axis']['maximum']))/unit_value)
    assert capacity == 12
    geometry = []

    for i,(category,value) in enumerate(zip(data['categories'],values)):
        top = row_top + i*row_h
        cy = top + row_h/2
        if i:
            line([(160,top),(2000,top)], RULE,1.5)

        # Generated portraits identify fictional accounts; their size carries no data.
        avatar_size = 112*S
        avatar = Image.open(ROOT / f'assets/avatar_{chr(65+i)}.png').convert('RGB')
        avatar = avatar.resize((avatar_size,avatar_size),Image.Resampling.LANCZOS)
        circular = Image.new('L',(avatar_size,avatar_size),0)
        ImageDraw.Draw(circular).ellipse((0,0,avatar_size-1,avatar_size-1),fill=255)
        canvas.paste(avatar,(round(166*S),round((cy-56)*S)),circular)

        text(315,cy-47,category,34,INK,True)
        number = plain_decimal(value)
        nwidth = text(314,cy+5,number,46,INK,True)
        text(314+nwidth+12,cy+18,'M',28,MUTED)

        y0 = round((cy-icon_h/2)*S)/S
        count = Decimal(str(value))/unit_value
        n = int(count.to_integral_value(rounding=ROUND_FLOOR))
        fraction = count-Decimal(n)
        # Empty positions are white, not ghost birds that could imply extra data.
        for j in range(n):
            put_mask(mask,x0+j*pitch,y0)
        pmask = fractional_mask(mask,fraction)
        px = x0+n*pitch
        if fraction:
            put_mask(pmask,px,y0)

        pb = pmask.getbbox()
        coverage_cols = np.asarray(pmask).max(axis=0)
        geometry.append({
            'category':category, 'value_million':number,
            'complete_birds':n, 'fraction_of_last_bird':plain_decimal(fraction),
            'tight_full_width_px':icon_width,
            'target_clip_width_px':float(fraction*icon_width),
            'fractional_mask_ink_bbox_px': [round(v/S,4) for v in pb] if pb else None,
            'fractional_mask_nonzero_columns':int(np.count_nonzero(coverage_cols)),
            'fractional_start_px':[px,y0],
            'clip_boundary_x_px':float(Decimal(px)+fraction*icon_width),
            'complete_symbol_origins_px':[[x0+j*pitch,y0] for j in range(n)],
            'row_box_px':[160,top,2000,top+row_h],
            'symbol_height_px':icon_h,
        })

        if i == 4:
            # A neutral leader locates the honest tiny crop without adding blue ink.
            if pb:
                mark_y = y0+(pb[1]+pb[3])/(2*S)
                line([(px+10,mark_y),(940,mark_y),(967,cy+4)],'#AAB7B8',1)
            text(984,cy-11,'Tiny slice: 2.148% of one bird',25,MUTED)

    # The hero photo sits over the data board, just as in the source composition.
    bird = Image.open(ROOT / 'assets/bluebird.png').convert('RGBA')
    bird_w = 780
    bird_h = round(bird.height*bird_w/bird.width)
    bird = bird.resize((bird_w*S,bird_h*S),Image.Resampling.LANCZOS)
    bx,by=1348,50
    shadow = Image.new('RGBA',canvas.size,(0,0,0,0))
    shadow_alpha = Image.new('L',canvas.size,0)
    shadow_alpha.paste(bird.getchannel('A'),((bx+4)*S,(by+16)*S))
    shadow_alpha = shadow_alpha.filter(ImageFilter.GaussianBlur(11*S)).point(lambda v:round(v*.13))
    shadow.putalpha(shadow_alpha)
    canvas = Image.alpha_composite(canvas.convert('RGBA'),shadow)
    canvas.alpha_composite(bird,(bx*S,by*S))
    draw = ImageDraw.Draw(canvas)

    # Compact footer gives the statistical scope and makes the clipping rule explicit.
    total = plain_decimal(data['series_totals'][series['id']])
    text(113,1662,'TOTAL',22,MUTED,True)
    text(213,1657,total+' million followers',32,INK,True)
    text(113,1716,'Read the birds by count and cropped width, not by inked area.  M = million followers.',25,MUTED)
    text(113,1755,'Synthetic values and fictional avatars. The large bluebird is illustrative and carries no quantity.',23,MUTED)

    final = canvas.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
    final.save(ROOT/'final.png',optimize=True)
    # Measurement inputs accompany every captured source state.
    save_json('artifacts/geometry.json',{
        'canvas_px':[W,H], 'supersampling':S, 'value_axis':{'minimum':0,'maximum':120,'scale':'linear'},
        'unit_million_per_bird':float(unit_value), 'tight_source_bbox_at_100x':list(source_bbox),
        'symbol_tight_width_px':icon_width, 'symbol_height_px':icon_h,
        'symbol_pitch_px':pitch, 'reserved_capacity_birds':capacity,
        'data_origin_x_px':x0, 'rows':geometry,
        'hero_box_px':[bx,by,bx+bird_w,by+bird_h], 'board_box_px':[110,552,2050,1620],
        'text_boxes':text_records,
    })


if __name__ == '__main__':
    main()
