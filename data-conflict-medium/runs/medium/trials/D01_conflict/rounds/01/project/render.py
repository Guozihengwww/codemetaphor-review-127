#!/usr/bin/env python3
"""Rebuild the infographic from local frozen data and generated object assets.

Run through capture_render.py for every complete rendering attempt.
The vertical data coordinate is shared by color bands, leader anchors and checks.
"""
from __future__ import annotations

import json
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import isotonic_regression


ROOT = Path(__file__).resolve().parent
S = 2
BG = (247, 241, 230)
INK = '#283335'
MUTED = '#707771'
RED = '#AD4134'
RULE = '#CCBFAA'
PALETTE = [
    '#3B7FA6', '#4F8774', '#C8664C', '#B29B4B',
    '#697BB1', '#3D9297', '#BD7350', '#7E82AD',
    '#739246', '#BA6372', '#4E90AA', '#BF9650',
    '#567B60', '#AD5250', '#4F769B', '#AA7D51',
    '#81739D', '#358A83', '#BB815B', '#668CB1',
    '#A4A252', '#B9668A', '#5C9980', '#AA6258',
    '#8784AB', '#BA985D', '#539AAC', '#94668B',
]


def rgb(color):
    return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))


@lru_cache(None)
def font(size, face='sans'):
    paths = {
        'sans': 'assets/fonts/sans.ttf',
        'bold': 'assets/fonts/sans-bold.ttf',
        'cjk': 'assets/fonts/cjk_regular.otf',
        'cjkbold': 'assets/fonts/cjk_bold.otf',
    }
    return ImageFont.truetype(str(ROOT / paths[face]), round(size * S))


def main():
    data = json.loads((ROOT / 'data.json').read_text(encoding='utf-8'), parse_float=Decimal)
    brief = json.loads((ROOT / 'brief.json').read_text(encoding='utf-8'))
    width, height = brief['width'], brief['height']
    assert (width, height) == (1440, 1991)
    series = data['series'][0]
    cats = data['categories']
    vals = [Decimal(v) for v in series['values']]
    exact_values = [str(v) for v in series['values']]
    total = sum(vals)
    assert len(cats) == len(vals) == len(PALETTE) == 28
    assert total == Decimal(data['value_axis']['maximum']) == 192
    assert data['value_axis']['minimum'] == 0
    unit = series['unit']
    total_label = format(total.normalize(), 'f')

    # Deterministic fine paper grain; only the object artwork is model-generated.
    rng = np.random.default_rng(1206)
    grain = rng.normal(0, .48, (height, width, 1))
    paper = np.clip(np.asarray(BG)[None, None, :] + grain, 0, 255).astype(np.uint8)
    canvas = Image.fromarray(paper).resize((width * S, height * S), Image.Resampling.BICUBIC).convert('RGBA')
    draw = ImageDraw.Draw(canvas)
    text_boxes = []

    def text(x, y, value, size, face='sans', fill=INK, anchor='lt', record=None):
        f = font(size, face)
        xy = (round(x * S), round(y * S))
        draw.text(xy, value, font=f, fill=fill, anchor=anchor, stroke_width=0)
        if record:
            box = draw.textbbox(xy, value, font=f, anchor=anchor)
            text_boxes.append({'id': record, 'text': value, 'bbox': [v / S for v in box]})

    def line(points, fill=RULE, stroke=1):
        draw.line([(round(x * S), round(y * S)) for x, y in points], fill=fill,
                  width=max(1, round(stroke * S)), joint='curve')

    def circle(x, y, r, fill, outline=None, stroke=1):
        draw.ellipse(((x-r)*S, (y-r)*S, (x+r)*S, (y+r)*S),
                     fill=fill, outline=outline, width=max(1, round(stroke*S)))

    def tracked(x, y, value, size, spacing=2.2, fill=MUTED, face='bold'):
        for ch in value:
            text(x, y, ch, size, face, fill)
            x += draw.textlength(ch, font=font(size, face)) / S + spacing

    # Header and reading notes.
    draw.rounded_rectangle((72*S, 54*S, 294*S, 94*S), radius=4*S, fill=RED)
    text(183, 75, data['visible_notice'], 19, 'bold', '#FFF8EF', 'mm', 'synthetic_notice')
    text(1368, 75, f'{len(cats):02d} GROUPS  /  {len(data["series"]):02d} SERIES',
         18, 'sans', MUTED, 'rm', 'category_count')
    text(72, 116, data['title'], 81, 'cjkbold', RED, record='title')
    text(76, 224, series['label'], 29, 'sans', INK, record='series')
    text(1368, 243, f'UNIT  {unit}  /  万亿立方米', 23, 'cjk', MUTED, 'rb', 'unit')
    line([(72, 281), (1368, 281)], RULE, 1.5)

    text(84, 356, '总量 / TOTAL', 21, 'cjk', MUTED)
    text(80, 399, total_label, 83, 'bold', RED, record='total')
    text(270, 451, unit, 26, 'sans', INK)
    text(86, 499, f'{len(cats)} 组 · 100%', 25, 'cjkbold', INK)
    line([(85, 548), (344, 548)], RULE, 1)
    text(85, 566, '按输入顺序自上而下排列', 19, 'cjk', MUTED)

    text(1083, 356, '读图 / ENCODING', 20, 'cjk', MUTED)
    text(1083, 401, '色带高度', 27, 'cjkbold', INK)
    text(1083, 440, '与数值成正比', 26, 'cjk', INK)
    text(1083, 494, '瓶肩至瓶底 = 100%', 20, 'cjk', MUTED)
    text(1083, 529, '护框与阀门不计入', 20, 'cjk', MUTED)

    # A single linear coordinate defines the shared axial data height.
    left, top, body_w, H = 480, 570, 480, 1210
    bottom = top + H
    edges_decimal = [Decimal(0)]
    for v in vals:
        edges_decimal.append(edges_decimal[-1] + v)
    edges = np.array([top + H * float(v / total) for v in edges_decimal])
    centers = (edges[:-1] + edges[1:]) / 2

    material = Image.open(ROOT / 'assets/body_material.png').convert('RGBA')
    source_h = material.height
    material = material.resize((body_w*S, H*S), Image.Resampling.LANCZOS)
    ma = np.asarray(material).astype(np.float32)
    lum = .2126*ma[:,:,0] + .7152*ma[:,:,1] + .0722*ma[:,:,2]
    shade = np.clip(.46 + .54 * lum / 190., .50, 1.13)
    row_colors = np.zeros((H*S, 3), dtype=np.float32)
    row_index = np.arange(H*S, dtype=np.float64)
    for i, color in enumerate(PALETTE):
        a, b = (edges[i]-top)*S, (edges[i+1]-top)*S
        coverage = np.clip(np.minimum(row_index+1, b) - np.maximum(row_index, a), 0, 1)
        row_colors += coverage[:,None] * np.asarray(rgb(color), dtype=np.float32)[None,:]
    pixels = np.zeros_like(ma, dtype=np.uint8)
    pixels[:,:,:3] = np.clip(row_colors[:,None,:] * shade[:,:,None], 0, 255).astype(np.uint8)
    pixels[:,:,3] = ma[:,:,3].astype(np.uint8)
    canvas.alpha_composite(Image.fromarray(pixels), (left*S, top*S))

    # The generated cage, valve and collar share the body's horizontal transform.
    hardware = Image.open(ROOT / 'assets/hardware.png').convert('RGBA')
    hardware_h = round(hardware.height * H / source_h * S)
    hardware = hardware.resize((body_w*S, hardware_h), Image.Resampling.LANCZOS)
    canvas.alpha_composite(hardware, (left*S, top*S-hardware_h))
    draw = ImageDraw.Draw(canvas)

    # Labels retain category order on each side. Monotone relocation avoids
    # crossing leader lines, while the band itself never gets a minimum height.
    label_y = np.zeros(len(vals))
    for side in (0, 1):
        ids = np.arange(side, len(vals), 2)
        min_gap = 64.
        offsets = np.arange(len(ids)) * min_gap
        fit = isotonic_regression(centers[ids] - offsets).x
        fit = np.clip(fit, 650, 1770-offsets[-1])
        label_y[ids] = fit + offsets

    bands = []
    leaders = []
    alpha = pixels[:,:,3]
    for i, (cat, value_text, color) in enumerate(zip(cats, exact_values, PALETTE)):
        cy, ly = float(centers[i]), float(label_y[i])
        row = min(H*S-1, max(0, round((cy-top)*S)))
        filled = np.flatnonzero(alpha[row] >= 128)
        is_left = i % 2 == 0
        edge_x = left + float(filled[0] if is_left else filled[-1]+1)/S
        badge_x = 330 if is_left else 1110
        end_x = badge_x + (29 if is_left else -29)
        elbow_near = 455 if is_left else 985
        elbow_far = 386 if is_left else 1054
        pts = [(edge_x,cy),(elbow_near,cy),(elbow_far,ly),(end_x,ly)]
        c = np.asarray(rgb(color))
        line_color = tuple(np.round(c*.79 + np.asarray(BG)*.21).astype(int))
        line(pts, line_color, 1.8)
        circle(edge_x,cy,3.2,color)
        circle(badge_x,ly,21,color)
        # The badge index is redundant, with the full category name next to it.
        lin = np.where(c/255 <= .04045, (c/255)/12.92, ((c/255+.055)/1.055)**2.4)
        lightness = float(lin @ np.asarray([.2126,.7152,.0722]))
        badge_ink = '#FFF9F0' if lightness < .31 else '#243234'
        text(badge_x, ly+.4, f'{i+1:02d}', 18, 'bold', badge_ink, 'mm')
        tx = 293 if is_left else 1147
        anchor = 'rt' if is_left else 'lt'
        text(tx,ly-26,cat,25,'sans',INK,anchor,f'category_{i+1:02d}')
        text(tx,ly+7,f'{value_text} {unit}',26,'bold',INK,anchor,f'value_{i+1:02d}')
        bands.append({'index':i+1,'category':cat,'value_exact':value_text,
                      'color':color,'top_y':float(edges[i]),'bottom_y':float(edges[i+1]),
                      'height_px':float(edges[i+1]-edges[i]),'share':float(vals[i]/total),
                      'center_y':cy,'side':'left' if is_left else 'right','label_center_y':ly})
        leaders.append({'index':i+1,'points':pts,'anchor_alpha':int(alpha[row,filled[0] if is_left else filled[-1]])})

    # A shared-height summary and a concise reading key beneath the object.
    text(720,1812,f'{total_label} {unit}  /  100%',22,'bold',INK,'mt')
    line([(72,1858),(1368,1858)],RULE,1.5)
    text(74,1883,'高度编码 / AXIAL SHARE',20,'cjkbold',RED)
    text(1366,1883,'数据：本地合成数据 · 非真实储量',20,'cjk',MUTED,'rt','provenance')
    text(74,1927,f'色带轴向高度 = 该组数值 ÷ {total_label} × H；面积不编码数据。',
         22,'cjk',INK,record='encoding_note')
    text(1366,1927,f'{unit} = 万亿立方米',22,'cjk',INK,'rt')

    final = canvas.convert('RGB').resize((width,height),Image.Resampling.LANCZOS)
    final.save(ROOT / 'final.png', compress_level=8)
    artifacts = ROOT / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    geometry = {
        'canvas':[width,height], 'supersampling':S, 'category_order':'top_to_bottom_as_listed',
        'total_exact':str(total),'data_height':{'top':top,'bottom':bottom,'height':H},
        'body_box':[left,top,left+body_w,bottom],
        'hardware_box':[left,top-hardware_h/S,left+body_w,top],
        'shade_formula':'0.46 + 0.54 * generated_surface_luminance / 190, clipped to [0.50,1.13]',
        'bands':bands,'leaders':leaders,'text_boxes':text_boxes,
    }
    (artifacts / 'render_geometry.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Rendered {width}×{height}; {len(vals)} bands; {total_label} {unit}; H={H}px.')


if __name__ == '__main__':
    main()
