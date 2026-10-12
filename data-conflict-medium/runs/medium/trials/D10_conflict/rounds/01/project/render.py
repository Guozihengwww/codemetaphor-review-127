#!/usr/bin/env python3
"""Rebuild the data-bound sand infographic using only local data and assets."""
from pathlib import Path
import json
import math
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops

ROOT = Path(__file__).resolve().parent
S = 2
LANCZOS = Image.Resampling.LANCZOS


def main():
    data = json.loads((ROOT / 'data.json').read_text(encoding='utf-8'))
    brief = json.loads((ROOT / 'brief.json').read_text(encoding='utf-8'))
    W, H = brief['width'], brief['height']
    assert (W, H) == (2000, 2000)
    categories = data['categories']
    values = data['series'][0]['values']
    times = np.asarray(data['time_points'], dtype=float)
    axis = data['value_axis']
    assert axis == {'minimum': 0, 'maximum': 400, 'scale': 'linear'}
    assert len(categories) == len(values) == len(times)

    # One mapping is shared by the material mask, crest, grid, labels and audit.
    left, right, top, zero = 250.0, 1850.0, 580.0, 1600.0
    px_per_unit = (zero - top) / (axis['maximum'] - axis['minimum'])
    xs = left + (times - times[0]) / (times[-1] - times[0]) * (right - left)
    ys = zero - (np.asarray(values) - axis['minimum']) * px_per_unit
    points = list(zip(xs.tolist(), ys.tolist()))
    def edge(x):
        return float(np.interp(x, xs, ys))
    def box(coords):
        return tuple(round(v * S) for v in coords)
    def xy(coords):
        return [(round(x * S), round(y * S)) for x, y in coords]

    bg = (246, 248, 247)
    navy = '#172c36'
    grey = '#617074'
    amber = '#b67931'
    canvas = Image.new('RGB', (W * S, H * S), bg)
    draw = ImageDraw.Draw(canvas)
    latin_regular = ROOT / 'assets/fonts/NotoSans-Regular.ttf'
    latin_bold = ROOT / 'assets/fonts/NotoSans-Bold.ttf'
    cjk_regular = ROOT / 'assets/fonts/NotoSansCJK-Regular.ttc'
    cjk_bold = ROOT / 'assets/fonts/NotoSansCJK-Bold.ttc'
    def font(size, bold=False, cjk=False):
        path = (cjk_bold if bold else cjk_regular) if cjk else (latin_bold if bold else latin_regular)
        return ImageFont.truetype(str(path), round(size * S), index=2 if cjk else 0)
    text_records = []
    def text(x, y, value, size, fill=navy, bold=False, cjk=False, anchor='lt', role='copy'):
        f = font(size, bold, cjk)
        bounds = draw.textbbox((round(x * S), round(y * S)), str(value), font=f, anchor=anchor)
        draw.text((round(x * S), round(y * S)), str(value), font=f, fill=fill, anchor=anchor)
        text_records.append({'role': role, 'text': str(value), 'bbox_px': [round(v / S, 2) for v in bounds]})
        return [v / S for v in bounds]

    # Editorial header.
    draw.rounded_rectangle(box((112, 103, 414, 153)), radius=25*S, fill='#ede4d3')
    text(263, 128, data['visible_notice'], 23, '#755124', True, anchor='mm', role='synthetic_notice')
    text(1888, 126, f'{categories[0]}—{categories[-1]}', 28, grey, anchor='rm')
    text(109, 190, data['title'], 88, navy, True, True, role='title')
    text(113, 307, data['series'][0]['label'], 36, grey, role='series')
    draw.line(xy([(112, 401), (1888, 401)]), fill='#d3d9d7', width=2*S)
    text(180, 487, data['series'][0]['unit'], 29, grey, role='unit')

    # Grid and literal 0–400 scale. Grid within sand is handled separately.
    for value in range(0, 401, 100):
        gy = zero - value * px_per_unit
        draw.line(xy([(209, gy), (1890, gy)]), fill='#cdd5d3', width=2*S)
        text(174, gy, str(value), 32, grey, anchor='rm', role='y_tick')

    # A soft ground shadow stays below the exact shared zero baseline.
    shadow = Image.new('RGBA', canvas.size)
    sd = ImageDraw.Draw(shadow)
    sd.ellipse(box((left + 7, zero - 14, right + 14, zero + 23)), fill=(87, 68, 42, 57))
    shadow = shadow.filter(ImageFilter.GaussianBlur(12*S))
    canvas.paste(shadow, (0, 0), shadow)

    # The generated sand is the entire encoded body, not a decoration.
    poly = points + [(right, zero), (left, zero)]
    mask = Image.new('L', canvas.size)
    ImageDraw.Draw(mask).polygon(xy(poly), fill=255)
    bbox = mask.getbbox()
    material = Image.open(ROOT / 'assets/sand_texture.png').convert('RGB').resize((bbox[2]-bbox[0], bbox[3]-bbox[1]), LANCZOS)
    rgb = np.asarray(material).astype(np.float32)
    local_x = np.arange(bbox[0], bbox[2], dtype=np.float32) / S
    local_y = np.arange(bbox[1], bbox[3], dtype=np.float32)[:, None] / S
    crest = np.interp(local_x, xs, ys)[None, :]
    depth = np.clip((local_y - crest) / (zero - crest), 0, 1)
    shade = (1.035 - 0.12 * depth + 0.025 * np.exp(-depth * 16))[..., None]
    rgb = (rgb * 0.83 + np.asarray([230, 184, 124], dtype=np.float32) * 0.17) * shade
    material = Image.fromarray(np.uint8(np.clip(rgb, 0, 255)))
    material_layer = Image.new('RGB', canvas.size, bg)
    material_layer.paste(material, (bbox[0], bbox[1]))
    canvas.paste(material_layer, (0, 0), mask)

    # Fine horizontal reading rules remain visible on the material.
    grid = Image.new('RGBA', canvas.size)
    gd = ImageDraw.Draw(grid)
    for value in (100, 200, 300):
        gy = zero - value * px_per_unit
        gd.line(xy([(left, gy), (right, gy)]), fill=(105, 76, 38, 53), width=2*S)
    grid.putalpha(ImageChops.multiply(grid.getchannel('A'), mask))
    canvas.paste(grid, (0, 0), grid)

    # The shovel sits inside the high sand face so no encoded crest is hidden.
    shovel = Image.open(ROOT / 'assets/shovel.png').convert('RGBA')
    shovel = shovel.resize((round(shovel.width * .65 * S), round(shovel.height * .65 * S)), LANCZOS)
    shovel = shovel.rotate(29, resample=Image.Resampling.BICUBIC, expand=True)
    shovel_pos = (round(269*S), round(874*S))
    shovel_layer = Image.new('RGBA', canvas.size)
    shovel_layer.alpha_composite(shovel, shovel_pos)
    shovel_alpha_full = shovel_layer.getchannel('A')
    # A sloping grain lip buries the lower blade, establishing front/back contact.
    a = np.asarray(shovel_alpha_full).copy()
    yp, xp = np.nonzero(a > 100)
    bottom = float(yp.max()) / S
    buried_center = float(np.median(xp[yp > yp.max() - 22*S])) / S
    yy, xx = np.mgrid[0:H*S, 0:W*S]
    burial_y = bottom - 33 + 0.10 * (xx / S - buried_center) + 2.0 * np.sin(xx / (4.2*S)) + 0.85*np.sin(xx / (1.9*S))
    burial = np.clip((burial_y - yy / S) / 1.6, 0, 1)
    a = np.uint8(a.astype(np.float32) * burial)
    shovel_layer.putalpha(Image.fromarray(a))
    del yy, xx, burial, a
    # Cast shadow is clipped to the data body and cannot enlarge its boundary.
    cast_alpha = ImageChops.offset(shovel_layer.getchannel('A'), 12*S, 15*S).filter(ImageFilter.GaussianBlur(9*S))
    cast_alpha = ImageChops.multiply(cast_alpha, mask).point(lambda v: round(v * .22))
    cast = Image.new('RGB', canvas.size, '#655031')
    canvas.paste(cast, (0, 0), cast_alpha)
    canvas.paste(shovel_layer, (0, 0), shovel_layer)

    # Fine grains at the buried blade are sampled from the generated texture.
    draw = ImageDraw.Draw(canvas)
    rng = random.Random(3108)
    for _ in range(94):
        gx = buried_center + rng.gauss(0, 35)
        gy = bottom - 32 + 0.1*(gx-buried_center) + rng.gauss(0, 5)
        if left < gx < right and edge(gx) + 18 < gy < zero - 10:
            r = rng.uniform(.75, 2.1)
            col = (rng.randrange(187, 229), rng.randrange(130, 178), rng.randrange(63, 107))
            draw.ellipse(box((gx-r, gy-r*.68, gx+r, gy+r*.68)), fill=col)

    # Detached edge grains: separated from the contour and all numerical labels.
    label_layout = []
    for i, (x, y, val) in enumerate(zip(xs, ys, values)):
        half = draw.textlength(str(val), font=font(36, True)) / (2*S)
        min_y = min(edge(x-half-5), edge(x+half+5), y)
        label_bottom = min(y-36, min_y-21)
        label_layout.append((float(x), float(y), val, float(label_bottom), half))
    def avoids_labels(gx, gy):
        return all(not (abs(gx-x) < half+17 and lb-51 < gy < lb+14) for x,y,v,lb,half in label_layout)
    grains = []
    for _ in range(78):
        gx = rng.uniform(left+14, right-15)
        gy = edge(gx) - rng.uniform(17, 93)
        if not avoids_labels(gx, gy):
            continue
        radius = rng.uniform(1.0, 3.0)
        shade_col = rng.choice(['#dabc8a', '#dec8a5', '#e8d7b9', '#cda873'])
        if rng.random() < .45:
            draw.polygon(xy([(gx-radius,gy),(gx,gy-radius),(gx+radius,gy),(gx+.4*radius,gy+radius)]), fill=shade_col)
        else:
            draw.ellipse(box((gx-radius,gy-radius*.7,gx+radius,gy+radius*.7)), fill=shade_col)
        grains.append([round(gx,2), round(gy,2), round(radius,2)])

    # Exact straight crest, visible at every year and throughout each interval.
    draw.line(xy(points), fill='#ab743a', width=2*S, joint='curve')
    label_bindings = []
    for i, (x, y, val, label_bottom, half) in enumerate(label_layout):
        if label_bottom + 8 < y - 11:
            draw.line(xy([(x, label_bottom+8), (x, y-11)]), fill='#a99271', width=S)
        radius = 5.5
        draw.ellipse(box((x-radius,y-radius,x+radius,y+radius)), fill=bg, outline='#95622e', width=2*S)
        bounds = text(x, label_bottom, str(val), 36, navy, True, anchor='mb', role='value')
        label_bindings.append({'category':categories[i], 'value':val, 'node_px':[round(x,4),round(y,4)], 'label_bbox_px':bounds})
    draw.line(xy([(209, zero), (1890, zero)]), fill='#536466', width=2*S)
    for i, x in enumerate(xs):
        draw.line(xy([(x, zero+1), (x, zero+12)]), fill='#778380', width=2*S)
        text(x, zero+38, categories[i], 30, navy, anchor='mt', role='category')

    # Quiet explanatory copy in the naturally open upper-right plot area.
    draw.line(xy([(1295, 763), (1363, 763)]), fill=amber, width=3*S)
    text(1295, 794, '沙体上缘，即年度数值', 28, '#536365', True, True)
    text(1295, 840, '共用零线 · 等距年份 · 线性量尺', 23, '#7a8584', cjk=True)

    # Explicit synthetic provenance; no inherited publisher or source claim.
    draw.line(xy([(112, 1755), (1888, 1755)]), fill='#d3d9d7', width=2*S)
    text(112, 1798, '数据来源：data.json', 27, grey, cjk=True, role='source')
    text(1888, 1798, '合成演示数据', 27, grey, cjk=True, anchor='rt', role='provenance')
    text(112, 1860, '读图：从共同零线量至沙体上缘；相邻年度节点以直线连接。', 26, grey, cjk=True, role='encoding')
    text(112, 1908, '铲子与游离沙粒不表示数值。', 24, '#84908e', cjk=True)

    final = canvas.resize((W, H), LANCZOS)
    final.save(ROOT / 'final.png', optimize=True)
    artifact_dir = ROOT / 'artifacts'
    artifact_dir.mkdir(exist_ok=True)
    # These are diagnostic masks and objective placement records, not previews.
    final_mask = mask.resize((W, H), LANCZOS)
    final_mask.save(artifact_dir / 'data_body_mask.png')
    shovel_layer.getchannel('A').resize((W,H), LANCZOS).save(artifact_dir / 'shovel_visible_mask.png')
    geometry = {
        'canvas_px': [W,H], 'plot': {'left':left,'right':right,'top':top,'zero':zero},
        'axis': axis, 'pixels_per_unit':px_per_unit,
        'mapping':'x = 250 + (year - 2008)/(2019 - 2008)*1600; y = 1600 - value*2.55',
        'label_bindings':label_bindings, 'text_records':text_records,
        'detached_grains': grains,
        'shovel':{'position_px':[v/S for v in shovel_pos], 'rotation_deg':29, 'scale':.65, 'burial_bottom_px':bottom, 'burial_center_x_px':buried_center},
    }
    (artifact_dir / 'geometry.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':'final.png','size':[W,H],'categories':len(categories),'visible_values':values},ensure_ascii=False))


if __name__ == '__main__':
    main()
