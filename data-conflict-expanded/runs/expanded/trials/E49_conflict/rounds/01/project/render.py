"""Offline data-bound aircraft infographic. Run through capture_render.py.

Every aircraft uses a single generated RGBA template. One isotropic affine
transform sets its area proportional to value and its nose on the linear axis.
The generated map supplies the background; no reference pixels are composited.
"""
from pathlib import Path
from decimal import Decimal
import json
import math
import hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
S = 4
INK = '#f3f3eb'
GOLD = '#f8dfa0'
MUTED = '#9bafc3'
GRID = '#283c51'


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def font(size, weight='regular'):
    name = {'regular': 'Sans-Regular.otf', 'bold': 'Sans-Bold.otf', 'mono': 'Mono.ttf'}[weight]
    return ImageFont.truetype(str(ROOT / 'assets/fonts' / name), round(size * S))


def box_scaled(box):
    return tuple(round(v * S) for v in box)


def text(draw, xy, value, size, fill=INK, weight='regular', anchor='la'):
    f = font(size, weight)
    draw.text((round(xy[0]*S), round(xy[1]*S)), value, font=f, fill=fill, anchor=anchor)
    bb = draw.textbbox((round(xy[0]*S), round(xy[1]*S)), value, font=f, anchor=anchor)
    return [round(v / S, 3) for v in bb]


def tracked(draw, xy, value, size, fill=MUTED, spacing=3, anchor='left'):
    f = font(size, 'bold')
    w = sum(draw.textlength(ch, font=f) / S for ch in value) + (len(value)-1)*spacing
    x = xy[0] - (w if anchor == 'right' else 0)
    for ch in value:
        draw.text((round(x*S), round(xy[1]*S)), ch, font=f, fill=fill)
        x += draw.textlength(ch, font=f)/S + spacing


def line(draw, xy, fill, width=1):
    draw.line(box_scaled(xy), fill=fill, width=max(1, round(width*S)))


def dashed(draw, x1, y1, x2, y2, fill, width=1, dash=8, gap=8):
    length = math.hypot(x2-x1, y2-y1)
    if length <= 0:
        return
    ux, uy = (x2-x1)/length, (y2-y1)/length
    for start in np.arange(0, length, dash+gap):
        end = min(start+dash, length)
        line(draw, (x1+ux*start, y1+uy*start, x1+ux*end, y1+uy*end), fill, width)


def aircraft_layer(template, meta, nose_x, row_y, scale, canvas_size):
    # All coordinates are in output pixels, not rounded mark dimensions.
    left = nose_x - meta['nose_x'] * scale
    top = row_y - meta['nose_y'] * scale
    box = (math.floor(left)-4, math.floor(top)-4,
           math.ceil(left+template.width*scale)+4,
           math.ceil(top+template.height*scale)+4)
    w, h = box[2]-box[0], box[3]-box[1]
    # One affine map for all four channels preserves fractional positioning.
    factor = scale*S
    affine = (1/factor, 0, (box[0]-left)/scale,
              0, 1/factor, (box[1]-top)/scale)
    layer = template.transform((w*S, h*S), Image.Transform.AFFINE, affine,
                               resample=Image.Resampling.BICUBIC)
    alpha = layer.getchannel('A').resize((w, h), Image.Resampling.LANCZOS)
    aa = np.array(alpha).astype(float)/255
    profile = aa.max(axis=0)
    indices = np.flatnonzero(profile >= .5)
    if len(indices):
        j = int(indices[-1])
        after = profile[j+1] if j+1 < len(profile) else 0
        edge = box[0]+j+.5+(profile[j]-.5)/(profile[j]-after)
        ys, xs = np.where(aa >= .5)
        visible_box = [box[0]+int(xs.min()), box[1]+int(ys.min()),
                       box[0]+int(xs.max())+1, box[1]+int(ys.max())+1]
    else:
        edge, visible_box = None, None
    metrics = {
        'affine_scale': scale,
        'nominal_nose_x': nose_x,
        'measured_nose_x_at_50pct_alpha': edge,
        'nose_error_px': None if edge is None else edge-nose_x,
        'measured_alpha_area_px2': float(aa.sum()),
        'opaque_pixel_count_at_50pct_alpha': int((aa >= .5).sum()),
        'visible_bbox_at_50pct_alpha': visible_box,
        'max_alpha': float(aa.max()),
        'raster_crop_box': box,
    }
    return layer, alpha, box, metrics


def add_trail(canvas, left, right, y, thickness):
    # The trailing atmosphere is explicitly not a quantity-bearing bar.
    if right <= left:
        return
    x0, x1 = math.floor(left), math.ceil(right)
    y0, y1 = math.floor(y-thickness/2-5), math.ceil(y+thickness/2+5)
    xx = (np.arange((x1-x0)*S)+.5)/S+x0
    yy = (np.arange((y1-y0)*S)+.5)/S+y0
    along = np.clip((xx-left)/(right-left), 0, 1)**1.5
    vertical = np.clip((thickness/2-np.abs(yy-y))/3, 0, 1)
    opacity = np.outer(vertical, along)*.69
    arr = np.empty((len(yy), len(xx), 4), dtype='uint8')
    arr[:, :, :3] = [233, 241, 244]
    arr[:, :, 3] = np.rint(opacity*255).astype('uint8')
    canvas.alpha_composite(Image.fromarray(arr), (x0*S, y0*S))


def main():
    paths = read_json('input.json')
    brief = read_json(paths['brief'])
    data = read_json(paths['data'])
    contract = read_json(paths['critical_metaphor_contract'])
    # Keep and validate all frozen dependencies without using reference pixels.
    with Image.open(ROOT / paths['reference']) as reference:
        reference_size = list(reference.size)
    assert contract['visible_title'] == data['title']
    assert data['synthetic'] is True
    assert len(data['series']) == 1
    series = data['series'][0]
    categories = data['categories']
    values = series['values']
    assert len(categories) == len(values) == 7
    assert data['value_axis'] == {'minimum': 0, 'maximum': 35, 'scale': 'linear'}
    assert sum(Decimal(str(v)) for v in values) == Decimal(str(data['series_totals'][series['id']]))
    W, H = brief['width'], brief['height']
    assert (W, H) == (2000, 2000)

    backdrop = Image.open(ROOT / 'assets/raw_02.png').convert('RGB').resize((W*S, H*S), Image.Resampling.LANCZOS)
    navy = Image.new('RGB', backdrop.size, '#102036')
    canvas = Image.blend(navy, backdrop, .41).convert('RGBA')
    draw = ImageDraw.Draw(canvas)
    labels = []

    # Editorial hierarchy, with an unmistakable synthetic-data notice.
    draw.rounded_rectangle(box_scaled((96, 80, 413, 135)), radius=5*S, fill=GOLD)
    tracked(draw, (117, 94), data['visible_notice'], 25, '#142337', spacing=2.25)
    tracked(draw, (1904, 96), 'AVIATION / MARKET VALUE', 23, MUTED, spacing=2.3, anchor='right')
    labels.append({'text': 'Airline market capitalization', 'bbox': text(draw, (93, 176), 'Airline market capitalization', 105, INK, 'bold')})
    labels.append({'text': series['label']+' | '+series['unit'], 'bbox': text(draw, (98, 310), series['label']+'  /  '+series['unit'], 34, MUTED)})
    line(draw, (96, 392, 1904, 392), '#4d6175', 1)

    x0, x35 = 492.0, 1622.0
    width = x35-x0
    axis_y = 508
    bottom = 1776
    tracked(draw, (98, 433), 'AIRLINE', 23, MUTED, spacing=3)
    tracked(draw, (1904, 433), 'USD BILLION', 23, MUTED, spacing=3, anchor='right')
    for tick in range(0, 36, 5):
        x = x0+width*tick/35
        text(draw, (x, 457), str(tick), 27, GOLD if tick == 0 else MUTED, 'regular', anchor='ma')
        line(draw, (x, axis_y-9, x, axis_y), '#71849a', 1.3)
        if tick == 0:
            dashed(draw, x, axis_y+8, x, bottom, '#7b8da0', .9, dash=6, gap=9)
        else:
            line(draw, (x, axis_y+8, x, bottom), GRID, .65)
    line(draw, (x0, axis_y, x35, axis_y), '#60758a', 1)

    template = Image.open(ROOT / 'assets/aircraft.png').convert('RGBA')
    meta = read_json('assets/aircraft_meta.json')
    maximum = max(values)
    max_width = 440.0
    base_scale = max_width/template.width
    rows = [653+171*i for i in range(len(values))]
    planes = []
    measures = []
    masks_dir = ROOT / 'artifacts/plane_masks'
    masks_dir.mkdir(parents=True, exist_ok=True)
    for i, (category, value, y) in enumerate(zip(categories, values, rows)):
        nose_x = x0+width*value/35
        scale = base_scale*math.sqrt(value/maximum)
        layer, mask, box, measure = aircraft_layer(template, meta, nose_x, y, scale, canvas.size)
        plane_left = nose_x-meta['nose_x']*scale
        add_trail(canvas, 388, plane_left+18*scale, y, 45)
        draw = ImageDraw.Draw(canvas)
        # The identifier modules carry category only, not market value.
        draw.rounded_rectangle(box_scaled((96, y-38, 373, y+38)), radius=5*S, fill='#1b3148')
        draw.rectangle(box_scaled((96, y-38, 101, y+38)), fill=GOLD)
        bb = text(draw, (124, y-23), category, 42, INK, 'bold')
        labels.append({'category': category, 'text': category, 'bbox': bb})
        value_label = '$'+str(value)
        vf = font(43, 'mono')
        value_left = 1904-draw.textlength(value_label, font=vf)/S
        dashed(draw, nose_x+17, y, value_left-24, y, '#607386', .85, dash=3, gap=9)
        bb = text(draw, (1904, y-25), value_label, 43, GOLD, 'mono', anchor='ra')
        labels.append({'category': category, 'text': value_label, 'value': value, 'bbox': bb})
        # Small, separated terminal witness lines keep the nose endpoint clear
        # without adding any area to the plane or widening its horizontal span.
        line(draw, (nose_x, y+23, nose_x, y+37), GOLD, .9)
        planes.append((layer, box))
        mask_name = f'artifacts/plane_masks/airline_{i+1:02d}.png'
        mask.save(ROOT / mask_name)
        measure.update(category=category, value=value, row_y=y,
                       target_nose_distance_px=nose_x-x0,
                       nominal_plane_width_px=template.width*scale,
                       mask_path=mask_name)
        measures.append(measure)

    for layer, box in planes:
        canvas.alpha_composite(layer, (box[0]*S, box[1]*S))
    draw = ImageDraw.Draw(canvas)

    # Reading notes explain the redundant encodings and synthetic scope.
    line(draw, (96, 1821, 1904, 1821), '#4d6175', 1)
    tracked(draw, (96, 1850), 'READING THE FLIGHT PATH', 23, GOLD, spacing=2.1)
    text(draw, (96, 1894), 'Nose position and aircraft area both scale with market value.', 30, INK)
    text(draw, (96, 1940), 'Linear scale from zero. Trails are decorative. Small values retain their true scale.', 26, MUTED)
    tracked(draw, (1904, 1850), '7 SYNTHETIC CARRIERS', 22, MUTED, spacing=2, anchor='right')

    final = canvas.convert('RGB').resize((W, H), Image.Resampling.LANCZOS)
    final.save(ROOT / 'final.png')

    # Screen-space evidence comes from the actual transformed raster masks.
    first_area = measures[0]['measured_alpha_area_px2']
    first_binary = measures[0]['opaque_pixel_count_at_50pct_alpha']
    for m in measures:
        ratio = m['value']/values[0]
        m['target_area_ratio'] = ratio
        m['measured_area_ratio'] = m['measured_alpha_area_px2']/first_area
        m['area_ratio_relative_error_pct'] = (m['measured_area_ratio']/ratio-1)*100
        m['binary_area_ratio_relative_error_pct'] = (m['opaque_pixel_count_at_50pct_alpha']/first_binary/ratio-1)*100
        m['measured_nose_distance_px'] = m['measured_nose_x_at_50pct_alpha']-x0
        m['nose_distance_relative_error_pct'] = m['nose_error_px']/m['target_nose_distance_px']*100
        m['nose_error_full_scale_pct'] = m['nose_error_px']/width*100
    manifest = {
        'canvas': [W,H], 'supersampling': S,
        'axis': {'zero_x': x0, 'maximum_x': x35, 'domain': [0,35], 'unit': series['unit'], 'scale': 'linear'},
        'area_method_zh': '同一生成机体按平方根比例等比缩放；在输出分辨率的独立透明度蒙版中积分覆盖面积，并另报50%透明度轮廓像素数。尾迹和定位细线不计入机体面积。',
        'nose_method_zh': '对最终分辨率机体蒙版逐列取最大透明度，在右端50%透明度处线性插值；与公共零点比较。',
        'reference_size_for_provenance_only': reference_size,
        'categories': categories, 'labels': labels, 'aircraft': measures,
        'frozen_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['input.json','brief.json','data.json','core.json','reference.png']},
    }
    (ROOT / 'artifacts/geometry_report.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'canvas': [W,H], 'aircraft_count': len(measures),
                      'max_area_error_pct': max(abs(m['area_ratio_relative_error_pct']) for m in measures),
                      'max_nose_error_px': max(abs(m['nose_error_px']) for m in measures)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
