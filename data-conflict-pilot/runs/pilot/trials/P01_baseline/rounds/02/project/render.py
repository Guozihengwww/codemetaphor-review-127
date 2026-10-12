#!/usr/bin/env python3
"""Reproduce the infographic from local data and generated coal cutouts.

Run through capture_render.py during this experiment. No model/network calls.
"""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent
NAVY = (18, 61, 90)
SECONDARY = (36, 83, 107)
FONT_DIR = ROOT / 'assets/fonts'

def font(size, bold=False):
    return ImageFont.truetype(str(FONT_DIR / ('NimbusSans-Bold.otf' if bold else 'NimbusSans-Regular.otf')), size)

def text(draw, xy, value, size, fill=NAVY, bold=False, anchor='lt'):
    draw.text(xy, value, font=font(size, bold), fill=fill, anchor=anchor, stroke_width=0)

def tracked_text(draw, center, value, size, tracking, fill, bold=False):
    f = font(size, bold)
    widths = [draw.textlength(c, font=f) for c in value]
    total = sum(widths) + tracking * (len(value) - 1)
    x = center[0] - total / 2
    for char, width in zip(value, widths):
        draw.text((x, center[1]), char, font=f, fill=fill, anchor='lt')
        x += width + tracking

def background(width, height):
    yy, xx = np.mgrid[:height, :width]
    x, y = xx / width, yy / height
    glow = np.exp(-(((x-.26)/.78)**2 + ((y-.54)/.82)**2))
    floor_light = np.exp(-((y-.77)/.19)**2)
    vignette = ((x-.45)**2 + .4*(y-.45)**2)
    rgb = np.empty((height, width, 3), dtype=np.float32)
    bases = [157, 203, 213]
    for c, base in enumerate(bases):
        rgb[..., c] = base + 17*glow + 4*floor_light - 10*vignette + (1-x)*3
    rng = np.random.default_rng(261012)
    grain = rng.normal(0, .52, (height, width, 1))
    rgb += grain
    return Image.fromarray(np.uint8(np.clip(rgb, 0, 255)), 'RGB').convert('RGBA')

def rgba_cutout(path, pixel_height):
    source = Image.open(ROOT / path).convert('RGBA')
    width = round(source.width * pixel_height / source.height)
    scaled = source.resize((width, pixel_height), Image.Resampling.LANCZOS)
    # Keep a tight, antialiased edge. The numerical endpoint is the visible coal,
    # not a transparent image margin. Extraction removed non-body components.
    arr = np.array(scaled)
    arr[..., 3][arr[..., 3] < 24] = 0
    return Image.fromarray(arr)

def add_shadows(scene, placements, baseline, objects):
    # Soft shadows are separate from the numerical object silhouettes.
    broad = Image.new('RGBA', scene.size)
    d = ImageDraw.Draw(broad)
    contact = Image.new('RGBA', scene.size)
    for item, obj in zip(placements, objects):
        x, w, h = item['x'], item['width'], item['height']
        cx = x + .515*w
        ry = max(7, min(20, w*.047))
        d.ellipse((cx-w*.54, baseline-ry*.66, cx+w*.54, baseline+ry*1.05), fill=(20,49,55,43))
        # Contact occlusion follows the generated bottom chunks, instead of a
        # generic ellipse. It is a separate ground effect, never a data mark.
        foot = obj.getchannel('A').crop((0, max(0, h-24), w, h))
        foot = foot.point(lambda a: round(a*.46))
        occlusion = Image.new('RGBA', foot.size, (7,29,34,0))
        occlusion.putalpha(foot)
        contact.alpha_composite(occlusion, (x, baseline-foot.height+3))
    scene.alpha_composite(broad.filter(ImageFilter.GaussianBlur(12)))
    scene.alpha_composite(contact.filter(ImageFilter.GaussianBlur(3)))

def main():
    brief = json.loads((ROOT / 'brief.json').read_text(encoding='utf-8'))
    data = json.loads((ROOT / 'data.json').read_text(encoding='utf-8'))
    extraction = json.loads((ROOT / 'assets/extraction.json').read_text(encoding='utf-8'))
    width, height = brief['width'], brief['height']
    categories = data['categories']
    series = data['series'][0]
    values = series['values']
    axis = data['value_axis']
    decimals = data['display_decimal_places']
    assert len(categories) == len(values) == len(extraction)
    assert len(data['series']) == 1 and axis['minimum'] == 0
    assert data['category_order'] == 'left_to_right'
    assert all(axis['minimum'] <= value <= axis['maximum'] for value in values)
    assert abs(sum(values)-data['total_percent']) < 1e-8

    baseline = 840
    full_scale_height = 370
    pixels_per_unit = full_scale_height / (axis['maximum'] - axis['minimum'])
    objects = [rgba_cutout(item['path'], round((value-axis['minimum'])*pixels_per_unit))
               for value, item in zip(values, extraction)]
    left, right = 130, 1900
    gap = (right-left-sum(obj.width for obj in objects)) / (len(objects)-1)
    assert gap > 25
    placements = []
    x = float(left)
    for category, value, obj, source in zip(categories, values, objects, extraction):
        obj_alpha = np.array(obj)[..., 3]
        top_ys, top_xs = np.nonzero(obj_alpha >= 128)
        top_y = int(top_ys.min())
        peak_xs = np.nonzero(obj_alpha[top_y:min(top_y+3, obj.height), :] >= 128)[1]
        item = {
            'category':category, 'value':value,
            'label':f'{value:.{decimals}f}{series["unit"]}',
            'asset':source['path'], 'x':round(x),
            'y':baseline-obj.height, 'width':obj.width, 'height':obj.height,
            'peak_x':round(x + float(np.median(peak_xs))),
            'body_top_alpha128_y':baseline-obj.height+top_y,
            'body_bottom_alpha128_y':baseline-obj.height+int(top_ys.max()),
        }
        placements.append(item)
        x += obj.width + gap

    scene = background(width, height)
    draw = ImageDraw.Draw(scene)
    text(draw, (100, 90), 'Solid-fuel import shares', 68, bold=True)
    text(draw, (103, 180), 'By fictional partner · percentage of total imported mass', 31, fill=SECONDARY)
    draw.rounded_rectangle((1546, 92, 1900, 155), radius=6, fill=NAVY)
    tracked_text(draw, (1723, 113), 'SYNTHETIC DATA', 25, 2.0, (185,229,234), bold=True)

    # A quiet calibration ruler accompanies the photographic marks.
    axis_ink = (62, 110, 130)
    text(draw, (77, baseline-full_scale_height-43), '%', 20, fill=axis_ink, anchor='rt')
    for tick in axis['ticks']:
        tick_y = round(baseline-(tick-axis['minimum'])*pixels_per_unit)
        draw.line((86, tick_y, 97, tick_y), fill=(86,137,151), width=1)
        text(draw, (74, tick_y), f'{tick:g}', 20, fill=axis_ink, anchor='rm')
    draw.line((97, baseline, 1910, baseline), fill=(124,174,184), width=1)
    add_shadows(scene, placements, baseline, objects)
    ground_rgb = np.asarray(scene.convert('RGB')).astype(np.float32)
    for item, obj in zip(placements, objects):
        scene.alpha_composite(obj, (item['x'], item['y']))

    draw = ImageDraw.Draw(scene)
    label_records = []
    for item in placements:
        center = item['x'] + item['width']/2
        tip_x = item['peak_x']
        coal_top = item['body_top_alpha128_y']
        label_y = coal_top - 133
        value_box = draw.textbbox((tip_x, label_y), item['label'], font=font(59), anchor='mt')
        text(draw, (tip_x, label_y), item['label'], 59, anchor='mt')
        line_top = value_box[3] + 17
        line_bottom = coal_top - 19
        draw.line((tip_x, line_top, tip_x, line_bottom), fill=NAVY, width=2)
        category_y = baseline + 66
        text(draw, (center, category_y), item['category'], 33, anchor='mt')
        category_box = draw.textbbox((center, category_y), item['category'], font=font(33), anchor='mt')
        label_records.append({'category':item['category'], 'value_text':item['label'],
                              'value_bbox_px':list(value_box), 'category_bbox_px':list(category_box),
                              'leader_px':[tip_x, line_top, tip_x, line_bottom]})

    draw.line((100, 1012, 1900, 1012), fill=(105,158,173), width=1)
    text(draw, (102, 1045), 'Pile height represents share', 25, bold=True)
    text(draw, (102, 1083), f'Common zero · linear scale {axis["minimum"]:g}–{axis["maximum"]:g}% · shadows excluded', 22, fill=SECONDARY)
    text(draw, (1900, 1045), f'TOTAL  {sum(values):.{decimals}f}%', 25, bold=True, anchor='rt')
    text(draw, (1900, 1083), 'Source: supplied synthetic data', 22, fill=SECONDARY, anchor='rt')

    out = scene.convert('RGB')
    out.save(ROOT / 'final.png', compress_level=6)
    # These measurements inspect actual pixels from the finished composition.
    # Difference against the actual pre-coal ground removes the separate shadow
    # and ruler pixels. The luminance condition rejects weak fringe pixels.
    rgb = np.asarray(out).astype(np.float32)
    luma = rgb @ np.array([.2126, .7152, .0722], dtype=np.float32)
    coal_difference = np.max(np.abs(rgb-ground_rgb), axis=2)
    measurements = []
    for item in placements:
        x0 = max(0, item['x']-2)
        x1 = min(width, item['x']+item['width']+2)
        y0 = item['y']-5
        y1 = baseline+5
        coal = ((luma[y0:y1, x0:x1] < 115) &
                (coal_difference[y0:y1, x0:x1] > 40))
        ys, xs = np.nonzero(coal)
        assert len(ys)
        observed_top = y0 + int(ys.min())
        observed_bottom = y0 + int(ys.max())
        observed_height = baseline-observed_top
        decoded = observed_height/pixels_per_unit + axis['minimum']
        measurements.append({
            'category':item['category'], 'source_value_percent':item['value'],
            'intended_object_bbox_xywh_px':[item['x'],item['y'],item['width'],item['height']],
            'measured_visible_dark_body_bbox_px':[x0+int(xs.min()),observed_top,x0+int(xs.max())+1,observed_bottom+1],
            'visible_height_from_common_zero_px':observed_height,
            'visible_value_percent':round(decoded,4),
            'absolute_error_percentage_points':round(abs(decoded-item['value']),4),
            'bottom_distance_from_zero_px':baseline-(observed_bottom+1),
            'alpha128_top_y':item['body_top_alpha128_y'],
            'alpha128_bottom_y':item['body_bottom_alpha128_y'],
        })
    artifacts = ROOT / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    geometry = {
        'canvas_px':[width,height], 'series_count':len(data['series']),
        'coal_pile_count':len(placements), 'category_order':categories,
        'axis':axis, 'baseline_y_px':baseline, 'full_scale_height_px':full_scale_height,
        'pixels_per_percentage_point':pixels_per_unit,
        'visible_measurement_method':'Final RGB pixels in coal-only ROIs; luminance <115 AND maximum channel difference from the actual pre-coal ground >40. Ground, ruler and separately composited shadows cancel in this difference. Leaders end 19px above coal and are outside these ROIs.',
        'measurements':measurements, 'labels':label_records,
        'maximum_absolute_error_percentage_points':max(r['absolute_error_percentage_points'] for r in measurements),
        'minimum_inter_object_gap_px':min(placements[i+1]['x']-(placements[i]['x']+placements[i]['width']) for i in range(len(placements)-1)),
    }
    (artifacts / 'geometry_checks.json').write_text(json.dumps(geometry, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'output':'final.png', 'dimensions':[width,height], 'geometry':geometry['measurements']}, ensure_ascii=False))

if __name__ == '__main__':
    main()
