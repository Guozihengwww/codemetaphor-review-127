"""Offline, area-calibrated egg composition. Run through capture_render.py.

Generated eggshell pixels supply the silhouette, material, pores and lighting.
Angular cut positions are solved from pixel-area integrals of that silhouette;
angles themselves never stand in for percentages.
"""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent
S = 2
RESAMPLE = Image.Resampling.LANCZOS


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def save_json(name, obj):
    p = ROOT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def rgb(h):
    return tuple(int(h[i:i+2], 16) for i in (1, 3, 5))


def font(size, bold=False):
    name = 'NimbusSans-Bold.otf' if bold else 'NimbusSans-Regular.otf'
    return ImageFont.truetype(str(ROOT / 'assets/fonts' / name), round(size * S))


def main():
    inp = read_json('input.json')
    brief = read_json(inp['brief'])
    data = read_json(inp['data'])
    core = read_json(inp['critical_metaphor_contract'])
    assert (ROOT / inp['reference']).is_file()
    assert core['topic_zh'] == data['topic_zh']
    W, H = brief['width'], brief['height']
    assert (W, H) == (2000, 1800)
    categories = data['categories']
    values = np.asarray(data['series'][0]['values'], dtype=float)
    assert len(categories) == len(values) == 9
    assert np.all(values > 0) and abs(values.sum() - 100) < 1e-8
    assert data['value_axis'] == {'minimum': 0, 'maximum': 100, 'scale': 'linear'}

    bg = rgb('#F1F5EF')
    ink = rgb('#213D35')
    muted = rgb('#60756A')
    palette_hex = ['#C3C69A', '#E8CF92', '#DCAB5B', '#BA803F',
                   '#CDDFD2', '#A5C7B6', '#7DAA95', '#568874', '#E4A668']
    palette = np.asarray([rgb(x) for x in palette_hex], dtype=np.float32)
    img = Image.new('RGB', (W*S, H*S), bg)
    draw = ImageDraw.Draw(img)
    text_boxes = []

    def line(points, fill, width=1):
        draw.line([(round(x*S), round(y*S)) for x, y in points], fill=fill,
                  width=max(1, round(width*S)), joint='curve')

    def text(x, y, value, size, fill=ink, bold=False, align='left', name=''):
        f = font(size, bold)
        tw = draw.textlength(value, font=f)
        xx = x*S - (tw if align == 'right' else tw/2 if align == 'center' else 0)
        xy = (round(xx), round(y*S))
        draw.text(xy, value, font=f, fill=fill, anchor='lt')
        box = draw.textbbox(xy, value, font=f, anchor='lt')
        text_boxes.append({'role': name, 'text': value, 'bbox': [round(v/S, 2) for v in box]})

    # Header: the synthetic notice is part of the visible title hierarchy.
    draw.rounded_rectangle((110*S, 76*S, 392*S, 126*S), radius=25*S, fill=ink)
    text(251, 91, data['visible_notice'], 23, bg, True, 'center', 'synthetic_notice')
    text(1890, 91, 'COMPOSITION / 9 CATEGORIES', 23, muted, False, 'right', 'category_count')
    text(105, 163, 'Egg nutrient composition', 96, ink, True, name='title')
    text(110, 278, 'Percent of egg composition  /  Total = 100%', 31, muted, name='unit_and_total')
    line([(110, 337), (1890, 337)], rgb('#C7D3C9'), 1.2)

    # Fixed non-data object placement. No subsequent nonuniform transformations.
    ex, ey, ew, eh = 440, 400, 837, 1190
    shell = Image.open(ROOT / 'assets/egg_shell.png').convert('RGBA').resize((ew*S, eh*S), RESAMPLE)
    arr = np.asarray(shell).astype(np.float32)
    alpha = arr[:, :, 3] / 255.0
    solid = alpha > 0
    sh, sw = alpha.shape
    hub = (sw * 0.50, sh * 0.52)
    yy, xx = np.mgrid[:sh, :sw]
    theta0 = math.radians(-95.0)
    angles = (np.arctan2(yy+0.5-hub[1], xx+0.5-hub[0]) - theta0) % (2*math.pi)

    # Alpha-weighted area quantiles determine radial boundaries in listed order.
    aa = angles[solid]
    ww = alpha[solid]
    order = np.argsort(aa, kind='stable')
    sorted_angles = aa[order]
    cumulative_area = np.cumsum(ww[order], dtype=np.float64)
    total_area_hi = float(cumulative_area[-1])
    targets = np.cumsum(values)[:-1] / 100 * total_area_hi
    indices = np.searchsorted(cumulative_area, targets)
    boundaries = np.r_[0., sorted_angles[indices], 2*math.pi]
    ids = np.searchsorted(boundaries[1:-1], angles, side='right').astype(np.uint8)

    # A common transfer of generated shell luminance retains actual material detail
    # and broad original lighting across every quantitative region.
    luminosity = arr[:, :, 0]*.2126 + arr[:, :, 1]*.7152 + arr[:, :, 2]*.0722
    shade = (luminosity - 172.0) * .72
    body_rgb = np.clip(palette[ids] + shade[:, :, None], 0, 255).astype(np.uint8)
    body = Image.fromarray(np.dstack([body_rgb, np.rint(alpha*255).astype(np.uint8)]), 'RGBA')

    # A restrained non-data contact shadow makes the intact shell read as an object.
    shadow = Image.new('RGBA', img.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.ellipse((640*S, 1565*S, 1084*S, 1620*S), fill=(35, 58, 38, 24))
    shadow = shadow.filter(ImageFilter.GaussianBlur(18*S))
    img.paste(shadow, (0, 0), shadow)
    img.paste(body, (ex*S, ey*S), body)
    draw = ImageDraw.Draw(img)

    def rim_at(angle):
        # Find the true generated silhouette's farthest occupied pixel on this ray.
        rr = np.arange(0, max(sh, sw)*1.3, .20)
        rx = np.floor(hub[0] + rr*np.cos(angle)).astype(int)
        ry = np.floor(hub[1] + rr*np.sin(angle)).astype(int)
        valid = (rx >= 0) & (rx < sw) & (ry >= 0) & (ry < sh)
        rr, rx, ry = rr[valid], rx[valid], ry[valid]
        good = alpha[ry, rx] >= .5
        radius = float(rr[good][-1])
        return np.array([ex + (hub[0]+radius*np.cos(angle))/S,
                         ey + (hub[1]+radius*np.sin(angle))/S]), radius/S

    def outward(point, angle, d):
        return point + d*np.array([np.cos(angle), np.sin(angle)])

    def node(p, color, r=3.1):
        draw.ellipse(((p[0]-r)*S, (p[1]-r)*S, (p[0]+r)*S, (p[1]+r)*S), fill=color)

    # Eight small components keep the input order clockwise along the egg and
    # top-to-bottom in the external callouts. The large water callout is on the left.
    row_tops = [416, 571, 696, 821, 946, 1071, 1196, 1321]
    leaders = []
    region_measurements = []
    for i in range(8):
        a_mid = theta0 + (boundaries[i]+boundaries[i+1])/2
        rim, radius = rim_at(a_mid)
        outside = outward(rim, a_mid, 9)
        shoulder = outward(rim, a_mid, 25)
        endpoint = [1450, row_tops[i]+33]
        points = [outside.tolist(), shoulder.tolist(), [1358, float(shoulder[1])], endpoint]
        leader_color = tuple(int(v) for v in np.clip(palette[i]*.60, 0, 255))
        line(points, leader_color, 1.65)
        node(outside, leader_color)
        leaders.append({'category': categories[i], 'points': points,
                        'rim_anchor': rim.tolist(), 'anchor_angle_degrees': math.degrees(a_mid)})
        label_lines = ['Vitamins minerals', 'and salt'] if i == 0 else [categories[i]]
        label_y = row_tops[i]
        for j, value in enumerate(label_lines):
            text(1485, label_y+34*j, value, 31, ink, name=f'category_{i}')
        value_y = label_y + (78 if i == 0 else 43)
        # g preserves all supplied numerical values including each 3.25.
        number = f'{values[i]:g}'
        text(1483, value_y, number, 53, ink, True, name=f'value_{i}')
        number_width = draw.textlength(number, font=font(53, True))/S
        text(1491+number_width, value_y+10, '%', 32, muted, name=f'unit_{i}')

    # Water remains an external label; no text obscures the data-bearing shell.
    rim, _ = rim_at(math.radians(151))
    p0 = outward(rim, math.radians(151), 8)
    water_line = [p0.tolist(), [407, float(p0[1])], [371, 1240]]
    wc = tuple(int(v) for v in palette[8]*.56)
    line(water_line, wc, 1.7)
    node(p0, wc)
    text(110, 1191, categories[8], 40, ink, name='category_8')
    text(105, 1252, f'{values[8]:g}', 96, ink, True, name='value_8')
    nw = draw.textlength(f'{values[8]:g}', font=font(96, True))/S
    text(115+nw, 1275, '%', 46, muted, name='unit_8')
    leaders.append({'category': categories[8], 'points': water_line, 'rim_anchor': rim.tolist()})

    # Quiet explanatory footer, with no invented source or nutritional claim.
    line([(110, 1662), (1890, 1662)], rgb('#C7D3C9'), 1.2)
    text(110, 1690, 'Area encodes share of the whole egg.', 27, ink, name='encoding')
    text(1890, 1690, 'Synthetic values · not a nutritional reference', 25, muted,
         align='right', name='synthetic_scope')
    text(110, 1737, 'Protein subcategories are mutually exclusive synthetic parts of Protein.',
         25, muted, name='subcategory_definition')

    final = img.resize((W, H), RESAMPLE)
    final.save(ROOT / 'final.png', compress_level=6)

    # Pixel-based measurements at the actual final display resolution. These
    # contribution masks are derived from the same compositing and downsampling
    # pipeline, not from the requested percentages.
    final_alpha = np.asarray(Image.fromarray(np.rint(alpha*255).astype(np.uint8)).resize((ew, eh), RESAMPLE), dtype=float)/255
    areas = []
    final_masks = []
    for i in range(len(values)):
        m = np.rint(alpha*(ids == i)*255).astype(np.uint8)
        mf = np.asarray(Image.fromarray(m).resize((ew, eh), RESAMPLE), dtype=float)/255
        final_masks.append(mf)
        areas.append(float(mf.sum()))
    area_total = float(sum(areas))
    measured_pct = np.asarray(areas)/area_total*100
    display_ids = np.argmax(np.stack(final_masks), axis=0).astype(np.uint8)+1
    display_ids[final_alpha < .5] = 0
    Image.fromarray(display_ids, 'L').save(ROOT / 'artifacts/region_ids.png')
    Image.fromarray(np.rint(final_alpha*255).astype(np.uint8), 'L').save(ROOT / 'artifacts/egg_mask.png')

    # Independently identify region colors in saved final.png. Grey-axis lighting
    # leaves palette chromatic differences invariant. Exclude only edge AA pixels.
    rendered = np.asarray(Image.open(ROOT / 'final.png').convert('RGB'))[ey:ey+eh, ex:ex+ew].astype(float)
    chroma = rendered - rendered.mean(axis=2, keepdims=True)
    pc = palette - palette.mean(axis=1, keepdims=True)
    distances = np.stack([np.sum((chroma-c)**2, axis=2) for c in pc])
    decoded = np.argmin(distances, axis=0)
    interior = final_alpha > .995
    decoded_counts = np.bincount(decoded[interior], minlength=len(values))
    decoded_pct = decoded_counts/decoded_counts.sum()*100
    expected_ids = display_ids.astype(int)-1
    agreement = float(np.mean(decoded[interior] == expected_ids[interior]))

    for i in range(len(values)):
        a0, a1 = theta0+boundaries[i], theta0+boundaries[i+1]
        p0, _ = rim_at(a0)
        p1, _ = rim_at(a1)
        # Measure the contour run using densely sampled true-silhouette points.
        samples = np.linspace(a0, a1, max(16, int((a1-a0)*220)))
        rim_points = np.array([rim_at(a)[0] for a in samples])
        rim_length = float(np.linalg.norm(np.diff(rim_points, axis=0), axis=1).sum())
        region_measurements.append({
            'category': categories[i], 'value_percent': float(values[i]),
            'final_projected_area_pixels': round(areas[i], 3),
            'measured_percent_of_mask': round(float(measured_pct[i]), 6),
            'error_percentage_points': round(float(measured_pct[i]-values[i]), 6),
            'saved_image_color_decoded_percent': round(float(decoded_pct[i]), 6),
            'boundary_degrees': [round(math.degrees(a0), 6), round(math.degrees(a1), 6)],
            'rim_span_pixels': round(rim_length, 3),
            'boundary_endpoints': [p0.tolist(), p1.tolist()]
        })

    union = sum(final_masks)
    bounds_ok = all(0 <= b['bbox'][0] <= b['bbox'][2] <= W and
                    0 <= b['bbox'][1] <= b['bbox'][3] <= H for b in text_boxes)
    report = {
        'canvas': [W, H], 'supersampling': S, 'category_count': len(categories),
        'data_source': inp['data'], 'core_contract': inp['critical_metaphor_contract'],
        'egg_bbox': [ex, ey, ex+ew, ey+eh],
        'shared_hub': [ex+hub[0]/S, ey+hub[1]/S],
        'partition_order': categories,
        'area_algorithm': 'Alpha-weighted angular CDF of the generated egg silhouette; quantile cuts; final-resolution masks measured after Lanczos downsampling.',
        'mask_area_pixels': round(float(final_alpha.sum()), 3),
        'sum_region_area_pixels': round(area_total, 3),
        'antialias_union_absolute_residual_pixels': round(float(np.abs(union-final_alpha).sum()), 4),
        'high_resolution_unassigned_pixels': int(np.sum(solid & (ids >= len(values)))),
        'max_absolute_area_error_percentage_points': round(float(np.max(np.abs(measured_pct-values))), 7),
        'saved_image_color_decode_agreement': round(agreement, 7),
        'saved_image_max_color_decoded_error_percentage_points': round(float(np.max(np.abs(decoded_pct-values))), 6),
        'all_text_within_canvas': bounds_ok,
        'regions': region_measurements, 'text_boxes': text_boxes, 'leader_lines': leaders,
        'non_data_elements': ['external labels', 'external leaders', 'contact shadow', 'eggshell texture and lighting'],
        'data_separator_stroke_width': 0,
        'limitation_zh': '分区在共同汇聚点处自然收窄；抗锯齿边缘存在亚像素混色。实际屏幕面积及最终图像颜色归类已单独测量。'
    }
    save_json('artifacts/geometry_check.json', report)
    print(json.dumps({k: report[k] for k in ['canvas', 'max_absolute_area_error_percentage_points',
          'saved_image_color_decode_agreement', 'saved_image_max_color_decoded_error_percentage_points',
          'all_text_within_canvas']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
