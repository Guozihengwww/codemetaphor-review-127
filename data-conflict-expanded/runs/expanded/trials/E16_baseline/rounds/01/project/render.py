#!/usr/bin/env python3
"""Rebuild the infographic from frozen local data and a generated garment.

Run through capture_render.py for every complete experiment rendering.
All numerical boundaries are quantiles of the FINAL projected visible cloth.
No model or network call is made by this script.
"""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parent
AA = 2
PALETTE = ['#456781', '#7c96b5', '#cb7765', '#345875',
           '#85927f', '#39363a', '#aaa29d', '#bb4838']
INK = '#29363a'
MUTED = '#64706e'
RULE = '#bfc4bd'


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def write_json(path, value):
    (ROOT / path).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def rgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float32)


def font(size, family='sans', bold=False):
    f = 'P052-Roman.otf' if family == 'serif' else ('NimbusSans-Bold.otf' if bold else 'NimbusSans-Regular.otf')
    return ImageFont.truetype(str(ROOT / 'assets/fonts' / f), round(size * AA))


def main():
    inputs = read_json('input.json')
    brief = read_json(inputs['brief'])
    data = read_json(inputs['data'])
    core = read_json(inputs['critical_metaphor_contract'])
    assert core['visible_title'] == data['title']
    assert data['synthetic'] and data['series'][0]['unit'] == 'percent'
    cats = data['categories']
    values = np.array(data['series'][0]['values'], np.float64)
    assert len(cats) == len(values) == 8 and abs(values.sum() - 100.0) < 1e-8
    assert data['value_axis'] == {'minimum': 0, 'maximum': 100, 'scale': 'linear'}
    W, H = brief['width'], brief['height']
    out = ROOT / 'artifacts'
    out.mkdir(exist_ok=True)

    # One projection for the source photograph, cloth union, identities and patterns.
    gx, gy, gw, gh = 445, 324, 956, 1434
    pw, ph = gw * AA, gh * AA
    src = Image.open(ROOT / 'assets/shirt_rgba.png').convert('RGBA').resize((pw, ph), Image.Resampling.LANCZOS)
    raw = np.array(src).astype(np.float32)
    opacity = raw[:, :, 3] / 255.0
    cloth_im = Image.open(ROOT / 'assets/cloth_mask.png').convert('L').resize((pw, ph), Image.Resampling.LANCZOS)
    S = np.asarray(cloth_im).astype(np.float32) / 255.0
    S = np.minimum(S, opacity)
    yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
    # Coordinates in the original generated garment, not the design-reference pixels.
    u = (xx + 0.5) * 1024 / pw
    v = (yy + 0.5) * 1536 / ph
    luminance = raw[:, :, 0] * 0.2126 + raw[:, :, 1] * 0.7152 + raw[:, :, 2] * 0.0722
    fold = gaussian_filter(luminance / 255, sigma=10 * AA)
    drape = (16 * np.sin((u - 125) / 192) + 5.0 * np.sin(u / 62)
             + 16 * (fold - 0.63) + 4 * np.sin(v / 240))
    q = v - drape
    visible = S > 0
    qflat = q[visible]
    wflat = S[visible].astype(np.float64)
    order = np.argsort(qflat, kind='stable')
    cdf = np.cumsum(wflat[order])
    targets = np.cumsum(values)[:-1] / 100 * cdf[-1]
    cuts = qflat[order[np.searchsorted(cdf, targets)]]
    ids = np.searchsorted(cuts, q, side='right')
    total_area = float(S.sum(dtype=np.float64)) / AA**2

    # Modulate palette/patterns with real generated cloth luminance.
    # Every detail (buttons, stitching, folds and cuffs) inherits its category.
    shade = np.clip((luminance / 163.0) ** 1.06, 0.26, 1.44)
    base = np.array([rgb(c) for c in PALETTE])[ids]
    tint = base.copy()
    # The weave coordinate gently follows the folds and the hanging sleeves.
    weave_u = u + 6.0 * np.sin(v / 140) + 12 * (fold - 0.63)
    weave_v = q
    ivory = rgb('#f0e7d7')
    coral = rgb('#bf7565')
    gold = rgb('#ad9a72')
    def blend_pattern(category, amount, colour):
        m = (ids == category)
        t = np.clip(amount[m], 0, 1)[:, None]
        tint[m] = tint[m] * (1 - t) + colour * t
    # Pinstripe and tartan are fabric treatments within area-controlled panels.
    stripe2 = np.exp(-((np.mod(weave_u + 2.9, 7.6) - 3.8) / 0.90) ** 4) * 0.79
    blend_pattern(1, stripe2, ivory)
    stripe3 = np.exp(-((np.mod(weave_u + 1.8, 12.5) - 6.25) / 2.10) ** 6) * 0.88
    blend_pattern(2, stripe3, ivory)
    stripe4 = np.exp(-((np.mod(weave_u, 7.0) - 3.5) / 0.66) ** 4) * 0.72
    blend_pattern(3, stripe4, gold)
    tartan_a = np.maximum(np.exp(-((np.mod(weave_u, 35) - 17.5) / 1.05) ** 4),
                          np.exp(-((np.mod(weave_v, 30) - 15) / 1.02) ** 4)) * 0.84
    blend_pattern(5, tartan_a, coral)
    tartan_b = np.maximum(np.exp(-((np.mod(weave_u + 7, 35) - 17.5) / 0.45) ** 4),
                          np.exp(-((np.mod(weave_v + 5, 30) - 15) / 0.45) ** 4)) * 0.51
    blend_pattern(5, tartan_b, gold)
    # Two subdued woven horizontal accents stay wholly inside the seventh category.
    t7 = (q - cuts[5]) / (cuts[6] - cuts[5])
    blend_pattern(6, np.exp(-((t7 - 0.31) / 0.090) ** 8) * 0.72, rgb('#d5b5bc'))
    blend_pattern(6, np.exp(-((t7 - 0.73) / 0.058) ** 8) * 0.83, rgb('#604b44'))
    # Very fine seam shading follows each computed boundary without removing cloth.
    seam_dist = np.min(np.abs(q[:, :, None] - cuts[None, None, :]), axis=2)
    seam_shade = 1 - 0.09 * np.exp(-(seam_dist / 0.74)**2)
    coloured = np.clip(tint * shade[:, :, None] * seam_shade[:, :, None], 0, 255)
    proportion = np.divide(S, opacity, out=np.zeros_like(S), where=opacity > 0)
    raw[:, :, :3] = raw[:, :, :3] * (1 - proportion[:, :, None]) + coloured * proportion[:, :, None]
    garment = Image.fromarray(np.rint(raw).astype(np.uint8), 'RGBA')

    # The image and the exact raster regions are saved for verifiable area checks.
    projected_S = Image.fromarray(np.rint(S * 255).astype(np.uint8)).resize((gw, gh), Image.Resampling.BOX)
    projected_S.save(ROOT / 'assets/visible_cloth_projected.png')
    band_info = []
    masks = []
    for i, (label, val) in enumerate(zip(cats, values)):
        band = S * (ids == i)
        mask = Image.fromarray(np.rint(band * 255).astype(np.uint8)).resize((gw, gh), Image.Resampling.BOX)
        mask.save(out / f'band_{i + 1:02d}.png')
        arr = np.asarray(mask).astype(np.float64) / 255
        masks.append(arr)
        row_area = arr.sum(axis=1)
        row = int(np.searchsorted(np.cumsum(row_area), arr.sum() / 2))
        xs = np.flatnonzero(arr[row] > 0.50)
        assert len(xs) > 0
        anchor_x, anchor_y = gx + int(xs[0]), gy + row
        share = float(band.sum(dtype=np.float64) / AA**2 / total_area * 100)
        band_info.append({'category': label, 'value_percent': float(val),
                          'visible_area_px': float(band.sum(dtype=np.float64) / AA**2),
                          'actual_area_percent': share,
                          'error_percentage_points': share - float(val),
                          'leader_anchor_canvas': [anchor_x, anchor_y],
                          'mask': f'artifacts/band_{i + 1:02d}.png'})
    del qflat, wflat, order, cdf, base, tint, coloured, xx, yy, ids, raw

    # A restrained warm-grey page, with a diffuse studio shadow behind the shirt.
    bw, bh = W * AA, H * AA
    yb, xb = np.mgrid[0:bh, 0:bw].astype(np.float32)
    vignette = np.exp(-(((xb / AA - 1000) / 605)**2 + ((yb / AA - 1080) / 1050)**2))
    canvas_arr = np.empty((bh, bw, 3), np.uint8)
    for c, v0 in enumerate([245, 244, 238]):
        canvas_arr[:, :, c] = np.clip(v0 - 6.8 * vignette, 0, 255).astype(np.uint8)
    page = Image.fromarray(canvas_arr, 'RGB').convert('RGBA')
    del canvas_arr, xb, yb, vignette
    shadow = Image.new('RGBA', (bw, bh), (0, 0, 0, 0))
    sm = src.getchannel('A').filter(ImageFilter.GaussianBlur(25 * AA))
    sm = sm.point(lambda x: round(x * 0.16))
    shadow.paste((72, 78, 76, 255), (gx * AA + 14 * AA, gy * AA + 20 * AA), sm)
    page.alpha_composite(shadow)
    page.alpha_composite(garment, (gx * AA, gy * AA))
    draw = ImageDraw.Draw(page)
    text_bounds = []
    def text_at(x, y, text, size=24, family='sans', fill=INK, bold=False, anchor='lt', track=0):
        f = font(size, family, bold)
        if track:
            assert anchor == 'lt'
            cursor = x * AA
            for ch in text:
                draw.text((cursor, y * AA), ch, font=f, fill=fill, anchor='lt')
                cursor += draw.textlength(ch, font=f) + track * AA
            bbox = (x * AA, y * AA, cursor, (y + size) * AA)
        else:
            draw.text((x * AA, y * AA), text, font=f, fill=fill, anchor=anchor)
            bbox = draw.textbbox((x * AA, y * AA), text, font=f, anchor=anchor)
        text_bounds.append({'text': text, 'bbox': [round(k / AA, 2) for k in bbox]})
        return bbox
    def line(x1,y1,x2,y2,fill=RULE,width=1):
        draw.line((round(x1*AA),round(y1*AA),round(x2*AA),round(y2*AA)),fill=fill,width=max(1,round(width*AA)))

    # Main typography is code-rendered, including the synthetic notice and all data.
    draw.rounded_rectangle((88*AA, 73*AA, 323*AA, 112*AA), radius=4*AA, fill='#a04d3d')
    text_at(103, 84, data['visible_notice'], 19, fill='#ffffff', bold=True, track=1.0)
    text_at(88, 145, 'Occupational', 98, family='serif')
    text_at(88, 244, 'composition', 98, family='serif')
    text_at(92, 375, 'Occupational groups and long-term unemployment', 26)
    text_at(92, 418, 'Share of the synthetic population (%)', 23, fill=MUTED)

    text_at(88, 565, 'OCCUPATIONAL GROUP', 16, bold=True, track=1.5)
    text_at(427, 565, '% OF TOTAL', 16, bold=True, anchor='rt')
    line(88, 599, 428, 599)
    wrapped = [
        ['Higher managerial', 'and professional'],
        ['Lower managerial', 'and professional'],
        ['Intermediate', 'occupations'],
        ['Small employers', 'and own account'],
        ['Lower supervisory', 'and technical'],
        ['Semi routine', 'occupations'],
        ['Routine', 'occupations'],
        ['Long term', 'unemployed'],
    ]
    assert [' '.join(lines) for lines in wrapped] == cats
    for i, info in enumerate(band_info):
        ax, ay = info['leader_anchor_canvas']
        # Straight leaders stay outside the garment, so they cannot conceal data area.
        line(448, ay, ax - 8, ay, fill='#959e97', width=1)
        draw.ellipse(((ax-8)*AA,(ay-2)*AA,(ax-4)*AA,(ay+2)*AA),fill=PALETTE[i])
        for j, part in enumerate(wrapped[i]):
            text_at(88, ay - 26 + 27*j, part, 24)
        text_at(430, ay - 21, f'{values[i]:.1f}%', 42, family='serif', anchor='rt')
    line(88, 1796, W - 88, 1796, fill='#b3bab2')
    text_at(88, 1830, 'One shirt,', 34, family='serif')
    text_at(88, 1870, 'one whole.', 34, family='serif')
    text_at(469, 1831, 'Coloured fabric area represents share, not the height of a band.', 24)
    text_at(469, 1869, 'The visible cloth totals 100%; the white collar and hanger are excluded.', 22, fill=MUTED)
    text_at(469, 1907, 'Illustrative synthetic values. No real population is described.', 21, fill=MUTED)
    text_at(88, 1934, f'{len(cats):02d} CATEGORIES', 15, bold=True, track=1.4, fill=MUTED)

    # Every complete image is emitted only here, for capture by the frozen wrapper.
    final = page.convert('RGB').resize((W,H), Image.Resampling.LANCZOS)
    final.save(ROOT / 'final.png', optimize=True)
    # Actual output-resolution masks and colour checks, not only target values.
    mask_total = sum(masks)
    area_rows = []
    for i, m in enumerate(masks):
        rendered_share = float(m.sum()/mask_total.sum()*100)
        # Compare painted pixels to the original garment where the region is opaque.
        actual_crop = np.asarray(final)[gy:gy+gh,gx:gx+gw].astype(np.float32)
        source_crop = np.asarray(src.resize((gw,gh),Image.Resampling.LANCZOS))[:,:,:3].astype(np.float32)
        opaque = m > 0.97
        chroma = np.ptp(actual_crop, axis=2)
        area_rows.append({**band_info[i], 'final_resolution_area_percent': rendered_share,
                          'final_resolution_error_pp': rendered_share - float(values[i]),
                          'opaque_pixels': int(opaque.sum()),
                          'mean_output_rgb': np.round(actual_crop[opaque].mean(axis=0),2).tolist(),
                          'mean_generated_to_recoloured_rgb_distance': round(float(np.linalg.norm(actual_crop[opaque]-source_crop[opaque],axis=1).mean()),3),
                          'median_final_chroma': round(float(np.median(chroma[opaque])),2)})
    write_json('artifacts/geometry_checks.json', {
        'canvas': [W,H], 'projection': {'x':gx,'y':gy,'width':gw,'height':gh,'antialias_scale':AA},
        'definition_zh': 'S为最终投影中的可见布料并集；直接采用照片可见表面，遮挡不重复计算；排除白领、衣架、挂钩及背景。',
        'quantile_boundaries_in_projected_garment_coordinate': [float(c) for c in cuts],
        'visible_cloth_area_final_px': total_area,
        'final_mask_union_area_px': float(mask_total.sum()),
        'mask_partition_max_pixel_sum_error': float(np.abs(mask_total - np.asarray(projected_S)/255).max()),
        'shares_sum': sum(x['final_resolution_area_percent'] for x in area_rows),
        'max_absolute_error_percentage_points': max(abs(x['final_resolution_error_pp']) for x in area_rows),
        'categories': area_rows,
        'text_bounds': text_bounds,
        'all_text_inside_canvas': all(b['bbox'][0]>=0 and b['bbox'][1]>=0 and b['bbox'][2]<=W and b['bbox'][3]<=H for b in text_bounds),
        'label_order_matches_data': [' '.join(a) for a in wrapped] == cats,
        'data_sha256': hashlib.sha256((ROOT/'data.json').read_bytes()).hexdigest(),
        'generated_raw_sha256': hashlib.sha256((ROOT/'assets/raw_01.png').read_bytes()).hexdigest(),
    })
    print(json.dumps({'canvas':[W,H], 'max_area_error_pp':max(abs(x['final_resolution_error_pp']) for x in area_rows),
                      'anchors': [x['leader_anchor_canvas'] for x in area_rows]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
