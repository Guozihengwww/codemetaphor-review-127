"""Rebuild the infographic from local data and generated cloth, without a network.

Run complete renders through capture_render.py during this experiment.
All colored boundaries use one affine transform and a 30% white exclusion.
"""
from pathlib import Path
from decimal import Decimal
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent
S = 2
PALETTE = ['#138985', '#354049', '#2879a2', '#596fb0',
           '#89669e', '#ae655f', '#c08a43', '#d4b55f']
INK = '#253739'
SECONDARY = '#647170'


def rgb(s):
    return tuple(int(s[i:i+2], 16) for i in (1, 3, 5))


def font(size, bold=False, cjk=False):
    name = 'NotoSansCJK-Regular.ttc' if cjk else ('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')
    return ImageFont.truetype(str(ROOT / 'assets/fonts' / name), round(size*S))


def txt(canvas, xy, value, size, color=INK, bold=False, anchor='la', cjk=False):
    d = ImageDraw.Draw(canvas)
    d.text((round(xy[0]*S), round(xy[1]*S)), value,
           font=font(size, bold, cjk), fill=color, anchor=anchor)


def line(canvas, xy, fill, width=1):
    ImageDraw.Draw(canvas).line(tuple(round(v*S) for v in xy), fill=fill, width=max(1, round(width*S)))


def vertical_value(canvas, cx, cy, value, size, color):
    f = font(size, bold=True)
    bb = f.getbbox(value)
    tile = Image.new('RGBA', (bb[2]-bb[0]+12*S, bb[3]-bb[1]+12*S))
    ImageDraw.Draw(tile).text((6*S-bb[0], 6*S-bb[1]), value, font=f, fill=color)
    tile = tile.transpose(Image.Transpose.ROTATE_90)
    canvas.alpha_composite(tile, (round(cx*S-tile.width/2), round(cy*S-tile.height/2)))


def main():
    brief = json.loads((ROOT / 'brief.json').read_text(encoding='utf-8'))
    data = json.loads((ROOT / 'data.json').read_text(encoding='utf-8'), parse_float=Decimal)
    W, H = int(brief['width']), int(brief['height'])
    assert (W, H) == (2000, 1400)
    values = [Decimal(str(v)) for v in data['series'][0]['values']]
    cats = data['categories']
    assert len(values) == len(cats) == 8 and sum(values) == Decimal(100)
    assert data['value_axis'] == {'minimum': 0, 'maximum': 100, 'scale': 'linear'}
    labels = [format(v, 'f') + data['series'][0]['unit'] for v in values]
    # Light neutral seamless studio ground.
    yy = np.linspace(0, 1, H*S, dtype=np.float32)[:, None, None]
    top = np.array([241, 242, 239], np.float32)[None, None, :]
    bottom = np.array([230, 232, 228], np.float32)[None, None, :]
    bg = np.broadcast_to(top*(1-yy) + bottom*yy, (H*S, W*S, 3)).astype('uint8').copy()
    canvas = Image.fromarray(bg).convert('RGBA')
    del bg

    # Editorial header; all semantic text is code controlled.
    txt(canvas, (120, 47), data['title'], 64, bold=True, cjk=True)
    txt(canvas, (125, 152), data['series'][0]['label'] + ' (%)', 26, SECONDARY)
    txt(canvas, (1880, 157), '8 groups  /  Total 100%', 22, SECONDARY, anchor='ra')
    d = ImageDraw.Draw(canvas)
    d.rounded_rectangle((1498*S, 67*S, 1880*S, 119*S), radius=26*S,
                        outline=rgb('#83918d'), width=1*S)
    txt(canvas, (1689, 92), data['visible_notice'], 22, INK, bold=True, anchor='mm')
    line(canvas, (120, 207, 1880, 207), '#c9ceca')

    # One affine resize for all object pixels and the alpha silhouette.
    px, py, pw, ph = 120, 245, 1760, 878
    asset = Image.open(ROOT / 'assets/pillow_cloth.png').convert('RGBA')
    pillow = asset.resize((pw*S, ph*S), Image.Resampling.LANCZOS)
    rgba = np.array(pillow)
    alpha = rgba[:, :, 3]
    visible_y, visible_x = np.where(alpha >= 128)
    ymid = (int(visible_y.min()) + int(visible_y.max()) + 1)//2
    row = np.flatnonzero(alpha[ymid] >= 128)
    left, right = int(row[0]), int(row[-1])+1
    width = right-left
    data_left = left + Decimal('0.30')*width
    data_width = Decimal('0.70')*width
    cumulative = [Decimal(0)]
    for v in values:
        cumulative.append(cumulative[-1] + v)
    boundaries = [float(data_left+data_width*c/100) for c in cumulative]

    # Studio floor shadow, derived from the generated lower silhouette.
    shadow = Image.new('RGBA', canvas.size)
    sd = ImageDraw.Draw(shadow)
    sd.ellipse((210*S, 1074*S, 1800*S, 1156*S), fill=(56, 62, 54, 46))
    shadow = shadow.filter(ImageFilter.GaussianBlur(21*S))
    canvas.alpha_composite(shadow)
    contact = Image.new('RGBA', canvas.size)
    cd = ImageDraw.Draw(contact)
    cd.ellipse((220*S, 1097*S, 1790*S, 1127*S), fill=(43, 49, 40, 57))
    contact = contact.filter(ImageFilter.GaussianBlur(9*S))
    canvas.alpha_composite(contact)
    del shadow, contact

    # Recolor the generated cotton without changing its shape or folds.
    # Adjacent half-open intervals cover the entire cloth without gutters.
    xx = np.arange(pw*S, dtype=np.float32) + .5
    ownership = np.full(pw*S, -1, np.int16)
    for i in range(len(values)):
        ownership[(xx >= boundaries[i]) & (xx < boundaries[i+1])] = i
    # Projecting outer corners inherit the nearest outer region; the encoding
    # itself is measured only across the specified middle horizontal section.
    ownership[xx >= boundaries[-1]] = len(values)-1
    original = rgba[:, :, :3].astype(np.float32)
    lum = (.2126*original[:, :, 0] + .7152*original[:, :, 1] + .0722*original[:, :, 2])
    shade = np.clip(lum/237, 0, 1.12)**1.65
    for i, c in enumerate(PALETTE):
        columns = ownership == i
        base = np.array(rgb(c), np.float32)
        painted = shade[:, columns, None]*base
        # A restrained diffuse highlight keeps the weave and wrinkles visible.
        highlights = np.maximum(lum[:, columns]-231, 0)[:, :, None]*.32
        original[:, columns, :] = np.clip(painted+highlights, 0, 255)
    rgba[:, :, :3] = original.astype('uint8')
    colored = Image.fromarray(rgba)
    canvas.alpha_composite(colored, (px*S, py*S))

    # Save a data mask for reproducible geometry audit, not as an alternate chart.
    audit_mask = np.zeros((H*S, W*S), dtype='uint8')
    indexed = np.broadcast_to((ownership+2).astype('uint8'), alpha.shape).copy()
    indexed[alpha < 128] = 0
    audit_mask[py*S:(py+ph)*S, px*S:(px+pw)*S] = indexed
    Image.fromarray(audit_mask).resize((W, H), Image.Resampling.NEAREST).save(ROOT / 'artifacts/geometry_mask.png')
    Image.fromarray(alpha).resize((pw, ph), Image.Resampling.LANCZOS).save(ROOT / 'assets/pillow_render_alpha.png')

    centers = [px + (boundaries[i]+boundaries[i+1])/(2*S) for i in range(8)]
    mid_y = py+ymid/S
    # Exact numerical labels are printed directly onto their own fabric stripe.
    white_cx = px+(left+float(data_left))/(2*S)
    txt(canvas, (white_cx, 660), '白布装饰区', 29, '#7d8986', anchor='mm', cjk=True)
    txt(canvas, (white_cx, 704), '30% · 非数据', 24, '#87928f', anchor='mm', cjk=True)
    txt(canvas, (centers[0], 604), cats[0], 33, '#eef8f3', anchor='mm')
    txt(canvas, (centers[0], 692), labels[0], 88, '#ffffff', bold=True, anchor='mm')
    for i in range(1, 8):
        color = '#fffdf6' if i < 6 else '#352f25'
        vertical_value(canvas, centers[i], 675, labels[i], 33, color)
    for i, cx in enumerate(centers):
        color = '#eff9f3' if i < 6 else '#403b2e'
        d = ImageDraw.Draw(canvas)
        d.ellipse(((cx-19)*S, 854*S, (cx+19)*S, 892*S), outline=rgb(color), width=1*S)
        txt(canvas, (cx, 873), f'{i+1:02}', 17, color, bold=True, anchor='mm')

    # A keyed, ordered legend repeats exact values at a comfortable reading size.
    for i, (cat, label, c) in enumerate(zip(cats, labels, PALETTE)):
        x = 120 + 220*i
        line(canvas, (x, 1190, x+191, 1190), c, 4)
        txt(canvas, (x, 1206), cat, 24, SECONDARY)
        txt(canvas, (x, 1246), label, 30, INK, bold=True)
    line(canvas, (120, 1310, 1880, 1310), '#c1c8c1')
    note = '读图：左侧 30% 白布不计入数据分母；其余布面按份额划分，只有中段横向宽度编码数值。'
    txt(canvas, (120, 1330), note, 23, SECONDARY, cjk=True)

    final = canvas.convert('RGB').resize((W, H), Image.Resampling.LANCZOS)
    final.save(ROOT / 'final.png', optimize=True)
    geometry = {
        'canvas_px': [W, H], 'supersampling': S,
        'source_asset': 'assets/pillow_cloth.png', 'asset_canvas_transform_xywh': [px, py, pw, ph],
        'measurement_alpha_threshold': 128,
        'midline_y_px': mid_y,
        'visible_midline_left_px': px+left/S,
        'visible_midline_right_px': px+right/S,
        'visible_midline_width_px': width/S,
        'white_excluded_width_px': float(Decimal('.30')*width)/S,
        'data_denominator_width_px': float(data_width)/S,
        'boundaries_x_px': [px+b/S for b in boundaries],
        'bands': [{'category': cat, 'value_text': label,
                   'expected_share_percent': float(value),
                   'x_start_px': px+boundaries[i]/S,
                   'x_end_px': px+boundaries[i+1]/S,
                   'center_x_px': centers[i], 'color': PALETTE[i]}
                  for i, (cat, label, value) in enumerate(zip(cats, labels, values))],
        'axis': {'minimum': 0, 'maximum': 100, 'scale': 'linear'},
        'encoding_zh': '在可见布面中间水平截面测端点，排除左侧30%白布，其余宽度按份额连续划分。边界竖直；面积和皱褶不编码。',
        'labels': labels, 'category_labels': cats,
    }
    (ROOT / 'artifacts/render_geometry.json').write_text(json.dumps(geometry, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('Rendered final.png:', W, H, 'data width:', round(float(data_width)/S, 3))


if __name__ == '__main__':
    main()
