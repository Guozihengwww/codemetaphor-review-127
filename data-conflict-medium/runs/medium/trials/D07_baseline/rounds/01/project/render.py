"""Reproduce the local infographic from frozen data and generated material.

Run: python render.py
For experimental captures, use the supplied capture_render.py wrapper.
No network or model calls. The final projected fabric union is counted once.
"""
from pathlib import Path
import json
import math
import hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'assets'
ART = ROOT / 'artifacts'
FONT = ASSETS / 'fonts'
ORIGIN = (342, 257)
PALETTE = [
    '#647c99', '#b7c8d6', '#cc8c7b', '#4b6586',
    '#899888', '#473339', '#c2bcb5', '#bd4c36',
]
INK = '#26332f'
MUTED = '#68706a'


def rgb(value):
    return np.array([int(value[i:i+2], 16) for i in (1, 3, 5)], dtype=np.float32)


def smoothstep(value):
    v = np.clip(value, 0, 1)
    return v * v * (3 - 2 * v)


def stripe(coord, period, width, edge=0.5):
    d = np.abs((coord + period / 2) % period - period / 2)
    return 1 - smoothstep((d - width / 2 + edge) / (2 * edge))


def fabric_coordinates(source):
    """A single projected cloth coordinate map for color, texture and seams."""
    h, w = source.shape[:2]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    lum = source[:, :, :3] @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    softened = gaussian_filter(lum, 8)
    broad = gaussian_filter(lum, 35)
    local_fold = np.clip((softened - broad) / 8, -2.5, 2.5)
    # The two sleeve edge curves were traced on the generated photograph.
    left_edge = np.interp(yy[:, 0], [240, 400, 580, 810, 970, 1100, 1260, 1330, 1450],
                          [345, 200, 242, 310, 265, 235, 231, 224, 230])[:, None]
    right_edge = np.interp(yy[:, 0], [270, 400, 600, 820, 1080, 1260, 1340, 1470],
                           [676, 846, 818, 816, 811, 821, 847, 870])[:, None]
    left = (xx < left_edge) & (yy > 340)
    right = (xx > right_edge) & (yy > 360)
    body_drape = 13 * np.sin((xx - 290) / 530 * np.pi) + 4 * np.sin(xx / 44)
    displacement = body_drape + 3.2 * local_fold
    displacement = np.where(left, 25 + 0.16 * (210 - xx) + 5 * np.sin(yy / 90) + 2.7 * local_fold, displacement)
    displacement = np.where(right, 8 + 0.15 * (xx - 830) + 4 * np.sin(yy / 130) + 3 * local_fold, displacement)
    q = yy - displacement
    u = xx + 3.8 * local_fold + 3 * np.sin(yy / 110)
    u = np.where(left, u - 0.09 * (yy - 600), u)
    u = np.where(right, u + 0.075 * (yy - 600), u)
    return q, u, lum


def calibrated_thresholds(q, fabric_alpha, shares):
    valid = fabric_alpha > 0
    order = np.argsort(q[valid], kind='stable')
    coordinates = q[valid][order]
    weights = fabric_alpha[valid][order]
    cumulative = np.cumsum(weights, dtype=np.float64)
    limits = []
    for share in np.cumsum(shares)[:-1] / 100:
        index = int(np.searchsorted(cumulative, share * cumulative[-1]))
        limits.append(float((coordinates[index] + coordinates[min(index+1, len(coordinates)-1)]) / 2))
    return np.array(limits, dtype=np.float32)


def pattern_color(index, q, u, xx, yy):
    """Category-specific fabrics. All pattern pixels inherit their category."""
    base = np.broadcast_to(rgb(PALETTE[index]), (*q.shape, 3)).copy()
    if index == 1:
        line = stripe(u, 8.5, 1.5, .35)
        base = base * (1 - line[:, :, None]) + rgb('#6c829a') * line[:, :, None]
    elif index == 2:
        line = stripe(u, 13.2, 3.4, .45)
        base = base * (1 - line[:, :, None]) + rgb('#eedbc9') * line[:, :, None]
    elif index == 3:
        line = stripe(u, 7.5, 1.0, .3)
        base = base * (1 - .65 * line[:, :, None]) + rgb('#b6bdb2') * .65 * line[:, :, None]
    elif index == 4:
        twill = np.sin((u + q * .65) * 2.2)
        base *= (1 + twill[:, :, None] * .019)
    elif index == 5:
        vertical = stripe(u, 37, 1.7, .4)
        horizontal = stripe(q, 42, 1.5, .4)
        fine = np.maximum(stripe(u + 5, 37, .6, .28), stripe(q + 5, 42, .6, .28))
        line = np.maximum(vertical, horizontal)
        base = base * (1 - line[:, :, None] * .81) + rgb('#b77865') * line[:, :, None] * .81
        base = base * (1 - fine[:, :, None] * .35) + rgb('#d5b9a0') * fine[:, :, None] * .35
    elif index == 6:
        # Delicate horizontal yarn lines, not extra broad quantitative bands.
        line = stripe(q, 17.0, 1.25, .4)
        base = base * (1 - line[:, :, None] * .18) + rgb('#ac777b') * line[:, :, None] * .18
    elif index == 7:
        ribs = .012 * np.sin(u * 2.6)
        base *= 1 + ribs[:, :, None]
    return base


def font(name, size):
    return ImageFont.truetype(str(FONT / name), round(size * 2))


class TypeLayer:
    def __init__(self, size):
        self.image = Image.new('RGBA', (size[0] * 2, size[1] * 2))
        self.draw = ImageDraw.Draw(self.image)
        self.boxes = []

    def text(self, xy, label, size=26, face='EditorialSans.otf', fill=INK, anchor=None, track=None, record=True):
        xy = tuple(round(v * 2) for v in xy)
        f = font(face, size)
        if track is None:
            self.draw.text(xy, label, font=f, fill=fill, anchor=anchor)
            box = self.draw.textbbox(xy, label, font=f, anchor=anchor)
        else:
            x, y = xy
            start = x
            for char in label:
                self.draw.text((x, y), char, font=f, fill=fill)
                x += self.draw.textlength(char, font=f) + track * 2
            box = (start, y, x - track * 2, y + size * 2)
        if record:
            self.boxes.append({'text': label, 'box': [round(v/2, 2) for v in box]})
        return box

    def line(self, points, fill, width=1):
        self.draw.line([(round(x * 2), round(y * 2)) for x, y in points], fill=fill, width=max(1,round(width*2)))

    def circle(self, box, fill, outline=None, width=1):
        self.draw.ellipse(tuple(round(v*2) for v in box), fill=fill, outline=outline, width=max(1,round(width*2)))

    def finish(self):
        return self.image.resize((self.image.width // 2, self.image.height // 2), Image.Resampling.LANCZOS)


def main():
    ART.mkdir(exist_ok=True)
    data = json.loads((ROOT / 'data.json').read_text(encoding='utf-8'))
    brief = json.loads((ROOT / 'brief.json').read_text(encoding='utf-8'))
    width, height = brief['width'], brief['height']
    names = data['categories']
    values = np.array(data['series'][0]['values'], dtype=float)
    assert len(names) == len(values) == 8
    assert np.isclose(values.sum(), 100), 'Area encoding requires the supplied shares to sum to 100.'
    assert data['value_axis'] == {'minimum':0, 'maximum':100, 'scale':'linear'}
    source = np.array(Image.open(ASSETS / 'garment_substrate.png').convert('RGBA'), dtype=np.float32)
    fabric = np.array(Image.open(ASSETS / 'fabric_alpha.png').convert('L'), dtype=np.float32) / 255
    membership = np.array(Image.open(ASSETS / 'fabric_membership.png').convert('L'), dtype=np.float32) / 255
    q, u, lum = fabric_coordinates(source)
    thresholds = calibrated_thresholds(q, fabric, values)
    classes = np.searchsorted(thresholds, q, side='right').astype(np.uint8)
    yy, xx = np.mgrid[:q.shape[0], :q.shape[1]].astype(np.float32)
    cumulative = [np.ones(q.shape, dtype=np.float32)]
    cumulative += [smoothstep((q-t) / .8 + .5) for t in thresholds]
    cumulative += [np.zeros(q.shape, dtype=np.float32)]
    coverage = [cumulative[i] - cumulative[i+1] for i in range(8)]
    colored = np.zeros((*q.shape, 3), dtype=np.float32)
    for i in range(8):
        colored += pattern_color(i, q, u, xx, yy) * coverage[i][:, :, None]
    # Photographic luminance supplies all major folds, occlusion shadows,
    # seam details, knitted cuffs and hem, rather than a synthetic flat shirt.
    shade = np.clip(lum / 170, .20, 1.38)
    colored *= shade[:, :, None]
    seam_shadow = np.zeros(q.shape, dtype=np.float32)
    seam_highlight = np.zeros(q.shape, dtype=np.float32)
    stitch = np.zeros(q.shape, dtype=np.float32)
    for t in thresholds:
        d = q-t
        seam_shadow = np.maximum(seam_shadow, np.exp(-(d/1.1)**2) * .16)
        seam_highlight = np.maximum(seam_highlight, np.exp(-((d-1.55)/.8)**2) * .045)
        stitch = np.maximum(stitch, np.exp(-((d-3.0)/.48)**2) * stripe(u, 5.5, 2.1, .3) * .065)
    colored *= (1 - seam_shadow + seam_highlight + stitch)[:, :, None]
    colored = np.clip(colored, 0, 255)
    composite_source = source.copy()
    composite_source[:, :, :3] = source[:, :, :3] * (1-membership[:, :, None]) + colored * membership[:, :, None]
    garment = Image.fromarray(np.uint8(np.round(composite_source.clip(0, 255))), 'RGBA')

    # Warm neutral editorial background with a very restrained studio falloff.
    gy, gx = np.mgrid[:height, :width].astype(np.float32)
    glow = np.exp(-(((gx-510)/850)**2 + ((gy-700)/1500)**2))
    bg = rgb('#e9e6de')[None, None, :] + glow[:, :, None] * np.array([9,9,8])
    canvas = Image.fromarray(np.uint8(np.round(bg.clip(0,255))), 'RGB').convert('RGBA')
    shadow = Image.new('L', (width,height))
    shadow.paste(Image.fromarray(np.uint8(source[:,:,3])), (ORIGIN[0]+16, ORIGIN[1]+18))
    shadow = shadow.filter(ImageFilter.GaussianBlur(22)).point(lambda p: round(p * .12))
    dark = Image.new('RGBA',(width,height),'#51534b')
    canvas.alpha_composite(Image.composite(dark, Image.new('RGBA',(width,height)), shadow))
    canvas.alpha_composite(garment, ORIGIN)

    typography = TypeLayer((width, height))
    typography.text((88,55), data['visible_notice'], 23, 'EditorialSansBold.otf', track=2.7)
    typography.text((1372,58), 'SHARE  /  PERCENT', 21, fill=MUTED, anchor='ra')
    typography.text((84,106), 'Synthetic', 68, 'EditorialSerifItalic.otf')
    typography.text((404,99), '衣服布面分层', 65, 'NotoSerifSC.otf')
    typography.text((88,204), '各色可见布料面积，就是该组份额。', 27, 'NotoSansSC.otf', MUTED)
    typography.line([(88,266),(326,266)], '#acafa5', 1)
    typography.text((88,313), 'AREA = SHARE', 21, 'EditorialSansBold.otf', track=1.3)
    typography.text((88,356), '按输入顺序', 23, 'NotoSansSC.otf', MUTED)
    typography.text((88,392), '从上至下排列', 23, 'NotoSansSC.otf', MUTED)
    typography.text((88,474), '类别', 20, 'NotoSansSC.otf', MUTED)
    typography.text((310,480), 'SHARE (%)', 19, 'EditorialSans.otf', MUTED, anchor='ra')

    anchors = []
    for i, (name, value) in enumerate(zip(names, values)):
        weight = fabric * coverage[i]
        rows = weight.sum(axis=1)
        center = int(np.searchsorted(np.cumsum(rows), rows.sum()/2))
        # A horizontal external leader terminates outside the fabric, and its
        # nearby visible pixel is explicitly checked to belong to its category.
        options = []
        for y in range(max(0,center-45), min(q.shape[0],center+46)):
            points = np.flatnonzero(fabric[y] > .98)
            if len(points):
                x = int(points[0])
                if classes[y,min(x+6,q.shape[1]-1)] == i and coverage[i][y,x+6] > .99:
                    options.append((abs(y-center),y,x))
        if not options:
            raise ValueError(f'No external label anchor for {name}')
        _, ay, ax = min(options)
        px, py = ax+ORIGIN[0], ay+ORIGIN[1]
        label_y = py
        typography.circle((88,label_y-35,96,label_y-27),PALETTE[i])
        typography.text((110,label_y-48),name,29,'EditorialSans.otf')
        typography.text((109,label_y-7),f'{value:.1f}',49,'EditorialSerif.otf')
        number_width = typography.draw.textlength(f'{value:.1f}', font=font('EditorialSerif.otf',49))/2
        typography.text((115+number_width,label_y+10),'%',25,'EditorialSerif.otf',MUTED)
        typography.line([(282,label_y+17),(px-17,label_y+17),(px-8,label_y)], '#8c948b',1.25)
        typography.circle((px-12,label_y-3,px-6,label_y+3),PALETTE[i])
        anchors.append({'category':name,'value_label':f'{value:.1f}%',
                        'external_dot_xy':[px-9,py],
                        'fabric_probe_xy':[px+6,py],
                        'probe_class':int(classes[ay,ax+6]),
                        'label_center_y':label_y})

    typography.line([(88,1806),(1372,1806)], '#b6b8ad',1)
    typography.text((88,1841),'面积即份额',26,'NotoSerifSC.otf')
    typography.text((343,1839),'白领、衣架与背景不计入；袖身重叠只计算前方一次。',24,'NotoSansSC.otf',MUTED)
    typography.text((343,1880),'颜色与纹理区分类别。单位：百分比（%）。',24,'NotoSansSC.otf',MUTED)
    typography.text((1372,1941),f'TOTAL  {values.sum():.0f}%   /   {len(values):02d} GROUPS',20,'EditorialSans.otf',MUTED,anchor='ra')
    typography.text((88,1941),'SYNTHETIC DATA',19,'EditorialSansBold.otf',MUTED,track=1.5)
    canvas.alpha_composite(typography.finish())
    final = canvas.convert('RGB')
    final.save(ROOT / 'final.png')

    total = float(fabric.sum(dtype=np.float64))
    geometry = []
    for i in range(8):
        hard_area = float(fabric[classes==i].sum(dtype=np.float64))
        blended_area = float((fabric*coverage[i]).sum(dtype=np.float64))
        ys, xs = np.nonzero((classes==i) & (fabric>.5))
        geometry.append({'category':names[i], 'supplied_percent':float(values[i]),
                         'measured_visible_area_px':round(hard_area,5),
                         'measured_percent':round(hard_area/total*100,6),
                         'antialiased_measured_percent':round(blended_area/total*100,6),
                         'error_percentage_points':round(blended_area/total*100-values[i],6),
                         'visible_y_range':[int(ys.min()+ORIGIN[1]),int(ys.max()+ORIGIN[1])],
                         'pixel_count_support':int(len(xs))})
    ids = np.where(fabric>0,classes+1,0).astype('uint8')
    Image.fromarray(ids,'L').save(ART / 'visible_category_ids.png')
    Image.fromarray(np.uint8(np.round(fabric*255)),'L').save(ART / 'visible_fabric_alpha.png')

    # Verify the saved visible pixels against the actual composited substrate.
    # Only opaque fabric interiors are tested, avoiding antialias edge mixtures.
    saved = np.asarray(Image.open(ROOT/'final.png').convert('RGB'))
    crop = saved[ORIGIN[1]:ORIGIN[1]+q.shape[0], ORIGIN[0]:ORIGIN[0]+q.shape[1]]
    probe_mask = (fabric > .999) & (membership > .999)
    expected = np.asarray(garment)[:,:,:3]
    pixel_diff = np.max(np.abs(crop.astype(np.int16)-expected.astype(np.int16)),axis=2)
    changed_fabric_pixels = int(np.sum((pixel_diff>1)&probe_mask))
    checks = {
        'canvas':[width,height], 'raw_asset_size':[source.shape[1],source.shape[0]],
        'object_origin':list(ORIGIN), 'object_scale':1.0,
        'fabric_union_area_px':round(total,5), 'category_count':len(names),
        'source_values':values.tolist(), 'sum_percent':float(values.sum()),
        'calibrated_q_thresholds':thresholds.tolist(),
        'mapping':'One cloth coordinate field; weighted quantiles on the final projected fabric alpha. Excludes hanger, hook, white collar and background. Every projected pixel counted once.',
        'geometry':geometry, 'labels':anchors,
        'label_boxes':typography.boxes,
        'max_abs_error_percentage_points':max(abs(g['error_percentage_points']) for g in geometry),
        'uncovered_fabric_px':float(np.sum(fabric*np.abs(sum(coverage)-1))),
        'saved_image_interior_fabric_pixels_checked':int(probe_mask.sum()),
        'saved_image_interior_fabric_pixels_changed_by_overlays':changed_fabric_pixels,
        'saved_image_sha256':hashlib.sha256((ROOT/'final.png').read_bytes()).hexdigest(),
    }
    (ART / 'geometry_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'canvas':[width,height], 'max_abs_area_error_pp':checks['max_abs_error_percentage_points'],
                      'fabric_overlay_pixels':changed_fabric_pixels,'source_values':values.tolist()}))


if __name__ == '__main__':
    main()
