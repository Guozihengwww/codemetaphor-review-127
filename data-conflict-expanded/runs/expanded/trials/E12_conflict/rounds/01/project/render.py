"""Offline, data-bound rendering. Launch complete renders through capture_render.py."""
from pathlib import Path
import json
import math
from PIL import Image, ImageDraw, ImageFont, ImageChops
import numpy as np

ROOT = Path(__file__).resolve().parent
S = 2
ANGLE = 38.0
PX_PER_PERCENT = 16.0
PITCH = 66.0
TINE_WIDTH = 38.0
ORIGINS = [(600.0, 1100.0), (1360.0, 910.0)]
COLORS = [(21, 78, 154), (174, 24, 44)]
NAMES = ['blue', 'red']
INK = (29, 32, 38)
MUTED = (102, 111, 120)

def font(size, bold=False):
    return ImageFont.truetype(str(ROOT / 'assets' / ('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')), round(size*S))

def text(draw, xy, value, size=24, fill=INK, bold=False, anchor='lt'):
    draw.text(tuple(round(v*S) for v in xy), value, font=font(size, bold), fill=fill, anchor=anchor)

def fmt(v):
    return f'{v:g}%'

def point(x, y, origin):
    a = math.radians(ANGLE); c, s = math.cos(a), math.sin(a)
    return [origin[0]+c*x+s*y, origin[1]-s*x+c*y]

def label_image(category, value, color):
    f, b = font(24), font(26, True)
    gap = 10*S
    cw = round(f.getlength(category))
    vw = round(b.getlength(value))
    out = Image.new('RGBA', (cw+gap+vw+4*S, 37*S))
    d = ImageDraw.Draw(out)
    d.text((0, 2*S), category, font=f, fill=INK, anchor='lt')
    d.text((cw+gap, 1*S), value, font=b, fill=color, anchor='lt')
    return out

def make_fork(series, i, width, height):
    local_w, local_h = 720, 1560
    rx, ry = 360, 720
    local = Image.new('RGBA', (local_w*S, local_h*S))
    name = NAMES[i]
    origin = ORIGINS[i]
    body = Image.open(ROOT / 'assets' / f'{name}_body.png').convert('RGBA')
    local.alpha_composite(body, (round((rx-270)*S), round(ry*S)))
    material = Image.open(ROOT / 'assets' / f'{name}_tine_material.png').convert('RGB')
    # One shared length-to-value mapping, used identically for both forks.
    records = []
    for j, (category, value) in enumerate(zip(series['category_labels'], series['values'])):
        x = (j-(len(series['values'])-1)/2)*PITCH
        length = float(value)*PX_PER_PERCENT
        left, top = (rx+x-TINE_WIDTH/2)*S, (ry-length)*S
        if value > 0:
            w, h = round(TINE_WIDTH*S), round(length*S)
            srcx = round((270+x-TINE_WIDTH/2)*S)
            tile = material.crop((srcx, 0, srcx+w, material.height)).resize((w, h), Image.Resampling.BICUBIC)
            arr = np.asarray(tile, dtype=float)
            # Subtle bevel lighting is perpendicular to the data axis; endpoints do not move.
            q = np.linspace(0, 1, w)[None, :, None]
            lighting = 0.92 + 0.12*np.sin(q*np.pi) + 0.28*np.exp(-((q-0.83)/0.065)**2)
            arr = np.clip(arr*lighting, 0, 255).astype('uint8')
            tine = Image.fromarray(arr).convert('RGBA')
            m = Image.new('L', (w, h), 255)
            md = ImageDraw.Draw(m)
            # Tiny rounded top corners stay strictly inside the true length rectangle.
            r = 2*S
            md.rectangle((0, 0, w-1, r), fill=0)
            md.rounded_rectangle((0, 0, w-1, min(h-1, 4*r)), radius=r, fill=255)
            md.rectangle((0, r, w-1, h-1), fill=255)
            tine.putalpha(m)
            local.alpha_composite(tine, (round(left), round(top)))
        lab = label_image(category, fmt(value), COLORS[i])
        # Read from upper left toward the tip, ending 13 px before its true endpoint.
        rotated = lab.transpose(Image.Transpose.ROTATE_270)
        lx = round((rx+x)*S-rotated.width/2)
        ly = round((ry-length-13)*S-rotated.height)
        local.alpha_composite(rotated, (lx, ly))
        records.append({'series_id': series['id'], 'category': category, 'value': value,
                        'local_x': x, 'length_px': length,
                        'root': point(x, 0, origin), 'tip': point(x, -length, origin),
                        'label_bounds_local': [lx/S-rx, ly/S-ry, (lx+rotated.width)/S-rx, (ly+rotated.height)/S-ry]})

    ld = ImageDraw.Draw(local)
    # The ruler shares the tine axis. The 0 tick lies on the common root line.
    ax = rx-287
    ld.line((round(ax*S), round((ry-320)*S), round(ax*S), round(ry*S)), fill=(175, 182, 190, 255), width=2)
    for val in [0, 10, 20]:
        yy = ry-PX_PER_PERCENT*val
        ld.line((round((ax-6)*S), round(yy*S), round((ax+6)*S), round(yy*S)), fill=(142, 151, 161, 255), width=2)
        text(ld, (ax-12, yy), f'{val}', size=17, fill=MUTED, anchor='rm')
    # Same zero guide for all category slots, without adding a mark at the zero-value slot.
    for x in np.arange(-258, 259, 12):
        ld.line((round((rx+x)*S), round(ry*S), round((rx+x+6)*S), round(ry*S)), fill=(244, 249, 255, 220), width=2)

    handle_label = 'TABLE SERVICE' if i == 0 else 'QUICK SERVICE'
    hf = font(24, True)
    im = Image.new('RGBA', (round(hf.getlength(handle_label))+4*S, 34*S))
    ImageDraw.Draw(im).text((0, 2*S), handle_label, font=hf, fill=(255,255,255,238), anchor='lt')
    im = im.transpose(Image.Transpose.ROTATE_270)
    local.alpha_composite(im, (round(rx*S-im.width/2), round((ry+402)*S)))

    a = math.radians(ANGLE); c, s = math.cos(a), math.sin(a)
    inverse = (c, -s, S*(rx-c*origin[0]+s*origin[1]),
               s, c, S*(ry-s*origin[0]-c*origin[1]))
    layer = local.transform((width*S, height*S), Image.Transform.AFFINE, inverse, Image.Resampling.BICUBIC)
    return layer, records

def main():
    inputs = json.loads((ROOT / 'input.json').read_text(encoding='utf-8'))
    brief = json.loads((ROOT / inputs['brief']).read_text(encoding='utf-8'))
    data = json.loads((ROOT / inputs['data']).read_text(encoding='utf-8'))
    core = json.loads((ROOT / inputs['critical_metaphor_contract']).read_text(encoding='utf-8'))
    assert (ROOT / inputs['reference']).is_file()
    assert core['topic_zh'] == data['topic_zh']
    assert data['value_axis'] == {'minimum': 0, 'maximum': 20, 'scale': 'linear'}
    assert len(data['series']) == 2
    for series in data['series']:
        assert len(series['values']) == len(series['category_labels']) == len(data['categories'])
        assert all(0 <= v <= 20 for v in series['values'])
    W, H = brief['width'], brief['height']
    canvas = Image.new('RGBA', (W*S, H*S), 'white')
    d = ImageDraw.Draw(canvas)
    d.rounded_rectangle((90*S, 48*S, 384*S, 88*S), radius=6*S, fill=INK)
    text(d, (108, 57), data['visible_notice'], 22, 'white', True)
    text(d, (86, 119), 'Restaurant', 91, INK, True)
    text(d, (86, 221), 'spending change', 91, INK, True)
    text(d, (92, 363), 'Each tine shows one restaurant category.', 28, MUTED)
    text(d, (92, 406), 'Length from the 0% root encodes spending change.', 25, MUTED)

    text(d, (1368, 61), 'TWO SERVICE TYPES', 20, MUTED, True)
    for i, title in enumerate(['Table service', 'Quick service']):
        y = 117+i*88
        d.rounded_rectangle((1368*S, (y+2)*S, 1376*S, (y+58)*S), radius=4*S, fill=COLORS[i])
        text(d, (1396, y), title, 29, COLORS[i], True)
        text(d, (1396, y+36), 'Spending change (%)', 23, MUTED)
    text(d, (1368, 310), 'Shared linear scale: 0–20%', 23, MUTED)

    geometry = []
    for i, series in enumerate(data['series']):
        layer, records = make_fork(series, i, W, H)
        canvas.alpha_composite(layer)
        geometry.extend(records)
    d = ImageDraw.Draw(canvas)
    d.line((90*S, 1761*S, 1910*S, 1761*S), fill=(220,224,229), width=2)
    text(d, (90, 1782), 'Illustrative synthetic values • Categories kept in the supplied order', 20, MUTED)
    text(d, (1910, 1782), 'SPENDING CHANGE  /  %', 18, MUTED, True, 'rt')
    output = canvas.convert('RGB').resize((W, H), Image.Resampling.LANCZOS)
    output.save(ROOT / 'final.png', optimize=True)
    artifact = ROOT / 'artifacts'; artifact.mkdir(exist_ok=True)
    (artifact / 'geometry.json').write_text(json.dumps({
        'canvas_px': [W,H], 'rotation_degrees': ANGLE,
        'px_per_percentage_point': PX_PER_PERCENT,
        'tine_width_px': TINE_WIDTH, 'origins': ORIGINS,
        'records': geometry,
    }, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('Rendered final.png from frozen data and local generated materials.')

if __name__ == '__main__':
    main()
