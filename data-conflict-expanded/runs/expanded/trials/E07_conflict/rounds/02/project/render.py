"""Offline pillow infographic. Run only through capture_render.py for trials."""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent
S = 2
COLORS = ['#52616A', '#83BEC3', '#43A79F', '#177D70', '#176FA0', '#E4AC40']
LABEL_COLORS = ['#43555F', '#35747C', '#257B73', '#176A60', '#176FA0', '#926819']
INK = '#263940'
MUTED = '#647278'

def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))

def rgb(color):
    return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))

def fmt(value):
    return f'{value:g}'

def main():
    contract = read('input.json')
    brief = read(contract['brief'])
    data = read(contract['data'])
    core = read(contract['critical_metaphor_contract'])
    assert (ROOT / contract['reference']).is_file()
    assert data['topic_zh'] == core['topic_zh']
    values = np.asarray(data['series'][0]['values'], dtype=float)
    categories = data['categories']
    assert len(values) == len(categories) == 6
    assert abs(values.sum() - 100) < 1e-8
    assert data['value_axis'] == {'minimum': 0, 'maximum': 100, 'scale': 'linear'}
    width, height = brief['width'], brief['height']
    canvas = Image.new('RGB', (width * S, height * S), '#EFF1F1')
    # Quiet neutral studio backdrop, slightly darker toward the floor.
    background = np.empty((height * S, width * S, 3), dtype=np.uint8)
    for y in range(height * S):
        t = y / (height * S)
        base = np.array([243., 245., 245.]) - 7 * t
        background[y, :, :] = np.round(base).astype('uint8')
    canvas = Image.fromarray(background).convert('RGBA')
    d = ImageDraw.Draw(canvas)
    fonts = {}
    text_boxes = []
    def font(size, bold=False, cjk=False):
        key = (size, bold, cjk)
        if key not in fonts:
            name = 'NotoSansCJK-Regular.ttc' if cjk else ('NotoSans-Bold.ttf' if bold else 'NotoSans-Regular.ttf')
            fonts[key] = ImageFont.truetype(str(ROOT / 'assets/fonts' / name), round(size * S))
        return fonts[key]
    def text(x, y, value, size=26, color=INK, bold=False, anchor='lt', cjk=False):
        f = font(size, bold, cjk)
        pos = (round(x * S), round(y * S))
        d.text(pos, value, font=f, fill=color, anchor=anchor, stroke_width=0)
        box = d.textbbox(pos, value, font=f, anchor=anchor)
        text_boxes.append({'text': value, 'bbox': [round(v / S, 2) for v in box]})
    def line(points, fill='#8A9699', w=1.5):
        d.line([(round(x * S), round(y * S)) for x, y in points], fill=fill, width=max(1, round(w * S)))

    # Header and explicit synthetic-data notice.
    text(100, 74, 'Sleep duration distribution', 76, bold=True)
    text(104, 179, '睡眠时长分布', 28, cjk=True)
    text(329, 184, 'Share by sleep duration (%)', 27, color=MUTED)
    badge = (1600 * S, 87 * S, 1900 * S, 143 * S)
    d.rounded_rectangle(badge, radius=9 * S, fill='#DDE7E7')
    text(1750, 114, data['visible_notice'], 25, bold=True, anchor='mm')
    line([(100, 230), (1900, 230)], '#CED5D6', 1)

    # Real generated pillow body; geometry is derived from its alpha silhouette.
    pillow = Image.open(ROOT / 'assets/pillow_neutral.png').convert('RGBA')
    object_width = 1800
    object_height = round(object_width * pillow.height / pillow.width)
    object_x, object_y = 100, 350
    pillow = pillow.resize((object_width * S, object_height * S), Image.Resampling.LANCZOS)
    a = np.array(pillow)
    alpha = a[:, :, 3]
    body_ys, body_xs = np.where(alpha >= 128)
    mid_y = int(round((int(body_ys.min()) + int(body_ys.max())) / 2))
    mid_xs = np.flatnonzero(alpha[mid_y] >= 128)
    left, right = float(mid_xs[0]), float(mid_xs[-1] + 1)
    visible_width = right - left
    data_start = left + 0.30 * visible_width
    data_width = 0.70 * visible_width
    boundaries = data_start + np.r_[0., np.cumsum(values)] / 100. * data_width
    centers = (boundaries[:-1] + boundaries[1:]) / 2
    x_positions = object_x + centers / S
    global_mid_y = object_y + mid_y / S

    # Cotton luminance modulates printed pigment. All boundaries remain vertical.
    lum = a[:, :, :3].astype(np.float32).mean(axis=2)
    shade = np.clip((lum / 236.0) ** 1.15, 0.28, 1.07)
    pixel_centers = np.arange(a.shape[1], dtype=float) + 0.5
    colored = a[:, :, :3].astype(np.float32)
    for i, color in enumerate(COLORS):
        lower = boundaries[i]
        upper = boundaries[i + 1] if i < len(values) - 1 else float(a.shape[1])
        # Exact pixel coverage, not minimum-size bands or rounded category widths.
        coverage = np.clip(np.minimum(pixel_centers + 0.5, upper) - np.maximum(pixel_centers - 0.5, lower), 0, 1)
        idx = coverage > 0
        pigment = shade[:, idx, None] * np.asarray(rgb(color), dtype=np.float32)
        colored[:, idx] = colored[:, idx] * (1 - coverage[idx][None, :, None]) + pigment * coverage[idx][None, :, None]
    a[:, :, :3] = np.clip(colored, 0, 255).astype('uint8')
    pillow_colored = Image.fromarray(a)

    # Ground contact shadow stays separate from the quantitative cloth geometry.
    shadow = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    floor_y = object_y + (int(body_ys.max()) + 1) / S
    sd.ellipse((178*S, (floor_y-35)*S, 1848*S, (floor_y+33)*S), fill=(49, 61, 65, 44))
    shadow = shadow.filter(ImageFilter.GaussianBlur(21*S))
    canvas.alpha_composite(shadow)
    contact = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
    cd = ImageDraw.Draw(contact)
    cd.ellipse((234*S, (floor_y-19)*S, 1798*S, (floor_y+7)*S), fill=(38, 50, 54, 60))
    canvas.alpha_composite(contact.filter(ImageFilter.GaussianBlur(7*S)))
    canvas.alpha_composite(pillow_colored, (object_x*S, object_y*S))
    d = ImageDraw.Draw(canvas)

    # Direct callouts for narrow bands; their lengths are never enlarged for text.
    label_x = [531, 751, 971, 1191, None, 1770]
    callouts = []
    for i in [0, 1, 2, 3, 5]:
        lx = label_x[i]
        xc = float(x_positions[i])
        xx = int(round(centers[i]))
        yy = int(np.flatnonzero(alpha[:, xx] >= 128)[0])
        top_y = object_y + yy / S
        text(lx, 264, fmt(values[i]) + '%', 43, color=LABEL_COLORS[i], bold=True, anchor='mt')
        parent = data['category_parent'][i]
        text(lx, 324, parent, 28, anchor='mt')
        if '/' in categories[i]:
            subcategory = categories[i].split('/', 1)[1].strip()
            text(lx, 360, subcategory, 21, color=MUTED, anchor='mt')
            leader_start = 394
        else:
            leader_start = 368
        # A short leader is anchored on the colored textile itself.
        anchor_y = top_y + 17
        bend_y = max(leader_start + 10, top_y - 14)
        line([(lx, leader_start), (lx, bend_y), (xc, anchor_y)], '#718085', 1.3)
        r = 3.5
        d.ellipse(((xc-r)*S, (anchor_y-r)*S, (xc+r)*S, (anchor_y+r)*S), fill=COLORS[i], outline='#F2F4F3', width=2*S)
        callouts.append({'category': categories[i], 'anchor_xy': [round(xc, 3), round(anchor_y, 3)]})

    # Largest share is printed directly onto its wide cloth band.
    main_x = float(x_positions[4])
    text(main_x, global_mid_y - 77, fmt(values[4]) + '%', 102, color='#FFFFFF', bold=True, anchor='mm')
    text(main_x, global_mid_y + 46, categories[4], 37, color='#FFFFFF', anchor='mm')
    # Clearly identify the white section as non-data fabric, without a false category.
    white_x = object_x + (left + data_start) / (2*S)
    text(white_x, global_mid_y - 16, 'WHITE FABRIC', 20, color='#7A8589', anchor='mm')
    text(white_x, global_mid_y + 20, 'not part of the data', 20, color='#7A8589', anchor='mm')

    # Reading guide explains the special mid-section width encoding and synthetic scope.
    text(100, 1303, 'READ THE WIDTH', 19, bold=True, color=INK)
    text(308, 1300, 'At the horizontal midpoint: white fabric = 30% of the pillow; colored fabric = 100% of the distribution.', 23, color=MUTED)
    text(100, 1348, '4–<6 h has three mutually exclusive synthetic subcategories. Illustrative data only; not a statistical survey.', 22, color=MUTED)

    output = canvas.convert('RGB').resize((width, height), Image.Resampling.LANCZOS)
    output.save(ROOT / 'final.png', compress_level=6)
    # These are mapping records, not a replacement for measurements of final pixels.
    layout = {
        'canvas': [width, height], 'supersampling': S,
        'pillow_xywh': [object_x, object_y, object_width, object_height],
        'midline_y': global_mid_y,
        'midline_cloth_left': object_x + left/S,
        'midline_cloth_right': object_x + right/S,
        'non_data_width_px': 0.30*visible_width/S,
        'data_width_px': data_width/S,
        'boundaries_x': [object_x + b/S for b in boundaries],
        'colors': COLORS, 'values': values.tolist(), 'categories': categories,
        'callouts': callouts, 'text_boxes': text_boxes,
        'title_zh': data['topic_zh'],
        'encoding_zh': core['data_carrying_relationship_zh']
    }
    (ROOT / 'artifacts/layout.json').write_text(json.dumps(layout, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Saved final.png (2000 × 1400); generated cloth with six data-controlled bands.')

if __name__ == '__main__':
    main()
