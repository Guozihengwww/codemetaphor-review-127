"""Offline infographic renderer. Run through capture_render.py for every attempt."""
from pathlib import Path
import json
import math
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def main():
    inputs = read_json('input.json')
    brief = read_json(inputs['brief'])
    data = read_json(inputs['data'])
    core = read_json(inputs['critical_metaphor_contract'])
    # Retain and verify all frozen local dependencies, without sampling reference pixels.
    assert (ROOT / inputs['reference']).is_file()
    assert core['visible_title'] == data['title']
    series = data['series'][0]
    values = series['values']
    categories = data['categories']
    assert len(values) == len(categories) == 8
    assert sum(values) == data['series_totals'][series['id']]
    assert data['value_axis'] == {'minimum': 0, 'maximum': 70, 'scale': 'linear'}
    W, H = brief['width'], brief['height']
    S = 2
    BG = '#F3F8FA'
    INK = '#123448'
    MUTED = '#526D7B'
    BLUE = '#5AACE0'
    GRID = '#DCE8EE'
    canvas = Image.new('RGBA', (W * S, H * S), BG)
    draw = ImageDraw.Draw(canvas)
    boxes = []

    def font(size, bold=False):
        return ImageFont.truetype(str(ROOT / 'assets/fonts' / ('sans_bold.otf' if bold else 'sans_regular.otf')), round(size * S))

    def rect(box, fill, outline=None, width=1):
        draw.rectangle(tuple(round(x * S) for x in box), fill=fill, outline=outline, width=round(width * S))

    def line(points, fill, width=1):
        draw.line([(round(x*S), round(y*S)) for x,y in points], fill=fill, width=max(1, round(width*S)))

    def text(x, y, value, size, fill=INK, bold=False, anchor='lt', role='text'):
        value = str(value)
        f = font(size, bold)
        xy = (round(x*S), round(y*S))
        draw.text(xy, value, font=f, fill=fill, anchor=anchor)
        b = draw.textbbox(xy, value, font=f, anchor=anchor)
        boxes.append({'text': value, 'role': role, 'bbox': [round(z / S, 2) for z in b]})

    def tracked(x, y, value, size, fill, spacing=2.5, bold=True):
        f = font(size, bold)
        left = x
        for char in value:
            draw.text((round(left*S), round(y*S)), char, font=f, fill=fill, anchor='lt')
            left += draw.textlength(char, font=f)/S + spacing
        boxes.append({'text': value, 'role': 'section_label', 'bbox': [x, y, left-spacing, y+size]})

    # Header hierarchy is deliberately separate from the numerical marks.
    rect((110, 100, 122, 136), BLUE)
    tracked(142, 102, data['visible_notice'], 30, '#286E99', 3.3)
    text(106, 184, 'Icebreaker fleet sizes', 116, bold=True, role='title')
    text(110, 337, series['label'], 44, role='metric')
    text(110, 399, f'Number of {series["unit"]} across {len(categories)} synthetic fleets', 34, MUTED, role='metric_unit')
    line([(110, 470), (1890, 470)], '#C6D9E3', 1.5)

    # A single mapping is used for ALL rows and the generated bow.
    x0, axis_right = 380.0, 1850.0
    axis_min, axis_max = data['value_axis']['minimum'], data['value_axis']['maximum']
    scale = (axis_right - x0) / (axis_max - axis_min)
    X = lambda value: x0 + (value-axis_min)*scale
    axis_y = 571
    row_y = [694.0, 874.0, 1018.0, 1162.0, 1306.0, 1450.0, 1594.0, 1738.0]
    tracked(110, 511, 'FLEET', 26, MUTED, 2.5)
    tracked(380, 511, 'NUMBER OF SHIPS', 26, MUTED, 2.1)
    for v in range(axis_min, axis_max + 1, 10):
        x = X(v)
        text(x, 548, v, 27, MUTED, anchor='mt', role='axis_tick')
        line([(x, axis_y+14), (x, 1796)], GRID, 1)
    line([(x0, axis_y+14), (axis_right, axis_y+14)], '#AFC7D3', 1.3)
    line([(x0, axis_y+14), (x0, 1796)], '#AFC7D3', 1.6)

    parts = read_json('assets/parts.json')
    asset_uses = []

    def asset(name, x, y, width, angle=0):
        im = Image.open(ROOT / parts[name]['path']).convert('RGBA')
        h = im.height * width / im.width
        im = im.resize((round(width*S), round(h*S)), Image.Resampling.LANCZOS)
        if angle:
            im = im.rotate(angle, Image.Resampling.BICUBIC, expand=True)
        canvas.alpha_composite(im, (round(x*S), round(y*S)))
        asset_uses.append({'asset': parts[name]['path'], 'bbox': [x,y,x+im.width/S,y+im.height/S], 'angle':angle})
        return im

    def channel(x1, y, h, first=False):
        # Boundary coordinates are exact; no minimum mark length is imposed.
        left, right = round(x0*S), round(x1*S)
        top, bottom = round((y-h/2)*S), round((y+h/2)*S)
        draw.rectangle((left, top, right-1, bottom-1), fill=BLUE)
        # Faint interior bands evoke a quiet cleared sea lane, clipped to its extent.
        if first:
            for yy in (y-28, y+18):
                draw.line([(left+14, round(yy*S)), (right-1, round(yy*S))], fill='#65B2E2', width=2*S)

    ship_w = 600.0
    ship_right = X(values[0])
    ship_left = ship_right - ship_w
    ship_info = parts['icebreaker']
    ship_top = row_y[0] - ship_info['nose_y'] * ship_w / ship_info['size'][0]
    geometry = []
    for i, (category, value, y) in enumerate(zip(categories, values, row_y)):
        text(110, y, category, 44, INK, bold=(i==0), anchor='lm', role='category')
        end = X(value)
        if i == 0:
            channel(ship_left+19, y, 96, True)
        else:
            channel(end, y, 84)
        text(end+29, y, value, 47, INK, bold=True, anchor='lm', role='value')
        geometry.append({'category': category, 'value': value, 'y': y, 'x0': x0, 'endpoint': end, 'span_px':end-x0, 'bar_height':96 if i==0 else 84})

    # Floes are context, behind the ship. They never define a quantitative endpoint.
    asset('ice_01', 1320, 598, 167, 12)
    asset('ice_02', 1491, 594, 160, -10)
    asset('ice_02', 1224, 731, 119, 18)
    asset('ice_01', 1471, 736, 115, -15)
    asset('ice_04', 1070, 641, 35, -10)
    asset('ice_03', 1051, 704, 25, 28)
    asset('ice_05', 1108, 607, 39, -25)
    asset('ice_06', 1111, 744, 37, 20)
    asset('ice_04', 1675, 743, 41, 25)
    asset('icebreaker', ship_left, ship_top, ship_w)

    # An unambiguous semantic footer; there are no real-country claims.
    line([(110, 1846), (1890, 1846)], '#C6D9E3', 1.5)
    text(110, 1880, 'Read each lane from 0 to its endpoint; Fleet A ends at the bow.', 29, MUTED, role='encoding_note')
    text(110, 1928, 'Synthetic categories and values. Not a real-world fleet inventory.', 28, MUTED, role='provenance')

    canvas.convert('RGB').resize((W,H), Image.Resampling.LANCZOS).save(ROOT/'final.png')
    artifacts = ROOT/'artifacts'
    artifacts.mkdir(exist_ok=True)
    ledger = {'canvas':[W,H], 'scale_px_per_ship':scale, 'axis':[x0,axis_right,axis_min,axis_max],
              'rows':geometry, 'ship':{'left':ship_left,'bow_x':ship_right,'top':ship_top,'width':ship_w,
                                      'nose_source_y':ship_info['nose_y'],'channel_end':ship_left+19},
              'text_boxes':boxes, 'asset_uses':asset_uses}
    (artifacts/'render_geometry.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
