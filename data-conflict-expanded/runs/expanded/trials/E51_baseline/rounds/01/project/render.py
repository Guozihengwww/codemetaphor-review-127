#!/usr/bin/env python3
"""Offline pictogram chart: generated shipyard artwork, code-bound quantities.
Run from the project with the supplied Python interpreter. No model/network calls.
Every complete experiment render must be launched through capture_render.py.
"""
from pathlib import Path
import json
from fractions import Fraction
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent

def read_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def main():
    inputs = read_json('input.json')
    brief = read_json(inputs['brief'])
    data = read_json(inputs['data'])
    contract = read_json(inputs['critical_metaphor_contract'])
    assert contract['visible_title'] == data['title']
    assert data['synthetic'] is True
    series = data['series'][0]
    assert len(data['series']) == 1
    years = data['categories']
    values = series['values']
    assert len(years) == len(values)
    unit_value = data['pictogram']['value']
    assert data['pictogram']['fraction_method'] == 'tight width left to right clipping'
    assert data['pictogram']['unit'] == series['unit']
    assert sum(values) == data['series_totals'][series['id']]
    assert all(data['value_axis']['minimum'] <= v <= data['value_axis']['maximum'] for v in values)

    W, H = brief['width'], brief['height']
    S = 2
    PAPER = '#F8F5ED'
    INK = '#252D32'
    SECONDARY = '#646B6C'
    BLUE = '#09548A'
    RULE = '#D8D8CF'
    RED = '#A94136'
    TINT = '#F4E2DB'
    canvas = Image.new('RGB', (W*S, H*S), PAPER)
    draw = ImageDraw.Draw(canvas)
    regular = ROOT/'assets/fonts/NimbusSans-Regular.otf'
    bold = ROOT/'assets/fonts/NimbusSans-Bold.otf'
    font_cache = {}
    text_records = []

    def font(size, weight='regular'):
        key = (size, weight)
        if key not in font_cache:
            font_cache[key] = ImageFont.truetype(str(bold if weight == 'bold' else regular), round(size*S))
        return font_cache[key]

    def label(x, y, content, size, fill=INK, weight='regular', align='left', role='annotation'):
        f = font(size, weight)
        left = x*S
        tw = draw.textlength(content, font=f)
        if align == 'right':
            left -= tw
        elif align == 'center':
            left -= tw/2
        draw.text((round(left), round(y*S)), content, font=f, fill=fill, anchor='lt')
        b = draw.textbbox((round(left), round(y*S)), content, font=f, anchor='lt')
        text_records.append({'text': content, 'role': role, 'bbox': [v/S for v in b]})

    def caps(x, y, content, size=21, spacing=2.0, fill=SECONDARY, align='left'):
        f = font(size, 'bold')
        widths = [draw.textlength(ch, font=f) for ch in content]
        total = sum(widths) + max(0,len(content)-1)*spacing*S
        start = x*S - (total if align == 'right' else 0)
        pos = start
        for ch, width in zip(content, widths):
            draw.text((round(pos), round(y*S)), ch, font=f, fill=fill, anchor='lt')
            pos += width+spacing*S
        text_records.append({'text':content,'role':'section_label','bbox':[start/S,y,(start+total)/S,y+size]})

    def line(x1, y1, x2, y2, color=RULE, width=1):
        draw.line((x1*S, y1*S, x2*S, y2*S), fill=color, width=max(1,round(width*S)))

    def rect(box, fill):
        draw.rectangle(tuple(round(v*S) for v in box), fill=fill)

    # Quiet archival title block; all statistical text comes from the frozen data.
    caps(84, 49, f'SHIPBUILDING / {len(years)} OBSERVED YEARS', 21, 2.1)
    caps(1916, 49, f'{years[0]}–{years[-1]}', 22, 1.9, align='right')
    label(80, 109, data['title'], 80, weight='bold', role='title')
    label(84, 208, 'Annual shipbuilding output · '+series['unit'], 29, SECONDARY, role='metric')
    rect((1590, 119, 1916, 172), RED)
    label(1753, 132, data['visible_notice'], 26, '#FFFFFF', 'bold', 'center', 'synthetic_notice')
    line(84, 257, 1916, 257, INK, 1.6)

    START_X = 236
    GLYPH_W, GLYPH_H = 46, 59
    PITCH = 52
    FIRST_CENTER = 351
    ROW_STEP = 83
    MAX_FIELD_END = 1740
    VALUE_X = 1916
    caps(84, 279, 'YEAR', 20, 1.6)
    caps(START_X, 279, 'SHIPBUILDING OUTPUT', 20, 1.6)
    caps(VALUE_X, 279, 'NET TONS', 20, 1.6, align='right')

    # Reference's emphasis strip is tied to the semantic year, never a rank or peak.
    highlight_year = '1915'
    if highlight_year in years:
        hi = years.index(highlight_year)
        cy = FIRST_CENTER + hi*ROW_STEP
        rect((72, cy-38, 1928, cy+38), TINT)
        # A restrained red halftone recalls the supplied source's emphasis band.
        for yy in range(cy-35, cy+37, 6):
            for xx in range(78+(3 if ((yy-(cy-35))//6)%2 else 0),1928,6):
                r = 0.43
                draw.ellipse(((xx-r)*S,(yy-r)*S,(xx+r)*S,(yy+r)*S), fill='#D7998C')
        rect((72,cy-38,76,cy+38), RED)

    rows = []
    for i, (year, value) in enumerate(zip(years, values)):
        cy = FIRST_CENTER + i*ROW_STEP
        y = cy - GLYPH_H//2
        count = Fraction(value, unit_value)
        full, remainder = divmod(value, unit_value)
        partial_width = Fraction(remainder*GLYPH_W, unit_value)
        assert partial_width.denominator == 1, 'This data has exact whole-pixel fraction widths.'
        partial_width = int(partial_width)
        marks = []
        for j in range(full):
            marks.append({'x':START_X+j*PITCH,'y':y,'w':GLYPH_W,'h':GLYPH_H,'fraction':1.0})
        if remainder:
            marks.append({'x':START_X+full*PITCH,'y':y,'w':partial_width,'h':GLYPH_H,'fraction':remainder/unit_value})
        right = max([m['x']+m['w'] for m in marks], default=START_X)
        assert right <= MAX_FIELD_END
        label(84, cy-19, year, 40, RED if year == highlight_year else INK, 'bold', role='category')
        label(VALUE_X, cy-15, f'{value:,}', 34, RED if year == highlight_year else INK, align='right', role='value')
        rows.append({'category':year,'value':value,'unit':series['unit'],'center_y':cy,
                     'full_symbols':full,'remainder_value':remainder,'fraction':remainder/unit_value,
                     'partial_width_px':partial_width,'tight_full_width_px':GLYPH_W,
                     'symbol_count':float(count),'visible_right_exclusive':right,'marks':marks})

    # Legend includes the only fractional unit used by this dataset.
    line(84, 1250, 1916, 1250, INK, 1.6)
    label(146, 1280, f'1 full symbol = {unit_value:,} {series["unit"]}', 27, INK, 'bold', role='legend')
    label(146, 1317, 'The same ship-in-dock symbol in every row.', 22, SECONDARY, role='legend_note')
    label(842, 1280, f'½ symbol = {unit_value//2:,} {series["unit"]}', 27, INK, 'bold', role='legend')
    label(842, 1317, 'Keep the left half of its tight width.', 22, SECONDARY, role='clipping_note')
    caps(1916, 1280, 'TOTAL SHOWN', 18, 1.6, align='right')
    label(1916, 1311, f'{sum(values):,} {series["unit"]}', 29, INK, 'bold', 'right', 'series_total')
    label(84, 1380, 'Years are equally spaced; gaps do not encode elapsed time.', 22, SECONDARY, role='spacing_note')
    label(1916, 1380, 'Source: supplied synthetic data · Not historical statistics.', 22, SECONDARY, align='right', role='source_note')

    # Finish text antialiasing before compositing tight integer-pixel data marks.
    canvas = canvas.resize((W,H), Image.Resampling.LANCZOS).convert('RGBA')
    glyph = Image.open(ROOT/'assets/shipyard_46x59.png').convert('RGBA')
    assert glyph.size == (GLYPH_W,GLYPH_H)
    assert glyph.getchannel('A').getbbox() == (0,0,GLYPH_W,GLYPH_H)
    for row in rows:
        for mark in row['marks']:
            # A fraction is clipped left-to-right; it is never scaled or area-matched.
            piece = glyph.crop((0,0,mark['w'],GLYPH_H))
            canvas.alpha_composite(piece, (mark['x'],mark['y']))
    canvas.alpha_composite(glyph,(84,1273))
    canvas.alpha_composite(glyph.crop((0,0,GLYPH_W//2,GLYPH_H)),(800,1273))

    # Save numeric geometry separately so checks can compare it with raster evidence.
    maximum_count = Fraction(data['value_axis']['maximum'], unit_value)
    full_max, rem_max = divmod(data['value_axis']['maximum'],unit_value)
    capacity_end = START_X + full_max*PITCH + float(Fraction(rem_max*GLYPH_W,unit_value))
    layout = {'canvas':[W,H],'glyph':'assets/shipyard_46x59.png','glyph_size':[GLYPH_W,GLYPH_H],
              'pitch_px':PITCH,'inter_symbol_gap_px':PITCH-GLYPH_W,'row_step_px':ROW_STEP,
              'plot_left_px':START_X,'plot_right_px':MAX_FIELD_END,
              'linear_unit_value':unit_value,'value_axis':data['value_axis'],
              'axis_maximum_symbol_count':float(maximum_count),'axis_maximum_extent_px':capacity_end,
              'highlight_category':highlight_year,'rows':rows,'text':text_records}
    assert capacity_end <= MAX_FIELD_END
    for item in text_records:
        l,t,r,b = item['bbox']
        assert l >= 0 and t >= 0 and r <= W and b <= H, item
    (ROOT/'artifacts').mkdir(exist_ok=True)
    (ROOT/'artifacts/layout.json').write_text(json.dumps(layout, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
    canvas.convert('RGB').save(ROOT/'final.png')

if __name__ == '__main__':
    main()
