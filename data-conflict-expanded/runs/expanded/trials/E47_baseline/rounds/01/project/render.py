#!/usr/bin/env python3
"""Offline renderer for this frozen case. Run from this directory: PYTHON render.py.

Complete research renders are launched with capture_render.py. All visual assets
are local. Three slices of the generated cigarette form each quantitative mark.
The tobacco cap and filter are INSIDE the length on the one linear data scale.
"""
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np

ROOT = Path(__file__).resolve().parent
S = 2
BG = '#fffdf8'
CHART = '#f3eddf'
INK = '#192736'
ORANGE = '#d76125'
MUTED = '#657078'
GRID = '#d6c7a9'
FONT = ROOT / 'assets/fonts'


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def font(name, size):
    return ImageFont.truetype(str(FONT / name), round(size*S))


def main():
    inputs = read_json('input.json')
    brief = read_json(inputs['brief'])
    data = read_json(inputs['data'])
    core = read_json(inputs['critical_metaphor_contract'])
    assert (ROOT / inputs['reference']).is_file()
    assert core['topic_zh'] == data['topic_zh']
    assert data['synthetic'] and data['value_axis']['scale'] == 'linear'
    assert data['value_axis']['minimum'] == 0
    assert len(data['series']) == 1
    series = data['series'][0]
    cats, vals = data['categories'], series['values']
    assert len(cats) == len(vals) == 13
    assert sum(vals) == data['series_totals'][series['id']]
    W, H = brief['width'], brief['height']
    im = Image.new('RGBA', (W*S, H*S), BG)
    draw = ImageDraw.Draw(im)
    text_records = []

    def rect(box, fill, radius=0):
        b = tuple(round(c*S) for c in box)
        if radius:
            draw.rounded_rectangle(b, radius=round(radius*S), fill=fill)
        else:
            draw.rectangle(b, fill=fill)

    def line(points, fill, width=1):
        draw.line([(round(x*S), round(y*S)) for x,y in points], fill=fill, width=max(1,round(width*S)))

    def text(x, y, value, size, face='Sans-Regular.otf', fill=INK, anchor='lt', tracking=0):
        f = font(face, size)
        if tracking:
            xx = x*S
            for c in value:
                draw.text((xx,y*S), c, font=f, fill=fill, anchor=anchor)
                xx += f.getlength(c) + tracking*S
            bbox = (x, y, xx/S, y+size)
        else:
            draw.text((x*S,y*S), value, font=f, fill=fill, anchor=anchor)
            bbox = tuple(a/S for a in draw.textbbox((x*S,y*S), value, font=f, anchor=anchor))
        text_records.append({'text': value, 'bbox': [round(a,2) for a in bbox]})

    # Header: a clear synthetic notice, the actual exposure topic, and the
    # generated city/smoking illustration. No reference-image pixels are used.
    rect((64,54,321,96), INK, 4)
    text(82,65,data['visible_notice'],22,'Sans-Bold.otf',BG,tracking=1)
    text(64,128,'Annual cigarette',84,'Serif-Bold.otf')
    text(64,217,'equivalent exposure',84,'Serif-Bold.otf',ORANGE)
    text(66,331,'Indirect smoking in cities',28,'Sans-Bold.otf')
    text(66,378,'Annual exposure expressed as cigarette equivalents,',25)
    text(66,412,'for one resident in each synthetic city.',25)
    text(66,473,'ILLUSTRATIVE VALUES  /  13 SYNTHETIC CITIES',17,'Sans-Bold.otf',MUTED,tracking=1.2)
    hero = Image.open(ROOT / 'assets/hero_city_smoking.png').convert('RGBA')
    hero_h = 447*S
    hero_w = round(hero.width*hero_h/hero.height)
    hero = hero.resize((hero_w,hero_h),Image.Resampling.LANCZOS)
    im.alpha_composite(hero,(round(918*S),round(41*S)))

    # Quiet warm chart field and a subtle non-data curve, echoing paper and smoke.
    rect((0,526,W,H),CHART)
    draw.ellipse((int(-360*S),int(1490*S),int(700*S),int(2430*S)),fill='#f0e8d8')
    line([(64,526),(1264,526)],'#dfd4c0',1)
    text(64,567,series['label'],31,'Serif-Bold.otf')
    text(64,611,series['unit'],23,'Sans-Regular.otf',MUTED)

    x0, x1 = 260.0, 1190.0
    axis_min, axis_max = data['value_axis']['minimum'], data['value_axis']['maximum']
    px_per_unit = (x1-x0)/(axis_max-axis_min)
    plot_top, plot_bottom = 709, 1884
    ticks = list(range(axis_min,axis_max+1,25))
    text(64,676,'CITY',17,'Sans-Bold.otf',MUTED,tracking=1.8)
    for t in ticks:
        x = x0+(t-axis_min)*px_per_unit
        line([(x,plot_top),(x,plot_bottom)],GRID,1.25 if t else 1.7)
        text(x,673,str(t),22,'Sans-Regular.otf',MUTED,anchor='mt')
        text(x,1905,str(t),21,'Sans-Regular.otf',MUTED,anchor='mt')
    line([(x0,plot_top),(x1,plot_top)],GRID,1)
    line([(x0,plot_bottom),(x1,plot_bottom)],GRID,1)

    pieces = [Image.open(ROOT / f'assets/{n}.png').convert('RGBA') for n in (
        'cigarette_filter','cigarette_paper','cigarette_tobacco_end')]
    object_h = 52
    filter_w, end_w = 112, 33
    first_y, row_step = 756, 90
    rows = []
    for i,(category,value) in enumerate(zip(cats,vals)):
        assert 0 < value <= axis_max
        center_y = first_y+i*row_step
        total_hi = round((value-axis_min)*px_per_unit*S)
        # Adapt non-data part sizes only if a future positive span is very small;
        # never inflate a quantitative mark to a minimum length.
        proportion = min(1.0,total_hi/(S*(filter_w+end_w+30)))
        f_hi = round(filter_w*S*proportion)
        e_hi = round(end_w*S*proportion)
        p_hi = total_hi-f_hi-e_hi
        widths = [f_hi,p_hi,e_hi]
        cigarette = Image.new('RGBA',(total_hi,object_h*S),(0,0,0,0))
        cur = 0
        for piece,width in zip(pieces,widths):
            resized = piece.resize((width,object_h*S),Image.Resampling.LANCZOS)
            cigarette.alpha_composite(resized,(cur,0))
            cur += width
        top = center_y-object_h/2
        im.alpha_composite(cigarette,(round(x0*S),round(top*S)))
        # These boxes come from rendered asset alpha, not just target lengths.
        a = np.asarray(cigarette.getchannel('A'))
        yy,xx = np.where(a>=128)
        measured_box_hi = [int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
        text(64,center_y-17,category,33,'Serif-Bold.otf')
        endpoint = x0+total_hi/S
        text(endpoint+17,center_y-15,str(value),30,'Sans-Bold.otf')
        rows.append({
            'category':category,'value':value,'center_y':center_y,
            'target_length_px':round(value*px_per_unit,6),
            'allocated_length_px':total_hi/S,
            'actual_asset_alpha_bbox_px':[round(x0+measured_box_hi[0]/S,3),round(top+measured_box_hi[1]/S,3),round(x0+measured_box_hi[2]/S,3),round(top+measured_box_hi[3]/S,3)],
            'alpha_measured_length_px':(measured_box_hi[2]-measured_box_hi[0])/S,
            'filter_slice_width_px':f_hi/S,'paper_slice_width_px':p_hi/S,'end_slice_width_px':e_hi/S,
            'value_label_left_px':round(endpoint+17,3),
        })

    # Explain the entire-object scale and the synthetic statistical status.
    line([(64,1960),(1264,1960)],'#b7ac98',1)
    text(64,1982,'READING THE OBJECT',16,'Sans-Bold.otf',ORANGE,tracking=1.6)
    text(64,2013,'Full cigarette length, including its filter, uses the 0–150 scale.',20)
    text(64,2042,'City categories remain in the supplied order.',20,'Sans-Regular.otf',MUTED)
    text(915,1982,'SYNTHETIC COMPARISON',16,'Sans-Bold.otf',ORANGE,tracking=1.1)
    text(915,2013,'Source: supplied synthetic data.',19)
    text(915,2042,'Not real city measurements.',19,'Sans-Regular.otf',MUTED)

    out = im.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
    out.save(ROOT / 'final.png', optimize=True)
    artifact_dir = ROOT / 'artifacts'
    artifact_dir.mkdir(exist_ok=True)
    (artifact_dir / 'render_geometry.json').write_text(json.dumps({
        'canvas':[W,H], 'scale_min':axis_min,'scale_max':axis_max,
        'shared_origin_x':x0,'axis_end_x':x1,'pixels_per_unit':px_per_unit,
        'object_height_px':object_h,'rows':rows,
        'labels':text_records,
        'generated_material_zh':'顶部插画及全部香烟的滤嘴、纸身、烟丝端面均来自assets/raw_01.png；原图参考像素未参与合成。',
    },ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Rendered {W}x{H}; {len(rows)} cigarette marks; scale {axis_min}–{axis_max}; {px_per_unit:g} px/unit.')


if __name__ == '__main__':
    main()
