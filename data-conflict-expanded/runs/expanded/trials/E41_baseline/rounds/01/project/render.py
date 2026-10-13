#!/usr/bin/env python3
"""Offline renderer: generated tissue materials + data-controlled paper geometry.
Run from this folder with: PYTHON render.py
Complete experimental renders must be launched through capture_render.py.
"""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'assets'
S = 2


def load_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def px(x):
    return int(round(x * S))


def main():
    inputs = load_json('input.json')
    brief = load_json(inputs['brief'])
    data = load_json(inputs['data'])
    core = load_json(inputs['critical_metaphor_contract'])
    assert (ROOT / inputs['reference']).is_file(), 'Frozen design reference is required.'
    assert core['semantic_roles_zh'] == ['Annual use (rolls/person/year)']
    assert data['value_axis'] == {'minimum':0, 'maximum':100, 'scale':'linear'}
    values = data['series'][0]['values']
    groups = data['categories']
    assert len(groups) == len(values) == 10
    assert all(0 <= v <= 100 for v in values)
    assert sum(values) == data['series_totals']['annual_use']
    W, H = brief['width'], brief['height']
    canvas = Image.new('RGB', (W*S, H*S), 'white')
    draw = ImageDraw.Draw(canvas)
    ink = '#223138'
    secondary = '#58666b'
    muted = '#748287'
    grid = '#edf0f0'
    zero = '#afbbbd'
    accent = '#156d74'
    fonts = {}

    def font(size, bold=False):
        key = (size, bold)
        if key not in fonts:
            fonts[key] = ImageFont.truetype(str(ASSETS/'fonts'/('NimbusSans-Bold.otf' if bold else 'NimbusSans-Regular.otf')), px(size))
        return fonts[key]

    text_records = []
    def text(x, y, value, size, color=ink, bold=False, anchor='lt', tracking=0):
        value = str(value)
        f = font(size, bold)
        if not tracking:
            draw.text((px(x), px(y)), value, font=f, fill=color, anchor=anchor)
            b = draw.textbbox((px(x),px(y)), value, font=f, anchor=anchor)
        else:
            widths = [f.getlength(c) for c in value]
            tw = sum(widths) + px(tracking) * max(0, len(value)-1)
            xx = px(x) - (tw/2 if anchor.startswith('m') else 0)
            first = xx
            for c, cw in zip(value, widths):
                draw.text((round(xx),px(y)), c, font=f, fill=color, anchor='lt')
                xx += cw + px(tracking)
            b = (first,px(y),first+tw,px(y+size))
        text_records.append({'text':value, 'bbox':[round(v/S,2) for v in b]})

    def line(points, color, width=1):
        draw.line([(px(x),px(y)) for x,y in points], fill=color, width=px(width))

    def dashed(x0,x1,y,color,width=1.5,dash=6,gap=7):
        x=x0
        while x < x1:
            line([(x,y),(min(x+dash,x1),y)],color,width)
            x += dash + gap

    # Editorial heading. Synthetic notice is conspicuous and unambiguous.
    draw.rounded_rectangle((px(94),px(66),px(397),px(116)), radius=px(7), fill=accent)
    text(113,80,data['visible_notice'],25,'white',True,tracking=1.1)
    text(1908,83,'ILLUSTRATIVE POPULATIONS  /  A–J',24,secondary,anchor='rt')
    text(88,163,'Annual toilet paper use',146,ink,True)
    text(95,326,data['series'][0]['label']+'  ·  '+data['series'][0]['unit'],42,secondary)
    line([(95,412),(1908,412)],'#d9dfe0',1.3)
    text(95,438,'SYNTHETIC POPULATION GROUPS',22,secondary,True,tracking=1.8)

    # A single consistent data transform. The roll head is outside this interval.
    y0 = 630.0
    plot_height = 1080.0
    scale = plot_height/(data['value_axis']['maximum']-data['value_axis']['minimum'])
    centers = [240.0 + 178.0*i for i in range(len(groups))]
    paper_width = 112.0
    tooth_amplitude = 4.0
    tooth_count = 8
    x_axis=110
    chart_right=1918
    text(96,547,'ROLLS',19,muted,True,anchor='mt',tracking=1.4)
    for tick in range(0,101,20):
        y = y0 + tick*scale
        if tick == 0:
            dashed(143,chart_right,y,zero,1.5,7,8)
        else:
            line([(143,y),(chart_right,y)],grid,1)
        line([(128,y),(138,y)],zero if tick==0 else '#c3ced0',1.4)
        text(x_axis,y,str(tick),26,secondary,anchor='rm')
    line([(135,y0),(135,y0+plot_height)],'#e1e7e8',1)

    # Generated materials. The head has a fixed size; tissue is tiled then clipped.
    geom = load_json('assets/asset_geometry.json')
    head = Image.open(ASSETS/'roll_head.png').convert('RGBA')
    cap_mask = np.array(Image.open(ASSETS/'cap_ink_mask.png').convert('L')).astype(float)/255
    head_arr = np.array(head).astype(float)
    paper = Image.open(ASSETS/'paper_texture.png').convert('RGB')
    factor = paper_width/(geom['tail_right_in_head_px']-geom['tail_left_in_head_px'])
    head_size = (px(head.width*factor), px(head.height*factor))
    tile_height = px(paper.height*factor)
    paper_tile = paper.resize((px(paper_width),tile_height),Image.Resampling.LANCZOS)
    colors=['#157d89','#5273ad','#82649d','#a76575','#b66c3f',
            '#a28a39','#748a4b','#468778','#558396','#717e8b']
    all_masks = Image.new('L', (W*S,H*S), 0)
    maskdraw = ImageDraw.Draw(all_masks)
    marks=[]

    for i,(category,value,cx,color) in enumerate(zip(groups,values,centers,colors)):
        left=cx-paper_width/2
        right=cx+paper_width/2
        end=y0+value*scale
        # Symmetric, zero-mean triangular tear. Endpoints share the same phase.
        bottom=[(left+j*paper_width/(2*tooth_count), end + (-tooth_amplitude if j%2==0 else tooth_amplitude))
                for j in range(2*tooth_count+1)]
        polygon=[(left,y0),(right,y0)]+list(reversed(bottom))
        body_height=px(end+tooth_amplitude-y0)+2
        body=Image.new('RGB',(px(paper_width),body_height),'white')
        for yy in range(0,body_height,tile_height):
            strip=paper_tile if (yy//tile_height)%2==0 else paper_tile.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            body.paste(strip,(0,yy))
        local_mask=Image.new('L',body.size,0)
        md=ImageDraw.Draw(local_mask)
        md.polygon([(px(x-left),px(y-y0)) for x,y in polygon],fill=255)
        canvas.paste(body,(px(left),px(y0)),local_mask)
        maskdraw.polygon([(px(x),px(y)) for x,y in polygon],fill=i+1)
        # Deliberately neutral and thin, but clearly visible on a white page.
        line([(left,y0),(left,bottom[0][1])],'#a5adaf',2.3)
        line([(right,y0),(right,bottom[-1][1])],'#a5adaf',2.3)
        line(bottom,'#657479',2.5)
        # Perforations are an object cue only, independent of the numeric scale.
        perforations=[]
        py=y0+80
        while py < end-93:
            dashed(left+5,right-5,py,'#9eabad',1.7,5,5)
            perforations.append(py)
            py+=80
        # Clear numeric label in the last paper segment.
        text(cx,end-60,str(value),55,ink,True,anchor='mt')
        # Preserve all generated shading while recolouring only the original ink.
        target=np.array(tuple(int(color[k:k+2],16) for k in (1,3,5)),dtype=float)
        luminance=head_arr[:,:,:3] @ np.array([.2126,.7152,.0722])
        tint=np.clip(target[None,None,:]*(.45+.55*luminance[:,:,None]/126),0,255)
        colored=head_arr.copy()
        colored[:,:,:3]=head_arr[:,:,:3]*(1-cap_mask[:,:,None])+tint*cap_mask[:,:,None]
        head_img=Image.fromarray(np.clip(colored,0,255).astype('uint8'),'RGBA').resize(head_size,Image.Resampling.LANCZOS)
        head_left=left-geom['tail_left_in_head_px']*factor
        head_top=y0-head.height*factor
        canvas.paste(head_img,(px(head_left),px(head_top)),head_img)
        text(cx,488,category,32,ink,True,anchor='mt')
        marks.append({'category':category,'value':value,'color':color,'paper_left':left,'paper_right':right,
                      'origin_y':y0,'endpoint_mean_y':end,'length_px':value*scale,
                      'tear_vertices':[[round(x,4),round(y,4)] for x,y in bottom],
                      'head_bbox':[head_left,head_top,head_left+head.width*factor,y0],
                      'perforation_y':perforations,'label':str(value)})

    # A restrained reading key occupies the natural negative space beneath short strips.
    key_x=1100
    line([(key_x,1516),(key_x+64,1516)],accent,4)
    text(key_x,1546,'Read the length',33,ink,True)
    text(key_x,1597,'From the shared zero line to the centre',27,secondary)
    text(key_x,1634,'of each torn edge. Longer paper means',27,secondary)
    text(key_x,1671,'more rolls used per person in one year.',27,secondary)

    line([(95,1797),(1908,1797)],'#d9dfe0',1.3)
    text(95,1830,'ONE COMMON SCALE',21,accent,True,tracking=1.6)
    text(95,1868,'0–100 rolls/person/year; all roll heads are the same size.',27,secondary)
    text(95,1906,'Coloured rings and perforation counts do not encode quantity.',27,secondary)
    text(1080,1830,'SYNTHETIC, NOT OBSERVED',21,accent,True,tracking=1.6)
    text(1080,1868,'Groups A–J are illustrative population groups.',27,secondary)
    text(1080,1906,'These values are not observed country statistics.',27,secondary)

    final=canvas.resize((W,H),Image.Resampling.LANCZOS)
    final.save(ROOT/'final.png',optimize=True)
    artifacts=ROOT/'artifacts'; artifacts.mkdir(exist_ok=True)
    # IDs in this mask are diagnostic only; the image itself is checked separately.
    all_masks.resize((W,H),Image.Resampling.NEAREST).save(artifacts/'data_mask.png')
    layout={'canvas':[W,H],'origin_y':y0,'scale_px_per_unit':scale,'axis':[0,100],
            'paper_width':paper_width,'tear_amplitude_px':tooth_amplitude,'marks':marks,
            'text_elements':text_records,'generated_raw_path':'assets/raw_01.png',
            'source_values':values,'synthetic':data['synthetic']}
    (artifacts/'render_geometry.json').write_text(json.dumps(layout,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Rendered final.png: 2000 × 1978, 10 paper strips, 0–100 shared linear scale.')


if __name__=='__main__':
    main()
