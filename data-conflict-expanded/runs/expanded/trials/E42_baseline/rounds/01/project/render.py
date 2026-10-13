#!/usr/bin/env python3
"""Offline renderer. Execute through capture_render.py during this experiment."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent
S = 2
INK = (85, 54, 34)
SOFT = (92, 77, 60)
RULE = (105, 116, 107)
CREAM = (253, 242, 215)


def load_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def main():
    inputs = load_json('input.json')
    brief = load_json(inputs['brief'])
    data = load_json(inputs['data'])
    contract = load_json(inputs['critical_metaphor_contract'])
    # Frozen inputs remain available; the reference is never used as source pixels.
    assert (ROOT / inputs['reference']).is_file()
    assert contract['topic_zh'] == data['topic_zh']
    assert data['synthetic'] is True
    assert len(data['series']) == 1
    series = data['series'][0]
    values = series['values']
    cats = data['categories']
    assert len(cats) == len(values)
    unit = data['unit_mark']['value']
    lo, hi = data['value_axis']['minimum'], data['value_axis']['maximum']
    assert data['value_axis']['scale'] == 'linear' and lo == 0
    assert data['unit_mark']['fractional_marks'] is False
    counts = [int(v / unit) for v in values]
    assert all(n*unit == v for n,v in zip(counts, values))
    total = sum(values)
    assert total == data['series_totals'][series['id']]
    w,h = brief['width'], brief['height']
    assert (w,h) == (2000,2005)

    # Single quantitative transform. One complete top occupies 25 final pixels.
    x_axis, x_end = 272.0, 1905.0
    y_zero, y_max = 1790.0, 415.0
    scale = (y_zero-y_max)/(hi-lo)
    y = lambda value: y_zero - (value-lo)*scale
    unit_h = unit*scale
    centers = np.linspace(510,1770,len(cats)).tolist()
    slab_w = 150.0

    # Subtle, deterministic blue tabletop: no reference-image sampling.
    rng = np.random.default_rng(4268)
    yy,xx = np.mgrid[0:h,0:w]
    light = 2.8*(1-xx/w) + 2.0*(1-yy/h) - 2*((xx/w-.43)**2+(yy/h-.4)**2)
    noise = rng.normal(0,.50,(h,w))
    bg = np.empty((h,w,3),dtype=np.uint8)
    for c, base in enumerate((142,202,226)):
        bg[:,:,c] = np.clip(base+light+noise,0,255).astype(np.uint8)
    im = Image.fromarray(bg).resize((w*S,h*S),Image.Resampling.BICUBIC).convert('RGBA')
    draw = ImageDraw.Draw(im)
    font_cache = {}
    label_boxes = []

    def font(size, face='sans'):
        key=(size,face)
        if key not in font_cache:
            font_cache[key]=ImageFont.truetype(str(ROOT/'assets/fonts'/f'{face}.otf'),round(size*S))
        return font_cache[key]

    def text_at(text, x, top, size=28, face='sans', fill=INK, align='left', record=True):
        f = font(size,face)
        box = draw.textbbox((0,0),text,font=f)
        length = draw.textlength(text,font=f)
        px=x*S
        if align=='center': px-=length/2
        elif align=='right': px-=length
        py=top*S-box[1]
        draw.text((px,py),text,font=f,fill=fill)
        visible=[px/S,top,(px+length)/S,top+(box[3]-box[1])/S]
        if record: label_boxes.append({'text':text,'bbox':visible})
        return visible

    def line(coords, fill=INK, width=1):
        draw.line([(round(x*S),round(y*S)) for x,y in coords],fill=fill,width=max(1,round(width*S)))

    def paste_shadow(asset, x0, y0, strength=65, blur=2.0, offset=(2,3)):
        layer=Image.new('RGBA',im.size)
        a=asset.getchannel('A')
        shade=Image.new('RGBA',asset.size,(26,62,67,0))
        shade.putalpha(a.point(lambda v:int(v*strength/255)))
        layer.alpha_composite(shade,(round((x0+offset[0])*S),round((y0+offset[1])*S)))
        if blur: layer=layer.filter(ImageFilter.GaussianBlur(blur*S))
        im.alpha_composite(layer)

    parts=load_json('assets/parts.json')
    slabs=[Image.open(ROOT/p['path']).convert('RGBA') for p in parts if p['role']=='sesame_slab_top']
    crumbs=[Image.open(ROOT/p['path']).convert('RGBA') for p in parts if p['role']=='sesame_or_caramel_crumb']

    # Use all extracted crumbs, repeated sparingly. None carries a data value.
    crumb_points=[
        (373,1515,10),(428,1695,8),(681,1751,13),(725,1611,5),(944,1727,8),
        (1040,1611,9),(1282,1755,11),(1393,1631,5),(1620,1722,8),(1902,1676,11),
        (348,1115,7),(1798,1231,8),(1510,1335,6),(774,1461,5),(1177,1381,7),
        (1894,921,8),(477,575,7),(1686,548,8),(1917,1440,9),(361,1729,6),
        (672,1684,5),(997,1761,6),(1303,1647,5),(1595,1763,7),(1848,1740,6),
        (802,1270,4),(1038,1330,4),(610,1590,4),(1842,1620,5),(1205,1540,4),
    ]
    crumb_records=[]
    for i,(cx,cy,size) in enumerate(crumb_points):
        src=crumbs[i%len(crumbs)]
        q=src.rotate((i*47)%360,resample=Image.Resampling.BICUBIC,expand=True)
        ratio=size/max(q.size)
        q=q.resize((max(1,round(q.width*ratio*S)),max(1,round(q.height*ratio*S))),Image.Resampling.LANCZOS)
        x0,y0=cx-q.width/S/2,cy-q.height/S/2
        paste_shadow(q,x0,y0,70,.8,(.7,1.2))
        im.alpha_composite(q,(round(x0*S),round(y0*S)))
        crumb_records.append({'asset':i%len(crumbs)+1,'center':[cx,cy],'max_dimension_px':size})
    draw=ImageDraw.Draw(im)

    # Editorial header; all numeric and categorical text derives from data.json.
    draw.rounded_rectangle((118*S,77*S,471*S,133*S),radius=7*S,fill=INK)
    text_at(data['visible_notice'],294.5,92,29,'sans_bold',CREAM,'center')
    text_at('REVENUE / WEEKLY',1877,93,24,'sans_bold',INK,'right')
    text_at('Weekly revenue',118,174,94,'display')
    text_at('in sesame slabs.',118,274,94,'display')
    text_at(f"{series['label']} ({series['unit']})",1877,355,28,'sans',INK,'right')

    # Full frozen scale, with extra 90,000 guide tick for the compact marks.
    line([(x_axis,y_max-24),(x_axis,y_zero),(x_end,y_zero)],width=3.5)
    line([(x_axis-8,y_max-12),(x_axis,y_max-24),(x_axis+8,y_max-12)],width=3.5)
    line([(x_end-13,y_zero-8),(x_end,y_zero),(x_end-13,y_zero+8)],width=3.5)
    text_at(series['unit'],x_axis-31,y_max-69,24,'sans_bold',INK,'right')
    tick_step=unit*5
    ticks=list(range(int(lo),int(hi)+1,tick_step))
    for t in ticks:
        py=y(t)
        major=(t % (unit*10)==0) or t==hi or t==max(values)
        line([(x_axis-(18 if major else 9),py),(x_axis,py)],width=2.1 if major else 1.4)
        if major:
            text_at(f'{t:,}',x_axis-35,py-13,27 if t!=max(values) else 23,'sans',INK,'right')
    # A deliberately short range cue, without a second scale or enlargement.
    line([(x_axis+19,y_max),(x_axis+107,y_max)],fill=(113,163,181),width=1)

    # Summary occupies empty tabletop, visually separate from the measurement axis.
    block_x=560
    text_at('FIVE-WEEK TOTAL',block_x,700,25,'sans_bold')
    text_at(f'${total:,}',block_x-7,759,132,'display')
    text_at(f'{sum(counts)} whole slabs across {len(cats)} weeks',block_x+3,924,32,'sans')
    line([(block_x,991),(1538,991)],fill=(102,160,179),width=1.2)
    legend_slab=slabs[0].resize((round(slab_w*S),round(unit_h*S)),Image.Resampling.LANCZOS)
    paste_shadow(legend_slab,block_x,1043,60,1.7,(1.5,2.2))
    im.alpha_composite(legend_slab,(round(block_x*S),round(1043*S)))
    draw=ImageDraw.Draw(im)
    text_at(f'1 complete top = ${unit:,}',block_x+180,1040,32,'sans_bold')
    text_at('Count the tops; shadows and crumbs add no value.',block_x,1104,26,'sans',SOFT)
    text_at(f'Fixed linear scale: {lo:,}–{hi:,} {series["unit"]}',block_x,1153,26,'sans',SOFT)

    # Quantitative objects: contiguous full top faces, no artificial minimum size.
    geometry=[]
    mask=Image.new('L',im.size,0)
    unit_masks=[]
    for i,(cat,val,count,cx) in enumerate(zip(cats,values,counts,centers)):
        x0=cx-slab_w/2
        bar=Image.new('RGBA',(round(slab_w*S),round(count*unit_h*S)))
        for j in range(count):
            src=slabs[(i+j)%len(slabs)]
            if (i+j)%2: src=src.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            patch=src.resize((round(slab_w*S),round(unit_h*S)),Image.Resampling.LANCZOS)
            # Complete asset, including its very thin side, fits the unit interval.
            # Side thickness and shadows never change positions of subsequent tops.
            local_y=round((count-1-j)*unit_h*S)
            bar.alpha_composite(patch,(0,local_y))
            alpha=np.asarray(patch.getchannel('A'))
            ys,xs=np.nonzero(alpha>=128)
            unit_masks.append({'category':cat,'unit_from_zero':j+1,
              'visible_alpha_bbox':[x0+xs.min()/S,y(val)+local_y/S+ys.min()/S,
                x0+(xs.max()+1)/S,y(val)+local_y/S+(ys.max()+1)/S],
              'layout_interval':[y((j+1)*unit),y(j*unit)]})
        paste_shadow(bar,x0,y(val),78,2.6,(2.0,3.0))
        im.alpha_composite(bar,(round(x0*S),round(y(val)*S)))
        mask.paste(bar.getchannel('A'),(round(x0*S),round(y(val)*S)))
        draw=ImageDraw.Draw(im)
        line([(cx,y_zero+2),(cx,y_zero+11)],width=2.2)
        text_at(cat,cx,1826,32,'sans_bold',INK,'center')
        text_at(f'${val:,}',cx,1875,34,'sans_bold',INK,'center')
        text_at(f'{count} '+('slab' if count==1 else 'slabs'),cx,1922,24,'sans',SOFT,'center')
        geometry.append({'category':cat,'value':val,'unit_count':count,'center_x':cx,
                         'top_y':y(val),'zero_y':y_zero,'height_px':val*scale,
                         'unit_height_px':unit_h,'width_px':slab_w})

    text_at('WEEK',x_axis-35,1830,22,'sans_bold',INK,'right')
    line([(118,1970),(1877,1970)],fill=(98,156,175),width=1)
    text_at('SYNTHETIC DATA  /  Illustrative revenue; no real business claims.',118,1981,17,'sans',INK)
    text_at('WHOLE TOPS ONLY',1877,1981,17,'sans_bold',INK,'right')

    final=im.convert('RGB').resize((w,h),Image.Resampling.LANCZOS)
    final.save(ROOT/'final.png',compress_level=7)
    (ROOT/'artifacts').mkdir(exist_ok=True)
    mask.resize((w,h),Image.Resampling.LANCZOS).save(ROOT/'artifacts/data_top_mask.png')
    evidence={'canvas':[w,h],'scale':{'minimum':lo,'maximum':hi,'zero_y':y_zero,'max_y':y_max,
          'pixels_per_usd':scale,'unit_value':unit,'unit_height_px':unit_h},
          'columns':geometry,'unit_alpha_bounds':unit_masks,'label_boxes':label_boxes,
          'crumbs':crumb_records,'total':total,'total_complete_tops':sum(counts),
          'frozen_dependencies':[inputs['brief'],inputs['data'],inputs['reference'],inputs['critical_metaphor_contract']]}
    (ROOT/'artifacts/render_geometry.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Rendered {w} x {h}; full scale {lo:,}–{hi:,}; complete tops {counts}.')

if __name__=='__main__':
    main()
