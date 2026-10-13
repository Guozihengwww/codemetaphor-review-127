#!/usr/bin/env python3
"""Offline, data-bound champagne infographic. Run through capture_render.py."""
from pathlib import Path
from decimal import Decimal
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
S = 4
BG = '#F5F3EB'
INK = '#203E32'
MUTED = '#677368'
GOLD = '#A38242'
GRID = '#DADFD4'

def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))

def main():
    data, brief, core = read('data.json'), read('brief.json'), read('core.json')
    inputs = read('input.json')
    assert inputs['critical_metaphor_contract'] == 'core.json'
    assert core['visible_title'] == data['title']
    w, h = brief['width'], brief['height']
    assert (w,h) == (2000,2000)
    assert data['value_axis'] == {'minimum':0,'maximum':30,'scale':'linear'}
    series = data['series'][0]
    cats, values = data['categories'], series['values']
    assert len(cats) == len(values) == 8
    im = Image.new('RGB',(w*S,h*S),BG)
    d = ImageDraw.Draw(im)
    fonts = {}
    text_boxes = []

    def font(size,kind='regular'):
        files={'regular':'Sans-Regular.otf','bold':'Sans-Bold.otf','serif':'EditorialSerif.otf'}
        key=(size,kind)
        if key not in fonts:
            fonts[key]=ImageFont.truetype(str(ROOT/'assets/fonts'/files[kind]),round(size*S))
        return fonts[key]

    def text(x,y,s,size=30,kind='regular',fill=INK,anchor='lt'):
        f=font(size,kind)
        d.text((x*S,y*S),s,font=f,fill=fill,anchor=anchor)
        box=d.textbbox((x*S,y*S),s,font=f,anchor=anchor)
        text_boxes.append({'text':s,'bbox':[round(v/S,3) for v in box]})

    def line(points,fill=GRID,width=1):
        d.line([(round(x*S),round(y*S)) for x,y in points],fill=fill,width=max(1,round(width*S)))

    def rectangle(box,fill,outline=None,width=1):
        d.rectangle(tuple(round(v*S) for v in box),fill=fill,outline=outline,width=round(width*S))

    # Editorial heading, precise data semantics, and visible synthetic notice.
    rectangle((96,83,430,137),INK)
    text(263,111,data['visible_notice'],26,'bold',BG,'mm')
    text(1900,111,'EXPORT MARKETS  /  VOLUME',23,'regular',MUTED,'rm')
    text(94,181,'Champagne shipments',113,'serif')
    text(99,322,'Synthetic champagne shipments by export market',34,fill=MUTED)
    line([(98,407),(1900,407)],'#B7C0B3',1.3)

    # Linear measurement space. Bottle silhouette is the data mark itself.
    plot_top, baseline, scale_h = 505., 1615., 1110.
    axis_x, plot_right = 162., 1900.
    px_per_unit=scale_h/30.
    text(99,445,series['unit'],27,'bold')
    text(1900,451,'WHOLE BOTTLE HEIGHT  ·  LINEAR SCALE',20,fill=MUTED,anchor='rt')
    for tick in range(0,31,5):
        y=baseline-tick*px_per_unit
        text(135,y,str(tick),25,fill=MUTED,anchor='rm')
        if tick:
            line([(axis_x,y),(plot_right,y)],GRID,1)
    # Ground line is interrupted beneath tiny marks so it cannot hide their height.
    xs=[274.+220.*i for i in range(8)]
    bottle=Image.open(ROOT/'assets/bottle_cutout.png').convert('RGBA')
    bw,bh=bottle.size
    geometry=[]
    widths=[]
    for value in values:
        widths.append(210.*(float(value)/float(max(values)))**0.60)
    left=axis_x
    for x,width in zip(xs,widths):
        line([(left,baseline),(x-width/2-2,baseline)],'#9AA992',1.3)
        left=x+width/2+2
    line([(left,baseline),(plot_right,baseline)],'#9AA992',1.3)

    # Record masks after the same resampling used in the actual composition.
    diagnostic_dir=ROOT/'artifacts/mark_masks'
    diagnostic_dir.mkdir(parents=True,exist_ok=True)
    for cat,value,x,width in zip(cats,values,xs,widths):
        height=float(value)*px_per_unit
        top=baseline-height
        x0=x-width/2
        ix=int(math.floor(x0*S))-2
        iy=int(math.floor(top*S))-2
        tw=int(math.ceil(width*S))+5
        th=int(math.ceil(height*S))+5
        coeff=(bw/(width*S),0,(ix-x0*S)*bw/(width*S),
               0,bh/(height*S),(iy-top*S)*bh/(height*S))
        tile=bottle.transform((tw,th),Image.Transform.AFFINE,coeff,
                              resample=Image.Resampling.BICUBIC)
        im.paste(tile,(ix,iy),tile)
        # A crop of the rendered alpha is a diagnostic, not another complete chart.
        mask=Image.new('L',(int(math.ceil(width))+12,int(math.ceil(height))+12),0)
        mx=int(math.floor(x0))-5
        my=int(math.floor(top))-5
        bigmask=Image.new('L',(mask.width*S,mask.height*S),0)
        bigmask.paste(tile.getchannel('A'),(ix-mx*S,iy-my*S))
        mask=bigmask.resize(mask.size,Image.Resampling.LANCZOS)
        mask.save(diagnostic_dir/f'{cat}.png')
        ar=np.array(mask)
        rows=np.max(ar,axis=1)
        bounds={}
        for threshold in (8,32,64,128):
            yy=np.where(rows>=threshold)[0]
            bounds[str(threshold)]=[int(my+yy[0]),int(my+yy[-1]+1)] if len(yy) else None
        geometry.append({'category':cat,'value':value,'center_x':x,'target_top_y':top,
                         'baseline_y':baseline,'target_height_px':height,'width_px_nonquantitative':width,
                         'resampled_alpha_row_bounds_by_threshold':bounds,
                         'alpha_mask_path':f'artifacts/mark_masks/{cat}.png'})
        # These are all supplied values, in their supplied precision.
        label=str(value)
        size=43 if cat=='US' else 33
        text(x,top-28,label,size,'bold',INK,'mb')
        # Small values retain their physical size; a clear, separated leader locates them.
        if height<90:
            line([(x,top-18),(x,top-5)],'#9AA992',1)

    # Quiet annotation in the unused high-value region of the plot.
    # A plain background preserves the graph's linearly measured context.
    rectangle((1017,665,1848,994),BG)
    line([(1017,665),(1848,665)],GOLD,2)
    text(1046,701,'TOTAL ACROSS THE 8 SHOWN MARKETS',24,'bold',MUTED)
    total=str(data['series_totals'][series['id']])
    text(1040,750,total,85,'serif')
    text(1046,853,'million bottles',30,fill=MUTED)
    text(1046,930,'Illustrative values, not reported market statistics.',24,fill=MUTED)

    names={'US':['United States'],'GB':['United Kingdom'],'JP':['Japan'],'DE':['Germany'],
           'IT':['Italy'],'AU':['Australia'],'BE':['Belgium'],'CH':['Switzerland']}
    # Category rail is outside the measurement space.
    for cat,x in zip(cats,xs):
        line([(x,baseline+10),(x,baseline+23)],'#9AA992',1.2)
        text(x,1650,cat,39,'bold',anchor='mt')
        for j,name in enumerate(names[cat]):
            text(x,1703+25*j,name,23,fill=MUTED,anchor='mt')

    line([(98,1781),(1900,1781)],'#B7C0B3',1.3)
    text(99,1825,'READING THE BOTTLES',22,'bold')
    text(99,1872,'Full height encodes volume on a shared 0–30 scale; width carries no data.',27,fill=MUTED)
    text(99,1916,'Tiny bottles are drawn to scale. Exact values are printed above each market.',27,fill=MUTED)
    text(1900,1825,'SOURCE: LOCAL SYNTHETIC DATA',22,fill=MUTED,anchor='rt')

    final=im.resize((w,h),Image.Resampling.LANCZOS)
    final.save(ROOT/'final.png',dpi=(300,300))
    artifact=ROOT/'artifacts'
    artifact.mkdir(exist_ok=True)
    (artifact/'geometry.json').write_text(json.dumps({'canvas':[w,h],'baseline_y':baseline,
        'scale_px_per_million_bottles':px_per_unit,'axis':[0,30],'marks':geometry},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (artifact/'text_layout.json').write_text(json.dumps(text_boxes,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    assert all(0<=b['bbox'][0]<b['bbox'][2]<=w and 0<=b['bbox'][1]<b['bbox'][3]<=h for b in text_boxes)
    print('Saved final.png: 2000 × 2000; 8 complete bottle marks; offline local assets.')

if __name__=='__main__':
    main()
