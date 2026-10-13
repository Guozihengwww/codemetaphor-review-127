#!/usr/bin/env python3
"""Regenerate final.png from frozen local inputs and the generated raster.

No network or model calls. All ten points and their labels are bound to data.json.
Complete experiment renders must be run through capture_render.py.
"""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from prepare_assets import prepare, chart_points

ROOT = Path(__file__).resolve().parent
S = 2
PLOT = {'left':300., 'right':1810., 'top':480., 'zero':1670.}
COL = {'paper':'#f5f0e6', 'ink':'#253b46', 'muted':'#586a6d',
       'grid':'#d7ddd8', 'major':'#c2cfc9', 'red':'#a02c46',
       'navy':'#153f54', 'water':'#4ba9b6'}
FONTS = {
    'sans':'assets/fonts/NotoSans-Regular.ttf',
    'bold':'assets/fonts/NotoSans-Bold.ttf',
    'serif':'assets/fonts/NotoSerif-Regular.ttf',
    'italic':'assets/fonts/NotoSerif-Italic.ttf',
}


def rgb(h):
    h=h.lstrip('#')
    return tuple(int(h[i:i+2],16) for i in (0,2,4))


def main():
    inputs=json.loads((ROOT/'input.json').read_text(encoding='utf-8'))
    data=json.loads((ROOT/inputs['data']).read_text(encoding='utf-8'))
    brief=json.loads((ROOT/inputs['brief']).read_text(encoding='utf-8'))
    core=json.loads((ROOT/inputs['critical_metaphor_contract']).read_text(encoding='utf-8'))
    # Preserve and validate the original visual source dependency without reusing pixels.
    with Image.open(ROOT/inputs['reference']) as reference:
        reference_size=reference.size
    assert data['title']==core['visible_title']
    assert data['value_axis']['scale']=='linear'
    assert len(data['categories'])==len(data['series'][0]['values'])==10
    W,H=brief['width'],brief['height']
    assert (W,H)==(2000,2000)
    body,points=prepare(data,PLOT,(W,H))
    rng=np.random.default_rng(3208)
    noise=rng.normal(0,.8,(H,W,1))
    paper=np.clip(np.array(rgb(COL['paper']))[None,None,:]+noise,0,255).astype('uint8')
    canvas=Image.fromarray(paper).resize((W*S,H*S),Image.Resampling.BICUBIC).convert('RGBA')
    d=ImageDraw.Draw(canvas)
    font_cache={}
    text_records=[]

    def font(size,style='sans'):
        key=(size,style)
        if key not in font_cache:
            font_cache[key]=ImageFont.truetype(str(ROOT/FONTS[style]),round(size*S))
        return font_cache[key]

    def line(coords,fill,width=1):
        d.line([(round(x*S),round(y*S)) for x,y in coords],fill=fill,width=max(1,round(width*S)),joint='curve')

    def ellipse(box,fill=None,outline=None,width=1):
        d.ellipse(tuple(round(x*S) for x in box),fill=fill,outline=outline,width=round(width*S))

    def rect(box,fill,outline=None,width=1,radius=0):
        coords=tuple(round(x*S) for x in box)
        if radius:
            d.rounded_rectangle(coords,radius=round(radius*S),fill=fill,outline=outline,width=round(width*S))
        else:
            d.rectangle(coords,fill=fill,outline=outline,width=round(width*S))

    def text(x,y,string,size=30,style='sans',fill=None,anchor='la',tag=None):
        f=font(size,style)
        d.text((round(x*S),round(y*S)),string,font=f,fill=fill or COL['ink'],anchor=anchor,
               stroke_width=0)
        box=d.textbbox((round(x*S),round(y*S)),string,font=f,anchor=anchor)
        text_records.append({'text':string,'role':tag or 'annotation',
                             'bbox_px':[round(v/S,2) for v in box]})
        return [v/S for v in box]

    # Editorial masthead.
    rect((124,74,447,126),COL['navy'],radius=5)
    text(285.5,99,data['visible_notice'],25,'bold','#ffffff','mm',tag='synthetic_notice')
    text(1872,99,'ILLUSTRATIVE SERIES  /  2000–2018',23,'sans',COL['muted'],'rm')
    text(120,151,data['title'],86,'serif',tag='title')
    text(126,279,'Economic output per person, traced through a diver’s pose.',31,'sans',COL['muted'])
    line([(126,363),(1875,363)],'#a9b9b4',1.5)
    text(126,397,data['series'][0]['unit'],29,'bold',COL['navy'],tag='unit')
    line([(1439,419),(1498,419)],COL['paper'],11)
    line([(1439,419),(1498,419)],COL['red'],4.5)
    ellipse((1462,412,1476,426),COL['paper'],COL['red'],3)
    text(1514,418,data['series'][0]['label'],27,'sans',COL['ink'],'lm',tag='series')

    # True rectangular coordinate system. Minor squares are approximately 42 px.
    grid_left,grid_right=230,1872
    dx=(PLOT['right']-PLOT['left'])/9
    for k in range(-1,39):
        x=PLOT['left']+k*dx/4
        if grid_left<=x<=grid_right:
            major=(k%4==0)
            line([(x,PLOT['top']),(x,1822)],COL['major'] if major else COL['grid'],1.1 if major else .7)
    lo,hi=data['value_axis']['minimum'],data['value_axis']['maximum']
    def yscale(value):
        return PLOT['zero']-(value-lo)/(hi-lo)*(PLOT['zero']-PLOT['top'])
    for value in np.arange(lo,hi+1,1250):
        y=yscale(value)
        line([(grid_left,y),(grid_right,y)],COL['major'] if value%5000==0 else COL['grid'],
             1.1 if value%5000==0 else .7)
    # Vertical observation guides are hidden by the body where appropriate.
    for x,y in points:
        yy=y+15
        while yy<PLOT['zero']:
            line([(x,yy),(x,min(yy+4,PLOT['zero']))],'#94b1b1',1.15)
            yy+=11

    # Painted water band is entirely below the exact, straight zero line.
    water=np.zeros((153,1743,4),dtype=np.uint8)
    wy,wx=np.mgrid[0:153,0:1743]
    pigment=(2.8*np.sin(wx/73+wy/29)+1.9*np.sin(wx/19-wy/45)+
             rng.normal(0,.9,wx.shape))
    base=np.asarray(rgb('#5eb8c0'))
    for ch in range(3):
        water[:,:,ch]=np.clip(base[ch]+pigment-wy*.085,0,255)
    water[:,:,3]=255
    water_im=Image.fromarray(water).resize((1743*S,153*S),Image.Resampling.BICUBIC)
    canvas.alpha_composite(water_im,(130*S,1670*S))
    d=ImageDraw.Draw(canvas)
    # Restrained horizontal dry-brush marks and small wave glyphs.
    for i in range(51):
        x=float(rng.uniform(145,1770));y=float(rng.uniform(1680,1810))
        length=float(rng.uniform(25,145))
        shade='#69bdc4' if i%2 else '#49a8b5'
        line([(x,y),(min(x+length,1865),y+float(rng.uniform(-2,2)))],shade,float(rng.uniform(1,2.5)))
    for y,offset in [(1706,0),(1792,58)]:
        for x in np.arange(185+offset,1860,142):
            # One quiet hand-drawn crest, kept clear of the year labels.
            line([(x-12,y+5),(x-5,y+1),(x,y-6),(x+6,y+1),(x+14,y+5)],'#287d92',1.8)
    line([(grid_left,480),(grid_left,1670)],COL['navy'],2)
    line([(130,1670),(1872,1670)],COL['navy'],2.7)
    for val in range(int(lo),int(hi)+1,5000):
        y=yscale(val)
        line([(grid_left-9,y),(grid_left,y)],COL['navy'],2)
        text(grid_left-23,y,f'{val:,}',27,'sans',COL['navy'],'rm',tag='value_tick')
    for category,(x,y) in zip(data['categories'],points):
        line([(x,1670),(x,1684)],COL['navy'],2)
        text(x,1748,category,30,'bold','#113d51','mm',tag='category')
    text(1849,1792,'Year',22,'sans','#164a5a','rm',tag='category_axis')

    # The generated drawing is the main chart body, not a detached icon.
    canvas.alpha_composite(body.resize((W*S,H*S),Image.Resampling.LANCZOS))
    d=ImageDraw.Draw(canvas)

    # A single quantitative channel, straight segments through all observations.
    line(points,COL['paper'],11)
    line(points,COL['red'],5)
    for x,y in points:
        ellipse((x-10,y-10,x+10,y+10),COL['paper'],COL['red'],4)

    # Exact values are outside the silhouette and linked by thin neutral leaders.
    # Offsets affect labels only; data coordinates never depend on typography.
    offsets=[(105,-77),(112,-83),(110,-85),(144,-90),(0,142),
             (220,77),(142,73),(120,86),(48,118),(-59,-85)]
    label_info=[]
    for i,((x,y),value,(ox,oy)) in enumerate(zip(points,data['series'][0]['values'],offsets)):
        cx,cy=x+ox,y+oy
        label=f'{value:,}'
        f=font(32,'bold')
        box=d.textbbox((0,0),label,font=f)
        tw=(box[2]-box[0])/S
        hw=tw/2+12; hh=24
        ux,uy=ox/math.hypot(ox,oy),oy/math.hypot(ox,oy)
        start=(x+ux*14,y+uy*14)
        # Intersect leader with its label box, preserving a clean edge.
        t=min(hw/abs(ox) if ox else 1e9,hh/abs(oy) if oy else 1e9)
        end=(cx-ox*t,cy-oy*t)
        line([start,end],COL['paper'],3.2)
        line([start,end],'#657878',1.15)
        rect((cx-hw,cy-hh,cx+hw,cy+hh),COL['paper'],radius=4)
        bbox=text(cx,cy,label,32,'bold',COL['red'],'mm',tag='data_value')
        label_info.append({'category':data['categories'][i],'value':value,
                           'point':[float(x),float(y)],'label_bbox':bbox,
                           'label_center':[float(cx),float(cy)]})

    # Caption supplies the decoding rule and the statistical scope explicitly.
    text(128,1860,'READ THE POSE',20,'bold',COL['navy'])
    text(369,1854,'Point height above the waterline = GDP per capita.',27,'sans',COL['ink'])
    text(369,1894,'Straight segments join the years; body thickness has no numerical meaning.',24,'sans',COL['muted'])
    text(128,1946,'Synthetic observations · USD per person · Not a country estimate',22,'sans',COL['muted'])
    text(1872,1946,'2000–2018',22,'sans',COL['muted'],'ra')

    # Save only here. No later compositing or editing of final.png is performed.
    final=canvas.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
    final.save(ROOT/'final.png')
    metadata={
        'canvas':[W,H], 'supersampling':S, 'plot':PLOT,
        'reference_size':list(reference_size), 'reference_pixels_reused':False,
        'linear_value_scale_px_per_usd':(PLOT['zero']-PLOT['top'])/(hi-lo),
        'category_spacing_px':dx, 'points':label_info, 'text_records':text_records,
        'source_inputs':inputs,
        'quantitative_path_color':COL['red'], 'quantitative_path_width_px':5,
        'node_radius_px':10,
    }
    (ROOT/'artifacts/render_geometry.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Wrote final.png: 2000 × 2000; ten ordered observations; one linear coordinate system.')


if __name__=='__main__':
    main()
