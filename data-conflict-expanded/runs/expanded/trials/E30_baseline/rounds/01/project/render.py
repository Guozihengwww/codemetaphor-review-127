#!/usr/bin/env python3
"""Offline, data-bound nose infographic. Run only through capture_render.py."""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent
S = 2


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def font(name, size):
    return ImageFont.truetype(str(ROOT / 'assets' / 'fonts' / (name + '.otf')), round(size*S))


def main():
    config = read_json('input.json')
    brief = read_json(config['brief'])
    data = read_json(config['data'])
    core = read_json(config['critical_metaphor_contract'])
    # The visual reference remains an input dependency; its pixels never enter the output.
    with Image.open(ROOT / config['reference']) as reference:
        reference.verify()
    assert core['visible_title'] == data['title']
    W, H = brief['width'], brief['height']
    assert (W, H) == (1600, 2000)
    values = np.asarray(data['series'][0]['values'], dtype=float)
    years = np.asarray(data['time_points'], dtype=float)
    categories = data['categories']
    lo, hi = data['value_axis']['minimum'], data['value_axis']['maximum']
    assert data['value_axis']['scale'] == 'linear'
    assert len(values) == len(years) == len(categories) == 5
    assert np.all(np.diff(years) > 0)
    assert np.all((values >= lo) & (values <= hi))

    # One mapping controls mask, texture, contour, points, grids and labels.
    X0, XMAX, YTOP, YBOTTOM = 350., 1310., 565., 1525.
    x_at = lambda v: X0 + (np.asarray(v)-lo)/(hi-lo)*(XMAX-X0)
    y_at = lambda t: YBOTTOM-(np.asarray(t)-years[0])/(years[-1]-years[0])*(YBOTTOM-YTOP)
    xs, ys = x_at(values), y_at(years)
    points = list(zip(xs.tolist(), ys.tolist()))
    sorted_points = list(reversed(points))
    color = {
        'ink': '#283731', 'green': '#28594b', 'muted': '#67766f',
        'paper': '#faf8ef', 'grid': '#dbe3da', 'major': '#c6d4c8',
        'zero': '#46544d', 'contour': '#534132', 'cream': '#fffbee',
    }

    canvas = Image.new('RGB', (W*S, H*S), '#e5e8e2')
    shadow = Image.new('RGBA', canvas.size)
    sd = ImageDraw.Draw(shadow)
    sd.rectangle((64*S, 63*S, 1548*S, 1958*S), fill=(37,48,42,36))
    shadow = shadow.filter(ImageFilter.GaussianBlur(11*S))
    canvas.paste(shadow, (0,0), shadow)
    d = ImageDraw.Draw(canvas)
    d.rectangle((57*S,48*S,1543*S,1952*S), fill=color['paper'])

    # Deterministic fine paper grain, generated locally rather than a copied reference.
    rng = np.random.default_rng(1047)
    noise = rng.normal(0, .47, (1904,1486,1))
    paper = np.clip(np.array([250,248,239])[None,None,:]+noise,0,255).astype(np.uint8)
    canvas.paste(Image.fromarray(paper).resize((1486*S,1904*S)),(57*S,48*S))
    d=ImageDraw.Draw(canvas)

    def line(coords, fill, width=1):
        d.line([(round(x*S),round(y*S)) for x,y in coords],fill=fill,width=max(1,round(width*S)))

    def rect(box, fill, radius=0, outline=None, width=1):
        box=tuple(round(v*S) for v in box)
        if radius:
            d.rounded_rectangle(box,round(radius*S),fill=fill,outline=outline,width=round(width*S))
        else:
            d.rectangle(box,fill=fill,outline=outline,width=round(width*S))

    text_boxes=[]
    def text(txt,x,y,size=28,face='sans',fill=None,anchor='lt'):
        f=font(face,size)
        p=(round(x*S),round(y*S))
        d.text(p,txt,font=f,fill=fill or color['ink'],anchor=anchor)
        b=d.textbbox(p,txt,font=f,anchor=anchor)
        text_boxes.append({'text':txt,'bbox_px':[round(q/S,2) for q in b]})

    # A square grid aligns to 0.2 million horizontally and one year vertically.
    grid_step=(XMAX-X0)/20
    for i in range(-6,25):
        x=X0+i*grid_step
        if 92 < x < 1508:
            line([(x,85),(x,1910)],color['grid'],.65)
    for i in range(-10,30):
        y=YTOP+i*grid_step
        if 85 < y < 1910:
            line([(92,y),(1508,y)],color['grid'],.65)

    # Restrained editorial heading on the graph-paper sheet.
    rect((130,118,433,165),color['green'],radius=23)
    text(data['visible_notice'],281.5,142,24,'sans_bold',color['cream'],'mm')
    text('1995–2015',1464,143,25,'sans',color['muted'],'rm')
    text('Nose procedures',130,226,132,'serif')
    text('Annual counts, in millions',138,381,33,'sans',color['muted'])
    line([(133,459),(1465,459)],color['major'],1.4)

    # Main data body: exact polygon, with no smoothing or minimum-size changes.
    poly=[(X0,YTOP)]+sorted_points+[(X0,YBOTTOM)]
    body_mask=Image.new('L',canvas.size)
    ImageDraw.Draw(body_mask).polygon([(round(x*S),round(y*S)) for x,y in poly],fill=255)

    # Warp the generated colored-pencil skin across the exact data width at each row.
    texture=Image.open(ROOT/'assets/skin_texture.png').convert('RGB')
    plot_w=round((XMAX-X0)*S)+1
    plot_h=round((YBOTTOM-YTOP)*S)+1
    source=np.asarray(texture.resize((plot_w,plot_h),Image.Resampling.LANCZOS))
    row_y=np.arange(plot_h)/S+YTOP
    right=np.interp(row_y,ys[::-1],xs[::-1])
    fraction=np.arange(plot_w)[None,:]/np.maximum((right-X0)*S,1)[:,None]
    u=np.clip(fraction*(plot_w-1),0,plot_w-1)
    left=np.floor(u).astype(int)
    next_x=np.minimum(left+1,plot_w-1)
    weight=(u-left)[...,None]
    rr=np.arange(plot_h)[:,None]
    tex=np.rint(source[rr,left]*(1-weight)+source[rr,next_x]*weight).astype(np.uint8)
    body=Image.new('RGB',canvas.size,color['paper'])
    body.paste(Image.fromarray(tex),(round(X0*S),round(YTOP*S)))
    canvas.paste(body,(0,0),body_mask)
    d=ImageDraw.Draw(canvas)

    # A faint grid remains visible through the pigmented body, as on sketch paper.
    grid=Image.new('RGBA',canvas.size)
    gd=ImageDraw.Draw(grid)
    for i in range(21):
        gx=X0+i*grid_step
        gd.line((round(gx*S),round(YTOP*S),round(gx*S),round(YBOTTOM*S)),fill=(104,104,79,45),width=1*S)
    for i in range(21):
        gy=YTOP+i*grid_step
        gd.line((round(X0*S),round(gy*S),round(XMAX*S),round(gy*S)),fill=(104,104,79,45),width=1*S)
    ga=np.asarray(grid.getchannel('A')).copy()
    ga=(ga.astype(np.uint16)*np.asarray(body_mask)/255).astype(np.uint8)
    grid.putalpha(Image.fromarray(ga))
    canvas.paste(grid,(0,0),grid)

    # Generated facial details are decorative overlays, never subtractions from the fill.
    def detail(path,box):
        asset=Image.open(ROOT/path).convert('RGBA')
        x,y,w,h=box
        asset=asset.resize((round(w*S),round(h*S)),Image.Resampling.LANCZOS)
        layer=Image.new('RGBA',canvas.size)
        layer.paste(asset,(round(x*S),round(y*S)))
        alpha=(np.asarray(layer.getchannel('A')).astype(np.uint16)*np.asarray(body_mask)/255).astype('uint8')
        layer.putalpha(Image.fromarray(alpha))
        canvas.paste(layer,(0,0),layer)

    detail('assets/eye.png',(408,606,300,240))
    detail('assets/nostril_wing.png',(838,1123,309,257))
    d=ImageDraw.Draw(canvas)

    # Data contour and common zero baseline. The face's outline IS the plot boundary.
    line([(X0,YTOP),(xs[-1],ys[-1])],color['contour'],3.2)
    line(sorted_points,color['contour'],4.2)
    line([(X0,YTOP-12),(X0,YBOTTOM)],color['zero'],3.1)
    line([(X0,YBOTTOM),(XMAX,YBOTTOM)],color['zero'],3.1)
    text('YEAR',296,517,23,'sans_bold',color['muted'],'rt')

    for year,cat,value,x,y in zip(years,categories,values,xs,ys):
        line([(X0-13,y),(X0,y)],color['zero'],2.5)
        text(cat,X0-36,y,33,'sans',color['ink'],'rm')
        # Numerical labels are placed directly beside the observed boundary nodes.
        is_bottom=year == years[0]
        if is_bottom:
            line([(x,y+12),(x,y+52)],color['green'],1.4)
        else:
            line([(x+11,y),(x+27,y)],color['green'],1.8)
        r=8.5
        d.ellipse((round((x-r)*S),round((y-r)*S),round((x+r)*S),round((y+r)*S)),fill=color['cream'],outline=color['green'],width=3*S)
        r=2.25
        d.ellipse((round((x-r)*S),round((y-r)*S),round((x+r)*S),round((y+r)*S)),fill=color['green'])
        if is_bottom:
            text(f'{value:.2f}',x,y+76,36,'sans_bold',color['green'],'mm')
        else:
            text(f'{value:.2f}',x+39,y,40,'sans_bold',color['green'],'lm')

    for v in range(int(lo),int(hi)+1):
        x=float(x_at(v))
        line([(x,YBOTTOM),(x,YBOTTOM+13)],color['zero'],2.4)
        text(str(v),x,YBOTTOM+34,31,'sans',color['ink'],'mt')
    text('Million procedures per indicated year',(X0+XMAX)/2,1632,30,'sans',color['ink'],'mt')

    # Reading key explicitly separates observed width from decorative body area.
    line([(133,1720),(1465,1720)],color['major'],1.4)
    text('READING THE OUTLINE',137,1755,23,'sans_bold',color['green'])
    text('Each dot marks a year. Read its count from the zero line to the nose’s edge.',137,1800,27,'sans')
    text('The edge joins observations; the total illustrated area does not encode a value.',137,1843,26,'sans',color['muted'])
    text('Synthetic study data · Not real-world statistics',137,1910,22,'sans',color['muted'],'lb')

    output=canvas.resize((W,H),Image.Resampling.LANCZOS)
    output.save(ROOT/'final.png',compress_level=6)

    # Diagnostic products do not constitute additional chart renders.
    a=ROOT/'artifacts';a.mkdir(exist_ok=True)
    body_mask.resize((W,H),Image.Resampling.LANCZOS).save(a/'logical_body_mask.png')
    geometry={
        'canvas':[W,H],
        'zero_x':X0,'maximum_x':XMAX,'year_top_y':YTOP,'year_bottom_y':YBOTTOM,
        'value_domain':[lo,hi],'year_domain':[float(years[0]),float(years[-1])],
        'interpolation':'piecewise linear; no added observations',
        'nodes':[{'year':cat,'value':float(v),'x':float(x),'y':float(y)} for cat,v,x,y in zip(categories,values,xs,ys)],
        'logical_fill_keeps_decorated_regions':True,
        'body_area_has_no_quantitative_role':True,
        'generated_texture_covers_entire_logical_body':True,
        'text_boxes':text_boxes,
    }
    (a/'geometry.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Saved final.png at 1600 × 2000; five data-bound nodes.')


if __name__=='__main__':
    main()
