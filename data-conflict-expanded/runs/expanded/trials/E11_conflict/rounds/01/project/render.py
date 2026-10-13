"""Offline, data-bound composition. Run through capture_render.py for a full render."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent
S = 3
BG = (248, 244, 237)
INK = (29, 38, 36)
MUTED = (99, 108, 103)
RED = (232, 57, 40)
RULE = (206, 210, 199)
TEAL = (7, 126, 124)
RESAMPLE = Image.Resampling.LANCZOS


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def font(size, bold=False):
    name = 'NimbusSans-Bold.otf' if bold else 'NimbusSans-Regular.otf'
    return ImageFont.truetype(str(ROOT / 'assets' / name), round(size * S))


def xy(p):
    return tuple(round(v * S) for v in p)


def text(draw, p, value, size, color=INK, bold=False, anchor='lt'):
    draw.text(xy(p), str(value), font=font(size, bold), fill=color, anchor=anchor)


def line(draw, points, color, width=1):
    draw.line([xy(p) for p in points], fill=color, width=max(1, round(width*S)))


def tracked(draw, p, value, size=18, spacing=2.2, color=MUTED, bold=False):
    x,y = p
    f = font(size, bold)
    for ch in value:
        draw.text(xy((x,y)), ch, font=f, fill=color, anchor='lt')
        x += draw.textlength(ch, font=f)/S + spacing


def star(cx,cy,r):
    return [(cx + (r if i%2==0 else r*.42)*math.sin(i*math.pi/5),
             cy - (r if i%2==0 else r*.42)*math.cos(i*math.pi/5)) for i in range(10)]


def make_flag(country, w=58, h=36):
    im = Image.new('RGBA', (w*S,h*S), (255,255,255,255))
    d = ImageDraw.Draw(im)
    def rect(box,fill): d.rectangle(xy(box), fill=fill)
    if country in ('Italy','France'):
        rect((0,0,w/3,h), '#008944' if country=='Italy' else '#14368C')
        rect((w*2/3,0,w,h), '#EA2537')
    elif country=='Germany':
        rect((0,0,w,h/3),'#181A18'); rect((0,h/3,w,2*h/3),'#D72F36')
        rect((0,2*h/3,w,h),'#EEC447')
    elif country=='Spain':
        rect((0,0,w,h),'#BA2636'); rect((0,h/4,w,3*h/4),'#F2C33E')
        # Small heraldic mark, separate from the numerical encoding.
        rect((w*.265,h*.42,w*.365,h*.68),'#BE423C')
        rect((w*.285,h*.38,w*.347,h*.43),'#BA2636')
        rect((w*.298,h*.47,w*.326,h*.62),'#F2C33E')
    elif country=='USA':
        for i in range(13):
            if i%2==0: rect((0,i*h/13,w,(i+1)*h/13),'#C83546')
        cw,ch = w*.43,h*7/13
        rect((0,0,cw,ch),'#23487C')
        for row in range(9):
            count = 6 if row%2==0 else 5
            for col in range(count):
                sx=(col+(.5 if count==6 else 1))*cw/6
                sy=(row+.65)*ch/9.5
                d.polygon([xy(p) for p in star(sx,sy,.84)],fill='white')
    elif country=='UK':
        rect((0,0,w,h),'#23447E')
        line(d,[(0,0),(w,h)],'white',7)
        line(d,[(0,h),(w,0)],'white',7)
        line(d,[(0,0),(w,h)],'#D13A47',2.6)
        line(d,[(0,h),(w,0)],'#D13A47',2.6)
        rect((w/2-6,0,w/2+6,h),'white')
        rect((0,h/2-6,w,h/2+6),'white')
        rect((w/2-3.2,0,w/2+3.2,h),'#D13A47')
        rect((0,h/2-3.2,w,h/2+3.2),'#D13A47')
    else:
        raise ValueError('No country identifier for '+country)
    return im


def paste(im, asset, x,y):
    im.alpha_composite(asset,(round(x*S), round(y*S)))


def pasta_at_height(asset, width, visible_height):
    """Use real generated tips and shafts, with one exact data-height mapping.

    The cap keeps its width-proportional photographic scale. Short bundles
    naturally reveal only the upper part of the cap. The shaft fills the rest.
    This does not impose a minimum data height.
    """
    out_w = round(width*S)
    out_h = max(1,round(visible_height*S))
    top_src = 48
    cap_h = round(top_src/asset.width * out_w)
    cap = asset.crop((0,0,asset.width,top_src)).resize((out_w,cap_h),RESAMPLE)
    if out_h <= cap_h:
        return cap.crop((0,0,out_w,out_h))
    im = Image.new('RGBA',(out_w,out_h))
    im.alpha_composite(cap,(0,0))
    shaft = asset.crop((0,top_src,asset.width,asset.height-5))
    shaft = shaft.resize((out_w,out_h-cap_h),RESAMPLE)
    im.alpha_composite(shaft,(0,cap_h))
    return im


def main():
    inputs = read_json('input.json')
    brief = read_json(inputs['brief'])
    data = read_json(inputs['data'])
    core = read_json(inputs['critical_metaphor_contract'])
    # Frozen files stay available, while reference pixels are never composited.
    if not (ROOT / inputs['reference']).is_file():
        raise FileNotFoundError(inputs['reference'])
    assert core['semantic_roles_zh'] == ['Synthetic pasta production (metric tonnes)']
    W,H = brief['width'],brief['height']
    assert (W,H)==(2000,1484)
    assert len(data['series'])==1
    series = data['series'][0]
    cats,parents,values = data['categories'],data['category_parent'],series['values']
    assert len(cats)==len(parents)==len(values)
    assert data['value_axis']=={'minimum':0,'maximum':3500000,'scale':'linear'}
    assert abs(sum(values)-data['series_totals'][series['id']]) < .0001

    im = Image.new('RGBA',(W*S,H*S),BG+(255,))
    d = ImageDraw.Draw(im)
    baseline, scale_h, scale_max = 1080.0,840.0,float(data['value_axis']['maximum'])
    centers = np.linspace(286,1870,len(cats)).tolist()
    pasta_width,sleeve_width,sleeve_height = 102.0,134.0,204.0

    # Publication-like header, independent of mark geometry.
    tracked(d,(91,79),'FOOD / REGIONAL PRODUCTION',18,2.1,INK,True)
    d.rounded_rectangle(xy((1611,62,1910,105)),radius=21*S,fill=RED)
    text(d,(1760.5,84),data['visible_notice'],22,(255,255,250),True,'mm')
    line(d,[(90,128),(1910,128)],RULE,1)
    text(d,(511,180),'Pasta',221,RED,True)
    text(d,(518,383),'production',144,RED,True)
    text(d,(522,566),'Synthetic pasta production by region',31,INK)
    text(d,(523,610),'metric tonnes  /  linear scale: 0–3,500,000',25,MUTED)

    line(d,[(1426,204),(1426,388)],RULE,1)
    tracked(d,(1470,221),'TOTAL PRODUCTION',18,2,MUTED,True)
    text(d,(1466,267),f'{sum(values):,.0f}',63,INK,True)
    text(d,(1471,342),'metric tonnes',24,MUTED)

    # Concise reading key beside the title; no extra data marks.
    line(d,[(524,698),(524,744)],TEAL,2)
    line(d,[(519,703),(524,698),(529,703)],TEAL,2)
    line(d,[(519,739),(524,744),(529,739)],TEAL,2)
    text(d,(548,694),'Read the height above the sleeve.',24,INK,True)
    text(d,(548,730),'All sleeve rims share zero; paper height carries no value.',23,MUTED)

    # Shared linear axis. Sparse ticks avoid competing with the object geometry.
    tracked(d,(91,190),'TONNES',16,1.4,MUTED,True)
    line(d,[(149,240),(149,baseline)],RULE,1)
    for v,label in [(3500000,'3.5m'),(3000000,'3m'),(2000000,'2m'),(1000000,'1m'),(0,'0')]:
        y=baseline-v/scale_max*scale_h
        line(d,[(143,y),(161,y)],MUTED,1)
        text(d,(130,y),label,22,MUTED,False,'rm')
    for x in np.arange(174,1930,14):
        line(d,[(float(x),baseline),(float(min(x+5,1930)),baseline)],RULE,1)

    # A semantic parent bracket identifies the four distinct USA categories.
    usa_ids=[i for i,p in enumerate(parents) if p=='USA']
    if len(usa_ids)>1:
        lo,hi=centers[usa_ids[0]]-61,centers[usa_ids[-1]]+61
        by=868
        line(d,[(lo,by+8),(lo,by),(hi,by),(hi,by+8)],RULE,1.2)
        text(d,((lo+hi)/2,839),f'USA  /  {len(usa_ids)} synthetic subcategories',22,MUTED,False,'mm')

    pasta = Image.open(ROOT/'assets/pasta.png').convert('RGBA')
    sleeve = Image.open(ROOT/'assets/sleeve.png').convert('RGBA')
    extraction = read_json('assets/extraction.json')
    rim_src = extraction['sleeve_front_rim_y_in_crop']
    # Resize the sleeve once. Split at its photographed straight front lip.
    sleeve_scaled = sleeve.resize((round(sleeve_width*S),round(sleeve_height*S)),RESAMPLE)
    rim_px = round(rim_src/sleeve.height*sleeve_scaled.height)
    sleeve_front = sleeve_scaled.crop((0,rim_px,sleeve_scaled.width,sleeve_scaled.height))
    body_h = sleeve_front.height/S

    shadows = Image.new('RGBA',im.size)
    sd = ImageDraw.Draw(shadows)
    for cx in centers:
        # Contact shadow is a non-data material cue beneath the sleeve.
        sd.ellipse(xy((cx-66,baseline+body_h-5,cx+92,baseline+body_h+9)),fill=(61,65,51,25))
    shadows=shadows.filter(ImageFilter.GaussianBlur(5*S))
    im.alpha_composite(shadows)

    geometry=[]
    mask=Image.new('L',im.size,0)
    for i,(cat,parent,v,cx) in enumerate(zip(cats,parents,values,centers)):
        height=float(v)/scale_max*scale_h
        top=baseline-height
        # Full sleeve behind, pasta inside, then only its front panel on top.
        sx=cx-59
        paste(im,sleeve_scaled,sx,baseline-rim_px/S)
        bundle=pasta_at_height(pasta,pasta_width,height)
        bx=round((cx-pasta_width/2)*S)
        by=round(baseline*S)-bundle.height
        im.alpha_composite(bundle,(bx,by))
        mask.paste(bundle.getchannel('A'),(bx,by))
        paste(im,sleeve_front,sx,baseline)

        d=ImageDraw.Draw(im)
        text(d,(cx,top-34),f'{v:,.0f}',37,INK,True,'mb')
        flag=make_flag(parent)
        paste(im,flag,cx-29,1303)
        d=ImageDraw.Draw(im)
        d.rectangle(xy((cx-29,1303,cx+29,1339)),outline=(155,163,151),width=1)
        text(d,(cx,1365),parent,28,INK,True,'mm')
        if ' / ' in cat:
            text(d,(cx,1399),cat.split(' / ',1)[1],20,MUTED,False,'mm')
        geometry.append({'category':cat,'parent':parent,'value':v,
                         'center_x':cx,'sleeve_front_rim_y':baseline,
                         'ideal_highest_tip_y':top,'ideal_exposed_height_px':height,
                         'placed_highest_tip_y':by/S,'placed_height_px':bundle.height/S,
                         'pasta_width_px':pasta_width,'sleeve_height_px':body_h,
                         'flag_parent':parent,'display_value':f'{v:,.0f}'})

    d=ImageDraw.Draw(im)
    line(d,[(90,1428),(1910,1428)],RULE,1)
    text(d,(91,1450),'Illustrative synthetic values; not a statistical survey.',20,MUTED)
    text(d,(1910,1450),'USA subcategories are mutually exclusive parts of USA.',20,MUTED,False,'rt')

    final=im.convert('RGB').resize((W,H),RESAMPLE)
    final.save(ROOT/'final.png',optimize=True)
    artifacts=ROOT/'artifacts'
    artifacts.mkdir(exist_ok=True)
    mask.resize((W,H),RESAMPLE).save(artifacts/'visible_pasta_alpha.png')
    (artifacts/'geometry.json').write_text(json.dumps({
        'canvas':[W,H],'linear_axis':[0,scale_max], 'baseline_y':baseline,
        'full_scale_height_px':scale_h,'supersampling':S,'marks':geometry,
        'series_label':series['label'],'unit':series['unit'],
        'series_total':sum(values),'generated_raster_source':'assets/raw_01.png'
    },ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Rendered {W} × {H}; {len(geometry)} data-bound generated pasta bundles.')


if __name__=='__main__':
    main()
