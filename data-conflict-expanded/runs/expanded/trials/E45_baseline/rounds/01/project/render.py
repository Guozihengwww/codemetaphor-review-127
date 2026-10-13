#!/usr/bin/env python3
"""Offline infographic renderer. Run through capture_render.py for experiment captures."""
from __future__ import annotations
from decimal import Decimal
from pathlib import Path
from functools import lru_cache
import hashlib, io, json, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cairosvg

ROOT = Path(__file__).resolve().parent
S = 2
BG = '#EFF0E9'
INK = '#17384A'
BLUE = '#26ABE0'
MUTED = '#667A80'
LINE = '#E3E9E8'

@lru_cache(maxsize=128)
def font(size, weight=400):
    f = ImageFont.truetype(str(ROOT/'assets/fonts/UbuntuSans.ttf'), round(size*S))
    f.set_variation_by_axes([100,weight])
    return f

def box(coords):
    return tuple(round(v*S) for v in coords)

def main():
    input_map=json.loads((ROOT/'input.json').read_text(encoding='utf-8'))
    brief=json.loads((ROOT/input_map['brief']).read_text(encoding='utf-8'))
    data=json.loads((ROOT/input_map['data']).read_text(encoding='utf-8'))
    core=json.loads((ROOT/input_map['critical_metaphor_contract']).read_text(encoding='utf-8'))
    assert core['topic_zh']==data['topic_zh']
    assert data['synthetic'] and data['visible_notice']=='SYNTHETIC DATA'
    assert data['pictogram']['fraction_method']=='tight width left to right clipping'
    assert len(data['series'])==1
    W,H=brief['width'],brief['height']
    assert (W,H)==(2160,1800)
    image=Image.new('RGBA',(W*S,H*S),BG)
    d=ImageDraw.Draw(image)
    text_boxes=[]
    def text(x,y,string,size=30,weight=400,fill=INK,anchor='ls',role=''):
        f=font(size,weight)
        d.text((round(x*S),round(y*S)),str(string),font=f,fill=fill,anchor=anchor)
        bb=d.textbbox((round(x*S),round(y*S)),str(string),font=f,anchor=anchor)
        text_boxes.append({'text':str(string),'role':role,'bbox':[v/S for v in bb]})
        return d.textlength(str(string),font=f)/S
    def rounded(coords,radius,fill,outline=None,width=1):
        d.rounded_rectangle(box(coords),round(radius*S),fill=fill,outline=outline,width=round(width*S))

    # The title and notice are native text, independent of generated assets.
    rounded((140,78,480,131),26,'#D5EDF5')
    text(310,115,data['visible_notice'],25,700,INK,'ms','synthetic_notice')
    text(140,268,'Social account',109,720,INK,role='topic_title')
    text(136,411,'followers',149,760,BLUE,role='metric_title')
    text(144,478,'Followers, measured in millions',31,420,MUTED,role='metric_unit')

    # White data board, with a small contact shadow and uninterrupted rows.
    panel=(140,562,2020,1624)
    shadow=Image.new('RGBA',image.size,(0,0,0,0))
    sd=ImageDraw.Draw(shadow)
    sd.rounded_rectangle(box((142,568,2024,1634)),radius=24*S,fill=(37,67,75,16))
    shadow=shadow.filter(ImageFilter.GaussianBlur(14*S))
    image.alpha_composite(shadow)
    d=ImageDraw.Draw(image)
    rounded(panel,18,'#FFFFFF')
    text(208,627,'SOCIAL ACCOUNT',23,680,MUTED,role='category_header')
    text(800,627,'FOLLOWERS  /  MILLIONS',23,680,MUTED,role='series_header')
    d.line(box((180,668,1980,668)),fill=LINE,width=2*S)

    # Native original bird silhouette. Its alpha bounding box, not the SVG canvas,
    # is normalized to 78 px. Every partial mark is a left-to-right clip of this
    # exact template; no bird is squeezed or area-normalized.
    svg='''<svg xmlns="http://www.w3.org/2000/svg" width="104" height="92" viewBox="-2 -1 104 92"><path fill="#26ABE0" d="M99 16 C95 18 91 19 87 19 C92 15 95 10 96 5 C91 8 86 10 80 11 C76 6 69 3 62 4 C51 5 45 14 45 25 C45 28 45 31 46 33 C28 32 14 25 4 13 C-1 23 0 32 7 39 L1 36 C1 48 9 56 18 59 L10 59 C13 67 20 73 30 75 C18 82 6 80 0 79 C12 86 24 90 39 88 C74 85 88 57 88 34 L88 30 C93 26 97 21 99 16 Z"/></svg>'''
    mask_source=Image.open(io.BytesIO(cairosvg.svg2png(bytestring=svg.encode('utf-8'),output_width=832,output_height=736))).convert('RGBA')
    raw_bbox=mask_source.getchannel('A').getbbox()
    tight=mask_source.crop(raw_bbox)
    icon_w=78
    icon_h=round(icon_w*tight.height/tight.width)
    alpha=tight.getchannel('A').resize((icon_w*S,icon_h*S),Image.Resampling.LANCZOS)
    # Crop again to the actual resized visible extent and restore fixed width.
    ab=alpha.getbbox()
    alpha=alpha.crop(ab).resize((icon_w*S,icon_h*S),Image.Resampling.LANCZOS)
    template=Image.new('RGBA',alpha.size,BLUE); template.putalpha(alpha)
    # This file is a code-generated template, not a generated raster asset.
    (ROOT/'pictogram.svg').write_text(svg+'\n',encoding='utf-8')

    values=data['series'][0]['values']
    unit=Decimal(str(data['pictogram']['value']))
    n_slots=int(Decimal(str(data['value_axis']['maximum']))/unit)
    assert n_slots==12 and data['value_axis']['minimum']==0
    x0=800; pitch=96; row_start=668; row_h=190
    geometry=[]
    for i,(category,value) in enumerate(zip(data['categories'],values)):
        center=row_start+row_h*(i+.5)
        if i:
            d.line(box((180,row_start+i*row_h,1980,row_start+i*row_h)),fill=LINE,width=2*S)
        # One circular fictional avatar for each account, with restrained rings.
        avatar=Image.open(ROOT/f'assets/avatar_{chr(97+i)}.png').convert('RGB')
        diameter=112
        avatar=avatar.resize((diameter*S,diameter*S),Image.Resampling.LANCZOS)
        circle=Image.new('L',avatar.size,0)
        ImageDraw.Draw(circle).ellipse((0,0,diameter*S-1,diameter*S-1),fill=255)
        rounded((199,center-57,313,center+57),57,'#E6EFF2')
        image.paste(avatar,(200*S,round((center-diameter/2)*S)),circle)
        d=ImageDraw.Draw(image)
        text(343,center-13,category,39,680,INK,role='category')
        value_str=f'{Decimal(str(value)):.1f}'
        num_width=text(342,center+49,value_str,54,680,INK,role='value')
        text(352+num_width,center+49,'M',29,520,MUTED,role='value_unit')

        q=Decimal(str(value))/unit
        whole=int(q); fraction=q-whole
        marks=[]
        for j in range(whole+(fraction>0)):
            frac=Decimal(1) if j<whole else fraction
            mark=template.copy()
            if frac<1:
                a=np.array(alpha,dtype=np.float64)
                # Pixel-coverage clipping at the exact fractional coordinate.
                clip_px=float(frac)*a.shape[1]
                coverage=np.clip(clip_px-np.arange(a.shape[1],dtype=float),0,1)
                a=np.rint(a*coverage[None,:]).astype(np.uint8)
                mark.putalpha(Image.fromarray(a))
            xx=x0+j*pitch; yy=round(center-icon_h/2)
            image.alpha_composite(mark,(xx*S,yy*S))
            bb=mark.getchannel('A').getbbox()
            marks.append({'index':j+1,'fraction':float(frac),'origin':[xx,yy],
                          'visible_bbox':[xx+bb[0]/S,yy+bb[1]/S,xx+bb[2]/S,yy+bb[3]/S],
                          'clip_width_px':float(frac)*icon_w})
        geometry.append({'category':category,'value':value,'whole_birds':whole,
                         'last_fraction':float(fraction),'row_center':center,
                         'icon_width':icon_w,'icon_height':icon_h,'pitch':pitch,
                         'capacity_birds':n_slots,'marks':marks})

    # The large photographic bird overlaps the board. It has no data meaning.
    contact=Image.new('RGBA',image.size,(0,0,0,0))
    cd=ImageDraw.Draw(contact)
    cd.ellipse(box((1370,557,1875,612)),fill=(28,48,67,45))
    contact=contact.filter(ImageFilter.GaussianBlur(20*S))
    image.alpha_composite(contact)
    bird=Image.open(ROOT/'assets/bluebird.png').convert('RGBA')
    bird_w=860; bird_h=round(bird.height/bird.width*bird_w)
    bird=bird.resize((bird_w*S,bird_h*S),Image.Resampling.LANCZOS)
    bird_xy=(1260,26)
    image.alpha_composite(bird,(bird_xy[0]*S,bird_xy[1]*S))
    d=ImageDraw.Draw(image)

    # A compact legend explains the count and the fractional-width convention.
    legend_w=48; legend_h=round(icon_h*legend_w/icon_w)
    legend=template.resize((legend_w*S,legend_h*S),Image.Resampling.LANCZOS)
    image.alpha_composite(legend,(148*S,(1675-legend_h//2)*S))
    text(214,1685,'= 10 million followers',29,620,INK,role='legend')
    text(146,1730,'Partial birds are clipped by width. Large bird is illustrative.',24,420,MUTED,role='encoding_note')
    text(146,1765,'Fictional accounts and portraits. All values are synthetic.',22,400,MUTED,role='provenance_note')
    total=sum(Decimal(str(v)) for v in values)
    assert total==Decimal(str(data['series_totals']['followers']))
    text(2016,1694,f'{total:.1f}M',49,710,INK,'rs','total')
    text(2016,1730,'Sum of account counts',24,430,MUTED,'rs','total_label')
    text(2016,1765,'Row capacity: 120 million',22,400,MUTED,'rs','range')

    final=image.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
    final.save(ROOT/'final.png')
    (ROOT/'artifacts').mkdir(exist_ok=True)
    dependency_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['input.json','brief.json','data.json','core.json','reference.png']}
    report={'canvas':[W,H],'pictogram':{'width_px':icon_w,'height_px':icon_h,'unit_million':float(unit),'scale':'10 million per full symbol, fractional tight width','native_source_bbox':list(raw_bbox)},
            'panel_bbox':list(panel),'hero_bbox':[bird_xy[0],bird_xy[1],bird_xy[0]+bird_w,bird_xy[1]+bird_h],
            'rows':geometry,'text_boxes':text_boxes,'total_million':float(total),'frozen_input_sha256':dependency_hashes}
    (ROOT/'artifacts/geometry.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':'final.png','canvas':[W,H],'categories':len(geometry),'full_birds':sum(r['whole_birds'] for r in geometry),'partial_birds':sum(r['last_fraction']>0 for r in geometry),'total_million':float(total)},ensure_ascii=False))

if __name__=='__main__':
    main()
