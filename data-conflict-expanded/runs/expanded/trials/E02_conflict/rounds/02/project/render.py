#!/usr/bin/env python3
"""Offline infographic renderer. Run with no arguments from the supplied case.

Numerical encoding: consecutive axial heights, from shoulder apex to flat base.
Generated material is recolored inside each exact, data-derived horizontal band.
"""
from pathlib import Path
from decimal import Decimal
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent
S = 3
BG = '#F8F0E4'
INK = '#25383E'
MUTED = '#65706C'
RED = '#B53F33'
RULE = '#D8CEC0'
PALETTE = [
    '#3C83AA',
    '#26796F', '#48988B', '#2F6A65', '#62A99B',
    '#BE624A', '#E38B62', '#AE493C', '#D57453',
    '#626FA6', '#8C9FC4',
    '#B88C35', '#DABC76',
    '#427D79', '#D97C69', '#9D4551', '#9290BE', '#569CB2',
    '#CEA644', '#85976D', '#C16E4D', '#436B93', '#97749B',
    '#4A977F', '#B99B50', '#77558C', '#AA85B4',
]


def read_json(path, decimals=False):
    return json.loads(path.read_text(encoding='utf-8'), **({'parse_float': Decimal} if decimals else {}))


def rgb(color):
    return tuple(int(color[j:j+2],16) for j in (1,3,5))


def badge_ink(color):
    """Select the more legible of white and page ink on a categorical badge."""
    def lum(c):
        s=np.array(rgb(c))/255.0
        lin=np.where(s<=.04045,s/12.92,((s+.055)/1.055)**2.4)
        return float(np.dot(lin,[.2126,.7152,.0722]))
    background=lum(color)
    white_contrast=1.05/(background+.05)
    dark_contrast=(background+.05)/(lum(INK)+.05)
    return '#FFFFFF' if white_contrast>=dark_contrast else INK


def pava(values):
    """Least-squares increasing regression for uncrossed leader destinations."""
    blocks=[]
    for i,v in enumerate(values):
        blocks.append([float(v),1,i,i+1])
        while len(blocks)>1 and blocks[-2][0]>blocks[-1][0]:
            b=blocks.pop(); a=blocks.pop(); n=a[1]+b[1]
            blocks.append([(a[0]*a[1]+b[0]*b[1])/n,n,a[2],b[3]])
    result=np.empty(len(values))
    for v,n,l,r in blocks: result[l:r]=v
    return result


def layout_labels(centers, minimum=74, lower=665, upper=1768):
    n=len(centers)
    offsets=np.arange(n)*minimum
    transformed=pava(np.array(centers)-offsets)
    transformed=np.clip(transformed,lower,upper-offsets[-1])
    return transformed+offsets


def main():
    inp=read_json(ROOT/'input.json')
    brief=read_json(ROOT/inp['brief'])
    core=read_json(ROOT/inp['critical_metaphor_contract'])
    data=read_json(ROOT/inp['data'],decimals=True)
    assert (ROOT/inp['reference']).is_file(), 'Frozen visual reference must remain available'
    assert core['topic_zh']==data['topic_zh']
    assert data['synthetic'] is True
    series=data['series'][0]
    assert len(data['series'])==1
    categories=data['categories']; parents=data['category_parent']
    decimal_values=[Decimal(str(v)) for v in series['values']]
    total=sum(decimal_values)
    assert total==Decimal(str(data['series_totals'][series['id']]))
    assert Decimal(str(data['value_axis']['minimum']))==0
    assert Decimal(str(data['value_axis']['maximum']))==total
    assert len(categories)==len(parents)==len(decimal_values)==len(PALETTE)
    assert all(v>0 for v in decimal_values)
    W,H=brief['width'],brief['height']
    canvas=Image.new('RGB',(W*S,H*S),BG)
    draw=ImageDraw.Draw(canvas)
    fonts={}
    text_boxes=[]

    def font(size,style='regular'):
        key=(size,style)
        if key not in fonts:
            name={'regular':'Text-Regular.otf','bold':'Text-Bold.otf','display':'Display-Bold.otf'}[style]
            fonts[key]=ImageFont.truetype(str(ROOT/'assets/fonts'/name),round(size*S))
        return fonts[key]

    def text(x,y,string,size=22,style='regular',fill=INK,anchor='lt',role=None):
        f=font(size,style)
        xy=(round(x*S),round(y*S))
        draw.text(xy,string,font=f,fill=fill,anchor=anchor)
        box=draw.textbbox(xy,string,font=f,anchor=anchor)
        text_boxes.append({'text':string,'role':role or 'annotation','bbox':[round(k/S,2) for k in box]})
        return box

    def tracked(x,y,string,size=14,tracking=1.5,fill=MUTED,anchor='left',style='bold'):
        f=font(size,style)
        lengths=[draw.textlength(ch,font=f)/S for ch in string]
        width=sum(lengths)+tracking*(len(string)-1)
        if anchor=='right': x-=width
        elif anchor=='center': x-=width/2
        for ch,n in zip(string,lengths):
            text(x,y,ch,size,style,fill,role='eyebrow'); x+=n+tracking

    def line(points,fill=RULE,width=1):
        draw.line([(round(x*S),round(y*S)) for x,y in points],fill=fill,width=max(1,round(width*S)),joint='curve')

    def ellipse(box,fill,outline=None,width=1):
        draw.ellipse(tuple(round(v*S) for v in box),fill=fill,outline=outline,width=max(1,round(width*S)))

    # Editorial title and an unmistakable synthetic-data notice.
    draw.rounded_rectangle((64*S,48*S,346*S,90*S),radius=6*S,fill=RED)
    tracked(205,61,data['visible_notice'],17,2.0,'#FFFFFF','center')
    tracked(1376,62,f'{len(categories)} CATEGORIES  /  1 ILLUSTRATIVE SERIES',14,1.3,anchor='right')
    title=data['title'].removeprefix('SYNTHETIC ').upper()
    text(720,120,title,116,'display',RED,anchor='mt',role='title')
    text(720,235,f"{series['label']} · trillion cubic meters ({series['unit']})",26,anchor='mt',role='metric_and_unit')
    line([(64,296),(1376,296)],width=1.2)

    total_string=format(total.normalize(),'f')
    text(64,369,total_string,94,'display',INK,role='total')
    text(226,424,series['unit'],29,'bold',INK,role='unit')
    tracked(64,474,'TOTAL ILLUSTRATIVE RESERVES',13,1.1)
    text(64,505,'One total, divided by region.',20,fill=MUTED)

    tracked(1040,368,'READING THE CYLINDER',13,1.25)
    text(1040,402,'Band height shows share.',23,'bold')
    text(1040,439,'Guard and valve are excluded.',19,fill=MUTED)
    text(1040,468,'Order runs from top to bottom.',19,fill=MUTED)
    text(1040,503,f'Linear scale · 0–{total_string} {series["unit"]}',19,fill=MUTED)

    # The only data coordinate transform used by the renderer.
    body_x=480; body_y=560; body_w=480; body_h=1240
    boundaries=[body_y]
    accumulated=Decimal(0)
    for value in decimal_values:
        accumulated+=value
        boundaries.append(body_y+float(accumulated/total)*body_h)
    centers=[(a+b)/2 for a,b in zip(boundaries,boundaries[1:])]

    # Subtle ground contact shadow, entirely behind the cylinder.
    shadow=Image.new('RGBA',canvas.size,(0,0,0,0))
    sd=ImageDraw.Draw(shadow)
    sd.ellipse((490*S,1790*S,950*S,1810*S),fill=(62,47,40,34))
    shadow=shadow.filter(ImageFilter.GaussianBlur(10*S))
    canvas.paste(shadow,(0,0),shadow)

    # Generated hardware retains its original red/brass colors.
    hardware=Image.open(ROOT/'assets/hardware.png').convert('RGBA').resize((body_w*S,224*S),Image.Resampling.LANCZOS)
    canvas.paste(hardware,(body_x*S,(body_y-224)*S),hardware)

    # Generated body supplies the shoulder shape and shading over all 27 bands.
    material=Image.open(ROOT/'assets/body_material.png').convert('RGBA').resize((body_w*S,body_h*S),Image.Resampling.LANCZOS)
    mat=np.asarray(material)
    luminance=np.dot(mat[...,:3].astype(np.float32),np.array([.2126,.7152,.0722],dtype=np.float32))
    shading=np.clip((luminance/185.0)**0.8,.53,1.20)
    colors=np.array([rgb(c) for c in PALETTE],dtype=np.float32)
    row_centers=body_y+(np.arange(body_h*S)+.5)/S
    band_indices=np.searchsorted(np.array(boundaries[1:-1]),row_centers,side='right')
    painted=np.clip(colors[band_indices,None,:]*shading[...,None],0,255).astype(np.uint8)
    body_rgba=np.dstack([painted,mat[...,3]])
    body=Image.fromarray(body_rgba,'RGBA')
    canvas.paste(body,(body_x*S,body_y*S),body)
    draw=ImageDraw.Draw(canvas)

    # Label columns remain semantic, with the complete parent + subcategory text.
    for left,right in ((64,392),(1040,1376)):
        tracked(left,609,'REGION / SUBCATEGORY',12,1.15)
        tracked(right,609,series['unit'],12,1.15,anchor='right')
        line([(left,637),(right,637)],width=1)
    label_y={}
    for side in (0,1):
        inds=list(range(side,len(categories),2))
        positions=layout_labels([centers[i] for i in inds])
        label_y.update(zip(inds,positions))

    geometry=[]
    for i,(name,parent,value,color,cy) in enumerate(zip(categories,parents,decimal_values,PALETTE,centers)):
        side=i%2
        ly=float(label_y[i])
        local_y=max(0,min(body_h*S-1,round((cy-body_y)*S)))
        occupied=np.flatnonzero(mat[local_y,:,3]>127)
        if side==0:
            sx=body_x+float(occupied[0])/S
            ex=413; badge_x=80; name_x=111; value_x=392; rule_x0=64; rule_x1=392
            direction=-1
        else:
            sx=body_x+float(occupied[-1]+1)/S
            ex=1027; badge_x=1056; name_x=1087; value_x=1376; rule_x0=1040; rule_x1=1376
            direction=1
        # Ordered cubic leaders; common interpolation keeps neighboring paths uncrossed.
        ey=ly-3
        p0=np.array([sx,cy]); p1=np.array([sx+direction*24,cy])
        p2=np.array([ex-direction*20,ey]); p3=np.array([ex,ey])
        ts=np.linspace(0,1,90)[:,None]
        curve=(1-ts)**3*p0+3*(1-ts)**2*ts*p1+3*(1-ts)*ts**2*p2+ts**3*p3
        c=rgb(color)
        leader_color=tuple(round(.80*k+.20*b) for k,b in zip(c,rgb(INK)))
        line(curve,fill=leader_color,width=1.55)
        ellipse((sx-3,cy-3,sx+3,cy+3),fill=color,outline=BG,width=1)
        ellipse((ex-2.3,ey-2.3,ex+2.3,ey+2.3),fill=color)
        ellipse((badge_x-14,ly-19,badge_x+14,ly+9),fill=color)
        text(badge_x,ly-12.3,f'{i+1:02d}',13,'bold',badge_ink(color),anchor='mt',role='category_order')
        text(name_x,ly-23,parent,23,'bold',role='category_parent')
        text(value_x,ly-25,str(value),28,'bold',anchor='rt',role='category_value')
        suffix=name[len(parent):].removeprefix(' / ')
        if suffix:
            text(name_x,ly+6,suffix,18,fill=MUTED,role='subcategory')
        line([(rule_x0,ly+34),(rule_x1,ly+34)],fill='#E6DCCF',width=.8)
        geometry.append({
            'index':i+1,'category':name,'parent':parent,'value':str(value),'unit':series['unit'],
            'color':color,'y0':boundaries[i],'y1':boundaries[i+1],
            'height_px':boundaries[i+1]-boundaries[i],
            'share':float(value/total),'label_side':'left' if side==0 else 'right',
            'label_y':ly,'source_point':[sx,cy],'label_endpoint':[ex,ey],
        })

    # Compact reading notes retain metric and synthetic-subcategory semantics.
    line([(64,1848),(1376,1848)],width=1.2)
    tracked(64,1874,'ENCODING',12,1.4)
    text(202,1868,f'Band height = value ÷ {total_string} × body height. Band area does not encode reserves.',20)
    text(64,1910,'tcm = trillion cubic meters. Subcategories are mutually exclusive parts of their named parent.',20,fill=MUTED)
    text(64,1948,'SYNTHETIC DATA · Illustrative categories, not a statistical survey.',17,'bold',fill=RED)
    text(1376,1948,'TOTAL  '+total_string+' '+series['unit'],17,'bold',fill=INK,anchor='rt')

    out=canvas.resize((W,H),Image.Resampling.LANCZOS)
    out.save(ROOT/'final.png')
    artifacts=ROOT/'artifacts'; artifacts.mkdir(exist_ok=True)
    (artifacts/'geometry.json').write_text(json.dumps({
        'canvas':[W,H],'body_box':[body_x,body_y,body_w,body_h],
        'encoding':'axial height; no minimum size; shoulder apex to flat bottom; area is not data',
        'total':str(total),'unit':series['unit'],'category_count':len(categories),
        'bands':geometry,'text_boxes':text_boxes,
    },ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Rendered {W} × {H}; {len(categories)} data bands; total {total} {series["unit"]}.')


if __name__=='__main__':
    main()
