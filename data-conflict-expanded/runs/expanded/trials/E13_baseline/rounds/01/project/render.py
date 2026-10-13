#!/usr/bin/env python3
"""Offline, data-bound cutaway oil drum infographic.
Run through capture_render.py to preserve each complete rendering attempt.
"""
from pathlib import Path
from decimal import Decimal
import hashlib
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent
S = 2
PALETTE = ['#008C91', '#20A660', '#F4D638', '#F39B29', '#E86A4E', '#D62E46', '#96379F']
INK = '#171E22'
MUTED = '#626A6C'
PAPER = '#F6F5F0'
CX = 685.0
TOP = 548.0
CURVE_A = 44.0
CURVE_HALF = 285.0
H = 1220.0
LEFT = [(520,418),(660,415),(790,402),(820,386),(965,380),(1090,395),
        (1160,413),(1270,408),(1350,398),(1520,399),(1580,420),(1670,418),(1775,450),(1840,475)]
RIGHT = [(520,947),(650,945),(790,938),(830,949),(975,942),(1110,950),
         (1175,944),(1320,938),(1460,932),(1610,948),(1740,940),(1840,930)]


def rgb(h):
    return np.array([int(h[i:i+2],16) for i in (1,3,5)],dtype=np.float32)


def curve(x):
    return TOP + CURVE_A * ((np.asarray(x) - CX) / CURVE_HALF) ** 2


def font(size, bold=False):
    return ImageFont.truetype(str(ROOT/'assets/fonts'/('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')), round(size*S))


def value_string(v):
    return format(Decimal(str(v)).normalize(), 'f')


def txt(draw, xy, text, size, fill=INK, bold=False, anchor='lt', **kw):
    draw.text((round(xy[0]*S),round(xy[1]*S)),text,font=font(size,bold),fill=fill,anchor=anchor,**kw)


def tracked(draw, xy, text, size, fill=INK, spacing=2.2, bold=False):
    x,y=xy
    f=font(size,bold)
    for char in text:
        draw.text((round(x*S),round(y*S)),char,font=f,fill=fill,anchor='lt')
        x += draw.textlength(char,font=f)/S + spacing
    return x


def line(draw, points, fill, width=1):
    draw.line([(round(x*S),round(y*S)) for x,y in points],fill=fill,width=max(1,round(width*S)),joint='curve')


def bezier(a,b,c,d,n=32):
    return [tuple((1-t)**3*a[k]+3*(1-t)**2*t*b[k]+3*(1-t)*t*t*c[k]+t**3*d[k] for k in (0,1)) for t in np.linspace(0,1,n)]


def droplet(canvas, x, y, color):
    w,h=66,82
    mask=Image.new('L',(w*S,h*S),0)
    md=ImageDraw.Draw(mask)
    pts=bezier((33,5),(40,19),(55,32),(55,48))
    pts+=bezier((55,48),(55,76),(11,76),(11,48))
    pts+=bezier((11,48),(11,32),(26,19),(33,5))
    md.polygon([(round(a*S),round(b*S)) for a,b in pts],fill=255)
    yy,xx=np.mgrid[0:h*S,0:w*S].astype(float)/S
    base=rgb(color)
    bright=np.exp(-(((xx-24)/16)**2+((yy-32)/22)**2))
    shade=.72+.27*bright+.12*(1-xx/w)
    arr=np.clip(base[None,None,:]*shade[:,:,None]+bright[:,:,None]*23,0,255).astype('uint8')
    obj=Image.fromarray(arr).convert('RGBA')
    obj.putalpha(mask)
    sh=Image.new('RGBA',obj.size,(20,32,32,0))
    sh.putalpha(mask.filter(ImageFilter.GaussianBlur(2*S)).point(lambda p:int(p*.15)))
    canvas.alpha_composite(sh,(round((x-w/2+1)*S),round((y-h/2+3)*S)))
    canvas.alpha_composite(obj,(round((x-w/2)*S),round((y-h/2)*S)))
    # A small reflected glint makes the legend marks read as oil drops.
    gloss=Image.new('RGBA',obj.size,(0,0,0,0))
    gd=ImageDraw.Draw(gloss)
    gd.ellipse((21*S,26*S,28*S,41*S),fill=(255,255,255,88))
    gloss=gloss.filter(ImageFilter.GaussianBlur(1.2*S))
    ga=np.asarray(gloss).copy()
    ga[:,:,3]=(ga[:,:,3].astype(float)*np.asarray(mask)/255).astype('uint8')
    canvas.alpha_composite(Image.fromarray(ga),(round((x-w/2)*S),round((y-h/2)*S)))


def main():
    inputs=json.loads((ROOT/'input.json').read_text(encoding='utf-8'))
    brief=json.loads((ROOT/inputs['brief']).read_text(encoding='utf-8'))
    data=json.loads((ROOT/inputs['data']).read_text(encoding='utf-8'))
    core=json.loads((ROOT/inputs['critical_metaphor_contract']).read_text(encoding='utf-8'))
    assert (ROOT/inputs['reference']).is_file()
    assert core['semantic_roles_zh'] == ['Synthetic output volume (US gal)']
    assert data['synthetic'] is True and data['value_axis']['scale']=='linear'
    categories=data['categories']
    series=data['series'][0]
    values=series['values']
    total=sum(Decimal(str(v)) for v in values)
    assert len(categories)==len(values)==7
    assert total==Decimal(str(data['series_totals'][series['id']]))==Decimal('44.0')
    assert data['value_axis']['minimum']==0 and Decimal(str(data['value_axis']['maximum']))==total
    cumulative=np.array([float(sum(Decimal(str(v)) for v in values[:i])/total)*H for i in range(len(values)+1)])
    W,BH=brief['width'],brief['height']
    canvas=Image.new('RGBA',(W*S,BH*S),PAPER)
    draw=ImageDraw.Draw(canvas)

    # Editorial header. All statistical text comes from the frozen input.
    draw.rounded_rectangle((110*S,80*S,442*S,132*S),radius=26*S,fill='#DCECE7')
    tracked(draw,(134,94),data['visible_notice'],22,fill='#176058',spacing=2.0,bold=True)
    txt(draw,(110,177),'Products from a',91,bold=True)
    txt(draw,(110,289),'barrel of crude oil',91,bold=True)
    txt(draw,(114,415),series['label']+' · US gallons ('+series['unit']+')',30,fill=MUTED)
    line(draw,[(1368,201),(1368,397)],'#CACEC8',1.3)
    txt(draw,(1420,207),value_string(total),89,bold=True)
    txt(draw,(1425,313),series['unit'],33,bold=True)
    tracked(draw,(1426,369),'TOTAL OUTPUT',17,fill=MUTED,spacing=1.8,bold=True)
    line(draw,[(110,478),(1690,478)],'#D3D7D0',1.2)

    # Grounding shadow and generated metal object material.
    shadow=Image.new('RGBA',canvas.size,(0,0,0,0))
    sd=ImageDraw.Draw(shadow)
    sd.ellipse((155*S,1775*S,1060*S,1898*S),fill=(35,43,40,33))
    shadow=shadow.filter(ImageFilter.GaussianBlur(29*S))
    canvas.alpha_composite(shadow)
    barrel=Image.open(ROOT/'assets/barrel_shell.png').convert('RGBA')
    barrel=barrel.resize((890*S,1380*S),Image.Resampling.LANCZOS)
    canvas.alpha_composite(barrel,(145*S,505*S))

    # The single data mapping: t = y - curve(x), height = H * value / total.
    # All eight boundaries are vertical translations of precisely the same curve.
    x0,y0,x1,y1=365,538,965,1850
    ny,nx=(y1-y0)*S,(x1-x0)*S
    yy,xx=np.mgrid[0:ny,0:nx].astype(np.float32)
    xx=xx/S+x0+.5/S
    yy=yy/S+y0+.5/S
    top=curve(xx)
    t=yy-top
    lft=np.interp(yy,[p[0] for p in LEFT],[p[1] for p in LEFT])
    rgt=np.interp(yy,[p[0] for p in RIGHT],[p[1] for p in RIGHT])
    valid=(t>=0)&(t<H)&(xx>=lft)&(xx<=rgt)
    ids=np.clip(np.searchsorted(cumulative,t,side='right')-1,0,len(values)-1)
    palette=np.stack([rgb(c) for c in PALETTE])
    # Smooth cylindrical reflection, with no geometry-dependent restyling.
    shine=np.exp(-((xx-632)/192)**2)
    dim=.69+.27*shine
    broad=.5+.5*np.cos((xx-650)/320)
    arr=palette[ids]*dim[:,:,None]+(shine*19+broad*3)[:,:,None]
    rgba=np.zeros((ny,nx,4),dtype='uint8')
    rgba[:,:,:3]=np.clip(arr,0,255).astype('uint8')
    rgba[:,:,3]=valid.astype('uint8')*255
    face=Image.fromarray(rgba,'RGBA')
    canvas.alpha_composite(face,(x0*S,y0*S))

    # Cut metal lips hug the opening. They do not establish any data height.
    draw=ImageDraw.Draw(canvas)
    edge_paths=[]
    for control,side in ((LEFT,'left'),(RIGHT,'right')):
        ys=np.linspace(540,1840,1501)
        xs=np.interp(ys,[p[0] for p in control],[p[1] for p in control])
        ok=(ys>=curve(xs))&(ys<=curve(xs)+H)
        points=list(zip(xs[ok],ys[ok]))
        edge_paths.append(points)
        outward=-1 if side=='left' else 1
        line(draw,[(x+outward*4.1,y) for x,y in points],'#191E20',8.0)
        line(draw,[(x+outward*3.0,y) for x,y in points],'#7D8586',3.4)
        line(draw,[(x+outward*.5,y) for x,y in points],'#D2D5D1',1.1)
    # Matching bottom cut: same curve + H; a metal lip is wholly outside the data.
    bx=np.linspace(420,960,1200)
    by=curve(bx)+H
    bl=np.interp(by,[p[0] for p in LEFT],[p[1] for p in LEFT])
    br=np.interp(by,[p[0] for p in RIGHT],[p[1] for p in RIGHT])
    keep=(bx>=bl)&(bx<=br)
    bottom=list(zip(bx[keep],by[keep]))
    line(draw,[(x,y+4) for x,y in bottom],'#1C2022',7)
    line(draw,[(x,y+2) for x,y in bottom],'#8A9293',2)
    # A subtle top meniscus identifies the exposed contents without adding a cap.
    tx=np.linspace(410,955,1000)
    ty=curve(tx)
    tl=np.interp(ty,[p[0] for p in LEFT],[p[1] for p in LEFT])
    tr=np.interp(ty,[p[0] for p in RIGHT],[p[1] for p in RIGHT])
    keep=(tx>tl+2)&(tx<tr-2)
    line(draw,list(zip(tx[keep],ty[keep]+.7)),'#87C5C1',1.1)

    # Direct values at exact data midpoints; sizes fit the genuine thin bands.
    sizes=[52,25,40,46,29,28,63]
    colors=['#FFFFFF','#112C22','#343315','#302411','#38231D','#FFFFFF','#FFFFFF']
    value_x=673
    bound_y=curve(value_x)+cumulative
    value_records=[]
    for i,v in enumerate(values):
        yc=float((bound_y[i]+bound_y[i+1])/2)
        val=value_string(v)
        txt(draw,(value_x,yc),val,sizes[i],colors[i],bold=True,anchor='mm')
        value_records.append({'category':categories[i],'value':v,'unit':series['unit'],
                              'text':val,'center_px':[value_x,yc],
                              'theoretical_thickness_px':float(cumulative[i+1]-cumulative[i])})

    # Oil-drop legend, preserving the supplied top-to-bottom category order.
    tracked(draw,(1095,548),'OUTPUT BY PRODUCT',21,fill=MUTED,spacing=2.0,bold=True)
    txt(draw,(1690,548),series['unit'],23,fill=MUTED,anchor='rt')
    rows=[650,794,938,1082,1226,1386,1560]
    label_lines=[['Diesel'],['Other Distillates'],['Jet Fuel'],['Other Products'],
                 ['Heavy Fuel Oil','(Residual)'],['Liquefied Petroleum','Gases (LPG)'],['Gasoline']]
    for i,(cat,y) in enumerate(zip(categories,rows)):
        droplet(canvas,1119,y+26,PALETTE[i])
        draw=ImageDraw.Draw(canvas)
        lines=label_lines[i]
        for j,label in enumerate(lines):
            txt(draw,(1170,y+j*40),label,30,INK,bold=(j==0))
        txt(draw,(1690,y+1),value_string(values[i]),35,INK,bold=True,anchor='rt')
        low=y+(83 if len(lines)>1 else 70)
        line(draw,[(1170,low),(1690,low)],'#D8DCD5',1)

    # Plain-language reading note and explicit synthetic provenance.
    txt(draw,(1095,1723),'Read the cutaway',25,bold=True)
    txt(draw,(1095,1768),'Layer thickness shows volume.',25,fill=MUTED)
    txt(draw,(1095,1806),'Top to bottom follows the list.',25,fill=MUTED)
    line(draw,[(110,1930),(1690,1930)],'#D3D7D0',1.2)
    txt(draw,(110,1963),'Synthetic illustration. Values are not real refinery statistics.',24,fill=MUTED)
    txt(draw,(1690,1963),'Metal shell and droplets are not data.',22,fill=MUTED,anchor='rt')

    out=canvas.convert('RGB').resize((W,BH),Image.Resampling.LANCZOS)
    out.save(ROOT/'final.png',optimize=True)
    metadata={
        'canvas_px':[W,BH], 'input_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest()
           for n in ('input.json','brief.json','data.json','core.json','reference.png')},
        'mapping':{'axis_min':0,'axis_max':float(total),'height_px':H,
                   'curve':'548 + 44 * ((x - 685) / 285)^2',
                   'cumulative_offsets_px':cumulative.tolist(),
                   'visible_unobstructed_test_columns_px':[480,810,860],
                   'category_order':'top to bottom, as supplied'},
        'labels':value_records,
        'generated_material':'assets/barrel_shell.png',
        'footer':'Synthetic illustration. Values are not real refinery statistics.'
    }
    (ROOT/'artifacts').mkdir(exist_ok=True)
    (ROOT/'artifacts/render_geometry.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Rendered final.png at',W,'x',BH,'using',len(values),'data-bound bands.')


if __name__=='__main__':
    main()
