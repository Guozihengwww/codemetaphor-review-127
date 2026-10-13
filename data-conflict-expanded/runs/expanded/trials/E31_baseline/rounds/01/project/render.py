#!/usr/bin/env python3
"""Offline infographic renderer. Run through the supplied capture_render.py.
The generated ocean material is color-graded and clipped to data-derived layers.
All data, numeric labels and category labels are read from frozen data.json.
"""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT=Path(__file__).resolve().parent
S=2
PALETTE=['#20AE92','#197ADD','#162E60','#9652C4']
BG='#FCFCF8'
INK='#172E3C'
MUTED='#566D76'
X0,X1,Y0,DEPTH=280.0,2100.0,370.0,800.0
LABEL_RECORDS=[]


def rgb(c): return tuple(int(c[i:i+2],16) for i in (1,3,5))
def p(v): return int(round(v*S))
def font(sz,kind='sans'):
    return ImageFont.truetype(str(ROOT/'assets/fonts'/f'{kind}.otf'),p(sz))

def main():
    inp=json.loads((ROOT/'input.json').read_text(encoding='utf-8'))
    data=json.loads((ROOT/inp['data']).read_text(encoding='utf-8'))
    brief=json.loads((ROOT/inp['brief']).read_text(encoding='utf-8'))
    core=json.loads((ROOT/inp['critical_metaphor_contract']).read_text(encoding='utf-8'))
    assert core['visible_title']==data['title']
    W,H=brief['width'],brief['height']
    assert (W,H)==(2200,1552)
    vals=np.array([s['values'] for s in data['series']],dtype=float)
    times=np.asarray(data['time_points'],float)
    cats=data['categories']
    assert vals.shape==(4,len(cats)) and len(times)==len(cats)
    assert np.all(np.diff(times)>0) and np.all(vals>=0)
    assert data['value_axis']=={'minimum':0,'maximum':7200,'scale':'linear'}
    scale=DEPTH/data['value_axis']['maximum']
    xs=X0+(times-times[0])/(times[-1]-times[0])*(X1-X0)
    bounds=np.vstack([np.zeros(len(cats)),np.cumsum(vals,axis=0)])
    totals=bounds[-1]
    assert np.max(totals)<=7200
    for k,s in enumerate(data['series']):
        assert int(vals[k].sum())==data['series_totals'][s['id']]
    im=Image.new('RGB',(p(W),p(H)),BG)
    draw=ImageDraw.Draw(im)
    def text(x,y,txt,size=24,color=INK,kind='sans',anchor='la',tag=None):
        f=font(size,kind)
        xy=(p(x),p(y))
        draw.text(xy,txt,font=f,fill=color,anchor=anchor)
        bb=draw.textbbox(xy,txt,font=f,anchor=anchor)
        LABEL_RECORDS.append({'text':txt,'box':[round(a/S,2) for a in bb],'role':tag or 'text'})
    def line(points,fill,width=1):
        draw.line([(p(x),p(y)) for x,y in points],fill=fill,width=max(1,p(width)))
    def rect(box,fill): draw.rectangle(tuple(p(v) for v in box),fill=fill)
    def circle(x,y,r,fill,outline=None,width=1):
        draw.ellipse((p(x-r),p(y-r),p(x+r),p(y+r)),fill=fill,outline=outline,width=p(width))
    def rounded(box,fill,r=8,outline=None,width=1):
        draw.rounded_rectangle(tuple(p(v) for v in box),radius=p(r),fill=fill,outline=outline,width=p(width))

    # Title and synthetic-data notice.
    rounded((80,42,326,83),'#E1F1EC',r=5)
    text(98,52,data['visible_notice'],21,'#195C51','sans_bold')
    text(80,100,'Plastic discharge',81,INK,'serif')
    text(80,184,'into the ocean',81,INK,'serif')
    text(84,277,'By source, by period  /  '+data['series'][0]['unit'],27,MUTED)
    text(2118,52,'16 PERIODS   /   4 SOURCES',19,MUTED,anchor='ra')

    # The legend spells out semantic roles, with order identical to the stack.
    legend_x=[84,382,680,978]
    for k,s in enumerate(data['series']):
        rect((legend_x[k],328,legend_x[k]+30,341),PALETTE[k])
        text(legend_x[k]+43,322,s['label'],23,INK)

    # Fixed axis and quiet guide lines sit outside / under the water object.
    ticks=list(range(0,7201,1200))
    for v in ticks:
        y=Y0+v*scale
        line([(259,y),(X1,y)],'#DDE4E3',0.8)
        text(242,y,f'{v:,}',21,MUTED,anchor='rm',tag='value_tick')
    line([(259,Y0),(259,Y0+DEPTH)],'#BECACB',1)
    text(80,380,'DISCHARGE',17,MUTED,'sans_bold')
    text(80,408,'Linear scale',19,MUTED)
    text(80,435,'increases',19,MUTED)
    text(80,462,'downward',19,MUTED)
    line([(122,508),(122,561)],MUTED,1.3)
    line([(115,553),(122,561),(129,553)],MUTED,1.3)
    for x in xs:
        line([(x,Y0),(x,Y0+DEPTH)],'#E8ECE8',0.8)

    # One shared mapping controls every source, all boundaries, and texture masks.
    cw,ch=p(X1-X0),p(DEPTH)
    xcoords=X0+(np.arange(cw,dtype=np.float32)+0.5)/S
    ycoords=(np.arange(ch,dtype=np.float32)+0.5)/S
    interp=np.array([np.interp(xcoords,xs,row) for row in bounds],dtype=np.float32)*scale
    ocean=Image.open(ROOT/'assets/ocean_material.png').convert('RGB').resize((cw,ch),Image.Resampling.LANCZOS)
    tex=np.asarray(ocean,dtype=np.float32)/255.0
    lum=0.2126*tex[:,:,0]+0.7152*tex[:,:,1]+0.0722*tex[:,:,2]
    # Preserve generated fluid detail through row-normalized luminance; no texture
    # coordinate is used for the chart's quantitative geometry.
    rowmean=np.mean(lum,axis=1,keepdims=True)
    factor=np.clip(0.93+1.52*(lum-rowmean),0.54,1.53)
    factor*=np.linspace(1.08,0.91,ch,dtype=np.float32)[:,None]
    painted=np.array(im.crop((p(X0),p(Y0),p(X1),p(Y0+DEPTH))),dtype=np.uint8)
    layer_pixels=[]
    for k in range(4):
        mask=(ycoords[:,None]>=interp[k][None,:])&(ycoords[:,None]<interp[k+1][None,:])
        base=np.array(rgb(PALETTE[k]),dtype=np.float32)
        shade=np.clip(factor[:,:,None]*base[None,None,:],0,255)
        # Bright refractive details are still derived from the generated image.
        glint=np.clip(lum-rowmean-0.13,0,1)[:,:,None]*64
        shade=np.clip(shade+glint,0,255).astype(np.uint8)
        painted[mask]=shade[mask]
        layer_pixels.append(int(mask.sum()))
    im.paste(Image.fromarray(painted),(p(X0),p(Y0)))
    draw=ImageDraw.Draw(im)

    # Very light guides in water: each follows an actual period position.
    guide=Image.new('RGBA',im.size,(0,0,0,0))
    gd=ImageDraw.Draw(guide)
    for j,x in enumerate(xs):
        gd.line([(p(x),p(Y0+1)),(p(x),p(Y0+totals[j]*scale))],fill=(255,255,255,40),width=1)
    im.paste(guide,(0,0),guide)
    draw=ImageDraw.Draw(im)
    # Crisp total edge; the total is a boundary, never an extra colored layer.
    total_xy=[(float(x),float(Y0+v*scale)) for x,v in zip(xs,totals)]
    line(total_xy,'#613C83',0.9)
    # Shared waterline remains rigorously horizontal at zero.
    line([(X0,Y0),(X1,Y0)],'#4FAF9F',1.1)
    text(289,Y0-27,'OCEAN SURFACE  /  ZERO',17,'#41746A','sans_bold')

    # Yacht touches the same surface and does not displace any data geometry.
    yacht=Image.open(ROOT/'assets/yacht_cutout.png').convert('RGBA')
    yacht_w=680
    yacht_h=yacht_w*yacht.height/yacht.width
    yacht=yacht.resize((p(yacht_w),p(yacht_h)),Image.Resampling.LANCZOS)
    yacht_x=1431
    yacht_y=Y0-yacht_h+0.5
    im.paste(yacht,(p(yacht_x),p(yacht_y)),yacht)
    draw=ImageDraw.Draw(im)
    # Small physical glints are confined to the first source, below zero.
    for a,b in [(1466,1518),(1543,1582),(1660,1711),(1777,1839),(1888,1960),(2012,2090)]:
        line([(a,Y0+2),(b,Y0+2)],'#A8DDD3',0.6)

    # Clear reading note occupies only the unfilled area below the early periods.
    text(306,646,'Four sources.',37,INK,'serif')
    text(306,689,'One period at a time.',37,INK,'serif')
    text(307,753,'Read downward from the surface.',24,MUTED)
    text(307,789,'Each layer’s thickness is its discharge.',24,MUTED)
    text(307,825,'The bottom edge is the period total.',24,MUTED)
    line([(307,883),(610,883)],'#CCD7D4',1)
    peak=int(np.argmax(totals))
    text(306,910,f'{int(totals[peak]):,}',70,INK,'serif')
    text(309,993,'Highest period total  /  '+cats[peak],23,MUTED)
    text(309,1030,'Sources add within each period;',21,MUTED)
    text(309,1061,'periods do not accumulate over time.',21,MUTED)

    # Minimal point labels, with exact complete values in the aligned table.
    j=0
    text(xs[j]+8,Y0+totals[j]*scale+19,f'{int(totals[j]):,}',22,'#613C83',tag='point_total')
    j=5
    circle(xs[j],Y0+totals[j]*scale,4,BG,'#613C83',1.3)
    text(xs[j]+10,Y0+totals[j]*scale+17,f'{int(totals[j]):,}',23,'#613C83',tag='point_total')
    j=peak
    circle(xs[j],Y0+totals[j]*scale,4,BG,'#613C83',1.3)
    text(xs[j],Y0+totals[j]*scale-22,f'{int(totals[j]):,}',23,'#FFFFFF','sans_bold',anchor='ms',tag='point_total')
    j=len(cats)-1
    circle(xs[j],Y0+totals[j]*scale,4,BG,'#613C83',1.3)
    text(xs[j]-4,Y0+totals[j]*scale+24,f'{int(totals[j]):,}',23,'#613C83',anchor='rt',tag='point_total')

    # In-water letters preserve the mapping when colors are difficult to distinguish.
    for k,s in enumerate(data['series']):
        y=Y0+scale*(bounds[k,-1]+bounds[k+1,-1])/2
        text(2072,y,s['id'],23,'#ECF8FC','sans_bold',anchor='rm',tag='layer_id')

    # The exact-value table doubles as the shared horizontal period axis.
    table_top=1190
    rect((80,table_top,2128,1233),'#EAF0ED')
    text(95,1203,'PERIOD / SOURCE',18,MUTED,'sans_bold')
    for j,cat in enumerate(cats):
        text(xs[j],1212,cat,23,INK,'sans_bold',anchor='mm',tag='period')
    row_y=[1258,1298,1338,1378]
    for k,s in enumerate(data['series']):
        if k%2==1:rect((80,row_y[k]-20,2128,row_y[k]+20),'#F2F5F0')
        rect((94,row_y[k]-7,109,row_y[k]+7),PALETTE[k])
        text(120,row_y[k],s['label'],20,INK,anchor='lm',tag='series')
        for j,v in enumerate(vals[k]):
            text(xs[j],row_y[k],f'{int(v):,}',24,INK,anchor='mm',tag='value')
    line([(80,1404),(2128,1404)],'#BCCBC7',1.1)
    text(95,1430,'Period total',22,INK,'sans_bold',anchor='lm',tag='total_label')
    for j,v in enumerate(totals):
        text(xs[j],1430,f'{int(v):,}',25,INK,'sans_bold',anchor='mm',tag='total')
    line([(80,1464),(2128,1464)],'#CFD9D5',1)
    text(80,1490,'SYNTHETIC DATA  /  Illustrative values, not observed ocean statistics.',20,MUTED)
    text(2128,1490,'Unit: '+data['series'][0]['unit']+'   ·   P01–P16 in supplied order',20,MUTED,anchor='ra')

    # Save diagnostic coordinates / text boxes, not a second chart render.
    geometry={
      'canvas':[W,H], 'plot':[X0,Y0,X1,Y0+DEPTH], 'axis_max':7200,
      'scale_px_per_unit':scale,'interpolation':'piecewise linear; each boundary uses every period',
      'categories':cats,'x_positions':xs.tolist(),'series_ids':[s['id'] for s in data['series']],
      'values':vals.astype(int).tolist(),'cumulative_values':bounds.astype(int).tolist(),
      'boundaries_y':(Y0+bounds*scale).tolist(),'palette':PALETTE,
      'supersample':S,'layer_pixel_areas_at_supersample':layer_pixels,
      'generated_texture_visible_pixels_estimate':sum(layer_pixels)/(S*S),
      'yacht_box':[yacht_x,yacht_y,yacht_x+yacht_w,Y0+0.5],
      'text_labels':LABEL_RECORDS,
    }
    (ROOT/'artifacts').mkdir(exist_ok=True)
    (ROOT/'artifacts/render_geometry.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    output=im.resize((W,H),Image.Resampling.LANCZOS)
    output.save(ROOT/'final.png',dpi=(300,300))
    print(json.dumps({'canvas':[W,H],'source_count':len(vals),'period_count':len(cats),'min_layer_px':float(vals.min()*scale),'max_total':int(totals.max()),'texture_pixels':sum(layer_pixels)/(S*S)}))

if __name__=='__main__': main()
