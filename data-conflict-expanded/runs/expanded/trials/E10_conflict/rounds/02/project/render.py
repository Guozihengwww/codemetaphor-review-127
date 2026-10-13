#!/usr/bin/env python3
"""Offline data-bound editorial oil-barrel portrait. Run through capture_render.py."""
from pathlib import Path
from decimal import Decimal
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent

def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))

def main():
    inputs=read('input.json')
    brief=read(inputs['brief'])
    data=read(inputs['data'])
    core=read(inputs['critical_metaphor_contract'])
    assert core['visible_title']==data['title']
    assert data['synthetic'] and len(data['series'])==1
    series=data['series'][0]
    values=[Decimal(str(v)) for v in series['values']]
    total=sum(values)
    assert total==Decimal(str(data['series_totals'][series['id']]))==100
    assert len(values)==len(data['categories'])==len(data['category_parent'])==11
    W,H=brief['width'],brief['height']
    S=2
    bg=(245,240,228)
    ink='#242E30'; muted='#6A706C'; rule='#CFCBBC'; accent='#A54F3D'
    page=Image.new('RGB',(W*S,H*S),bg)
    draw=ImageDraw.Draw(page)
    fonts={}
    def font(size,style='sans'):
        key=(size,style)
        if key not in fonts:
            fonts[key]=ImageFont.truetype(str(ROOT/'assets/fonts'/({'sans':'sans.otf','bold':'bold.otf','mono':'mono.ttf'}[style])),round(size*S))
        return fonts[key]
    text_boxes=[]
    def text(x,y,content,size=28,style='sans',fill=ink,anchor='ls',track=0):
        f=font(size,style)
        if not track:
            draw.text((x*S,y*S),content,font=f,fill=fill,anchor=anchor)
            b=draw.textbbox((x*S,y*S),content,font=f,anchor=anchor)
            text_boxes.append({'text':content,'bbox':[round(k/S,2) for k in b]})
        else:
            xx=x*S
            for c in content:
                draw.text((xx,y*S),c,font=f,fill=fill,anchor='ls')
                xx+=draw.textlength(c,font=f)+track*S
    def line(points,fill=rule,width=1):
        draw.line([(round(x*S),round(y*S)) for x,y in points],fill=fill,width=max(1,round(width*S)))
    def ellipse(box,fill=None,outline=None,width=1):
        draw.ellipse(tuple(round(v*S) for v in box),fill=fill,outline=outline,width=round(width*S))
    def asset(name,box):
        im=Image.open(ROOT/'assets'/name).convert('RGBA')
        im=im.resize((round(box[2]*S),round(box[3]*S)),Image.Resampling.LANCZOS)
        page.paste(im,(round(box[0]*S),round(box[1]*S)),im)
    def path(points,fill,width=1):
        line(points,fill,width)

    # Fixed editorial typography: all data-bearing text below is bound to data.json.
    draw.rounded_rectangle((104*S,65*S,413*S,111*S),radius=6*S,fill='#A44E3D')
    text(123,97,data['visible_notice'],24,'bold',fill='#FFF8E8',track=1.2)
    text(103,218,data['title'].replace('SYNTHETIC ',''),100,'bold')
    text(106,272,f"{series['label']} ({series['unit']})",31)
    text(1945,132,'TOTAL SHARE',23,'bold',fill=muted,anchor='rs')
    text(1948,235,f'{total.normalize():f}%',83,'bold',anchor='rs')
    line([(104,310),(1946,310)],ink,1.5)

    # Generated portrait parts: all visible headwear, beard and shoulders come from raw_01.
    asset('beard_shoulders.png',(158,1390,1270,484))
    asset('turban.png',(300,344,875,806))

    # One common axial coordinate system, without minimum-size inflation.
    x0,x1,y0,body_h,depth=515.,1105.,605.,900.,53.
    cx=(x0+x1)/2; rx=(x1-x0)/2
    boundaries=[y0]
    for v in values:
        boundaries.append(boundaries[-1]+float(v/total)*body_h)
    colors=['#BA614B','#CDA27C','#ECD2AE','#496E70','#98B0A7','#CCD7C7','#C69849','#E8C77B','#657D70','#ACB99E','#985747']
    rgbs=np.array([tuple(int(c[i:i+2],16) for i in (1,3,5)) for c in colors],dtype=np.float32)
    def arc(x):
        return depth*math.sqrt(max(0,1-((x-cx)/rx)**2))
    xs=np.linspace(x0,x1,721)

    # A steel lid and rolled edge lie outside the measured data-bearing front wall.
    ellipse((x0-3,y0-depth-3,x1+3,y0+depth+3),fill='#334848')
    ellipse((x0+5,y0-depth+6,x1-5,y0+depth-6),fill='#81928A',outline='#CED0BC',width=2.0)
    ellipse((x0+21,y0-depth+13,x1-21,y0+depth-13),fill='#B5BAA5',outline='#4A615B',width=1.3)
    ellipse((x0+54,y0-17,x0+140,y0+1),fill='#3F5652',outline='#D1D0B5',width=1.5)
    ellipse((x0+61,y0-13,x0+133,y0-1),fill='#253C39')

    # Rasterize analytic front-surface masks. For every x, interfaces are translations
    # of the SAME ellipse arc: y_i(x) = y0 + H * cumulative_share_i + arc(x).
    aw=round((x1-x0)*S)+1
    ah=round((body_h+depth)*S)+3
    xx=(np.arange(aw,dtype=np.float32)+0.5)/S+x0
    yy=(np.arange(ah,dtype=np.float32)+0.5)/S+y0
    u=(xx-cx)/rx
    curved=depth*np.sqrt(np.clip(1-u*u,0,1))
    axial=yy[:,None]-curved[None,:]
    bid=np.searchsorted(np.array(boundaries[1:]),axial,side='right')
    valid=(axial>=y0)&(axial<y0+body_h)&(np.abs(u)[None,:]<=1)
    bid=np.clip(bid,0,len(values)-1)
    shade=.72+.28*np.sqrt(np.clip(1-u*u,0,1))+.085*np.exp(-((u+.36)/.23)**2)-.035*u
    noise=np.random.default_rng(124).normal(0,.52,(ah,aw)).astype(np.float32)
    rgb=np.clip(rgbs[bid]*shade[None,:,None]+noise[:,:,None],0,255).astype(np.uint8)
    body=np.dstack([rgb,(valid*255).astype(np.uint8)])
    page.paste(Image.fromarray(body),(round(x0*S),round(y0*S)),Image.fromarray(body[:,:,3]))
    # Fine consistent seams, rather than arbitrary thick non-quantitative bands.
    for y in boundaries:
        pts=[(float(x),y+arc(float(x))) for x in xs]
        path(pts,'#443F32',1.1)
        if y>y0 and y<y0+body_h:
            path([(x,py+1) for x,py in pts],'#E8DFBA',.55)
    line([(x0,y0),(x0,y0+body_h)],'#334341',2)
    line([(x1,y0),(x1,y0+body_h)],'#334341',2)
    # Both rolled rims are explicitly outside the H-height face.
    for off,col,w in [(-4,'#2D4240',2),(-2,'#E2DBC4',1.6)]:
        path([(float(x),y0+arc(float(x))+off) for x in xs],col,w)
    for off,col,w in [(2,'#E4D9B8',1.5),(4,'#364844',2),(6,'#718179',1.2)]:
        path([(float(x),y0+body_h+arc(float(x))+off) for x in xs],col,w)

    # Small barrel stencil: material identity only; it carries no additional data.
    text(650,747,'OIL',39,'bold',fill='#743D31',track=7)
    line([(649,763),(738,763)],'#8C4939',1)

    # The H bracket measures the axis, not the curved front arc or headwear.
    line([(276,y0),(257,y0),(257,y0+body_h),(276,y0+body_h)],'#7C8278',1.3)
    text(244,y0-13,'0%',23,'sans',fill=muted,anchor='rs')
    text(244,y0+body_h+10,'100%',23,'sans',fill=muted,anchor='rs')
    axis_label=Image.new('RGBA',(500*S,46*S),(0,0,0,0))
    al=ImageDraw.Draw(axis_label)
    al.text((250*S,23*S),'SHARE OF TOTAL EXPORTS',font=font(22,'bold'),fill=muted,anchor='mm')
    axis_label=axis_label.rotate(90,expand=True,resample=Image.Resampling.BICUBIC)
    page.paste(axis_label,(185*S,round((y0+body_h/2-250)*S)),axis_label)

    # Right-side interpretation and direct, one-to-one label binding.
    text(1280,410,f'{len(values)} destinations',44,'bold')
    text(1280,459,'One barrel represents all',29)
    text(1280,498,'synthetic oil exports.',29)
    line([(1280,543),(1946,543)],rule,1)
    text(1280,587,'Layer height shows share of the total.',26)
    text(1280,627,'Subcategories split their named parent.',26)
    text(1280,666,'Headwear, beard and rims carry no data.',25,fill=muted)
    text(1280,699,'DESTINATION',21,'bold',fill=muted,track=1.3)
    text(1945,699,'SHARE',21,'bold',fill=muted,anchor='rs')
    line([(1280,721),(1946,721)],ink,1.2)
    rows=[]
    anchor_x=x1-23
    arc_at_anchor=arc(anchor_x)
    for i,(cat,parent,v,col) in enumerate(zip(data['categories'],data['category_parent'],values,colors)):
        ym=(boundaries[i]+boundaries[i+1])/2+arc_at_anchor
        # The lead starts inside its own data layer, never at an unrelated chart row.
        line([(anchor_x,ym),(1251,ym)],'#8F9386',1.25)
        ellipse((anchor_x-4,ym-4,anchor_x+4,ym+4),fill=col,outline='#F6F0DF',width=1)
        ellipse((1247,ym-4,1255,ym+4),fill=col)
        baseline=ym+9
        if cat==parent:
            text(1280,baseline,cat,31,'bold')
        else:
            text(1280,baseline,parent,28,'bold')
            pw=draw.textlength(parent,font=font(28,'bold'))/S
            text(1280+pw+9,baseline,'/ '+cat.split(' / ',1)[1],25)
        value_text=f'{v.normalize():f}%'
        text(1945,baseline,value_text,40 if i==0 else 30,'mono',anchor='rs')
        if i:
            line([(1280,ym+28),(1946,ym+28)],'#DDD6C7',.8)
        rows.append({'category':cat,'value_text':value_text,'layer_color':col,'axial_top':boundaries[i],'axial_bottom':boundaries[i+1],'height_px':boundaries[i+1]-boundaries[i],'leader_anchor':[anchor_x,ym],'label_baseline':baseline})

    line([(104,1896),(1946,1896)],ink,1.3)
    text(104,1940,'Illustrative data only. Market names and shares are synthetic; this is not a real statistical survey.',25,fill=muted)
    text(1945,1940,'100% TOTAL',22,'bold',fill=muted,anchor='rs')

    # Save only this one complete chart. The frozen wrapper preserves each attempt.
    page.resize((W,H),Image.Resampling.LANCZOS).save(ROOT/'final.png',optimize=True)
    geom={'canvas':[W,H],'supersampling':S,'front_body':{'x_left':x0,'x_right':x1,'axis_top':y0,'axis_height':body_h,'ellipse_depth':depth},'mapping':'height = H * share / sum(shares); identical vertically translated front-ellipse arcs','total':str(total),'rows':rows,'text_boxes':text_boxes,'palette':colors,'frozen_inputs_used':[inputs['brief'],inputs['data'],inputs['critical_metaphor_contract']],'generated_assets_used':['assets/turban.png','assets/beard_shoulders.png']}
    (ROOT/'artifacts').mkdir(exist_ok=True)
    (ROOT/'artifacts/geometry.json').write_text(json.dumps(geom,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':'final.png','size':[W,H],'layers':len(values),'sum':str(total),'axis_height_px':body_h}))

if __name__=='__main__':
    main()
