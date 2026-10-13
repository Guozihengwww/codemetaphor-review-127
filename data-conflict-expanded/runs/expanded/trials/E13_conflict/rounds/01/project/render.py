#!/usr/bin/env python3
"""Offline oil-barrel infographic. Run from this directory; no network/model calls."""
from pathlib import Path
import io
import json
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent
S = 2
BG = (248, 247, 243)
INK = (26, 31, 34)
MUTED = (95, 105, 108)
PALETTE = [
    '#09696E', '#14868B', '#2CA2A7', '#70C2C2',
    '#47A559', '#FFD441', '#EC912B', '#F4B355',
    '#E16C50', '#C8324C', '#9D4A9F',
]
LIGHT_TEXT = [True, True, False, False, False, False, False, False, False, True, True]
# All product interfaces are F(x) + cumulative_volume * H / total.
H = 1210.0
BARREL_BOX = (90, 455, 900, 1390)
RAW_CROP = (98, 67, 911, 1475)
KNOTS = np.array([0, .13, .16, .24, .38, .41, .62, .74, .79, .91, .96, 1.0])
LEFT = np.array([370, 370, 354, 348, 380, 382, 371, 390, 413, 430, 470, 500])
RIGHT = np.array([906, 905, 919, 910, 905, 900, 917, 912, 900, 900, 880, 878])


def interface(x):
    return 502.0 + 54.0 * ((np.asarray(x) - 565.0) / 430.0) ** 2


def rgb(h):
    return tuple(int(h[i:i+2], 16) for i in (1, 3, 5))


def font(size, bold=False):
    return ImageFont.truetype(str(ROOT / 'assets' / ('NimbusSans-Bold.otf' if bold else 'NimbusSans-Regular.otf')), round(size*S))


def text(draw, pos, value, size, fill=INK, bold=False, anchor='ls'):
    draw.text(tuple(round(v*S) for v in pos), value, font=font(size, bold), fill=fill, anchor=anchor)


def centered(draw, xy, value, size, fill, bold=False):
    f = font(size, bold)
    b = draw.textbbox((0, 0), value, font=f)
    draw.text((round(xy[0]*S-(b[0]+b[2])/2), round(xy[1]*S-(b[1]+b[3])/2)), value, font=f, fill=fill)


def line(draw, coords, fill, width=1):
    draw.line([(round(x*S), round(y*S)) for x,y in coords], fill=fill, width=max(1,round(width*S)), joint='curve')


def rect(draw, box, fill, radius=0, outline=None, width=1):
    draw.rounded_rectangle(tuple(round(v*S) for v in box), radius=round(radius*S), fill=fill, outline=outline, width=round(width*S))


def number(v):
    return format(v, '.12g')


def droplet(canvas, cx, cy, color):
    """Small dimensional oil-drop legend; no data quantity encoded by its size."""
    w, h = 40*S, 52*S
    # Cubic curves are rasterized through a densely sampled Bezier outline.
    def cubic(a,b,c,d):
        u=np.linspace(0,1,55)[:,None]
        return (1-u)**3*np.array(a)+3*(1-u)**2*u*np.array(b)+3*(1-u)*u**2*np.array(c)+u**3*np.array(d)
    p=np.vstack([
        cubic((20,1),(16,12),(4,24),(4,34)),
        cubic((4,34),(3,55),(37,55),(36,34)),
        cubic((36,34),(36,24),(24,12),(20,1)),
    ])
    mask=Image.new('L',(w,h)); md=ImageDraw.Draw(mask)
    md.polygon([(round(x*S),round(y*S)) for x,y in p],fill=255)
    xx,yy=np.meshgrid(np.arange(w)/S,np.arange(h)/S)
    base=np.array(rgb(color),dtype=float)
    reflection=np.exp(-((xx-13)/10)**2-((yy-27)/24)**2)
    shade=.73+.29*reflection
    arr=np.minimum(255,base[None,None,:]*shade[:,:,None]+28*reflection[:,:,None]).astype('uint8')
    out=Image.fromarray(arr,'RGB').convert('RGBA');out.putalpha(mask)
    shadow=Image.new('RGBA',(w+10*S,h+10*S));shadow_alpha=Image.new('L',shadow.size)
    shadow_alpha.paste(mask,(4*S,4*S));shadow_alpha=shadow_alpha.filter(ImageFilter.GaussianBlur(2*S)).point(lambda z:int(z*.17))
    shadow.putalpha(shadow_alpha)
    px,py=round((cx-20)*S),round((cy-26)*S)
    canvas.alpha_composite(shadow,(px,py));canvas.alpha_composite(out,(px,py))
    d=ImageDraw.Draw(canvas)
    d.ellipse((px+11*S,py+23*S,px+15*S,py+34*S),fill=(255,255,255,100))


def make_barrel(values, total):
    bx,by,bw,bh=BARREL_BOX
    raw=Image.open(ROOT/'assets/raw_01.png').convert('RGBA')
    metal=raw.crop(RAW_CROP).resize((bw*S,bh*S),Image.Resampling.LANCZOS)
    x=(np.arange(bw*S,dtype=np.float32)+.5)/S+bx
    y=(np.arange(bh*S,dtype=np.float32)+.5)/S+by
    depth=y[:,None]-interface(x)[None,:]
    lft=np.interp(depth,KNOTS*H,LEFT)
    rgt=np.interp(depth,KNOTS*H,RIGHT)
    aperture=(depth>=0)&(depth<H)&(x[None,:]>=lft)&(x[None,:]<=rgt)
    cumulative=np.concatenate([[0.0],np.cumsum(values)])
    boundaries=cumulative*H/total
    idx=np.clip(np.searchsorted(boundaries,depth,side='right')-1,0,len(values)-1)
    bases=np.array([rgb(c) for c in PALETTE],dtype=np.float32)
    # A shared horizontal light field preserves every vertical data thickness.
    shade=.80+.22*np.exp(-((x-620)/220)**2)-.025*((x-680)/360)
    bright=5.0*np.exp(-((x-580)/190)**2)
    colors=np.clip(bases[idx]*shade[None,:,None]+bright[None,:,None],0,255)
    # Recessed cut-edge shading stays within 14 px of the shell, away from the channel.
    edge=np.minimum(x[None,:]-lft,rgt-x[None,:])
    edge_shade=.72+.28*np.minimum(np.maximum(edge,0)/14,1)
    colors*=edge_shade[:,:,None]
    face=np.zeros((bh*S,bw*S,4),dtype=np.uint8)
    face[:,:,:3]=colors.astype(np.uint8)
    face[:,:,3]=aperture.astype(np.uint8)*255
    face=Image.fromarray(face,'RGBA')
    shell_arr=np.array(metal)
    shell_arr[aperture,3]=0
    shell=Image.fromarray(shell_arr,'RGBA')
    # This processed generated asset is retained separately from its untouched original.
    shell.resize((bw,bh),Image.Resampling.LANCZOS).save(ROOT/'assets/barrel_cutaway.png')
    return face,shell,cumulative,boundaries,shade,bright


def main():
    inputs=json.loads((ROOT/'input.json').read_text(encoding='utf-8'))
    brief=json.loads((ROOT/inputs['brief']).read_text(encoding='utf-8'))
    data=json.loads((ROOT/inputs['data']).read_text(encoding='utf-8'))
    core=json.loads((ROOT/inputs['critical_metaphor_contract']).read_text(encoding='utf-8'))
    assert core['semantic_roles_zh']==['Synthetic output volume (US gal)']
    W,HH=brief['width'],brief['height']
    values=np.array(data['series'][0]['values'],dtype=float)
    cats=data['categories'];total=float(values.sum())
    assert len(values)==len(cats)==len(PALETTE)
    assert abs(total-data['series_totals'][data['series'][0]['id']])<1e-9
    assert data['value_axis']=={'minimum':0,'maximum':44,'scale':'linear'}
    canvas=Image.new('RGBA',(W*S,HH*S),BG+(255,));d=ImageDraw.Draw(canvas)
    rect(d,(100,66,414,112),INK,radius=4)
    text(d,(121,98),data['visible_notice'],24,(255,255,255),True)
    text(d,(1700,98),'ONE BARREL  /  PRODUCT VOLUMES',22,MUTED,anchor='rs')
    text(d,(96,213),'Products from a',102,INK,True)
    text(d,(96,321),'barrel of crude oil',102,INK,True)
    text(d,(1700,176),'TOTAL OUTPUT',24,MUTED,True,anchor='rs')
    text(d,(1700,280),number(total),112,INK,True,anchor='rs')
    text(d,(1700,323),'US gal',32,INK,anchor='rs')
    line(d,[(100,369),(1700,369)],(189,196,194),1.5)
    text(d,(100,418),data['series'][0]['label']+' (US gal)',29,INK)
    text(d,(1700,418),'Listed order: top to bottom',24,MUTED,anchor='rs')

    face,shell,cumulative,boundaries,shade,bright=make_barrel(values,total)
    bx,by,bw,bh=BARREL_BOX
    canvas.alpha_composite(face,(bx*S,by*S))
    canvas.alpha_composite(shell,(bx*S,by*S))
    d=ImageDraw.Draw(canvas)
    # Metal bevels follow the two cut edges; no horizontal structural hoop crosses the data channel.
    for side,outer in [(LEFT,-1),(RIGHT,1)]:
        pts=[(float(x),float(interface(x)+t)) for x,t in zip(side,KNOTS*H)]
        line(d,[(x+outer*2,y) for x,y in pts],(12,16,19),6)
        line(d,[(x+outer*2,y) for x,y in pts],(112,120,122),3)
        line(d,[(x+outer*3,y) for x,y in pts],(205,209,207),.8)
    # Numeric annotations are centered within each exact layer, with no minimum-size adjustment.
    label_x=662
    for i,value in enumerate(values):
        yc=float(interface(label_x)+(boundaries[i]+boundaries[i+1])/2)
        label_size=56 if i==len(values)-1 else (28 if boundaries[i+1]-boundaries[i]<43 else 32)
        fill=(255,255,249) if LIGHT_TEXT[i] else (29,45,42)
        centered(d,(label_x,yc),number(value),label_size,fill,True)

    # Oil-drop legend repeats exact values and preserves all semantic category names.
    text(d,(1161,512),'PRODUCT',22,MUTED,True)
    text(d,(1700,512),'US gal',22,MUTED,True,anchor='rs')
    line(d,[(1080,534),(1700,534)],(176,185,184),1.4)
    row_centers=[]
    for i,(cat,value,color) in enumerate(zip(cats,values,PALETTE)):
        cy=588+111*i
        row_centers.append(cy)
        droplet(canvas,1102,cy,color);d=ImageDraw.Draw(canvas)
        if ' / ' in cat:
            parent,sub=cat.split(' / ',1)
            text(d,(1161,cy-3),parent,31,INK,True)
            text(d,(1161,cy+29),sub,24,MUTED)
        elif cat=='Heavy Fuel Oil (Residual)':
            text(d,(1161,cy-3),'Heavy Fuel Oil',31,INK,True)
            text(d,(1161,cy+29),'(Residual)',24,MUTED)
        elif cat=='Liquefied Petroleum Gases (LPG)':
            text(d,(1161,cy-3),'Liquefied Petroleum',29,INK,True)
            text(d,(1161,cy+29),'Gases (LPG)',25,MUTED)
        else:
            text(d,(1161,cy+11),cat,31,INK,True)
        text(d,(1700,cy+12),number(value),37,INK,True,anchor='rs')
        if i<len(cats)-1:
            line(d,[(1161,cy+55),(1700,cy+55)],(224,227,222),1)

    # A compact explanatory footer identifies the metric, synthetic scope and visual encoding.
    line(d,[(100,1871),(1700,1871)],(189,196,194),1.5)
    text(d,(100,1920),'Layer thickness = product volume / 44 US gal',29,INK,True)
    text(d,(100,1960),'Synthetic illustration. Subcategories are mutually exclusive parts of their named parent.',25,MUTED)
    text(d,(1700,2007),'11 CATEGORIES  •  1 SYNTHETIC BARREL',20,MUTED,anchor='rs')

    final=canvas.convert('RGB').resize((W,HH),Image.Resampling.LANCZOS)
    final.save(ROOT/'final.png')
    # Target geometry is an audit aid. Actual raster measurements are performed separately.
    channels=[740,800,840]
    geometry={
        'canvas':[W,HH], 'H_px':H, 'total_us_gal':total,
        'interface_formula':'502 + 54 * ((x - 565) / 430) ** 2 + cumulative_volume * 1210 / 44',
        'category_order':'top to bottom, exactly data.json order',
        'clear_scan_columns_px':channels,
        'barrel_box':list(BARREL_BOX), 'raw_crop':list(RAW_CROP),
        'labels':[{'category':cat,'value':float(v),'unit':data['series'][0]['unit'],'color':c,'thickness_px':float(H*v/total),'center_y_px':float(interface(label_x)+(boundaries[i]+boundaries[i+1])/2)} for i,(cat,v,c) in enumerate(zip(cats,values,PALETTE))],
        'interfaces_at_scan_columns':{str(x):[float(interface(x)+b) for b in boundaries] for x in channels},
        'legend_centers_y':row_centers,
        'font_files':['assets/NimbusSans-Regular.otf','assets/NimbusSans-Bold.otf'],
    }
    (ROOT/'artifacts').mkdir(exist_ok=True)
    (ROOT/'artifacts/geometry.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':'final.png','size':[W,HH],'categories':len(cats),'total':total,'H_px':H}))


if __name__=='__main__':
    main()
