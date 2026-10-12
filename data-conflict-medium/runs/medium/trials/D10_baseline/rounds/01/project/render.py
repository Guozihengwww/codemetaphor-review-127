#!/usr/bin/env python3
"""Rebuild the sand infographic from frozen data and local generated assets."""
from pathlib import Path
import json
import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter


ROOT = Path(__file__).resolve().parent
S = 2
BG = '#f6f5ef'
INK = '#22372f'
MUTED = '#657168'
AMBER = '#ba7133'
EDGE = '#995d29'


def scaled(points):
    return [(round(x*S), round(y*S)) for x, y in points]


def font(size, bold=False, cjk=False):
    name = ('CJK-Bold.ttc' if bold else 'CJK-Regular.ttc') if cjk else ('Sans-Bold.ttf' if bold else 'Sans.ttf')
    return ImageFont.truetype(str(ROOT/'assets'/'fonts'/name), round(size*S))


def text(draw, xy, value, size, color=INK, bold=False, anchor='lt', cjk=False):
    draw.text((round(xy[0]*S), round(xy[1]*S)), str(value), font=font(size, bold, cjk), fill=color, anchor=anchor)


def line(draw, points, fill, width=1):
    draw.line(scaled(points), fill=fill, width=max(1,round(width*S)))


def box(draw, rect, fill, radius=0, outline=None, width=1):
    bounds=tuple(round(v*S) for v in rect)
    if radius:
        draw.rounded_rectangle(bounds, radius=round(radius*S), fill=fill, outline=outline, width=round(width*S))
    else:
        draw.rectangle(bounds, fill=fill, outline=outline, width=round(width*S))


def ellipse(draw, rect, fill, outline=None, width=1):
    draw.ellipse(tuple(round(v*S) for v in rect), fill=fill, outline=outline, width=max(1,round(width*S)))


def main():
    brief=json.loads((ROOT/'brief.json').read_text(encoding='utf-8'))
    data=json.loads((ROOT/'data.json').read_text(encoding='utf-8'))
    W,H=brief['width'],brief['height']
    assert (W,H)==(2000,2000)
    series=data['series'][0]
    cats=data['categories']
    values=series['values']
    times=data['time_points']
    vmin,vmax=data['value_axis']['minimum'],data['value_axis']['maximum']
    assert vmin==0 and vmax==400 and data['value_axis']['scale']=='linear'
    assert len(cats)==len(values)==len(times)==12
    assert all(vmin<=v<=vmax for v in values)
    x0,x1,y0,yt=255.,1830.,1560.,560.
    px_per_unit=(y0-yt)/(vmax-vmin)
    xs=[x0+(t-times[0])/(times[-1]-times[0])*(x1-x0) for t in times]
    ys=[y0-(v-vmin)*px_per_unit for v in values]
    points=list(zip(xs,ys))
    polygon=points+[(x1,y0),(x0,y0)]
    canvas=Image.new('RGB',(W*S,H*S),BG)
    d=ImageDraw.Draw(canvas)

    # Clear hierarchy: provenance, title, series label, and a dedicated unit label.
    box(d,(130,100,445,152),'#ece2cf',radius=26)
    text(d,(287.5,126),data['visible_notice'],24,color='#77522d',bold=True,anchor='mm')
    text(d,(1870,126),'2008—2019',26,color=MUTED,anchor='rm')
    text(d,(130,207),'Synthetic',100,bold=True)
    text(d,(728,200),'沙堆轮廓',100,bold=True,cjk=True)
    text(d,(132,347),series['label'],37,color=MUTED)
    line(d,[(132,439),(1870,439)],'#d7dbd1',1.4)
    box(d,(132,484,158,510),AMBER,radius=3)
    text(d,(173,482),series['unit'],29,color=MUTED)

    # A shadow below the common baseline anchors the granular object to the page.
    shadow=Image.new('L',canvas.size,0)
    sd=ImageDraw.Draw(shadow)
    sd.ellipse(tuple(round(v*S) for v in (235,1549,1854,1584)),fill=41)
    shadow=shadow.filter(ImageFilter.GaussianBlur(11*S))
    canvas.paste('#5a4b32',(0,0),shadow)

    # One data transform controls both the clipping geometry and the annotations.
    sand_mask=Image.new('L',canvas.size,0)
    ImageDraw.Draw(sand_mask).polygon(scaled(polygon),fill=255)
    texture=Image.open(ROOT/'assets/sand_texture.png').convert('RGB')
    # Crop without stretching individual grains; add a subtle depth gradient.
    tw,th=round((x1-x0)*S),round((y0-min(ys))*S)+2
    source=texture.resize((tw,round(texture.height*tw/texture.width)),Image.Resampling.LANCZOS)
    source=source.crop((0,0,tw,th))
    arr=np.asarray(source).astype(np.float32)
    xx=np.arange(tw)/S+x0
    yy=np.arange(th)/S+min(ys)
    profile=np.interp(xx,xs,ys)
    depth=np.clip((yy[:,None]-profile[None,:])/(y0-profile[None,:]),0,1)
    shading=1.025-.105*depth
    ridge=np.exp(-np.maximum(0,yy[:,None]-profile[None,:])/10.)*.06
    arr=np.clip(arr*(shading+ridge)[:,:,None],0,255).astype(np.uint8)
    surface=Image.new('RGB',canvas.size,BG)
    surface.paste(Image.fromarray(arr),(round(x0*S),round(min(ys)*S)))
    canvas.paste(surface,(0,0),sand_mask)

    # Isolated grains are explicitly separated from the continuous data boundary.
    rng=random.Random(1017)
    d=ImageDraw.Draw(canvas)
    label_bounds=[(x-51,y-86,x+51,y-35) for x,y in points]
    grain_colors=['#c49b5f','#d5b67d','#bb8d4b','#e2c694','#b98743']
    scattered=0
    for _ in range(140):
        gx=rng.uniform(x0+10,x1-9)
        gy=float(np.interp(gx,xs,ys))-rng.uniform(11,43)
        if any(a-9<gx<c+9 and b-9<gy<e+9 for a,b,c,e in label_bounds):
            continue
        if 755<gx<835 and 1140<gy<1260:
            continue
        size=rng.uniform(1.1,3.0)
        grain=[(gx-size,gy-.5*size),(gx+.2*size,gy-size),(gx+size,gy+.5*size),(gx-.2*size,gy+size)]
        d.polygon(scaled(grain),fill=rng.choice(grain_colors))
        scattered+=1
    for _ in range(95):
        gx=rng.uniform(x0-13,x1+14)
        gy=rng.uniform(y0+5,y0+22)
        r=rng.uniform(.7,2.1)
        ellipse(d,(gx-r,gy-r*.65,gx+r,gy+r*.65),rng.choice(grain_colors))

    # Horizontal reference lines stay faint on both the background and the sand.
    grid=Image.new('RGBA',canvas.size,(0,0,0,0))
    gd=ImageDraw.Draw(grid)
    for val in [100,200,300,400]:
        gy=y0-val*px_per_unit
        line(gd,[(x0-34,gy),(x1+32,gy)],(81,96,83,49),1.1)
    canvas=Image.alpha_composite(canvas.convert('RGBA'),grid)

    # The photographed cutout is tilted, cast onto the sand, then partly buried.
    shovel=Image.open(ROOT/'assets/shovel_cutout.png').convert('RGBA')
    new_h=780*S
    shovel=shovel.resize((round(shovel.width*new_h/shovel.height),new_h),Image.Resampling.LANCZOS)
    shovel=shovel.rotate(24,resample=Image.Resampling.BICUBIC,expand=True)
    pos=(454*S,596*S)
    shovel_layer=Image.new('RGBA',canvas.size,(0,0,0,0))
    shovel_layer.alpha_composite(shovel,pos)
    a=np.asarray(shovel_layer.getchannel('A')).copy()
    # An irregular contact line is deep inside the sand; it never changes its top.
    yy_all=np.arange(H*S)[:,None]/S
    xx_all=np.arange(W*S)[None,:]/S
    burial_y=1307+.035*(xx_all-850)+1.8*np.sin(xx_all*.22)+1.1*np.sin(xx_all*.61)
    bury=yy_all>burial_y
    a[bury]=0
    shovel_layer.putalpha(Image.fromarray(a))
    shadow_alpha=Image.fromarray((a*.29).astype('uint8')).filter(ImageFilter.GaussianBlur(5*S))
    shifted=Image.new('L',canvas.size,0)
    shifted.paste(shadow_alpha,(8*S,7*S))
    shifted_arr=np.asarray(shifted).astype(np.uint16)*np.asarray(sand_mask).astype(np.uint16)//255
    cast=Image.new('RGBA',canvas.size,(67,38,16,0))
    cast.putalpha(Image.fromarray(shifted_arr.astype('uint8')))
    canvas=Image.alpha_composite(canvas,cast)
    canvas=Image.alpha_composite(canvas,shovel_layer)
    d=ImageDraw.Draw(canvas)
    # Small sand grains at the contact point visually establish the burial depth.
    for _ in range(40):
        gx=rng.uniform(790,924)
        gy=1307+.035*(gx-850)+rng.uniform(-3.0,6.0)
        r=rng.uniform(.8,2.6)
        ellipse(d,(gx-r,gy-r*.62,gx+r,gy+r*.62),rng.choice(grain_colors))

    # The precise ridge is drawn on top so the single shaft crossing remains readable.
    line(d,points,EDGE,2.0)
    for x,y in points:
        ellipse(d,(x-5,y-5,x+5,y+5),BG,outline=EDGE,width=2)

    # All values are exact, placed directly over their own year and data node.
    for cat,val,x,y in zip(cats,values,xs,ys):
        box(d,(x-50,y-82,x+50,y-37),BG,radius=8)
        text(d,(x,y-58),str(val),36,bold=True,anchor='mm')
        line(d,[(x,y-28),(x,y-10)],'#b2ab94',1.1)
        text(d,(x,y0+56),cat,29,color=INK,anchor='mm')
        line(d,[(x,y0),(x,y0+10)],INK,1.2)
    line(d,[(x0-34,y0),(x1+32,y0)],INK,1.7)
    for val in [0,100,200,300,400]:
        text(d,(203,y0-val*px_per_unit),str(val),31,color=MUTED,anchor='rm')

    # A compact reading guide makes the encoded dimension and provenance explicit.
    line(d,[(132,1703),(1870,1703)],'#d7dbd1',1.4)
    text(d,(132,1747),'读图方式',25,color=AMBER,bold=True,cjk=True)
    text(d,(132,1800),'沙体上缘 = 年度值',37,bold=True,cjk=True)
    text(d,(132,1860),'从共同零线量取垂直高度，年度节点间直线连接。',27,color=MUTED,cjk=True)
    text(d,(1870,1751),'12',48,bold=True,anchor='rt')
    text(d,(1790,1769),'年度节点',25,color=MUTED,cjk=True,anchor='rt')
    text(d,(1870,1821),'0–400  /  线性量尺',26,color=MUTED,cjk=True,anchor='rt')
    text(d,(132,1940),'数据：本地 data.json · 合成示例，非真实开采统计',23,color=MUTED,cjk=True)

    final=canvas.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
    final.save(ROOT/'final.png',optimize=True)
    (ROOT/'artifacts').mkdir(exist_ok=True)
    # This is a diagnostic mask crop, not another completed chart candidate.
    sand_mask.resize((W,H),Image.Resampling.NEAREST).crop((220,530,1865,1585)).save(ROOT/'artifacts/sand_mask_crop.png')
    geometry={
        'canvas':[W,H], 'plot':{'x_min':x0,'x_max':x1,'y_zero':y0,'y_400':yt},
        'pixels_per_unit':px_per_unit,'contour':'piecewise linear; closed to common zero line',
        'points':[{'category':c,'value':v,'x':x,'y':y} for c,v,x,y in zip(cats,values,xs,ys)],
        'font_assets':['assets/fonts/Sans.ttf','assets/fonts/Sans-Bold.ttf','assets/fonts/CJK-Regular.ttc','assets/fonts/CJK-Bold.ttc'],
        'shovel':{'angle_degrees':24,'cutout_height_px':780,'paste_origin':[454,596],'burial_y_near':1307},
        'isolated_ridge_grain_count':scattered,
        'visible_text':[data['visible_notice'],data['title'],series['label'],series['unit']]+cats+[str(v) for v in values],
    }
    (ROOT/'artifacts/geometry.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Wrote final.png (2000 × 2000), with 12 exact data nodes.')


if __name__=='__main__':
    main()
