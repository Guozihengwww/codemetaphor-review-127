#!/usr/bin/env python3
"""Rebuild the frozen nose-profile infographic using only local data and assets."""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parent
S = 2
PAPER = (248, 245, 234)
INK = (53, 48, 41)
BROWN = (79, 49, 31)
MUTED = (111, 107, 91)
GREEN = (44, 92, 72)
BLUE = (96, 129, 154)


def font(size, family='sans'):
    file = {'sans':'sans.ttf', 'bold':'sans_bold.ttf', 'italic':'serif_italic.ttf',
            'kai':'kai.ttc', 'cjk':'cjk.ttc'}[family]
    return ImageFont.truetype(str(ROOT/'fonts'/file), round(size*S), index=0)


def text(canvas, xy, value, size, family='sans', fill=INK, anchor='lt'):
    ImageDraw.Draw(canvas).text(tuple(round(v*S) for v in xy), str(value),
        font=font(size,family), fill=fill, anchor=anchor)


def line(canvas, points, fill, width=1):
    ImageDraw.Draw(canvas).line([(round(x*S),round(y*S)) for x,y in points],
                                fill=fill, width=max(1,round(width*S)), joint='curve')


def track_text(canvas, xy, value, size, spacing, fill=INK, family='sans'):
    x,y = xy
    f = font(size,family)
    draw = ImageDraw.Draw(canvas)
    for ch in value:
        draw.text((round(x*S),round(y*S)),ch,font=f,fill=fill,anchor='lt')
        x += draw.textlength(ch,font=f)/S + spacing


def rgba_warp(source, sx, sy):
    """Premultiplied bilinear sampling keeps transparent illustration edges clean."""
    a = np.asarray(source.convert('RGBA'),dtype=np.float32) / 255
    alpha = a[:,:,3]
    coords = [sy.ravel(), sx.ravel()]
    sampled_alpha = map_coordinates(alpha,coords,order=1,mode='constant',cval=0).reshape(sx.shape)
    out = np.zeros((*sx.shape,4), dtype=np.uint8)
    for c in range(3):
        premult = map_coordinates(a[:,:,c]*alpha,coords,order=1,mode='constant',cval=0).reshape(sx.shape)
        out[:,:,c] = np.clip(premult / np.maximum(sampled_alpha,1e-8)*255,0,255).astype('uint8')
    out[:,:,3] = np.clip(sampled_alpha*255,0,255).astype('uint8')
    return Image.fromarray(out)


def main():
    data = json.loads((ROOT/'data.json').read_text(encoding='utf-8'))
    brief = json.loads((ROOT/'brief.json').read_text(encoding='utf-8'))
    W,H = brief['width'],brief['height']
    assert (W,H)==(1600,2000)
    times = np.array(data['time_points'],float)
    values = np.array(data['series'][0]['values'],float)
    assert len(times)==len(values)==len(data['categories'])==5
    assert np.all(np.diff(times)>0)
    axis_min, axis_max = data['value_axis']['minimum'], data['value_axis']['maximum']
    assert axis_min==0 and axis_max==4 and data['value_axis']['scale']=='linear'
    # A single affine mapping controls the texture mask, boundary, nodes, and scales.
    x0,y_top,y_bottom,plot_width = 400.,550.,1550.,1000.
    y_axis_bottom = 1600.
    x_scale = plot_width/(axis_max-axis_min)
    year_scale = (y_bottom-y_top)/(times[-1]-times[0])
    y_for = lambda t: y_bottom-(np.asarray(t)-times[0])*year_scale
    x_for = lambda v: x0+(np.asarray(v)-axis_min)*x_scale
    interp = PchipInterpolator(times,values,extrapolate=False)
    boundary = lambda y: x_for(interp(times[0]+(y_bottom-np.asarray(y))/year_scale))
    node_x = x_for(values)
    node_y = y_for(times)
    assert np.allclose(boundary(node_y),node_x,atol=1e-8)

    # Almost front-on paper, with a small physical edge and reproducible paper grain.
    canvas = Image.new('RGBA',(W*S,H*S),(226,226,217,255))
    shadow = Image.new('RGBA',canvas.size,(0,0,0,0))
    ImageDraw.Draw(shadow).rounded_rectangle((56*S,52*S,1560*S,1958*S),radius=10*S,fill=(44,43,32,47))
    shadow = shadow.filter(ImageFilter.GaussianBlur(12*S))
    canvas.alpha_composite(shadow)
    paper = Image.new('RGBA',canvas.size,(0,0,0,0))
    ImageDraw.Draw(paper).rectangle((45*S,35*S,1545*S,1945*S), fill=(*PAPER,255))
    canvas.alpha_composite(paper)
    rng = np.random.default_rng(12035)
    grain = rng.normal(0,0.65,(H,W,1))
    base = np.clip(np.array(PAPER)[None,None,:]+grain,0,255).astype('uint8')
    grain_im = Image.fromarray(base).resize((W*S,H*S),Image.Resampling.BILINEAR)
    paper_mask = Image.new('L',canvas.size,0)
    ImageDraw.Draw(paper_mask).rectangle((46*S,36*S,1544*S,1944*S),fill=255)
    canvas.paste(grain_im,(0,0),paper_mask)

    # Body material is warped in normalized horizontal distance and clipped to data.
    local_w = round(plot_width*S)+1
    local_h = round((y_bottom-y_top)*S)+1
    yy,xx = np.mgrid[:local_h,:local_w].astype('float32')
    global_y = y_top+yy/S
    right = boundary(global_y)
    width_here = (right-x0)
    texture = Image.open(ROOT/'assets/skin_texture.png').convert('RGBA')
    tw,th = texture.size
    tx = np.clip(xx/S/width_here,0,1)*(tw-1)
    ty = (global_y-y_top)/(y_bottom-y_top)*(th-1)
    skin = rgba_warp(texture,tx,ty)
    body_alpha = np.clip((width_here-xx/S)*S+0.5,0,1)
    skin_a = np.array(skin)
    skin_a[:,:,3] = (body_alpha*255).astype('uint8')
    skin = Image.fromarray(skin_a)
    canvas.alpha_composite(skin,(round(x0*S),round(y_top*S)))

    logical_mask = Image.new('L',canvas.size,0)
    logical_mask.paste(Image.fromarray((body_alpha*255).astype('uint8')),(round(x0*S),round(y_top*S)))

    # Generated alar shading/nostril is registered against the CURRENT data boundary.
    reg = json.loads((ROOT/'assets/nose_registration.json').read_text())
    nose = Image.open(ROOT/'assets/nose_detail.png').convert('RGBA')
    sy = global_y-600.0
    source_right = np.interp(sy,reg['source_y'],reg['source_right_edge_x'])
    sx = source_right-(right-(x0+xx/S))/1.50
    nose_warp = rgba_warp(nose,sx.astype('float32'),sy.astype('float32'))
    na = np.array(nose_warp)
    na[:,:,3] = (na[:,:,3]*body_alpha).astype('uint8')
    canvas.alpha_composite(Image.fromarray(na),(round(x0*S),round(y_top*S)))

    # Square graph paper remains visible through the colored-pencil skin material.
    grid = Image.new('RGBA',canvas.size,(0,0,0,0))
    gd = ImageDraw.Draw(grid)
    for x in range(100,1501,50):
        alpha = 57 if x%250==150 else 31
        gd.line((x*S,95*S,x*S,1900*S),fill=(*BLUE,alpha),width=S)
    for y in range(100,1901,50):
        alpha = 57 if y%250==50 else 31
        gd.line((95*S,y*S,1505*S,y*S),fill=(*BLUE,alpha),width=S)
    # The text region has a lighter grid while preserving the same paper surface.
    fade = Image.new('L',canvas.size,255)
    fd = ImageDraw.Draw(fade)
    fd.rectangle((80*S,80*S,1515*S,440*S),fill=70)
    fd.rectangle((80*S,1680*S,1515*S,1915*S),fill=65)
    garr = np.array(grid)
    garr[:,:,3] = (garr[:,:,3].astype(float)*np.array(fade)/255).astype('uint8')
    canvas.alpha_composite(Image.fromarray(garr))

    # Upper small eye overlay. Its opacity does not subtract from the logical area.
    eye = Image.open(ROOT/'assets/eye.png').convert('RGBA')
    eye = eye.resize((round(274*S),round(230.7*S)),Image.Resampling.LANCZOS)
    canvas.alpha_composite(eye,(round(435*S),round(575*S)))

    # Exact, smoothly interpolated data outline; no extra observed nodes.
    ys = np.linspace(y_top,y_bottom,2001)
    xs = boundary(ys)
    outline = [(x0,y_top), *list(zip(xs,ys)), (x0,y_bottom)]
    line(canvas,outline,BROWN,4.6)
    # A thin second pencil stroke sits inside the same geometry, never expanding it.
    trace = [(float(x)-2.6,float(y)+0.55*math.sin((y-y_top)/28)) for x,y in zip(xs,ys)]
    line(canvas,trace,(131,82,49),0.65)
    line(canvas,[(x0,y_top-8),(x0,y_axis_bottom)],INK,3.1)
    line(canvas,[(x0,y_axis_bottom),(x0+plot_width,y_axis_bottom)],INK,2.7)

    # Observations are bound to the contour, with short direct leaders and exact labels.
    draw = ImageDraw.Draw(canvas)
    text_boxes=[]
    for category,value,x,y in zip(data['categories'],data['series'][0]['values'],node_x,node_y):
        line(canvas,[(x0-12,y),(x0,y)],INK,2.6)
        text(canvas,(x0-31,y),category,33,'sans',INK,anchor='rm')
        line(canvas,[(x+9,y),(x+23,y)],MUTED,1.8)
        label = str(value)
        text(canvas,(x+33,y),label,37,'bold',INK,anchor='lm')
        box=draw.textbbox((round((x+33)*S),round(y*S)),label,font=font(37,'bold'),anchor='lm')
        text_boxes.append({'text':label,'box_px':[round(z/S,2) for z in box]})
        r = 8.0
        draw.ellipse(((x-r)*S,(y-r)*S,(x+r)*S,(y+r)*S),fill=GREEN)
        r = 3.0
        draw.ellipse(((x-r)*S,(y-r)*S,(x+r)*S,(y+r)*S),fill=PAPER)

    for v in range(int(axis_min),int(axis_max)+1):
        x = float(x_for(v))
        line(canvas,[(x,y_axis_bottom),(x,y_axis_bottom+13)],INK,2.4)
        text(canvas,(x,y_axis_bottom+29),str(v),31,'sans',INK,anchor='mt')
    text(canvas,(400,1687),'Million procedures per indicated year',29,'sans',INK)
    text(canvas,(400,1729),'对应年度的鼻部手术量（百万例）',26,'cjk',MUTED)

    axis_label = Image.new('RGBA',(430*S,65*S),(0,0,0,0))
    text(axis_label,(0,3),'YEAR / 年份',29,'cjk',INK)
    axis_label=axis_label.rotate(90,expand=True)
    canvas.alpha_composite(axis_label,(145*S,865*S))

    # Code typesets every semantic and mandatory text element.
    hd=ImageDraw.Draw(canvas)
    hd.rounded_rectangle((128*S,124*S,447*S,173*S),radius=5*S,outline=GREEN,width=2*S,fill=(*PAPER,235))
    track_text(canvas,(145,138),data['visible_notice'],23,2.15,GREEN,'bold')
    track_text(canvas,(1090,132),'NOSE PROCEDURES',17,2.0,MUTED)
    text(canvas,(1468,171),f'{data["categories"][0]}—{data["categories"][-1]}',23,'sans',MUTED,anchor='rt')
    title = data['title']
    english, chinese = title.split(' ',1)
    text(canvas,(125,226),english,91,'italic',INK)
    text(canvas,(673,232),chinese,88,'kai',INK)
    text(canvas,(132,356),'Nose procedures  /  鼻部手术',31,'cjk',MUTED)
    line(canvas,[(132,419),(451,419)],(171,135,104),2)
    hd.ellipse((1116*S,373*S,1128*S,385*S),fill=GREEN)
    text(canvas,(1141,364),'观测节点',23,'cjk',MUTED)
    text(canvas,(1114,404),'曲线仅作插值',23,'cjk',MUTED)

    line(canvas,[(132,1806),(1467,1806)],(158,154,135),1)
    text(canvas,(132,1830),'横向距离 = 手术量；年份由下至上，按真实时间间距排列。',25,'cjk',INK)
    text(canvas,(132,1872),'眼睛与鼻翼为装饰覆盖层；鼻形总面积不编码。',23,'cjk',MUTED)
    text(canvas,(1466,1888),'合成数据 · 非真实统计',21,'cjk',MUTED,anchor='rs')

    final = canvas.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
    final.save(ROOT/'final.png')
    (ROOT/'artifacts').mkdir(exist_ok=True)
    logical_mask.resize((W,H),Image.Resampling.LANCZOS).save(ROOT/'artifacts/data_mask.png')
    geometry={
        'canvas':[W,H], 'axis_min':axis_min,'axis_max':axis_max,
        'zero_x':x0,'pixels_per_million':x_scale,'top_y':y_top,'bottom_y':y_bottom,
        'pixels_per_year':year_scale,'interpolation':'PCHIP; shape preserving; no extra observations',
        'observations':[{'category':str(c),'year':float(t),'value':float(v),'x':float(x),'y':float(y)}
                        for c,t,v,x,y in zip(data['categories'],times,values,node_x,node_y)],
        'numeric_label_boxes':text_boxes,
        'body_mask_nonzero_pixels':int((np.asarray(logical_mask.resize((W,H)))>127).sum()),
        'title':data['title'], 'unit':data['series'][0]['unit'],
        'visible_notice':data['visible_notice']}
    (ROOT/'artifacts/geometry.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Rendered final.png: 1600 x 2000; five data-bound observations; local assets only.')

if __name__=='__main__':
    main()
