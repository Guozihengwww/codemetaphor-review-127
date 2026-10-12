"""Rebuild final.png from frozen local data and generated assets, without network.

Run complete renders only through capture_render.py. Geometry is computed once:
equal ordered x positions, common monotone interpolation weights, and cumulative
positive y scale from a fixed waterline. All text is drawn by executable code.
"""
from pathlib import Path
import json
import math
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent
S = 2
W, H = 2200, 1552
X0, X1 = 160.0, 2080.0
Y0, Y1 = 340.0, 1132.0
INK = '#152e3a'
MUTED = '#60737b'
PALE = '#d5dfe0'
BG = '#fbfcfb'
COLORS = ['#11ac79', '#287ce6', '#16285b', '#8c4bbb']
FONT_FILES = {'sans': 'sans.otf', 'serif': 'serif.otf', 'num': 'numbers.ttf', 'bold': 'numbers_bold.ttf'}
FONTS = {}

def font(size, face='sans'):
    key = (size, face)
    if key not in FONTS:
        FONTS[key] = ImageFont.truetype(str(ROOT / 'assets/fonts' / FONT_FILES[face]), round(size * S))
    return FONTS[key]

def main():
    data = json.loads((ROOT / 'data.json').read_text(encoding='utf-8'))
    brief = json.loads((ROOT / 'brief.json').read_text(encoding='utf-8'))
    assert (brief['width'], brief['height']) == (W, H)
    categories = data['categories']
    values = np.array([s['values'] for s in data['series']], dtype=float)
    assert values.shape == (4, 16) and np.all(values > 0)
    assert data['value_axis'] == {'minimum': 0, 'maximum': 7200, 'scale': 'linear'}
    totals = values.sum(axis=0)
    cumulative = np.cumsum(values, axis=0)
    xs = np.linspace(X0, X1, len(categories))
    scale = (Y1 - Y0) / data['value_axis']['maximum']

    def layers_at(x):
        x = np.asarray(x)
        coord = np.clip((x-X0)/(X1-X0)*(len(categories)-1), 0, len(categories)-1)
        left = np.minimum(np.floor(coord).astype(int), len(categories)-2)
        t = coord - left
        blend = t*t*(3-2*t)
        return values[:, left]*(1-blend) + values[:, left+1]*blend

    def bounds_at(x):
        return Y0 + np.cumsum(layers_at(x), axis=0)*scale

    canvas = Image.new('RGB', (W*S, H*S), BG)
    draw = ImageDraw.Draw(canvas)
    text_records = []

    def text(x, y, string, size=24, fill=INK, face='sans', anchor='lt', record=True):
        f = font(size, face)
        draw.text((round(x*S), round(y*S)), str(string), font=f, fill=fill, anchor=anchor)
        if record:
            box = draw.textbbox((round(x*S), round(y*S)), str(string), font=f, anchor=anchor)
            text_records.append({'text': str(string), 'bbox': [round(v/S,2) for v in box]})

    def line(points, fill, width=1):
        draw.line([(round(x*S), round(y*S)) for x,y in points], fill=fill, width=max(1,round(width*S)), joint='curve')

    def rect(box, fill, outline=None, width=1, radius=0):
        b = tuple(round(v*S) for v in box)
        if radius:
            draw.rounded_rectangle(b, round(radius*S), fill=fill, outline=outline, width=max(1,round(width*S)))
        else:
            draw.rectangle(b, fill=fill, outline=outline, width=max(1,round(width*S)))

    def dot(x,y,r,fill,outline=None,width=1):
        draw.ellipse((round((x-r)*S),round((y-r)*S),round((x+r)*S),round((y+r)*S)), fill, outline, max(1,round(width*S)))

    # Editorial header: title, provenance notice, and the complete reading rule.
    rect((120, 42, 320, 76), '#e6f4ee', radius=4)
    text(220, 59, data['visible_notice'], 17, '#167851', 'bold', 'mm')
    text(342, 60, 'OCEAN CROSS-SECTION  /  16 PERIODS · 4 SOURCES', 17, MUTED, 'num', 'lm')
    text(120, 103, data['title'], 58, INK, 'serif')
    text(120, 193, '层厚表示分项，边界向下累计；四项之和构成海洋总深度。', 26, '#4b646d')
    text(120, 238, 'Layer thickness = source output   ·   Total depth = A + B + C + D', 20, MUTED, 'num')
    text(2080, 57, '合成数据示意  /  非真实海洋统计', 18, MUTED, 'sans', 'rm')

    # Grid behind the cutaway, with a low-contrast second pass within the water.
    ticks = list(range(0, 7201, 1200))
    for value in ticks:
        y = Y0 + value*scale
        line([(126,y),(X1,y)], '#dfe5e5' if value else '#b9c9cb', .8)
        text(116,y,str(value),19,MUTED,'num','rm')
    line([(126,Y0),(126,Y1)], '#b9c9cb', .85)
    text(122, 278, '累计输出 ↓', 22, INK, 'sans', 'lt')
    text(122, 324, '100 tonnes', 17, MUTED, 'num', 'lb')

    # Material coordinates are mapped inside the exact per-layer masks.
    pw, ph = round((X1-X0)*S), round((Y1-Y0)*S)
    sample_x = X0 + (np.arange(pw)+.5)/S
    interp_values = layers_at(sample_x)
    boundary = np.cumsum(interp_values,axis=0)*scale*S
    top_edges = np.vstack([np.zeros((1,pw)),boundary[:-1]])
    yy = (np.arange(ph,dtype=np.float32)+.5)[:,None]
    source = np.asarray(Image.open(ROOT / 'assets/water_material.png').convert('RGB'),dtype=np.float32)/255
    lum = source @ np.array([.2126,.7152,.0722],dtype=np.float32)
    lo, hi = np.percentile(lum,[1,99.7])
    lum = np.clip((lum-lo)/(hi-lo),0,1)
    # Local photographic detail remains visible even inside the deepest layer.
    detail = lum-gaussian_filter(lum,9)
    lum = np.clip(lum + detail*.9,0,1)
    palettes = [
        ((3,91,66),(14,178,125),(143,249,212)),
        ((17,56,166),(38,107,221),(106,200,255)),
        ((8,16,45),(19,36,87),(46,73,129)),
        ((64,27,103),(131,68,180),(203,151,242)),
    ]
    ocean = np.zeros((ph,pw,4),dtype=np.uint8)
    labels = np.zeros((ph,pw),dtype=np.uint8)
    tex_x_base = (sample_x-X0)/(X1-X0)*(lum.shape[1]-1)
    for k in range(4):
        top, bottom = top_edges[k], boundary[k]
        inside = (yy>=top[None,:]) & (yy<bottom[None,:])
        v = np.clip((yy-top[None,:])/(bottom-top)[None,:],0,1)
        # Different portions of one continuous plate avoid obvious tiled bands.
        offset, depth = [(0,.96),(.05,.83),(.10,.90),(.08,.85)][k]
        ty = (offset + depth*v)*(lum.shape[0]-1)
        tx = tex_x_base if k%2==0 else lum.shape[1]-1-tex_x_base
        tx = np.broadcast_to(tx[None,:],ty.shape)
        material = map_coordinates(lum,[ty,tx],order=1,mode='nearest')
        material = np.clip(material,0,1)**[.80,.83,.62,.80][k]
        dark, middle, light = [np.array(c,dtype=np.float32) for c in palettes[k]]
        a = np.minimum(material*2,1)[:,:,None]
        b = np.maximum(material*2-1,0)[:,:,None]
        rgb = dark+(middle-dark)*a+(light-middle)*b
        ocean[inside,:3] = np.clip(rgb[inside],0,255).astype(np.uint8)
        ocean[inside,3] = 255
        labels[inside] = k+1
    water = Image.fromarray(ocean, 'RGBA')

    # The boat touches the shared zero line; its reflection is clipped to Source A.
    boat = Image.open(ROOT / 'assets/yacht.png').convert('RGBA')
    boat_width = 536
    boat_height = round(boat.height/boat.width*boat_width)
    boat = boat.resize((boat_width*S,boat_height*S),Image.Resampling.LANCZOS)
    boat_x, boat_y = 1544, Y0-boat_height
    reflection = boat.transpose(Image.Transpose.FLIP_TOP_BOTTOM).resize((boat.width,42*S),Image.Resampling.BICUBIC)
    ref_array = np.array(reflection)
    ref_array[:,:,3] = (ref_array[:,:,3].astype(float)*np.linspace(.15,0,42*S)[:,None]).astype(np.uint8)
    reflection = Image.fromarray(ref_array).filter(ImageFilter.GaussianBlur(1.4*S))
    reflected = Image.new('RGBA',water.size)
    reflected.alpha_composite(reflection,(round((boat_x-X0)*S),0))
    reflected.putalpha(Image.fromarray(np.minimum(np.array(reflected.getchannel('A')),np.where(labels==1,255,0).astype(np.uint8))))
    water = Image.alpha_composite(water,reflected)
    canvas.paste(water,(round(X0*S),round(Y0*S)),water)
    draw = ImageDraw.Draw(canvas)

    # Fine guides cross the actual colored object, never masking a thin layer.
    grid = Image.new('RGBA',canvas.size)
    gd = ImageDraw.Draw(grid)
    for x in xs:
        for y in np.arange(Y0+8,Y1,9):
            gd.line((round(x*S),round(y*S),round(x*S),round((y+3)*S)),fill=(255,255,255,42),width=1)
    for value in ticks[1:]:
        y = Y0+value*scale
        gd.line((round(X0*S),round(y*S),round(X1*S),round(y*S)),fill=(239,251,254,38),width=1)
    ga = np.array(grid.getchannel('A'))
    body_mask = np.zeros((H*S,W*S),dtype=np.uint8)
    body_mask[round(Y0*S):round(Y0*S)+ph,round(X0*S):round(X0*S)+pw] = np.where(labels>0,255,0)
    grid.putalpha(Image.fromarray(np.minimum(ga,body_mask)))
    canvas = Image.alpha_composite(canvas.convert('RGBA'),grid).convert('RGB')
    draw = ImageDraw.Draw(canvas)

    dense_x = np.linspace(X0,X1,1601)
    dense_b = bounds_at(dense_x)
    for k in range(3):
        line(list(zip(dense_x,dense_b[k])),['#91e9ca','#85b8ed','#7186bc'][k],.5)
    line(list(zip(dense_x,dense_b[3])), '#674284', 1.0)
    line([(X0,Y0),(X1,Y0)], '#40c59b', 1.4)
    # Tiny wave highlights sit within the topmost layer and do not displace zero.
    for start, end in [(310,440),(688,792),(1020,1194),(1328,1494),(1870,1980)]:
        xx = np.linspace(start,end,65)
        yy_r = Y0+2.5+1.3*np.sin(np.linspace(0,2*math.pi,65))
        line(list(zip(xx,yy_r)),'#99efcf',.55)
    canvas.paste(boat,(round(boat_x*S),round(boat_y*S)),boat)
    draw = ImageDraw.Draw(canvas)
    line([(boat_x+20,Y0),(X1,Y0)], '#55c8a8', .9)
    text(1130,318,'共同水面 = 0',18,'#4e766f','sans','mm')

    # White-space reading key, within the shallow left part of the section.
    assert float(np.max(bounds_at(np.linspace(181,603,100))[-1])) < 625
    rect((181,625,603,1060),BG)
    text(193, 635, 'HOW TO READ', 17, MUTED, 'bold')
    text(193, 670, '从水面向下，逐层累计', 30, INK, 'serif')
    text(193, 720, '每层厚度对应当期分项值', 22, MUTED)
    for k,s in enumerate(data['series']):
        y = 774 + k*48
        rect((194,y+2,229,y+15),COLORS[k],radius=2)
        text(249,y+9,s['label'],23,INK,'num','lm')
        text(467,y+9,'第 '+str(k+1)+' 层',20,MUTED,'sans','lm')
    line([(194,981),(229,981)], '#674284', 2)
    text(249,981,'总值 = A + B + C + D',22,INK,'sans','lm')
    text(193,1019,'底部轮廓表示合计，不增加第五层。',19,MUTED)

    # Direct source names sit on their respective material regions.
    label_x = 1708
    bound = bounds_at(label_x)
    upper = np.r_[Y0,bound[:-1]]
    for k,s in enumerate(data['series']):
        y = (upper[k]+bound[k])/2
        text(label_x,y,s['label'],23 if k!=2 else 28,'#f8ffff','bold','mm')

    # Sparse point annotations bind values to visible outer boundaries.
    total_annotations=[]
    for i, dx, dy, anchor in [(0,33,51,'lm'),(5,-18,42,'mm')]:
        x,y=xs[i],Y0+totals[i]*scale
        dot(x,y,3.7,'#ffffff','#674284',1.2)
        line([(x,y+5),(x+dx,y+dy-17)],'#a897b5',.8)
        txt=f'{categories[i]} · 总值 {int(totals[i])}'
        text(x+dx,y+dy,txt,21,'#684884','sans',anchor)
        total_annotations.append({'category':categories[i],'value':int(totals[i])})
    for i in [14,15]:
        x,y=xs[i],Y0+totals[i]*scale
        dot(x,y,3.8,'#fffaff','#674284',1)
        text(x-3 if i==15 else x,y-19,'Σ '+str(int(totals[i])),19,'#ffffff','bold','rm' if i==15 else 'mm')
        total_annotations.append({'category':categories[i],'value':int(totals[i])})

    # Explicit equal-period ticks, followed by the complete, exact numerical register.
    for i,x in enumerate(xs):
        line([(x,Y1+7),(x,Y1+15)],'#839ba1',.8)
        text(x,Y1+31,categories[i],20,INK,'bold','mm')
    text(X1,1202,'有序合成观测期  →',19,MUTED,'sans','rm')
    line([(120,1227),(2080,1227)], '#b9c9cb', 1)
    text(120,1245,'完整数值',25,INK,'serif')
    text(270,1250,'单位：100 吨 / 合成观测期  ·  100 tonnes per synthetic observation period',20,MUTED)
    groups=[(120,680,list(range(0,6))),(850,680,list(range(6,12))),(1580,500,list(range(12,16)))]
    table_records=[]
    for gx,gw,indices in groups:
        label_w=64
        colw=(gw-label_w)/len(indices)
        centers=[gx+label_w+(j+.5)*colw for j in range(len(indices))]
        header_y=1312
        text(gx,header_y,'来源',18,MUTED,'sans','lm')
        for j,i in enumerate(indices):
            text(centers[j],header_y,categories[i],20,INK,'bold','mm')
        line([(gx,1333),(gx+gw,1333)],'#c7d5d8',.8)
        for k,s in enumerate(data['series']):
            y=1356+k*32
            rect((gx,y-6,gx+10,y+6),COLORS[k],radius=1)
            text(gx+23,y,s['id'],20,INK,'bold','lm')
            for j,i in enumerate(indices):
                value=str(s['values'][i])
                text(centers[j],y,value,21,INK,'num','mm')
                table_records.append({'category':categories[i],'series':s['id'],'value':s['values'][i]})
        line([(gx,1472),(gx+gw,1472)],'#c7d5d8',.8)
        text(gx,1492,'总值',18,INK,'sans','lm')
        for j,i in enumerate(indices):
            text(centers[j],1492,str(int(totals[i])),21,INK,'bold','mm')
    text(120,1530,'数据来源：本地冻结合成数据  ·  相邻时期采用无过冲平滑连接',16,MUTED,'sans','lm')
    text(2080,1530,'SYNTHETIC DATA   /   A + B + C + D',16,MUTED,'num','rm')

    # Save the finished render. No post-render pixel changes are made by other scripts.
    canvas = canvas.resize((W,H),Image.Resampling.LANCZOS)
    canvas.save(ROOT / 'final.png',compress_level=6)
    artifact_dir=ROOT / 'artifacts'
    artifact_dir.mkdir(exist_ok=True)
    # Diagnostic material mask, not another full chart render.
    Image.fromarray(labels).save(artifact_dir / 'layer_mask_2x.png')
    raster_measurements=[]
    for i,x in enumerate(xs):
        col=min(pw-1,max(0,round((x-X0)*S)))
        scan=labels[:,col]
        found=[]
        for k in range(4):
            hits=np.where(scan==k+1)[0]
            found.append({'series':data['series'][k]['id'],
                'visible_mask_thickness_px':float(len(hits)/S),
                'expected_thickness_px':round(float(values[k,i]*scale),6),
                'visible_mask_lower_y':float(Y0+(hits[-1]+1)/S) if len(hits) else None,
                'expected_lower_y':round(float(Y0+cumulative[k,i]*scale),6)})
        raster_measurements.append({'category':categories[i],'x':float(x),'layers':found})
    diagnostics={
        'canvas':[W,H], 'plot':{'x0':X0,'x1':X1,'y0':Y0,'y1':Y1,'pixels_per_unit':scale},
        'interpolation':'Shared convex smoothstep weights 3t^2-2t^3 for every series; no overshoot or negative thickness.',
        'category_order':categories, 'series_order':[s['id'] for s in data['series']],
        'totals':[int(x) for x in totals], 'numeric_table':table_records,
        'rasterized_mask_measurements':raster_measurements,
        'generated_material_coverage_px':round(float(np.count_nonzero(labels)/(S*S)),2),
        'text_boxes':text_records,
    }
    (artifact_dir/'render_diagnostics.json').write_text(json.dumps(diagnostics,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Wrote final.png at 2200 × 1552; four textured layers, 16 ordered periods, 64 values and 16 totals.')

if __name__=='__main__':
    main()
