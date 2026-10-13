"""Offline data-bound pasta infographic. Run through capture_render.py.

Generated photographs provide the pasta and every paper face. A single linear
screen-height mapping places each pasta tip relative to the common front rim.
No network, model call, reference pixels, minimum mark, or per-value scale.
"""
from pathlib import Path
import json
import math
import numpy as np
from scipy.ndimage import map_coordinates
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent
S = 3
BG = '#f8f5ef'
INK = '#252a27'
RED = '#ef392a'
MUTED = '#736f64'
BASE = 1070.0
SCALE_HEIGHT = 840.0
CENTERS = [300, 600, 900, 1200, 1500, 1800]
FRONT_W = 136
SIDE_W = 23
SLEEVE_H = 238
OPENING_DEPTH = 4
PASTA_W = 123

def load_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))

def font(size, face='body'):
    suffix = '.ttf' if face == 'numbers' else '.otf'
    return ImageFont.truetype(str(ROOT / 'assets/fonts' / (face+suffix)), round(size*S))

def text(draw, xy, message, size, color=INK, face='body', anchor='lt'):
    draw.text(tuple(round(x*S) for x in xy), message, font=font(size,face), fill=color, anchor=anchor)

def tracked(draw, xy, message, size, spacing, color=INK, face='body'):
    x,y = xy
    f=font(size,face)
    for char in message:
        draw.text((round(x*S),round(y*S)), char, font=f, fill=color, anchor='lt')
        x += f.getlength(char)/S + spacing
    return x

def line(draw, pts, fill, width=1):
    draw.line([(round(x*S),round(y*S)) for x,y in pts], fill=fill, width=max(1,round(width*S)))

def paste_perspective(canvas, asset, points):
    """Map a rectangular generated face into an explicit 4-corner paper plane."""
    ps = np.array(points,dtype=float)*S
    minx,miny=np.floor(ps.min(axis=0)).astype(int)
    maxx,maxy=np.ceil(ps.max(axis=0)).astype(int)
    target=ps-np.array([minx,miny])
    source=np.array([[0,0],[asset.width,0],[asset.width,asset.height],[0,asset.height]],float)
    mat=[]; rhs=[]
    for (x,y),(u,v) in zip(target,source):
        mat.extend([[x,y,1,0,0,0,-u*x,-u*y],[0,0,0,x,y,1,-v*x,-v*y]])
        rhs.extend([u,v])
    coeff=np.linalg.solve(np.array(mat),np.array(rhs))
    tile=asset.transform((maxx-minx,maxy-miny),Image.Transform.PERSPECTIVE,coeff,Image.Resampling.BICUBIC)
    mask=Image.new('L',tile.size)
    ImageDraw.Draw(mask).polygon([tuple(p) for p in target],fill=255)
    tile.putalpha(mask)
    canvas.alpha_composite(tile,(int(minx),int(miny)))

def pasta_tile(asset, height):
    """Use the same physical texture map for every bundle, then crop at its rim.

    A fixed shallow tip transform preserves many distinct photographic tips in
    the smallest mark. It is identical across categories and does not change
    the encoded top-to-rim distance. Longer marks extend through the same raw
    spaghetti material; they are not differently scaled photographs.
    """
    width=round(PASTA_W*S)
    out_h=math.ceil((height+12)*S)
    xx,yy=np.meshgrid(np.arange(width)/S,np.arange(out_h)/S)
    sx=xx/(PASTA_W)*(asset.shape[1]-1)
    sy=np.where(yy<9, yy*(60/9), 60+(yy-9)*1.20)
    channels=[map_coordinates(asset[:,:,c].astype(float),[sy,sx],order=1,mode='nearest') for c in range(4)]
    return Image.fromarray(np.stack(channels,axis=2).clip(0,255).astype('uint8'),'RGBA')

def star(cx,cy,r):
    return [(cx+math.cos(-math.pi/2+i*math.pi/5)*r*(1 if i%2==0 else .40),
             cy+math.sin(-math.pi/2+i*math.pi/5)*r*(1 if i%2==0 else .40)) for i in range(10)]

def flag(name):
    # Country identifiers are non-quantitative, constructed independently of data.
    im=Image.new('RGBA',(180,112),'white'); d=ImageDraw.Draw(im)
    if name=='Italy':
        d.rectangle((0,0,60,112),fill='#008c46');d.rectangle((120,0,180,112),fill='#e73137')
    elif name=='France':
        d.rectangle((0,0,60,112),fill='#163d8a');d.rectangle((120,0,180,112),fill='#e43242')
    elif name=='Germany':
        d.rectangle((0,0,180,37),fill='#20201e');d.rectangle((0,37,180,75),fill='#d82732');d.rectangle((0,75,180,112),fill='#f5c847')
    elif name=='Spain':
        d.rectangle((0,0,180,28),fill='#c92834');d.rectangle((0,28,180,84),fill='#f3c846');d.rectangle((0,84,180,112),fill='#c92834')
        # Compact emblem identifying the national flag at final print size.
        d.rectangle((45,44,59,66),fill='#b6343c');d.polygon([(45,66),(59,66),(52,73)],fill='#b6343c')
        d.rectangle((47,39,57,44),fill='#b6343c');d.rectangle((42,49,45,68),fill='#f0ebe1');d.rectangle((60,49,63,68),fill='#f0ebe1')
        d.rectangle((49,49,55,57),fill='#f3c846')
    elif name=='USA':
        for k in range(13):
            d.rectangle((0,round(k*112/13),180,round((k+1)*112/13)),fill='#c93e47' if k%2==0 else '#fffdf8')
        d.rectangle((0,0,78,60),fill='#27456a')
        for row in range(9):
            count=6 if row%2==0 else 5
            for col in range(count):
                d.polygon(star(7+col*13+(0 if row%2==0 else 6.5),5+row*6.25,2.4),fill='white')
    elif name=='UK':
        d.rectangle((0,0,180,112),fill='#203f77')
        d.line((0,0,180,112),fill='#fffdf8',width=24);d.line((0,112,180,0),fill='#fffdf8',width=24)
        d.polygon([(0,0),(12,0),(180,104),(180,112),(168,112),(0,8)],fill='#cc303f')
        d.polygon([(0,112),(0,104),(168,0),(180,0),(180,8),(12,112)],fill='#cc303f')
        d.rectangle((0,38,180,74),fill='#fffdf8');d.rectangle((72,0,108,112),fill='#fffdf8')
        d.rectangle((0,44,180,68),fill='#cc303f');d.rectangle((79,0,101,112),fill='#cc303f')
    else:
        raise ValueError('Unspecified category identifier: '+name)
    return im

def main():
    brief=load_json('brief.json'); data=load_json('data.json'); core=load_json('core.json')
    assert core['visible_title']==data['title']
    assert data['value_axis']['scale']=='linear' and data['value_axis']['minimum']==0
    assert len(data['series'])==1 and len(data['categories'])==len(CENTERS)
    series=data['series'][0]
    values=series['values']; maximum=data['value_axis']['maximum']
    assert sum(values)==data['series_totals'][series['id']]
    assert all(0<=v<=maximum for v in values)
    W,H=brief['width'],brief['height']
    canvas=Image.new('RGBA',(W*S,H*S),BG); d=ImageDraw.Draw(canvas)

    # Editorial typography and the unambiguous synthetic-data notice.
    tracked(d,(95,70),'SIX REGIONS / ONE SCALE',18,2.1,MUTED,'heading')
    d.rounded_rectangle((800*S,66*S,1100*S,110*S),radius=4*S,fill=RED)
    tracked(d,(818,77),data['visible_notice'],21,1.6,'#fffaf3','heading')
    text(d,(780,140),'PASTA',335,RED,'heading')
    tracked(d,(800,417),'PRODUCTION',96,5,RED,'heading')
    text(d,(805,551),series['label']+' · '+series['unit'],27,INK)
    text(d,(805,607),'Exposed spaghetti height represents production.',25,MUTED)
    text(d,(805,644),'Shared linear scale: 0–3,500,000 tonnes.',25,MUTED)

    # A discreet scale supports reading without replacing the object chart.
    text(d,(96,165),'METRIC',17,MUTED,'heading',anchor='rt')
    text(d,(96,186),'TONNES',17,MUTED,'heading',anchor='rt')
    line(d,[(116,BASE-SCALE_HEIGHT),(116,BASE)],'#c9c4ba',1)
    ticks=[0,500000,1000000,1500000,2000000,2500000,3000000,3500000]
    for val in ticks:
        y=BASE-val/maximum*SCALE_HEIGHT
        line(d,[(109,y),(126,y)],'#ada99d',1)
        label='0' if val==0 else (str(val/1000000).rstrip('0').rstrip('.')+'M')
        text(d,(96,y),label,19,MUTED,anchor='rm')
    for x in range(145,1906,10):
        line(d,[(x,BASE),(x+3,BASE)],'#c4bfb4',.75)

    # Non-data cast shadows sit below each paper sleeve.
    shadow=Image.new('RGBA',canvas.size)
    sd=ImageDraw.Draw(shadow)
    for cx in CENTERS:
        sd.ellipse(((cx-55)*S,(BASE+SLEEVE_H-7)*S,(cx+125)*S,(BASE+SLEEVE_H+14)*S),fill=(102,77,42,40))
    shadow=shadow.filter(ImageFilter.GaussianBlur(8*S)); canvas.alpha_composite(shadow)

    front=Image.open(ROOT/'assets/sleeve_front.png').convert('RGBA')
    side=Image.open(ROOT/'assets/sleeve_side.png').convert('RGBA')
    opening=Image.open(ROOT/'assets/sleeve_opening.png').convert('RGBA')
    pasta=np.array(Image.open(ROOT/'assets/pasta_bundle.png').convert('RGBA'))
    records=[]
    for cx,cat,value in zip(CENTERS,data['categories'],values):
        h=value/maximum*SCALE_HEIGHT
        top=BASE-h
        left=cx-FRONT_W/2
        right=cx+FRONT_W/2
        # Back of the empty sleeve is laid down first. Pasta enters it, and the
        # opaque front is laid down last at exactly BASE for every category.
        paste_perspective(canvas,opening,[(left+12,BASE-OPENING_DEPTH),(right+SIDE_W,BASE-OPENING_DEPTH),(right,BASE),(left,BASE)])
        if value>0:
            tile=pasta_tile(pasta,h)
            xpix=round((cx-PASTA_W/2)*S); ypix=round(top*S)
            canvas.alpha_composite(tile,(xpix,ypix))
        else:
            ypix=round(BASE*S)
        paste_perspective(canvas,side,[(right,BASE),(right+SIDE_W,BASE-OPENING_DEPTH),(right+SIDE_W,BASE+SLEEVE_H-11),(right,BASE+SLEEVE_H)])
        face=front.resize((round(FRONT_W*S),round(SLEEVE_H*S)),Image.Resampling.LANCZOS)
        canvas.alpha_composite(face,(round(left*S),round(BASE*S)))
        d=ImageDraw.Draw(canvas)
        # A fine front paper edge, strictly at or below the baseline, cannot
        # occlude any positive-height pasta tips above it.
        line(d,[(left,BASE+.25),(right,BASE+.25)],'#46a2a6',.65)
        line(d,[(right-.3,BASE+1),(right-.3,BASE+SLEEVE_H-.5)],'#217781',.55)
        label=f'{value:,}'
        label_y=top-56
        text(d,(cx,label_y),label,40,INK,'numbers',anchor='mt')
        # A small amount of vertical breathing room keeps all tips unobstructed.
        fl=flag(cat).resize((54*S,34*S),Image.Resampling.LANCZOS)
        canvas.alpha_composite(fl,(round((cx-27)*S),1341*S))
        d=ImageDraw.Draw(canvas)
        d.rectangle(((cx-27)*S,1341*S,(cx+27)*S,1375*S),outline='#ddd7ca',width=1)
        text(d,(cx,1391),cat,29,INK,'heading',anchor='mt')
        records.append({'category':cat,'value':value,'unit':series['unit'],'center_x':cx,
            'front_rim_y':BASE,'target_height_px':h,'target_top_y':top,
            'raster_placement_top_y':ypix/S,'numeric_label':label,'label_top_y':label_y,
            'label_bbox':list(d.textbbox((round(cx*S),round(label_y*S)),label,font=font(40,'numbers'),anchor='mt'))})

    d=ImageDraw.Draw(canvas)
    line(d,[(95,1441),(1905,1441)],'#d8d2c6',1)
    text(d,(96,1457),'Source: supplied synthetic dataset. Figures are illustrative.',19,MUTED)
    text(d,(1905,1457),'Height above the rim encodes output; paper sleeves do not.',19,MUTED,anchor='rt')
    final=canvas.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
    final.save(ROOT/'final.png',compress_level=8)
    recipe={'canvas':[W,H],'supersampling':S,'scale_range':[0,maximum],
        'scale_height_px':SCALE_HEIGHT,'baseline_y':BASE,'no_minimum_mark_height':True,
        'category_order':data['categories'],'series':series['label'],'records':records,
        'asset_use_zh':'六束意面及六个纸套的前面、侧面、开口均来自 assets/raw_01.png；参考图像素未参与合成。'}
    (ROOT/'artifacts/geometry.json').write_text(json.dumps(recipe,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Saved final.png at',W,H)

if __name__=='__main__':
    main()
