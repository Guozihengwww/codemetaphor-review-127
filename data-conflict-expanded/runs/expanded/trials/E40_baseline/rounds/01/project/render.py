#!/usr/bin/env python3
"""Offline composition: generated illustration parts + data-defined leg geometry."""
from pathlib import Path
from io import BytesIO
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cairosvg

ROOT = Path(__file__).resolve().parent
S = 2
BRIEF = json.loads((ROOT / 'brief.json').read_text(encoding='utf-8'))
DATA = json.loads((ROOT / 'data.json').read_text(encoding='utf-8'))
CORE = json.loads((ROOT / 'core.json').read_text(encoding='utf-8'))
W, H = BRIEF['width'], BRIEF['height']
PAPER = '#F3EFE6'
INK = '#272421'
MUTED = '#68615A'
RED = '#DC424B'
CORE_COLOR = '#FFE8BB'
canvas = Image.new('RGBA', (W*S, H*S), PAPER)
draw = ImageDraw.Draw(canvas)
text_records = []


def font(size, face='sans'):
    return ImageFont.truetype(str(ROOT / 'fonts' / (face+'.otf')), round(size*S))


def text(x, y, value, size, face='sans', fill=INK, anchor='lt', label=None):
    f = font(size, face)
    at = (round(x*S), round(y*S))
    draw.text(at, str(value), font=f, fill=fill, anchor=anchor)
    bbox = draw.textbbox(at, str(value), font=f, anchor=anchor)
    text_records.append({'label': label or str(value), 'text': str(value),
                         'bbox': [round(z/S, 2) for z in bbox]})


def line(coords, fill, width=1):
    draw.line([(round(x*S), round(y*S)) for x,y in coords], fill=fill,
              width=max(1,round(width*S)), joint='curve')


def ellipse(x,y,r,fill,outline=None,width=1):
    draw.ellipse([(round((x-r)*S),round((y-r)*S)),
                  (round((x+r)*S),round((y+r)*S))],fill=fill,outline=outline,width=round(width*S))


def svg_layer(path, fill, stroke=None, sw=0):
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><path d="{path}" fill="{fill}" stroke="{stroke or "none"}" stroke-width="{sw}" stroke-linejoin="round"/></svg>'
    return Image.open(BytesIO(cairosvg.svg2png(bytestring=svg.encode('utf-8'), output_width=W*S, output_height=H*S))).convert('RGBA')


def affine_asset(path, source, target):
    im = Image.open(ROOT / path).convert('RGBA')
    src = np.column_stack((np.asarray(source,float),np.ones(3)))
    dst = np.asarray(target,float)*S
    mat = np.linalg.solve(src,dst).T
    inv = np.linalg.inv(np.vstack((mat,[0,0,1])))[:2]
    return im.transform((W*S,H*S), Image.Transform.AFFINE, tuple(inv.ravel()), Image.Resampling.BICUBIC)


def shoe_layer(ankle, toe):
    # Fixed image landmarks identify the shoe opening and pointed toe.
    src_a = np.array([130.,140.])
    src_b = np.array([492.,322.])
    a, b = np.asarray(ankle,float),np.asarray(toe,float)
    v = src_b-src_a
    q = b-a
    k = np.linalg.norm(q)/np.linalg.norm(v)
    angle = math.atan2(q[1],q[0])-math.atan2(v[1],v[0])
    R = np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])*k
    third = src_a + np.array([-v[1],v[0]])
    target_third = a+R@(third-src_a)
    return affine_asset('assets/heel.png',[src_a,src_b,third],[a,b,target_third])


def stocking(path, centerline, rear=False):
    mask_hi = svg_layer(path,'white').getchannel('A')
    mask = mask_hi.resize((W,H),Image.Resampling.LANCZOS)
    box = mask.getbbox()
    x0,y0,x1,y1 = box
    yy,xx = np.mgrid[y0:y1,x0:x1].astype(np.float32)
    mind = np.full(xx.shape,np.inf,dtype=np.float32)
    along = np.zeros_like(xx)
    across = np.zeros_like(xx)
    cumulative = 0.
    for a,b in zip(centerline[:-1],centerline[1:]):
        a,b = np.asarray(a,float),np.asarray(b,float)
        v = b-a
        length = np.linalg.norm(v)
        t = np.clip(((xx-a[0])*v[0]+(yy-a[1])*v[1])/(length*length),0,1)
        dx,dy = xx-(a[0]+t*v[0]),yy-(a[1]+t*v[1])
        dist = dx*dx+dy*dy
        keep = dist < mind
        mind[keep] = dist[keep]
        along[keep] = (cumulative+t*length)[keep]
        across[keep] = ((dx*(-v[1])+dy*v[0])/length)[keep]
        cumulative += length
    tex = np.array(Image.open(ROOT/'assets/net_fabric.png').convert('RGB'))
    th,tw = tex.shape[:2]
    def reflect(x,n):
        u = np.mod(x,2*(n-1))
        return np.where(u<n,u,2*(n-1)-u).astype(np.int32)
    u = reflect(across*2.2+130,tw)
    v = reflect(along*2.2+43,th)
    pixels = tex[v,u].astype(np.float32)
    # Engraved fabric follows the limb; soft edge shading supplies volume.
    falloff = np.minimum(abs(across)/170,1)
    shade = 1.-0.10*falloff
    pixels *= shade[:,:,None]
    if rear:
        pixels = pixels*.84 + np.array([221,216,204])*.16
    rgba = np.dstack((np.clip(pixels,0,255).astype(np.uint8), np.array(mask)[y0:y1,x0:x1]))
    tile = Image.fromarray(rgba,'RGBA').resize(((x1-x0)*S,(y1-y0)*S),Image.Resampling.LANCZOS)
    # Replace the upsampled alpha with the high-resolution path mask.
    tile.putalpha(mask_hi.crop((x0*S,y0*S,x1*S,y1*S)))
    canvas.alpha_composite(tile,(x0*S,y0*S))
    canvas.alpha_composite(svg_layer(path,'none',INK,3.2 if not rear else 2.8))


def main():
    global draw
    assert DATA['synthetic'] and len(DATA['series']) == 1
    series=DATA['series'][0]
    times=np.asarray(DATA['time_points'],float)
    values=np.asarray(series['values'],float)
    lo,hi=DATA['value_axis']['minimum'],DATA['value_axis']['maximum']
    assert DATA['value_axis']['scale']=='linear'
    xmin,xmax,ytop,ybottom = 650.,1730.,600.,1570.
    xs=xmin+(times-times.min())/(times.max()-times.min())*(xmax-xmin)
    ys=ybottom-(values-lo)/(hi-lo)*(ybottom-ytop)
    pts=list(zip(xs.tolist(),ys.tolist()))
    p0,p1,p2,p3,p4=[np.array(p,float) for p in pts]

    # Publication masthead; every string is rendered in code.
    line([(100,73),(1900,73)],INK,2)
    draw.rounded_rectangle((100*S,105*S,408*S,152*S),radius=5*S,fill=RED)
    text(254,128.5,DATA['visible_notice'],22,'sans-bold',PAPER,'mm')
    text(1900,116,'THE PRICE PERFORMANCE',22,'sans-bold',MUTED,'rt')
    text(1900,153,'1978 — 1982  /  ONE CARAT',21,'sans',MUTED,'rt')
    text(96,192,'ONE CARAT',116,'narrow-bold')
    text(96,306,'DIAMOND PRICES',116,'narrow-bold')
    text(103,433,'A fictional series, performed as a bent leg.',29,'serif-italic',MUTED)

    line([(1396,278),(1452,278)],RED,10)
    ellipse(1424,278,9,RED)
    text(1481,278,series['label'],29,'sans-bold',INK,'lm')
    text(1395,325,series['unit'],22,'sans',MUTED)

    # Restrained theatrical backdrop with a quiet grid, separate from the data path.
    bgpath='M 102 528 L 1850 528 L 1850 1595 L 102 1595 Z'
    canvas.alpha_composite(svg_layer(bgpath,'#D9D6CF'))
    draw=ImageDraw.Draw(canvas)
    # Muted stage floor reinforces the reclining pose without encoding data.
    stage='M 102 1488 L 1850 1550 L 1850 1595 L 102 1595 Z'
    canvas.alpha_composite(svg_layer(stage,'#CC9B96'))
    draw=ImageDraw.Draw(canvas)
    ticks=[10000,20000,30000,40000,50000,60000,65000]
    tick_records=[]
    for value in ticks:
        y=ybottom-(value-lo)/(hi-lo)*(ybottom-ytop)
        line([(600,y),(1790,y)],'#BCB8B0',1.1)
        line([(1790,y),(1802,y)],MUTED,1.5)
        text(1875,y,f'{value:,}',22,'sans',MUTED,'rm')
        tick_records.append({'value':value,'y':y})
    text(1875,557,'USD',20,'sans-bold',MUTED,'rt')

    # Rear leg and elevated shoe are identity geometry, not a second data series.
    rear_ankle=np.array([1737.,491.])
    rear_toe=np.array([1839.,291.])
    rearpath=(f'M {p0[0]-62} {p0[1]-175} '
              f'C 809 969 1240 652 1698 466 '
              f'Q 1727 456 1750 475 L 1780 530 '
              f'C 1510 683 1200 974 {p0[0]+157} {p0[1]-29} '
              f'Q {p0[0]+49} {p0[1]+14} {p0[0]-62} {p0[1]-175} Z')
    stocking(rearpath,[(680,1250),(1180,844),rear_ankle],rear=True)
    canvas.alpha_composite(shoe_layer(rear_ankle,rear_toe))

    # The foreground stocking's inner edge is the EXACT data polyline.
    frontpath=(f'M {p0[0]-90} {p0[1]-175} '
        f'C {p0[0]+120} {p0[1]-220} {p2[0]-180} {p2[1]-190} {p2[0]-75} {p2[1]-118} '
        f'Q {p2[0]-8} {p2[1]-161} {p2[0]+33} {p2[1]-65} '
        f'C {p2[0]+76} {p2[1]+55} {p3[0]+40} {p3[1]-130} {p3[0]+64} {p3[1]-40} '
        f'L {p3[0]+90} {p3[1]+42} L {p3[0]-27} {p3[1]+68} '
        f'L {p3[0]-40} {p3[1]+12} L {p3[0]} {p3[1]} '
        f'L {p2[0]} {p2[1]} L {p1[0]} {p1[1]} L {p0[0]} {p0[1]} '
        f'Q {p0[0]-80} {p0[1]+14} {p0[0]-90} {p0[1]-175} Z')
    stocking(frontpath,[p0+[-10,-95],p1+[-75,-210],p2+[-10,-60],p3+[22,-18]])
    canvas.alpha_composite(shoe_layer(p3,p4))

    # The generated upper body is attached over the two stocking roots.
    upper = affine_asset('assets/performer_upper.png',
        [(710,860),(120,965),(455,265)],
        [(650,1280),(135,1500),(360,790)])
    canvas.alpha_composite(upper)
    draw=ImageDraw.Draw(canvas)

    # The data path is a five-vertex polyline. No curve interpolation or overshoot.
    line(pts,'#F8F1E4',22)
    line(pts,RED,14)
    for x,y in pts:
        ellipse(x,y,17,RED)
        ellipse(x,y,7,CORE_COLOR)

    # All value labels are bound to their corresponding data points.
    offsets=[(-6,58),(-14,70),(38,-89),(93,-34),(0,75)]
    label_anchors=['mt','mt','lt','lt','mt']
    for i,((x,y),value,(dx,dy),anchor) in enumerate(zip(pts,values,offsets,label_anchors)):
        tx,ty=x+dx,y+dy
        text(tx,ty,f'${int(value):,}',32,'sans-bold',INK,anchor,label=f'value_{DATA["categories"][i]}')

    # Shared year axis: positions follow the supplied numerical time points.
    line([(xmin,1627),(xmax,1627)],INK,2)
    for i,(x,y) in enumerate(pts):
        line([(x,1616),(x,1637)],INK,2)
        text(x,1655,DATA['categories'][i],31,'sans-bold',INK,'mt',label=f'year_{i}')
    text(1898,1657,'YEAR',21,'sans-bold',MUTED,'rt')

    line([(100,1750),(1900,1750)],INK,1.5)
    text(101,1787,'THE LINE FOLLOWS THE LEG.',25,'sans-bold')
    text(101,1830,'Dot height shows price; years are equally spaced.',26,'sans',MUTED)
    text(1898,1787,'FIVE FICTIONAL OBSERVATIONS',22,'sans-bold',MUTED,'rt')
    text(1898,1827,'Illustrative synthetic values — not market data.',23,'serif-italic',MUTED,'rt')
    text(101,1890,'SYNTHETIC  /  ONE-CARAT DIAMOND',16,'sans-bold',MUTED)
    text(1898,1890,'LINEAR SCALE  ·  10,000–65,000 USD',16,'sans-bold',MUTED,'rt')

    out=canvas.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
    out.save(ROOT/'final.png',optimize=True)
    metrics={'canvas':[W,H],'axis':{'min':lo,'max':hi,'scale':'linear',
        'left':xmin,'right':xmax,'top':ytop,'bottom':ybottom},
        'points':[{'category':c,'time':float(t),'value':int(v),'center':[float(x),float(y)]}
            for c,t,v,(x,y) in zip(DATA['categories'],times,values,pts)],
        'ticks':tick_records,'path_type':'piecewise_linear','marker_core_rgb':[255,232,187],
        'text_bboxes':text_records,'front_leg_inner_edge':pts[:4],
        'generated_components':['performer_upper','net_fabric','heel'],
        'identity_geometry_note':'Rear leg and limb thickness are non-data geometry.'}
    (ROOT/'artifacts').mkdir(exist_ok=True)
    (ROOT/'artifacts/render_metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
