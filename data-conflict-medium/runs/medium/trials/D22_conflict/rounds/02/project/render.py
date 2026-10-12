#!/usr/bin/env python3
"""Render the frozen synthetic data using local generated paint and hand assets.

Run from this directory: PYTHON render.py
Complete experimental renders must be invoked through capture_render.py.
"""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parent
S = 3
INK = (24, 25, 24, 255)
GRAY = (86, 85, 79, 255)
R0 = 32.0
RMAX = 178.0
CENTERS = [(510, 570), (1250, 590), (310, 1050), (890, 1030),
           (1480, 1070), (500, 1540), (1080, 1505)]
HAND_X, HAND_Y, HAND_SCALE = 900, 1600, 0.40
PINCH = (HAND_X + 1145 * HAND_SCALE, HAND_Y + 145 * HAND_SCALE)


def font(name, size):
    return ImageFont.truetype(str(ROOT / 'assets/fonts' / name), round(size*S))


def pxy(p):
    return tuple(round(t*S) for t in p)


def text(draw, xy, content, size=24, face='EditorialSans.otf', fill=INK, anchor='ls'):
    draw.text(pxy(xy), str(content), font=font(face, size), fill=fill, anchor=anchor)


def tracking(draw, xy, content, size=18, gap=2.3, fill=INK, center=False):
    f = font('EditorialSans-Bold.otf', size)
    widths = [draw.textlength(c, font=f)/S for c in content]
    total = sum(widths) + gap*(len(content)-1)
    x, y = xy
    if center:
        x -= total/2
    for c, width in zip(content, widths):
        draw.text(pxy((x,y)),c,font=f,fill=fill,anchor='ls')
        x += width+gap


def line(draw, points, fill=INK, width=1.5):
    draw.line([pxy(p) for p in points], fill=fill, width=max(1,round(width*S)), joint='curve')


def cubic(points, steps=110):
    p = np.asarray(points, dtype=float)
    t = np.linspace(0,1,steps)[:,None]
    return (1-t)**3*p[0]+3*(1-t)**2*t*p[1]+3*(1-t)*t*t*p[2]+t**3*p[3]


def curved_line(draw, start, segments, width=1.6):
    path = []
    for a,b,c in segments:
        path.extend(cubic([start,a,b,c]).tolist())
        start = c
    line(draw,path,width=width)


def circle_points(cx,cy,r,start=-90,end=270):
    a=np.radians(np.linspace(start,end,950))
    return np.column_stack([cx+r*np.cos(a),cy+r*np.sin(a)]).tolist()


def arc_text(canvas, center, radius, label, size=34):
    """Place glyph baselines on one circular arc, preserving font advances."""
    f=font('EditorialSerif.otf',size)
    d=ImageDraw.Draw(canvas)
    widths=[d.textlength(label[:i+1],font=f)/S for i in range(len(label))]
    total=widths[-1]
    before=0
    for c,after in zip(label,widths):
        adv=after-before
        a=-math.pi/2+((before+adv/2)-total/2)/radius
        before=after
        if c.isspace():
            continue
        tile=Image.new('RGBA',(round(size*5*S),round(size*5*S)))
        td=ImageDraw.Draw(tile)
        pivot=(tile.width//2,tile.height//2)
        td.text(pivot,c,font=f,fill=INK,anchor='ms')
        tile=tile.rotate(-math.degrees(a)-90,resample=Image.Resampling.BICUBIC,center=pivot)
        x=center[0]+radius*math.cos(a)
        y=center[1]+radius*math.sin(a)
        canvas.alpha_composite(tile,(round(x*S-pivot[0]),round(y*S-pivot[1])))


def make_paper(w,h):
    rng=np.random.default_rng(20261012)
    fine=rng.normal(0,0.85,(h,w)).astype(np.float32)
    cloud=gaussian_filter(rng.normal(0,1,(h,w)).astype(np.float32),18)
    cloud=cloud/(cloud.std()+1e-6)*0.40
    base=np.array([247.0,245.5,239.2],dtype=np.float32)
    rgb=np.clip(base[None,None,:]+fine[:,:,None]+cloud[:,:,None],0,255).astype(np.uint8)
    paper=Image.fromarray(rgb).convert('RGBA').resize((w*S,h*S),Image.Resampling.BICUBIC)
    d=ImageDraw.Draw(paper)
    for _ in range(3900):
        x=float(rng.uniform(0,w)); y=float(rng.uniform(0,h))
        radius=float(rng.uniform(.15,.6))
        v=int(rng.uniform(165,225))
        d.ellipse(pxy((x-radius,y-radius,x+radius,y+radius)),fill=(v,v,v,100))
    # A restrained trace of pigment at the trim edges echoes the paint materials.
    edge_colors=[(237,105,47,165),(43,83,220,150),(232,180,31,140)]
    for i in range(200):
        side=int(rng.integers(0,3))
        if side==0:
            x=abs(float(rng.normal(0,15))); y=float(rng.uniform(430,1540))
        elif side==1:
            x=w-abs(float(rng.normal(0,13))); y=float(rng.uniform(70,730))
        else:
            x=w-abs(float(rng.normal(0,12))); y=float(rng.uniform(1760,h))
        r=float(rng.uniform(.3,1.55))
        d.ellipse(pxy((x-r,y-r,x+r,y+r)),fill=edge_colors[[1,2,0][side]])
    return paper


def disk(canvas, center, radius, asset, index):
    """Analytic subpixel disk mask is the only data boundary; shadow is separate."""
    cx,cy=center
    pad=math.ceil(radius*.32+10)
    x0=math.floor(cx-radius-pad); y0=math.floor(cy-radius-pad)
    x1=math.ceil(cx+radius+pad); y1=math.ceil(cy+radius+pad)
    w=(x1-x0)*S;h=(y1-y0)*S
    yy,xx=np.mgrid[0:h,0:w]
    dx=(xx+.5)/S+x0-cx;dy=(yy+.5)/S+y0-cy
    dist=np.hypot(dx,dy)
    alpha=np.clip((radius-dist)*S+.5,0,1)
    mask=Image.fromarray(np.rint(alpha*255).astype(np.uint8))
    # Material is stretched to an exact square and is opaque under the circle.
    src=Image.open(ROOT/asset).convert('RGB')
    tex=src.transform((w,h),Image.Transform.AFFINE,
        (src.width/(2*radius*S),0,(x0-cx+radius)*src.width/(2*radius),
         0,src.height/(2*radius*S),(y0-cy+radius)*src.height/(2*radius)),
        resample=Image.Resampling.BICUBIC)
    shadow_mask=mask.filter(ImageFilter.GaussianBlur(radius*.075*S))
    shadow_mask=shadow_mask.point(lambda v: round(v*.19))
    shadow=Image.new('RGBA',(w,h),(34,28,24,0));shadow.putalpha(shadow_mask)
    canvas.alpha_composite(shadow,(x0*S+round(radius*.10*S),y0*S+round(radius*.14*S)))
    tex.putalpha(mask)
    canvas.alpha_composite(tex,(x0*S,y0*S))
    # Local mask is evidence for the visible shape, not a second full chart render.
    mask.save(ROOT/f'artifacts/disk_mask_{index:02}.png')
    mask_n=np.asarray(mask)>127
    ys,xs=np.where(mask_n)
    return {'center':list(center),'radius_formula_px':radius,'mask_origin':[x0,y0],
            'mask_scale':S,'mask_bbox_px':[x0+xs.min()/S,y0+ys.min()/S,
                                        x0+(xs.max()+1)/S,y0+(ys.max()+1)/S],
            'mask_area_px2':float(np.asarray(mask).sum()/255/S**2),
            'asset':asset}


def main():
    brief=json.loads((ROOT/'brief.json').read_text(encoding='utf-8'))
    data=json.loads((ROOT/'data.json').read_text(encoding='utf-8'))
    w,h=brief['width'],brief['height']
    assert (w,h)==(1800,2160)
    cats=data['categories']; values=data['series'][0]['values']
    assert len(cats)==len(values)==len(CENTERS)==7
    assert data['value_axis']=={'minimum':0,'maximum':50,'scale':'linear'}
    radii=[R0+(RMAX-R0)*v/50 for v in values]
    canvas=make_paper(w,h)
    draw=ImageDraw.Draw(canvas)

    # Header uses the frozen title and series, with a conspicuous data notice.
    tracking(draw,(900,79),data['visible_notice'],size=20,gap=3.1,center=True)
    line(draw,[(105,110),(1695,110)],fill=(81,79,72,170),width=.75)
    lat='Synthetic '
    zh=data['title'].removeprefix(lat)
    f1=font('EditorialSerif-Bold.otf',76);f2=font('ChineseSerif-Bold.ttc',68)
    widths=[draw.textlength(lat,font=f1)/S,draw.textlength(zh,font=f2)/S]
    x=(w-sum(widths))/2
    draw.text(pxy((x,209)),lat,font=f1,fill=INK,anchor='ls')
    draw.text(pxy((x+widths[0],209)),zh,font=f2,fill=INK,anchor='ls')
    text(draw,(900,270),data['series'][0]['label'],size=29,face='EditorialSerif.otf',anchor='ms')
    text(draw,(900,308),'Seven independent categories  /  Percent (%)  /  Shared 0–50 scale',size=20,fill=GRAY,anchor='ms')

    # The generated photographic hand is placed below the fine string paths.
    hand=Image.open(ROOT/'assets/hand.png').convert('RGBA')
    hand=hand.resize((round(hand.width*HAND_SCALE*S),round(hand.height*HAND_SCALE*S)),Image.Resampling.LANCZOS)
    canvas.alpha_composite(hand,pxy((HAND_X,HAND_Y)))
    draw=ImageDraw.Draw(canvas)
    starts=[(cx,cy+RMAX+87) for cx,cy in CENTERS]
    routes=[
      [((510,962),(579,989),(590,1110)),((602,1300),(621,1337),(721,1440)),
       ((780,1620),(864,1690),(973,1693)),((1080,1704),(1245,1680),PINCH)],
      [((1280,1000),(1236,1180),(1310,1310)),((1335,1420),(1317,1590),PINCH)],
      [((307,1460),(280,1570),(331,1700)),((340,1790),(365,1810),(400,1815)),
       ((620,1850),(1080,1845),(1175,1798)),((1260,1763),(1330,1695),PINCH)],
      [((884,1360),(780,1324),(788,1451)),((797,1650),(904,1702),(997,1704)),
       ((1118,1707),(1295,1680),PINCH)],
      [((1480,1420),(1444,1550),PINCH)],
      [((700,1825),(1070,1810),(1185,1775)),((1270,1740),(1330,1686),PINCH)],
      [((1110,1810),(1290,1730),PINCH)]
    ]
    for (cx,cy),radius,start,route in zip(CENTERS,radii,starts,routes):
        line(draw,[(cx,cy+radius),(cx,cy+RMAX+29)],fill=(72,69,63,220),width=1.3)
        curved_line(draw,start,route,width=1.65)

    # The foreground thumb covers the last string pixels at the pinch contact.
    raw_hand=Image.open(ROOT/'assets/hand.png').convert('RGBA')
    thumb_mask=Image.new('L',raw_hand.size,0)
    ImageDraw.Draw(thumb_mask).polygon([(836,497),(934,336),(1068,197),(1142,133),
                                      (1200,133),(1240,185),(1180,321),(1120,430),
                                      (1020,553),(890,604)],fill=255)
    thumb_alpha=np.minimum(np.asarray(raw_hand.getchannel('A')),np.asarray(thumb_mask))
    raw_hand.putalpha(Image.fromarray(thumb_alpha))
    raw_hand=raw_hand.resize(hand.size,Image.Resampling.LANCZOS)
    canvas.alpha_composite(raw_hand,pxy((HAND_X,HAND_Y)))

    # Every group uses one common radius mapping and the same concentric ticks.
    records=[]
    for i,(cat,v,center,radius) in enumerate(zip(cats,values,CENTERS,radii),1):
        cx,cy=center;draw=ImageDraw.Draw(canvas)
        for tick in range(5,51,5):
            r=R0+(RMAX-R0)*tick/50
            line(draw,circle_points(cx,cy,r),fill=(61,59,52,235),width=1.25 if tick<50 else 1.6)
        # The zero core is completely occluded for every positive supplied value.
        line(draw,circle_points(cx,cy,R0),fill=(105,101,91,210),width=1)
        rec=disk(canvas,center,radius,f'assets/paint_{i:02}.png',i)
        rec.update(category=cat,value=v,label=f'{v:g}%',order=i)
        records.append(rec)
        arc_text(canvas,center,RMAX+24,f'{i:02}  {cat}',size=33)
        draw=ImageDraw.Draw(canvas)
        top=cy+RMAX+29
        draw.rectangle(pxy((cx-63,top,cx+63,top+48)),fill=INK)
        tail=cubic([(cx-15,top+48),(cx-2,top+48),(cx-1,top+54),(cx,top+58)]).tolist()
        tail+=cubic([(cx,top+58),(cx+1,top+52),(cx+4,top+48),(cx+15,top+48)]).tolist()
        draw.polygon([pxy(p) for p in tail],fill=INK)
        text(draw,(cx,top+35),f'{v:g}%',size=31,fill=(250,249,245,255),anchor='ms')

    # The quarter-circle key has precisely the same r0, R and tick spacing.
    draw=ImageDraw.Draw(canvas)
    lx,ly=211,2000
    for tick in range(0,51,5):
        r=R0+(RMAX-R0)*tick/50
        line(draw,circle_points(lx,ly,r,-90,0),fill=(68,65,59,235),width=1.2 if tick<50 else 1.6)
        if tick%10==0:
            line(draw,[(lx-7,ly-r),(lx-1,ly-r)],width=1)
            text(draw,(lx-16,ly-r+6),f'{tick}%',size=19,anchor='rs')
    line(draw,[(lx,ly+10),(lx+R0,ly+10)],fill=GRAY,width=1)
    line(draw,[(lx,ly+6),(lx,ly+14)],fill=GRAY,width=1)
    line(draw,[(lx+R0,ly+6),(lx+R0,ly+14)],fill=GRAY,width=1)
    text(draw,(lx+R0/2,ly+38),'zero core',size=16,fill=GRAY,anchor='ms')
    tracking(draw,(447,1854),'HOW TO READ',size=16,gap=2.2)
    text(draw,(447,1895),'A shared radial scale',size=30,face='EditorialSerif-Bold.otf')
    text(draw,(447,1933),'Fixed zero core + a linear 0–50% radial increment.',size=21)
    text(draw,(447,1966),'Each ring adds 5 percentage points. Area carries no value.',size=21)
    text(draw,(447,1999),'Independent shares; they need not sum to 100%.',size=21)

    line(draw,[(105,2080),(790,2080)],fill=(117,113,103,170),width=.75)
    tracking(draw,(106,2111),'LOCAL SYNTHETIC STUDY',size=15,gap=1.7,fill=GRAY)
    text(draw,(106,2137),'Source: supplied data.json  ·  Values shown exactly as supplied',size=16,fill=GRAY)
    text(draw,(1692,2128),'07 CATEGORIES  /  ONE SHARED SCALE',size=15,fill=GRAY,anchor='rs')
    out=canvas.convert('RGB').resize((w,h),Image.Resampling.LANCZOS)
    out.save(ROOT/'final.png',dpi=(180,180))
    (ROOT/'artifacts').mkdir(exist_ok=True)
    mapping={'canvas':[w,h],'zero_core_radius_px':R0,'full_scale_radius_px':RMAX,
             'value_domain':[0,50],'tick_values':list(range(0,51,5)),
             'legend':{'center':[lx,ly],'zero_core_radius_px':R0,'full_scale_radius_px':RMAX},
             'strings':{'count':len(routes),'pinch_point':list(PINCH)},'objects':records}
    (ROOT/'artifacts/render_geometry.json').write_text(json.dumps(mapping,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Rendered {w}x{h}; 7 generated paint spheres, 7 strings and 1 generated grayscale hand.')


if __name__=='__main__':
    main()
