"""Offline renderer. All quantitative geometry and all typography are code-owned.
Run through capture_render.py to preserve complete rendering attempts.
"""
from pathlib import Path
import json, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent
S = 2
INK = (24, 25, 24)
MUTED = (90, 90, 85)
R0, RMAX = 32.0, 180.0
RING_STEP = 5
CENTERS = [(520,610), (1280,610), (300,1060), (900,1010),
           (1500,1060), (565,1510), (1185,1510)]
PINCH = (1170,1852)
FONT_FILES = {'serif':'serif.otf', 'bold':'serif-bold.otf',
              'sans':'sans.otf', 'sansbold':'sans-bold.otf'}
FONT_CACHE = {}

def font(size, face='sans'):
    key = (size, face)
    if key not in FONT_CACHE:
        FONT_CACHE[key] = ImageFont.truetype(str(ROOT/'assets/fonts'/FONT_FILES[face]), round(size*S))
    return FONT_CACHE[key]

def coords(box):
    return tuple(round(x*S) for x in box)

def write(im, xy, text, size=28, face='sans', color=INK, anchor='la'):
    ImageDraw.Draw(im).text((xy[0]*S,xy[1]*S), text, font=font(size,face), fill=color, anchor=anchor)

def curved_text(im, text, center, radius, size=39):
    f = font(size, 'serif')
    tracking = .1*S
    total = f.getlength(text) + tracking*(len(text)-1)
    for k,ch in enumerate(text):
        before = f.getlength(text[:k]) + k*tracking
        charw = f.getlength(ch)
        theta = (before + charw/2 - total/2)/(radius*S)
        px = (center[0] + radius*math.sin(theta))*S
        py = (center[1] - radius*math.cos(theta))*S
        side = math.ceil(size*S*4)
        glyph = Image.new('RGBA',(side,side))
        d=ImageDraw.Draw(glyph)
        d.text((side/2,side/2),ch,font=f,fill=INK+(255,),anchor='ms')
        glyph=glyph.rotate(-math.degrees(theta),resample=Image.Resampling.BICUBIC)
        im.alpha_composite(glyph,(round(px-side/2),round(py-side/2)))

def bezier_points(start, segments, samples=90):
    out=[start]
    p0=np.array(start,dtype=float)
    for c1,c2,end in segments:
        c1,c2,p3=map(lambda p:np.array(p,dtype=float),(c1,c2,end))
        for t in np.linspace(0,1,samples)[1:]:
            p=(1-t)**3*p0+3*(1-t)**2*t*c1+3*(1-t)*t*t*c2+t**3*p3
            out.append(tuple(p))
        p0=p3
    return out

def line_path(im,start,segments,color=INK,width=1.8):
    p=bezier_points(start,segments)
    ImageDraw.Draw(im).line([(round(x*S),round(y*S)) for x,y in p],fill=color,width=round(width*S),joint='curve')
    return p

def paper(w,h):
    rng=np.random.default_rng(20261014)
    grain=rng.normal(0,.85,(h,w)).astype(np.float32)
    low=rng.normal(0,.65,(max(1,h//40),max(1,w//40)))
    lowim=Image.fromarray(np.uint8(np.clip(low*35+128,0,255))).resize((w,h),Image.Resampling.BICUBIC)
    grain+=(np.asarray(lowim).astype(np.float32)-128)/40
    base=np.array([248,247,242],dtype=np.float32)[None,None,:]+grain[:,:,None]
    im=Image.fromarray(np.uint8(np.clip(base,0,255))).resize((w*S,h*S),Image.Resampling.BICUBIC).convert('RGBA')
    speck=Image.new('RGBA',im.size)
    d=ImageDraw.Draw(speck)
    for _ in range(9000):
        x,y=rng.uniform(0,w),rng.uniform(0,h)
        radius=rng.choice([.25,.38,.6,.9],p=[.45,.3,.2,.05])
        d.ellipse(coords((x-radius,y-radius,x+radius,y+radius)),fill=(55,54,48,int(rng.integers(9,28))))
    # Delicate pigment dust at the margins, leaving all labels and marks clear.
    for side,color in [('left',(38,77,197)),('right',(231,128,35))]:
        for _ in range(1550):
            inset=float(rng.exponential(13))
            x=inset if side=='left' else w-inset
            y=float(rng.uniform(350,h-60))
            radius=float(rng.uniform(.35,1.8))
            d.ellipse(coords((x-radius,y-radius,x+radius,y+radius)),fill=color+(int(rng.integers(12,65)),))
    im.alpha_composite(speck)
    return im

def radius(value,axismax):
    return R0+(RMAX-R0)*value/axismax

def paint_ball(im, center, radius_px, asset_path):
    cx,cy=center
    r=radius_px*S
    cxh,cyh=cx*S,cy*S
    left=math.floor(cxh-r-2);top=math.floor(cyh-r-2)
    right=math.ceil(cxh+r+2);bottom=math.ceil(cyh+r+2)
    ww,hh=right-left,bottom-top
    xx=(np.arange(ww)+left+.5-cxh)/r
    yy=(np.arange(hh)+top+.5-cyh)/r
    dist=np.sqrt(xx[None,:]**2+yy[:,None]**2)*r
    coverage=np.uint8(np.clip((r-dist+.5),0,1)*255)
    mask=Image.fromarray(coverage)
    tex=Image.open(ROOT/asset_path).convert('RGB')
    # The same affine map binds normalized material coordinates to each data radius.
    sourcebox=((left-cxh+r)/(2*r)*tex.width,(top-cyh+r)/(2*r)*tex.height,
               (right-cxh+r)/(2*r)*tex.width,(bottom-cyh+r)/(2*r)*tex.height)
    tex=tex.transform((ww,hh),Image.Transform.EXTENT,sourcebox,Image.Resampling.BICUBIC).convert('RGBA')
    tex.putalpha(mask)
    shadow=Image.new('RGBA',(ww+80*S,hh+80*S))
    sm=Image.new('L',shadow.size)
    sm.paste(mask,(40*S+7*S,40*S+10*S))
    sm=sm.filter(ImageFilter.GaussianBlur(7*S)).point(lambda a:round(a*.20))
    shadow.putalpha(sm)
    im.alpha_composite(shadow,(left-40*S,top-40*S))
    im.alpha_composite(tex,(left,top))
    return {'alpha_bbox_high_resolution':list(mask.getbbox()),'tile_origin_high_resolution':[left,top],
            'mask_area_high_resolution':float(np.asarray(mask).sum()/255)}

def tag(im,cx,y,label):
    d=ImageDraw.Draw(im)
    d.rectangle(coords((cx-73,y,cx+73,y+51)),fill=INK)
    # The tapered black label flows into its thread, matching the visual metaphor.
    pp=[(cx-17,y+51)]
    pp+=bezier_points(pp[0],[((cx-5,y+51),(cx-1,y+57),(cx,y+69)),
                           ((cx+1,y+57),(cx+5,y+51),(cx+17,y+51))],35)[1:]
    d.polygon([(round(x*S),round(y*S)) for x,y in pp],fill=INK)
    write(im,(cx,y+25.5),label,36,'sans',(255,255,251),'mm')


def main():
    # Load and retain all frozen inputs. The reference is never sampled as artwork.
    config=json.loads((ROOT/'input.json').read_text(encoding='utf-8'))
    brief=json.loads((ROOT/config['brief']).read_text(encoding='utf-8'))
    data=json.loads((ROOT/config['data']).read_text(encoding='utf-8'))
    core=json.loads((ROOT/config['critical_metaphor_contract']).read_text(encoding='utf-8'))
    assert (ROOT/config['reference']).is_file()
    assert core['visible_title']==data['title']
    w,h=brief['width'],brief['height']
    values=data['series'][0]['values'];cats=data['categories']
    assert len(cats)==len(values)==len(CENTERS)==7
    assert data['value_axis']=={'minimum':0,'maximum':50,'scale':'linear'}
    assert sum(values)==data['series_totals']['synthetic_response_share']
    maxv=data['value_axis']['maximum']
    im=paper(w,h)
    d=ImageDraw.Draw(im)
    # All mandatory text comes from data or explicitly describes its encoding.
    d.rounded_rectangle(coords((729,68,1071,113)),radius=3*S,fill=INK)
    write(im,(900,91),data['visible_notice'],24,'sansbold',(255,255,251),'mm')
    headline=data['title'].removeprefix('SYNTHETIC ')
    write(im,(900,193),headline,111,'bold',INK,'mm')
    write(im,(900,280),data['series'][0]['label']+' (%)',33,'sans',INK,'mm')
    write(im,(900,325),'Independent multiple selections; percentages need not total 100%.',26,'sans',MUTED,'mm')
    # Seven threads, drawn below chart objects and the photographic hand.
    starts=[(cx,cy+RMAX+97) for cx,cy in CENTERS]
    threads=[
       [((520,1020),(610,1100),(635,1170)),((650,1240),(790,1260),(800,1380)),((835,1650),(1010,1680),(1085,1790)),((1160,1785),(1210,1810),PINCH)],
       [((1280,1010),(1160,1120),(1210,1195)),((1240,1240),(1390,1220),(1425,1335)),((1450,1500),(1410,1660),(1320,1740)),((1260,1800),(1200,1815),PINCH)],
       [((300,1460),(300,1670),(390,1780)),((530,1930),(985,1760),(1110,1780)),((1160,1780),(1205,1820),PINCH)],
       [((900,1460),(920,1660),(1090,1770)),((1130,1790),(1218,1805),PINCH)],
       [((1500,1500),(1510,1680),(1350,1760)),((1280,1800),(1200,1818),PINCH)],
       [((565,1920),(985,1740),(1130,1790)),((1175,1795),(1200,1817),PINCH)],
       [((1208,1800),(1200,1815),PINCH)]
    ]
    thread_geometry=[]
    for start,segments in zip(starts,threads):
        line_path(im,start,segments)
        thread_geometry.append({'start':start,'segments':segments,'end':PINCH})
    rings=[radius(v,maxv) for v in range(0,maxv+1,RING_STEP)]
    geometry=[]
    for i,(category,value,center) in enumerate(zip(cats,values,CENTERS)):
        cx,cy=center
        for k,rr in enumerate(rings):
            color=(52,53,50) if k==len(rings)-1 else (80,81,76)
            ImageDraw.Draw(im).ellipse(coords((cx-rr,cy-rr,cx+rr,cy+rr)),outline=color,width=round((1.55 if k==len(rings)-1 else 1.2)*S))
        rr=radius(value,maxv)
        measured=paint_ball(im,center,rr,f'assets/sphere_{i+1:02d}.png')
        curved_text(im,category,center,RMAX+22,39)
        tag(im,cx,cy+RMAX+28,f'{value:g}%')
        geometry.append({'category':category,'value_percent':value,'center':center,'radius_px':rr,
                         'circle_bbox':[cx-rr,cy-rr,cx+rr,cy+rr],
                         'tag_bbox':[cx-73,cy+RMAX+28,cx+73,cy+RMAX+79],**measured})
    # Photographic hand. Anchor its inspected thumb/index contact to all seven lines.
    hand=Image.open(ROOT/'assets/hand.png').convert('RGBA')
    hand_scale=.48
    hand_angle=-20
    oldw,oldh=hand.size
    hand=hand.rotate(hand_angle,resample=Image.Resampling.BICUBIC,expand=True)
    c=math.cos(math.radians(hand_angle));s=math.sin(math.radians(hand_angle))
    dx,dy=1108-oldw/2,231-oldh/2
    contact=(c*dx+s*dy+hand.width/2,-s*dx+c*dy+hand.height/2)
    hand=hand.resize((round(hand.width*hand_scale*S),round(hand.height*hand_scale*S)),Image.Resampling.LANCZOS)
    origin=(PINCH[0]-contact[0]*hand_scale,PINCH[1]-contact[1]*hand_scale)
    im.alpha_composite(hand,(round(origin[0]*S),round(origin[1]*S)))
    # A short visible tie ends exactly on the contact seam; it has no data meaning.
    ImageDraw.Draw(im).line(coords((PINCH[0]+2.5,PINCH[1]-2.3,*PINCH)),fill=INK,width=3)
    # The legend is a physical copy of the marks' radial scale, with no resizing.
    lx,ly=1480,1975
    ImageDraw.Draw(im).pieslice(coords((lx-R0,ly-R0,lx+R0,ly+R0)),270,360,fill=(224,222,214))
    for v,rr in zip(range(0,maxv+1,RING_STEP),rings):
        ImageDraw.Draw(im).arc(coords((lx-rr,ly-rr,lx+rr,ly+rr)),270,360,fill=INK,width=round(1.2*S))
        if v%10==0:
            ImageDraw.Draw(im).line(coords((lx-9,ly-rr,lx-2,ly-rr)),fill=MUTED,width=2)
            write(im,(lx-15,ly-rr),f'{v}%',21,'sans',INK,'rm')
    write(im,(1550,1740),'ONE RADIAL SCALE',21,'sansbold',INK,'mm')
    write(im,(1550,2011),'0–50% for every sphere',23,'sans',INK,'mm')
    write(im,(100,1950),'HOW TO READ',21,'sansbold',INK)
    write(im,(100,1991),'Each ring adds 5 percentage points.',27,'sans',INK)
    write(im,(100,2028),'The fixed inner core represents 0%.',27,'sans',INK)
    write(im,(100,2065),'Radius carries the value; area does not.',27,'sans',INK)
    # Small provenance line retains the synthetic context without claiming a survey.
    write(im,(100,2121),f"{len(cats)} synthetic categories • Shares total {sum(values):g}%",21,'sans',MUTED)
    final=im.convert('RGB').resize((w,h),Image.Resampling.LANCZOS)
    final.save(ROOT/'final.png')
    (ROOT/'artifacts').mkdir(exist_ok=True)
    report={'canvas':[w,h],'render_scale':S,'radius_scale':{'zero_core_px':R0,'maximum_radius_px':RMAX,
             'axis_minimum':0,'axis_maximum':maxv,'ring_step_percentage_points':RING_STEP,'ring_radii_px':rings},
             'balls':geometry,'threads':thread_geometry,'hand':{'source_contact_xy':[1108,231],
             'scale':hand_scale,'rotation_degrees':hand_angle,'rotated_contact_xy':contact,'origin':origin,'target_contact_xy':PINCH},
             'legend':{'center':[lx,ly],'ring_radii_px':rings,'same_absolute_scale_as_marks':True},
             'labels_from_data':{'title':data['title'],'categories':cats,'series':data['series'][0]['label'],
             'value_labels':[f'{v:g}%' for v in values]}}
    (ROOT/'artifacts/render_geometry.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Rendered final.png at',w,h,'with seven generated paint materials and one generated hand.')

if __name__=='__main__':
    main()
