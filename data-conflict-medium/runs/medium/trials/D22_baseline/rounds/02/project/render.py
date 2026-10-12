"""Deterministic poster renderer. Run from this case: PYTHON render.py.
No network or model calls. Numeric geometry and all text come from data.json.
"""
from pathlib import Path
import json
import math
import hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.ndimage import map_coordinates, gaussian_filter

ROOT=Path(__file__).resolve().parent
ART=ROOT/'artifacts'
ART.mkdir(exist_ok=True)
DATA=json.loads((ROOT/'data.json').read_text(encoding='utf-8'))
BRIEF=json.loads((ROOT/'brief.json').read_text(encoding='utf-8'))
W,H=BRIEF['width'],BRIEF['height']
S=2
INK=(22,23,23,255)
MUTED=(87,88,84,255)
R0,R=62.0,180.0
VMAX=float(DATA['value_axis']['maximum'])
assert DATA['value_axis']['minimum']==0 and VMAX==50
CATEGORIES=DATA['categories']
VALUES=DATA['series'][0]['values']
assert len(CATEGORIES)==len(VALUES)==7
CENTERS=[(500,565),(1270,650),(890,860),(335,1100),(1470,1115),(980,1430),(520,1690)]

def radius(v):
    return R0+(R-R0)*float(v)/VMAX

def font(name,size):
    return ImageFont.truetype(str(ROOT/'assets/fonts'/name),round(size*S))

# A restrained paper texture at the actual output resolution.
rng=np.random.default_rng(20261012)
noise=rng.normal(0,1.05,(H,W))
fiber=gaussian_filter(rng.normal(0,1,(H,W)),2)*2.3
base=np.stack([246+noise+fiber,245+noise+fiber,240+noise+fiber],axis=2)
paper=Image.fromarray(np.uint8(np.clip(base,0,255)),'RGB').resize((W*S,H*S),Image.Resampling.BICUBIC).convert('RGBA')
im=paper
D=ImageDraw.Draw(im)
# Fine paper flecks, deliberately limited to low contrast.
for _ in range(4800):
    x,y=rng.uniform(0,W),rng.uniform(0,H)
    rr=rng.choice([.30,.45,.65],p=[.62,.29,.09])
    col=tuple(int(v) for v in rng.choice([[213,212,207,255],[224,223,217,255],[235,234,228,255]]))
    D.ellipse(((x-rr)*S,(y-rr)*S,(x+rr)*S,(y+rr)*S),fill=col)
# Sparse pigment dust confined to the outer paper margins.
for side,color,ylo,yhi,count in [('left',(106,135,195,255),350,1500,130),('right',(228,193,86,255),20,650,140),('right',(229,157,113,255),1740,2120,75)]:
    for _ in range(count):
        edge=rng.exponential(13)
        if edge>46:continue
        x=edge if side=='left' else W-edge
        y=rng.uniform(ylo,yhi);rr=rng.uniform(.45,1.8)
        D.ellipse(((x-rr)*S,(y-rr)*S,(x+rr)*S,(y+rr)*S),fill=color)

TEXT_RECORDS=[]
def text(x,y,t,name='sans.otf',size=24,color=INK,anchor='la'):
    D.text((round(x*S),round(y*S)),t,font=font(name,size),fill=color,anchor=anchor)
    TEXT_RECORDS.append({'text':t,'position':[x,y],'size_px':size})

def tracking_center(t,cx,y,size=20,spacing=3.0,color=INK):
    f=font('sans_bold.otf',size)
    ws=[f.getlength(c)/S for c in t]
    total=sum(ws)+spacing*(len(t)-1)
    x=cx-total/2
    for c,w in zip(t,ws):
        D.text((x*S,y*S),c,font=f,fill=color,anchor='la');x+=w+spacing
    TEXT_RECORDS.append({'text':t,'position':[cx,y],'size_px':size})

tracking_center(DATA['visible_notice'],W/2,67,size=19,spacing=4.0)
text(W/2,125,DATA['title'],'title.ttc',72,anchor='ma')
text(W/2,238,DATA['series'][0]['label'],'serif.otf',35,anchor='ma')
# The header states the common unit while leaving the actual scale at the hand.
text(W/2,289,'Seven independent shares · Percent (%)','sans.otf',22,MUTED,'ma')

# Seven independent strings; each path starts below its own black numeric tag.
hand_x,hand_y,hand_w=780.0,1785.0,650.0
hand_src=Image.open(ROOT/'assets/processed/hand.png').convert('RGBA')
hand_scale=hand_w/hand_src.width
hand_h=hand_src.height*hand_scale
CONTACT=(hand_x+1390*hand_scale,hand_y+68*hand_scale)
AX,AY=CONTACT
BADGE_W,BADGE_H=126,48
badge_tops=[cy+R+22 for cx,cy in CENTERS]
starts=[(cx,y+BADGE_H+10) for (cx,cy),y in zip(CENTERS,badge_tops)]
segments=[
 [((500,1040),(725,1060),(651,1300)),((595,1570),(826,1760),(1120,1745)),((1260,1738),(1340,1780),CONTACT)],
 [((1270,1085),(1165,1160),(1190,1400)),((1200,1605),(1275,1725),CONTACT)],
 [((890,1170),(1145,1120),(1220,1270)),((1330,1460),(1230,1695),CONTACT)],
 [((335,1490),(733,1380),(815,1660)),((905,1825),(1265,1737),CONTACT)],
 [((1470,1430),(1320,1470),(1335,1610)),((1350,1700),(1375,1770),CONTACT)],
 [((980,1765),(1235,1675),CONTACT)],
 [((520,2120),(886,2020),(1040,1895)),((1190,1790),(1270,1789),CONTACT)]
]

def cubic(p0,p1,p2,p3,n=180):
    t=np.linspace(0,1,n)[:,None]
    arr=(1-t)**3*np.array(p0)+3*(1-t)**2*t*np.array(p1)+3*(1-t)*t*t*np.array(p2)+t**3*np.array(p3)
    return [tuple(p) for p in arr]

def path_points(start,segs):
    points=[start];last=start
    for a,b,c in segs:
        points.extend(cubic(last,a,b,c)[1:]);last=c
    return points

PATHS=[]
for start,segs in zip(starts,segments):
    points=path_points(start,segs);PATHS.append(points)
    D.line([(round(x*S),round(y*S)) for x,y in points],fill=(44,44,42,255),width=3,joint='curve')

# Concentric circles: a 0% zero-core ring, then 5 pp increments through 50%.
for cx,cy in CENTERS:
    for tick in range(0,51,5):
        rr=radius(tick)
        color=(64,65,61,255) if tick in (0,50) else (102,103,97,255)
        D.ellipse(((cx-rr)*S,(cy-rr)*S,(cx+rr)*S,(cy+rr)*S),outline=color,width=3 if tick in (0,50) else 2)

GEOMETRY=[]
for i,((cx,cy),v) in enumerate(zip(CENTERS,VALUES)):
    rr=radius(v)
    # Local blurred shadow, offset from the material edge; not part of encoding.
    pad=math.ceil(rr+45)
    shadow=Image.new('RGBA',(pad*2*S,pad*2*S),(0,0,0,0))
    sd=ImageDraw.Draw(shadow)
    sd.ellipse(((pad-rr+9)*S,(pad-rr+14)*S,(pad+rr+9)*S,(pad+rr+14)*S),fill=(18,21,27,55))
    shadow=shadow.filter(ImageFilter.GaussianBlur(7*S))
    im.alpha_composite(shadow,(round((cx-pad)*S),round((cy-pad)*S)))
    n=math.ceil(2*rr*S)+6
    left=round(cx*S-n/2);top=round(cy*S-n/2)
    yy,xx=np.mgrid[:n,:n]
    dx=(left+xx+.5-cx*S)/S;dy=(top+yy+.5-cy*S)/S
    dist=np.hypot(dx,dy)
    # This single antialiased mask defines the entire visible data-bearing boundary.
    mask=np.uint8(np.clip((rr-dist)*S+.5,0,1)*255)
    tex=np.array(Image.open(ROOT/f'assets/processed/sphere_{i+1:02d}.png').convert('RGB'))
    tx=(dx/rr+1)*.5*(tex.shape[1]-1)
    ty=(dy/rr+1)*.5*(tex.shape[0]-1)
    rgb=np.stack([map_coordinates(tex[:,:,k],[ty,tx],order=1,mode='nearest') for k in range(3)],axis=-1)
    layer=Image.fromarray(np.dstack((rgb,mask)).astype('uint8'),'RGBA')
    im.alpha_composite(layer,(left,top))
    # Raster measurements of the mask actually applied, independently sampled in 720 directions.
    angles=np.linspace(0,2*np.pi,720,endpoint=False)
    radial_samples=np.arange(max(0,rr-4),rr+4,.025)
    sx=(cx*S-left-.5)+np.cos(angles)[:,None]*radial_samples*S
    sy=(cy*S-top-.5)+np.sin(angles)[:,None]*radial_samples*S
    samples=map_coordinates(mask.astype(float),[sy,sx],order=1,mode='constant',cval=0)
    measured=[]
    for ray in samples:
        ok=np.flatnonzero(ray>=127.5)
        measured.append(float(radial_samples[ok[-1]]) if len(ok) else None)
    radii=np.array(measured)
    mask_path=f'artifacts/ball_mask_{i+1:02d}.png'
    Image.fromarray(mask).save(ROOT/mask_path)
    GEOMETRY.append({'category':CATEGORIES[i],'value_percent':v,'center_px':[cx,cy],
        'radius_target_px':rr,'raster_edge_radius_mean_px':float(radii.mean()),
        'raster_edge_radius_min_px':float(radii.min()),'raster_edge_radius_max_px':float(radii.max()),
        'decoded_percent_from_mean_raster_radius':float((radii.mean()-R0)/(R-R0)*VMAX),
        'max_edge_error_px':float(np.max(abs(radii-rr))),'mask':mask_path,
        'alpha_area_px2':float(mask.sum()/255/S/S)})
D=ImageDraw.Draw(im)

# Native per-glyph placement follows each shared circular arc; category strings are exact.
def arc_text(t,cx,cy,rr=205,size=36):
    f=font('serif.otf',size)
    # Pair-kerning-aware glyph advances at output resolution.
    widths=[]
    for j,c in enumerate(t):
        widths.append((f.getlength(t[:j+1])-f.getlength(t[:j]))/S)
    total=sum(widths);advance=0
    tile_size=round(110*S)
    for c,ww in zip(t,widths):
        theta=-np.pi/2+(advance+ww/2-total/2)/rr
        x=cx+rr*np.cos(theta);y=cy+rr*np.sin(theta)
        if c!=' ':
            tile=Image.new('RGBA',(tile_size,tile_size),(0,0,0,0))
            td=ImageDraw.Draw(tile)
            td.text((tile_size/2,tile_size/2),c,font=f,fill=INK,anchor='ms')
            tile=tile.rotate(-math.degrees(theta+np.pi/2),Image.Resampling.BICUBIC,center=(tile_size/2,tile_size/2))
            im.alpha_composite(tile,(round(x*S-tile_size/2),round(y*S-tile_size/2)))
        advance+=ww
    TEXT_RECORDS.append({'text':t,'arc_center':[cx,cy],'arc_radius_px':rr,'size_px':size})

for i,((cx,cy),category,value) in enumerate(zip(CENTERS,CATEGORIES,VALUES)):
    arc_text(category,cx,cy)
    D=ImageDraw.Draw(im)
    y=badge_tops[i]
    D.rectangle(((cx-BADGE_W/2)*S,y*S,(cx+BADGE_W/2)*S,(y+BADGE_H)*S),fill=INK)
    # Small narrowing attachment into the string.
    D.polygon([((cx-12)*S,(y+BADGE_H)*S),((cx+12)*S,(y+BADGE_H)*S),(cx*S,(y+BADGE_H+12)*S)],fill=INK)
    label=f'{value:g}%'
    text(cx,y+BADGE_H/2+1,label,'sans.otf',33,(255,255,252,255),'mm')

# A same-size quarter-circle legend: same r0, R and function as every sphere.
LX,LY=1475.0,1695.0
text(1570,1470,'RADIAL SCALE','sans_bold.otf',21,INK,'ma')
D.pieslice(((LX-R0)*S,(LY-R0)*S,(LX+R0)*S,(LY+R0)*S),270,360,fill=(227,226,218,255))
for tick in range(0,51,5):
    rr=radius(tick)
    D.arc(((LX-rr)*S,(LY-rr)*S,(LX+rr)*S,(LY+rr)*S),270,360,fill=(65,66,61,255),width=3 if tick in (0,50) else 2)
    if tick%10==0:
        D.line(((LX-9)*S,(LY-rr)*S,LX*S,(LY-rr)*S),fill=INK,width=2)
        text(LX-15,LY-rr,f'{tick}%','sans.otf',19,INK,'rm')
text(1564,1725,'Each ring: +5 pp','sans.otf',19,MUTED,'ma')
text(1564,1755,'Fixed inner core: 0%','sans.otf',19,MUTED,'ma')

# The photographic hand, then terminal string sections, then the front thumb.
hand=hand_src.resize((round(hand_w*S),round(hand_h*S)),Image.Resampling.LANCZOS)
im.alpha_composite(hand,(round(hand_x*S),round(hand_y*S)))
D=ImageDraw.Draw(im)
for pts in PATHS:
    # Continue every strand across the rear index finger to the pinch without gaps.
    # The foreground thumb then provides the final anatomical occlusion.
    terminal=[p for p in pts if math.hypot(p[0]-AX,p[1]-AY)<120]
    if len(terminal)>1:
        D.line([(round(x*S),round(y*S)) for x,y in terminal],fill=(35,35,34,255),width=3,joint='curve')
thumb_mask=Image.new('L',hand_src.size,0)
tm=ImageDraw.Draw(thumb_mask)
tm.polygon([(1075,195),(1160,131),(1250,98),(1325,79),(1380,63),(1414,68),(1434,90),(1411,142),(1335,185),(1230,240),(1100,280)],fill=255)
thumb_mask=np.minimum(np.array(thumb_mask),np.array(hand_src.getchannel('A')))
thumb=hand_src.copy();thumb.putalpha(Image.fromarray(thumb_mask))
thumb=thumb.resize(hand.size,Image.Resampling.LANCZOS)
im.alpha_composite(thumb,(round(hand_x*S),round(hand_y*S)))
D=ImageDraw.Draw(im)

# Short reading note and source, kept away from the hand and the numerical tags.
text(92,1986,'READ THE RADIUS','sans_bold.otf',19,INK)
text(92,2020,'Radius increment represents the share.','sans.otf',22,MUTED)
text(92,2051,'The fixed core and sphere area do not encode values.','sans.otf',22,MUTED)
D.line((92*S,2100*S,1708*S,2100*S),fill=(192,192,184,255),width=2)
text(92,2120,'SOURCE: data.json · Synthetic values','sans.otf',17,MUTED)
text(1708,2120,'Independent multiple-choice shares; totals need not equal 100%.','sans.otf',17,MUTED,'ra')

final=im.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
final.save(ROOT/'final.png',compress_level=6)
# Geometry evidence is derived from applied masks, not an independent quality score.
(ART/'geometry_check.json').write_text(json.dumps({'canvas':[W,H],'scale':{'zero_core_radius_px':R0,'maximum_radius_px':R,'maximum_percent':VMAX,'ring_step_pp':5,'formula':'r0 + (R-r0) * value / 50','area_encodes_data':False},'objects':GEOMETRY,'legend':{'center_px':[LX,LY],'radii_px':[radius(v) for v in range(0,51,5)],'same_function_as_objects':True},'strings':{'count':len(PATHS),'contact_px':list(CONTACT),'layering':'strings / hand / terminal strands / foreground thumb'},'rendered_text':TEXT_RECORDS},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'output':'final.png','size':[W,H],'objects':len(GEOMETRY),'max_mask_edge_error_px':max(v['max_edge_error_px'] for v in GEOMETRY)},ensure_ascii=False))
