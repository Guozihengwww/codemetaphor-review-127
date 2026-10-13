#!/usr/bin/env python3
"""Offline data-bound diver infographic. Run only through capture_render.py for a complete render."""
from pathlib import Path
import json
import hashlib
import math
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent

def read_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))

def save_json(name, value):
    p = ROOT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

BRIEF = read_json('brief.json')
DATA = read_json('data.json')
CORE = read_json('core.json')
INPUT = read_json('input.json')
assert INPUT['brief'] == 'brief.json' and INPUT['data'] == 'data.json'
assert CORE['visible_title'] == DATA['title']
W, H = BRIEF['width'], BRIEF['height']
assert (W, H) == (2000, 2000)
assert DATA['value_axis']['scale'] == 'linear'
assert len(DATA['series']) == 1
SERIES = DATA['series'][0]
YEARS = np.asarray(DATA['time_points'], dtype=float)
VALUES = np.asarray(SERIES['values'], dtype=float)
assert len(YEARS) == len(VALUES) == len(DATA['categories']) == 10
X0, X1 = 250.0, 1780.0
ZERO_Y, TOP_Y = 1670.0, 480.0
VMIN, VMAX = DATA['value_axis']['minimum'], DATA['value_axis']['maximum']
PX = X0 + (YEARS - YEARS[0]) / (YEARS[-1] - YEARS[0]) * (X1-X0)
PY = ZERO_Y - (VALUES - VMIN) / (VMAX-VMIN) * (ZERO_Y-TOP_Y)
POINTS = np.column_stack((PX, PY))
# Explicit landmarks on the generated figure: foot, calf, knee, thigh,
# hip, torso, shoulder, elbow, wrist, fingers. These do not encode other data.
SRC_X = np.array([157,300,452,610,775,935,1090,1230,1360,1452], dtype=float)
SRC_Y = np.array([728,572,452,313,223,323,432,618,772,886], dtype=float)
BODY_THICKNESS_SCALE = 1.10
S = 2
INK = '#253e48'
MUTED = '#627475'
PAPER = '#f6f0e4'
RUST = '#a64432'


def extended_interp(q, xp, yp):
    q = np.asarray(q)
    result = np.interp(q, xp, yp)
    result = np.where(q < xp[0], yp[0] + (q-xp[0])*(yp[1]-yp[0])/(xp[1]-xp[0]), result)
    result = np.where(q > xp[-1], yp[-1] + (q-xp[-1])*(yp[-1]-yp[-2])/(xp[-1]-xp[-2]), result)
    return result


def build_diver():
    """One x-dependent map is shared by anatomy and all ten data anchors."""
    src = np.asarray(Image.open(ROOT / 'assets/diver_clean.png').convert('RGBA'), dtype=np.float32)
    xx = np.arange(W, dtype=np.float32)
    yy = np.arange(H, dtype=np.float32)
    u = extended_interp(xx, PX, SRC_X)
    body_source_y = extended_interp(u, SRC_X, SRC_Y)
    body_target_y = extended_interp(xx, PX, PY)
    v = body_source_y[None,:] + (yy[:,None] - body_target_y[None,:]) / BODY_THICKNESS_SCALE
    coords = [v, np.broadcast_to(u[None,:], v.shape)]
    alpha = src[:,:,3] / 255.0
    out_a = map_coordinates(alpha, coords, order=1, mode='constant', cval=0, prefilter=False)
    out = np.zeros((H,W,4), dtype=np.uint8)
    good = out_a > 1e-4
    for c in range(3):
        premul = map_coordinates(src[:,:,c] * alpha, coords, order=1, mode='constant', cval=0, prefilter=False)
        color = np.zeros_like(premul)
        color[good] = premul[good] / out_a[good]
        out[:,:,c] = np.clip(color,0,255).astype(np.uint8)
    out[:,:,3] = np.clip(out_a*255,0,255).astype(np.uint8)
    Image.fromarray(out).save(ROOT / 'assets/diver_warped.png')
    return Image.fromarray(out), out[:,:,3]


def paper_image():
    rng = np.random.default_rng(832)
    noise = rng.normal(0,0.76,(H,W)).astype(np.float32)
    cloud = gaussian_filter(rng.normal(0,1,(H//10,W//10)), 4)
    cloud = np.asarray(Image.fromarray(cloud.astype('float32')).resize((W,H),Image.Resampling.BICUBIC))
    grain = noise + cloud * 4
    base = np.zeros((H,W,3),dtype=np.uint8)
    for i,c in enumerate((246,240,228)):
        base[:,:,i] = np.clip(c+grain,0,255).astype('uint8')
    return Image.fromarray(base).resize((W*S,H*S),Image.Resampling.BICUBIC).convert('RGBA')


FONT_DIR = ROOT / 'assets/fonts'
FONT_PATHS = {'serif': FONT_DIR/'NimbusRoman-Regular.otf',
              'italic': FONT_DIR/'NimbusRoman-Italic.otf',
              'sans': FONT_DIR/'NimbusSans-Regular.otf',
              'bold': FONT_DIR/'NimbusSans-Bold.otf'}

def font(size, style='sans'):
    return ImageFont.truetype(str(FONT_PATHS[style]),round(size*S))

canvas = paper_image()
draw = ImageDraw.Draw(canvas)
TEXT_LOG = []

def xy(p): return tuple(float(v)*S for v in p)

def line(coords, fill=INK, width=1):
    draw.line([xy(p) for p in coords],fill=fill,width=max(1,round(width*S)),joint='curve')

def rect(box, fill, outline=None, width=1, radius=0):
    b=xy(box)
    if radius: draw.rounded_rectangle(b,radius*S,fill=fill,outline=outline,width=round(width*S))
    else: draw.rectangle(b,fill=fill,outline=outline,width=round(width*S))

def ellipse(box, fill, outline=None, width=1):
    draw.ellipse(xy(box),fill=fill,outline=outline,width=round(width*S))

def text(s,x,y,size=30,style='sans',fill=INK,align='left',record=True):
    f=font(size,style)
    length=draw.textlength(s,font=f)
    if align=='center': x-=length/(2*S)
    elif align=='right': x-=length/S
    b=draw.textbbox((x*S,y*S),s,font=f,anchor='lt')
    draw.text((x*S,y*S),s,font=f,fill=fill,anchor='lt')
    if record: TEXT_LOG.append({'text':s,'bbox':[round(v/S,2) for v in b]})
    return (b[0]/S,b[1]/S,b[2]/S,b[3]/S)

def tracked_text(s,x,y,size=22,tracking=3,fill=MUTED):
    f=font(size,'bold')
    for c in s:
        draw.text((x*S,y*S),c,font=f,fill=fill,anchor='lt')
        x+=draw.textlength(c,font=f)/S+tracking

# Editorial header.
line([(130,116),(220,116)],RUST,4)
tracked_text('ECONOMIC OUTPUT / A POSE IN DATA',242,103,22,2.0)
rect((1477,85,1868,145), '#ead9c8', radius=5)
text(DATA['visible_notice'],1672.5,103,27,'bold',RUST,'center')
text(DATA['title'],126,190,123,'serif')
text('Economic output per person, traced through a diver’s pose.',132,331,33,'sans',MUTED)
line([(132,399),(1866,399)],'#b8b6aa',1.25)
text(f"{SERIES['label']} ({SERIES['unit']})",165,432,27,'bold')
text('2000–2018',1865,432,27,'sans',MUTED,'right')

# Square graph paper. Four squares equal two years and USD 5,000.
for i in range(41):
    x=165+42.5*i
    major = (i-2)%4==0
    line([(x,480),(x,1840)],'#c4ceca' if major else '#dce0d7',1.05 if major else .65)
for j in range(33):
    y=480+42.5*j
    major=j%4==0
    line([(165,y),(1865,y)],'#bac9c6' if major else '#d7ded4',1.15 if major else .65)

# Water is the shared 0 baseline; decorative waves are below it.
water=np.zeros((170,1701,4),dtype=np.uint8)
rng=np.random.default_rng(1208)
wave_noise=rng.normal(0,1.4,(170,1701))
for k,(top,bottom) in enumerate(zip((63,148,157),(44,124,141))):
    grad=np.linspace(top,bottom,170)[:,None]+wave_noise
    water[:,:,k]=np.clip(grad,0,255).astype('uint8')
water[:,:,3]=255
wi=Image.fromarray(water).resize((1701*S,170*S),Image.Resampling.BICUBIC)
canvas.alpha_composite(wi,(165*S,1670*S))
draw=ImageDraw.Draw(canvas)
for i in range(41):
    x=165+42.5*i
    line([(x,1671),(x,1840)],'#4c929c',.65)
for yy in (1757,1797,1825):
    for j in range(10):
        xx=202+j*176 + (23 if yy==1797 else 0)
        pts=[]
        for t in np.linspace(0,1,21):
            pts.append((xx+62*t,yy+3.5*math.sin(2*math.pi*t)))
        line(pts,'#6eacb0',1.4)
line([(165,ZERO_Y),(1865,ZERO_Y)],INK,2.6)
line([(165,480),(165,ZERO_Y)],'#637676',1.6)
# Vertical value axis uses the exact same linear mapping as the data.
for value in range(int(VMIN),int(VMAX)+1,5000):
    y=ZERO_Y-(value-VMIN)/(VMAX-VMIN)*(ZERO_Y-TOP_Y)
    line([(156,y),(165,y)],INK,1.5)
    text(f'{value:,}',138,y-12,28,'sans',INK,'right')
for category,x in zip(DATA['categories'],PX):
    line([(x,ZERO_Y),(x,ZERO_Y+13)],'#e9eee0',1.75)
    text(category,float(x),1709,33,'bold','#f6f0df','center')
text('Year',1865,1866,27,'bold',MUTED,'right')

# Dominant image-generated metaphor, quantitatively adapted in code.
diver, body_alpha = build_diver()
canvas.alpha_composite(diver.resize((W*S,H*S),Image.Resampling.BICUBIC))
draw=ImageDraw.Draw(canvas)

# Exact straight-line data channel; the figure's outer thickness has no metric meaning.
line(POINTS.tolist(), '#fbf4e4', 10)
line(POINTS.tolist(), INK, 5.4)
for x,y in POINTS:
    ellipse((x-13,y-13,x+13,y+13),'#fbf4e4',INK,2.6)
    ellipse((x-6.2,y-6.2,x+6.2,y+6.2),RUST)

# Value labels use short leaders outside the silhouette. All values are printed in full.
# Anchor placement is non-quantitative; the round data marks retain their true coordinates.
label_specs = [
    (250,1390,'below'),
    (415,935,'above'),
    (585,697,'above'),
    (765,539,'above'),
    (944,482,'above'),
    (1110,570,'above'),
    (1296,730,'above'),
    (1466,1158,'below'),
    (1604,1371,'below'),
    (1793,1556,'below'),
]
LABELS=[]
for i,((x,y),value,spec) in enumerate(zip(POINTS,VALUES,label_specs)):
    lx,ly,side=spec
    label=f'{int(value):,}'
    f=font(37,'bold')
    tw=draw.textlength(label,font=f)/S
    # A small paper patch lifts the label from the graph ruling.
    rect((lx-tw/2-11,ly-8,lx+tw/2+11,ly+36),PAPER,radius=3)
    bbox=text(label,lx,ly,37,'bold',RUST if i==int(np.argmax(VALUES)) else INK,'center')
    end_y=ly+40 if side=='above' else ly-13
    start_y=y-19 if side=='above' else y+19
    line([(x,start_y),(lx,end_y)],'#687a78',1.25)
    LABELS.append({'year':DATA['categories'][i],'value':int(value),'data_point':[float(x),float(y)],'label_bbox':bbox})

# Quiet legend and encoding note inside the arch.
tracked_text('READING THE POSE',625,1247,22,2.7)
line([(625,1319),(699,1319)],'#fbf4e4',10)
line([(625,1319),(699,1319)],INK,5.4)
ellipse((649,1306,675,1332),'#fbf4e4',INK,2.6)
ellipse((656,1313,668,1325),RUST)
text(SERIES['label'],725,1304,33,'bold')
text('Height above the waterline shows',625,1370,29,'sans',MUTED)
text('USD per person. Body thickness',625,1413,29,'sans',MUTED)
text('is illustrative; follow the marked line.',625,1456,29,'sans',MUTED)

# Provenance is explicit and does not claim a national or real-world observation.
line([(132,1904),(1866,1904)],'#b8b6aa',1.25)
text('10 observations · 2000–2018 · Synthetic values, not national statistics.',132,1934,25,'sans',MUTED)
text('GDP / PERSON',1866,1934,22,'bold',MUTED,'right')

final=canvas.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
final.save(ROOT/'final.png',optimize=True)

# Concise machine evidence; verification of the final pixels is performed separately.
segment_inside=[]
for i in range(len(POINTS)-1):
    ts=np.linspace(0,1,301)
    coords=POINTS[i][None,:]*(1-ts[:,None])+POINTS[i+1][None,:]*ts[:,None]
    al=map_coordinates(body_alpha.astype(float),[coords[:,1],coords[:,0]],order=1)
    segment_inside.append({'from':DATA['categories'][i],'to':DATA['categories'][i+1],
                           'samples':len(ts),'alpha_min':round(float(al.min()),2),
                           'fraction_inside_alpha_200':round(float(np.mean(al>=200)),5)})
coverage=[]
for x,y in POINTS:
    coverage.append(round(float(map_coordinates(body_alpha.astype(float),[[y],[x]],order=1)[0]),2))
save_json('artifacts/render_geometry.json',{
    'canvas':[W,H], 'axis':{'x_year_min':float(YEARS[0]),'x_year_max':float(YEARS[-1]),
        'x_min_px':X0,'x_max_px':X1,'value_min':VMIN,'value_max':VMAX,
        'zero_y_px':ZERO_Y,'max_y_px':TOP_Y,'scale':'linear'},
    'data_polyline':LABELS,'body_alpha_at_points':coverage,'body_path_samples':segment_inside,
    'source_landmarks':np.column_stack((SRC_X,SRC_Y)).tolist(),
    'body_thickness_scale_non_data':BODY_THICKNESS_SCALE,'text_bboxes':TEXT_LOG,
    'visible_generated_asset_pixels_alpha_gt_200':int((body_alpha>200).sum()),
    'input_hashes':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                    ['input.json','brief.json','data.json','core.json','reference.png']}})
