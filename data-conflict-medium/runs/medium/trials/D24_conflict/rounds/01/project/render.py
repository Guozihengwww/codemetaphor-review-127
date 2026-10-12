"""Reproduce final.png using only frozen local data and generated local materials.

All top faces use one ellipse mapping (x, y) = (r cos(t), .96 r sin(t)).
The exploded pieces retain that mapping and differ only by translation.
Run complete renders through capture_render.py.
"""
from pathlib import Path
from functools import lru_cache
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops, ImageEnhance

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / 'data.json').read_text(encoding='utf-8'))
BRIEF = json.loads((ROOT / 'brief.json').read_text(encoding='utf-8'))
W, H = BRIEF['width'], BRIEF['height']
S = 3
R, Q = 300, .96
PAD = 38
L = 2*(R+PAD)
ORIGIN = R+PAD
INK = (54, 64, 55)
MUTED = (109, 117, 101)
GOLD = (165, 129, 34)
LINE = (129, 136, 118)
VALUES = DATA['series'][0]['values']
CATS = DATA['categories']
assert len(VALUES) == len(CATS) == 6
assert sum(VALUES) == 100 and DATA['left_capacity'] == DATA['right_capacity'] == 50
assert VALUES[0] == 50
OUT = ROOT / 'artifacts'
(OUT / 'masks').mkdir(parents=True, exist_ok=True)

@lru_cache(None)
def font(kind, size):
    files = {'cjk':'cjk.ttc', 'sans':'sans.otf', 'narrow':'narrow.otf', 'serif':'serif.otf'}
    return ImageFont.truetype(str(ROOT / 'assets/fonts' / files[kind]), round(size*S))

yy, xx = np.mgrid[0:H,0:W]
light = np.exp(-(((xx-W*.43)/(W*.85))**2+((yy-H*.42)/(H*.72))**2))
bg = np.empty((H,W,3), dtype=np.uint8)
for k, base in enumerate((219,222,206)):
    bg[:,:,k] = np.clip(base+light*14,0,255)
canvas = Image.fromarray(bg, 'RGB').resize((W*S,H*S),Image.Resampling.BICUBIC).convert('RGBA')
del bg, yy, xx, light
draw = ImageDraw.Draw(canvas)
text_records = []

def txt(text, x, y, size, kind='sans', color=INK, anchor='lt', track=0):
    global draw
    f = font(kind,size)
    if track:
        length = sum(draw.textlength(c,font=f) for c in text) + max(0,len(text)-1)*track*S
        xpos = x*S - (length/2 if anchor.startswith('m') else 0)
        for c in text:
            draw.text((xpos,y*S), c, font=f, fill=color, anchor='lt')
            xpos += draw.textlength(c,font=f)+track*S
        box = (x-length/S/2 if anchor.startswith('m') else x,y,x+length/S/2 if anchor.startswith('m') else x+length/S,y+size)
    else:
        box0 = draw.textbbox((x*S,y*S), text, font=f, anchor=anchor)
        draw.text((x*S,y*S), text, font=f, fill=color, anchor=anchor)
        box = tuple(v/S for v in box0)
    text_records.append({'text':text, 'bbox':[round(v,2) for v in box]})

def line(points, fill=LINE, width=1):
    draw.line([(round(x*S),round(y*S)) for x,y in points], fill=fill, width=max(1,round(width*S)), joint='curve')

def dashed_curve(p0,p1,p2,p3, fill=(170,177,157), width=.9):
    t = np.linspace(0,1,220)
    pts = [((1-u)**3*p0[0]+3*(1-u)**2*u*p1[0]+3*(1-u)*u*u*p2[0]+u**3*p3[0],
            (1-u)**3*p0[1]+3*(1-u)**2*u*p1[1]+3*(1-u)*u*u*p2[1]+u**3*p3[1]) for u in t]
    phase = 0.
    for a,b in zip(pts[:-1],pts[1:]):
        if phase % 13 < 6:
            line([a,b], fill=fill,width=width)
        phase += math.dist(a,b)

def fmt(v):
    return str(v)

def xy(center,angle,r=R):
    a = math.radians(angle)
    return (center[0]+r*math.cos(a), center[1]+Q*r*math.sin(a))

# Header and exact data labels are all code typeset.
draw.ellipse((90*S,88*S,101*S,99*S), fill=GOLD)
txt(DATA['visible_notice'],118,85,21,'sans',track=2.2)
txt('ALLOCATION  /  01',1396,85,20,'sans',MUTED,anchor='rt')
line([(90,126),(1396,126)],fill=(173,180,160),width=.9)
txt('Synthetic',87,166,100,'serif')
txt('双瓜面薄切片',88,285,79,'cjk')
txt('两个等面积瓜面，合计 100%',91,403,28,'cjk')
txt(DATA['series'][0]['label']+'  ·  Share of combined total (%)',92,457,24,'sans',MUTED)

left=(375,1005)
right=(1045,1005)
for i,c in enumerate((left,right)):
    txt(CATS[i],c[0],550,29,'sans',MUTED,anchor='mt')
    txt(fmt(VALUES[i])+'%',c[0],596,108,'narrow',anchor='mt')
    line([(c[0],704),(c[0],714)],width=1.1)

# B, C, D, E, F follow input order counterclockwise. The 7.2-degree gap
# lies at 64 degrees in screen coordinates. No angular minimum is applied.
tiny_total = sum(VALUES[2:])
gap_degrees = 360*tiny_total/DATA['right_capacity']
gap_mid = 64.
gap_low = gap_mid-gap_degrees/2
gap_high = gap_mid+gap_degrees/2
parts = [
    {'i':0,'center':left,'a0':0.,'a1':360.,'translation':[0,0]},
    {'i':1,'center':right,'a0':gap_high,'a1':gap_low+360,'translation':[0,0]},
]
slice_tips=[(730,1410),(895,1410),(1060,1410),(1225,1410)]
cursor=gap_high
for i,c in zip(range(2,6),slice_tips):
    deg=360*VALUES[i]/DATA['right_capacity']
    parts.append({'i':i,'center':c,'a0':cursor-deg,'a1':cursor,
                  'translation':[c[0]-right[0],c[1]-right[1]]})
    cursor -= deg

# Fine dashed provenance connectors are behind the data objects.
for p in parts[2:]:
    mid=(p['a0']+p['a1'])/2
    source=xy(right,mid,R+2)
    tip=p['center']
    dashed_curve(source,(source[0],source[1]+47),(tip[0],tip[1]-45),tip)

# One canonical source texture and one geometry mapping for all top faces.
skin = Image.open(ROOT / 'assets/melon_top.png').convert('RGB')
skin = skin.resize((2*R*S,round(2*R*Q*S)), Image.Resampling.LANCZOS)
toptex = Image.new('RGB',(L*S,L*S),(225,187,55))
toptex.paste(skin,(PAD*S,round((ORIGIN-R*Q)*S)))
flesh0 = Image.open(ROOT / 'assets/melon_flesh.png').convert('RGB').resize((L*S,180*S),Image.Resampling.LANCZOS)
flesh0 = ImageEnhance.Brightness(flesh0).enhance(.86)
flesh = Image.new('RGB',(L*S,L*S))
for y in range(0,L*S,180*S):
    flesh.paste(flesh0,(0,y))
del flesh0

grid=(np.arange(L*S,dtype=np.float32)+.5)/S-ORIGIN
gx=grid[None,:]
gy=grid[:,None]/Q
rad2=gx*gx+gy*gy
angles=np.mod(np.degrees(np.arctan2(gy,gx)),360)
inside=rad2 <= R*R
del rad2
all_masks=[]
geometry=[]
EXTRUDE=(-3,8)

def shifted(mask,dx,dy):
    m=Image.new('L',mask.size,0)
    m.paste(mask,(dx,dy))
    return m

for p in parts:
    i=p['i']
    a0,a1=p['a0'],p['a1']
    span=a1-a0
    angle_ok = np.mod(angles-a0,360) < span if span < 359.999999 else np.ones(inside.shape,dtype=bool)
    m=Image.fromarray((inside & angle_ok).astype('uint8')*255,'L')
    union=m.copy()
    for k in range(1,EXTRUDE[1]*S+1):
        union=ImageChops.lighter(union,shifted(m,round(EXTRUDE[0]*S*k/(EXTRUDE[1]*S)),k))
    side=ImageChops.subtract(union,m)
    # A soft shadow of the actual silhouette, not an encoded region.
    sh=shifted(union,3*S,9*S).filter(ImageFilter.GaussianBlur(7*S))
    sh=sh.point(lambda x: round(x*.21))
    shadow=Image.new('RGBA',(L*S,L*S),(59,63,39,0))
    shadow.putalpha(sh)
    pos=(round((p['center'][0]-ORIGIN)*S),round((p['center'][1]-ORIGIN)*S))
    canvas.alpha_composite(shadow,pos)
    layer=Image.new('RGBA',(L*S,L*S),(0,0,0,0))
    layer.paste(flesh,(0,0),side)
    layer.paste(toptex,(0,0),m)
    canvas.alpha_composite(layer,pos)
    # Pixel-area checks use the same actual mask passed to the compositor.
    full=Image.new('L',(W*S,H*S),0)
    full.paste(m,pos)
    mask_final=full.resize((W,H),Image.Resampling.LANCZOS)
    mask_final.save(OUT/'masks'/f'category_{chr(65+i)}.png')
    maskarr=np.asarray(mask_final,dtype=np.float32)/255.
    all_masks.append(maskarr)
    true_area=math.pi*R*R*Q*span/360
    geometry.append({'category':CATS[i], 'value':VALUES[i], 'degrees':round(span,8),
                     'angle_start_screen_degrees':round(a0,8), 'angle_end_screen_degrees':round(a1,8),
                     'center':list(p['center']), 'translation_from_source':p['translation'],
                     'top_area_analytic_px2':round(true_area,6),
                     'top_area_compositor_subpixels_px2':round(float(np.asarray(m,dtype=np.float64).sum()/255/S**2),6),
                     'top_area_final_antialiased_px2':round(float(maskarr.sum()),6),
                     'tip_to_arc_radius_px':R, 'vertical_scale':Q,
                     'mask_path':f'artifacts/masks/category_{chr(65+i)}.png'})
    del full,mask_final,shadow,sh,m,layer,side,union

draw=ImageDraw.Draw(canvas)
txt('完整瓜面',left[0],1337,24,'cjk',MUTED,anchor='mt')

# The small portions remain small; external labels provide legibility.
for p in parts[2:]:
    i=p['i']
    mid=(p['a0']+p['a1'])/2
    at=xy(p['center'],mid,R)
    label_x=p['center'][0]+104
    line([(at[0]+2,at[1]+9),(at[0]+2,1720),(label_x,1720),(label_x,1741)],fill=LINE,width=1.05)
    txt(CATS[i],label_x,1754,23,'sans',MUTED,anchor='mt')
    txt(fmt(VALUES[i])+'%',label_x,1798,49,'narrow',INK,anchor='mt')

txt('C—F',92,1456,23,'sans',GOLD,track=1.4)
txt('四片合计',92,1500,32,'cjk')
txt(f'{tiny_total:g}%',88,1557,103,'narrow')
txt('扇角不放大，只平移展开。',92,1700,24,'cjk',MUTED)
txt('虚线指向右瓜面的切片来源。',92,1739,23,'cjk',MUTED)

line([(90,1899),(1396,1899)],fill=(173,180,160),width=.9)
txt('读图方式',91,1931,22,'cjk',INK)
txt('每个完整瓜面 = 50%。占比以两个瓜面的合计顶面面积为分母。',256,1931,22,'cjk',MUTED)
txt('顶面统一纵向缩放 0.96；侧壁、阴影与分离空隙不计面积。',256,1971,22,'cjk',MUTED)
txt('UNIT  /  合计占比（%）',91,2040,18,'cjk',MUTED)
txt('SYNTHETIC ALLOCATION',1396,2040,18,'sans',MUTED,anchor='rt')

total_area=sum(g['top_area_final_antialiased_px2'] for g in geometry)
for g in geometry:
    g['visible_top_share_percent']=round(100*g['top_area_final_antialiased_px2']/total_area,6)
    g['absolute_error_percentage_points']=round(abs(g['visible_top_share_percent']-g['value']),6)
stack=np.stack(all_masks)
overlap_pixels=int(np.count_nonzero(np.sum(stack>.5,axis=0)>1))
report={'canvas':[W,H], 'supersampling':S, 'equal_base_radius':R, 'common_vertical_scale':Q,
        'equal_base_top_area_px2':math.pi*R*R*Q,
        'category_order':CATS, 'right_order_direction':'counterclockwise',
        'sidewall_extrusion_px':list(EXTRUDE), 'sidewalls_encode_data':False,
        'top_mask_overlap_pixels_at_alpha_gt_half':overlap_pixels,
        'parts':geometry, 'labels':text_records,
        'all_text_within_canvas':all(t['bbox'][0]>=0 and t['bbox'][1]>=0 and t['bbox'][2]<=W and t['bbox'][3]<=H for t in text_records),
        'generated_top_material_used_for_all_six_categories':True}
(OUT/'geometry.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
canvas.convert('RGB').resize((W,H),Image.Resampling.LANCZOS).save(ROOT/'final.png')
print(json.dumps({'canvas':[W,H],'top_mask_overlap_pixels':overlap_pixels,
                  'top_share_percent':[g['visible_top_share_percent'] for g in geometry]},ensure_ascii=False))
