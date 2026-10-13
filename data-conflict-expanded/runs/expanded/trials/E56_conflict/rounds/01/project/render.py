#!/usr/bin/env python3
"""Offline crop-harvest infographic. Run through capture_render.py for every full render."""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent
S = 3
INPUT = json.loads((ROOT / 'input.json').read_text(encoding='utf-8'))
BRIEF = json.loads((ROOT / INPUT['brief']).read_text(encoding='utf-8'))
DATA = json.loads((ROOT / INPUT['data']).read_text(encoding='utf-8'))
CORE = json.loads((ROOT / INPUT['critical_metaphor_contract']).read_text(encoding='utf-8'))
assert (ROOT / INPUT['reference']).is_file()
W, H = BRIEF['width'], BRIEF['height']
VALUES = DATA['series'][0]['values']
CATEGORIES = DATA['categories']
TOTAL = DATA['series_totals'][DATA['series'][0]['id']]
assert len(VALUES) == len(CATEGORIES) == 5 and sum(VALUES) == TOTAL
assert DATA['value_axis'] == {'minimum': 0, 'maximum': 72000, 'scale': 'linear'}
BASE_Y, SCALE_HEIGHT, BASE_THICKNESS = 1345.0, 995.0, 32.0
RADIUS = 130.0
ANGLE = 17.0
RY = RADIUS * math.sin(math.radians(ANGLE))
INNER_R = RADIUS - 7.0
INNER_RY = INNER_R * math.sin(math.radians(ANGLE))
CENTERS = [275.0, 600.0, 925.0, 1250.0, 1575.0]
INK = (30, 56, 46)
GREEN = (34, 102, 49)
MUTED = (96, 113, 104)
FONT_DIR = ROOT / 'assets/fonts'

def font(size, bold=False):
    return ImageFont.truetype(str(FONT_DIR / ('Sans-Bold.otf' if bold else 'Sans-Regular.otf')), round(size*S))

def text(draw, xy, value, size=24, fill=INK, bold=False, anchor='lt'):
    draw.text((xy[0]*S,xy[1]*S), value, font=font(size,bold), fill=fill, anchor=anchor, stroke_width=0)

def tracking(draw, xy, value, size=20, spacing=2.0, fill=MUTED, bold=False):
    x,y=xy
    ft=font(size,bold)
    for c in value:
        draw.text((x*S,y*S),c,font=ft,fill=fill,anchor='lt')
        x += draw.textlength(c,font=ft)/S + spacing

def line(draw, points, fill, width=1):
    draw.line([(round(x*S),round(y*S)) for x,y in points],fill=fill,width=max(1,round(width*S)))

def ellipse(draw, box, fill=None, outline=None, width=1):
    draw.ellipse(tuple(round(v*S) for v in box), fill=fill, outline=outline, width=max(1,round(width*S)))

def arc(draw, box, start, end, fill, width=1):
    draw.arc(tuple(round(v*S) for v in box),start,end,fill=fill,width=max(1,round(width*S)))

# Light photographic studio backdrop. Deterministic; no network or model calls.
yy,xx=np.mgrid[0:H,0:W].astype(np.float32)
glow=np.exp(-(((xx-955)/1170)**2 + ((yy-910)/840)**2)*1.35)
floor=np.exp(-((yy-1400)/220)**2)
bg=np.zeros((H,W,3),dtype=np.float32)
for c,base in enumerate([231,237,232]):
    bg[:,:,c]=base + 17*glow - 2.5*floor
im=Image.fromarray(np.uint8(np.clip(bg,0,255))).convert('RGBA').resize((W*S,H*S),Image.Resampling.BICUBIC)
del yy,xx,bg,glow,floor
D=ImageDraw.Draw(im)

# Editorial header and explicit provenance.
tracking(D,(120,86),'AGRICULTURE / ANNUAL OUTPUT',21,2.3,GREEN,True)
text(D,(116,147),'Annual crop harvest',94,INK,True)
text(D,(121,266),'Synthetic harvest · metric tonnes per year (t/year)',29,MUTED)
D.rounded_rectangle((1510*S,80*S,1880*S,140*S),radius=30*S,fill=(226,236,225),outline=(95,132,103),width=2*S)
text(D,(1695,111),DATA['visible_notice'],24,GREEN,True,anchor='mm')
text(D,(1880,176),'FIVE CROP GROUPS',20,MUTED,False,anchor='rt')
line(D,[(120,333),(1880,333)],(177,192,180),1.2)

tracking(D,(120,405),'TOTAL ANNUAL HARVEST',22,2.0,GREEN,True)
text(D,(113,453),f'{TOTAL:,}',126,GREEN,True)
text(D,(546,524),'t/year',32,INK)
text(D,(121,601),'Five crop groups · one synthetic series',24,MUTED)

# A single shared zero-based scale, positioned behind all vessels.
axis_x = 1833
text(D,(1833,303),'t/year',20,MUTED,anchor='mt')
line(D,[(axis_x,350),(axis_x,BASE_Y)],(162,178,165),1.2)
for tick in [0,18000,36000,54000,72000]:
    y=BASE_Y-tick/72000*SCALE_HEIGHT
    line(D,[(axis_x-9,y),(axis_x+9,y)],(128,148,133),1.2)
    text(D,(axis_x+24,y),f'{tick:,}',23,MUTED,anchor='lm')
    if tick in [0,18000,36000]:
        # Quiet dotted guide lines retain the spacious studio composition.
        for x in range(145,1785,13):
            line(D,[(x,y),(x+3,y)],(197,208,199),0.7)

# Ground contact shadows are separate from the 32-pixel non-data bases.
shadow=Image.new('RGBA',im.size,(0,0,0,0))
sd=ImageDraw.Draw(shadow)
for cx in CENTERS:
    ellipse(sd,(cx-147,BASE_Y+BASE_THICKNESS+RY-7,cx+150,BASE_Y+BASE_THICKNESS+RY+24),(45,66,58,43))
shadow=shadow.filter(ImageFilter.GaussianBlur(11*S))
im=Image.alpha_composite(im,shadow)
D=ImageDraw.Draw(im)

geometry=[]
artifact_dir=ROOT/'artifacts'
artifact_dir.mkdir(exist_ok=True)

for i,(cx,value,name) in enumerate(zip(CENTERS,VALUES,CATEGORIES)):
    height=value/72000*SCALE_HEIGHT
    top=BASE_Y-height
    local_left=math.floor(cx-RADIUS-18)
    local_top=math.floor(top-RY-8)
    local_right=math.ceil(cx+RADIUS+18)
    local_bottom=math.ceil(BASE_Y+BASE_THICKNESS+RY+7)
    lw,lh=(local_right-local_left)*S,(local_bottom-local_top)*S
    # Sample at supersampled pixel centers. All masks use exactly the same mapping.
    y,x=np.mgrid[0:lh,0:lw].astype(np.float32)
    X=local_left+(x+.5)/S-cx
    Y=local_top+(y+.5)/S
    u=X/INNER_R
    profile=INNER_RY*np.sqrt(np.maximum(0,1-u*u))
    crop_mask=(np.abs(u)<=1)&(Y>=top-profile)&(Y<=BASE_Y+profile)
    top_mask=(u*u+((Y-top)/INNER_RY)**2)<=1
    outer_u=X/RADIUS
    outer_profile=RY*np.sqrt(np.maximum(0,1-outer_u*outer_u))
    outer_mask=(np.abs(outer_u)<=1)&(Y>=top-outer_profile)&(Y<=BASE_Y+BASE_THICKNESS+outer_profile)
    base_mask=(np.abs(outer_u)<=1)&(Y>=BASE_Y-outer_profile)&(Y<=BASE_Y+BASE_THICKNESS+outer_profile)
    front_base=(np.abs(outer_u)<=1)&(Y>=BASE_Y+outer_profile)&(Y<=BASE_Y+BASE_THICKNESS+outer_profile)
    # Clear thick glass base with refracted bright and cool-gray bands.
    base=np.zeros((lh,lw,4),dtype=np.uint8)
    p=(Y-BASE_Y-outer_profile)/BASE_THICKNESS
    shade=(213+21*np.cos(outer_u*np.pi/2) + 14*np.exp(-((p-.29)/.11)**2)-34*np.exp(-((p-.80)/.09)**2))
    base[:,:,0]=np.uint8(np.clip(shade-9,0,255))
    base[:,:,1]=np.uint8(np.clip(shade,0,255))
    base[:,:,2]=np.uint8(np.clip(shade-1,0,255))
    base[:,:,3]=np.where(front_base,222,np.where(base_mask,105,0)).astype(np.uint8)
    layer=Image.fromarray(base,'RGBA')
    ld=ImageDraw.Draw(layer)
    def lb(yc,rx=RADIUS,ry=RY):
        return (cx-rx-local_left,yc-ry-local_top,cx+rx-local_left,yc+ry-local_top)
    ellipse(ld,lb(BASE_Y+BASE_THICKNESS),outline=(108,128,121,150),width=1.3)
    arc(ld,lb(BASE_Y+BASE_THICKNESS-2),0,180,(250,255,251,240),2)
    arc(ld,lb(BASE_Y+BASE_THICKNESS-7),4,176,(114,135,125,160),1)
    arc(ld,lb(BASE_Y+9),2,178,(255,255,255,235),2)
    arc(ld,lb(BASE_Y+15),8,172,(252,255,254,135),1.1)

    tex=Image.open(ROOT/f'assets/{name.lower()}_texture.png').convert('RGB')
    src=np.asarray(tex,dtype=np.float32)
    th,tw=src.shape[:2]
    # Curved mapping on the visible cylinder face. No vertical compression of data height.
    tx=((np.arcsin(np.clip(u,-1,1))/np.pi+.5)*(tw-1)).astype(np.int32)
    ty=np.mod((Y-top-profile)*(tw/(2*INNER_R))+.36*th,th-1).astype(np.int32)
    side=src[ty,tx]
    cylindrical_light=.66+.35*np.sqrt(np.maximum(0,1-u*u))
    side=np.clip(side*cylindrical_light[:,:,None],0,255)
    # Project only the circular end-cap texture, not the quantitative screen-space height.
    ttx=np.clip((u+1)*.5*(tw-1),0,tw-1).astype(np.int32)
    tty=np.clip(.34*th+((Y-top)/INNER_RY)*tw*.5,0,th-1).astype(np.int32)
    cap=np.clip(src[tty,ttx]*1.075+3,0,255)
    rgb=np.where(top_mask[:,:,None],cap,side)
    rgba=np.zeros((lh,lw,4),dtype=np.uint8)
    rgba[:,:,:3]=rgb.astype(np.uint8)
    rgba[:,:,3]=crop_mask.astype(np.uint8)*255
    layer=Image.alpha_composite(layer,Image.fromarray(rgba,'RGBA'))

    # Translucent glass walls, integral reflections and rounded rim.
    glass=np.zeros((lh,lw,4),dtype=np.uint8)
    glass[:,:,:3]=(248,255,251)
    reflect=(.22*np.exp(-((outer_u+.78)/.056)**2)+.12*np.exp(-((outer_u+.64)/.024)**2)+.32*np.exp(-((outer_u-.84)/.043)**2)+.08*np.exp(-((outer_u-.69)/.045)**2))
    glass[:,:,3]=np.uint8(np.clip(reflect*255,0,180))*outer_mask.astype(np.uint8)
    layer=Image.alpha_composite(layer,Image.fromarray(glass,'RGBA'))
    ld=ImageDraw.Draw(layer)
    for side_sign in [-1,1]:
        edge_x=cx+side_sign*RADIUS-local_left
        line(ld,[(edge_x,top-local_top),(edge_x,BASE_Y+BASE_THICKNESS-local_top)],(117,138,128,170),1.25)
        inner_x=edge_x-side_sign*2.5
        line(ld,[(inner_x,top-local_top+1),(inner_x,BASE_Y+BASE_THICKNESS-local_top-2)],(255,255,255,215),1.6)
    ellipse(ld,lb(top),outline=(115,137,124,200),width=2.0)
    ellipse(ld,lb(top-.7,RADIUS-1.0,RY-1.0),outline=(252,255,251,245),width=2.2)
    arc(ld,lb(top+.8,RADIUS-3.8,RY-1.6),1,179,(104,129,113,200),1.25)
    arc(ld,lb(top-1.3,RADIUS-2,RY-1.5),188,352,(255,255,255,255),2.1)
    arc(ld,lb(BASE_Y,INNER_R,INNER_RY),0,180,(131,153,138,120),.85)
    im.alpha_composite(layer,(local_left*S,local_top*S))

    # Store only a local coverage mask, not another complete chart render.
    coverage=Image.fromarray(crop_mask.astype(np.uint8)*255,'L').resize((local_right-local_left,local_bottom-local_top),Image.Resampling.LANCZOS)
    coverage.save(artifact_dir/f'content_mask_{i+1:02d}.png')
    D=ImageDraw.Draw(im)
    label_y=top-RY-143
    text(D,(cx,label_y),f'{value:,}',63,GREEN,True,anchor='mt')
    text(D,(cx,label_y+72),DATA['series'][0]['unit'],24,INK,anchor='mt')
    text(D,(cx,label_y+108),f'{value/TOTAL*100:.2f}% of total',23,MUTED,anchor='mt')

    # Bracket endpoints align with the exact two ellipse centers. No minimum height.
    bx=cx+RADIUS+16
    line(D,[(bx,top),(bx,BASE_Y)],(76,115,86),1.0)
    line(D,[(bx-4,top),(bx+4,top)],(76,115,86),1.0)
    line(D,[(bx-4,BASE_Y),(bx+4,BASE_Y)],(76,115,86),1.0)
    text(D,(cx,1456),name,33,INK,False,anchor='mt')
    geometry.append({'category':name,'value':value,'unit':'t/year','share_percent':value/TOTAL*100,
                     'center_x':cx,'mouth_center_y':top,'content_bottom_center_y':BASE_Y,
                     'data_height_px':height,'base_thickness_px':BASE_THICKNESS,
                     'outer_radius_x':RADIUS,'outer_radius_y':RY,
                     'content_radius_x':INNER_R,'content_radius_y':INNER_RY,
                     'bracket_x':bx,'mask_path':f'artifacts/content_mask_{i+1:02d}.png',
                     'mask_origin':[local_left,local_top], 'label_top_y':label_y})

D=ImageDraw.Draw(im)
line(D,[(120,1525),(1880,1525)],(177,192,180),1.2)
text(D,(120,1552),'Vessel height = annual harvest',24,INK,True)
text(D,(523,1552),'· Shared linear scale: 0–72,000 t/year',24,MUTED)
text(D,(120,1591),'Measured between ellipse centers. Clear 32 px bases are excluded; all vessels use a 17° view.',22,MUTED)
text(D,(1880,1634),'SYNTHETIC DATA · Illustrative quantities, not observed crop statistics.',18,MUTED,anchor='rb')

final=im.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
final.save(ROOT/'final.png',optimize=True)
(artifact_dir/'geometry.json').write_text(json.dumps({'canvas':[W,H], 'supersampling':S, 'scale_height_px':SCALE_HEIGHT, 'plot_bottom_y':BASE_Y, 'angle_deg':ANGLE,'vessels':geometry},indent=2)+'\n',encoding='utf-8')
print('Rendered final.png from frozen data and local generated crop materials.')
