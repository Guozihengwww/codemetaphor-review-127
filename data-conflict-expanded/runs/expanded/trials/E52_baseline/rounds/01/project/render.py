"""Offline renderer. Run only through capture_render.py for complete attempts.

One linear mapping controls all boundaries: x = 125 + 1750 * cumulative_share/100.
Generated antique book faces are reconstructed inside, and clipped to, those bands.
"""
from pathlib import Path
from decimal import Decimal
from functools import lru_cache
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / 'data.json').read_text(encoding='utf-8'))
brief = json.loads((ROOT / 'brief.json').read_text(encoding='utf-8'))
core = json.loads((ROOT / 'core.json').read_text(encoding='utf-8'))
inputs = json.loads((ROOT / 'input.json').read_text(encoding='utf-8'))
assert all((ROOT / inputs[k]).is_file() for k in ('brief', 'data', 'reference', 'critical_metaphor_contract'))
assert data['synthetic'] and core['topic_zh'] == data['topic_zh']
assert len(data['categories']) == 7 and len(data['series']) == 1
series = data['series'][0]
assert series['unit'] == 'percent of population'
values = [Decimal(str(v)) for v in series['values']]
assert sum(values) == Decimal('100')

S = 4
W, H = brief['width'], brief['height']
assert (W, H) == (2000, 1392)
PAPER = (244, 242, 232)
INK = (39, 40, 33)
MUTED = (100, 98, 84)
ACCENT = (131, 82, 46)
RULE = (170, 161, 139)

@lru_cache(maxsize=100)
def font(name, size):
    return ImageFont.truetype(str(ROOT / 'assets' / 'fonts' / name), round(size*S))

canvas = Image.new('RGB', (W*S, H*S), PAPER)
draw = ImageDraw.Draw(canvas)

def text(x, y, s, size=28, face='serif.otf', color=INK, anchor='mm'):
    draw.text((round(x*S), round(y*S)), s, font=font(face,size), fill=color, anchor=anchor,
              stroke_width=0)

def line(points, color=RULE, width=1):
    draw.line([(round(x*S), round(y*S)) for x,y in points], fill=color, width=max(1,round(width*S)))

def tracked(x, y, s, size=18, tracking=3.5, color=MUTED):
    f=font('sans.ttf',size)
    lengths=[draw.textlength(c,font=f)/S for c in s]
    total=sum(lengths)+tracking*(len(s)-1)
    left=x-total/2
    for c, length in zip(s,lengths):
        text(left,y,c,size,'sans.ttf',color,'lm')
        left+=length+tracking
    return total

# Editorial heading: topic, unit and synthetic notice are distinct and readable.
notice_width=tracked(1000,80,data['visible_notice'],20,4.2,ACCENT)
line([(650,80),(1000-notice_width/2-26,80)], (173,150,119), 1)
line([(1000+notice_width/2+26,80),(1350,80)], (173,150,119), 1)
title=data['title'].removeprefix('SYNTHETIC ')
text(1000,169,title,82,'serif_bold.otf')
text(1000,243,'Religious affiliation and unaffiliated population',29,'serif.otf',MUTED)
text(1000,285,'Percent of population',21,'sans.ttf',MUTED)

# Exact linear front-view data strip; no minimum widths, gaps, protrusions or overlap.
X0=Decimal('125')
STRIP_W=Decimal('1750')
TOP, BOTTOM=367,1008
HEIGHT=BOTTOM-TOP
bounds=[X0]
for v in values:
    bounds.append(bounds[-1]+STRIP_W*v/Decimal('100'))
assert bounds[-1] == X0+STRIP_W
assert all(b*S == int(b*S) for b in bounds)

line([(125,351),(125,337),(866,337)],RULE,1)
line([(1134,337),(1875,337),(1875,351)],RULE,1)
tracked(1000,337,'100% TOTAL',16,2.9,MUTED)

# A shared grounding/contact shadow has no statistical meaning.
shadow=Image.new('RGBA',canvas.size,(0,0,0,0))
sd=ImageDraw.Draw(shadow)
sd.rounded_rectangle((123*S,995*S,1878*S,1022*S),radius=12*S,fill=(46,34,21,85))
shadow=shadow.filter(ImageFilter.GaussianBlur(10*S))
canvas=Image.alpha_composite(canvas.convert('RGBA'),shadow).convert('RGB')
draw=ImageDraw.Draw(canvas)

asset_names=['vermilion','ochre','emerald','tobacco','charcoal','indigo','chestnut']
label_font_sizes=[52,36,56,47,58,None,44]
spine_inks=[(248,218,149),(48,40,24),(242,231,201),(248,221,167),(243,234,209),None,(250,228,183)]

def book_face(path, target_width, target_height):
    """Preserve binding edges and use reflected interior strips to retain fine grain.

    Odd repetitions join both original edges without seams; repeats affect material
    texture only. All image geometry is finally clipped to its exact data band.
    """
    src=Image.open(path).convert('RGB')
    sw,sh=src.size
    if target_width < 22*S:
        # The tiny book is compressed, never enlarged to an invented minimum.
        return src.resize((target_width,target_height),Image.Resampling.LANCZOS)
    edge=30
    final_edge=min(round(edge*target_height/sh),target_width//4)
    left=src.crop((0,0,edge,sh)).resize((final_edge,target_height),Image.Resampling.LANCZOS)
    right=src.crop((sw-edge,0,sw,sh)).resize((final_edge,target_height),Image.Resampling.LANCZOS)
    mid=src.crop((edge,0,sw-edge,sh))
    natural_mid=mid.width*target_height/sh
    desired=(target_width-2*final_edge)/natural_mid
    reps=max(1,int(round((desired-1)/2))*2+1)
    tiled=Image.new('RGB',(mid.width*reps,sh))
    for r in range(reps):
        part=mid if r%2==0 else mid.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        tiled.paste(part,(r*mid.width,0))
    tiled=tiled.resize((target_width-2*final_edge,target_height),Image.Resampling.LANCZOS)
    out=Image.new('RGB',(target_width,target_height))
    out.paste(left,(0,0));out.paste(tiled,(final_edge,0));out.paste(right,(target_width-final_edge,0))
    # Contact shading and a restrained reflected highlight stay INSIDE this book.
    arr=np.asarray(out,dtype=np.float32)
    t=np.arange(target_width,dtype=np.float32)/S
    mult=1 - .09*np.exp(-t/2.1) - .20*np.exp(-(target_width/S-1/S-t)/2.1)
    mult+=.035*np.exp(-((t-3.0)/1.2)**2)
    arr=np.clip(arr*mult[None,:,None],0,255).astype('uint8')
    return Image.fromarray(arr)

def vertical_title(face, title, size, color):
    f=font('serif.otf',size)
    box=f.getbbox(title)
    padding=16*S
    layer=Image.new('RGBA',(box[2]-box[0]+padding*2,box[3]-box[1]+padding*2),(0,0,0,0))
    ld=ImageDraw.Draw(layer)
    px,py=padding-box[0],padding-box[1]
    ld.text((px+S,py+S/2),title,font=f,fill=(22,15,7,145))
    ld.text((px,py),title,font=f,fill=color+(255,))
    layer=layer.rotate(90,expand=True,resample=Image.Resampling.BICUBIC)
    center_y=round(face.height*0.494)
    face=face.convert('RGBA')
    face.alpha_composite(layer,((face.width-layer.width)//2,center_y-layer.height//2))
    return face.convert('RGB')

geometries=[]
for i,(name,value,asset) in enumerate(zip(data['categories'],values,asset_names)):
    left,right=float(bounds[i]),float(bounds[i+1])
    left_hi,right_hi=int(bounds[i]*S),int(bounds[i+1]*S)
    face=book_face(ROOT/'assets'/f'spine_{asset}.png',right_hi-left_hi,HEIGHT*S)
    if label_font_sizes[i] is not None:
        face=vertical_title(face,name,label_font_sizes[i],spine_inks[i])
    canvas.paste(face,(left_hi,TOP*S))
    geometries.append({'category':name,'value_percent':float(value),
        'left_px':left,'right_px':right,'width_px':right-left,
        'top_px':TOP,'bottom_px':BOTTOM,'asset':f'assets/spine_{asset}.png',
        'geometry_width_at_4x':right_hi-left_hi})

draw=ImageDraw.Draw(canvas)

# Alternating external leaders retain readable labels for every category.
for i,g in enumerate(geometries):
    center=(g['left_px']+g['right_px'])/2
    lower=i%2==1
    value_y=1214 if lower else 1092
    name_y=value_y+48
    end_y=value_y-39
    line([(center,BOTTOM+10),(center,end_y)],(91,85,68),1.25)
    rad=1.8
    draw.ellipse(((center-rad)*S,(BOTTOM+10-rad)*S,(center+rad)*S,(BOTTOM+10+rad)*S),fill=(80,73,58))
    value_text=f"{g['value_percent']:.1f}%"
    text(center,value_y,value_text,46,'serif_bold.otf',INK)
    text(center,name_y,g['category'],27,'serif.otf',INK)
    g['leader_x_px']=center
    g['label_value']=value_text
    g['label_value_center_y_px']=value_y
    g['label_category_center_y_px']=name_y

line([(125,1305),(1875,1305)],(192,184,163),1)
text(125,1344,'Spine width = share of population',20,'serif_italic.otf',MUTED,'lm')
text(1875,1344,'Illustrative synthetic data • Not real-world estimates',18,'sans.ttf',MUTED,'rm')

canvas=canvas.resize((W,H),Image.Resampling.LANCZOS)
canvas.save(ROOT/'final.png',compress_level=7)
(ROOT/'artifacts').mkdir(exist_ok=True)
geometry_report={'canvas_px':[W,H],'supersample':S,'strip':{'x':float(X0),'width':float(STRIP_W),'top':TOP,'bottom':BOTTOM},
 'mapping':'x = 125 + 1750 × cumulative percent / 100','minimum_width_applied':False,
 'category_order':data['categories'],'geometry':geometries,
 'source_files':['input.json','brief.json','data.json','core.json','reference.png'],
 'reference_pixels_used':False}
(ROOT/'artifacts'/'geometry.json').write_text(json.dumps(geometry_report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Rendered final.png (2000 × 1392), seven contiguous proportional book spines.')
