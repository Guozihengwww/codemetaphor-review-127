"""Rebuild the infographic from frozen data and local generated raster parts.

Usage: PYTHON render.py
All complete experimental runs must be launched with capture_render.py.
No network access or generative model calls occur here.
"""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / 'data.json').read_text(encoding='utf-8'))
BRIEF = json.loads((ROOT / 'brief.json').read_text(encoding='utf-8'))
PARTS = json.loads((ROOT / 'assets/parts.json').read_text(encoding='utf-8'))
W, H = BRIEF['width'], BRIEF['height']
S = 2
INK = '#111D20'
MUTED = '#657A81'
TEAL = '#148AA0'
PALE = '#EAF4F5'
GRID = '#E4EDEF'
BG = '#FFFFFF'
FONT_DIR = ROOT / 'assets/fonts'
im = Image.new('RGBA', (W*S,H*S), BG)
draw = ImageDraw.Draw(im)

def p(value):
    return int(round(value*S))

def font(size, bold=False, cjk=False):
    name = ('NotoSansCJK-Bold.ttc' if bold else 'NotoSansCJK-Regular.ttc') if cjk else ('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')
    return ImageFont.truetype(str(FONT_DIR/name), p(size), index=0)

text_boxes=[]
def text(x,y,value,size=32,fill=INK,bold=False,anchor='lt',cjk=False,track=None):
    f=font(size,bold,cjk)
    if track is None:
        draw.text((p(x),p(y)),str(value),font=f,fill=fill,anchor=anchor)
        box=draw.textbbox((p(x),p(y)),str(value),font=f,anchor=anchor)
    else:
        width=sum(draw.textlength(ch,font=f) for ch in str(value))+p(track)*max(0,len(str(value))-1)
        xx=p(x) - (width/2 if anchor.startswith('m') else width if anchor.startswith('r') else 0)
        box=[xx,p(y),xx+width,p(y)+p(size)]
        for ch in str(value):
            draw.text((int(round(xx)),p(y)),ch,font=f,fill=fill,anchor='lt')
            xx += draw.textlength(ch,font=f)+p(track)
    text_boxes.append({'text':str(value),'bbox_xyxy':[round(v/S,2) for v in box]})

def line(points,fill=GRID,width=1):
    draw.line([(p(x),p(y)) for x,y in points],fill=fill,width=max(1,p(width)))

def rect(box,fill,outline=None,width=1):
    draw.rectangle(tuple(p(v) for v in box),fill=fill,outline=outline,width=p(width))

# Header: fixed context and the exact supplied title.
rect((108,83,455,132),INK)
text(281.5,97,DATA['visible_notice'],22,'#FFFFFF',True,'mt',track=2.1)
text(1688,101,'FIVE REGIONS  /  ANNUAL',22,MUTED,False,'rt',track=1.15)
text(108,189,DATA['title'],83,INK,True,cjk=True)
text(108,302,DATA['series'][0]['label'],43,MUTED)
line([(108,401),(1692,401)],INK,1.5)

axis=DATA['value_axis']
assert axis['minimum']==0 and axis['scale']=='linear'
values=DATA['series'][0]['values']
categories=DATA['categories']
assert len(values)==len(categories)==5
Y0=734.0
FULL_LENGTH=1120.0
VMAX=float(axis['maximum'])
SCALE=FULL_LENGTH/VMAX
CENTERS=[407.0+i*301.0 for i in range(5)]
FAUCET_WIDTH=244.0
STREAM_WIDTH=38.0
SPLASH_WIDTH=156.0
SPLASH_HEIGHT=100.0

# A shared downward scale. Grid is light enough to leave the water foremost.
text(168,682,DATA['series'][0]['unit'],23,MUTED,False,'rt')
for value in range(0,int(VMAX)+1,1500):
    yy=Y0+value*SCALE
    text(138,yy,format(value,','),25,MUTED,False,'rm')
    line([(154,yy),(170,yy)],MUTED,1)
    line([(181,yy),(1692,yy)],GRID,1)
line([(166,Y0),(166,Y0+FULL_LENGTH)],GRID,1)

faucet=Image.open(ROOT/'assets/faucet.png').convert('RGBA')
faucet=faucet.crop((0,0,faucet.width,PARTS['faucet']['outlet_bottom_y']))
texture=Image.open(ROOT/'assets/water_texture.png').convert('RGBA')
splash=Image.open(ROOT/'assets/splash.png').convert('RGBA')
fw=p(FAUCET_WIDTH)
fh=round(faucet.height/faucet.width*fw)
faucet_scaled=faucet.resize((fw,fh),Image.Resampling.LANCZOS)
outlet_offset=PARTS['faucet']['outlet_center_x']/faucet.width*FAUCET_WIDTH
geometry=[]

# Preserve raster part masks for independent pixel-edge checks.
water_mask=Image.new('L',im.size,0)
faucet_mask=Image.new('L',im.size,0)
for index,(cx,category,value) in enumerate(zip(CENTERS,categories,values)):
    length=float(value)*SCALE
    yend=Y0+length
    top=p(Y0)
    bottom=p(yend)
    nrows=bottom-top
    fx=p(cx-outlet_offset)
    fy=top-fh
    category_x=fx/S+FAUCET_WIDTH/2
    text(category_x,476,category,35,INK,True,'mt')

    # Water geometry is controlled in one common value-to-y mapping.
    # The entire splash is above the endpoint; its area and width encode nothing.
    layer=Image.new('RGBA',im.size,(0,0,0,0))
    if nrows>0:
        tw=p(STREAM_WIDTH)
        # A continuous pale underprint meets the outlet with no raster gap.
        # Generated water texture supplies the visible internal material.
        ImageDraw.Draw(layer).rectangle((p(cx)-tw//2,top,p(cx)+tw//2-1,bottom-1),fill='#AAE6ED')
        flowing=texture.resize((tw,nrows),Image.Resampling.LANCZOS)
        # Confine all generated pale edge pixels to the exact quantitative rectangle.
        layer.alpha_composite(flowing,(p(cx)-tw//2,top))
        sh=min(p(SPLASH_HEIGHT),nrows)
        sw=p(SPLASH_WIDTH) if sh==p(SPLASH_HEIGHT) else max(1,round(p(SPLASH_WIDTH)*sh/p(SPLASH_HEIGHT)))
        landing=splash.resize((sw,sh),Image.Resampling.LANCZOS)
        layer.alpha_composite(landing,(p(cx)-sw//2,bottom-sh))
        # Code-controlled impact line clarifies the bottom of the generated splash.
        ld=ImageDraw.Draw(layer)
        ld.line([(p(cx)-sw//2+p(3),bottom-2),(p(cx)+sw//2-p(3),bottom-2)],fill='#8BDCE7',width=2)
        # Keep every visible water pixel in [outlet lower edge, impact baseline).
        a=np.asarray(layer.getchannel('A')).copy()
        a[:top,:]=0
        a[bottom:,:]=0
        layer.putalpha(Image.fromarray(a,'L'))
        im.alpha_composite(layer)
        water_mask=Image.fromarray(np.maximum(np.asarray(water_mask),a),'L')

    im.alpha_composite(faucet_scaled,(fx,fy))
    faucet_mask.paste(faucet_scaled.getchannel('A'),(fx,fy))
    draw=ImageDraw.Draw(im)
    # Exact supplied values, each anchored directly under its own impact.
    text(cx,yend+28,format(value,','),46,TEAL,True,'mt')
    text(cx,yend+87,DATA['series'][0]['unit'],23,MUTED,False,'mt')
    geometry.append({'category':category,'value':value,'outlet_center_x':cx,
        'outlet_lower_edge_y':Y0,'expected_impact_y':yend,'expected_length_px':length,
        'raster_bottom_exclusive_px':bottom/S,
        'faucet_bbox_xywh':[fx/S,fy/S,fw/S,fh/S],
        'splash_bbox_xywh':[(p(cx)-sw//2)/S,(bottom-sh)/S,sw/S,sh/S]})

# Current-data numeric callout uses only an explicit sum of supplied inputs.
rect((810,1490,1692,1778),BG)
line([(832,1511),(832,1749)],'#8BDCE7',5)
text(877,1515,'TOTAL ACROSS FIVE REGIONS',23,MUTED,True,track=1.5)
text(873,1572,format(sum(values),','),107,TEAL,True)
text(879,1703,DATA['series'][0]['unit'],31,TEAL)
text(1110,1710,'combined annual resources',24,MUTED)

# Concise reading key, unit definition, and synthetic-data provenance.
line([(108,1974),(1692,1974)],INK,1.5)
text(108,2013,'READING THE FLOW',22,INK,True,track=1.5)
text(505,2008,'Vertical drop from outlet to splash = annual resources',28,INK)
text(505,2055,'Gm³/year = 十亿立方米/年（billion m³ per year）',27,MUTED,cjk=True)
text(108,2130,'Synthetic inputs · Region order as supplied',23,MUTED)
text(1692,2130,'Linear scale  /  0–9,000 Gm³/year',23,MUTED,False,'rt')

(ROOT/'artifacts').mkdir(exist_ok=True)
# Masks are diagnostics, not additional complete chart candidates.
water_mask.resize((W,H),Image.Resampling.LANCZOS).save(ROOT/'artifacts/water_mask.png')
faucet_mask.resize((W,H),Image.Resampling.LANCZOS).save(ROOT/'artifacts/faucet_mask.png')
geometry_report={'canvas':[W,H],'shared_outlet_y':Y0,'full_scale_length_px':FULL_LENGTH,
    'value_axis':axis,'pixels_per_Gm3_year':SCALE,'marks':geometry,'text_boxes':text_boxes,
    'derived_total':sum(values),'supersampling':S}
(ROOT/'artifacts/geometry.json').write_text(json.dumps(geometry_report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
im=im.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
im.save(ROOT/'final.png')
print('Saved final.png: %d × %d; %d data-bound faucet marks.'%(W,H,len(values)))
