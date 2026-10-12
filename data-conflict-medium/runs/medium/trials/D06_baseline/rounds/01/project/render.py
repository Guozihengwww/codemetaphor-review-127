"""Render the frozen synthetic fork infographic from local data and assets only.

Run complete charts only through capture_render.py. No network/model calls.
"""
from pathlib import Path
import json
import math
import hashlib
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parent
S = 2
PX_PER_PERCENT = 20.0
ANGLE = math.radians(-38)
C, SN = math.cos(ANGLE), math.sin(ANGLE)
BLUE = '#124BA3'
RED = '#BC142A'
INK = '#11151A'
GRAY = '#555E68'
PALE = '#D4DAE1'

@lru_cache(None)
def font(size, style='sans'):
    files = {'sans':'sans.ttf', 'bold':'sans_bold.ttf', 'serif':'serif_bold.ttf', 'cjk':'cjk_black.ttc'}
    return ImageFont.truetype(str(ROOT / 'assets' / 'fonts' / files[style]), round(size*S))

def text(draw, xy, value, size, color=INK, style='sans', anchor='lt'):
    draw.text(tuple(round(v*S) for v in xy), str(value), font=font(size,style), fill=color, anchor=anchor)

def line(draw, coords, fill, width=1):
    draw.line([(round(x*S),round(y*S)) for x,y in coords], fill=fill, width=round(width*S))

def screen(origin, x, y):
    return (origin[0]+C*x-SN*y, origin[1]+SN*x+C*y)

def exact(v):
    return str(v) if isinstance(v, int) else format(v, '.12g')

def tine_label(category, value, color):
    cat = category+' '
    val = exact(value)+'%'
    cw = font(28).getlength(cat)
    vw = font(30,'bold').getlength(val)
    label = Image.new('RGBA', (math.ceil(cw+vw)+4*S, 42*S))
    d = ImageDraw.Draw(label)
    d.text((0,0), cat, font=font(28), fill=GRAY, anchor='lt')
    d.text((cw,0), val, font=font(30,'bold'), fill=color, anchor='lt')
    return label.rotate(-90, expand=True)

def fork_layer(series, categories, name, color, origin):
    # The same local coordinates and a single rigid transform are used for
    # every tine, root line, label and body. Only lengths depend on values.
    lx, ly = 300, 650
    layer = Image.new('RGBA', (760*S, 1480*S))
    body = Image.open(ROOT/'assets'/f'body_{name}.png').convert('RGBA')
    assert body.size == (400*S,816*S)
    strip = Image.open(ROOT/'assets'/f'tine_{name}.png').convert('RGB')
    layer.alpha_composite(body, (int((lx-200)*S), ly*S))
    lanes = np.linspace(-156,156,len(categories))
    geometry = []
    for index, (category, v, x) in enumerate(zip(categories,series['values'],lanes)):
        length = PX_PER_PERCENT*v
        w = 32*S
        if v > 0:
            hp = round(length*S)
            material = np.array(strip.resize((w,hp),Image.Resampling.LANCZOS)).astype(float)
            # Join the generated tine material to the generated head surface.
            bx = round((200+x-16)*S)
            base_colors = np.array(body)[2, bx:bx+w, :3].astype(float)
            blend_rows = min(24*S,hp)
            blend = np.linspace(0,1,blend_rows)[:,None,None]
            material[-blend_rows:] = material[-blend_rows:]*(1-blend) + base_colors[None,:,:]*blend
            tine = Image.fromarray(material.clip(0,255).astype(np.uint8)).convert('RGBA')
            mask = Image.new('L',(w,hp),255)
            # A 2px bevel is within the data boundary; the center tip is exact.
            if hp >= 4*S:
                md = ImageDraw.Draw(mask)
                md.rectangle((0,0,w-1,2*S), fill=0)
                md.rounded_rectangle((0,0,w-1,min(hp-1,5*S)),radius=2*S,fill=255)
                md.rectangle((0,2*S,w-1,hp-1),fill=255)
            tine.putalpha(mask)
            layer.alpha_composite(tine,(round((lx+x-16)*S),round((ly-length)*S)))
        label = tine_label(category,v,color)
        # Clockwise 90 degrees in local space; after the fork transform its
        # baseline reads naturally from upper left to lower right.
        layer.alpha_composite(label,(round((lx+x-18)*S),round((ly-length-17)*S)-label.height))
        geometry.append({'series':series['id'],'category':category,'value':v,'unit':series['unit'],
                         'slot':index,'local_x':float(x),'expected_length_px':length,
                         'root_xy':screen(origin,x,0),'tip_xy':screen(origin,x,-length)})
    d=ImageDraw.Draw(layer)
    # Root ticks lie at or below zero and cannot create a spurious zero tine.
    line(d,[(lx-213,ly),(lx-201,ly)],GRAY,1.4)
    line(d,[(lx+201,ly),(lx+217,ly)],GRAY,1.4)
    line(d,[(lx-200,ly+1),(lx+200,ly+1)],'#E8EBF0',1.1)
    text(d,(lx+221,ly-9),'ROOT · 0%',19,GRAY)
    for x in lanes:
        line(d,[(lx+x,ly+5),(lx+x,ly+13)],'#FFFFFF',1.5)
    transform=(C,SN,S*(lx-C*origin[0]-SN*origin[1]),
               -SN,C,S*(ly+SN*origin[0]-C*origin[1]))
    transformed=layer.transform((2000*S,1825*S),Image.Transform.AFFINE,transform,Image.Resampling.BICUBIC)
    return transformed,geometry

def pixel_geometry_check(final, records):
    rgb=np.array(final.convert('RGB')).astype(float)
    checks=[]
    # Trace the actual final colored pixels from the geometric zero root,
    # detecting the first sustained white gap before the external text.
    t=np.arange(1.25,461,0.25)
    for g in records:
        rx,ry=g['root_xy']
        xs=rx+SN*t
        ys=ry-C*t
        samples=np.stack([map_coordinates(rgb[:,:,ch],[ys,xs],order=1,mode='constant',cval=255) for ch in range(3)],axis=1)
        if g['series']=='table_service':
            colored=(samples[:,2]-samples[:,0]>27)&(samples[:,2]-samples[:,1]>8)
        else:
            colored=(samples[:,0]-samples[:,2]>30)&(samples[:,0]-samples[:,1]>25)
        gaps=np.flatnonzero(np.convolve((~colored).astype(int),np.ones(8,dtype=int),mode='valid')==8)
        end_index=int(gaps[0]) if len(gaps) else len(t)-1
        measured=0.0 if end_index==0 else float(t[end_index]-0.125)
        checks.append({**g,'measured_visible_centerline_px':round(measured,3),
                       'error_full_scale_percent':round(abs(measured-g['expected_length_px'])/400*100,4),
                       'within_3_percent_full_scale':bool(abs(measured-g['expected_length_px'])<=12)})
    return checks

def main():
    data=json.loads((ROOT/'data.json').read_text(encoding='utf-8'))
    brief=json.loads((ROOT/'brief.json').read_text(encoding='utf-8'))
    assert (brief['width'],brief['height'])==(2000,1825)
    assert data['value_axis']=={'minimum':0,'maximum':20,'scale':'linear'}
    assert len(data['series'])==2
    im=Image.new('RGBA',(brief['width']*S,brief['height']*S),'white')
    d=ImageDraw.Draw(im)
    # Quiet editorial framing, intentionally white as in the design brief.
    line(d,[(90,72),(1910,72)],INK,2)
    text(d,(90,103),'Synthetic',118,INK,'serif')
    text(d,(85,249),'双叉齿',117,INK,'cjk')
    text(d,(93,425),'Two service types, five categories.',29,GRAY)
    # Explicit synthetic notice; this is a data disclosure, not a real source.
    d.rounded_rectangle((1510*S,104*S,1910*S,158*S),radius=3*S,fill=INK)
    text(d,(1710,131),data['visible_notice'],22,'white','bold','mm')
    text(d,(1485,204),'Same root.',38,INK,'bold')
    text(d,(1485,257),'Same scale.',38,INK,'bold')
    text(d,(1487,322),'Percent (%) · linear 0–20',23,GRAY)

    records=[]
    origins=[(580,1035),(1190,850)]
    for series,name,color,origin in zip(data['series'],['blue','red'],[BLUE,RED],origins):
        fork,geo=fork_layer(series,data['categories'],name,color,origin)
        im.alpha_composite(fork)
        records.extend(geo)
    d=ImageDraw.Draw(im)
    # Direct series captions and short leaders tie the categories to the bodies.
    text(d,(100,1207),'TABLE',38,BLUE,'bold')
    text(d,(100,1257),'SERVICE',38,BLUE,'bold')
    text(d,(101,1318),'5 categories · %',22,GRAY)
    line(d,[(320,1240),(418,1240),(519,1175)],'#97A5B8',1.4)
    d.ellipse((515*S,1171*S,523*S,1179*S),fill=BLUE)
    text(d,(1540,784),'QUICK',38,RED,'bold')
    text(d,(1540,834),'SERVICE',38,RED,'bold')
    text(d,(1541,895),'5 categories · %',22,GRAY)
    line(d,[(1517,843),(1457,843),(1398,883)],'#BBA1A7',1.4)
    d.ellipse((1394*S,879*S,1402*S,887*S),fill=RED)

    text(d,(100,1468),'READ FROM ROOT TO TIP',19,GRAY,'bold')
    text(d,(100,1512),'Each tine encodes one value.',25,INK)
    text(d,(100,1552),'Zero keeps its place, with no tine.',25,INK)
    # Shared 400px ruler, equal to the maximum data length before rigid rotation.
    sx,sy=105,1650
    line(d,[(sx,sy),(sx+400,sy)],INK,1.5)
    for value in range(0,21,5):
        x=sx+PX_PER_PERCENT*value
        line(d,[(x,sy-7),(x,sy+7)],INK,1.4)
        text(d,(x,sy+18),f'{value}'+('%' if value==20 else ''),20,GRAY,anchor='mt')
    text(d,(sx+440,sy),'SHARED SCALE',17,GRAY,'bold','lm')
    line(d,[(90,1742),(1910,1742)],PALE,1)
    text(d,(90,1769),'All values are synthetic. Category order: Table A → Table E.',20,GRAY)
    text(d,(1910,1769),'Tine length ∝ percentage',20,GRAY,anchor='rt')

    final=im.convert('RGB').resize((brief['width'],brief['height']),Image.Resampling.LANCZOS)
    final.save(ROOT/'final.png')
    checks=pixel_geometry_check(final,records)
    artifact={'canvas_px':[brief['width'],brief['height']], 'pixels_per_percentage_point':PX_PER_PERCENT,
              'shared_full_scale_px':400,'rotation_degrees':-38,'origins_px':origins,
              'measurement_method':'Final RGB raster, quarter-pixel samples along each tine centerline from the zero root; first sustained non-material gap defines the visible tip.',
              'tines':checks,'all_lengths_within_tolerance':all(x['within_3_percent_full_scale'] for x in checks),
              'positive_tine_counts':[sum(v>0 for v in s['values']) for s in data['series']],
              'data_sha256':hashlib.sha256((ROOT/'data.json').read_bytes()).hexdigest()}
    (ROOT/'artifacts').mkdir(exist_ok=True)
    (ROOT/'artifacts'/'geometry_check.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'saved':'final.png','size':final.size,'length_checks_pass':artifact['all_lengths_within_tolerance']},ensure_ascii=False))

if __name__=='__main__':
    main()
