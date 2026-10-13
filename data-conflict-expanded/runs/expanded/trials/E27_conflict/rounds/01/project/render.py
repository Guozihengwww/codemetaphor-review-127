#!/usr/bin/env python3
"""Offline renderer. Run complete chart builds only through capture_render.py."""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
S = 2
BG = '#fcf9ef'
INK = '#202d27'
GREEN = '#00874f'
MUTED = '#62695b'

def load(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))

def main():
    entry = load('input.json')
    brief, data, core = load(entry['brief']), load(entry['data']), load(entry['critical_metaphor_contract'])
    assert data['synthetic'] is True
    assert data['value_axis'] == {'minimum': 0, 'maximum': 10, 'scale': 'linear'}
    assert core['visible_title'] == data['title']
    width, height = brief['width'], brief['height']
    values = data['series'][0]['values']
    years = data['time_points']
    categories = data['categories']
    assert len(values) == len(years) == len(categories)
    # The only mapping from statistical values to canvas coordinates.
    x0, x1, ytop, ybase = 190.0, 1450.0, 500.0, 1430.0
    xmin, xmax = min(years), max(years)
    vmin, vmax = data['value_axis']['minimum'], data['value_axis']['maximum']
    def xy(t, v):
        return (x0 + (t-xmin)/(xmax-xmin)*(x1-x0), ybase-(v-vmin)/(vmax-vmin)*(ybase-ytop))
    pts = [xy(t,v) for t,v in zip(years,values)]
    # Ivory-to-amber sky, pale enough to preserve both grid and black lattice details.
    yy = np.arange(height, dtype=np.float32)[:,None]
    xx = np.arange(width, dtype=np.float32)[None,:]
    strength = np.clip((yy-330)/1180,0,1)**1.7
    glow = np.clip(0.76+0.24*xx/width,0,1)
    a = strength * glow
    top = np.array([252,249,239],np.float32)
    bottom = np.array([232,190,85],np.float32)
    pixels = np.clip(top[None,None,:]*(1-a[:,:,None])+bottom[None,None,:]*a[:,:,None],0,255).astype('uint8')
    canvas = Image.fromarray(pixels).convert('RGBA').resize((width*S,height*S),Image.Resampling.BICUBIC)
    d = ImageDraw.Draw(canvas)
    fonts={}
    labels=[]
    def font(size,bold=False):
        key=(size,bold)
        if key not in fonts:
            fonts[key]=ImageFont.truetype(str(ROOT/'assets/fonts'/('NimbusSans-Bold.otf' if bold else 'NimbusSans-Regular.otf')),round(size*S))
        return fonts[key]
    def text(content,pos,size=28,fill=INK,bold=False,anchor='la',tag=None):
        p=tuple(round(z*S) for z in pos)
        f=font(size,bold)
        d.text(p,str(content),font=f,fill=fill,anchor=anchor)
        if tag:
            box=d.textbbox(p,str(content),font=f,anchor=anchor)
            labels.append({'tag':tag,'text':str(content),'bbox_px':[round(z/S,2) for z in box]})
    def line(points,fill=INK,w=1):
        d.line([(round(x*S),round(y*S)) for x,y in points],fill=fill,width=max(1,round(w*S)),joint='curve')
    def circle(c,r,fill,outline=None,w=1):
        x,y=c
        d.ellipse(((x-r)*S,(y-r)*S,(x+r)*S,(y+r)*S),fill=fill,outline=outline,width=round(w*S))
    def rect(box,fill,r=0,outline=None,w=1):
        coords=tuple(round(v*S) for v in box)
        if r:d.rounded_rectangle(coords,radius=round(r*S),fill=fill,outline=outline,width=round(w*S))
        else:d.rectangle(coords,fill=fill,outline=outline,width=round(w*S))
    def bezier(p0,p1,p2,p3,n=260):
        return [tuple((1-t)**3*np.array(p0)+3*(1-t)**2*t*np.array(p1)+3*(1-t)*t*t*np.array(p2)+t**3*np.array(p3)) for t in np.linspace(0,1,n)]
    def spaced(content,pos,size,spacing,fill=INK,bold=False):
        x,y=pos
        for char in content:
            text(char,(x,y),size,fill,bold)
            x+=d.textlength(char,font=font(size,bold))/S+spacing

    # Header: synthetic status is never relegated to a footnote.
    spaced('ENERGY / ECONOMIC OUTPUT',(96,73),23,2.7,MUTED,True)
    rect((1575,70,1904,123),GREEN,r=26)
    text(data['visible_notice'],(1739.5,84),25,'#ffffff',True,anchor='ma',tag='synthetic_notice')
    text('SYNTHETIC',(95,132),35,GREEN,True,tag='title_prefix')
    text('Energy intensity',(89,179),112,INK,True,tag='title_main')
    text('Energy consumed per dollar of economic output',(98,312),34,INK,tag='metric_description')
    line([(97,383),(1902,383)],'#cfceba',1.4)

    # Shared linear axis; the eight years retain their actual temporal distances.
    text(data['series'][0]['unit'],(190,444),27,INK,True,tag='unit')
    text('0–10 linear scale',(570,447),23,MUTED,tag='scale_description')
    for v in range(0,11,2):
        y=xy(xmin,v)[1]
        line([(x0-10,y),(x1,y)],'#ac9e75' if v==0 else '#dbcfaa',1.45 if v==0 else 1.05)
        text(str(v),(157,y-14),27,MUTED,anchor='ra',tag=f'axis_{v}')
    text('YEAR',(92,1473),22,MUTED,True,tag='year_axis')
    for i,(x,y) in enumerate(pts):
        line([(x,ybase),(x,ybase+11)],'#756f56',1.5)
        if i<6:
            text(categories[i],(x,ybase+29),27,INK,anchor='ma',tag=f'year_{categories[i]}')
    # The last two years are one year apart; leaders separate typography only.
    x09,x10=pts[-2][0],pts[-1][0]
    line([(x09,ybase+11),(x09,ybase+26),(1390,ybase+46)],'#756f56',1.2)
    text(categories[-2],(1381,ybase+40),27,INK,anchor='ra',tag=f'year_{categories[-2]}')
    line([(x10,ybase+11),(x10,ybase+73),(1493,ybase+73)],'#756f56',1.2)
    text(categories[-1],(1503,ybase+60),27,INK,tag=f'year_{categories[-1]}')

    # An explicitly labelled derived comparison, separated from data labels.
    spaced('2010 RELATIVE TO 1950',(223,560),20,2.0,MUTED,True)
    text(f'{values[-1]/values[0]:.2f}×',(216,600),97,GREEN,True,tag='derived_ratio')
    text('the energy per dollar of output',(223,712),27,INK,tag='derived_ratio_description')
    text(f'{values[-1]:.2f} ÷ {values[0]:.2f}',(224,754),23,MUTED,tag='derived_ratio_formula')

    # Generated pylon: its upper-left insulator is positioned at the final point.
    # Raw landmark measured on the generated image, not inferred from chart data.
    crop_left,crop_top = 158,11
    raw_anchor=(221.5,320.0)
    raw_base=1225.0
    tower_scale=(1450-pts[-1][1])/(raw_base-raw_anchor[1])
    tower_left=pts[-1][0]-(raw_anchor[0]-crop_left)*tower_scale
    tower_top=pts[-1][1]-(raw_anchor[1]-crop_top)*tower_scale
    def tower_point(raw):
        return (tower_left+(raw[0]-crop_left)*tower_scale,tower_top+(raw[1]-crop_top)*tower_scale)
    terminals={k:tower_point(v) for k,v in {
      'upper_left':(221.5,320),'middle_left':(169.5,535),'lower_left':(169.7,759),
      'upper_right':(699.5,320),'middle_right':(751,536),'lower_right':(750.5,759)}.items()}
    # Non-data overhead conductors visibly terminate on the pylon hardware.
    wire1=bezier((-45,1337),(380,1450),(940,1345),terminals['middle_left'])
    wire2=bezier((-45,1442),(455,1560),(1010,1456),terminals['lower_left'])
    wire3=bezier((-45,1452),(455,1570),(1010,1466),(terminals['lower_left'][0]+4,terminals['lower_left'][1]+4))
    for wire in (wire1,wire2,wire3):line(wire,'#45473a',2.2)
    for i,key in enumerate(('upper_right','middle_right','lower_right')):
        p=terminals[key]
        path=bezier(p,(p[0]+80,p[1]+28),(1980,p[1]-12),(2055,p[1]-44))
        line(path,'#45473a',2.2)
    incoming=bezier((-50,1090),(12,1137),(99,1217),pts[0],n=150)
    line(incoming,'#45473a',2.2)
    tower=Image.open(ROOT/'assets/tower.png').convert('RGBA')
    tower=tower.resize((round(tower.width*tower_scale*S),round(tower.height*tower_scale*S)),Image.Resampling.LANCZOS)
    canvas.alpha_composite(tower,(round(tower_left*S),round(tower_top*S)))
    d=ImageDraw.Draw(canvas)

    # Green geometry is code-only; no generated raster defines the data path.
    line(pts,GREEN,10)
    for p in pts:
        circle(p,7,GREEN)
        circle(p,2.2,'#fcf7df')

    # Affine bird placement keeps the feet in contact while leaving the body upright.
    # A very small shear only adjusts the foot plane to the cable slope.
    parts=load('assets/parts.json')
    bird_placements=[]
    def bird(name,width_px,foot,slope):
        source=Image.open(ROOT/f'assets/{name}.png').convert('RGBA')
        box=parts['crops'][name]
        raw_foot=parts['bird_foot_raw'][name]
        fx,fy=raw_foot[0]-box[0],raw_foot[1]-box[1]
        scale=width_px/source.width
        # y' = y + slope * (x - foot_x), preserving the bird's vertical axis.
        corners=[(0,0),(source.width,0),(0,source.height),(source.width,source.height)]
        corners_out=[(x*scale,(y+slope*(x-fx))*scale) for x,y in corners]
        min_y=min(y for x,y in corners_out)
        max_y=max(y for x,y in corners_out)
        out_w=math.ceil(source.width*scale*S)+4
        out_h=math.ceil((max_y-min_y)*S)+4
        # Inverse affine map from output to source pixels.
        bird_im=source.transform((out_w,out_h),Image.Transform.AFFINE,
            (1/(scale*S),0,0,-slope/(scale*S),1/(scale*S),min_y/scale+slope*fx),
            resample=Image.Resampling.BICUBIC)
        local_foot=(fx*scale*S,(fy*scale-min_y)*S)
        dest=(round(foot[0]*S-local_foot[0]),round(foot[1]*S-local_foot[1]))
        canvas.alpha_composite(bird_im,dest)
        bird_placements.append({'asset':f'assets/{name}.png','foot_px':list(foot),'slope':slope,'size_px':[out_w/S,out_h/S]})
    def point_on_segment(i,t):
        a,b=pts[i],pts[i+1]
        return (a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1])),(b[1]-a[1])/(b[0]-a[0])
    foot,slope=point_on_segment(0,.53);bird('bird_01',91,foot,slope)
    foot,slope=point_on_segment(2,.38);bird('bird_02',94,foot,slope)
    foot,slope=point_on_segment(4,.48);bird('bird_03',89,foot,slope)
    idx=175
    foot=wire1[idx];slope=(wire1[idx+1][1]-wire1[idx-1][1])/(wire1[idx+1][0]-wire1[idx-1][0])
    bird('bird_04',101,foot,slope)
    d=ImageDraw.Draw(canvas)

    # Mandatory numerical labels use original input precision.
    positions=[(190,1191,'ma'),(414,1205,'ma'),(591,1125,'ma'),(841,1063,'ma'),
               (1007,922,'ma'),(1231,793,'ma'),(1347,745,'ra'),(1510,675,'la')]
    for i,(x,y,anchor) in enumerate(positions):
        text(f'{values[i]:.2f}',(x,y),38,GREEN,True,anchor=anchor,tag=f'value_{categories[i]}')
    # Leaders are typography connectors, thinner and lighter than the green path.
    line([(pts[-2][0]-8,pts[-2][1]-8),(1386,768),(1359,768)],'#69815a',1.5)
    line([(pts[-1][0]+11,pts[-1][1]),(1490,695),(1500,695)],'#69815a',1.5)

    # Footer is out of the plotted scene and contains the exact series role.
    rect((0,1556,width,height),BG)
    line([(96,1556),(1904,1556)],'#c9c7ad',1.2)
    line([(100,1608),(161,1608)],GREEN,8)
    circle((130,1608),5,GREEN)
    text(data['series'][0]['label'],(183,1589),28,INK,True,tag='series')
    text('Synthetic study values · 1950–2010',(1904,1593),25,MUTED,anchor='ra',tag='scope')
    text('Only the green line encodes data. Towers, birds and dark cables are illustrative.',(98,1640),25,MUTED,tag='illustration_note')
    text('Years use their actual spacing. Values shown in BTU per GDP dollar.',(98,1684),25,MUTED,tag='spacing_note')

    final=canvas.resize((width,height),Image.Resampling.LANCZOS).convert('RGB')
    final.save(ROOT/'final.png',optimize=True)
    evidence={'canvas':[width,height],'plot_bounds_px':[x0,ytop,x1,ybase],
      'mapping':{'time_range':[xmin,xmax],'value_range':[vmin,vmax],'pixels_per_year':(x1-x0)/(xmax-xmin),'pixels_per_value':(ybase-ytop)/(vmax-vmin)},
      'points':[{'year':c,'value':v,'xy':list(p)} for c,v,p in zip(categories,values,pts)],
      'labels':labels,'data_line_width_px':10,'marker_radius_px':7,
      'tower':{'bbox_px':[tower_left,tower_top,tower_left+tower.width/S,tower_top+tower.height/S],
               'scale':tower_scale,'terminals_px':{k:list(v) for k,v in terminals.items()}},
      'birds':bird_placements,'generated_assets':['assets/tower.png']+[b['asset'] for b in bird_placements]}
    (ROOT/'artifacts/layout.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':
    main()
