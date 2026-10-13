#!/usr/bin/env python3
"""Regenerate final.png offline from frozen data and locally saved image assets.
Complete executions must be launched with the supplied capture_render.py wrapper.
"""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parent

def load_json(rel):
    return json.loads((ROOT / rel).read_text(encoding='utf-8'))

spec = load_json('input.json')
brief = load_json(spec['brief'])
data = load_json(spec['data'])
core = load_json(spec['critical_metaphor_contract'])
assert (ROOT / spec['reference']).is_file(), 'Keep the supplied reference available.'
assert core['visible_title'] == data['title']
assert len(data['series']) == 1
series = data['series'][0]
values = np.asarray(series['values'], dtype=float)
times = np.asarray(data['time_points'], dtype=float)
categories = data['categories']
assert len(values) == len(times) == len(categories)
assert np.all(np.diff(times) > 0)
assert data['value_axis']['scale'] == 'linear'
W, H = brief['width'], brief['height']
SS = 2
PAPER = '#F6F2E8'
INK = '#24343C'
MUTED = '#626D6C'
RED = '#A74632'
GRID = '#DCDDD3'
CREAM_DOT = '#FFF9EB'
canvas = Image.new('RGBA', (W * SS, H * SS), PAPER)
draw = ImageDraw.Draw(canvas)

# All geometry below is in final output pixels. Only rendering is supersampled.
def xy(p):
    return tuple(float(v) * SS for v in p)

def poly(points, fill, outline=None, width=1):
    draw.polygon([xy(p) for p in points], fill=fill)
    if outline:
        line(points + [points[0]], outline, width)

def line(points, fill, width=1):
    draw.line([xy(p) for p in points], fill=fill, width=max(1,round(width * SS)), joint='curve')

def ellipse(box, fill, outline=None, width=1):
    draw.ellipse(xy(box), fill=fill, outline=outline, width=max(1,round(width * SS)))

def rect(box, fill, outline=None, width=1, radius=0):
    if radius:
        draw.rounded_rectangle(xy(box), radius=radius*SS, fill=fill, outline=outline, width=max(1,round(width*SS)))
    else:
        draw.rectangle(xy(box), fill=fill, outline=outline, width=max(1,round(width*SS)))

fonts = {}
def font(size, face='sans'):
    key=(size,face)
    if key not in fonts:
        names={'sans':'sans.otf','bold':'sans_bold.otf','serif':'serif_italic.otf'}
        fonts[key]=ImageFont.truetype(str(ROOT/'fonts'/names[face]), round(size*SS))
    return fonts[key]

text_boxes=[]
def text_at(text, x, y, size=24, fill=INK, face='sans', align='left', track=0, record=None):
    f=font(size,face)
    if track:
        widths=[draw.textlength(c,font=f) for c in text]
        tw=sum(widths)+(len(text)-1)*track*SS
    else:
        tw=draw.textlength(text,font=f)
    tx=x*SS-(tw/2 if align=='center' else tw if align=='right' else 0)
    # Use a top-left anchor so all declared text positions are actual ink top.
    if track:
        start=tx
        for c,cw in zip(text,widths):
            draw.text((tx,y*SS),c,font=f,fill=fill,anchor='lt')
            tx+=cw+track*SS
        bbox=(start/SS,y,(start+tw)/SS,y+size)
    else:
        draw.text((tx,y*SS),text,font=f,fill=fill,anchor='lt')
        b=draw.textbbox((tx,y*SS),text,font=f,anchor='lt')
        bbox=tuple(q/SS for q in b)
    if record:
        text_boxes.append({'id':record,'text':text,'bbox':bbox})
    return bbox

# Header: all wording and figures are sourced or derived from data.json.
line([(88,72),(1672,72)],INK,2)
rect((88,105,392,152),RED,radius=3)
text_at(data['visible_notice'],240,117,23,'#FFF9EB','bold','center',track=1.3,record='synthetic_notice')
text_at('ANNUAL HOUSING  /  USD',1672,120,20,MUTED,align='right',track=1.5)
text_at('SYNTHETIC',88,184,38,INK,'bold',track=2.0,record='title_synthetic')
text_at('House prices',82,230,115,INK,'bold',record='title_prices')
text_at(series['label']+' ('+series['unit']+')',90,366,30,INK,record='metric')
line([(1235,190),(1235,391)],'#CCCFC4',1.5)
text_at(categories[0].upper()+'  →  '+categories[-1].upper(),1280,210,20,MUTED,'bold',track=.6)
delta=int(values[-1]-values[0])
text_at(('+' if delta>=0 else '−')+'$'+f'{abs(delta):,}',1280,251,66,RED,'bold',record='change')
text_at(f'from ${int(values[0]):,} to ${int(values[-1]):,}',1280,343,23,MUTED)
line([(88,425),(1672,425)],INK,1.6)

# Single quantitative mapping. Perspective only affects the non-data parts.
axis_min=float(data['value_axis']['minimum'])
axis_max=float(data['value_axis']['maximum'])
X0, X1=400.0, 1510.0
Y0, YTOP=1490.0, 500.0
x=X0+(times-times[0])/(times[-1]-times[0])*(X1-X0)
y=Y0-(values-axis_min)/(axis_max-axis_min)*(Y0-YTOP)
ridge=np.c_[x,y]
shift=np.array([-126.,210.])
wall_height=260.
eave=ridge+shift
bottom=eave+np.array([0,wall_height])
foundation=bottom+np.array([24.,26.])

text_at('PRICE (USD)',88,465,19,MUTED,'bold',track=1.0)
for value in np.linspace(axis_min,axis_max,5):
    yy=Y0-(value-axis_min)/(axis_max-axis_min)*(Y0-YTOP)
    line([(212,yy),(1580,yy)],GRID,1.1)
    line([(203,yy),(216,yy)],'#939C94',1.3)
    text_at(f'{int(value):,}',188,yy-11,23,MUTED,align='right',record=f'axis_{int(value)}')
line([(211,YTOP),(211,Y0)],'#ABB2A7',1.15)
text_at('0–80,000 · linear scale',88,1523,19,MUTED)

# Ground and cast shadows establish the suspended-house relationship.
poly([(88,1724),(385,1715),(685,1723),(992,1714),(1325,1722),(1672,1716),
      (1672,1814),(88,1814)],'#CED4AB')
line([(88,1814),(1672,1814)],'#A8B18D',1)
# Deterministic sparse print flecks, restricted to the ground.
rng=np.random.default_rng(3301)
for px,py,r in zip(rng.uniform(92,1667,900),rng.uniform(1734,1810,900),rng.uniform(.3,.95,900)):
    ellipse((px-r,py-r,px+r,py+r),'#B9C198')
shadow=Image.new('RGBA',canvas.size,(0,0,0,0))
sd=ImageDraw.Draw(shadow)
sd.ellipse(xy((276,1723,1350,1782)),fill=(28,43,42,40))
sd.ellipse(xy((1290,1763,1601,1799)),fill=(28,43,42,37))
shadow=shadow.filter(ImageFilter.GaussianBlur(10*SS))
canvas.alpha_composite(shadow)
draw=ImageDraw.Draw(canvas)

# Piecewise texture warps implement the same profile as the roof coordinates.
# The original raster sheet contains no chart geometry; each surface is bounded
# by a code-defined polygon and every output pixel maps back into a local asset.
def warped_asset(asset, polygon, inverse, tint=None):
    global draw
    pts=np.asarray(polygon,dtype=float)
    l=max(0,math.floor(pts[:,0].min()-2)); r=min(W,math.ceil(pts[:,0].max()+2))
    t=max(0,math.floor(pts[:,1].min()-2)); b=min(H,math.ceil(pts[:,1].max()+2))
    gx=(np.arange((r-l)*SS,dtype=np.float32)+.5)/SS+l
    gy=(np.arange((b-t)*SS,dtype=np.float32)+.5)/SS+t
    xx,yy=np.meshgrid(gx,gy)
    uu,vv=inverse(xx,yy)
    src=np.asarray(Image.open(ROOT/asset).convert('RGBA'),dtype=np.float32)
    coords=np.array([np.clip(vv,0,1)*(src.shape[0]-1),np.clip(uu,0,1)*(src.shape[1]-1)])
    arr=np.stack([map_coordinates(src[:,:,k],coords,order=1,mode='nearest') for k in range(4)],axis=-1)
    if tint is not None:
        arr[:,:,:3]=arr[:,:,:3]*np.array(tint)
    layer=Image.fromarray(np.clip(arr,0,255).astype(np.uint8),'RGBA')
    mask=Image.new('L',layer.size,0)
    md=ImageDraw.Draw(mask)
    md.polygon([((a-l)*SS,(b0-t)*SS) for a,b0 in polygon],fill=255)
    layer.putalpha(mask)
    canvas.alpha_composite(layer,(l*SS,t*SS))
    draw=ImageDraw.Draw(canvas)

# Understructure behind the facade, with small material thickness (non-data).
poly([tuple(p) for p in bottom]+[tuple(p) for p in foundation[::-1]],'#34434A')
line([tuple(p) for p in foundation],INK,2.2)
back_bottom=ridge[-1]+[0,wall_height]
poly([tuple(bottom[-1]),tuple(back_bottom),tuple(back_bottom+[24,26]),tuple(foundation[-1])],'#23323B',INK,2)
for j in range(1,11):
    a=bottom[j-1]+[4,7]; b=bottom[j]+[4,7]
    line([tuple(a),tuple(b)],'#5A6567',1)

# End wall is deliberately a perspective face, outside the data plane.
right_quad=[tuple(eave[-1]),tuple(ridge[-1]),tuple(back_bottom),tuple(bottom[-1])]
def side_inverse(xx,yy):
    uu=(xx-eave[-1,0])/(-shift[0])
    top=eave[-1,1]-shift[1]*uu
    return uu,(yy-top)/wall_height
warped_asset('assets/side_wall.png',right_quad,side_inverse,tint=(.82,.94,1.04))
line(right_quad+[right_quad[0]],INK,3)

# Front wall has the generated door and three generated windows fully integrated.
front_poly=[tuple(p) for p in eave]+[tuple(p) for p in bottom[::-1]]
def front_inverse(xx,yy):
    yy_top=np.interp(xx,eave[:,0],eave[:,1])
    return (xx-eave[0,0])/(eave[-1,0]-eave[0,0]),(yy-yy_top)/wall_height
warped_asset('assets/facade.png',front_poly,front_inverse)
line([tuple(p) for p in bottom],INK,3.5)
line([tuple(eave[0]),tuple(bottom[0])],INK,3)
line([tuple(eave[-1]),tuple(bottom[-1])],INK,3)

# Roof transform: points (x(u), y(u)) + v * shift, exactly at v=0.
# Because y(u) is piecewise linear, this inverse is analytical on each segment.
roof_poly=[tuple(p) for p in ridge]+[tuple(p) for p in eave[::-1]]
def roof_inverse(xx,yy):
    out_u=np.zeros_like(xx); out_v=np.zeros_like(xx)
    best=np.full_like(xx,np.inf)
    for i in range(len(ridge)-1):
        xa,ya=ridge[i]; xb,yb=ridge[i+1]
        m=(yb-ya)/(xb-xa)
        v=(yy-ya-m*(xx-xa))/(shift[1]-m*shift[0])
        rx=xx-v*shift[0]
        u=(rx-X0)/(X1-X0)
        err=np.maximum(xa-rx,0)+np.maximum(rx-xb,0)
        choose=err<best
        out_u[choose]=u[choose]; out_v[choose]=v[choose]; best[choose]=err[choose]
    return out_u,out_v
warped_asset('assets/roof_tiles.png',roof_poly,roof_inverse)
# A slim light fascia sits strictly below the eave. It is not a second data line.
line([tuple(p+[0,4]) for p in eave],'#FBEDDB',5.0)
line([tuple(p) for p in eave],INK,2.5)
line([tuple(eave[0]),tuple(ridge[0])],INK,3.0)
line([tuple(ridge[-1]),tuple(eave[-1])],INK,3.0)
line([tuple(p) for p in ridge],INK,4.0)

# Data markers and direct labels, positioned from exactly the same mapping.
for i,((px,py),value,category) in enumerate(zip(ridge,values,categories)):
    line([(px,py-26),(px,py-9)],'#7D8982',1.2)
    text_at(category,px,py-83,22,MUTED,'sans','center',record=f'category_{i}')
    text_at(f'${int(value):,}',px,py-53,26,RED,'bold','center',record=f'value_{i}')
    ellipse((px-7,py-7,px+7,py+7),INK)
    ellipse((px-3.1,py-3.1,px+3.1,py+3.1),CREAM_DOT)

# Human scale from the generated group, anchored to the ground below the house.
family=Image.open(ROOT/'assets/family.png').convert('RGBA')
fh=374
fw=round(fh*family.width/family.height)
family=family.resize((fw*SS,fh*SS),Image.Resampling.LANCZOS)
canvas.alpha_composite(family,(1298*SS,(1786-fh)*SS))
draw=ImageDraw.Draw(canvas)

# A small editorial note occupies the clear space below the raised end.
text_at('A roof shaped by price.',905,1390,31,INK,'serif')
text_at('One vertex for each year.',905,1434,23,MUTED)

# Footer describes the reading convention without inferring affordability.
line([(88,1850),(1672,1850)],INK,1.4)
text_at('READING THE ROOF',88,1878,19,RED,'bold',track=1.4)
text_at('Each vertex is an annual price; height follows the USD scale at left.',399,1874,24,INK)
text_at('Years are equally spaced. House depth and the family are illustrative.',399,1915,23,MUTED)
text_at('11 ANNUAL VALUES  ·  SYNTHETIC DATA',88,1960,16,MUTED,'bold',track=.9)
text_at('LOCAL ILLUSTRATIVE SERIES',1672,1960,16,MUTED,align='right',track=.9)

# Save the code-controlled geometric record; the separate check script measures
# the final pixels rather than treating these coordinates as observations.
geometry={
    'canvas':[W,H], 'series_id':series['id'], 'unit':series['unit'],
    'mapping':{'x_min':X0,'x_max':X1,'time_min':float(times[0]),'time_max':float(times[-1]),
               'y_zero':Y0,'y_axis_top':YTOP,'value_min':axis_min,'value_max':axis_max,
               'pixels_per_usd':(Y0-YTOP)/(axis_max-axis_min)},
    'points':[{'category':c,'time':float(t),'value':int(v),'x':float(p[0]),'y':float(p[1])}
              for c,t,v,p in zip(categories,times,values,ridge)],
    'roof_polygon':[[float(q) for q in p] for p in np.r_[ridge,eave[::-1]]],
    'non_data_perspective':{'eave_shift':shift.tolist(),'wall_height':wall_height,'foundation_shift':[24,26]},
    'family_bbox':[1298,1786-fh,1298+fw,1786],
    'ground_y_at_family':1719,
    'lowest_house_y':float(foundation[:,1].max()),
    'text_boxes':text_boxes,
}
(ROOT/'artifacts').mkdir(exist_ok=True)
(ROOT/'artifacts/geometry.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
canvas.convert('RGB').resize((W,H),Image.Resampling.LANCZOS).save(ROOT/'final.png')
print(f'Wrote final.png: {W}x{H}; {len(values)} exact-price roof vertices; local assets only.')

if __name__ == '__main__':
    pass
