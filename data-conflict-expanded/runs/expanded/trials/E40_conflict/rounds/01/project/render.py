#!/usr/bin/env python3
"""Offline, data-bound cabaret illustration. All coordinates derive from data.json."""
from pathlib import Path
import json, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.ndimage import map_coordinates

ROOT=Path(__file__).resolve().parent
D=json.loads((ROOT/'data.json').read_text(encoding='utf-8'))
B=json.loads((ROOT/'brief.json').read_text(encoding='utf-8'))
C=json.loads((ROOT/'core.json').read_text(encoding='utf-8'))
I=json.loads((ROOT/'input.json').read_text(encoding='utf-8'))
W,H=B['width'],B['height']; S=2
BG=(246,242,233); INK=(35,34,32); MUTED=(105,102,94); CORAL=(214,57,45)
PAPER=(255,251,242); GRID=(206,200,189)
FONT=ROOT/'assets/fonts'

def font(size,bold=False,italic=False):
    name='NimbusRoman-Italic.otf' if italic else ('NimbusSans-Bold.otf' if bold else 'NimbusSans-Regular.otf')
    return ImageFont.truetype(str(FONT/name),round(size*S))

canvas=Image.new('RGBA',(W*S,H*S),BG+(255,))
draw=ImageDraw.Draw(canvas)
def line(points,fill=INK,width=1):
    draw.line([(round(x*S),round(y*S)) for x,y in points],fill=fill,width=round(width*S),joint='curve')
def rect(box,fill,outline=None,width=1,radius=0):
    xy=tuple(round(v*S) for v in box)
    if radius: draw.rounded_rectangle(xy,round(radius*S),fill=fill,outline=outline,width=round(width*S))
    else: draw.rectangle(xy,fill=fill,outline=outline,width=round(width*S))
def ellipse(box,fill,outline=None,width=1):
    draw.ellipse(tuple(round(v*S) for v in box),fill=fill,outline=outline,width=round(width*S))
def text(x,y,t,size=28,fill=INK,bold=False,italic=False,anchor='la'):
    draw.text((round(x*S),round(y*S)),t,font=font(size,bold,italic),fill=fill,anchor=anchor)
def paste(im,x,y,w=None,h=None):
    if w is None: w=im.width
    if h is None: h=im.height
    im=im.resize((round(w*S),round(h*S)),Image.Resampling.LANCZOS)
    canvas.alpha_composite(im,(round(x*S),round(y*S)))

# One mapping for every marker, line vertex, year tick, label attachment and limb warp.
times=np.asarray(D['time_points'],float)
vals=np.asarray(D['series'][0]['values'],float)
vmin,vmax=float(D['value_axis']['minimum']),float(D['value_axis']['maximum'])
xmin,xmax=705.,1785.; ytop,ybottom=600.,1600.
xs=xmin+(times-times[0])/(times[-1]-times[0])*(xmax-xmin)
ys=ybottom-(vals-vmin)/(vmax-vmin)*(ybottom-ytop)
pts=np.column_stack([xs,ys])

# Deliberately restrained poster typography; synthetic status is immediately visible.
rect((110,66,420,112),CORAL,radius=4)
text(132,77,D['visible_notice'],24,PAPER,True)
text(110,143,'One carat',94,INK,True)
text(110,243,'diamond prices',94,INK,True)
text(115,365,D['series'][0]['label']+'  /  '+D['series'][0]['unit'],29,MUTED)
line([(115,435),(859,435)],INK,2)
text(1470,79,'PRICE / PERFORMANCE',23,INK,True)
text(1470,119,'1978—1982',38,INK,italic=True)
text(1470,176,'Five years, one carat.',25,MUTED)
# Small geometrical diamond, a semantic marker for the measured object.
line([(1821,88),(1847,88),(1861,103),(1834,131),(1807,103),(1821,88)],CORAL,2)
line([(1807,103),(1861,103)],CORAL,1.5)
line([(1821,88),(1825,103),(1834,131),(1843,103),(1847,88)],CORAL,1.5)

# The full requested 10,000–65,000 domain is explicit, with quiet horizontal rules.
for v in [10000,20000,30000,40000,50000,60000,65000]:
    yy=ybottom-(v-vmin)/(vmax-vmin)*(ybottom-ytop)
    line([(649,yy),(1849,yy)],GRID,1)
    text(1866,yy,f'{v:,}',23,MUTED,anchor='lm')
text(1864,558,'USD',20,MUTED,True)
# Category guides have the same true-year x spacing as the curve.
for xx,yy in pts:
    for g in np.arange(yy+24,1651,15):
        line([(xx,g),(xx,min(g+4,1651))],(195,188,176),1)
line([(648,1654),(1853,1654)],INK,1.5)
for xx,cat in zip(xs,D['categories']):
    line([(xx,1647),(xx,1663)],INK,2)
    text(xx,1682,cat,34,INK,True,anchor='ma')
text(1867,1693,'YEAR',18,MUTED,True)

# Data-carrying foreground stocking: tapered continuous limb, not an independent chart.
# Its first four skeleton points use the SAME mapping as the quantitative path.
leg_pts=pts[:4].copy()
radii=np.array([88.,94.,64.,44.])
mask=Image.new('L',canvas.size,0);md=ImageDraw.Draw(mask)
for j in range(3):
    a,b=leg_pts[j:j+2]; u=b-a; u=u/np.linalg.norm(u); n=np.array([-u[1],u[0]])
    poly=[a+n*radii[j],b+n*radii[j+1],b-n*radii[j+1],a-n*radii[j]]
    md.polygon([tuple(q*S) for q in poly],fill=255)
for p,r in zip(leg_pts,radii):
    md.ellipse(tuple(np.r_[p-r,p+r]*S),fill=255)
# Remove the rounded proximal cap outside the body. The body hides this attachment.
# Material coordinates follow each limb segment, so mesh narrows with the anatomy.
bbox=(int(leg_pts[:,0].min()-110),int(leg_pts[:,1].min()-90),int(leg_pts[:,0].max()+90),int(leg_pts[:,1].max()+110))
x0,y0,x1,y1=bbox
Y,X=np.mgrid[y0:y1,x0:x1].astype(float)
best=np.full(X.shape,np.inf); U=np.zeros(X.shape); V=np.zeros(X.shape)
swatch=np.asarray(Image.open(ROOT/'assets/stocking_swatch.png').convert('RGBA'))
sw,sh=swatch.shape[1],swatch.shape[0]; cum=0.
for j in range(3):
    a,b=leg_pts[j:j+2]; seg=b-a; length=np.linalg.norm(seg); unit=seg/length
    t=np.clip(((X-a[0])*seg[0]+(Y-a[1])*seg[1])/(length*length),0,1)
    px=a[0]+t*seg[0];py=a[1]+t*seg[1]
    radius=radii[j]+t*(radii[j+1]-radii[j])
    distance=((X-px)**2+(Y-py)**2)/(radius*radius)
    q=distance<best
    normal=(X-px)*(-unit[1])+(Y-py)*unit[0]
    U[q]=np.clip((normal[q]/radius[q]+1)/2*(sw-1),1,sw-2)
    V[q]=((cum+t[q]*length)*0.69)%(sh-3)+1
    best[q]=distance[q]; cum+=length
sample=np.stack([map_coordinates(swatch[:,:,k],[V,U],order=1,mode='nearest') for k in range(4)],-1).astype(np.uint8)
tex=Image.fromarray(sample,'RGBA').resize(((x1-x0)*S,(y1-y0)*S),Image.Resampling.BICUBIC)
layer=Image.new('RGBA',canvas.size,(0,0,0,0));layer.alpha_composite(tex,(x0*S,y0*S))
layer.putalpha(mask)
# Ink outline at the cutout edge, with a small warm highlight on the inside.
outline=mask.filter(ImageFilter.MaxFilter(9))
inklayer=Image.new('RGBA',canvas.size,INK+(255,));inklayer.putalpha(outline)
canvas.alpha_composite(inklayer);canvas.alpha_composite(layer)

# Generated torso, two arms, hat, rear fishnet leg and rear high heel are one assembly.
body=Image.open(ROOT/'assets/performer.png').convert('RGBA')
body_scale=1.1
# Original source hip landmark (540,1084), with the crop origin restored here.
body_x=pts[0,0]-(540-30)*body_scale
body_y=pts[0,1]-(1084-31)*body_scale
paste(body,body_x,body_y,body.width*body_scale,body.height*body_scale)

# Warp the generated pump, preserving its heel and material while flattening the
# instep onto the low-slope 1981–1982 segment. The source map is an object warp only.
shoe=np.asarray(Image.open(ROOT/'assets/shoe.png').convert('RGBA'))
# Source curve follows the solid material through heel body, instep and toe.
src_x=np.array([0,30,50,70,100,140,180,220,270,300,335],float)
src_y=np.array([109,103,96,99,162,224,257,264,272,278,284],float)
x_scale=(pts[4,0]-pts[3,0])/(270.-70.)
y_scale=.94
sxleft=pts[3,0]-70*x_scale
shoe_box=(math.floor(sxleft)-2,math.floor(pts[3,1]-135),math.ceil(sxleft+shoe.shape[1]*x_scale)+2,math.ceil(pts[3,1]+230))
a0,b0,a1,b1=shoe_box
TY,TX=np.mgrid[b0:b1,a0:a1].astype(float)
SX=(TX-sxleft)/x_scale
center=np.interp(SX,src_x,src_y)
target_line=pts[3,1]+(TX-pts[3,0])*(pts[4,1]-pts[3,1])/(pts[4,0]-pts[3,0])
SY=(TY-target_line)/y_scale+center
warped=np.stack([map_coordinates(shoe[:,:,k],[SY,SX],order=1,mode='constant',cval=0) for k in range(4)],-1).astype(np.uint8)
paste(Image.fromarray(warped,'RGBA'),a0,b0)

# Draw the quantitative line LAST, in exact, non-overshooting piecewise-linear form.
draw=ImageDraw.Draw(canvas)
line(pts.tolist(),CORAL,14)
for xx,yy in pts:
    ellipse((xx-17,yy-17,xx+17,yy+17),PAPER)
    ellipse((xx-10.5,yy-10.5,xx+10.5,yy+10.5),CORAL)

# Value labels are all code-bound to data. A pale plate protects them from mesh.
def value_plate(x,y,i,align='center'):
    label=f'${int(vals[i]):,}'
    f=font(31,True);bb=draw.textbbox((0,0),label,font=f)
    tw=(bb[2]-bb[0])/S
    if align=='left': left=x-10
    elif align=='right': left=x-tw-10
    else:left=x-tw/2-10
    rect((left,y-5,left+tw+20,y+36),BG,radius=4)
    text(left+10,y,label,31,INK,True)
# Proximal and middle-thigh values sit outside the line, with clear attachments.
line([tuple(pts[0]+[0,22]),(pts[0,0],1596)],CORAL,1.5)
value_plate(pts[0,0],1602,0)
line([tuple(pts[1]+[-18,-7]),(892,1252)],CORAL,1.5)
value_plate(884,1210,1,align='right')
# The peak callout sits in clean space, away from the rear leg.
line([tuple(pts[2]+[18,-10]),(1340,590),(1420,590)],CORAL,1.5)
text(1435,550,'1980 / PEAK',21,CORAL,True)
text(1430,581,f'${int(vals[2]):,}',48,INK,True)
# Low-slope final segment: both distinct dots and both exact values remain visible.
line([tuple(pts[3]+[-11,19]),(1437,1438)],CORAL,1.5)
value_plate(1434,1452,3)
line([tuple(pts[4]+[0,-22]),(1785,1291)],CORAL,1.5)
value_plate(1785,1240,4)

# Closing band explains the encoding and the synthetic status without obscuring art.
line([(110,1770),(1890,1770)],INK,2)
text(110,1811,'A curve with a stage presence.',36,INK,italic=True)
text(110,1864,'Illustrative values only. Not historical market prices.',24,MUTED)
line([(1182,1818),(1244,1818)],CORAL,7)
ellipse((1204,1808,1224,1828),PAPER)
ellipse((1209,1813,1219,1823),CORAL)
text(1260,1801,'Dot height = fictional price',26,INK,True)
text(1182,1847,'Linear scale · $10,000–$65,000',24,MUTED)
text(1182,1880,'USD per one-carat diamond',24,MUTED)

final=canvas.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
final.save(ROOT/'final.png')
geometry={'canvas':[W,H],'axis':{'x_range':[xmin,xmax],'y_pixels':[ytop,ybottom],'value_domain':[vmin,vmax],'scale':'linear'},'points':[{'category':cat,'value':int(v),'center':[float(x),float(y)]} for cat,v,(x,y) in zip(D['categories'],vals,pts)],'line':'piecewise linear, no interpolation overshoot','marker_inner_color':list(CORAL),'marker_inner_radius':10.5,'body_transform':{'scale':body_scale,'xy':[float(body_x),float(body_y)]},'shoe_source_to_data_anchors':{'source':[70,270],'targets':[pts[3].tolist(),pts[4].tolist()]}}
(ROOT/'artifacts').mkdir(exist_ok=True)
(ROOT/'artifacts/render_geometry.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
