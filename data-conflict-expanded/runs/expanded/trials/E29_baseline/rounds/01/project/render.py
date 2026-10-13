#!/usr/bin/env python3
"""Offline, data-driven rendering of the frozen hand/cohort infographic."""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'assets'
ART = ROOT / 'artifacts'
ART.mkdir(exist_ok=True)

def read_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))

data = read_json('data.json')
brief = read_json('brief.json')
core = read_json('core.json')
inputs = read_json('input.json')
assert all((ROOT / inputs[k]).is_file() for k in ('brief','data','reference','critical_metaphor_contract'))
assert data['synthetic'] and data['series'][0]['unit'] == 'percent'
W, H = brief['width'], brief['height']
assert (W, H) == (1560, 2000)
S = 2
PAPER = '#F7F4EC'
INK = '#342F32'
MUTED = '#69646A'
PLUM = '#743A61'
GRID = '#DEE1D9'
GRID_MAJOR = '#CDD3CB'

im = Image.new('RGBA', (W*S, H*S), PAPER)
draw = ImageDraw.Draw(im)
fonts = {}
def font(name, size):
    key = (name, size)
    if key not in fonts:
        files = {'serif':'serif_regular.otf','italic':'serif_italic.otf','sans':'sans_regular.ttf','bold':'sans_bold.ttf'}
        fonts[key] = ImageFont.truetype(str(ASSETS/'fonts'/files[name]), int(round(size*S)))
    return fonts[key]

def text(x, y, txt, name='sans', size=26, fill=INK, anchor='lt'):
    draw.text((round(x*S),round(y*S)), txt, font=font(name,size), fill=fill, anchor=anchor)

def line(points, fill, width=1):
    draw.line([(round(x*S),round(y*S)) for x,y in points], fill=fill, width=max(1,round(width*S)))

def rect(box, fill, radius=0, outline=None, width=1):
    box = tuple(round(v*S) for v in box)
    if radius:
        draw.rounded_rectangle(box, radius=round(radius*S), fill=fill, outline=outline, width=max(1,round(width*S)))
    else:
        draw.rectangle(box, fill=fill, outline=outline, width=max(1,round(width*S)))

# Editorial heading. All quantitative and semantic copy is code controlled.
rect((110,58,376,101), PLUM, radius=6)
text(129,68,data['visible_notice'],'bold',21,'#FFF9F1')
text(1450, seventy:=73, '1880–2000  /  BIRTH COHORTS','sans',20,MUTED,anchor='rt')
text(106,126,'Left handedness','serif',94)
text(108,224,'by birth year','serif',94)
text(111,340,'Synthetic left-handedness rate · percent of each birth cohort','sans',26,MUTED)
line([(110,387),(1450,387)],'#B5B8AD',1.2)

# A single linear coordinate system; ascending birth years are higher on paper.
years = np.asarray(data['time_points'], dtype=float)
values = np.asarray(data['series'][0]['values'], dtype=float)
assert len(years) == len(values) == len(data['categories'])
assert np.all(np.diff(years)>0)
x0, x1 = 270.0, 1320.0
y_top, y_bottom = 640.0, 1510.0
vmin, vmax = (float(data['value_axis'][k]) for k in ('minimum','maximum'))
xscale = (x1-x0)/(vmax-vmin)
yscale = (y_bottom-y_top)/(years[-1]-years[0])
def xx(v): return x0+(np.asarray(v)-vmin)*xscale
def yy(t): return y_bottom-(np.asarray(t)-years[0])*yscale
curve = PchipInterpolator(years,values,extrapolate=False)

# Graph paper stays visible around the generated hand.
for n in range(-8,64):
    x = x0+n*(xscale/4)
    if 110 <= x <= 1450:
        line([(x,480),(x,1884)], GRID_MAJOR if n%8==0 else GRID, .8 if n%8==0 else .5)
for n in range(-15,105):
    y = y_top+n*(yscale*2)
    if 480<=y<=1884:
        line([(110,y),(1450,y)],GRID,.6)
for t in years:
    y = float(yy(t))
    line([(x0,y),(1442,y)],'#C3CCC2',1.1)
line([(x0,480),(x1,480)],'#777C75',1.6)
line([(x0,480),(x0,y_bottom+12)],'#777C75',1.6)
text((x0+x1)/2,414,'LEFT-HANDEDNESS RATE (%)','bold',20,MUTED,anchor='mt')
for v in np.arange(vmin,vmax+.01,1):
    x = float(xx(v))
    line([(x,480),(x,470 if int(v)%2==0 else 475)],'#777C75',1.3)
    if int(v)%2==0:
        text(x,441,f'{v:g}','sans',26,INK,anchor='mt')
text(231,563,'BIRTH YEAR','bold',17,MUTED,anchor='rt')
text(231,589,'COHORT','bold',17,MUTED,anchor='rt')
for cat,t in zip(data['categories'],years):
    y=float(yy(t))
    text(234,y,cat,'sans',29,INK,anchor='rm')
    line([(x0-8,y),(x0,y)],'#777C75',1.4)

# Map generated skin and contours into a code-defined silhouette, scanline by
# scanline. Only the center is quantitative; the horizontal width is constant.
source = np.asarray(Image.open(ASSETS/'hand_source.png').convert('RGBA'),dtype=np.float32)
sh,sw = source.shape[:2]
left = np.full(sh,np.nan)
right = np.full(sh,np.nan)
for r in range(sh):
    inds=np.flatnonzero(source[r,:,3]>128)
    if len(inds): left[r],right[r] = inds[0],inds[-1]
valid=np.flatnonzero(np.isfinite(left))
left=np.interp(np.arange(sh),valid,left[valid])
right=np.interp(np.arange(sh),valid,right[valid])
body_start,body_end=170.0,995.0
width=110.0
source_top_center=(left[170]+right[170])/2
source_bottom_center=(left[995]+right[995])/2
top_scale=width/(right[170]-left[170]+1)
bottom_scale=width/(right[995]-left[995]+1)
cap_y_scale=.90
palm_y_scale=.70
bottom_slope = float(curve.derivative()(years[0]))*(-1/yscale)*xscale
extension=50.0

def centers_at(Y):
    T=years[0]+(y_bottom-np.clip(Y,y_top,y_bottom))/yscale
    C=np.asarray(xx(curve(T)))
    s=np.clip(Y-y_bottom,0,extension)
    C += bottom_slope*(s-s*s/(2*extension))
    return C

box=(425,480,1310,1885)
bw,bh=(box[2]-box[0])*S,(box[3]-box[1])*S
X=box[0]+(np.arange(bw)+.5)/S
Y=box[1]+(np.arange(bh)+.5)/S
C=centers_at(Y)
qY=np.where(Y<y_top,body_start+(Y-y_top)/cap_y_scale,
      np.where(Y<=y_bottom,body_start+(Y-y_top)*(body_end-body_start)/(y_bottom-y_top),
               body_end+(Y-y_bottom)/palm_y_scale))
L=np.interp(qY,np.arange(sh),left)
R=np.interp(qY,np.arange(sh),right)
qX=L[:,None]-.5+(X[None,:]-(C[:,None]-width/2))*(R-L+1)[:,None]/width
upper=Y<y_top
lower=Y>y_bottom
qX[upper,:]=source_top_center+(X[None,:]-C[upper,None])/top_scale
qX[lower,:]=source_bottom_center+(X[None,:]-C[lower,None])/bottom_scale
qY2=np.broadcast_to(qY[:,None],qX.shape)
coords=np.array([qY2,qX])
rgba=np.empty((bh,bw,4),dtype=np.float32)
for ch in range(4):
    rgba[:,:,ch]=map_coordinates(source[:,:,ch],coords,order=1,mode='constant',cval=0,prefilter=False)
body=(Y>=y_top)&(Y<=y_bottom)
coverage=np.clip((width/2-np.abs(X[None,:]-C[:,None]))*S+.5,0,1)
rgba[body,:,3]=coverage[body,:]*255
hand=Image.fromarray(np.clip(rgba,0,255).astype(np.uint8),'RGBA')
im.alpha_composite(hand,(box[0]*S,box[1]*S))
# Diagnostic mask, not a second chart render.
mask=Image.new('L',im.size,0)
mask.paste(hand.getchannel('A'),(box[0]*S,box[1]*S))
mask.resize((W,H),Image.Resampling.LANCZOS).save(ART/'object_mask.png')

# Precisely bound observed midpoints, leaders and value labels.
for cat,t,v in zip(data['categories'],years,values):
    x,y=float(xx(v)),float(yy(t))
    # Small midpoint marker; neither its radius nor finger width encodes data.
    rr=7.0
    draw.ellipse(((x-rr)*S,(y-rr)*S,(x+rr)*S,(y+rr)*S),fill=PAPER,outline=PLUM,width=round(2*S))
    draw.ellipse(((x-2.5)*S,(y-2.5)*S,(x+2.5)*S,(y+2.5)*S),fill=PLUM)
    # Horizontal leader starts at the contour, not a displaced numeric anchor.
    line([(x+width/2+3,y),(x+width/2+31,y)],PLUM,1.7)
    lx=x+width/2+42
    label=f'{v:.1f}%'
    bbox=draw.textbbox((0,0),label,font=font('bold',30))
    label_w=(bbox[2]-bbox[0])/S
    rect((lx-9,y-23,lx+label_w+9,y+23),PAPER,radius=6)
    text(lx,y,label,'bold',30,PLUM,anchor='lm')

# A restrained insight in the natural negative space of the hand chart.
rect((347,749,955,924),PAPER)
line([(354,770),(418,770)],PLUM,3)
text(352,792,'A dip, then a rise.','italic',43,PLUM)
min_i=int(np.argmin(values))
text(354,854,f'{data["categories"][min_i]} is the lowest cohort in','sans',25,MUTED)
text(354,892,f'this synthetic sequence: {values[min_i]:.1f}%.','sans',25,MUTED)

# Non-data hand/palm remains below the earliest observed cross-section.
rect((1017,1627,1450,1838),PAPER)
text(1030,1643,'READ THE MIDPOINT','bold',19,PLUM)
for i,s in enumerate(['At each birth year, the center','of the finger reads the rate.','Width and nail are illustrative.']):
    text(1030,1686+i*39,s,'sans',23,MUTED)
line([(110,1914),(1450,1914)],'#B5B8AD',1.1)
text(111,1941,'SYNTHETIC DATA  ·  Illustrative cohorts, not observed statistics.','sans',23,MUTED)

final = im.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
final.save(ROOT/'final.png')

# Record raster measurements, not only requested coordinates.
mask_arr=np.asarray(Image.open(ART/'object_mask.png'))
rendered=np.asarray(final).astype(np.int16)
checks=[]
for cat,t,v in zip(data['categories'],years,values):
    x,y=float(xx(v)),float(yy(t))
    row=int(round(y))
    # The local cross-section excludes all text, the leader and background.
    a,b=int(np.floor(x-width)),int(np.ceil(x+width))
    ind=np.flatnonzero(mask_arr[row,a:b]>=128)+a
    xl,xr=float(ind[0])-.5,float(ind[-1])+.5
    midpoint=(xl+xr)/2
    pix=rendered[row,a:b,:]
    warm=(pix[:,0]>pix[:,1]+15)&(pix[:,1]>pix[:,2]+12)
    vis=np.flatnonzero(warm)+a
    vx_l,vx_r=float(vis[0])-.5,float(vis[-1])+.5
    visible_midpoint=(vx_l+vx_r)/2
    checks.append({'cohort':cat,'value_percent':float(v),'center_target_px':[x,y],
        'mask_contour_x_px':[xl,xr],'mask_midpoint_px':midpoint,
        'mask_inferred_percent':vmin+(midpoint-x0)/xscale,
        'visible_warm_contour_x_px':[vx_l,vx_r],
        'visible_midpoint_px':visible_midpoint,
        'visible_inferred_percent':vmin+(visible_midpoint-x0)/xscale,
        'visible_error_percentage_points':abs(vmin+(visible_midpoint-x0)/xscale-v)})
interpolation=[]
for i in range(len(years)-1):
    vals=curve(np.linspace(years[i],years[i+1],201))
    lo,hi=min(values[i:i+2]),max(values[i:i+2])
    interpolation.append({'birth_year_interval':[float(years[i]),float(years[i+1])],
        'sampled_min_percent':float(vals.min()),'sampled_max_percent':float(vals.max()),
        'overshoot_percent':float(max(0,lo-vals.min(),vals.max()-hi))})
measurement={'canvas':[W,H],'axis':{'x_origin_px':x0,'x_end_px':x1,
  'minimum_percent':vmin,'maximum_percent':vmax,'pixels_per_percentage_point':xscale,
  'birth_year_pixels_per_year':yscale,'2000_y':y_top,'1880_y':y_bottom},
  'cross_sections':checks,'interpolation_checks':interpolation,
  'generated_asset':'assets/hand_source.png','generated_raw':'assets/raw_01.png',
  'observed_region_y_px':[y_top,y_bottom],
  'source_body_rows':[body_start,body_end],
  'fingertip_above_observed_region':True,'palm_below_observed_region':True,
  'font_sources':'assets/fonts/','method':'Generated raster scanline remap; PCHIP midpoints; constant horizontal width; local raster contour measurements.'}
(ART/'geometry_check.json').write_text(json.dumps(measurement,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Wrote final.png; measured',len(checks),'visible cross-sections.')
