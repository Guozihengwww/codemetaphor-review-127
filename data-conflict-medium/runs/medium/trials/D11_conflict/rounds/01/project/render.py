#!/usr/bin/env python3
"""Rebuild the infographic from frozen local data and the generated hand asset.
Run only through capture_render.py for experimental complete rendering attempts.
No model calls, network access, or writes to frozen inputs.
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.interpolate import PchipInterpolator, CubicHermiteSpline

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'assets'
OUT = ROOT / 'artifacts'
OUT.mkdir(exist_ok=True)
data = json.loads((ROOT / 'data.json').read_text(encoding='utf-8'))
brief = json.loads((ROOT / 'brief.json').read_text(encoding='utf-8'))
W, H = brief['width'], brief['height']
S = 2
X0, X1 = 234.0, 1354.0
YT, YB = 460.0, 1420.0
A0, A1 = float(data['value_axis']['minimum']), float(data['value_axis']['maximum'])
times = np.array(data['time_points'], dtype=float)
values = np.array(data['series'][0]['values'], dtype=float)
assert len(times) == len(values) == len(data['categories'])
assert np.all(np.diff(times) > 0)

def x_of(v):
    return X0 + (np.asarray(v)-A0)/(A1-A0)*(X1-X0)

def y_of(t):
    return YB - (np.asarray(t)-times[0])/(times[-1]-times[0])*(YB-YT)

node_y = y_of(times)[::-1]
node_x = x_of(values)[::-1]
pchip = PchipInterpolator(node_y, node_x)
slopes = pchip.derivative()(node_y)
# A vertical tangent at either end joins the unmeasured tip/palm smoothly.
# Zeroing these endpoint slopes preserves interval monotonicity and no overshoot.
slopes[0] = slopes[-1] = 0.0
curve = CubicHermiteSpline(node_y, node_x, slopes)
dcurve = curve.derivative()

def half_width(y):
    # Non-data width: approximately constant normal thickness, expressed in
    # horizontal cross-sections to preserve the required midpoint encoding.
    radius = 51.0 + 3.0*(np.asarray(y)-YT)/(YB-YT)
    return radius * np.sqrt(1.0 + dcurve(y)**2)

paper = (246, 244, 234)
ink = (48, 51, 45)
muted = (104, 110, 100)
purple = (109, 48, 86)
grid = (221, 225, 212)
grid_major = (201, 211, 193)
canvas = Image.new('RGB', (W*S, H*S), paper)
d = ImageDraw.Draw(canvas)

def line(points, fill, width=1):
    d.line([(round(x*S), round(y*S)) for x,y in points], fill=fill, width=max(1,round(width*S)))

def rect(box, fill, outline=None, width=1, radius=0):
    b = tuple(round(z*S) for z in box)
    if radius:
        d.rounded_rectangle(b, radius=round(radius*S), fill=fill, outline=outline, width=round(width*S))
    else:
        d.rectangle(b, fill=fill, outline=outline, width=round(width*S))

fontfiles={'serif':'serif.ttc','sans':'sans.ttc','num':'numbers.ttf','bold':'numbers_bold.ttf'}
fonts={}
def font(kind, size):
    key=(kind,size)
    if key not in fonts:
        fonts[key]=ImageFont.truetype(str(ASSETS/'fonts'/fontfiles[kind]), round(size*S))
    return fonts[key]

def text(x,y,content,size=26,kind='sans',fill=ink,anchor='lt'):
    d.text((round(x*S),round(y*S)),str(content),font=font(kind,size),fill=fill,anchor=anchor)

def tracking(x,y,content,size=18,spacing=2,fill=purple):
    for c in content:
        text(x,y,c,size,'bold',fill)
        x += d.textlength(c,font=font('bold',size))/S + spacing
    return x

# Subtle full-page squared paper, with a protected editorial header/footer.
for x in np.arange(X0 % 20, W, 20):
    line([(x,52),(x,H-52)], grid, .55)
for y in np.arange(YT % 20, H-52, 20):
    if y >= 52:
        line([(66,y),(W-66,y)], grid, .55)
for x in np.arange(X0 % 80, W, 80):
    line([(x,52),(x,H-52)], grid_major, .7)
for y in np.arange(YT % 80, H-52, 80):
    if y >= 52:
        line([(66,y),(W-66,y)], grid_major, .7)
rect((65,51,W-65,255),paper)
rect((65,1886,W-65,1955),paper)
line([(65,51),(W-65,51),(W-65,H-51),(65,H-51),(65,51)], (183,192,177),1)

# Header: exact title, series identity, and prominent synthetic notice.
rect((110,80,415,119), purple, radius=3)
tracking(127,90,data['visible_notice'],17,2,(250,245,235))
text(W-110,92,'虚构数据 · 可视化示例',21,'sans',muted,'rt')
text(110,136,data['title'],66,'serif',ink)
text(114,218,'左利手比例  /  '+data['series'][0]['label'],25,'sans',muted)
text(W-110,151,f'{data["categories"][0]}—{data["categories"][-1]}',32,'serif',ink,'rt')
text(W-110,207,f'{len(values)} 个年份 · 单位 %',22,'sans',muted,'rt')

# Shared horizontal linear rate axis and vertical time axis.
text(X0,266,'比例 / RATE (%)',22,'sans',ink)
line([(X0,320),(X1,320)],ink,1.5)
for v in np.arange(A0,A1+.01,1):
    x=float(x_of(v))
    length=12 if v%2==0 else 6
    line([(x,320),(x,320+length)],ink,1.3)
    if v%2==0:
        text(x,309,f'{v:g}',25,'num',ink,'mb')
text(118,390,'年份',24,'sans',ink)
text(118,425,'YEAR',17,'bold',muted)
line([(X0,YT),(X0,YB)],ink,1.5)
for category,t in zip(data['categories'],times):
    y=float(y_of(t))
    line([(X0,y),(X1,y)],(171,186,161),1.05)
    line([(X0-9,y),(X0,y)],ink,1.5)
    # Paper backing keeps labels clear over the graph-paper substrate.
    rect((110,y-23,215,y+22),paper)
    text(205,y,category,29,'num',ink,'rm')

# Warp the generated raster. Within [YT,YB], the alpha-contour midpoint of
# every horizontal row is mapped to curve(y); its width carries no quantity.
source=np.array(Image.open(ASSETS/'hand_clean.png').convert('RGBA'))
sh,sw=source.shape[:2]
left=np.full(sh,np.nan); right=np.full(sh,np.nan)
for sy in range(sh):
    xs=np.flatnonzero(source[sy,:,3]>=128)
    if xs.size:
        left[sy],right[sy]=xs[0],xs[-1]
valid=np.flatnonzero(np.isfinite(left))
source_rows=np.arange(sh)
left=np.interp(source_rows,valid,left[valid])
right=np.interp(source_rows,valid,right[valid])
SRC_TOP,SRC_BOTTOM=135.0,1010.0
TIP_SY,PALM_SY=1.13,.92
lt=float(np.interp(SRC_TOP,source_rows,left)); rt=float(np.interp(SRC_TOP,source_rows,right))
lb=float(np.interp(SRC_BOTTOM,source_rows,left)); rb=float(np.interp(SRC_BOTTOM,source_rows,right))
top_sx=float(2*half_width(YT)/(rt-lt))
bot_sx=float(2*half_width(YB)/(rb-lb))
hand=np.zeros((H*S,W*S,4),dtype=np.uint8)
dest_x=(np.arange(W*S)+.5)/S
for dy in range(max(0,round((YT-150)*S)),min(H*S,round((YB+465)*S))):
    y=(dy+.5)/S
    if y<YT:
        sy=SRC_TOP+(y-YT)/TIP_SY
        center=float(node_x[0]); sx=top_sx; src_center=(lt+rt)/2
    elif y<=YB:
        sy=SRC_TOP+(y-YT)/(YB-YT)*(SRC_BOTTOM-SRC_TOP)
        l=float(np.interp(sy,source_rows,left)); r=float(np.interp(sy,source_rows,right))
        center=float(curve(y)); sx=float(2*half_width(y)/(r-l)); src_center=(l+r)/2
    else:
        sy=SRC_BOTTOM+(y-YB)/PALM_SY
        center=float(node_x[-1]); sx=bot_sx; src_center=(lb+rb)/2
    if not 0<=sy<sh-1:
        continue
    sx_all=src_center+(dest_x-center)/sx
    chosen=(sx_all>=0)&(sx_all<sw-1)
    xx=sx_all[chosen]
    ix=xx.astype(int); fx=(xx-ix)[:,None]
    iy=int(sy); fy=sy-iy
    a=source[iy,ix]*(1-fx)+source[iy,ix+1]*fx
    b=source[iy+1,ix]*(1-fx)+source[iy+1,ix+1]*fx
    hand[dy,chosen]=np.round(a*(1-fy)+b*fy).clip(0,255).astype(np.uint8)
hand_image=Image.fromarray(hand)
# Keep a native-size, visible-alpha mask for independent pixel measurements.
alpha_image=Image.fromarray(hand[:,:,3]).resize((W,H),Image.Resampling.LANCZOS)
alpha_image.save(OUT/'hand_alpha.png')
canvas.paste(hand_image,(0,0),hand_image)

# Data-bearing cross-sections and exact labels. No independent plotted curve.
labels=[]
for category,t,v in zip(data['categories'],times,values):
    y=float(y_of(t)); x=float(x_of(v)); hw=float(half_width(y))
    l,r=x-hw,x+hw
    line([(l+3,y),(r-3,y)],purple,1.8)
    for edge in [l+3,r-3]:
        line([(edge,y-4),(edge,y+4)],purple,1.5)
    d.ellipse(((x-7)*S,(y-7)*S,(x+7)*S,(y+7)*S),fill=paper)
    d.ellipse(((x-4.6)*S,(y-4.6)*S,(x+4.6)*S,(y+4.6)*S),fill=purple)
    line([(r+5,y),(r+18,y)],purple,1.3)
    label=f'{v:g}%'
    tx=r+25
    tw=d.textlength(label,font=font('num',29))/S
    rect((tx-7,y-21,tx+tw+7,y+22),paper,radius=3)
    text(tx,y,label,29,'num',purple,'lm')
    labels.append({'category':category,'value':float(v),'y':y,'expected_midpoint_x':x,'designed_left':l,'designed_right':r,'label':label,'label_box':[tx-7,y-21,tx+tw+7,y+22]})

# Reading key in the open area below the measured span.
rect((111,1540,823,1777),paper,radius=5)
text(131,1557,'读指身的中点',34,'serif',ink)
# The miniature section is a legend for the marks used on the actual finger.
line([(136,1630),(314,1630)],purple,2)
for xx in [136,314]:
    line([(xx,1622),(xx,1638)],purple,2)
d.ellipse((219*S,1624*S,231*S,1636*S),fill=purple)
text(338,1613,'横截面中点 = 当年比例',24,'sans',ink)
text(132,1666,'指宽和甲片不表示数值。',24,'sans',muted)
text(132,1710,'年份由下向上；相邻刻度间隔 20 年。',24,'sans',muted)

line([(111,1908),(W-111,1908)],(166,177,157),1)
text(112,1927,'数据为合成示例，不代表真实历史统计。',21,'sans',muted)
text(W-112,1927,'0–14%  ·  线性比例尺',21,'sans',muted,'rt')
canvas=canvas.resize((W,H),Image.Resampling.LANCZOS)
canvas.save(ROOT/'final.png')

# Evidence calculated from the rasterized alpha silhouette, not target echoes.
a=np.array(alpha_image)
measurements=[]
for item in labels:
    iy=round(item['y'])
    xs=np.flatnonzero(a[iy]>=128)
    ml,mr=int(xs[0]),int(xs[-1]); mid=(ml+mr)/2
    measured_value=A0+(mid-X0)/(X1-X0)*(A1-A0)
    measurements.append({**item,'measured_visible_left_px':ml,'measured_visible_right_px':mr,'measured_midpoint_px':mid,'measured_value_percent':round(measured_value,6),'absolute_error_percentage_points':round(abs(measured_value-item['value']),6)})
max_error=0
observed_range=[]
for iy in range(round(YT),round(YB)+1):
    xs=np.flatnonzero(a[iy]>=128)
    if len(xs):
        mid=(int(xs[0])+int(xs[-1]))/2
        max_error=max(max_error,abs(mid-float(curve(iy))))
for i in range(len(node_y)-1):
    yy=np.linspace(node_y[i],node_y[i+1],1001)
    xx=curve(yy)
    observed_range.append({'year_interval':[str(data['categories'][::-1][i]),str(data['categories'][::-1][i+1])], 'no_overshoot':bool(xx.min()>=min(node_x[i:i+2])-1e-8 and xx.max()<=max(node_x[i:i+2])+1e-8)})
geometry={'canvas':[W,H],'value_axis':[A0,A1],'horizontal_axis_px':[X0,X1], 'year_span_px':[YB,YT],'measurements':measurements,'continuous_max_midpoint_error_px':round(max_error,4),'continuous_max_midpoint_error_percentage_points':round(max_error/(X1-X0)*(A1-A0),6),'interpolation_segments':observed_range,'generated_hand_visible_pixels':int((a>=128).sum()),'observed_hand_is_connected':None,'tip_and_palm_source_rows':[SRC_TOP,SRC_BOTTOM]}
(OUT/'geometry.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'output':'final.png','size':[W,H],'max_geometry_error_pp':geometry['continuous_max_midpoint_error_percentage_points']},ensure_ascii=False))
