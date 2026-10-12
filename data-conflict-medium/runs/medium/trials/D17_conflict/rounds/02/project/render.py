"""Render the supplied data with generated local object parts. Offline, no arguments.

All numerical geometry is controlled here. The generated non-data roll heads are
identical in dimensions. The white hanging strips begin at a common baseline;
their mean serrated endpoints follow one linear 0–100 mapping.
"""
from pathlib import Path
import json
import math
from decimal import Decimal
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / 'data.json').read_text(encoding='utf-8'))
brief = json.loads((ROOT / 'brief.json').read_text(encoding='utf-8'))
W, H = brief['width'], brief['height']
S = 4
CANVAS = Image.new('RGB', (W*S,H*S), 'white')
draw = ImageDraw.Draw(CANVAS)
INK = '#263c3b'
MUTED = '#687875'
FAINT = '#e8edeb'
TEAL = '#247c79'
BODY_EDGE = '#465e58'
BASELINE = 547.0
AXIS_BOTTOM = 1757.0
AXIS_X = 115.0
AXIS_RIGHT = 1925.0
HEAD_W, HEAD_H = 147.0, 116.0
HEAD_X0, STEP = 182.0, 175.5
SHEET_OFFSET = (301.0-168.0)/681.0*HEAD_W
SHEET_W = (819.0-301.0)/681.0*HEAD_W
AXIS_MIN = float(data['value_axis']['minimum'])
AXIS_MAX = float(data['value_axis']['maximum'])
PPU = (AXIS_BOTTOM-BASELINE)/(AXIS_MAX-AXIS_MIN)
assert data['value_axis']['scale'] == 'linear' and AXIS_MIN == 0
assert len(data['series']) == 1
series = data['series'][0]
assert len(series['values']) == len(data['categories']) == 10

font_cache = {}
def font(size, bold=False, latin=False):
    key = (size,bold,latin)
    if key not in font_cache:
        path = ('DejaVuSans' if latin else 'NotoSansCJK') + ('-Bold' if bold else '-Regular')
        if latin:
            path = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
        else:
            path += '.ttc'
        font_cache[key] = ImageFont.truetype(str(ROOT/'assets/fonts'/path),round(size*S),index=0 if latin else 2)
    return font_cache[key]

def text(x,y,value,size=24,fill=INK,bold=False,latin=False,anchor='lt'):
    draw.text((round(x*S),round(y*S)),str(value),font=font(size,bold,latin),fill=fill,anchor=anchor)

def line(points, fill, width=1):
    draw.line([(round(x*S),round(y*S)) for x,y in points],fill=fill,width=max(1,round(width*S)))

def exact_number(v):
    return format(Decimal(str(v)), 'f')

def dashed_horizontal(x1,x2,y,fill,width=1.2,dash=6,gap=5):
    x=x1
    while x<x2:
        line([(x,y),(min(x+dash,x2),y)],fill,width)
        x += dash+gap

def serrated_edge(x, y, width, amplitude, teeth=10):
    # Four equal segments per tooth. Each complete tooth has signed area zero.
    points=[]
    for k in range(teeth*4+1):
        shift = (0,-1,0,1)[k % 4] * amplitude
        points.append((x+width*k/(teeth*4), y+shift))
    return points

# Editorial title and mandatory data notice.
draw.rounded_rectangle((100*S,70*S,368*S,109*S),radius=6*S,fill='#e7f1ef')
text(116,79,data['visible_notice'],21,TEAL,True,True)
text(1920,80,'ANNUAL USE  /  10 GROUPS',21,MUTED,False,True,'rt')
text(97,144,data['title'],76,INK,True)
text(100,245,series['label'],30,INK,True,True)
text(310,247,'·  '+series['unit'],29,MUTED,False,True)
line([(100,294),(1920,294)],'#dbe3df',1.2)

# Shared, downward increasing linear ruler and quiet guides.
text(96,BASELINE,'0',25,MUTED,False,True,'rm')
line([(AXIS_X,BASELINE),(AXIS_X,AXIS_BOTTOM)],'#afbfba',1.2)
line([(AXIS_X,BASELINE),(AXIS_RIGHT,BASELINE)],'#9aaca5',1.2)
for v in range(20,101,20):
    y = BASELINE+(v-AXIS_MIN)*PPU
    line([(AXIS_X,y),(AXIS_RIGHT,y)],FAINT,0.8)
    line([(AXIS_X-7,y),(AXIS_X+6,y)],'#a5b6ae',1.1)
    text(96,y,str(v),24,MUTED,False,True,'rm')

texture = Image.open(ROOT / 'assets/paper_texture.png').convert('RGB')
records=[]
for i,(category,value) in enumerate(zip(data['categories'],series['values'])):
    head_x = HEAD_X0+STEP*i
    x = head_x+SHEET_OFFSET
    c = x+SHEET_W/2
    length = (float(value)-AXIS_MIN)*PPU
    endpoint = BASELINE+length
    amplitude = min(2.7,length*0.18)
    edge = serrated_edge(x,endpoint,SHEET_W,amplitude)
    # Separate local mask permits fractional lengths and symmetric tear endpoints.
    left_px = math.floor(x*S)
    right_px = math.ceil((x+SHEET_W)*S)
    top_px = round(BASELINE*S)
    bottom_px = math.ceil((endpoint+amplitude+1)*S)
    pw,ph = right_px-left_px,bottom_px-top_px
    # Coverage mask with fractional endpoints: no inclusive raster-polygon bias.
    x_centers=(np.arange(pw)+left_px+0.5)/S
    edge_y=np.interp(x_centers,[p[0] for p in edge],[p[1] for p in edge])
    row_top=(np.arange(ph)+top_px)/S
    cov=np.clip((edge_y[None,:]-row_top[:,None])*S,0,1)
    x_left=(np.arange(pw)+left_px)/S
    x_cov=np.clip((np.minimum(x_left+1/S,x+SHEET_W)-np.maximum(x_left,x))*S,0,1)
    mask=Image.fromarray(np.round(cov*x_cov[None,:]*255).astype('uint8'))
    sheet = texture.resize((pw,ph),Image.Resampling.LANCZOS)
    # Subtle edge light falloff, applied to the generated paper surface.
    arr=np.asarray(sheet).astype(float)
    u=np.linspace(0,1,pw)
    shading=1-0.022*np.exp(-u*22)-0.016*np.exp(-(1-u)*22)
    arr=np.minimum(255,arr*shading[None,:,None])
    sheet=Image.fromarray(arr.astype('uint8'))
    CANVAS.paste(sheet,(left_px,top_px),mask)

    # Side outlines and a real serrated lower boundary; no minimum data length.
    edge_weight = min(1.25, max(0.5,length*0.17))
    line([(x,BASELINE),(x,endpoint)],BODY_EDGE,edge_weight)
    line([(x+SHEET_W,BASELINE),(x+SHEET_W,endpoint)],BODY_EDGE,edge_weight)
    line(edge,BODY_EDGE,edge_weight)

    # Perforations are visual material detail, never count-based data symbols.
    perforations=[]
    py=BASELINE+88
    while py<endpoint-21:
        dashed_horizontal(x+3,x+SHEET_W-3,py,'#a3aaa5',1.25,6,5)
        perforations.append(py)
        py+=88

    head=Image.open(ROOT/f'assets/head_{i+1:02d}.png').convert('RGBA')
    head=head.resize((round(HEAD_W*S),round(HEAD_H*S)),Image.Resampling.LANCZOS)
    CANVAS.paste(head,(round(head_x*S),round((BASELINE-HEAD_H)*S)),head)
    # This same non-data perforation appears on every curled head, including tiny values.
    dashed_horizontal(x+3,x+SHEET_W-2,BASELINE-27,'#9ba39d',1.35,6,5)

    text(c,326,category,25,INK,True,True,'mt')
    text(c,368,exact_number(value),43,INK,False,True,'mt')

    # A local high-resolution diagnostic alpha measurement of each physical strip.
    mask_arr=np.asarray(mask,dtype=float)/255
    mid=mask_arr[:,4:-4]
    area_mean_length=float(mid.sum(axis=0).mean()/S)
    records.append({'category':category,'value_label':exact_number(value),
                    'head_bbox_px':[head_x,BASELINE-HEAD_H,head_x+HEAD_W,BASELINE],
                    'strip_x_px':[x,x+SHEET_W],'zero_y_px':BASELINE,
                    'encoded_length_px':length,'mean_endpoint_y_px':endpoint,
                    'serration_half_amplitude_px':amplitude,
                    'raster_mask_mean_length_px':area_mean_length,
                    'mask_raster_error_px':area_mean_length-length,
                    'perforation_y_px':perforations})

# Explanatory text occupies unused space without altering the data silhouette.
line([(1120,1143),(1180,1143)],TEAL,3)
text(1120,1177,'长度如何读',30,INK,True)
text(1120,1234,'从共同零线向下量取，',26,MUTED)
text(1120,1278,'到锯齿撕口的平均线为止。',26,MUTED)
text(1120,1349,'卷头等大；彩色端面仅区分类别。',24,MUTED)
text(1120,1390,'小值保留真实比例，不设最小长度。',24,MUTED)

# Footer provenance and concise encoding disclosure.
line([(100,1829),(1920,1829)],'#dbe3df',1.2)
text(100,1866,'合成数据 · 仅用于视觉编码研究',24,MUTED)
text(1920,1867,'0–100  /  LINEAR SCALE',23,MUTED,False,True,'rt')
text(100,1910,'纸带长度编码年度使用量；卷头、纹样与穿孔数量不编码。',23,MUTED)
text(1920,1910,'Source: supplied data.json',21,MUTED,False,True,'rt')

CANVAS.resize((W,H),Image.Resampling.LANCZOS).save(ROOT/'final.png')
(ROOT/'artifacts').mkdir(exist_ok=True)
geometry={'canvas':[W,H],'supersampling':S,'mapping':{'minimum':AXIS_MIN,'maximum':AXIS_MAX,
          'baseline_y_px':BASELINE,'maximum_y_px':AXIS_BOTTOM,'pixels_per_unit':PPU},
          'data_strip_measurement':'Alpha area per interior column at render resolution; excludes decorative heads.',
          'records':records}
(ROOT/'artifacts/geometry.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Rendered final.png from local data and generated assets:',W,H)
