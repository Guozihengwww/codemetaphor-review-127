#!/usr/bin/env python3
"""Rebuild the hand-chart from frozen data and local raster material. No network."""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import PchipInterpolator, CubicHermiteSpline
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parent
S = 2
PAPER = (247, 245, 236)
INK = (50, 47, 45)
MUTED = (111, 112, 104)
PLUM = (110, 44, 82)


def save_json(path, obj):
    (ROOT / path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def font(name, size):
    return ImageFont.truetype(str(ROOT / 'assets/fonts' / name), int(size*S))


def main():
    data = json.loads((ROOT/'data.json').read_text(encoding='utf-8'))
    brief = json.loads((ROOT/'brief.json').read_text(encoding='utf-8'))
    W, H = brief['width'], brief['height']
    assert (W, H) == (1560, 2000)
    series = data['series'][0]
    values = np.asarray(series['values'], dtype=float)
    times = np.asarray(data['time_points'], dtype=float)
    assert len(values) == len(times) == len(data['categories'])
    assert np.all(np.diff(times)>0)
    vmin, vmax = data['value_axis']['minimum'], data['value_axis']['maximum']
    x0, x1 = 240.0, 1360.0
    ytop, ybottom = 510.0, 1410.0
    px_per_unit = (x1-x0)/(vmax-vmin)
    data_x = x0+(values-vmin)*px_per_unit
    data_y = ybottom-(times-times[0])/(times[-1]-times[0])*(ybottom-ytop)
    # One monotone piecewise cubic centerline, with no inter-node overshoot.
    center = PchipInterpolator(data_y[::-1], data_x[::-1])
    palm_join_y = 1490.0
    palm_center = float(data_x[0]+52)
    lower = CubicHermiteSpline([ybottom, palm_join_y],
                               [data_x[0], palm_center],
                               [float(center.derivative()(ybottom)), 0])
    width = PchipInterpolator([510,660,810,960,1110,1260,1410,1490],
                              [140,145,153,190,240,172,176,140.4])
    # Material-space registration: all parts come from the same generated hand.
    source_y_map = PchipInterpolator([375,510,955,1248,1490],
                                    [40,160,410,690,905])
    raw = np.asarray(Image.open(ROOT/'assets/hand_clean.png').convert('RGBA'))
    edge_rows = json.loads((ROOT/'assets/hand_rows.json').read_text(encoding='utf-8'))['rows']
    src_left = np.asarray([r[0] if r[0] is not None else np.nan for r in edge_rows])
    src_right = np.asarray([r[1] if r[1] is not None else np.nan for r in edge_rows])
    valid = np.isfinite(src_left)
    src_left = np.interp(np.arange(len(src_left)), np.flatnonzero(valid), src_left[valid])
    src_right = np.interp(np.arange(len(src_right)), np.flatnonzero(valid), src_right[valid])
    # A dense inverse warp. In the observed area, both actual source contour
    # edges are mapped symmetrically about the data-derived center.
    bx0, bx1, by0, by1 = 310, 1375, 372, 1844
    yy = np.arange(by0*S, by1*S, dtype=np.float32)/S
    xx = np.arange(bx0*S, bx1*S, dtype=np.float32)/S
    sy = np.empty_like(yy)
    cx = np.empty_like(yy)
    sx_center = np.empty_like(yy)
    scale_x = np.empty_like(yy)
    cap = yy < ytop
    body = (yy >= ytop) & (yy <= palm_join_y)
    palm = yy > palm_join_y
    sy[cap | body] = source_y_map(yy[cap | body])
    cx[cap] = data_x[-1]
    sx_center[cap] = 551
    scale_x[cap] = 1.4
    obs = body & (yy <= ybottom)
    connection = body & (yy > ybottom)
    cx[obs] = center(yy[obs])
    cx[connection] = lower(yy[connection])
    left = np.interp(sy[body], np.arange(len(src_left)), src_left)
    right = np.interp(sy[body], np.arange(len(src_right)), src_right)
    sx_center[body] = (left+right)/2
    scale_x[body] = width(yy[body])/(right-left)
    sy[palm] = 905+(yy[palm]-palm_join_y)/0.72
    cx[palm] = palm_center
    sx_center[palm] = 556
    scale_x[palm] = 1.3
    source_x = (xx[None,:]-cx[:,None])/scale_x[:,None]+sx_center[:,None]
    source_y = np.broadcast_to(sy[:,None], source_x.shape)
    # Premultiplication prevents dark transparent-pixel halos at the silhouette.
    src = raw.astype(np.float32)
    src[:,:,:3] *= src[:,:,3:4]/255
    coords = np.stack([source_y, source_x])
    sampled = np.empty(source_x.shape+(4,), dtype=np.float32)
    for channel in range(4):
        sampled[:,:,channel] = map_coordinates(src[:,:,channel], coords, order=1, mode='constant', cval=0)
    sampled[:,:,:3] = sampled[:,:,:3]*255/np.maximum(sampled[:,:,3:4], 1e-5)
    patch = Image.fromarray(np.clip(sampled,0,255).astype(np.uint8), 'RGBA')
    hand = Image.new('RGBA', (W*S,H*S), (0,0,0,0))
    hand.paste(patch, (bx0*S,by0*S))
    hand_native = hand.resize((W,H), Image.Resampling.LANCZOS)
    hand_native.save(ROOT/'assets/hand_warped.png')

    # Warm graph paper; texture is deterministic and unobtrusive.
    rng = np.random.default_rng(314159)
    noise = rng.normal(0, .43, (H,W,1))
    bg = np.clip(np.asarray(PAPER)[None,None,:]+noise,0,255).astype(np.uint8)
    canvas = Image.fromarray(bg, 'RGB').resize((W*S,H*S), Image.Resampling.BICUBIC).convert('RGBA')
    draw = ImageDraw.Draw(canvas)
    def line(points, fill, w=1):
        draw.line([(int(x*S),int(y*S)) for x,y in points], fill=fill, width=max(1,int(w*S)))
    def rect(box, fill, outline=None, w=1):
        draw.rectangle(tuple(int(z*S) for z in box), fill=fill, outline=outline, width=max(1,int(w*S)))
    def text(x,y,content,size=26,color=INK,face='NotoSansSC.otf',anchor='lt'):
        draw.text((int(x*S),int(y*S)),content,font=font(face,size),fill=color,anchor=anchor)
    def dot(x,y,r,fill,outline=None,w=1):
        draw.ellipse((int((x-r)*S),int((y-r)*S),int((x+r)*S),int((y+r)*S)),fill=fill,outline=outline,width=int(w*S))
    def dash(xa,xb,y,fill,w=1,on=6,off=10):
        for a in np.arange(xa,xb,on+off):
            line([(a,y),(min(a+on,xb),y)],fill,w)

    # Grid is behind both the chart and its fused object.
    for gx in np.arange(72,1489,24):
        line([(gx,280),(gx,1870)], (222,225,216), .65)
    for gy in np.arange(280,1871,24):
        line([(72,gy),(1488,gy)], (222,225,216), .65)
    for gx in np.arange(72,1489,120):
        line([(gx,280),(gx,1870)], (213,219,210), .85)
    for gy in np.arange(280,1871,120):
        line([(72,gy),(1488,gy)], (213,219,210), .85)

    # Header and axis typography are code-bound, never part of the generated art.
    rect((96,61,362,100), PLUM)
    text(229,80,data['visible_notice'],20,PAPER,'SansBold.ttf','mm')
    text(96,121,data['title'],74,INK,'NotoSerifSC.otf')
    text(100,222,series['label'],28,INK,'SerifItalic.ttf')
    text(1464,230,f"{data['categories'][0]}—{data['categories'][-1]}",26,MUTED,'Sans.ttf','rm')
    text(x0,291,'比例（%）',25,INK)
    text(105,411,'年份',25,INK)
    line([(x0,352),(x1,352)], (105,108,101), 1.5)
    line([(x0,352),(x0,1452)], (105,108,101), 1.5)
    axis_ticks = []
    for value in np.arange(vmin,vmax+0.01,2):
        px = x0+(value-vmin)*px_per_unit
        line([(px,346),(px,360)], (87,92,85), 1.5)
        text(px,329,f'{value:g}',25,INK,'Sans.ttf','mm')
        axis_ticks.append({'value':float(value),'x_px':float(px)})
    # Year guide lines end at the contour; solid short spans measure the section.
    for category,value,px,py in zip(data['categories'],values,data_x,data_y):
        wl = float(width(py))
        le,ri = px-wl/2, px+wl/2
        line([(x0-8,py),(x0,py)],(82,87,80),1.5)
        text(x0-27,py,category,31,INK,'Serif.ttf','rm')
        dash(x0+3,le-12,py,(174,179,168),1,5,9)
    canvas.alpha_composite(hand)
    draw = ImageDraw.Draw(canvas)
    label_boxes = []
    for category,value,px,py in zip(data['categories'],values,data_x,data_y):
        wl = float(width(py))
        le,ri = px-wl/2, px+wl/2
        # True horizontal section and midpoint; no added centerline between years.
        line([(le,py),(ri,py)],(110,44,82,255),1.2)
        line([(le,py-5),(le,py+5)],PLUM,1.4)
        line([(ri,py-5),(ri,py+5)],PLUM,1.4)
        dot(px,py,6.5,PAPER,PLUM,2)
        dot(px,py,2.5,PLUM)
        line([(ri,py),(ri+18,py)],PLUM,1.2)
        label=f'{value:g}%'
        label_x=ri+30
        f=font('SerifItalic.ttf',39)
        bbox=draw.textbbox((int(label_x*S),int(py*S)),label,font=f,anchor='lm')
        # A tiny paper knockout prevents grid intersections inside the digits.
        box=tuple(int(z) for z in (bbox[0]-6*S,bbox[1]-3*S,bbox[2]+5*S,bbox[3]+3*S))
        draw.rectangle(box,fill=PAPER)
        text(label_x,py,label,39,PLUM,'SerifItalic.ttf','lm')
        label_boxes.append({'year':category,'text':label,'bbox_px':[round(z/S,2) for z in box]})

    # Small reading key occupies unobserved space, clear of the hand.
    rect((1047,1604,1468,1817),PAPER)
    line([(1070,1623),(1122,1623)],PLUM,2)
    text(1070,1641,'读取方式',21,PLUM)
    text(1070,1685,'截面中点 → 比例',28,INK,'NotoSerifSC.otf')
    text(1070,1733,'指宽与甲片不表示数量',21,MUTED)
    line([(1072,1790),(1240,1790)],PLUM,1.2)
    line([(1072,1783),(1072,1797)],PLUM,1.4)
    line([(1240,1783),(1240,1797)],PLUM,1.4)
    dot(1156,1790,6.5,PAPER,PLUM,2)
    dot(1156,1790,2.5,PLUM)
    text(1262,1790,'中点',21,MUTED,anchor='lm')

    line([(96,1894),(1464,1894)],(173,177,165),1)
    text(96,1914,'合成数据，无真实人群统计含义。',22,INK)
    text(96,1953,'年份从下到上，按真实时间间距排列。',20,MUTED)
    text(1464,1953,'0–14%  /  percent',20,MUTED,'Sans.ttf','rt')
    final = canvas.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
    final.save(ROOT/'final.png')

    # Measure the rasterized silhouette, not the nominal centerline coordinates.
    alpha = np.asarray(hand_native)[:,:,3]
    checks=[]
    for category,value,px,py in zip(data['categories'],values,data_x,data_y):
        row=int(round(py))
        xs=np.flatnonzero(alpha[row]>=128)
        left_visible,right_visible=int(xs[0]),int(xs[-1])
        visible_mid=(left_visible+right_visible)/2
        visible_value=vmin+(visible_mid-x0)/px_per_unit
        checks.append({'year':category,'supplied_value_percent':float(value),
                       'sample_y_px':row,'visible_left_px':left_visible,'visible_right_px':right_visible,
                       'visible_midpoint_px':visible_mid,'decoded_visible_percent':round(visible_value,6),
                       'absolute_error_percentage_points':round(abs(visible_value-value),6)})
    interval_checks=[]
    for k in range(len(data_y)-1):
        ys=np.linspace(data_y[k+1],data_y[k],1001)
        vv=vmin+(center(ys)-x0)/px_per_unit
        interval_checks.append({'years':[data['categories'][k],data['categories'][k+1]],
                                'interpolated_min':float(vv.min()),'interpolated_max':float(vv.max()),
                                'within_adjacent_values':bool(vv.min()>=min(values[k:k+2])-1e-9 and vv.max()<=max(values[k:k+2])+1e-9)})
    save_json('artifacts/geometry_check.json',{
        'canvas_px':[W,H],'encoding':'horizontal silhouette midpoint; PCHIP; year linear bottom to top',
        'horizontal_axis':{'minimum':vmin,'maximum':vmax,'x0':x0,'x1':x1,'pixels_per_percentage_point':px_per_unit,'ticks':axis_ticks},
        'observed_y_range_px':[ytop,ybottom],'palm_join_y_px':palm_join_y,
        'raw_finger_y_range':[160,905],'source_asset':'assets/raw_01.png',
        'visible_silhouette_measurements':checks,'interpolation_checks':interval_checks,
        'label_boxes':label_boxes,'max_error_percentage_points':max(c['absolute_error_percentage_points'] for c in checks),
        'measurement_method':'Threshold 128 on the actual antialiased, warped hand alpha at each displayed year row.'})
    print('Rendered final.png at',W,H,'; maximum measured silhouette midpoint error:',
          max(c['absolute_error_percentage_points'] for c in checks),'percentage points')


if __name__ == '__main__':
    main()
