#!/usr/bin/env python3
"""Offline, data-bound hand infographic. Run only through capture_render.py.

All times and values come from data.json. The generated hand's upper alpha
boundary is column-warped onto a monotone PCHIP. The lower edge and anatomy do
not encode another statistic. The small distal cap is outside the time domain.
"""
from pathlib import Path
import json
import math
import hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parent
S = 2
INK = '#363332'
MUTED = '#777573'
PURPLE = '#885b89'
PAPER = '#f8f8f4'


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def save_json(name, value):
    p = ROOT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def font(size, style='body'):
    return ImageFont.truetype(str(ROOT / 'fonts' / (style + '.otf')), round(size*S))


def main():
    inputs = read('input.json')
    brief = read(inputs['brief'])
    data = read(inputs['data'])
    core = read(inputs['critical_metaphor_contract'])
    assert core['semantic_roles_zh'] == ['Women parliamentary seats (%)']
    assert data['synthetic'] and data['value_axis']['scale'] == 'linear'
    width, height = brief['width'], brief['height']
    xs = np.array(data['time_points'], dtype=float)
    ys = np.array(data['series'][0]['values'], dtype=float)
    cats = data['categories']
    assert len(xs) == len(ys) == len(cats) and np.all(np.diff(xs) > 0)
    vmin, vmax = data['value_axis']['minimum'], data['value_axis']['maximum']
    x0, x1 = 310.0, 1750.0
    ytop, yzero = 385.0, 1155.0
    ppu = (yzero-ytop)/(vmax-vmin)
    xscale = (x1-x0)/(xs[-1]-xs[0])
    node_x = x0 + (xs-xs[0])*xscale
    node_y = yzero-(ys-vmin)*ppu
    interp = PchipInterpolator(xs, ys, extrapolate=False)

    def upper(x):
        time = xs[0]+(np.asarray(x)-x0)/xscale
        return yzero-(interp(np.clip(time, xs[0], xs[-1]))-vmin)*ppu

    image = Image.new('RGB', (width*S, height*S), PAPER)
    draw = ImageDraw.Draw(image)
    text_boxes = []

    def line(points, fill=INK, w=1):
        draw.line([(round(x*S), round(y*S)) for x, y in points], fill=fill, width=max(1, round(w*S)))

    def text(x, y, s, size=28, fill=INK, style='body', align='left', halo=False):
        f = font(size, style)
        bb = draw.textbbox((0, 0), s, font=f, anchor='lt')
        tw = bb[2]-bb[0]
        tx = x*S - (tw/2 if align == 'center' else tw if align == 'right' else 0)
        ty = y*S
        opts = dict(font=f, fill=fill, anchor='lt')
        if halo:
            opts.update(stroke_width=4*S, stroke_fill=PAPER)
        draw.text((round(tx), round(ty)), s, **opts)
        bounds = draw.textbbox((round(tx), round(ty)), s, font=f, anchor='lt')
        text_boxes.append({'text':s, 'bounds_px':[round(v/S, 2) for v in bounds]})

    # A restrained editorial header, with the synthetic notice at first glance.
    text(125, 64, 'REPRESENTATION  /  A SYNTHETIC STUDY', 22, PURPLE, 'bold')
    text(125, 118, data['series'][0]['label'], 81, INK, 'title')
    text(128, 227, 'Share of all parliamentary seats held by women (%)', 31, MUTED)
    badge = (1565, 60, 1874, 104)
    draw.rounded_rectangle(tuple(round(v*S) for v in badge), radius=7*S, fill='#eee6ef')
    text((badge[0]+badge[2])/2, 72, data['visible_notice'], 22, PURPLE, 'bold', 'center')
    line([(125, 282), (1875, 282)], '#c6c7c3', 1)

    # Graph paper. Minor 2-percentage-point vertical spacing shares the scale.
    gx0, gx1, gy0, gy1 = 178, 1876, 316, 1230
    grid_step = ppu*2
    for i in range(-4, 54):
        x = x0 + i*grid_step
        if gx0 <= x <= gx1:
            line([(x, gy0), (x, gy1)], '#e2e4e0', .85)
    for i in range(-2, 29):
        y = yzero-i*grid_step
        if gy0 <= y <= gy1:
            line([(gx0, y), (gx1, y)], '#e2e4e0', .85)
    for value in range(0, 51, 10):
        y = yzero-(value-vmin)*ppu
        line([(x0, y), (1830, y)], '#cdd0cc', 1.15)

    # Explicit equality benchmark, independent of the terminal observation.
    benchmark = data['semantic_target']['benchmark_percent']
    by = yzero-(benchmark-vmin)*ppu
    for start in np.arange(x0, 1830, 18):
        line([(start, by), (min(start+9, 1830), by)], '#ab8daa', 1.7)
    text(339, 332, f'{benchmark}%  EQUALITY TARGET', 27, PURPLE, 'bold', halo=True)
    text(339, 413, 'Half of all parliamentary seats', 25, MUTED, halo=True)

    # Warp the actual generated pixels, including all five nails and creases.
    source = np.array(Image.open(ROOT/'assets/hand_rgba.png').convert('RGBA'), dtype=np.float32)/255
    alpha = source[:,:,3]
    src_x = np.arange(source.shape[1])
    src_top = np.full(source.shape[1], np.nan, dtype=float)
    for xx in src_x:
        iy = np.flatnonzero(alpha[:,xx] >= .5)
        if len(iy):
            y = int(iy[0])
            if y and alpha[y,xx] != alpha[y-1,xx]:
                src_top[xx] = y-1+(.5-alpha[y-1,xx])/(alpha[y,xx]-alpha[y-1,xx])
            else:
                src_top[xx] = y
    present = np.isfinite(src_top)
    src_top = np.interp(src_x, src_x[present], src_top[present])
    # The last observation lies on the final finger before its rounded cap.
    src_start, src_end = 51., 1435.
    sx_per_px = (src_end-src_start)/(x1-x0)
    anatomy_scale_y = .94
    right = math.ceil(x1+(1488-src_end)/sx_per_px)
    left, top, bottom = int(x0), int(ytop), int(yzero)
    ww, hh = (right-left)*S, (bottom-top)*S
    warped = np.zeros((hh, ww, 4), np.uint8)
    x_dest = left+(np.arange(ww)+.5)/S
    sx = src_start+(x_dest-x0)*sx_per_px
    # Hold the deformation anchor past the observation, leaving the tip rounded.
    anchor_src_y = np.interp(np.minimum(sx, src_end), src_x, src_top)
    anchor_dest_y = upper(np.minimum(x_dest, x1))
    premult = source.copy()
    premult[:,:,:3] *= alpha[:,:,None]
    for row0 in range(0, hh, 160):
        row1 = min(hh, row0+160)
        yy = top+(np.arange(row0,row1)+.5)/S
        sy = anchor_src_y[None,:]+(yy[:,None]-anchor_dest_y[None,:])/anatomy_scale_y
        xx = np.broadcast_to(sx[None,:], sy.shape)
        coords = np.array([sy,xx])
        values = np.stack([map_coordinates(premult[:,:,c], coords, order=1, mode='constant', cval=0, prefilter=False) for c in range(4)],axis=-1)
        a = values[:,:,3]
        # No generated fringe or original texture may project above the data edge.
        above = yy[:,None] < anchor_dest_y[None,:]
        above &= (x_dest <= x1)[None,:]
        a[above] = 0
        values[:,:,:3] = np.divide(values[:,:,:3],a[:,:,None],out=np.zeros_like(values[:,:,:3]),where=a[:,:,None]>.0001)
        warped[row0:row1] = np.clip(np.rint(values*255),0,255).astype(np.uint8)
    layer = Image.fromarray(warped, 'RGBA')
    image.paste(layer, (left*S, top*S), layer)
    # A continuous thin ink edge belongs to the hand, not a detached chart line.
    # It also makes the very low values clear without changing their heights.
    edge_x = np.linspace(x0,x1,3601)
    edge_y = upper(edge_x)
    line(list(zip(edge_x,edge_y+.95)), '#504035', 2.0)

    # Save an anatomy-only diagnostic mask; this is not a chart preview.
    alpha_canvas = Image.new('L',(width*S,height*S),0)
    alpha_canvas.paste(layer.getchannel('A'),(left*S,top*S))
    alpha_low = alpha_canvas.resize((width,height),Image.Resampling.LANCZOS)
    alpha_low.save(ROOT/'artifacts/subject_alpha.png')
    layer.resize((right-left,bottom-top),Image.Resampling.LANCZOS).save(ROOT/'assets/hand_warped.png')

    # Shared zero baseline and real-year ticks.
    line([(x0, ytop-4), (x0, yzero)], '#777572', 1.7)
    line([(x0, yzero), (x1, yzero)], '#57524f', 2.1)
    for value in range(0, 51, 10):
        y = yzero-(value-vmin)*ppu
        line([(x0-9, y), (x0, y)], '#696360', 1.5)
        text(x0-24, y-13, str(value), 27, PURPLE if value==50 else MUTED, 'bold' if value==50 else 'body', 'right', True)
    text(x0-23, ytop-52, '%', 26, MUTED, 'body', 'right', True)

    label_records=[]
    for i, (x,y,value,category) in enumerate(zip(node_x,node_y,ys,cats)):
        line([(x,yzero), (x,yzero+12)], '#696360', 1.5)
        text(x,yzero+27,category,28,INK,'body','center',True)
        # Label the two plateau observations above the 50% guide with leaders.
        label_y = 332 if i >= len(xs)-2 else y-56
        if i >= len(xs)-2:
            line([(x, label_y+34), (x,y-9)], '#8d788c',1.1)
        radius = 4.5
        draw.ellipse(((x-radius)*S,(y-radius)*S,(x+radius)*S,(y+radius)*S),fill=PURPLE,outline=PAPER,width=2*S)
        value_s = f'{value:.1f}%'
        text(x,label_y,value_s,32,PURPLE,'bold','center',True)
        label_records.append({'category':category,'value_text':value_s,'x_px':round(float(x),4),'y_px':round(float(y),4),'label_top_px':round(float(label_y),4)})
    text(x1,1212,'YEAR',22,MUTED,'bold','right',True)

    # Short reading key in open paper space; all data remain on the object.
    line([(463,665),(498,665)],PURPLE,3)
    text(514,649,'The upper edge traces',29,INK)
    text(514,689,"women's share of seats.",29,INK)

    line([(125,1275),(1875,1275)],'#c6c7c3',1)
    text(128,1310,'Nine synthetic observations. Year spacing is proportional; the vertical scale is 0–50%.',25,MUTED)
    text(128,1351,'Hand thickness and finger details are illustrative. The rounded tip adds no time point.',24,MUTED)
    image = image.resize((width,height),Image.Resampling.LANCZOS)
    image.save(ROOT/'final.png')

    # Checks of the rasterized silhouette, rather than targets alone.
    visible_alpha = np.asarray(alpha_low)
    contour=[]
    for i, (x,y,value) in enumerate(zip(node_x,node_y,ys)):
        col = min(width-1,max(0,int(round(x))))
        yy = np.flatnonzero(visible_alpha[:,col]>=128)
        measured = float(yy[0]) if len(yy) else None
        measured_value = (yzero-measured)/ppu+vmin if measured is not None else None
        contour.append({'year':cats[i],'supplied_percent':float(value),'target_x_px':round(float(x),4),'target_y_px':round(float(y),4),'raster_column_px':col,'visible_upper_y_px':measured,'visible_percent':round(measured_value,4) if measured_value is not None else None,'absolute_error_percentage_points':round(abs(measured_value-value),4) if measured_value is not None else None,'height_above_zero_px':round(yzero-measured,3) if measured is not None else None})
    dense_x = np.linspace(x0+1,x1-1,1400)
    err=[]
    for x in dense_x:
        col=int(round(x))
        visible=np.flatnonzero(visible_alpha[:,col]>=128)
        if len(visible):
            err.append(abs(float(visible[0])-float(upper(col)))/ppu)
    overshoot=[]
    for i in range(len(xs)-1):
        vv=interp(np.linspace(xs[i],xs[i+1],1001))
        overshoot.append({'from':cats[i],'to':cats[i+1],'within_neighbor_range':bool(vv.min()>=min(ys[i:i+2])-1e-10 and vv.max()<=max(ys[i:i+2])+1e-10)})
    save_json('artifacts/geometry_checks.json',{
        'canvas_px':[width,height],
        'mapping':{'x_start_px':x0,'x_end_px':x1,'first_year':float(xs[0]),'last_year':float(xs[-1]),'pixels_per_year':xscale,'zero_y_px':yzero,'fifty_percent_y_px':ytop,'pixels_per_percentage_point':ppu,'interpolation':'PCHIP, shape preserving; no interval overshoot','decorative_cap_width_px':right-x1},
        'rendered_subject_mask_measurements':contour,
        'max_dense_silhouette_error_pp':max(err),
        'interpolation_checks':overshoot,
        'bound_labels':label_records,
        'text_boxes':text_boxes,
        'method_zh':'读取实际变形素材的alpha光栅边缘，阈值128；数据节点标记与坐标轴未计入轮廓测量。非外部评价。',
        'source_files':{'raw':'assets/raw_01.png','processed':'assets/hand_rgba.png','warped_diagnostic':'assets/hand_warped.png'},
        'frozen_data_sha256':hashlib.sha256((ROOT/inputs['data']).read_bytes()).hexdigest()
    })
    print(json.dumps({'size':[width,height],'max_dense_silhouette_error_pp':round(max(err),4),'node_count':len(xs)},ensure_ascii=False))


if __name__ == '__main__':
    main()
