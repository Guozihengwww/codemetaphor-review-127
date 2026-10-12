#!/usr/bin/env python3
"""Rebuild the area-calibrated egg infographic from local, frozen data and assets.
Run from this directory with the provided Python interpreter. No network/model calls.
"""
from pathlib import Path
from decimal import Decimal
import json
import math
import hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent
S = 2
INK = '#203b32'
MUTED = '#687d70'
BG = '#edf4ed'
COLORS = ['#a4794b', '#e6c56f', '#a7b98a', '#dfa56e', '#719482',
          '#edcf93', '#bf784d', '#e8ad7e', '#a7b795', '#d0a658',
          '#b96541', '#d99a61']


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def rgb(value):
    return np.array([int(value[i:i+2], 16) for i in (1, 3, 5)], dtype=np.float32)


def main():
    brief = json.loads((ROOT/'brief.json').read_text(encoding='utf-8'))
    data = json.loads((ROOT/'data.json').read_text(encoding='utf-8'), parse_float=Decimal)
    cats = data['categories']
    raw_vals = data['series'][0]['values']
    values = np.array([float(v) for v in raw_vals], dtype=float)
    assert len(cats) == len(values) == 12
    assert abs(values.sum() - 100) < 1e-9
    value_strings = [str(v) + '%' for v in raw_vals]
    W, H = brief['width'], brief['height']
    assert (W, H) == (2000, 1800)
    (ROOT/'artifacts').mkdir(exist_ok=True)

    canvas = Image.new('RGB', (W*S, H*S), BG)
    draw = ImageDraw.Draw(canvas)
    font_cache = {}
    text_boxes = []
    def font(size, face='sans'):
        key = (size, face)
        if key not in font_cache:
            filename = {'sans':'sans.otf', 'bold':'sans_bold.otf', 'cjk':'cjk.ttc'}[face]
            font_cache[key] = ImageFont.truetype(str(ROOT/'assets'/filename), round(size*S), index=0)
        return font_cache[key]
    def text(x, y, label, size, face='sans', color=INK, anchor='lt', record=True):
        f = font(size, face)
        pos = (round(x*S), round(y*S))
        box = draw.textbbox(pos, label, font=f, anchor=anchor)
        draw.text(pos, label, font=f, fill=color, anchor=anchor)
        if record:
            text_boxes.append({'text':label, 'bbox':[round(k/S, 2) for k in box]})
        return [k/S for k in box]
    def line(points, color, width=1):
        draw.line([(round(x*S), round(y*S)) for x,y in points], fill=color, width=max(1,round(width*S)), joint='curve')
    def rect(box, color):
        draw.rectangle([round(k*S) for k in box], fill=color)

    # Header typography and all labels are drawn by executable code.
    text(120, 64, 'SYNTHETIC COMPOSITION', 20, 'bold', MUTED)
    text(116, 110, data['title'], 66, 'cjk')
    text(120, 209, 'Unit: ' + data['series'][0]['unit'], 27, color=MUTED)
    draw.rounded_rectangle((1505*S,90*S,1880*S,148*S),radius=29*S,fill='#dce8d8')
    text(1692.5, 106, data['visible_notice'], 25, 'bold', anchor='mt')
    text(1880, 209, f'{len(cats)} groups  /  {float(values.sum()):g}%', 27, color=MUTED, anchor='rt')
    line([(120,260),(1880,260)], '#c7d6c9', 1)

    # One continuous generated eggshell. Its alpha is the authoritative object mask.
    ox, oy, ew, eh = 570, 405, 860, 1204
    common = np.array([1000.0, 900.0])
    shell = Image.open(ROOT/'assets/eggshell_01.png').convert('RGBA').resize((ew*S,eh*S),Image.Resampling.LANCZOS)
    shell_array = np.asarray(shell)
    alpha = shell_array[:,:,3].astype(np.float64)/255.0
    yy, xx = np.indices(alpha.shape, dtype=np.float64)
    xx = (xx+.5)/S + ox
    yy = (yy+.5)/S + oy
    theta = np.arctan2(xx-common[0], common[1]-yy)  # clockwise from the top
    valid = alpha > 0
    tv, av = theta[valid], alpha[valid]
    order = np.argsort(tv)
    sorted_theta = tv[order]
    cumulative = np.cumsum(av[order])
    total = float(cumulative[-1])
    top_cumulative = float(np.interp(0.0,sorted_theta,cumulative))
    small_total = float(values[:-1].sum())
    first_weight = top_cumulative - total*small_total/200.0
    targets = first_weight + np.r_[0.0,np.cumsum(values[:-1])]*total/100.0
    bounds = np.interp(targets,cumulative,sorted_theta)
    ids = np.searchsorted(bounds,theta,side='right') - 1
    ids[(ids < 0) | (ids >= len(cats)-1)] = len(cats)-1
    ids = ids.astype(np.uint8)
    observed = np.bincount(ids[valid],weights=alpha[valid],minlength=len(cats))/total*100.0

    # Retain generated pores, freckles, and the entire generated lighting field.
    # The color field is selected exclusively by the area-calibrated partition ID.
    palette = np.array([rgb(c) for c in COLORS])
    reference_shell = np.array([226.0,195.0,164.0], dtype=np.float32)
    illumination = shell_array[:,:,:3].astype(np.float32)/reference_shell
    # Compress only extreme highlights, without modifying the data mask.
    illumination = np.clip(illumination,0.28,1.38)
    tinted = np.clip(palette[ids] * illumination,0,255).astype(np.uint8)
    rgba = np.dstack((tinted,shell_array[:,:,3]))
    egg = Image.fromarray(rgba,'RGBA')

    # A small contact shadow does not belong to the quantitative mask.
    shadow = Image.new('RGBA',canvas.size,(0,0,0,0))
    sd = ImageDraw.Draw(shadow)
    sd.ellipse((755*S,1594*S,1245*S,1630*S),fill=(62,72,51,27))
    shadow = shadow.filter(ImageFilter.GaussianBlur(14*S))
    canvas.paste(shadow,(0,0),shadow)
    canvas.paste(egg,(ox*S,oy*S),egg)
    draw = ImageDraw.Draw(canvas)

    def edge_at_angle(angle):
        # Locate the rendered shell's actual 50%-alpha edge on the specified ray.
        rs = np.arange(0,1500,0.125)
        gx = common[0]+np.sin(angle)*rs
        gy = common[1]-np.cos(angle)*rs
        lx = np.floor((gx-ox)*S).astype(int)
        ly = np.floor((gy-oy)*S).astype(int)
        inside = (lx>=0)&(lx<ew*S)&(ly>=0)&(ly<eh*S)
        sampled = np.zeros(rs.size)
        sampled[inside] = alpha[ly[inside],lx[inside]]
        good = np.where(sampled>=.5)[0]
        radius = rs[good[-1]] if len(good) else 0
        return np.array([common[0]+np.sin(angle)*radius,common[1]-np.cos(angle)*radius])

    # Hairline seams are drawn on top; they never cut gaps out of the data mask.
    seam_layer = Image.new('RGBA',egg.size,(0,0,0,0))
    seams = ImageDraw.Draw(seam_layer)
    for angle in bounds:
        end = edge_at_angle(angle)
        seams.line(((round((common[0]-ox)*S),round((common[1]-oy)*S)),
                    (round((end[0]-ox)*S),round((end[1]-oy)*S))),
                   fill=(255,244,213,100),width=2)
    seam_layer.putalpha(Image.fromarray((np.asarray(seam_layer.getchannel('A')).astype(float)*alpha).astype('uint8')))
    canvas.paste(seam_layer,(ox*S,oy*S),seam_layer)
    draw = ImageDraw.Draw(canvas)

    # Endpoints follow the arc order; top labels absorb the narrow apex wedges.
    ends = [(430,1057),(430,857),(430,657),(430,477),
            (730,382),(1000,369),(1260,382),
            (1570,477),(1570,657),(1570,857),(1590,1007)]
    leader_records = []
    anchors=[]
    def add_leader(i, anchor, endpoint):
        anchor=np.asarray(anchor,float); endpoint=np.asarray(endpoint,float)
        vec=endpoint-anchor
        vec=vec/np.linalg.norm(vec)
        start=anchor+vec*4.0
        line([start,endpoint], '#81917f', 1.35)
        r=3.1
        draw.ellipse(((start[0]-r)*S,(start[1]-r)*S,(start[0]+r)*S,(start[1]+r)*S),fill='#627760')
        ts=np.linspace(.012,1.0,500)
        points=anchor[None,:]+(endpoint-anchor)[None,:]*ts[:,None]
        xi=np.floor((points[:,0]-ox)*S).astype(int)
        yi=np.floor((points[:,1]-oy)*S).astype(int)
        in_box=(xi>=0)&(xi<ew*S)&(yi>=0)&(yi<eh*S)
        interior=np.zeros(len(ts),dtype=bool)
        interior[in_box]=alpha[yi[in_box],xi[in_box]]>.8
        leader_records.append({'category':cats[i],'anchor_xy':[round(v,3) for v in anchor],
                               'endpoint_xy':list(endpoint),'samples_inside_egg_excluding_anchor':int(interior.sum())})
        anchors.append(anchor)
    for i in range(11):
        mid_target=(targets[i]+targets[i+1])/2
        angle=float(np.interp(mid_target,cumulative,sorted_theta))
        add_leader(i,edge_at_angle(angle),ends[i])
    big_angle=math.radians(126)
    add_leader(11,edge_at_angle(big_angle),(1525,1319))

    # Category names and exact supplied values, all outside the intact egg.
    left_y=[995,795,595,415]
    for i,y in enumerate(left_y):
        rect((139,y+3,147,y+27),COLORS[i])
        text(170,y,cats[i],29,'bold')
        text(168,y+42,value_strings[i],49)
    for i,cx,y in [(4,730,296),(5,1000,283),(6,1260,296)]:
        text(cx,y,cats[i],26,'bold',anchor='mt')
        text(cx,y+36,value_strings[i],43,anchor='mt')
        rect((cx-21,y+80,cx+21,y+83),COLORS[i])
    for i,y in [(7,415),(8,595),(9,795),(10,945)]:
        x=1610 if i==10 else 1590
        rect((x,y+3,x+8,y+27),COLORS[i])
        text(x+31,y,cats[i],29,'bold')
        text(x+29,y+42,value_strings[i],49)
    rect((1565,1235,1575,1262),COLORS[11])
    text(1600,1232,cats[11],32,'bold')
    text(1562,1280,value_strings[11],98)

    # A light explanatory footer, separate from the chart and labels.
    line([(120,1663),(1880,1663)], '#c7d6c9', 1)
    text(120,1694,'区域面积 = 构成占比',28,'cjk')
    text(120,1740,'合成数据，不代表真实鸡蛋成分。',23,'cjk',MUTED)
    text(1880,1696,'Group 01 → Group 12 · clockwise',25,color=MUTED,anchor='rt')
    text(1880,1742,'One egg. Twelve area-calibrated regions.',23,color=MUTED,anchor='rt')

    # Save actual native-resolution coverage masks used to audit visible geometry.
    alpha_final=Image.fromarray(np.rint(alpha*255).astype('uint8')).resize((ew,eh),Image.Resampling.BOX)
    alpha_final.save(ROOT/'artifacts/egg_data_mask.png')
    coverage=[]
    area_native=[]
    for i in range(len(cats)):
        region=Image.fromarray(np.rint((ids==i)*alpha*255).astype('uint8')).resize((ew,eh),Image.Resampling.BOX)
        r=np.array(region)
        coverage.append(r)
        area_native.append(float(r.sum()/255.0))
    coverage=np.array(coverage,dtype=np.uint8)
    np.savez_compressed(ROOT/'artifacts/region_coverage.npz',coverage=coverage,origin=np.array([ox,oy]),common_point=common)
    native_share=np.array(area_native)/np.sum(area_native)*100
    # The partition map assigns every nonzero shell pixel exactly once.
    assignment_counts=np.zeros(alpha.shape,dtype=np.uint8)
    for i in range(len(cats)):assignment_counts += (ids==i).astype(np.uint8)
    overlap=int(np.sum(valid&(assignment_counts>1)))
    unassigned=int(np.sum(valid&(assignment_counts==0)))
    boxes_ok=all(0<=b['bbox'][0]<b['bbox'][2]<=W and 0<=b['bbox'][1]<b['bbox'][3]<=H for b in text_boxes)
    overlaps=[]
    for i,b in enumerate(text_boxes):
        for c in text_boxes[i+1:]:
            a=b['bbox'];z=c['bbox']
            if min(a[2],z[2])-max(a[0],z[0])>0 and min(a[3],z[3])-max(a[1],z[1])>0:
                overlaps.append([b['text'],c['text']])
    geometry={
        'canvas':[W,H], 'supersampling':S,
        'mask_origin_xy':[ox,oy], 'mask_size':[ew,eh],
        'shared_interior_point_xy':common.tolist(),
        'area_method_zh':'按生成蛋壳透明度遮罩的实际像素面积，计算顺时针极角累计面积分位点；非按中心角直接取百分比。',
        'category_order':cats,
        'angle_boundaries_clockwise_from_top_degrees':np.rad2deg(bounds).tolist(),
        'measurements':[{'category':cats[i],'label':value_strings[i],'target_percent':float(values[i]),
                         'supersampled_mask_percent':float(observed[i]),
                         'native_pixel_coverage_percent':float(native_share[i]),
                         'native_area_pixels':area_native[i],
                         'absolute_error_percentage_points':abs(float(native_share[i]-values[i]))} for i in range(len(cats))],
        'max_absolute_error_percentage_points':float(np.max(np.abs(native_share-values))),
        'overlap_pixels_in_data_masks':overlap,'unassigned_pixels_in_data_masks':unassigned,
        'all_text_in_canvas':boxes_ok,'text_box_overlaps':overlaps,
        'leaders':leader_records,'text_boxes':text_boxes,
        'generated_texture_coverage_of_egg_percent':100.0,
        'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['brief.json','data.json','reference.png','assets/raw_01.png','assets/eggshell_01.png']}
    }
    save_json(ROOT/'artifacts/geometry_check.json',geometry)
    out=canvas.resize((W,H),Image.Resampling.LANCZOS)
    out.save(ROOT/'final.png')
    print(json.dumps({'size':[W,H],'categories':len(cats),'max_area_error_pp':geometry['max_absolute_error_percentage_points'],
                      'all_text_in_canvas':boxes_ok,'text_box_overlap_count':len(overlaps),
                      'leader_interior_samples':[x['samples_inside_egg_excluding_anchor'] for x in leader_records]}))


if __name__=='__main__':
    main()
