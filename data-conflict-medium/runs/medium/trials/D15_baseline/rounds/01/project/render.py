#!/usr/bin/env python3
"""Regenerate the infographic locally. Launch complete renders through capture_render.py."""
from pathlib import Path
import json
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parent
S = 2
INK = (57, 53, 52)
PURPLE = (120, 72, 116)
MUTED = (112, 106, 102)
PAPER = (249, 248, 244)
FONT_CJK = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONT_CJK_BOLD = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
FONT_NUM = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONT_NUM_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def font(size, bold=False, cjk=False):
    path = (FONT_CJK_BOLD if bold else FONT_CJK) if cjk else (FONT_NUM_BOLD if bold else FONT_NUM)
    return ImageFont.truetype(path, round(size*S), index=0)


def main():
    data = json.loads((ROOT/'data.json').read_text(encoding='utf-8'))
    brief = json.loads((ROOT/'brief.json').read_text(encoding='utf-8'))
    W, H = brief['width'], brief['height']
    values = np.asarray(data['series'][0]['values'], dtype=float)
    years = np.asarray(data['time_points'], dtype=float)
    lo, hi = data['value_axis']['minimum'], data['value_axis']['maximum']
    x0, x1, y0, y1 = 280.0, 1792.0, 1130.0, 380.0
    xs = x0+(years-years[0])/(years[-1]-years[0])*(x1-x0)
    ys = y0-(values-lo)/(hi-lo)*(y0-y1)
    curve = PchipInterpolator(xs, ys)
    ART = ROOT/'artifacts'
    ART.mkdir(exist_ok=True)

    # A quiet sheet of square graph paper. Grid lines are never a second series.
    canvas = Image.new('RGB', (W*S, H*S), (232, 231, 227))
    shadow = Image.new('RGBA', canvas.size)
    sd = ImageDraw.Draw(shadow)
    sd.rectangle((70*S, 64*S, 1940*S, 1392*S), fill=(56, 50, 42, 34))
    shadow = shadow.filter(ImageFilter.GaussianBlur(14*S))
    canvas = Image.alpha_composite(canvas.convert('RGBA'), shadow)
    d = ImageDraw.Draw(canvas)
    d.rectangle((64*S, 50*S, 1936*S, 1384*S), fill=PAPER+(255,))
    for x in range(70, 1937, 30):
        major = (x-280) % 150 == 0
        d.line((x*S, 50*S, x*S, 1384*S), fill=(210, 211, 208, 255) if major else (227, 227, 223, 255), width=1*S)
    for y in range(50, 1385, 30):
        major = (y-1130) % 150 == 0
        d.line((64*S, y*S, 1936*S, y*S), fill=(210, 211, 208, 255) if major else (227, 227, 223, 255), width=1*S)

    def text(x, y, content, size, fill=INK, bold=False, cjk=False, anchor='la'):
        d.text((round(x*S), round(y*S)), content, font=font(size, bold, cjk), fill=fill, anchor=anchor)

    # Header is intentionally neutral about direction and proximity to a threshold.
    d.rectangle((112*S, 91*S, 1877*S, 255*S), fill=PAPER+(249,))
    text(137, 104, data['title'], 70, bold=True, cjk=True)
    text(140, 212, f"{data['series'][0]['label']}  ·  {data['categories'][0]}–{data['categories'][-1]}  ·  {len(values)} 个观测点", 27, fill=MUTED, cjk=True)
    badge = (1411*S, 119*S, 1856*S, 177*S)
    d.rounded_rectangle(badge, radius=7*S, fill=(237, 226, 236), outline=(172, 141, 168), width=1*S)
    text(1633.5, 132, data['visible_notice'], 27, fill=PURPLE, bold=True, anchor='ma')
    d.line((139*S, 277*S, 1856*S, 277*S), fill=(161, 151, 147), width=S)
    text(x0, 306, f"{data['series'][0]['label']} ({data['series'][0]['unit']})", 27, fill=INK, bold=True)

    # Axis positions use exactly the same affine mapping as the hand boundary.
    d.line((x0*S, (y1-8)*S, x0*S, (y0+1)*S), fill=INK, width=3*S)
    d.line((x0*S, y0*S, (x1+25)*S, y0*S), fill=INK, width=3*S)
    for v in range(int(lo), int(hi)+1, 10):
        y = y0-(v-lo)/(hi-lo)*(y0-y1)
        d.line(((x0-13)*S, y*S, x0*S, y*S), fill=INK, width=2*S)
        text(x0-27, y-20, f'{v}%', 30, anchor='ra')
    for x, cat in zip(xs, data['categories']):
        d.line((x*S, y0*S, x*S, (y0+12)*S), fill=INK, width=2*S)
        text(x, y0+30, cat, 31, anchor='ma')
    text(x1, 1232, 'YEAR / 年份', 26, cjk=True, anchor='ra')

    # Read the generated subject, retain all ink, skin texture, nails and gaps.
    source = np.asarray(Image.open(ROOT/'assets/hand_clean.png').convert('RGBA')).astype(np.float32)
    alpha = source[:,:,3]/255.0
    source_start, source_end = 58.0, 1450.0
    source_right = 1471.5
    columns = np.arange(source.shape[1], dtype=float)
    top = np.zeros(source.shape[1], dtype=float)
    bottom = np.zeros_like(top)
    occupied = alpha >= .5
    valid = occupied.any(axis=0)
    for u in np.flatnonzero(valid):
        rows = np.flatnonzero(occupied[:,u])
        first, last = rows[0], rows[-1]
        # Subpixel 50% alpha contour, not a bounding rectangle.
        a0 = alpha[first-1,u] if first else 0
        top[u] = first-1+(.5-a0)/max(alpha[first,u]-a0, 1e-6)
        a1 = alpha[last+1,u] if last+1 < alpha.shape[0] else 0
        bottom[u] = last+(alpha[last,u]-.5)/max(alpha[last,u]-a1, 1e-6)
    top[~valid] = np.interp(columns[~valid], columns[valid], top[valid])
    bottom[~valid] = np.interp(columns[~valid], columns[valid], bottom[valid])
    sx = (x1-x0)/(source_end-source_start)
    obj_l, obj_r = int(x0), math.ceil(x0+(source_right-source_start)*sx+2)
    obj_t, obj_b = int(min(ys)-5), int(y0+3)
    px = (np.arange(obj_l*S, obj_r*S)+.5)/S
    py = (np.arange(obj_t*S, obj_b*S)+.5)/S
    u = source_start+(px-x0)/sx
    st = np.interp(u, columns, top)
    sb = np.interp(u, columns, bottom)
    target = curve(np.clip(px, x0, x1))
    # Beyond the last observed point only the original rounded fingertip remains.
    cap = px > x1
    target[cap] = ys[-1]+np.maximum(0, st[cap]-np.interp(source_end, columns, top))*1.06
    vscale = np.minimum(1.06, np.maximum(.0001, y0-target)/np.maximum(sb-st, .1))
    srcy = st[None,:]+(py[:,None]-target[None,:])/vscale[None,:]
    srcx = np.broadcast_to(u[None,:], srcy.shape)
    coord = np.stack((srcy, srcx))
    a = map_coordinates(alpha, coord, order=1, mode='constant', cval=0)
    # Enforce the exact analytic upper surface on the same warped object.
    coverage = np.clip((py[:,None]-target[None,:])*S+.5, 0, 1)
    a *= coverage
    a *= (px[None,:] >= x0)
    warped = np.zeros(srcy.shape+(4,), dtype=np.uint8)
    for c in range(3):
        premult = source[:,:,c]*alpha
        channel = map_coordinates(premult, coord, order=1, mode='constant', cval=0)
        warped[:,:,c] = np.clip(channel/np.maximum(a/np.maximum(coverage,1e-5),1e-5), 0, 255).astype(np.uint8)
    warped[:,:,3] = np.clip(a*255, 0, 255).astype(np.uint8)
    subject = Image.fromarray(warped, 'RGBA')
    canvas.alpha_composite(subject, (obj_l*S, obj_t*S))

    # Save the actual raster silhouette for independently repeatable measurements.
    mask = Image.new('L', (W*S,H*S), 0)
    mask.paste(subject.getchannel('A'), (obj_l*S,obj_t*S))
    mask.resize((W,H), Image.Resampling.LANCZOS).save(ART/'object_alpha.png')
    subject.resize((obj_r-obj_l,obj_b-obj_t), Image.Resampling.LANCZOS).save(ART/'warped_subject.png')

    d = ImageDraw.Draw(canvas)
    # Thin ink at the true physical top edge, continuous with the generated skin.
    trace_x = np.linspace(x0, x1, 1900)
    trace = list(zip(trace_x*S, curve(trace_x)*S))
    d.line(trace, fill=(72, 52, 45), width=2*S, joint='curve')
    # Every supplied value gets one directly attached marker and its exact label.
    for i, (x,y,v) in enumerate(zip(xs,ys,values)):
        label = f'{v:g}%'
        label_y = y-65
        d.line((x*S, (label_y+39)*S, x*S, (y-7)*S), fill=(152,110,146), width=1*S)
        f = font(33, True)
        box = d.textbbox((x*S,label_y*S), label, font=f, anchor='ma')
        d.rounded_rectangle((box[0]-8*S, box[1]-4*S, box[2]+8*S,box[3]+5*S), radius=5*S,fill=PAPER+(255,))
        text(x, label_y, label, 33, fill=PURPLE, bold=True, anchor='ma')
        r=5.5
        d.ellipse(((x-r)*S,(y-r)*S,(x+r)*S,(y+r)*S), fill=PAPER, outline=PURPLE, width=2*S)

    # One unobtrusive encoding key, in the large blank area above the early years.
    key_x, key_y = 426, 446
    d.line((key_x*S, key_y*S, (key_x+46)*S,key_y*S), fill=(72,52,45), width=2*S)
    d.ellipse(((key_x+18)*S,(key_y-5)*S,(key_x+28)*S,(key_y+5)*S), fill=PAPER,outline=PURPLE,width=2*S)
    text(key_x+64, key_y-19, '上轮廓读取数值', 26, cjk=True, fill=MUTED)
    text(key_x+64, key_y+24, 'Upper edge = Group A', 22, fill=MUTED)

    d.line((140*S, 1294*S, 1856*S,1294*S), fill=(185,175,170), width=S)
    text(142, 1313, '合成数据 · 0–50% 线性量尺 · 年份按真实间距排列', 23, cjk=True, fill=MUTED)
    text(1855, 1316, '末端圆角不增加观测年份', 21, cjk=True, fill=MUTED, anchor='ra')
    canvas.convert('RGB').resize((W,H),Image.Resampling.LANCZOS).save(ROOT/'final.png')

    geometry = {
        'canvas': [W,H], 'plot': {'x_start':x0,'x_end':x1,'y_zero':y0,'y_fifty':y1},
        'scale': {'pixels_per_year':(x1-x0)/(years[-1]-years[0]),'pixels_per_percentage_point':(y0-y1)/(hi-lo)},
        'interpolation':'scipy.interpolate.PchipInterpolator, no extrapolation of observations',
        'points':[{'year':str(c),'value':float(v),'x':float(x),'y':float(y)} for c,v,x,y in zip(data['categories'],values,xs,ys)],
        'source_control': {'start_column':source_start,'observation_end_column':source_end,'rightmost_column':source_right},
        'subject_bbox':[obj_l,obj_t,obj_r,obj_b],
        'generated_asset':'assets/hand_clean.png',
    }
    (ART/'geometry.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
