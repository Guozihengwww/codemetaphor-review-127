#!/usr/bin/env python3
"""Recreate the infographic from the frozen data and a local generated hand.

Run this file from the case directory. Complete experimental renders are launched
by capture_render.py. The only quantitative boundary is the hand's upper edge.
"""
from pathlib import Path
import json
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy import ndimage
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parent
S = 2
INK = (64, 61, 58)
MUTED = (111, 109, 104)
PURPLE = (131, 84, 125)
PAPER = (248, 248, 244)


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def font(size, serif=False):
    return ImageFont.truetype(str(ROOT / "assets/fonts" / ("serif.ttf" if serif else "sans.ttf")), round(size*S))


def text(draw, xy, value, size, fill=INK, anchor="lt", serif=False, halo=0):
    draw.text(tuple(round(a*S) for a in xy), value, font=font(size, serif),
              fill=fill, anchor=anchor, stroke_width=round(halo*S), stroke_fill=PAPER)


def line(draw, points, fill, width=1):
    draw.line([(round(x*S), round(y*S)) for x,y in points], fill=fill,
              width=max(1, round(width*S)), joint="curve")


def overlap_upper_fingers(raw):
    """Tuck the second finger behind the first; these finger widths are not data."""
    height,width = raw.shape[:2]
    xx = np.arange(width,dtype=float)
    # Follow the existing ink seam between the first and second generated digits.
    upper_cut = np.interp(xx,[1740,1800,1850,1900,1950,2000,2050,2107],
                            [202,196,190,194,199,201,207,202])
    lower_cut = np.interp(xx,[1740,1800,1850,1900,1950,2000,2050,2107],
                            [286,281,276,280,289,301,311,306])
    yy = np.arange(height,dtype=float)[:,None]
    selected = ((xx[None,:]>=1740) & (yy>upper_cut[None,:]) & (yy<lower_cut[None,:]))
    digit = raw.copy()
    digit[...,3][~selected] = 0
    base = raw.copy()
    base[...,3][selected] = 0
    t = np.clip((xx-1760)/345,0,1)
    ease = t*t*(3-2*t)
    destination_x = xx - 35*ease
    src_x_line = np.interp(xx,destination_x,xx)
    shift = np.interp(src_x_line,xx,65*ease)
    src_y = np.broadcast_to(yy+shift[None,:],(height,width))
    src_x = np.broadcast_to(src_x_line[None,:],(height,width))
    moved = np.empty_like(raw)
    for c in range(4):
        moved[...,c] = ndimage.map_coordinates(digit[...,c],[src_y,src_x],order=1,
                                              mode="constant",cval=0,prefilter=False)
    composed = Image.fromarray(moved,"RGBA")
    composed.alpha_composite(Image.fromarray(base,"RGBA"))
    composed.save(ROOT / "assets/arm_layered.png")
    return np.asarray(composed)


def prepare_source():
    raw = np.array(Image.open(ROOT / "assets/raw_01.png").convert("RGBA"))
    raw = overlap_upper_fingers(raw)
    mask = raw[...,3] > 110
    labels, n = ndimage.label(mask)
    areas = np.bincount(labels.ravel()); areas[0] = 0
    mask = labels == areas.argmax()
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_closing(mask, iterations=1)
    # Extend the nearest real edge color into transparent pixels before sampling.
    # This removes colored RGB garbage in transparent parts of the raw PNG.
    _, nearest = ndimage.distance_transform_edt(~mask, return_indices=True)
    rgb = raw[...,:3].copy()
    rgb[~mask] = rgb[nearest[0][~mask],nearest[1][~mask]]
    alpha = ndimage.gaussian_filter(mask.astype(np.float32), .52)
    clean = np.dstack([rgb, np.clip(alpha*255,0,255).astype(np.uint8)])
    Image.fromarray(clean).save(ROOT / "assets/arm_clean.png")
    top = np.full(mask.shape[1], np.nan, dtype=float)
    for x in range(mask.shape[1]):
        ys = np.flatnonzero(mask[:,x])
        if len(ys): top[x] = ys[0] - .5
    ok = np.isfinite(top)
    top = np.interp(np.arange(len(top)), np.flatnonzero(ok), top[ok])
    return clean, top


def map_hand(clean, source_top, curve, plot):
    x0, x1, y50, y0 = plot
    # The small natural tip cap is explicitly outside the observation interval.
    source_start, source_end, source_limit = 72., 2085., 2107.
    scale_x = (x1-x0)/(source_end-source_start)
    scale_y = scale_x * .84
    left = int(x0)
    right = math.ceil(x1 + (source_limit-source_end)*scale_x)
    upper = math.floor(min(curve(np.linspace(x0,x1,1000)))) - 3
    lower = 1239
    ww, hh = (right-left)*S, (lower-upper)*S
    xx = left + (np.arange(ww)+.5)/S
    yy = upper + (np.arange(hh)+.5)/S
    sx = source_start + (xx-x0)/scale_x
    boundary = curve(np.clip(xx,x0,x1))
    st = np.interp(np.minimum(sx,source_end), np.arange(len(source_top)), source_top)
    src_y = st[None,:] + (yy[:,None]-boundary[None,:])/scale_y
    src_x = np.broadcast_to(sx[None,:], src_y.shape)
    coords = [src_y, src_x]
    out = np.empty((hh,ww,4),dtype=np.uint8)
    for c in range(4):
        out[...,c] = np.clip(ndimage.map_coordinates(clean[...,c], coords, order=1,
                               mode="constant", cval=0, prefilter=False),0,255).astype(np.uint8)
    # Code owns the real silhouette. Keep the generated surface, fingers, palm,
    # and lower contour; fit only the upper ink edge to the PCHIP data geometry.
    depth = yy[:,None] - boundary[None,:]
    encoded = (xx <= x1)[None,:]
    out[...,3][encoded & (depth < 0)] = 0
    band = encoded & (depth >= 0) & (depth <= 2.15)
    out[band,:3] = (62,55,51)
    out[band,3] = 255
    mapped = Image.fromarray(out, "RGBA")
    mapped_small = mapped.resize((right-left,lower-upper),Image.Resampling.LANCZOS)
    mapped_small.save(ROOT / "assets/arm_mapped.png")
    return mapped, (left,upper), mapped_small, {
        "source_data_interval_x": [source_start,source_end],
        "destination_data_interval_x": [x0,x1],
        "source_to_destination_x_scale": scale_x,
        "source_to_destination_y_scale": scale_y,
        "mapped_asset_origin": [left,upper],
        "non_data_tip_cap_width_px": (source_limit-source_end)*scale_x,
    }


def main():
    data = json.loads((ROOT / "data.json").read_text(encoding="utf-8"))
    brief = json.loads((ROOT / "brief.json").read_text(encoding="utf-8"))
    w,h = brief["width"],brief["height"]
    assert (w,h) == (2000,1440)
    years = np.array(data["time_points"],dtype=float)
    values = np.array(data["series"][0]["values"],dtype=float)
    categories = data["categories"]
    assert len(years)==len(values)==len(categories)
    assert np.all(np.diff(years)>0)
    assert data["value_axis"] == {"minimum":0,"maximum":50,"scale":"linear"}
    x0,x1,y50,y0 = 250.,1770.,330.,940.
    xs = x0 + (years-years[0])/(years[-1]-years[0])*(x1-x0)
    ys = y0 - values/50*(y0-y50)
    curve = PchipInterpolator(xs,ys)

    # A lightly textured graph-paper sheet, with restrained charcoal rules.
    rng = np.random.default_rng(714)
    noise = rng.normal(0,.36,(h*S,w*S)).astype(np.float32)
    base = np.empty((h*S,w*S,3),dtype=np.uint8)
    for c,v in enumerate((238,237,233)):
        base[...,c] = np.clip(v+noise,0,255)
    canvas = Image.fromarray(base,"RGB").convert("RGBA")
    shadow = Image.new("RGBA",canvas.size,(0,0,0,0))
    sd = ImageDraw.Draw(shadow)
    sd.rounded_rectangle((72*S,72*S,1934*S,1400*S),radius=2*S,fill=(74,70,61,25))
    shadow = shadow.filter(ImageFilter.GaussianBlur(7*S))
    canvas.alpha_composite(shadow)
    d = ImageDraw.Draw(canvas)
    d.rectangle((68*S,61*S,1930*S,1389*S),fill=PAPER)
    for i,x in enumerate(range(96,1905,34)):
        line(d,[(x,82),(x,1365)],(214,215,211) if i%5==0 else (227,228,223),.75 if i%5==0 else .55)
    for i,y in enumerate(range(82,1370,34)):
        line(d,[(93,y),(1907,y)],(214,215,211) if i%5==0 else (227,228,223),.75 if i%5==0 else .55)

    text(d,(199,119),data["title"],73,serif=True,halo=3)
    text(d,(205,219),"Group A · 1918–2086",28,MUTED,halo=2)
    # A plain, explicit synthetic-data label, separate from the art.
    d.rounded_rectangle((1480*S,136*S,1841*S,191*S),radius=4*S,
                        fill=(240,232,238),outline=(162,132,158),width=1*S)
    text(d,(1660.5,164),data["visible_notice"],25,PURPLE,anchor="mm")

    # Shared linear value axis. The time axis sits below the non-data hand body.
    axis_x = 201
    line(d,[(axis_x,y50),(axis_x,y0)],INK,1.9)
    for value in range(0,51,10):
        y = y0-value/50*(y0-y50)
        line(d,[(axis_x-9,y),(axis_x+1,y)],INK,1.6)
        text(d,(axis_x-21,y),str(value),27,INK,anchor="rm",halo=2)
        if value:
            line(d,[(axis_x+2,y),(1817,y)],(198,200,194),.7)
    axis_title = Image.new("RGBA",(180*S,42*S),(0,0,0,0))
    ad = ImageDraw.Draw(axis_title)
    text(ad,(90,21),"比例（%）",25,INK,anchor="mm",halo=2)
    axis_title = axis_title.rotate(90,expand=True)
    canvas.alpha_composite(axis_title,(105*S,545*S))
    d = ImageDraw.Draw(canvas)
    # Zero remains visible wherever it is not occluded by the hand.
    for x in range(axis_x,1815,18):
        line(d,[(x,y0),(min(x+10,1817),y0)],(128,130,124),1)
    time_axis = 1265
    for x in xs:
        line(d,[(x,315),(x,time_axis)],(204,206,199),.75)
    line(d,[(x0,time_axis),(x1,time_axis)],INK,1.8)
    for x,cat in zip(xs,categories):
        line(d,[(x,time_axis),(x,time_axis+10)],INK,1.7)
        text(d,(x,time_axis+36),cat,27,INK,anchor="mm",halo=2)
    text(d,(1860,time_axis+34),"YEAR",24,MUTED,anchor="mm",halo=2)

    clean, source_top = prepare_source()
    arm, origin, arm_small, transform = map_hand(clean,source_top,curve,(x0,x1,y50,y0))
    canvas.alpha_composite(arm,(origin[0]*S,origin[1]*S))
    d = ImageDraw.Draw(canvas)

    # Numeric labels are data-derived, each attached to its contour node.
    label_boxes = []
    for i,(x,y,value) in enumerate(zip(xs,ys,values)):
        label = f"{value:.1f}%"
        local_box = d.textbbox((0,0),label,font=font(34),anchor="mm",stroke_width=3*S)
        label_xs = np.linspace(max(x0,x+local_box[0]/S),min(x1,x+local_box[2]/S),80)
        label_y = y-43
        label_y = min(label_y,float(np.min(curve(label_xs)))-local_box[3]/S-15)
        line(d,[(x,y-9),(x,label_y+19)],PURPLE,1.15)
        r = 5.5
        d.ellipse(((x-r)*S,(y-r)*S,(x+r)*S,(y+r)*S),fill=PURPLE,
                  outline=PAPER,width=1*S)
        text(d,(x,label_y),label,34,PURPLE,anchor="mm",halo=3)
        box = d.textbbox((x*S,label_y*S),label,font=font(34),anchor="mm",stroke_width=3*S)
        label_boxes.append({"category":categories[i],"value":label,"bbox_px":[round(v/S,2) for v in box]})

    line(d,[(1268,464),(1322,464)],PURPLE,2.5)
    d.ellipse((1288*S,459*S,1298*S,469*S),fill=PURPLE)
    text(d,(1343,465),data["series"][0]["label"],35,INK,anchor="lm",serif=True,halo=2)
    text(d,(1268,508),"沿着手的上缘读数",31,INK,halo=3)
    text(d,(1268,554),"0–50% · 线性量尺",25,MUTED,halo=2)

    # The lower silhouette and all finger widths are deliberately non-quantitative.
    text(d,(202,1351),"合成数据 · 上轮廓编码比例；下缘、宽度与其余手指不编码数值。",23,MUTED,anchor="lm",halo=2)

    out = canvas.convert("RGB").resize((w,h),Image.Resampling.LANCZOS)
    out.save(ROOT / "final.png")

    # Scan the actual antialiased mapped object's alpha, not just target coordinates.
    alpha = np.array(arm_small)[...,3]
    trace = []
    for xi in range(math.ceil(x0),math.floor(x1)+1):
        col = xi-origin[0]
        found = np.flatnonzero(alpha[:,col]>=128)
        if len(found):
            measured = origin[1] + int(found[0])
            trace.append([xi,measured,float(curve(xi))])
    trace = np.asarray(trace)
    residual = trace[:,1]-trace[:,2]
    measured_nodes = []
    for year,x,y,v in zip(years,xs,ys,values):
        col = int(round(x))-origin[0]
        found = np.flatnonzero(alpha[:,col]>=128)
        observed_y = origin[1]+int(found[0])
        observed_v = (y0-observed_y)/(y0-y50)*50
        measured_nodes.append({"year":int(year),"input_value":float(v),
            "x_px":round(float(x),4),"target_y_px":round(float(y),4),
            "observed_boundary_y_px":observed_y,"observed_value":round(observed_v,4),
            "absolute_value_error_pp":round(abs(observed_v-v),4)})
    mapped_arr = np.asarray(arm_small)
    nail = ((mapped_arr[...,2].astype(int)-mapped_arr[...,1].astype(int)>10)
          & (mapped_arr[...,0].astype(int)-mapped_arr[...,1].astype(int)>15)
          & (mapped_arr[...,3]>200))
    labs,n = ndimage.label(nail)
    sizes = np.bincount(labs.ravel());sizes[0]=0
    nail_components = [int(s) for s in sizes if s>=150]
    body_labels,n = ndimage.label(alpha>=128)
    body_sizes = np.bincount(body_labels.ravel()); body_sizes[0]=0
    all_dense_values = (y0-curve(np.linspace(x0,x1,10001)))/(y0-y50)*50
    report = {
        "canvas_px":[w,h],"encoding":"上轮廓；真实年份间距；0–50%共同线性纵轴；PCHIP无过冲插值",
        "sample_count":len(values),"category_order":categories,
        "time_spacing_px":[round(float(a),4) for a in np.diff(xs)],
        "last_interval_to_first_interval_ratio":round(float((xs[-1]-xs[-2])/(xs[1]-xs[0])),6),
        "measured_nodes":measured_nodes,
        "raster_upper_edge_check":{"alpha_threshold":128,"sampled_columns":len(trace),
            "max_absolute_error_px":round(float(np.max(np.abs(residual))),4),
            "max_absolute_error_pp":round(float(np.max(np.abs(residual))/(y0-y50)*50),4)},
        "interpolation_nonincreasing":bool(np.all(np.diff(all_dense_values)<=1e-9)),
        "interpolation_value_range":[round(float(all_dense_values.min()),4),round(float(all_dense_values.max()),4)],
        "visible_purple_nail_components":len(nail_components),"nail_component_areas_px":nail_components,
        "large_connected_body_components":int(np.sum(body_sizes>200)),
        "numeric_labels":label_boxes,"transform":transform,
        "measurement_scope":"从实际输出物件透明度扫描上缘；节点标记和文字不计入物体边界。",
    }
    save_json(ROOT / "artifacts/geometry_check.json",report)


if __name__ == "__main__":
    main()
