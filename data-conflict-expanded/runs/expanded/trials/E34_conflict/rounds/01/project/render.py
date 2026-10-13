#!/usr/bin/env python3
"""Offline, deterministic data-bound arm infographic. Run from this directory."""
from pathlib import Path
import json
import math
import hashlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import map_coordinates, label, binary_opening

ROOT = Path(__file__).resolve().parent
S = 2


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def save_json(path, value):
    (ROOT / path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def font(size, face="regular"):
    name = {"regular": "NimbusSans-Regular.otf", "bold": "NimbusSans-Bold.otf", "serif": "Bookman-Light.otf"}[face]
    return ImageFont.truetype(str(ROOT / "assets/fonts" / name), round(size * S))


def text(draw, xy, content, size, fill, face="regular", anchor="lt"):
    draw.text(tuple(round(v * S) for v in xy), content, font=font(size, face), fill=fill, anchor=anchor)


def line(draw, points, fill, width=1):
    draw.line([(round(x*S), round(y*S)) for x, y in points], fill=fill, width=round(width*S), joint="curve")


def dashed(draw, start, end, color, width=1, dash=10, gap=8):
    x0, y0 = start
    x1, y1 = end
    length = math.hypot(x1-x0, y1-y0)
    for d in np.arange(0, length, dash+gap):
        e = min(d+dash, length)
        line(draw, [(x0+(x1-x0)*d/length, y0+(y1-y0)*d/length),
                    (x0+(x1-x0)*e/length, y0+(y1-y0)*e/length)], color, width)


def rounded_rect(draw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(tuple(round(v*S) for v in box), radius=round(radius*S), fill=fill, outline=outline, width=round(width*S))


def build_hand(xs, ys, curve):
    """Column map: x is linear time; first opaque source edge is mapped to y(value).

    Beyond the observed endpoint, the fixed transform preserves the natural
    rounded fingertip. That cap carries no additional observation or time.
    """
    source = np.array(Image.open(ROOT / "assets/hand_clean.png").convert("RGBA"))
    source_alpha = source[:, :, 3].astype(np.float32) / 255
    sh, sw = source_alpha.shape
    occupied = source_alpha > .5
    first = occupied.argmax(axis=0).astype(np.float32)
    occupied_columns = occupied.any(axis=0)
    source_x = np.arange(sw, dtype=np.float32)
    first = np.interp(source_x, source_x[occupied_columns], first[occupied_columns])

    # Anatomical endpoint is on the uppermost fingertip, before its rounded cap.
    source_endpoint = 1915.0
    source_last = 1959.0
    x_scale = (xs[-1]-xs[0]) / source_endpoint
    thickness_scale = .64  # Non-quantitative anatomy only; does not affect the top edge.
    left = int(xs[0]*S)
    right = int(math.ceil((xs[0]+source_last*x_scale)*S))+2
    top = int((min(ys)-5)*S)
    bottom = 1300*S
    dest_x = np.arange(left, right, dtype=np.float32) / S
    dest_y = np.arange(top, bottom, dtype=np.float32) / S
    src_x = (dest_x-xs[0])/x_scale
    clamped_x = np.minimum(dest_x, xs[-1])
    target_y = curve(clamped_x).astype(np.float32)
    first_y = np.interp(np.minimum(src_x, source_endpoint), source_x, first).astype(np.float32)
    # Freeze the translation beyond the last data point so tips remain rounded.
    src_y = first_y[None, :] + (dest_y[:, None]-target_y[None, :])/thickness_scale
    src_x_grid = np.broadcast_to(src_x[None, :], src_y.shape)
    coordinates = np.array([src_y, src_x_grid], dtype=np.float32)
    alpha = map_coordinates(source_alpha, coordinates, order=1, mode="constant", cval=0)
    clip = (dest_y[:, None] >= target_y[None, :]) | (dest_x[None, :] > xs[-1])
    alpha *= clip
    rgba = np.empty((*alpha.shape, 4), dtype=np.uint8)
    # Premultiplied sampling avoids dark fringes at transparent finger gaps.
    for c in range(3):
        premul = source[:, :, c].astype(np.float32)*source_alpha
        sampled = map_coordinates(premul, coordinates, order=1, mode="constant", cval=0)
        rgba[:, :, c] = np.clip(np.divide(sampled, alpha, out=np.zeros_like(sampled), where=alpha>.005), 0, 255).astype(np.uint8)
    rgba[:, :, 3] = np.clip(alpha*255, 0, 255).astype(np.uint8)
    hand = Image.fromarray(rgba, "RGBA")

    # A fine inward contour gives the generated skin a crisp quantitative boundary.
    hd = ImageDraw.Draw(hand)
    contour_x = np.arange(xs[0], xs[-1]+.25, .25)
    contour_y = curve(contour_x)
    contour = [(round(x*S-left), round((y+1.35)*S-top)) for x, y in zip(contour_x, contour_y)]
    hd.line(contour, fill=(74, 47, 39, 255), width=round(2.7*S), joint="curve")
    alpha_after = np.array(hand.getchannel("A"))
    alpha_after[~clip] = 0
    hand.putalpha(Image.fromarray(alpha_after))

    # Measure the RASTER contour, not just the analytic target.
    measurements = []
    for x, y in zip(xs, ys):
        c = round(x*S)-left
        visible = np.flatnonzero(alpha_after[:, c] >= 128)
        measured = (top + int(visible[0]))/S
        measurements.append({"x_px": float(x), "target_y_px": float(y),
                             "raster_top_y_px": measured,
                             "edge_error_px": round(measured-float(y), 5)})
    # The interior purple regions remain visible after the actual warp.
    warped = np.array(hand)
    r, g, b = [warped[:, :, c].astype(np.float32) for c in range(3)]
    purple = (warped[:, :, 3]>200) & (b>g*1.05) & (r>g*1.08) & (r<220) & (g<165)
    purple = binary_opening(purple, iterations=2)
    regions, count = label(purple)
    nails = []
    for n in range(1, count+1):
        yy, xx = np.where(regions == n)
        if len(xx) > 400:
            nails.append({"area_px2": round(len(xx)/(S*S), 2),
                          "center_px": [round((xx.mean()+left)/S,2),round((yy.mean()+top)/S,2)],
                          "bbox_px": [round((xx.min()+left)/S,2),round((yy.min()+top)/S,2),
                                      round((xx.max()+left)/S,2),round((yy.max()+top)/S,2)]})
    return hand, (left, top), measurements, nails


def main():
    inputs = read_json("input.json")
    brief = read_json(inputs["brief"])
    data = read_json(inputs["data"])
    core = read_json(inputs["critical_metaphor_contract"])
    assert (ROOT / inputs["reference"]).is_file()
    assert "Women parliamentary seats (%)" in core["semantic_roles_zh"]
    assert len(data["series"]) == 1
    assert data["value_axis"] == {"minimum": 0, "maximum": 50, "scale": "linear"}
    w, h = brief["width"], brief["height"]
    assert (w, h) == (2000, 1440)
    series = data["series"][0]
    years = np.array(data["time_points"], dtype=float)
    values = np.array(series["values"], dtype=float)
    categories = data["categories"]
    assert len(years) == len(values) == len(categories)
    assert np.all(np.diff(years)>0)
    x0, x1 = 246.0, 1790.0
    y50, y0 = 348.0, 1048.0
    xs = x0+(years-years[0])/(years[-1]-years[0])*(x1-x0)
    ys = y0-values/50*(y0-y50)
    curve = PchipInterpolator(xs, ys, extrapolate=False)

    ink = (46, 43, 43)
    muted = (101, 102, 105)
    purple = (114, 76, 127)
    light_purple = (162, 136, 169)
    paper = (247, 247, 244)
    # Deterministic, barely visible paper grain.
    rng = np.random.default_rng(34)
    grain = rng.normal(0, .6, (h*S, w*S, 1)).astype(np.float32)
    background = np.clip(np.array(paper)[None,None,:]+grain, 0, 255).astype(np.uint8)
    canvas = Image.fromarray(background, "RGB").convert("RGBA")
    d = ImageDraw.Draw(canvas)

    # Fine square paper grid; the numeric axes have their own shared linear scale.
    for x in np.arange(78, 1924, 28):
        line(d, [(x,273),(x,1342)], (222,224,222), .65)
    for y in np.arange(278, 1343, 28):
        line(d, [(78,y),(1924,y)], (222,224,222), .65)

    rounded_rect(d, (94,55,355,91), 3, purple)
    text(d, (112,63), data["visible_notice"], 22, (255,255,255), "bold")
    text(d, (93,114), series["label"], 83, ink, "serif")
    text(d, (96,214), "Women members of parliament · share of total seats (%)", 28, muted)
    text(d, (1906,122), f"{categories[0]}—{categories[-1]}", 35, ink, "regular", "rt")
    text(d, (1906,175), "Synthetic observation years", 22, muted, anchor="rt")

    # Shared percent scale and the benchmark.
    for value in range(0,51,10):
        y = y0-value/50*(y0-y50)
        if value == 50:
            dashed(d, (x0,y),(1872,y), light_purple, 1.7, 12, 9)
        else:
            line(d, [(x0,y),(1872,y)], (185,189,187) if value else (139,145,142), 1.1 if value else 1.5)
        line(d, [(x0-10,y),(x0,y)], (106,111,108), 1.4)
        text(d, (x0-23,y), f"{value}%", 25, purple if value==50 else muted, anchor="rm")
    line(d, [(x0,y50),(x0,y0)], (122,126,123), 1.4)
    text(d, (97,289), "SEAT SHARE", 19, muted, "bold")
    rounded_rect(d, (1272,301,1877,337), 1, paper)
    text(d, (1872,309), "50%  /  EQUAL REPRESENTATION", 24, purple, "bold", "rt")

    # Every date uses its actual numerical interval. Guides stop behind the anatomy.
    for x in xs:
        dashed(d, (x,y50+14),(x,1284), (202,205,202), .75, 3, 7)

    hand, offset, measurements, nails = build_hand(xs, ys, curve)
    canvas.alpha_composite(hand, offset)
    d = ImageDraw.Draw(canvas)

    # Numeric labels and observation markers are entirely code-controlled.
    label_boxes = []
    for i, (x,y,v,category) in enumerate(zip(xs,ys,values,categories)):
        label_y = y-51
        label_x = x + (60 if i==0 else 0)
        label_text = f"{v:.1f}%"
        f = font(31,"bold")
        tw = d.textlength(label_text, font=f)/S
        rounded_rect(d, (label_x-tw/2-7,label_y-2,label_x+tw/2+7,label_y+35), 3, paper)
        text(d, (label_x,label_y), label_text, 31, ink, "bold", "mt")
        if i == 0:
            line(d, [(x+34,label_y+37),(x+7,y-12),(x,y-5)], purple, 1.25)
        else:
            line(d, [(x,y-13),(x,y-5)], purple, 1.25)
        d.ellipse((round((x-3.5)*S),round((y-3.5)*S),round((x+3.5)*S),round((y+3.5)*S)), fill=purple, outline=paper, width=round(S))
        label_boxes.append({"year":category,"text":label_text,"bbox_px":[round(label_x-tw/2-7,2),round(label_y-2,2),round(label_x+tw/2+7,2),round(label_y+35,2)]})

    # A derived, explicitly dated reading occupies the open space above the hand.
    gap = data["semantic_target"]["benchmark_percent"]-values[-1]
    rounded_rect(d, (1274,492,1850,784), 0, (247,247,244))
    line(d, [(1274,493),(1274,777)], light_purple, 3)
    text(d, (1305,510), "GAP TO THE 50% TARGET", 22, purple, "bold")
    text(d, (1301,552), f"{gap:.1f}", 88, purple, "serif")
    text(d, (1305,655), "percentage points", 30, ink)
    text(d, (1305,709), f"{categories[-1]} · {values[-1]:.1f}% of parliamentary seats", 24, muted)

    # Time labels sit below the non-quantitative palm and fingers.
    line(d, [(x0,1307),(x1,1307)], (139,145,142), 1.25)
    for x, year in zip(xs,categories):
        line(d, [(x,1307),(x,1317)], (106,111,108), 1.4)
        text(d, (x,1332), year, 27, ink, anchor="mt")
    text(d, (1904,1332), "YEAR", 21, muted, "bold", "rt")
    text(d, (96,1390), "SYNTHETIC DATA · Illustrative values, not a historical record or forecast.", 21, muted)
    text(d, (1905,1390), "Upper edge = seat share. Other anatomy is illustrative.", 21, muted, anchor="rt")

    final = canvas.convert("RGB").resize((w,h), Image.Resampling.LANCZOS)
    final.save(ROOT / "final.png", optimize=True)

    samples = np.linspace(xs[0],xs[-1],10001)
    cy = curve(samples)
    max_error = max(abs(m["edge_error_px"]) for m in measurements)
    for i, m in enumerate(measurements):
        m.update(year=categories[i], input_value_pct=float(values[i]),
                 raster_inferred_pct=round((y0-m["raster_top_y_px"])/(y0-y50)*50,6))
    save_json("artifacts/geometry_checks.json", {
        "canvas_px":[w,h], "supersample":S,
        "mapping":{"x_start":x0,"x_end":x1,"year_start":float(years[0]),"year_end":float(years[-1]),"y_50":y50,"y_0":y0,"pixels_per_percentage_point":(y0-y50)/50},
        "interpolation":"scipy.interpolate.PchipInterpolator, no overshoot",
        "raster_measurement":"First alpha >= 128 sample of the actual warped and inward-outlined hand, measured before chart markers; coordinates in final-image pixels.",
        "nodes":measurements, "max_raster_edge_error_px":round(max_error,6),
        "max_raster_edge_error_percentage_points":round(max_error/14,6),
        "curve_monotone_descending_values":bool(np.all(np.diff(cy)>=-1e-8)),
        "curve_stays_within_observed_range":bool(cy.min()>=ys.min()-1e-8 and cy.max()<=ys.max()+1e-8),
        "nail_color_component_count":len(nails), "nail_color_components":nails,
        "numeric_label_boxes":label_boxes,
        "endpoint_cap_zh":"2086节点之后仅保留指尖自然圆钝收尾，时间轴和观测点在2086终止。",
        "frozen_input_sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ["input.json","brief.json","data.json","core.json","reference.png"]},
        "final_sha256":hashlib.sha256((ROOT/"final.png").read_bytes()).hexdigest()
    })


if __name__ == "__main__":
    main()
