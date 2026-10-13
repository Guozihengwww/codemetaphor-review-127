#!/usr/bin/env python3
"""Offline, deterministic compositor for the frozen sand-extraction case.

Run from this directory: PYTHON render.py
For this experiment, all complete renders are invoked by capture_render.py.
"""
from pathlib import Path
import json
import math
import hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent
S = 2
BG = (246, 245, 240)
INK = (35, 43, 43)
MUTED = (102, 108, 106)
ACCENT = (168, 108, 48)
EDGE = (135, 86, 39)


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def font(size, bold=False, numeric=False):
    name = "DejaVuSans.ttf" if numeric else ("NimbusSans-Bold.otf" if bold else "NimbusSans-Regular.otf")
    return ImageFont.truetype(str(ROOT / "assets/fonts" / name), round(size * S))


def main():
    data, brief, core, inputs = [read_json(n) for n in ("data.json", "brief.json", "core.json", "input.json")]
    assert core["topic_zh"] == data["topic_zh"]
    for name in inputs.values():
        assert (ROOT / name).exists(), name
    assert data["value_axis"] == {"minimum": 0, "maximum": 400, "scale": "linear"}
    assert len(data["series"]) == 1
    series = data["series"][0]
    categories, values = data["categories"], series["values"]
    assert len(categories) == len(values) == len(data["time_points"])
    W, H = brief["width"], brief["height"]
    assert (W, H) == (2000, 2000)
    width, height = W * S, H * S
    page = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(page)
    label_boxes = []

    def line(points, fill, w=1):
        draw.line([(round(x*S), round(y*S)) for x, y in points], fill=fill, width=round(w*S))

    def txt(x, y, text, size, fill=INK, bold=False, align="left", numeric=False, role=None):
        f = font(size, bold, numeric)
        length = draw.textlength(text, font=f)
        xx = x*S - (length/2 if align == "center" else length if align == "right" else 0)
        # Anchor to visual top, making text positions predictable across fonts.
        draw.text((round(xx), round(y*S)), text, font=f, fill=fill, anchor="lt")
        bb = draw.textbbox((round(xx), round(y*S)), text, font=f, anchor="lt")
        label_boxes.append({"text": text, "role": role, "bbox": [round(v/S, 2) for v in bb]})

    # Editorial header; the badge is deliberately prominent.
    draw.rectangle((120*S, 93*S, 139*S, 112*S), fill=ACCENT)
    txt(158, 91, "MATERIALS / ANNUAL EXTRACTION", 25, MUTED, bold=True)
    txt(120, 172, "Sand and gravel", 112, bold=True)
    txt(120, 287, "extraction", 112, bold=True)
    txt(124, 430, series["label"], 34, MUTED, role="series")
    draw.rounded_rectangle((1480*S, 91*S, 1878*S, 154*S), radius=5*S, fill=(234, 213, 181))
    txt(1679, 111, data["visible_notice"], 28, (100, 65, 30), bold=True, align="center", role="synthetic_notice")
    txt(1878, 199, f"{categories[0]}–{categories[-1]}", 49, INK, align="right")
    txt(1878, 265, "Annual series", 28, MUTED, align="right")
    line([(120, 518), (1880, 518)], (209, 211, 205), 1.2)

    # One mapping, used for the material mask, rim, nodes, labels and checks.
    x0, x1 = 250.0, 1840.0
    y0, y400 = 1580.0, 650.0
    span = y0 - y400
    xs = np.linspace(x0, x1, len(values))
    ys = y0 - np.asarray(values, dtype=float) / 400.0 * span
    points = list(zip(xs.tolist(), ys.tolist()))
    poly = points + [(x1, y0), (x0, y0)]

    txt(120, 587, series["unit"], 29, MUTED, role="unit")
    for value in (0, 100, 200, 300, 400):
        yy = y0 - value / 400.0 * span
        line([(220, yy), (1880, yy)], (211, 215, 209), 1.2)
        txt(191, yy-18, str(value), 32, MUTED, align="right", numeric=True, role="axis_tick")

    mask = Image.new("L", page.size, 0)
    md = ImageDraw.Draw(mask)
    md.polygon([(round(x*S), round(y*S)) for x,y in poly], fill=255)

    # Soft contact shadow under the common zero line, not a second data surface.
    ground_shadow = Image.new("L", page.size, 0)
    gd = ImageDraw.Draw(ground_shadow)
    gd.ellipse((int((x0-4)*S), int((y0-9)*S), int((x1+8)*S), int((y0+24)*S)), fill=60)
    ground_shadow = ground_shadow.filter(ImageFilter.GaussianBlur(12*S))
    page.paste((91, 76, 54), (0,0), ground_shadow)

    # Generated sand texture supplies the full visible sand face.
    texture = Image.open(ROOT / "assets/sand_texture.png").convert("RGB")
    material_width, material_height = round((x1-x0)*S)+2, round(span*S)+2
    texture = texture.resize((material_width, material_height), Image.Resampling.LANCZOS)
    arr = np.asarray(texture).astype(np.float32)
    tx = np.arange(material_width) / S + x0
    ty = np.arange(material_height) / S + y400
    surface = np.interp(tx, xs, ys)
    depth = np.maximum(0, ty[:,None] - surface[None,:])
    local_h = np.maximum(1, y0-surface)[None,:]
    frac = np.clip(depth / local_h, 0, 1)
    # Subtle relief: a lit upper lip and a gently darker toe, all inside the exact mask.
    tone = 1.018 - 0.085*frac + 0.045*np.exp(-depth/15.0)
    arr = np.clip(arr*tone[...,None], 0, 255).astype(np.uint8)
    material = Image.new("RGB", page.size, BG)
    material.paste(Image.fromarray(arr), (round(x0*S), round(y400*S)))
    page.paste(material, (0,0), mask)

    # A few loose grains sit outside the silhouette; none bridge the data rim.
    rng = np.random.default_rng(28)
    grain_count = 0
    for i in range(105):
        gx = rng.uniform(x0-13, x1+13)
        gy = rng.uniform(y0+9, y0+39)
        r = rng.uniform(0.8, 3.2)
        if i < 20:
            gx = rng.uniform(x0+35, x1-35)
            rim = float(np.interp(gx, xs, ys))
            gy = rim-rng.uniform(19, 68)
            # Keep grit away from the numeric labels.
            if any(abs(gx-x)<60 and abs(gy-(y-49))<40 for x,y in points):
                continue
        col = [(181,142,90), (203,166,115), (151,117,75), (223,193,149)][i%4]
        d = ImageDraw.Draw(page)
        d.ellipse((round((gx-r)*S), round((gy-r*.65)*S), round((gx+r)*S), round((gy+r*.65)*S)), fill=col)
        grain_count += 1

    # The generated shovel is an integrated, non-quantitative object.
    shovel = Image.open(ROOT / "assets/shovel.png").convert("RGBA")
    shovel_h = 650
    shovel_w = round(shovel.width / shovel.height * shovel_h)
    shovel = shovel.resize((shovel_w*S, shovel_h*S), Image.Resampling.LANCZOS)
    shovel_x, shovel_y = 600, 840
    sa = Image.new("L", page.size, 0)
    sa.paste(shovel.getchannel("A"), (shovel_x*S, shovel_y*S))
    shadow = Image.new("L", page.size, 0)
    shadow.paste(shovel.getchannel("A"), ((shovel_x+9)*S, (shovel_y+11)*S))
    shadow = shadow.filter(ImageFilter.GaussianBlur(6*S))
    sh = np.asarray(shadow, dtype=np.uint16)
    sh = ((sh * np.asarray(mask, dtype=np.uint16) // 255) * 36 // 100).astype(np.uint8)
    page.paste((67, 50, 29), (0,0), Image.fromarray(sh))
    page.paste(shovel, (shovel_x*S, shovel_y*S), shovel)

    # Sand crosses the lower blade to establish insertion/contact.
    cover = Image.new("L", page.size, 0)
    cd = ImageDraw.Draw(cover)
    bury_y = shovel_y + shovel_h - 35
    cd.polygon([(shovel_x*S, (bury_y+5)*S), ((shovel_x+shovel_w*.3)*S, (bury_y-3)*S),
                ((shovel_x+shovel_w*.65)*S, (bury_y+1)*S), ((shovel_x+shovel_w)*S, (bury_y-6)*S),
                ((shovel_x+shovel_w)*S, (shovel_y+shovel_h+2)*S), (shovel_x*S, (shovel_y+shovel_h+2)*S)], fill=255)
    cover_arr = (np.asarray(cover, dtype=np.uint16)*np.asarray(sa, dtype=np.uint16)//255).astype(np.uint8)
    cover_arr = (cover_arr.astype(np.uint16)*np.asarray(mask, dtype=np.uint16)//255).astype(np.uint8)
    page.paste(material, (0,0), Image.fromarray(cover_arr))

    # The explicit data rim remains legible through the local object overlap.
    draw = ImageDraw.Draw(page)
    line(points, EDGE, 3)
    line([(x0,y0), (x1,y0)], INK, 2.1)
    for category, value, x, y in zip(categories, values, xs, ys):
        line([(x,y-13), (x,y-25)], (146,127,97), 1.2)
        draw.ellipse((round((x-5)*S), round((y-5)*S), round((x+5)*S), round((y+5)*S)), fill=BG, outline=EDGE, width=2*S)
        txt(x, y-69, str(value), 40, INK, bold=True, align="center", role="data_value")
        line([(x,y0+3), (x,y0+13)], (142,147,139), 1.2)
        txt(x, y0+45, category, 32, INK, align="center", numeric=True, role="year")

    # Footer explains the measured relationship without inventing provenance.
    line([(120,1749), (1880,1749)], (209,211,205), 1.2)
    txt(120, 1792, "READING THE SURFACE", 25, ACCENT, bold=True)
    txt(120, 1840, "Height above zero shows annual extraction.", 29, MUTED)
    txt(120, 1884, "The shovel and loose grains are illustrative.", 29, MUTED)
    txt(1880, 1792, "SYNTHETIC SERIES", 25, ACCENT, bold=True, align="right")
    txt(1880, 1840, "Illustrative values for a local study.", 29, MUTED, align="right")
    txt(1880, 1884, "Not measured extraction statistics.", 29, MUTED, align="right")

    final = page.resize((W,H), Image.Resampling.LANCZOS)
    final.save(ROOT / "final.png")
    # Diagnostic alpha is the actual rasterized material boundary, not another chart.
    mask.resize((W,H), Image.Resampling.LANCZOS).save(ROOT / "artifacts/sand_mask.png")
    geometry = {
        "canvas": [W,H], "axis_min": 0, "axis_max": 400,
        "plot": {"x_first":x0,"x_last":x1,"zero_y":y0,"max_y":y400,"pixels_per_unit":span/400},
        "nodes": [{"category":c,"value":v,"x":x,"y":y} for c,v,x,y in zip(categories,values,xs.tolist(),ys.tolist())],
        "labels":label_boxes, "loose_grains_drawn":grain_count,
        "shovel_bbox":[shovel_x,shovel_y,shovel_x+shovel_w,shovel_y+shovel_h],
        "shovel_buried_below_y":bury_y,
        "data_sha256":hashlib.sha256((ROOT/"data.json").read_bytes()).hexdigest(),
    }
    (ROOT/"artifacts/render_geometry.json").write_text(json.dumps(geometry,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("Wrote final.png (2000 x 2000), offline generated-material composition.")


if __name__ == "__main__":
    main()
