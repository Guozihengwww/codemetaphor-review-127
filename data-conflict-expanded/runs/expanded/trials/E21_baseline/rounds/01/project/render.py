#!/usr/bin/env python3
"""Offline watercolor forest infographic. Run from this directory, no arguments.

Only data.json determines the numerical path, order, axis range, and statistics.
The generated forest is a non-quantitative overlay; its silhouette is unmodified.
Complete experimental renders must be invoked via capture_render.py.
"""
from pathlib import Path
import json
import math
import hashlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
S = 2


def load_json(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def save_json(name, value):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    inputs = load_json("input.json")
    brief = load_json(inputs["brief"])
    data = load_json(inputs["data"])
    contract = load_json(inputs["critical_metaphor_contract"])
    assert (ROOT / inputs["reference"]).is_file()  # Preserved design source, never composited.
    assert data["synthetic"] and data["value_axis"]["scale"] == "linear"
    assert contract["topic_zh"] == data["topic_zh"]
    W, H = brief["width"], brief["height"]
    assert (W, H) == (2000, 1600)
    series = data["series"][0]
    assert len(data["series"]) == 1
    years = np.asarray(data["time_points"], dtype=float)
    values = np.asarray(series["values"], dtype=float)
    cats = data["categories"]
    assert len(cats) == len(values) == len(years)
    assert all(float(cat) == year for cat, year in zip(cats, years))
    assert np.all(np.diff(years) > 0)
    low, high = data["value_axis"]["minimum"], data["value_axis"]["maximum"]
    assert low == 0 and high == 8000000
    assert np.all((values >= low) & (values <= high))
    assert int(values.sum()) == data["series_totals"][series["id"]]

    # One shared transform, applied without smoothing or magnitude clamping.
    plot = {"left": 225.0, "right": 1895.0, "top": 360.0, "bottom": 1300.0}
    def px(year):
        return plot["left"] + (year - years[0]) / (years[-1] - years[0]) * (plot["right"] - plot["left"])
    def py(value):
        return plot["bottom"] - (value - low) / (high - low) * (plot["bottom"] - plot["top"])
    points = [(float(px(y)), float(py(v))) for y, v in zip(years, values)]
    paper = (232, 240, 230)
    ink = "#1c4037"
    muted = "#506d60"
    terracotta = "#c64a34"
    faint = "#b6cabc"

    # Subtle mint paper. Fixed seed ensures offline regeneration is byte stable.
    yy, xx = np.mgrid[0:H, 0:W]
    glow = np.exp(-(((xx - 1050) / 1300) ** 2 + ((yy - 700) / 1200) ** 2))
    rng = np.random.default_rng(214)
    grain = rng.normal(0, .45, (H, W))
    base = np.empty((H, W, 3), dtype=np.uint8)
    for c, (edge, bright) in enumerate(zip((218, 234, 222), (239, 244, 233))):
        base[:, :, c] = np.clip(edge + glow * (bright - edge) + grain, 0, 255).astype(np.uint8)
    im = Image.fromarray(base).convert("RGBA").resize((W*S, H*S), Image.Resampling.BICUBIC)

    font_cache = {}
    def font(size, bold=False, italic=False):
        name = "LiberationSans-Italic.ttf" if italic else ("LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf")
        key = (name, size)
        if key not in font_cache:
            font_cache[key] = ImageFont.truetype(str(ROOT / "assets/fonts" / name), round(size*S))
        return font_cache[key]
    draw = ImageDraw.Draw(im)
    text_records = []
    def text(x, y, content, size=26, color=ink, bold=False, anchor="ls", halo=0, italic=False):
        f = font(size, bold, italic)
        draw.text((x*S, y*S), str(content), font=f, fill=color, anchor=anchor,
                  stroke_width=round(halo*S), stroke_fill="#eaf1e6")
        box = draw.textbbox((x*S, y*S), str(content), font=f, anchor=anchor,
                            stroke_width=round(halo*S))
        text_records.append({"text": str(content), "bbox": [round(v/S, 2) for v in box]})
    def line(coords, color, width=1):
        draw.line([(round(x*S), round(y*S)) for x, y in coords], fill=color, width=round(width*S), joint="curve")
    def circle(x, y, radius, fill, outline=None, width=1):
        draw.ellipse(((x-radius)*S, (y-radius)*S, (x+radius)*S, (y+radius)*S),
                     fill=fill, outline=outline, width=round(width*S))
    def tracked(x, y, content, size=20, spacing=2, color=muted, bold=True):
        for ch in content:
            text(x, y, ch, size=size, color=color, bold=bold)
            x += draw.textlength(ch, font=font(size, bold))/S + spacing

    # Header and explicit study-data notice.
    draw.rounded_rectangle((90*S, 57*S, 383*S, 104*S), radius=5*S, fill=terracotta)
    text(236.5, 89, data["visible_notice"], size=24, color="#fff9ed", bold=True, anchor="ms")
    text(88, 202, "Annual forest fire area", size=91, bold=True)
    text(94, 251, "Synthetic area burned, measured annually in hectares", size=29, color=muted)
    text(1906, 153, f"{cats[0]}–{cats[-1]}", size=43, bold=True, anchor="rs")
    text(1906, 197, f"{len(years)} annual observations", size=25, color=muted, anchor="rs")
    line([(94, 278), (1906, 278)], "#9cb7a6", 1.2)
    text(94, 326, "AREA BURNED  /  HECTARES", size=23, color=muted, bold=True)
    line([(1537, 319), (1587, 319)], terracotta, 7.5)
    text(1604, 327, series["label"], size=25, color=ink)

    # Large, complete generated forest occupying the same space as the line.
    # Uniform aspect-preserving scale and translation; no data-driven tree warps.
    forest = Image.open(ROOT / "assets/forest_watercolor.png").convert("RGBA")
    forest_w = 1750
    forest_h = round(forest.height * forest_w / forest.width)
    forest_xy = (188, 391)
    forest = forest.resize((forest_w*S, forest_h*S), Image.Resampling.LANCZOS)
    im.alpha_composite(forest, (forest_xy[0]*S, forest_xy[1]*S))

    # Restrained grid above the watercolor to retain visible units over foliage.
    grid = Image.new("RGBA", im.size)
    gd = ImageDraw.Draw(grid)
    tick_values = list(range(low, high + 1, 1000000))
    major_years = [int(years[0])] + [int(y) for y in years if int(y) % 5 == 0 and y > years[0]+2 and y < years[-1]-2] + [int(years[-1])]
    for v in tick_values:
        gy = py(v)
        gd.line([(plot["left"]*S, gy*S), (plot["right"]*S, gy*S)], fill=(61, 101, 79, 38), width=2)
    for year in major_years:
        gx = px(year)
        gd.line([(gx*S, plot["top"]*S), (gx*S, plot["bottom"]*S)], fill=(62, 103, 80, 22), width=2)
    im.alpha_composite(grid)
    draw = ImageDraw.Draw(im)
    line([(plot["left"], plot["bottom"]), (plot["right"], plot["bottom"])], "#738f7c", 1.5)
    for v in tick_values:
        label = "0" if v == 0 else f"{v//1000000}M"
        text(198, py(v), label, size=27, color=muted, anchor="rm")
    for year in years:
        gx = px(year)
        line([(gx, plot["bottom"]+3), (gx, plot["bottom"]+11)], "#8fa996", 1)
    for year in major_years:
        gx = px(year)
        line([(gx, plot["bottom"]), (gx, plot["bottom"]+15)], "#597e65", 1.5)
        text(gx, 1350, str(year), size=26, anchor="ms")
    text(1895, 1393, "Year", size=24, color=muted, anchor="rs")
    text(225, 1393, "M = million hectares  ·  One vertex per year", size=23, color=muted)

    # Chart foreground: all 61 vertices, precisely spaced by the supplied years.
    halo_layer = Image.new("RGBA", im.size)
    hd = ImageDraw.Draw(halo_layer)
    hd.line([(x*S, y*S) for x, y in points], fill=(242, 246, 235, 177), width=round(13*S), joint="curve")
    im.alpha_composite(halo_layer)
    draw = ImageDraw.Draw(im)
    line(points, terracotta, 8)
    for x, y in points:
        circle(x, y, 3.9, terracotta)

    # Peak / endpoint callouts are read from the same series and coordinates.
    max_i = int(values.argmax())
    min_i = int(values.argmin())
    runner_up = int(np.argsort(values)[-2])
    label_specs = [
        (max_i, points[max_i][0], points[max_i][1]-59, "ms", "highest", None),
        (runner_up, points[runner_up][0]+52, points[runner_up][1]-40, "ls", "", "right"),
    ]
    for idx, tx, ty, anch, qualifier, leader in label_specs:
        x, y = points[idx]
        if leader:
            line([(tx-11, ty+15), (x+8, y-7)], "#48694f", 1.5)
        else:
            line([(x, ty+39), (x, y-11)], "#48694f", 1.5)
        title = cats[idx] + (f" · {qualifier}" if qualifier else "")
        text(tx, ty, title, size=23, color=ink, bold=True, anchor=anch, halo=3)
        text(tx, ty+34, f"{int(values[idx]):,} ha", size=29, color=ink, bold=True, anchor=anch, halo=3)
        circle(x, y, 7, "#f5f5e8", terracotta, 3)
        circle(x, y, 2.7, terracotta)

    # Compact numerical context below the image. All numbers are computed.
    line([(94, 1423), (1906, 1423)], "#9cb7a6", 1.2)
    footer = [
        (94, f"TOTAL OVER {len(years)} YEARS", f"{int(values.sum()):,} ha", "Sum of the annual values"),
        (808, f"HIGHEST YEAR · {cats[max_i]}", f"{int(values[max_i]):,} ha", "Largest annual burned area"),
        (1432, f"LOWEST YEAR · {cats[min_i]}", f"{int(values[min_i]):,} ha", "Smallest annual burned area"),
    ]
    for x, title, value, explanation in footer:
        tracked(x, 1466, title, size=19, spacing=1.5)
        text(x, 1513, value, size=42, bold=True)
        text(x, 1546, explanation, size=23, color=muted)
    text(94, 1582, "Synthetic study data. Forest height and texture are illustrative; only the red line encodes area burned.", size=19, color=muted)

    final = im.convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
    final.save(ROOT / "final.png", optimize=True)

    save_json("artifacts/render_geometry.json", {
        "canvas": [W, H], "series_id": series["id"], "unit": series["unit"],
        "plot": plot, "axis": data["value_axis"], "line_color": terracotta,
        "line_width_px": 8, "vertex_count": len(points), "segment_count": len(points)-1,
        "x_spacing_px_per_year": (plot["right"]-plot["left"])/(years[-1]-years[0]),
        "y_spacing_px_per_million_hectares": (plot["bottom"]-plot["top"])/(high-low)*1000000,
        "points": [{"year": c, "value": int(v), "x": x, "y": y} for c,v,(x,y) in zip(cats,values,points)],
        "forest": {"path": "assets/forest_watercolor.png", "position": list(forest_xy), "size": [forest_w, forest_h], "data_encoding": False},
        "major_year_labels": major_years,
        "numeric_callout_years": [cats[i[0]] for i in label_specs],
        "text_bounding_boxes": text_records,
        "data_sha256": hashlib.sha256((ROOT / "data.json").read_bytes()).hexdigest(),
    })
    print(f"Rendered {W} × {H}; {len(points)} vertices; range {low:,}–{high:,} {series['unit']}.")


if __name__ == "__main__":
    main()
