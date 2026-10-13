#!/usr/bin/env python3
"""Offline infographic renderer. All quantitative geometry comes from data.json.

Run complete candidates only through the supplied capture_render.py wrapper.
The generated infant is a non-quantitative surface; the chart uses one unwarped
linear mapping and is printed directly onto the generated diaper texture.
"""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageChops

ROOT = Path(__file__).resolve().parent
S = 2


def load_json(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def save_json(name, obj):
    p = ROOT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    inputs = load_json("input.json")
    brief = load_json(inputs["brief"])
    data = load_json(inputs["data"])
    core = load_json(inputs["critical_metaphor_contract"])
    assert core["topic_zh"] == data["topic_zh"]
    assert (ROOT / inputs["reference"]).is_file()
    W, H = brief["width"], brief["height"]
    assert data["value_axis"]["scale"] == "linear"
    assert len(data["series"]) == 1
    series = data["series"][0]
    years = np.array(data["time_points"], dtype=float)
    values = np.array(series["values"], dtype=float)
    assert len(years) == len(values) == len(data["categories"])
    assert np.all(np.diff(years) > 0)
    vmin, vmax = data["value_axis"]["minimum"], data["value_axis"]["maximum"]
    assert np.all((values >= vmin) & (values <= vmax))
    first, last = data["categories"][0], data["categories"][-1]
    peak, trough = int(np.argmax(values)), int(np.argmin(values))

    paper = (246, 243, 235)
    ink = (37, 43, 44)
    muted = (90, 99, 98)
    blue_ink = (109, 143, 159)
    red = (173, 53, 43)
    canvas = Image.new("RGBA", (W*S, H*S), paper + (255,))
    draw = ImageDraw.Draw(canvas)
    fonts = {}

    def font(size, style="sans"):
        key = (size, style)
        if key not in fonts:
            file = {"sans": "Sans-Regular.ttf", "bold": "Sans-Bold.ttf",
                    "serif": "Serif-Regular.ttf", "italic": "Serif-Italic.ttf"}[style]
            fonts[key] = ImageFont.truetype(str(ROOT / "assets/fonts" / file), round(size*S))
        return fonts[key]

    text_records = []

    def text(x, y, string, size=28, style="sans", fill=ink, anchor="lt", halo=0):
        f = font(size, style)
        draw.text((round(x*S), round(y*S)), str(string), font=f, fill=fill,
                  anchor=anchor, stroke_width=round(halo*S),
                  stroke_fill=(249, 248, 243) if halo else fill)
        bounds = draw.textbbox((round(x*S), round(y*S)), str(string), font=f, anchor=anchor,
                              stroke_width=round(halo*S))
        text_records.append({"text": str(string), "bbox": [round(b/S, 2) for b in bounds]})

    def line(points, fill=ink, width=1):
        draw.line([(round(x*S), round(y*S)) for x, y in points], fill=fill,
                  width=max(1, round(width*S)), joint="curve")

    def rect(box, fill, outline=None, width=1):
        draw.rectangle(tuple(round(a*S) for a in box), fill=fill, outline=outline, width=round(width*S))

    def paragraph(x, y, message, width, size=29, leading=42, style="serif", fill=ink):
        words = message.split()
        current = ""
        lines = []
        for word in words:
            attempt = f"{current} {word}".strip()
            if font(size, style).getlength(attempt) / S <= width:
                current = attempt
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        for i, ln in enumerate(lines):
            text(x, y + i*leading, ln, size, style, fill)
        return y + len(lines)*leading

    def tracked(x, y, string, size=21, spacing=2.3, fill=muted):
        for char in string:
            text(x, y, char, size=size, style="bold", fill=fill)
            x += font(size, "bold").getlength(char)/S + spacing

    # Editorial page, retaining the source's text-column / blue-plate structure.
    tracked(96, 66, "PICTURING POPULATION", 20, 2.7, ink)
    text(1984, 66, f"{len(values)} YEARS · ONE CONTINUOUS SERIES", 20, fill=muted, anchor="rt")
    line([(96, 117), (1984, 117)], (176, 180, 173), 1.4)

    rect((748, 244, 1980, 1372), (216, 231, 237), (123, 148, 157), 1.6)
    # The generated asset is used in full. No reference pixels enter the renderer.
    raw = Image.open(ROOT / "assets/raw_01.png").convert("RGBA")
    asset_box = (710, 132, 1379, 1342)
    ax, ay, aw, ah = asset_box
    infant = raw.resize((aw*S, ah*S), Image.Resampling.LANCZOS)
    canvas.alpha_composite(infant, (ax*S, ay*S))
    draw = ImageDraw.Draw(canvas)

    # A single affine chart mapping; neither the curve nor the grid is warped.
    plot = {"left": 952.0, "right": 1858.0, "top": 799.0, "bottom": 1140.0}
    pw, ph = plot["right"]-plot["left"], plot["bottom"]-plot["top"]

    def xmap(year):
        return plot["left"] + (float(year)-years[0])/(years[-1]-years[0])*pw

    def ymap(value):
        return plot["bottom"] - (float(value)-vmin)/(vmax-vmin)*ph

    points = [(xmap(t), ymap(v)) for t, v in zip(years, values)]
    mask_original = Image.open(ROOT / "assets/diaper_mask.png").convert("L")
    mask = Image.new("L", canvas.size)
    mask.paste(mask_original.resize((aw*S, ah*S), Image.Resampling.LANCZOS), (ax*S, ay*S))
    # Grid ink crosses the cloth, but stops at the diaper's actual generated edge.
    grid = Image.new("RGBA", canvas.size)
    gd = ImageDraw.Draw(grid)
    for year in range(int(years[0]), int(years[-1])+1, 5):
        gx = xmap(year)*S
        major = year % 10 == 0
        gd.line([(gx, 725*S), (gx, 1240*S)],
                fill=blue_ink + (106 if major else 69,), width=round((1.45 if major else 1.0)*S))
    for value in range(int(vmin), int(vmax)+1, 20):
        gy = ymap(value)*S
        gd.line([(807*S, gy), (1983*S, gy)], fill=blue_ink + (105,), width=round(1.4*S))
    grid.putalpha(ImageChops.multiply(grid.getchannel("A"), mask))
    canvas.alpha_composite(grid)
    draw = ImageDraw.Draw(canvas)

    line([(plot["left"], plot["top"]), (plot["left"], plot["bottom"]),
          (plot["right"], plot["bottom"])], (105, 135, 148), 1.8)
    for value in range(int(vmin), int(vmax)+1, 20):
        gy = ymap(value)
        line([(plot["left"]-8, gy), (plot["left"], gy)], (105, 135, 148), 1.8)
        text(plot["left"]-20, gy, str(value), 25, fill=(63, 86, 95), anchor="rm", halo=1.8)
    for year in range(int(years[0]), int(years[-1])+1):
        gx = xmap(year)
        line([(gx, plot["bottom"]), (gx, plot["bottom"]+(9 if year % 10 == 0 else 4))],
             (105, 135, 148), 1.3)
        if year % 10 == 0:
            text(gx, plot["bottom"]+18, str(year), 24, fill=(54, 74, 83), anchor="mt", halo=1.6)
    text((plot["left"]+plot["right"])/2, 1214, "Year", 22, fill=(68, 89, 96), anchor="mt", halo=1)

    # Unsmoothed polyline uses every supplied annual observation, in input order.
    line(points, red, 7)
    for idx in (0, peak, trough, len(points)-1):
        x, y = points[idx]
        draw.ellipse(((x-5)*S, (y-5)*S, (x+5)*S, (y+5)*S), fill=red)

    # Callouts contain only values drawn from the current series.
    px, py = points[peak]
    line([(px, py), (px+42, py-33), (px+61, py-33)], red, 1.6)
    text(px+70, py-87, "SERIES HIGH", 18, "bold", red, halo=2)
    text(px+70, py-57, f"{data['categories'][peak]} · {values[peak]:.1f}", 29, "bold", red, halo=2)
    lx, ly = points[trough]
    line([(lx, ly), (lx-42, ly+65), (lx-97, ly+65)], red, 1.6)
    text(lx-315, ly+19, "SERIES LOW", 18, "bold", red, halo=2)
    text(lx-315, ly+49, f"{data['categories'][trough]} · {values[trough]:.1f}", 29, "bold", red, halo=2)

    # Left-hand book-page explanation and title.
    rect((96, 194, 410, 243), (238, 220, 211))
    text(113, 203, data["visible_notice"], 23, "bold", red)
    display_title = data["title"].removeprefix("SYNTHETIC ")
    first_title, second_title = display_title.split(" ", 1)
    text(91, 281, first_title, 90, "serif", ink)
    text(91, 390, second_title, 90, "serif", ink)
    unit_parts = series["unit"].rsplit(" per year", 1)
    text(98, 527, unit_parts[0], 29, "sans", ink)
    text(98, 568, "per year", 29, "sans", ink)
    text(98, 625, f"{first}–{last}", 31, "sans", muted)
    line([(96, 696), (609, 696)], (176, 180, 173), 1.3)
    line([(99, 739), (145, 739)], red, 7)
    text(166, 721, series["label"], 27, "bold", ink)

    text(95, 806, "Each year leaves a mark.", 33, "italic", ink)
    y = paragraph(98, 866,
        "An oversized diaper carries the series, with every annual value joined in chronological order. "
        "Equal vertical steps mean equal changes in the rate.", 507, 28, 42)
    paragraph(98, y+25,
        f"The series begins at {values[0]:.1f} in {first} and closes at {values[-1]:.1f} in {last}.",
        505, 28, 42)

    tracked(98, 1228, "READING THE SCALE", 18, 2, muted)
    text(98, 1271, f"{len(values)} annual observations", 25, fill=ink)
    text(98, 1310, f"{vmin}–{vmax} on a linear scale", 25, fill=ink)
    text(98, 1349, "Decade labels; all years plotted", 25, fill=ink)
    text(98, 1410, "Constructed values, not observed statistics.", 20, "italic", muted)

    # Final page furniture.
    line([(96, 1506), (1984, 1506)], (176, 180, 173), 1.4)
    text(96, 1531, "Source: supplied synthetic data  /  Annual birth rate", 21, fill=muted)
    text(1984, 1531, f"{first}–{last}     ·     {len(values)} OBSERVATIONS", 20, fill=muted, anchor="rt")

    # Save measurement metadata, not a second image or an uncaptured preview.
    source_mask = np.array(mask_original)
    source_coords = [((x-ax)*raw.width/aw, (y-ay)*raw.height/ah) for x, y in points]
    mask_membership = [bool(source_mask[round(y), round(x)] > 127) for x, y in source_coords]
    geometry = {
        "canvas": [W, H], "series_id": series["id"], "unit": series["unit"],
        "plot": plot, "axis": data["value_axis"], "asset_box_xywh": list(asset_box),
        "mapping": {"x_pixels_per_year": pw/(years[-1]-years[0]),
                    "y_pixels_per_rate_unit": ph/(vmax-vmin), "smoothing": False},
        "point_count": len(points), "segment_count": len(points)-1,
        "points": [{"year": str(cat), "value": float(val), "x": x, "y": y,
                    "inside_generated_diaper_mask": member}
                   for cat, val, (x,y), member in zip(data["categories"], values, points, mask_membership)],
        "peak": {"year": data["categories"][peak], "value": float(values[peak])},
        "minimum": {"year": data["categories"][trough], "value": float(values[trough])},
        "text_bounds": text_records,
    }
    save_json("artifacts/geometry.json", geometry)
    canvas.convert("RGB").resize((W,H), Image.Resampling.LANCZOS).save(ROOT / "final.png")
    print(f"Rendered {W}x{H}; {len(points)} points; {len(points)-1} segments; "
          f"{sum(mask_membership)} point centers on generated diaper.")


if __name__ == "__main__":
    main()
