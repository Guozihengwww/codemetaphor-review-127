#!/usr/bin/env python3
"""Offline pillow infographic. Run only through capture_render.py in this study."""
from pathlib import Path
import json
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def save_json(name, value):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    inputs = read_json("input.json")
    brief = read_json(inputs["brief"])
    data = read_json(inputs["data"])
    core = read_json(inputs["critical_metaphor_contract"])
    assert (ROOT / inputs["reference"]).is_file()
    assert core["topic_zh"] == data["topic_zh"]
    width, height = brief["width"], brief["height"]
    assert (width, height) == (2000, 1400)
    series = data["series"][0]
    values = np.array(series["values"], dtype=float)
    categories = data["categories"]
    assert len(data["series"]) == 1 and len(values) == len(categories) == 4
    assert series["unit"] == "%" and np.isclose(values.sum(), 100)
    assert data["value_axis"] == {"minimum": 0, "maximum": 100, "scale": "linear"}

    # Light-gray studio background. All geometry below is in final-image pixels.
    yy, xx = np.mgrid[0:height, 0:width]
    radial = np.minimum(1, ((xx - 1000) / 1550) ** 2 + ((yy - 630) / 1550) ** 2)
    level = 242.5 - 6.0 * radial - 3.0 * np.maximum(0, yy - 780) / 620
    background = np.stack([level, level + .3, level - 1.0], axis=-1)
    canvas = Image.fromarray(np.uint8(np.clip(background, 0, 255))).convert("RGBA")

    px, py, pw, ph = 140, 305, 1720, 840
    pillow = Image.open(ROOT / "assets/pillow_clean.png").convert("RGBA")
    pillow = pillow.resize((pw, ph), Image.Resampling.LANCZOS)
    source = np.asarray(pillow).astype(np.float64)
    alpha = source[:, :, 3]
    mid_y = ph // 2
    opaque = np.flatnonzero(alpha[mid_y] >= 127.5)
    left, right = int(opaque[0]), int(opaque[-1]) + 1
    full_width = right - left
    data_start = left + .30 * full_width
    data_width = .70 * full_width
    boundaries = data_start + np.r_[0, np.cumsum(values)] / 100 * data_width

    # Dye the generated fabric, keeping its wrinkles, weave, piping, and volume.
    palette_hex = ["#493E37", "#078F86", "#0878B7", "#E6A62C"]
    palette = np.array([tuple(bytes.fromhex(s[1:])) for s in palette_hex], dtype=float)
    luminance = (source[:, :, :3] * np.array([.2126, .7152, .0722])).sum(axis=2)
    shade = np.clip(luminance / 236.0, 0, 1.085) ** 1.60
    shine = np.maximum(luminance - 238, 0)[:, :, None] * .14
    output_rgb = source[:, :, :3].copy()
    columns = np.arange(pw, dtype=float)
    for i, color in enumerate(palette):
        lower = boundaries[i]
        # The final band continues to the outer corners. Only the midpoint
        # cross section is quantitative, as specified in the frozen contract.
        upper = boundaries[i + 1] if i < len(values) - 1 else pw
        coverage = np.clip(np.minimum(columns + 1, upper) - np.maximum(columns, lower), 0, 1)
        dyed = np.clip(shade[:, :, None] * color + shine, 0, 255)
        output_rgb += (dyed - source[:, :, :3]) * coverage[None, :, None]
    dyed_pillow = Image.fromarray(np.uint8(np.clip(np.dstack([output_rgb, alpha]), 0, 255)))

    # Ground shadows remain decorative, below the pillow, and never carry data.
    shadow = Image.new("RGBA", (width, height))
    sd = ImageDraw.Draw(shadow)
    sd.ellipse((145, 1098, 1855, 1181), fill=(35, 42, 46, 33))
    shadow = shadow.filter(ImageFilter.GaussianBlur(23))
    canvas.alpha_composite(shadow)
    contact = Image.new("RGBA", (width, height))
    cd = ImageDraw.Draw(contact)
    cd.ellipse((193, 1124, 1815, 1154), fill=(34, 39, 40, 63))
    contact = contact.filter(ImageFilter.GaussianBlur(9))
    canvas.alpha_composite(contact)
    canvas.alpha_composite(dyed_pillow, (px, py))

    # Supersampled vector-like typography and leaders. Required strings are data-bound.
    scale = 2
    ink = (29, 43, 52, 255)
    muted = (96, 107, 114, 255)
    light = (131, 140, 146, 255)
    text_layer = Image.new("RGBA", (width * scale, height * scale))
    draw = ImageDraw.Draw(text_layer)
    text_records = []
    font_cache = {}

    def font(size, weight="regular"):
        key = (size, weight)
        if key not in font_cache:
            name = {"regular": "NotoSans-Regular.ttf", "bold": "NotoSans-Bold.ttf", "cjk": "NotoSansCJK-Regular.ttc"}[weight]
            font_cache[key] = ImageFont.truetype(str(ROOT / "assets/fonts" / name), int(size * scale))
        return font_cache[key]

    def text(x, y, label, size, color=ink, weight="regular", anchor="lt", role="annotation"):
        f = font(size, weight)
        draw.text((x * scale, y * scale), label, font=f, fill=color, anchor=anchor)
        box = draw.textbbox((x * scale, y * scale), label, font=f, anchor=anchor)
        text_records.append({"text": label, "role": role, "bbox": [v / scale for v in box]})

    def line(points, color, thickness=1.5):
        draw.line([(int(x * scale), int(y * scale)) for x, y in points], fill=color, width=round(thickness * scale))

    def spaced_text(x, y, label, size, spacing, color):
        f = font(size, "bold")
        initial = x
        for character in label:
            draw.text((x * scale, y * scale), character, font=f, fill=color, anchor="lt")
            x += draw.textlength(character, font=f) / scale + spacing
        text_records.append({"text": label, "role": "notice", "bbox": [initial, y, x, y + size]})

    badge = (112, 65, 402, 111)
    draw.rounded_rectangle(tuple(v * scale for v in badge), radius=5 * scale,
                           fill=(237, 226, 213, 255))
    spaced_text(130, 77, data["visible_notice"], 21, 1.6, (122, 73, 40, 255))
    title = re.sub(r"^SYNTHETIC\s+", "", data["title"])
    text(108, 138, title, 78, weight="bold", role="title")
    text(113, 239, data["topic_zh"] + "  ·  Share of total (%)", 28, muted, "cjk", role="metric")
    text(1884, 81, f"{len(categories)} DURATION BANDS", 20, muted, anchor="rt")

    centers = (boundaries[:-1] + boundaries[1:]) / 2
    for i, (category, value, cx) in enumerate(zip(categories, values, centers)):
        gx = px + float(cx)
        ci = int(round(cx))
        column_rows = np.flatnonzero(alpha[:, ci] >= 127.5)
        bottom = py + int(column_rows[-1])
        leader_color = tuple(int(v * .68) for v in palette[i]) + (230,)
        line([(gx, bottom + 4), (gx, 1180)], leader_color, 1.6)
        r = 3.2
        draw.ellipse(((gx-r)*scale, 1177*scale, (gx+r)*scale, 1183.4*scale), fill=leader_color)
        text(gx, 1199, category, 27, muted, anchor="mt", role=f"category_{i}")
        text(gx, 1237, f"{value:.1f}{series['unit']}", 47, ink, "bold", "mt", f"value_{i}")

    # Explicitly identify the white, non-data region without implying a fifth group.
    white_center = px + (left + data_start) / 2
    text(white_center, 1202, "White fabric", 25, muted, anchor="mt")
    text(white_center, 1244, "Decorative · not data", 23, light, anchor="mt")

    line([(112, 1320), (1888, 1320)], (196, 201, 202, 255), 1)
    text(112, 1344, "Width at the midpoint encodes share. White fabric (30%), area and folds do not encode values.", 21, muted)
    text(1888, 1343, f"TOTAL  {data['series_totals'][series['id']]:.1f}%", 21, ink, "bold", "rt")

    canvas.alpha_composite(text_layer.resize((width, height), Image.Resampling.LANCZOS))
    canvas.convert("RGB").save(ROOT / "final.png")

    geometry = {
        "canvas": [width, height],
        "pillow_bbox": [px, py, px + pw, py + ph],
        "measurement_y": py + mid_y,
        "alpha_threshold": 127.5,
        "midpoint_endpoints_x": [px + left, px + right],
        "visible_full_width_px": full_width,
        "white_fraction": .30,
        "data_region_width_px": data_width,
        "data_boundaries_x": (boundaries + px).tolist(),
        "expected_band_widths_px": (values / 100 * data_width).tolist(),
        "category_order": categories,
        "values_percent": values.tolist(),
        "palette": palette_hex,
        "labels": text_records,
        "mapping_zh": "在中部水平截面取可见布面端点；左侧30%白布排除，右侧70%按份额作连续竖直色带。",
    }
    save_json("artifacts/render_geometry.json", geometry)
    print("Saved final.png: 2000 × 1400; generated pillow with code-bound vertical bands.")


if __name__ == "__main__":
    main()
