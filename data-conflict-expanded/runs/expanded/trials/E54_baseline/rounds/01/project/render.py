#!/usr/bin/env python3
"""Offline data-bound balloon infographic. Run only through capture_render.py.

The generated fabric envelope is a single shared template. A uniform affine
scale sqrt(CEO pay / area reference) controls both axes of its silhouette.
The center of every identical basket follows one linear worker-pay transform.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
S = 2


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


INPUT = read_json("input.json")
BRIEF = read_json(INPUT["brief"])
DATA = read_json(INPUT["data"])
CORE = read_json(INPUT["critical_metaphor_contract"])
W, H = BRIEF["width"], BRIEF["height"]
assert (W, H) == (1980, 2000)
assert DATA["synthetic"] and len(DATA["categories"]) == 5
series = {item["id"]: item for item in DATA["series"]}
CEO = series["ceo_pay"]
WORKER = series["worker_median_pay"]
REF = DATA["ceo_area_reference"]
VMIN, VMAX = DATA["worker_axis"]["minimum"], DATA["worker_axis"]["maximum"]
assert CEO["unit"] == WORKER["unit"] == "USD/year"
assert REF == DATA["value_axis"]["maximum"]
assert VMIN == 0

INK = "#243F48"
MUTED = "#617780"
GRID = "#D3E0DF"
COLORS = ["#D9523B", "#E4B633", "#398BB3", "#79A44F", "#DD852F"]
CATEGORY_X = [385, 738, 1091, 1444, 1797]
ZERO_Y = 1740.0
AXIS_TOP = 1020.0
AXIS_X = 181.0
REF_WIDTH = 336.0
BASKET_WIDTH = 45.0
ROPE_GAP = 62.0

FONTS = {
    "regular": ROOT / "assets/fonts/NimbusSans-Regular.otf",
    "bold": ROOT / "assets/fonts/NimbusSans-Bold.otf",
}
font_cache = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in font_cache:
        font_cache[key] = ImageFont.truetype(str(FONTS["bold" if bold else "regular"]), round(size * S))
    return font_cache[key]


def rgb(hex_color):
    return tuple(int(hex_color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))


# A clean sky shared by all objects, becoming slightly greener at the ground.
top = np.array(rgb("#FAFAF6"), float)
bottom = np.array(rgb("#EDF4F0"), float)
yy = np.linspace(0, 1, H * S)[:, None, None]
gradient = np.repeat((top + (bottom - top) * yy).astype(np.uint8), W * S, axis=1)
im = Image.fromarray(gradient, "RGB").convert("RGBA")
draw = ImageDraw.Draw(im)
text_records = []


def line(points, fill, width=1):
    draw.line([(round(x * S), round(y * S)) for x, y in points], fill=fill, width=max(1, round(width * S)))


def ellipse(box, fill, outline=None, width=1):
    draw.ellipse(tuple(round(v * S) for v in box), fill=fill, outline=outline, width=round(width * S))


def rect(box, fill, radius=0, outline=None, width=1):
    p = tuple(round(v * S) for v in box)
    if radius:
        draw.rounded_rectangle(p, round(radius * S), fill=fill, outline=outline, width=round(width * S))
    else:
        draw.rectangle(p, fill=fill, outline=outline, width=round(width * S))


def text(x, y, value, size=28, fill=INK, bold=False, anchor="lt", track=0, record=True):
    f = font(size, bold)
    if track:
        total = sum(draw.textlength(c, font=f) for c in value) + max(0, len(value) - 1) * track * S
        px = x * S - (total / 2 if anchor.startswith("m") else total if anchor.startswith("r") else 0)
        for c in value:
            draw.text((px, y * S), c, font=f, fill=fill, anchor="lt")
            px += draw.textlength(c, font=f) + track * S
        box = (x - total / S / 2, y, x + total / S / 2, y + size) if anchor.startswith("m") else (x, y, x + total / S, y + size)
    else:
        draw.text((x * S, y * S), value, font=f, fill=fill, anchor=anchor)
        box = tuple(v / S for v in draw.textbbox((x * S, y * S), value, font=f, anchor=anchor))
    if record:
        text_records.append({"text": value, "bbox_px": [round(v, 2) for v in box], "font_px": size})


def dashed_horizontal(y, start, end, color=GRID, width=1, dash=4, gap=9):
    x = start
    while x < end:
        line([(x, y), (min(x + dash, end), y)], color, width)
        x += dash + gap


def cloud(cx, cy, scale=1):
    # Restrained atmospheric clouds; no quantitative meaning.
    color = "#FDFDF9"
    for x, y, w, h in [(-68, -5, 100, 43), (-34, -30, 75, 73), (14, -44, 83, 89), (62, -8, 86, 48)]:
        ellipse((cx + x * scale, cy + y * scale, cx + (x + w) * scale, cy + (y + h) * scale), color)


def tree(x, ground_y, scale=1):
    line([(x, ground_y), (x, ground_y - 40 * scale)], "#69834D", 3 * scale)
    for dx, dy, r, color in [(-11, -40, 17, "#AFBF75"), (10, -47, 21, "#BDCA8A"), (0, -61, 18, "#C8D495")]:
        ellipse((x + (dx - r) * scale, ground_y + (dy - r) * scale,
                 x + (dx + r) * scale, ground_y + (dy + r) * scale), color)
    line([(x, ground_y), (x, ground_y - 60 * scale)], "#69834D", 2 * scale)
    line([(x, ground_y - 25 * scale), (x - 10 * scale, ground_y - 37 * scale)], "#69834D", 1.5 * scale)
    line([(x, ground_y - 39 * scale), (x + 12 * scale, ground_y - 53 * scale)], "#69834D", 1.5 * scale)


# Header and the two clearly distinguished metric roles.
rect((80, 61, 351, 107), INK, radius=8)
text(216, 74, DATA["visible_notice"], 22, "#FFFFFF", bold=True, anchor="mt", track=1.35)
text(1900, 77, f"PAY COMPARISON  /  {len(DATA['categories']):02d} COMPANIES", 21, MUTED, anchor="rt")
text(77, 151, "Executive and worker pay", 108, bold=True)
text(82, 282, "Five synthetic companies, compared in US dollars per year.", 33, MUTED)
line([(80, 349), (1900, 349)], "#C9D6D3", 1.4)

ellipse((84, 398, 113, 427), None, INK, 1.8)
ellipse((100, 388, 149, 437), None, INK, 1.8)
text(175, 384, CEO["label"], 31, bold=True)
text(175, 427, "Balloon envelope area  ·  USD/year", 26, MUTED)
line([(1051, 384), (1051, 444)], "#D1DDD9", 1)
line([(1101, 390), (1101, 439)], INK, 1.8)
line([(1094, 398), (1101, 390), (1108, 398)], INK, 1.8)
line([(1094, 431), (1101, 439), (1108, 431)], INK, 1.8)
text(1130, 384, WORKER["label"], 31, bold=True)
text(1130, 427, "Basket-center height  ·  USD/year", 26, MUTED)

# Clouds sit behind the data and remain deliberately subdued.
cloud(1690, 668, 1.2)
cloud(680, 1418, 1.05)
cloud(1130, 1565, 0.7)
cloud(122, 760, 0.48)

# The only quantitative positional axis is the worker salary axis.
text(66, 923, "Worker pay", 25, bold=True)
text(66, 958, "USD/year", 22, MUTED)
for val in np.linspace(VMIN, VMAX, 5):
    y = ZERO_Y - (val - VMIN) / (VMAX - VMIN) * (ZERO_Y - AXIS_TOP)
    if val > 0:
        dashed_horizontal(y, AXIS_X, 1928)
    line([(AXIS_X - 10, y), (AXIS_X + 2, y)], "#8BA09E", 1.6)
    text(150, y, f"${val:,.0f}", 25, MUTED, anchor="rm")
line([(AXIS_X, AXIS_TOP), (AXIS_X, ZERO_Y)], "#A7B9B3", 1.3)

# Level zero-ground is common to all five balloons, with a non-data green lawn.
rect((75, ZERO_Y, 1928, 1815), "#91AC70", radius=0)
ellipse((75, 1787, 1928, 1840), "#91AC70")
rect((75, 1787, 1928, 1816), "#91AC70")
line([(AXIS_X, ZERO_Y), (1928, ZERO_Y)], "#6C8858", 2.2)
tree(251, ZERO_Y - 1, 0.93)
tree(944, ZERO_Y - 1, 0.73)
tree(1879, ZERO_Y - 1, 0.97)
for x in CATEGORY_X:
    ellipse((x - 35, ZERO_Y + 11, x + 35, ZERO_Y + 22), "#B1C68D")
text(1050, 1781, "COMMON ZERO GROUND  ·  $0 WORKER PAY", 20, "#334C38", anchor="mt", track=1.2)

envelope = Image.open(ROOT / "assets/processed/envelope.png").convert("RGBA")
basket = Image.open(ROOT / "assets/processed/basket.png").convert("RGBA")
env_array = np.asarray(envelope).astype(float)
native_area = float(env_array[:, :, 3].sum() / 255)


def tint_envelope(color):
    # Luminance mapping preserves the generated seams, wrinkles and broad shading.
    lum = env_array[:, :, 0] / 255
    base = np.array(rgb(color), float)
    mid = 0.70
    dark = np.clip(lum / mid, 0, 1) ** 1.0
    shade = base[None, None, :] * dark[:, :, None]
    highlight = np.clip((lum - mid) / (1 - mid), 0, 1) * 0.43
    shade = shade + (255 - shade) * highlight[:, :, None]
    rgba = np.dstack((np.clip(shade, 0, 255).astype(np.uint8), env_array[:, :, 3].astype(np.uint8)))
    return Image.fromarray(rgba, "RGBA")


def affine_asset(asset, x, y, scale):
    """Place with a single uniform scale, preserving subpixel geometry."""
    left, top = math.floor(x * S) - 2, math.floor(y * S) - 2
    right = math.ceil((x + asset.width * scale) * S) + 2
    bottom = math.ceil((y + asset.height * scale) * S) + 2
    tile = asset.transform((right - left, bottom - top), Image.Transform.AFFINE,
        (1 / (scale * S), 0, (left / S - x) / scale,
         0, 1 / (scale * S), (top / S - y) / scale), Image.Resampling.BICUBIC)
    im.alpha_composite(tile, (left, top))
    return tile, (left, top)


geometry = []
mask_dir = ROOT / "artifacts/masks"
mask_dir.mkdir(parents=True, exist_ok=True)

for i, (cat, x, ceo, worker, color) in enumerate(zip(DATA["categories"], CATEGORY_X, CEO["values"], WORKER["values"], COLORS)):
    by = ZERO_Y - (worker - VMIN) / (VMAX - VMIN) * (ZERO_Y - AXIS_TOP)
    bscale = BASKET_WIDTH / basket.width
    bw, bh = basket.width * bscale, basket.height * bscale
    # The basket's visible bounding-box center is the precise data anchor.
    ba = np.asarray(basket)[:, :, 3]
    b_ys, b_xs = np.where(ba >= 128)
    native_bcx = (b_xs.min() + b_xs.max() + 1) / 2
    native_bcy = (b_ys.min() + b_ys.max() + 1) / 2
    bx, btop = x - native_bcx * bscale, by - native_bcy * bscale
    linear_scale = math.sqrt(ceo / REF)
    escale = REF_WIDTH / envelope.width * linear_scale
    ew, eh = envelope.width * escale, envelope.height * escale
    ebottom = btop - ROPE_GAP
    ex, ey = x - ew / 2, ebottom - eh

    # Thin guide stops at the basket center; labels knock out only a short segment.
    guide = tuple(round(v * 0.47 + 247 * 0.53) for v in rgb(color))
    line([(x, by), (x, ZERO_Y)], guide, 1.6)

    # Four suspension ropes are non-data material, attached to the envelope and basket.
    for sign in (-1, 1):
        line([(x + sign * ew * 0.22, ey + eh * 0.77), (x + sign * bw * 0.40, btop + 5)], "#736B52", 1.55)
        line([(x + sign * ew * 0.08, ebottom - 10), (x + sign * bw * 0.31, btop + 3)], "#8E876B", 1.2)
    colored = tint_envelope(color)
    tile, offset = affine_asset(colored, ex, ey, escale)
    btile, boffset = affine_asset(basket, bx, btop, bscale)

    # Small external level ticks mark the same center used by the salary transform.
    line([(x - 35, by), (x - 27, by)], color, 2)
    line([(x + 27, by), (x + 35, by)], color, 2)

    text(x, ey - 49, cat, 31, INK, bold=True, anchor="mt")
    fill = "#FFFFFF" if i in (0, 2) else "#293B2B"
    ty = ey + eh * 0.373
    text(x, ty, "CEO / YEAR", 19, fill, bold=True, anchor="mt", track=1.3)
    text(x, ty + 33, f"${ceo:,}", 36, fill, bold=True, anchor="mt")
    # A pale label backing interrupts the guide, without moving the basket anchor.
    rect((x - 102, by + 32, x + 102, by + 106), "#F2F6F1", radius=11)
    text(x, by + 35, f"${worker:,}", 36, INK, bold=True, anchor="mt")
    text(x, by + 78, "MEDIAN WORKER", 17, MUTED, anchor="mt", track=0.9)

    # Save cropped masks only, for a raster-based geometry audit after rendering.
    alpha = tile.getchannel("A")
    alpha.save(mask_dir / f"envelope_{i+1:02d}_2x.png")
    btile.getchannel("A").save(mask_dir / f"basket_{i+1:02d}_2x.png")
    arr = np.asarray(alpha)
    alpha_area = float(arr.sum() / 255 / S ** 2)
    yy2, xx2 = np.where(arr >= 128)
    barr = np.asarray(btile.getchannel("A"))
    bys, bxs = np.where(barr >= 128)
    actual_basket_center = [(boffset[0] + (bxs.min() + bxs.max() + 1) / 2) / S,
                            (boffset[1] + (bys.min() + bys.max() + 1) / 2) / S]
    geometry.append({
        "category": cat, "ceo_pay_usd_year": ceo, "median_worker_pay_usd_year": worker,
        "color": color, "uniform_scale_relative_to_reference": linear_scale,
        "envelope_bbox_target_px": [ex, ey, ex + ew, ey + eh],
        "envelope_mask_offset_2x": list(offset),
        "envelope_visible_bbox_px": [(offset[0] + xx2.min()) / S, (offset[1] + yy2.min()) / S,
                                     (offset[0] + xx2.max() + 1) / S, (offset[1] + yy2.max() + 1) / S],
        "envelope_alpha_area_px2": alpha_area,
        "envelope_threshold_area_px2": int((arr >= 128).sum()) / S ** 2,
        "basket_mask_offset_2x": list(boffset),
        "basket_target_center_px": [x, by],
        "basket_visible_center_px": actual_basket_center,
        "basket_visible_size_px": [(bxs.max() - bxs.min() + 1) / S, (bys.max() - bys.min() + 1) / S],
        "worker_pay_recovered_from_visible_center": (ZERO_Y - actual_basket_center[1]) / (ZERO_Y - AXIS_TOP) * (VMAX - VMIN) + VMIN,
    })

# Method and provenance, stated without borrowing any values from the reference.
line([(80, 1874), (1900, 1874)], "#CAD7CF", 1.3)
text(80, 1903, f"AREA ∝ CEO PAY", 23, INK, bold=True)
text(333, 1903, f"Reference: ${REF:,}  ·  HEIGHT ∝ WORKER PAY  ·  Scale: ${VMIN:,}–${VMAX:,}  ·  Fixed basket size", 23, MUTED)
text(80, 1952, "Source: supplied synthetic dataset. Company labels and pay values are illustrative.", 24, MUTED)

# One final image; the frozen wrapper retains this and the corresponding sources.
final = im.convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
final.save(ROOT / "final.png", optimize=True)
audit = {
    "canvas_px": [W, H], "supersampling": S,
    "worker_transform": {"zero_y_px": ZERO_Y, "maximum_y_px": AXIS_TOP, "minimum": VMIN, "maximum": VMAX},
    "envelope_reference": {"value_usd_year": REF, "template_px": list(envelope.size), "reference_width_px": REF_WIDTH,
                           "alpha_area_at_reference_px2": native_area * (REF_WIDTH / envelope.width) ** 2},
    "objects": geometry, "text": text_records,
}
(ROOT / "artifacts/geometry.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"image": "final.png", "size": [W, H], "balloons": len(geometry), "generated_template": "assets/processed/envelope.png"}))
