"""Offline renderer for the frozen forest-share composition.

Run from this directory with the provided Python interpreter. During the study,
all complete renders are invoked through the frozen capture_render.py wrapper.
"""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent
S = 2
INK = "#203e36"
MUTED = "#536c63"
BG = "#e5eeea"
COLORS = [
    "#518c87", "#d7aa54", "#cf795c", "#8c7ea9", "#91a269",
    "#587c9e", "#d39b64", "#b36878", "#78ada1", "#d48c42",
    "#817eb1", "#559374", "#b97251", "#829fac", "#b990b0",
    "#c6a34e", "#bb9868",
]


def rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def font(kind, size):
    return ImageFont.truetype(str(ROOT / "assets/fonts" / f"{kind}.otf"), round(size * S))


def main():
    paths = json.loads((ROOT / "input.json").read_text(encoding="utf-8"))
    brief = json.loads((ROOT / paths["brief"]).read_text(encoding="utf-8"))
    data = json.loads((ROOT / paths["data"]).read_text(encoding="utf-8"))
    core = json.loads((ROOT / paths["critical_metaphor_contract"]).read_text(encoding="utf-8"))
    assert (ROOT / paths["reference"]).is_file()  # Retained reference; its pixels are not copied.
    assert core["visible_title"] == data["title"]
    assert data["synthetic"] and len(data["series"]) == 1
    series = data["series"][0]
    cats = data["categories"]
    values = np.array(series["values"], dtype=np.float64)
    assert len(cats) == len(values) == len(COLORS) == 17
    assert series["unit"] == "percent" and abs(values.sum() - 100) < 1e-9
    w, h = brief["width"], brief["height"]
    cx, cy, radius, yscale = 860., 1040., 525., .96
    start_angle = -90.
    ends = np.cumsum(values) * 3.6
    starts = np.r_[0., ends[:-1]]
    mids = start_angle + (starts + ends) / 2
    palette = np.array([rgb(x) for x in COLORS], dtype=np.float32)

    # Quiet mint paper, with deterministic subpixel texture.
    base = np.array(rgb(BG), dtype=np.float32)
    yy = np.linspace(0, 1, h * S, dtype=np.float32)
    gradient = np.clip(base[None, :] + (1.4 - 2.8 * yy[:, None]), 0, 255).astype(np.uint8)
    canvas = Image.fromarray(gradient[:, None, :], "RGB").resize((w * S, h * S)).convert("RGBA")
    draw = ImageDraw.Draw(canvas)
    text_boxes = []

    def text(x, y, content, kind="body", size=28, color=INK, align="left", key=None):
        f = font(kind, size)
        tw = draw.textlength(content, font=f)
        px = x * S - (tw if align == "right" else tw / 2 if align == "center" else 0)
        py = y * S
        draw.text((px, py), content, font=f, fill=color, anchor="lt")
        box = draw.textbbox((px, py), content, font=f, anchor="lt")
        text_boxes.append({"key": key or content, "text": content,
                           "bbox": [round(t / S, 3) for t in box]})

    def line(points, fill, width=1.5):
        draw.line([(round(x * S), round(y * S)) for x, y in points],
                  fill=fill, width=round(width * S), joint="curve")

    # Editorial hierarchy and explicit synthetic provenance.
    draw.rounded_rectangle((90*S, 78*S, 397*S, 132*S), radius=27*S, fill=INK)
    text(243.5, 94, data["visible_notice"], "bold", 24, "#f1f5eb", "center")
    text(1595, 96, "REGIONAL DISTRIBUTION", "bold", 23, MUTED, "right")
    text(86, 174, "Forest area shares", "title", 138)
    text(94, 328, "Share of total forest area, by synthetic region (%)", "body", 31, MUTED)
    line([(90, 397), (1595, 397)], "#b4c6bc", 1)

    # Sample the generated wood into a single, explicitly transformed object.
    # The quantitative domain is an exact circle before the uniform .96 y-scale.
    x0, y0, pw, ph = cx - 620, cy - 600, 1240, 1200
    geom = json.loads((ROOT / "assets/wood_geometry.json").read_text(encoding="utf-8"))
    wood = Image.open(ROOT / geom["processed"]).convert("RGBA")
    ar = geom["processed_data_radius"]
    acx, acy = geom["processed_center_xy"]
    ax = ar / (radius * S)
    ay = ar / (radius * yscale * S)
    patch = wood.transform((pw*S, ph*S), Image.Transform.AFFINE,
                           (ax, 0, acx - 620 * ar / radius,
                            0, ay, acy - 600 * ar / (radius * yscale)),
                           resample=Image.Resampling.BICUBIC)
    pixels = np.asarray(patch).copy()
    gy, gx = np.indices((ph*S, pw*S), dtype=np.float32)
    gx = (gx + .5) / S - 620
    gy = ((gy + .5) / S - 600) / yscale
    r = np.hypot(gx, gy)
    angle = np.mod(np.degrees(np.arctan2(gy, gx)) - start_angle, 360)
    ids = np.searchsorted(ends, angle, side="right").clip(0, len(cats)-1)
    domain = r <= radius
    lum = pixels[..., :3] @ np.array([.2126, .7152, .0722], dtype=np.float32)
    shade = np.power(np.clip(lum / 220., 0, 1.16), .9)
    tinted = np.clip(palette[ids] * shade[..., None], 0, 255).astype(np.uint8)
    pixels[..., :3][domain] = tinted[domain]
    pixels[..., 3][domain] = 255
    patch = Image.fromarray(pixels, "RGBA")

    # Non-data bark and a diffuse shadow establish the material object.
    shadow_a = patch.getchannel("A").filter(ImageFilter.GaussianBlur(21*S))
    shadow_a = shadow_a.point(lambda a: int(a * .19))
    shadow = Image.new("RGBA", patch.size, "#273c33")
    shadow.putalpha(shadow_a)
    canvas.alpha_composite(shadow, (round(x0*S + 7*S), round(y0*S + 20*S)))
    canvas.alpha_composite(patch, (round(x0*S), round(y0*S)))
    draw = ImageDraw.Draw(canvas)

    def point(theta, rr):
        t = math.radians(float(theta))
        return (cx + rr * math.cos(t), cy + rr * yscale * math.sin(t))

    label_records = []
    leaders = []

    def leader(idx, rest):
        p0 = point(mids[idx], radius - 1.5)
        p1 = point(mids[idx], radius + 53)
        pts = [p0, p1] + rest
        col = tuple(round(c*.64) for c in rgb(COLORS[idx]))
        line(pts, col, 1.5)
        # Endpoint sits on the edge of its own data wedge, not on another one.
        x, y = p0
        draw.ellipse(((x-3.5)*S, (y-3.5)*S, (x+3.5)*S, (y+3.5)*S),
                     fill=COLORS[idx], outline="#f0f3e9", width=round(1.2*S))
        leaders.append({"category": cats[idx], "points": [[round(x,3),round(y,3)] for x,y in pts]})

    # Large sectors use spacious two-line labels; all remain outside the wood.
    large = {
        0: (1305, 502, [(1278, 556)]),
        1: (1452, 1049, [(1436, point(mids[1], radius+53)[1])]),
        2: (1452, 1437, [(1385, 1482), (1431, 1482)]),
        3: (1140, 1650, [(1101, 1692), (1120, 1692)]),
        16: (91, 552, [(370, 598), (308, 598)]),
    }
    for i, (lx, ly, pts) in large.items():
        leader(i, pts)
        text(lx, ly, cats[i], "body", 32, INK, key=f"category_{i}")
        text(lx-1, ly+42, f"{values[i]:.1f}%", "bold", 46, INK, key=f"value_{i}")
        label_records.append({"index":i,"category":cats[i],"value_text":f"{values[i]:.1f}%"})

    # Dense small shares fan out in spatial order. Their angles are not enlarged.
    small = sorted(range(4, 16), key=lambda i: point(mids[i], radius)[1])
    for k, i in enumerate(small):
        ly = 1012 + k * 56
        leader(i, [(278, ly+14)])
        text(60, ly+2, cats[i], "body", 28, INK, key=f"category_{i}")
        text(258, ly, f"{values[i]:.1f}%", "bold", 32, INK, "right", key=f"value_{i}")
        label_records.append({"index":i,"category":cats[i],"value_text":f"{values[i]:.1f}%"})

    # Measurement key and provenance are part of the finished graphic.
    line([(90, 1784), (1595, 1784)], "#b4c6bc", 1)
    text(92, 1820, "ONE WOOD FACE = 100%", "bold", 25)
    text(92, 1870, "Sector area encodes forest-area share; bark is excluded. Listed order, clockwise from 12 o’clock.",
         "body", 25, MUTED)
    text(92, 1920, "Source: supplied synthetic data. Regional labels are illustrative, not real-world estimates.",
         "body", 23, MUTED)
    text(1595, 1820, series["label"] + " (%)", "body", 24, MUTED, "right")

    counts = np.bincount(ids[domain], minlength=len(cats))
    observed = counts / counts.sum() * 100
    geometry = {
        "canvas": [w,h], "supersampling":S,
        "data_center_xy":[cx,cy], "unprojected_radius":radius,
        "vertical_scale":yscale, "start_angle_degrees":start_angle,
        "direction":"clockwise", "wood_asset":geom["processed"],
        "encoding":"sector angle = 360 × percent / 100, followed by one uniform y-scale",
        "data_domain_bounds":[cx-radius,cy-radius*yscale,cx+radius,cy+radius*yscale],
        "bark_in_denominator":False,
        "sectors":[{
            "index":i,"category":cats[i],"value":float(values[i]),"unit":series["unit"],
            "start_deg":float(starts[i]+start_angle),"end_deg":float(ends[i]+start_angle),
            "fill_rgb":list(rgb(COLORS[i])),"raster_area_share_percent":float(observed[i]),
            "raster_area_error_pp":float(observed[i]-values[i]),
            "outer_midpoint":list(point(mids[i],radius)),
        } for i in range(len(cats))],
        "labels":sorted(label_records,key=lambda x:x["index"]),
        "leader_paths":leaders, "text_boxes":text_boxes,
    }
    (ROOT / "artifacts/render_geometry.json").write_text(
        json.dumps(geometry, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    canvas.convert("RGB").resize((w,h), Image.Resampling.LANCZOS).save(ROOT / "final.png")
    print(f"Rendered {w} × {h}; {len(cats)} sectors; max raster area error {abs(observed-values).max():.6f} pp")


if __name__ == "__main__":
    main()
