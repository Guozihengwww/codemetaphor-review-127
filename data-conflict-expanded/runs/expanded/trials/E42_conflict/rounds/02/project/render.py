"""Offline, data-bound rendering of the supplied sesame revenue composition.

Run from this project with Python. Complete experimental renders must be
launched through the supplied capture_render.py wrapper.
"""
from pathlib import Path
import json
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parent
S = 2
RESAMPLE = Image.Resampling.LANCZOS


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def main():
    inputs = read_json("input.json")
    brief = read_json(inputs["brief"])
    data = read_json(inputs["data"])
    core = read_json(inputs["critical_metaphor_contract"])
    assert (ROOT / inputs["reference"]).is_file()
    assert core["visible_title"] == data["title"]
    assert data["synthetic"] is True
    width, height = brief["width"], brief["height"]
    series, = data["series"]
    categories = data["categories"]
    values = series["values"]
    unit = data["unit_mark"]["value"]
    assert series["unit"] == "USD"
    assert len(categories) == len(values) == 5
    assert not data["unit_mark"]["fractional_marks"]
    assert all(v % unit == 0 for v in values)
    counts = [v // unit for v in values]
    total = sum(values)
    assert total == data["series_totals"][series["id"]]
    vmin, vmax = data["value_axis"]["minimum"], data["value_axis"]["maximum"]
    assert vmin == 0 and data["value_axis"]["scale"] == "linear"

    # A single mapping controls the ruler, every whole top face, and labels.
    plot_top, baseline = 445, 1765
    plot_height = baseline - plot_top
    x_axis, x_right = 274, 1900
    centers = np.linspace(480, 1720, len(categories)).tolist()
    slab_width = 156
    pixels_per_dollar = plot_height / (vmax - vmin)
    unit_height = unit * pixels_per_dollar
    assert abs(unit_height - round(unit_height)) < 1e-8

    def y_of(value):
        return baseline - (value - vmin) * pixels_per_dollar

    # Matte blue tabletop, with restrained photographic grain.
    rng = np.random.default_rng(42017)
    yy, xx = np.mgrid[0:height, 0:width]
    light = 5.5 * np.exp(-(((xx - 720) / 1200) ** 2 + ((yy - 520) / 1700) ** 2))
    vignette = 3.0 * ((xx - width / 2) / width) ** 2 + 2.0 * (yy / height)
    grain = rng.normal(0, .42, (height, width))
    base = np.array([137., 199., 220.])
    bg = np.clip(base[None, None, :] + (light - vignette + grain)[:, :, None], 0, 255).astype(np.uint8)
    canvas = Image.fromarray(bg).convert("RGBA").resize((width * S, height * S), RESAMPLE)
    draw = ImageDraw.Draw(canvas)
    ink = (77, 49, 30, 255)
    secondary = (49, 91, 104, 255)
    rule = (73, 119, 133, 105)
    cream = (255, 242, 209, 255)

    def font(size, kind="sans"):
        return ImageFont.truetype(str(ROOT / "assets/fonts" / (kind + ".otf")), round(size * S))

    def text_at(x, y, text, size=30, kind="sans", fill=ink, anchor="lt"):
        draw.text((round(x * S), round(y * S)), text, font=font(size, kind), fill=fill, anchor=anchor)

    def tracked(x, y, text, size=22, spacing=2.3, fill=secondary, anchor="left"):
        f = font(size)
        widths = [draw.textlength(c, font=f) / S for c in text]
        length = sum(widths) + spacing * max(0, len(text)-1)
        if anchor == "right":
            x -= length
        for ch, adv in zip(text, widths):
            text_at(x, y, ch, size, fill=fill)
            x += adv + spacing

    def line(points, fill=ink, w=1):
        draw.line([(round(x*S), round(y*S)) for x,y in points], fill=fill, width=max(1, round(w*S)))

    # Header and explicit synthetic provenance.
    draw.rounded_rectangle((112*S, 65*S, 384*S, 111*S), radius=23*S, fill=ink)
    text_at(248, 89, data["visible_notice"], 23, "sans_bold", cream, "mm")
    tracked(1888, 81, f"{len(categories)} WEEKS / {len(data['series'])} SERIES", 21, 2.5, anchor="right")
    text_at(108, 140, "Weekly revenue", 108, "serif")
    text_at(108, 239, "in sesame slabs", 108, "serif")
    text_at(1890, 172, f"${total:,}", 62, "sans", anchor="rt")
    tracked(1888, 250, "TOTAL SYNTHETIC REVENUE", 20, 1.6, anchor="right")
    text_at(113, 362, f"{series['label']} · {series['unit']}", 27, fill=secondary)

    faces = [Image.open(ROOT / f"assets/slab_{i:02d}.png").convert("RGBA") for i in range(1,5)]

    def place_face(face, x, y, w, h, brightness=1.0):
        tile = face.resize((round(w*S), round(h*S)), RESAMPLE)
        if brightness != 1:
            ar = np.array(tile)
            ar[:,:,:3] = np.clip(ar[:,:,:3].astype(float)*brightness,0,255).astype(np.uint8)
            tile = Image.fromarray(ar)
        # Edge bevels stay strictly inside the top-face extent.
        td = ImageDraw.Draw(tile)
        tw, th = tile.size
        td.line((1, 0, tw-2, 0), fill=(255,224,157,255), width=1)
        td.line((1, th-1, tw-2, th-1), fill=(175,110,42,255), width=1)
        alpha = Image.new("L", tile.size, 0)
        ImageDraw.Draw(alpha).rounded_rectangle((0,0,tw-1,th-1), radius=2*S, fill=255)
        tile.putalpha(alpha)
        canvas.alpha_composite(tile, (round(x*S), round(y*S)))

    # Legend uses the exact same top-face size as all the data marks.
    place_face(faces[0], 1302, 359, slab_width, unit_height)
    text_at(1480, 358, f"1 whole slab = ${unit:,}", 27)
    line([(112,411),(1888,411)], fill=(64,112,127,130),w=.8)

    # Axis tick labels are explicit USD amounts; no axis break or clipping.
    tick_values = list(range(0, vmax+1, unit*5))
    if vmax not in tick_values:
        tick_values.append(vmax)
    for v in tick_values:
        y = y_of(v)
        is_major = v % (unit*10) == 0 or v == vmax
        if v > 0 and is_major:
            line([(x_axis+1,y),(x_right,y)], fill=(207,233,241,155), w=.85)
        line([(x_axis-(13 if is_major else 7),y),(x_axis,y)],fill=(114,80,53,255),w=1.6)
        if is_major:
            text_at(239, y, f"${v:,}", 29, anchor="rm")
    line([(x_axis,plot_top),(x_axis,baseline),(x_right,baseline)],fill=(110,71,43,255),w=2.5)

    # All generated crumbs are used on the tabletop, away from complete marks.
    crumb_positions = [
        (340,1606,13,-21), (617,1670,17,27), (916,1731,11,63),
        (1201,1527,12,12), (1537,1603,15,-35), (1861,1643,11,81),
        (1570,1116,10,35), (1849,867,11,-30), (1533,710,8,25),
        (683,1571,4,31), (954,1651,5,-25), (1278,1733,4,78),
        (1477,1465,4,49), (1651,1522,4,-36), (354,1394,4,11),
    ]
    for k, (cx, cy, target_w, angle) in enumerate(crumb_positions, 1):
        crumb = Image.open(ROOT/f"assets/crumb_{k:02d}.png").convert("RGBA")
        cw = round(target_w*S)
        ch = round(cw * crumb.height/crumb.width)
        crumb = crumb.resize((cw,max(1,ch)), RESAMPLE).rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
        shadow = Image.new("RGBA", (crumb.width+12*S,crumb.height+12*S), (0,0,0,0))
        shadow_alpha = Image.new("L",shadow.size)
        shadow_alpha.paste(crumb.getchannel("A"),(6*S,6*S))
        shadow_alpha = shadow_alpha.filter(ImageFilter.GaussianBlur(1.5*S)).point(lambda a:int(a*.22))
        shadow.paste((28,65,74),mask=shadow_alpha)
        canvas.alpha_composite(shadow,(round(cx*S)-6*S+S,round(cy*S)-6*S+S))
        canvas.alpha_composite(crumb,(round(cx*S),round(cy*S)))

    records = []
    for i, (category,value,count,cx) in enumerate(zip(categories,values,counts,centers)):
        x0, x1 = cx-slab_width/2, cx+slab_width/2
        ytop = y_of(value)
        # Soft rightward cast shadow has no vertical extent outside the column.
        sh_w = round((slab_width+18)*S)
        sh_h = round((baseline-ytop)*S)
        sh_alpha = Image.new("L", (sh_w,sh_h),0)
        sd = ImageDraw.Draw(sh_alpha)
        sd.rectangle((7*S,0,(slab_width+7)*S,sh_h-1),fill=63)
        sh_alpha = sh_alpha.filter(ImageFilter.GaussianBlur(3*S))
        sh = Image.new("RGBA",sh_alpha.size,(38,71,68,0))
        sh.putalpha(sh_alpha)
        canvas.alpha_composite(sh,(round(x0*S),round(ytop*S)))
        cells=[]
        for j in range(count):
            top = y_of((j+1)*unit)
            bottom = y_of(j*unit)
            face = faces[(j*3+i)%len(faces)]
            place_face(face,x0,top,slab_width,bottom-top,1.0 + ((j+i)%3-1)*.016)
            cells.append({"unit_from_baseline":j+1,"rect_xyxy":[x0,top,x1,bottom]})
        draw = ImageDraw.Draw(canvas)
        text_at(cx, ytop-18, f"${value:,}", 38, "sans_bold", anchor="mb")
        line([(cx,baseline+1),(cx,baseline+12)],fill=(110,71,43,255),w=2)
        tracked_label = category
        text_at(cx,1800,tracked_label,34,"sans_bold",anchor="mt")
        text_at(cx,1848,f"{count} {'slab' if count==1 else 'slabs'}",26,fill=secondary,anchor="mt")
        records.append({"category":category,"series":series['label'],"value":value,"unit":"USD",
                        "whole_slab_count":count,"top_face_union_xyxy":[x0,ytop,x1,baseline],
                        "height_px":baseline-ytop,"cells":cells})

    draw = ImageDraw.Draw(canvas)
    tracked(239,1805,"WEEK",19,2,anchor="right")
    tracked(239,1851,"SLABS",19,2,anchor="right")

    # A compact note uses the negative space created by the outlying week.
    # It is calculated from the current data and is not a separate chart.
    peak = max(range(len(values)), key=values.__getitem__)
    note_x, note_y = 402, 648
    text_at(note_x,note_y,f"{counts[peak]} of {sum(counts)} slabs",54,"serif",fill=ink)
    text_at(note_x,note_y+69,f"{categories[peak]} contributes ${values[peak]:,} of the",29,fill=secondary)
    text_at(note_x,note_y+108,f"${total:,} five-week total.",29,fill=secondary)

    line([(112,1905),(1888,1905)],fill=rule,w=1)
    text_at(112,1937,f"Linear scale · ${vmin:,}–${vmax:,} · Whole slabs only",23,fill=secondary)
    text_at(1888,1937,"Crumbs and shadows carry no value.",23,fill=secondary,anchor="rt")

    output = canvas.convert("RGB").resize((width,height),RESAMPLE)
    output.save(ROOT/"final.png",compress_level=7)
    geometry = {
        "canvas_px":[width,height],"axis_range_usd":[vmin,vmax],"axis_scale":"linear",
        "plot_top_px":plot_top,"baseline_px":baseline,"pixels_per_dollar":pixels_per_dollar,
        "unit_value_usd":unit,"unit_height_px":unit_height,"unit_width_px":slab_width,
        "data_top_faces":sum(counts),"legend_top_faces":1,"records":records,
        "generated_raw":"assets/raw_01.png","reference_pixels_used":False,
    }
    (ROOT/"artifacts").mkdir(exist_ok=True)
    (ROOT/"artifacts/render_geometry.json").write_text(json.dumps(geometry,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"Rendered {width} x {height}; {sum(counts)} whole data slabs, {counts}; linear 0–{vmax} USD.")


if __name__ == "__main__":
    main()
