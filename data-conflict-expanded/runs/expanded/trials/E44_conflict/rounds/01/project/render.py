#!/usr/bin/env python3
"""Offline garden-yield infographic. Launch complete renders via capture_render.py."""
from pathlib import Path
from decimal import Decimal
import json, math, hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
ART = ROOT / "artifacts"
S = 4
BG = "#f5f3ea"
INK = "#21382c"
MUTED = "#657165"
PALETTE = ["#97edce", "#f4d673", "#68dbee", "#63adfc", "#c798f0", "#fa93a8"]
PHOTO = (50, 205, 2000, 1000)
BASELINE = 1070.0
TOP90 = 280.0
NET_WIDTH = 176.0
CENTERS = [380.0 + 274.0*i for i in range(6)]
RESAMPLE = Image.Resampling.LANCZOS

def rgb(h):
    return tuple(int(h[i:i+2], 16) for i in (1,3,5))

def pt(p):
    return tuple(round(v*S) for v in p)

def font(size, style="regular"):
    names = {"regular":"DejaVuSans.ttf", "bold":"DejaVuSans-Bold.ttf", "serif":"P052-Roman.otf"}
    return ImageFont.truetype(str(ASSETS / names[style]), round(size*S))

def draw_text(draw, xy, value, size, fill, style="regular", anchor="la"):
    draw.text(pt(xy), str(value), font=font(size,style), fill=fill, anchor=anchor)

def line(draw, coords, fill, width=1):
    draw.line([pt(p) for p in coords], fill=fill, width=max(1,round(width*S)))

def rounded(draw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(pt(box),radius=radius*S,fill=fill,outline=outline,width=round(width*S))

def source_sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

class Materials:
    def __init__(self):
        original = Image.open(ASSETS/"twine.png").convert("RGBA")
        ar = np.array(original)
        ys,xs=np.nonzero(ar[:,:,3]>180)
        self.twine = original.crop((int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)))
        self.peg = Image.open(ASSETS/"wooden_peg.png").convert("RGBA")
        self.tile_cache={}

    def rope_tile(self, color, thickness, opacity):
        key=(color,thickness,opacity)
        if key not in self.tile_cache:
            w,h=self.twine.size
            h2=max(2,round(thickness*S))
            # Keep photographic fibre proportions. Tile rather than stretch all texture.
            p=self.twine.resize((round(w*h2/h),h2),RESAMPLE)
            a=np.array(p).astype(float)
            lum=a[:,:,:3].mean(2)/255.
            shade=.58+.42*lum
            for ch,c in enumerate(rgb(color)):
                a[:,:,ch]=np.clip(c*shade,0,255)
            a[:,:,3]*=opacity
            self.tile_cache[key]=Image.fromarray(a.astype(np.uint8),"RGBA")
        return self.tile_cache[key]

    def rope(self, image, a, b, color, thickness=1.45, opacity=1., shadow=True):
        dx,dy=b[0]-a[0],b[1]-a[1]
        length=math.hypot(dx,dy)
        if length<.005:return
        tile=self.rope_tile(color,thickness,opacity)
        lengthpx=max(1,round(length*S))
        strip=Image.new("RGBA",(lengthpx,tile.height),(0,0,0,0))
        for x in range(0,lengthpx,tile.width):strip.alpha_composite(tile,(x,0))
        angle=-math.degrees(math.atan2(dy,dx))
        rot=strip.rotate(angle,resample=Image.Resampling.BICUBIC,expand=True)
        cx,cy=(a[0]+b[0])/2*S,(a[1]+b[1])/2*S
        pos=(round(cx-rot.width/2),round(cy-rot.height/2))
        if shadow:
            sh=Image.new("RGBA",rot.size,(10,9,8,0))
            sh.putalpha(rot.getchannel("A").point(lambda q:round(q*.56)))
            image.alpha_composite(sh,(pos[0]+round(.9*S),pos[1]+round(1.3*S)))
        image.alpha_composite(rot,pos)

    def stake(self,image,tip,tail):
        length=math.dist(tip,tail)
        p=self.peg.resize((round(length*S),round(4.5*S)),RESAMPLE)
        angle=-math.degrees(math.atan2(tip[1]-tail[1],tip[0]-tail[0]))
        p=p.rotate(angle,resample=Image.Resampling.BICUBIC,expand=True)
        cx,cy=(tip[0]+tail[0])/2*S,(tip[1]+tail[1])/2*S
        pos=(round(cx-p.width/2),round(cy-p.height/2))
        sh=Image.new("RGBA",p.size,(13,12,8,0))
        sh.putalpha(p.getchannel("A").point(lambda q:round(q*.8)))
        sh=sh.filter(ImageFilter.GaussianBlur(1.0*S))
        image.alpha_composite(sh,(pos[0]+round(1.3*S),pos[1]+round(2*S)))
        image.alpha_composite(p,pos)

def clipped_diagonal(xl,xr,yt,yb,c,sign):
    # A 45-degree line y = sign*(x-xl) + c clipped to the exact data rectangle.
    points=[]
    for x in [xl,xr]:
        y=sign*(x-xl)+c
        if yt-1e-7<=y<=yb+1e-7:points.append((x,y))
    for y in [yt,yb]:
        x=xl+(y-c)/sign
        if xl-1e-7<=x<=xr+1e-7:points.append((x,y))
    unique=[]
    for p in points:
        if not any(math.dist(p,q)<1e-5 for q in unique):unique.append(p)
    if len(unique)<2:return None
    return max(((a,b) for i,a in enumerate(unique) for b in unique[i+1:]),key=lambda t:math.dist(*t))

def main():
    ART.mkdir(exist_ok=True)
    inputs=json.loads((ROOT/"input.json").read_text(encoding="utf-8"))
    brief=json.loads((ROOT/inputs["brief"]).read_text(encoding="utf-8"))
    data=json.loads((ROOT/inputs["data"]).read_text(encoding="utf-8"),parse_float=Decimal)
    core=json.loads((ROOT/inputs["critical_metaphor_contract"]).read_text(encoding="utf-8"))
    assert (ROOT/inputs["reference"]).is_file(), "Frozen visual reference must remain available."
    assert data["synthetic"] and data["value_axis"]["minimum"]==0 and data["value_axis"]["maximum"]==90
    assert data["value_axis"]["scale"]=="linear"
    assert core["semantic_roles_zh"]==["Harvest yield (kg)"]
    categories=data["categories"]
    series=data["series"][0]
    values=series["values"]
    assert len(categories)==len(values)==6 and len(data["series"])==1
    assert sum(values)==data["series_totals"][series["id"]]
    W,H=brief["width"],brief["height"]
    assert (W,H)==(2100,1295)
    canvas=Image.new("RGBA",(W*S,H*S),BG)
    d=ImageDraw.Draw(canvas)
    mat=Materials()

    # Clear editorial header.
    draw_text(d,(64,28),"G A R D E N   T R I A L S",17,INK,"bold")
    draw_text(d,(61,58),data["title"].removeprefix("SYNTHETIC "),76,INK,"serif")
    draw_text(d,(66,153),series["label"]+" ("+series["unit"]+")  /  Six experimental garden plots",23,MUTED)
    rounded(d,(1651,55,2034,108),26,INK)
    draw_text(d,(1842.5,81.5),data["visible_notice"],22,BG,"bold","mm")
    draw_text(d,(2033,147),"A shared linear scale · 0–90 kg",21,MUTED,anchor="ra")

    # Image-generated earth, grass, tools and human bodies form the main field.
    px,py,pw,ph=PHOTO
    photo=Image.open(ASSETS/"raw_01.png").convert("RGBA").resize((pw*S,ph*S),RESAMPLE)
    canvas.alpha_composite(photo,pt((px,py)))
    d=ImageDraw.Draw(canvas)
    # A thin keyline keeps the editorial frame precise without boxing individual nets.
    d.rectangle(pt((px,py,px+pw,py+ph)),outline="#d8dacb",width=S)

    scale=(BASELINE-TOP90)/float(data["value_axis"]["maximum"])
    axis_x=246.
    grid=Image.new("RGBA",canvas.size,(0,0,0,0))
    gd=ImageDraw.Draw(grid)
    for val in range(0,91,15):
        y=BASELINE-val*scale
        if val>0:
            line(gd,[(axis_x,y),(1870,y)],(221,218,187,52),.65)
        line(gd,[(axis_x-8,y),(axis_x+5,y)],(246,241,216,220),1.2)
    line(gd,[(axis_x,TOP90-5),(axis_x,BASELINE)],(246,241,216,190),1)
    canvas.alpha_composite(grid)
    d=ImageDraw.Draw(canvas)
    for val in range(0,91,15):
        draw_text(d,(axis_x-17,BASELINE-val*scale),str(val),19,"#fffdf1",anchor="rm")
    draw_text(d,(axis_x-3,TOP90-31),series["unit"],18,"#fffdf1","bold","mm")
    # Physical pale common baseline, with a discreet 0 kg designation.
    mat.rope(canvas,(axis_x,BASELINE),(1870,BASELINE),"#e9e3c9",1.35,.87)
    d=ImageDraw.Draw(canvas)
    draw_text(d,(1868,BASELINE+26),"0 kg · shared baseline",17,"#f8f5e6",anchor="ra")

    # A data-derived reading note in the genuine empty part of this distribution.
    # It does not carry a second geometric scale.
    ratio=max(values)/min(values)
    rounded(d,(347,365,1088,579),12,(22,33,25,208))
    draw_text(d,(379,389),"Y I E L D   R A N G E",15,"#bcd4bd","bold")
    draw_text(d,(374,418),format(ratio,".0f")+"×",65,"#f8f7e7","serif")
    draw_text(d,(584,433),"highest to lowest",25,"#f8f7e7")
    draw_text(d,(584,470),str(min(values))+" kg → "+str(max(values))+" kg",22,"#e1e7d5")
    draw_text(d,(379,535),"Net length from the baseline represents yield.",21,"#e1e7d5")

    geometry=[]
    for i,(category,value,color,center) in enumerate(zip(categories,values,PALETTE,CENTERS)):
        height=float(value)*scale
        top=BASELINE-height
        xl,xr=center-NET_WIDTH/2,center+NET_WIDTH/2
        # Four anchors stay outside the quantitative rectangle. Their size does not encode data.
        for x,side in [(xl,-1),(xr,1)]:
            for y,orientation in [(top,-1),(BASELINE,1)]:
                tip=(x+side*5.5,y)
                tail=(x+side*29,y+orientation*13)
                mat.stake(canvas,tip,tail)
                mat.rope(canvas,tip,(x,y),color,1.05)
        # Diagonal twine is clipped, without increasing any short data height.
        for sign in [-1,1]:
            lo=math.floor((top-NET_WIDTH-BASELINE)/26)-1
            hi=math.ceil((BASELINE+NET_WIDTH-BASELINE)/26)+1
            for k in range(lo,hi+1):
                segment=clipped_diagonal(xl,xr,top,BASELINE,BASELINE+k*26,sign)
                if segment:
                    mat.rope(canvas,*segment,color,1.10,.91)
        for x in [xl+44,xl+88,xl+132]:
            mat.rope(canvas,(x,top),(x,BASELINE),color,.95,.78)
        # The top and bottom rope centerlines define the measured length.
        mat.rope(canvas,(xl,top),(xr,top),color,1.45)
        mat.rope(canvas,(xl,BASELINE),(xr,BASELINE),color,1.45)
        mat.rope(canvas,(xl,top),(xl,BASELINE),color,1.45)
        mat.rope(canvas,(xr,top),(xr,BASELINE),color,1.45)
        # Tie knots are tiny physical construction details, not extra length.
        kd=ImageDraw.Draw(canvas)
        for x in [xl,xr]:
            for y in [top,BASELINE]:
                kd.ellipse(pt((x-1.25,y-1.25,x+1.25,y+1.25)),fill=color)
        geometry.append({"category":category,"value":str(value),"unit":series["unit"],
                         "center_x_px":center,"left_x_px":xl,"right_x_px":xr,
                         "top_rope_center_y_px":top,"baseline_rope_center_y_px":BASELINE,
                         "encoded_length_px":height,"color":color,
                         "anchor_count":4,"mesh_pitch_px":26})
    # Restore the generated foreground gardener over the twine at the touch point.
    # The source photo stays in precisely the same coordinate transform.
    fg=Image.open(ASSETS/"gardener_foreground.png").convert("RGBA")
    crop=(1554,183,1770,310)
    fx,fy=px+crop[0]*pw/1774,py+crop[1]*ph/887
    fg=fg.resize((round(fg.width*pw/1774*S),round(fg.height*ph/887*S)),RESAMPLE)
    canvas.alpha_composite(fg,pt((fx,fy)))

    # Values sit in a quiet table strip, each under the corresponding physical plot.
    d=ImageDraw.Draw(canvas)
    rounded(d,(269,1138,1861,1237),8,"#f5f3ea")
    for i,(g,color) in enumerate(zip(geometry,PALETTE)):
        cx=g["center_x_px"]
        line(d,[(cx-76,1150),(cx+76,1150)],color,3)
        if i:
            line(d,[(cx-137,1158),(cx-137,1220)],"#d9ded1",1)
        draw_text(d,(cx,1165),g["category"],22,INK,"bold","ma")
        draw_text(d,(cx,1198),g["value"]+" "+g["unit"],21,INK,anchor="ma")
    draw_text(d,(66,1257),"READING KEY",15,INK,"bold")
    draw_text(d,(217,1254),"Length = yield. Width, mesh spacing, stakes and people do not encode values.",19,MUTED)
    draw_text(d,(2031,1256),"Synthetic values · not field observations",16,MUTED,anchor="ra")

    output=canvas.convert("RGB").resize((W,H),RESAMPLE)
    output.save(ROOT/"final.png",compress_level=6)
    (ART/"geometry.json").write_text(json.dumps({
        "canvas":[W,H],"series":series["label"],"unit":series["unit"],
        "value_axis":{"minimum":0,"maximum":90,"scale":"linear"},
        "pixel_scale_per_kg":scale,"common_baseline_y_px":BASELINE,
        "net_width_px":NET_WIDTH,"supersampling":S,
        "measurement_definition":"Top and baseline rope centerlines, in ground screen pixels.",
        "plots":geometry,"derived_max_to_min_ratio":str(ratio),
        "foreground_overlap":{"plot":"Plot F","source_crop":list(crop),"source_to_canvas_scale":pw/1774},
        "frozen_input_sha256":{n:source_sha(ROOT/n) for n in ["input.json","brief.json","data.json","core.json","reference.png"]},
        "generated_raw_sha256":{n:source_sha(ASSETS/n) for n in ["raw_01.png","raw_02.png"]}
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("Saved final.png (2100×1295); all six plot lengths mapped linearly from data.json.")

if __name__=="__main__":
    main()

