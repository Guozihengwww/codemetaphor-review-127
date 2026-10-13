#!/usr/bin/env python3
"""Offline renderer for the frozen synthetic melting-Earth infographic.

Run through capture_render.py when creating a complete experimental candidate.
All quantitative positions use one unprojected linear coordinate transform.
"""
from pathlib import Path
import json, math, hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
S = 2
def read_json(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))
data, brief, core = read_json("data.json"), read_json("brief.json"), read_json("core.json")
inputs = read_json("input.json")
assert inputs["data"] == "data.json" and inputs["critical_metaphor_contract"] == "core.json"
assert len(data["categories"]) == len(data["time_points"])
assert data["value_axis"] == {"minimum": -1, "maximum": 15, "scale": "linear"}
assert [s["id"] for s in data["series"]] == ["price_inflation", "output_growth"]
W, H = brief["width"], brief["height"]
BG = "#eee9e1"
BOARD = "#303639"
PAPER = "#f5f0e8"
INK = "#121313"
RED = "#d72727"
CREAM = "#f2ece2"
MUTED = "#bec3c2"
COLORS = {"price_inflation": INK, "output_growth": RED}
FONT_REG = ROOT / "assets/fonts/DejaVuSans.ttf"
FONT_BOLD = ROOT / "assets/fonts/DejaVuSans-Bold.ttf"
font_cache = {}
def font(size, bold=False):
    key=(size,bold)
    if key not in font_cache:
        font_cache[key]=ImageFont.truetype(str(FONT_BOLD if bold else FONT_REG), round(size*S))
    return font_cache[key]

# Full-canvas drawing is performed only from this executable entry point.
im = Image.new("RGB", (W*S,H*S), BG)
D = ImageDraw.Draw(im, "RGBA")
def refresh():
    global D
    D=ImageDraw.Draw(im,"RGBA")
def rect(box, fill, outline=None, width=1, radius=0):
    b=tuple(round(v*S) for v in box)
    if radius:
        D.rounded_rectangle(b, radius=round(radius*S), fill=fill, outline=outline, width=round(width*S))
    else:
        D.rectangle(b,fill=fill,outline=outline,width=round(width*S))
def line(pts, fill, width=1):
    pp=[(round(x*S),round(y*S)) for x,y in pts]
    D.line(pp,fill=fill,width=max(1,round(width*S)),joint="curve")
def polygon(pts, fill, outline=None, width=1):
    pp=[(round(x*S),round(y*S)) for x,y in pts]
    D.polygon(pp,fill=fill)
    if outline: D.line(pp+[pp[0]],fill=outline,width=round(width*S),joint="curve")
def circle(x,y,r,fill,outline=None,width=1):
    D.ellipse((round((x-r)*S),round((y-r)*S),round((x+r)*S),round((y+r)*S)),fill=fill,outline=outline,width=round(width*S))
text_records=[]
def text(x,y,txt,size=24,fill=INK,bold=False,align="left",record=True):
    f=font(size,bold)
    tw=D.textlength(txt,font=f)/S
    if align=="center": x-=tw/2
    elif align=="right": x-=tw
    D.text((round(x*S),round(y*S)),txt,font=f,fill=fill,anchor="lt",stroke_width=0)
    if record: text_records.append({"text":txt,"x":round(x,3),"y":round(y,3),"size":size})
    return tw
def dashed(a,b,fill,width=1,dash=9,gap=7):
    dx,dy=b[0]-a[0],b[1]-a[1]
    length=math.hypot(dx,dy)
    if not length: return
    t=0
    while t<length:
        e=min(t+dash,length)
        line([(a[0]+dx*t/length,a[1]+dy*t/length),(a[0]+dx*e/length,a[1]+dy*e/length)],fill,width)
        t+=dash+gap

# Header.
rect((108,32,325,64), RED, radius=3)
text(216.5,39,data["visible_notice"],17,CREAM,True,"center")
text(108,83,"Price inflation and output growth",53,INK,True)
text(111,150,"Annual change from the previous year · Price level and real output",23,"#5c6160")
text(1889,58,"1960–1982",31,INK,True,"right")
text(1889,103,"Annual series (%)",20,"#5c6160",False,"right")
text(1889,137,"1982 projected",19,"#5c6160",False,"right")

# Dark backboard and shallow-perspective plinth. These are not data transforms.
rect((200,191,1800,865),BOARD,outline="#252b2e",width=2)
line([(200,191),(1800,191)],"#4b5355",2)
polygon([(200,865),(1800,865),(1930,998),(70,998)],"#d9d5cf","#363d40",2)
polygon([(70,998),(1930,998),(1930,1070),(70,1070)],"#ccc8c2","#363d40",2)
line([(74,1001),(1926,1001)],"#e9e5df",2)
line([(200,865),(1800,865)],"#737b7c",1)

# Generated globe: continuous sphere -> pool -> drips, with decorative-only warp.
asset=Image.open(ROOT/"assets/globe_melt.png").convert("RGBA")
asset_x=155
asset_w=1690
segments=[(0,700,200,860),(700,835,860,986),(835,1024,986,1055)]
rgba=im.convert("RGBA")
for raw_y0,raw_y1,dst_y0,dst_y1 in segments:
    part=asset.crop((0,raw_y0,asset.width,raw_y1))
    part=part.resize((asset_w*S,round((dst_y1-dst_y0)*S)),Image.Resampling.LANCZOS)
    rgba.alpha_composite(part,(asset_x*S,round(dst_y0*S)))
im=rgba.convert("RGB")
refresh()

# Mapping: time is horizontal, percent vertical; absolutely no perspective.
X0,X1=575.0,1448.0
YTOP,YBOTTOM=300.0,918.0
VMIN,VMAX=data["value_axis"]["minimum"],data["value_axis"]["maximum"]
T0,T1=data["time_points"][0],data["time_points"][-1]
def xy(t,v):
    return (X0+(t-T0)/(T1-T0)*(X1-X0),YBOTTOM-(v-VMIN)/(VMAX-VMIN)*(YBOTTOM-YTOP))
zero_y=xy(T0,0)[1]
points={}
for s in data["series"]:
    assert len(s["values"])==len(data["categories"])
    points[s["id"]]=[xy(t,v) for t,v in zip(data["time_points"],s["values"])]

# Left semantic key, legible on the backboard.
text(253,250,"ANNUAL CHANGE",18,MUTED,True)
text(253,281,"Percent (%)",25,CREAM,True)
rect((254,377,335,396),PAPER,radius=9)
line([(263,386),(326,386)],INK,6.5)
text(253,416,"Price inflation",23,CREAM,True)
text(253,451,"Price level",18,MUTED)
line([(259,527),(331,527)],RED,6.5)
text(253,557,"Output growth",23,CREAM,True)
text(253,592,"Real output",18,MUTED)
line([(254,651),(444,651)],"#596064",1)
text(253,678,"Same linear scale",18,MUTED)
text(253,710,"23 years · 2 series",18,MUTED)

# Horizontal scale stays independent of the globe's curved graticule.
for tick in [-1,0,5,10,15]:
    y=xy(T0,tick)[1]
    if tick==0:
        line([(531,y+0.8),(1511,y+0.8)],(245,240,231,220),4.0)
        dashed((531,y),(1511,y),"#59615f",1.8,8,6)
    else:
        line([(531,y),(1504,y)],(243,241,231,95),1.25)
    line([(514,y),(529,y)],CREAM if y<865 else "#48504f",1.7)
    text(502,y-11,str(tick).replace("-","−"),22,CREAM if y<865 else INK,False,"right")
line([(521,YTOP),(521,YBOTTOM)],(194,201,199,170),1)
text(513,953,"Year",18,"#5c615d",False,"right")

# Forecast identity is attached to 1982, not to a y-value.
forecast=float(data["forecast_boundary"])
forecast_i=data["time_points"].index(forecast)
forecast_x=xy(forecast,0)[0]
dashed((forecast_x,292),(forecast_x,948),(232,231,222,160),1.25,4,8)
text(1560,746,"1982",26,CREAM,True)
text(1560,782,"PROJECTED",17,MUTED,True)
dashed((1561,822),(1662,822),CREAM,3,10,8)
circle(1662,822,4.5,BOARD,CREAM,2)
text(1560,842,"Open endpoints",15,MUTED)

# Draw translucent halos first, then both cores, so close values keep both colors.
for sid,pp in points.items():
    for i in range(len(pp)-1):
        a,b=pp[i],pp[i+1]
        if i+1>=forecast_i: dashed(a,b,(246,241,231,150),10.5,10,7)
        else: line([a,b],(246,241,231,150),10.5)
for sid,pp in points.items():
    color=COLORS[sid]
    for i in range(len(pp)-1):
        a,b=pp[i],pp[i+1]
        if i+1>=forecast_i: dashed(a,b,color,6.4,10,7)
        else: line([a,b],color,6.4)
    for i,(x,y) in enumerate(pp):
        if i>=forecast_i:
            circle(x,y,6.2,PAPER,color,2.8)
        else:
            circle(x,y,3.4,color)

# Minimal key-point labels. All values are read from the frozen JSON.
def value(sid,year):
    s=next(s for s in data["series"] if s["id"]==sid)
    return s["values"][data["categories"].index(str(year))]
def valstr(v):
    return f"{v:.1f}%".replace("-","−")
def point(sid,year):
    return points[sid][data["categories"].index(str(year))]
def label_card(x,y,year,sid,w=119):
    rect((x,y,x+w,y+62),(247,242,233,243),radius=5)
    text(x+11,y+8,str(year),16,"#666960")
    text(x+11,y+29,valstr(value(sid,year)),28,COLORS[sid],True)

p=point("price_inflation",1974)
line([p,(p[0]+17,p[1]-15)],INK,1.3)
label_card(p[0]+18,p[1]-66,1974,"price_inflation",125)
p=point("price_inflation",1980)
label_card(p[0]-55,p[1]-91,1980,"price_inflation",122)
line([(p[0],p[1]-25),p],INK,1.3)
p=point("output_growth",1962)
label_card(p[0]-70,p[1]-87,1962,"output_growth",110)
line([(p[0]-9,p[1]-26),p],RED,1.3)

# Exact paired label makes the 0.1-point 1965 separation readable without inflation.
p1=point("price_inflation",1965)
p2=point("output_growth",1965)
line([((p1[0]+p2[0])/2,(p1[1]+p2[1])/2),(p1[0],798)],"#5e635e",1.2)
rect((p1[0]-107,798,p1[0]+133,838),(247,242,233,240),radius=4)
text(p1[0]-96,809,"1965",16,"#666960")
text(p1[0]-37,805,f'{value("price_inflation",1965):.1f}',22,INK,True)
text(p1[0]+13,807,"/",20,"#787c76")
text(p1[0]+31,805,valstr(value("output_growth",1965)),22,RED,True)

# Direct end values sit outside the object, on light keyed tabs.
for sid in ["price_inflation","output_growth"]:
    p=points[sid][-1]
    line([(p[0]+8,p[1]),(1514,p[1])],COLORS[sid],1.4)
    rect((1514,p[1]-24,1634,p[1]+24),PAPER,radius=4)
    text(1574,p[1]-15,valstr(value(sid,1982)),27,COLORS[sid],True,"center")
p=point("output_growth",1975)
line([p,(p[0]+21,p[1]+21)],RED,1.5)
rect((p[0]+20,p[1]+12,p[0]+162,p[1]+46),(247,242,233,240),radius=3)
text(p[0]+31,p[1]+17,valstr(value("output_growth",1975)),26,RED,True)
text(p[0]-5,p[1]+26,"1975",16,"#656860",False,"right")

# Calendar axis: every point has a tick; alternate years carry labels.
axis_y=949
line([(X0,axis_y),(X1,axis_y)],"#626965",1.3)
for i,(t,category) in enumerate(zip(data["time_points"],data["categories"])):
    x=xy(t,0)[0]
    line([(x,axis_y-3),(x,axis_y+7 if i%2==0 else axis_y+3)],"#59615d",1.2)
    if i%2==0 or i==len(data["categories"])-1:
        # Paper-colored backing is confined to text, and does not alter marks.
        labelw=D.textlength(category,font=font(20))/S
        rect((x-labelw/2-4,axis_y+11,x+labelw/2+4,axis_y+39),(236,232,223,220),radius=2)
        text(x,axis_y+15,category,20,INK,i==len(data["categories"])-1,"center")

# The physical front face remains clean; long drips are visible across it.
text(105,1024,"PRICE LEVEL",17,"#535a57",True)
text(105,1048,"and real output",15,"#656b66")
text(1894,1026,"One shared scale",17,"#535a57",True,"right")
text(1894,1050,"−1% to 15%",15,"#656b66",False,"right")

text(107,1091,"SYNTHETIC RESEARCH SERIES",17,INK,True)
text(455,1090,"Illustrative values only · Not observed economic statistics",19,"#666b66")
text(1894,1091,"1982 projected",18,"#666b66",False,"right")

im=im.resize((W,H),Image.Resampling.LANCZOS)
im.save(ROOT/"final.png")
geometry={
    "canvas":[W,H],"data_axis":{"minimum":VMIN,"maximum":VMAX,"scale":"linear"},
    "plot":{"x0":X0,"x1":X1,"y_top":YTOP,"y_bottom":YBOTTOM,"zero_y":zero_y,"pixels_per_percent":(YBOTTOM-YTOP)/(VMAX-VMIN)},
    "data_transform":"x=x0+(time-first)/(last-first)*(x1-x0); y=y_bottom-(value-min)/(max-min)*(y_bottom-y_top)",
    "forecast":{"year":forecast,"index":forecast_i,"x":forecast_x,"open_endpoint":True,"incoming_segment":"dashed"},
    "series":[],
    "asset_transform":{"x":asset_x,"width":asset_w,"raw_size":list(asset.size),"vertical_segments":segments,"changes_data_coordinates":False},
    "text":text_records,
    "frozen_dependencies":{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ["input.json","brief.json","core.json","data.json","reference.png"]},
}
for s in data["series"]:
    geometry["series"].append({
        "id":s["id"],"label":s["label"],"unit":s["unit"],"color":COLORS[s["id"]],
        "points":[{"category":c,"time":t,"value":v,"x":p[0],"y":p[1],"projected":t>=forecast}
                  for c,t,v,p in zip(data["categories"],data["time_points"],s["values"],points[s["id"]])]
    })
(ROOT/"artifacts").mkdir(exist_ok=True)
(ROOT/"artifacts/geometry.json").write_text(json.dumps(geometry,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(f"Rendered {W} × {H}; {sum(len(v) for v in points.values())} data vertices; forecast {forecast:g}.")

