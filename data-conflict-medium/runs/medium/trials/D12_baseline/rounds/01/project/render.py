"""Render the frozen nose-profile infographic using local data and generated assets.

Usage: python render.py
All complete experimental renders are invoked through capture_render.py.
No network or model calls. The same x/y mapping drives fill, contour and nodes.
"""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parent
S = 2
INK = (48, 53, 51)
GREEN = (33, 77, 72)
RUST = (137, 64, 39)
MUTED = (92, 103, 100)
PAPER = (250, 247, 238)

FONTS = {
    'serif': '/usr/share/fonts/opentype/noto/NotoSerifCJK-Medium.ttc',
    'sans': '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    'bold': '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
    'number': '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf',
    'number_bold': '/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf',
    'italic': '/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf',
}

def font(kind, size):
    return ImageFont.truetype(FONTS[kind], round(size*S), index=2 if kind in ('serif','sans','bold') else 0)

def line(draw, points, fill=INK, width=1):
    draw.line([(round(x*S),round(y*S)) for x,y in points], fill=fill, width=max(1,round(width*S)), joint='curve')

def text(draw, xy, value, size=28, kind='sans', fill=INK, anchor='lt'):
    draw.text((round(xy[0]*S),round(xy[1]*S)), str(value), font=font(kind,size), fill=fill, anchor=anchor)

def ellipse(draw, xy, radius, fill, outline=None, width=1):
    x,y=xy
    draw.ellipse(((x-radius)*S,(y-radius)*S,(x+radius)*S,(y+radius)*S),fill=fill,outline=outline,width=round(width*S))

def grid_layer(size, x0, yt, xb, yb, opacity=255):
    grid=Image.new('RGBA', size, (0,0,0,0))
    d=ImageDraw.Draw(grid)
    step=(xb-x0)/20
    for k in range(-10,30):
        x=x0+k*step
        if 97<=x<=1502:
            col=(181,200,204,opacity if k%5==0 else round(opacity*.65))
            line(d,[(x,90),(x,1908)],col,0.85 if k%5==0 else 0.65)
    # A one-year step also gives square graph paper for this frozen data extent.
    for k in range(-15,40):
        y=yt+k*step
        if 93<=y<=1905:
            col=(181,200,204,opacity if k%5==0 else round(opacity*.65))
            line(d,[(94,y),(1507,y)],col,0.85 if k%5==0 else 0.65)
    return grid

def main():
    brief=json.loads((ROOT/'brief.json').read_text(encoding='utf-8'))
    data=json.loads((ROOT/'data.json').read_text(encoding='utf-8'))
    W,H=brief['width'],brief['height']
    assert (W,H)==(1600,2000)
    vals=np.asarray(data['series'][0]['values'],dtype=float)
    times=np.asarray(data['time_points'],dtype=float)
    categories=data['categories']
    assert len(vals)==len(times)==len(categories)
    assert np.all(np.diff(times)>0)
    assert data['value_axis']['scale']=='linear'
    vmin,vmax=data['value_axis']['minimum'],data['value_axis']['maximum']
    x0,xb,yt,yb=366.,1286.,590.,1510.
    xp=lambda v: x0+(np.asarray(v)-vmin)/(vmax-vmin)*(xb-x0)
    yp=lambda t: yb-(np.asarray(t)-times[0])/(times[-1]-times[0])*(yb-yt)
    curve=PchipInterpolator(times,vals,extrapolate=False)
    value_at_y=lambda y: curve(times[0]+(yb-np.asarray(y))/(yb-yt)*(times[-1]-times[0]))

    # Near-frontal sheet with a subtle physical shadow and reproducible paper grain.
    rng=np.random.default_rng(2307)
    im=Image.new('RGBA',(W*S,H*S),(231,236,235,255))
    shadow=Image.new('RGBA',im.size,(0,0,0,0))
    d=ImageDraw.Draw(shadow)
    d.polygon([(91*S,84*S),(1517*S,89*S),(1525*S,1929*S),(88*S,1934*S)],fill=(60,65,53,44))
    shadow=shadow.filter(ImageFilter.GaussianBlur(13*S))
    im=Image.alpha_composite(im,shadow)
    noise=rng.normal(0,.65,(H,W,1))
    xx=np.linspace(0,1,W)[None,:,None]
    tint=np.array(PAPER,dtype=float)[None,None,:]-1.5*xx
    paperarr=np.clip(tint+noise,0,255).astype(np.uint8)
    paper=Image.fromarray(paperarr,'RGB').convert('RGBA').resize(im.size,Image.Resampling.BICUBIC)
    papermask=Image.new('L',im.size,0)
    pd=ImageDraw.Draw(papermask)
    pd.polygon([(85*S,77*S),(1510*S,78*S),(1514*S,1916*S),(86*S,1920*S)],fill=255)
    im=Image.composite(paper,im,papermask)
    im=Image.alpha_composite(im,grid_layer(im.size,x0,yt,xb,yb,155))
    d=ImageDraw.Draw(im)
    line(d,[(86,78),(1509,79),(1513,1915),(87,1919),(86,78)],(212,214,204),.9)

    # Editorial title and the explicit synthetic-data notice.
    text(d,(145,138),'NOSE PROCEDURES  /  鼻部手术',23,'sans',MUTED)
    d.rounded_rectangle((1136*S,125*S,1451*S,178*S),radius=7*S,fill=GREEN)
    text(d,(1293,151),data['visible_notice'],23,'bold',PAPER,'mm')
    text(d,(143,228),data['title'],84,'serif',INK)
    text(d,(146,350),'把年度数量，读成一道鼻形轮廓。',33,'sans',INK)
    text(d,(1450,363),'1995—2015  ·  5 个年度样本',23,'sans',MUTED,'rm')
    line(d,[(146,425),(1450,425)],(128,147,144),1.4)

    # Light row guides span the figure without introducing extra observations.
    for t in times:
        y=float(yp(t))
        line(d,[(x0,y),(xb,y)],(180,193,190),1.25)

    # Map every source row from its measured alpha silhouette into the current
    # data width. Source anatomy/materials stay inside a full logical area mask.
    body=np.asarray(Image.open(ROOT/'assets/nose_material.png').convert('RGB'),dtype=np.float32)
    mapping=np.load(ROOT/'assets/source_mapping.npz')
    peak_y=float(yp(times[int(np.argmax(vals))]))
    sy_anchors=np.array([mapping['y_top'],mapping['y_tip'],mapping['y_bottom']],dtype=float)
    ty_anchors=np.array([yt,peak_y,yb],dtype=float)
    # The frozen maximum is interior, so the anatomical tip maps to that sample.
    assert yt<peak_y<yb
    dy=np.arange(round((yb-yt)*S)+1,dtype=np.float32)/S+yt
    dx=np.arange(round((xb-x0)*S)+1,dtype=np.float32)/S+x0
    right_edges=xp(value_at_y(dy))
    sy=np.interp(dy,ty_anchors,sy_anchors).astype(np.float32)
    leftsrc=np.interp(sy,np.arange(len(mapping['left'])),mapping['left'])+6
    rightsrc=np.interp(sy,np.arange(len(mapping['right'])),mapping['right'])-6
    u=(dx[None,:]-x0)/(right_edges[:,None]-x0)
    sx=leftsrc[:,None]+np.clip(u,0,1)*(rightsrc-leftsrc)[:,None]
    coords=np.stack([np.broadcast_to(sy[:,None],sx.shape),sx])
    warped=np.stack([map_coordinates(body[:,:,c],coords,order=1,mode='nearest') for c in range(3)],axis=-1)
    warped=np.clip(warped,0,255).astype(np.uint8)
    region=Image.fromarray(warped,'RGB').convert('RGBA')
    regionmask=Image.fromarray((u<=1).astype(np.uint8)*255,'L')
    region.putalpha(regionmask)
    im.alpha_composite(region,(round(x0*S),round(yt*S)))
    fullmask=Image.new('L',im.size,0)
    fullmask.paste(regionmask,(round(x0*S),round(yt*S)))
    # Grid remains faintly visible through colored pencil; eye is placed later.
    interiorgrid=grid_layer(im.size,x0,yt,xb,yb,62)
    ia=np.asarray(interiorgrid.getchannel('A')).astype(np.uint16)
    ma=np.asarray(fullmask).astype(np.uint16)
    interiorgrid.putalpha(Image.fromarray((ia*ma//255).astype(np.uint8),'L'))
    im=Image.alpha_composite(im,interiorgrid)

    # Small non-data eye overlay. It occludes pigment but never removes area.
    eye=Image.open(ROOT/'assets/eye_overlay.png').convert('RGBA')
    eye_w=268
    eye_h=round(eye.height*eye_w/eye.width)
    eye=eye.resize((eye_w*S,eye_h*S),Image.Resampling.LANCZOS)
    im.alpha_composite(eye,(410*S,620*S))
    d=ImageDraw.Draw(im)

    # Stroke exactly follows the same curve as the fill, with no invented nodes.
    rows=np.linspace(yt,yb,1841)
    outline=[(float(x),float(y)) for x,y in zip(xp(value_at_y(rows)),rows)]
    line(d,[(x0,yt),outline[0]],(93,65,49),3)
    line(d,outline,(75,57,43),4.2)
    line(d,[(x0,yb),outline[-1]],(75,57,43),3.4)
    # A very fine pencil trace remains sub-pixel close to the data boundary.
    line(d,[(x-.8,y+.25) for x,y in outline],(92,67,45),.6)

    # Common axes; the chronological scale runs bottom to top.
    line(d,[(x0,563),(x0,yb+3)],INK,3.4)
    line(d,[(x0,yb),(xb+10,yb)],INK,3.4)
    line(d,[(x0-7,575),(x0,563),(x0+7,575)],INK,2.2)
    text(d,(322,523),'年份 / YEAR',24,'sans',MUTED,'rm')
    for t,cat in zip(times,categories):
        y=float(yp(t))
        line(d,[(x0-13,y),(x0,y)],INK,3)
        text(d,(322,y),cat,36,'number',INK,'rm')
    for tick in range(int(vmin),int(vmax)+1):
        x=float(xp(tick))
        line(d,[(x,yb),(x,yb+13)],INK,2.4)
        text(d,(x,yb+35),str(tick),34,'number',INK,'mt')

    # Point markers and exact supplied decimal strings, anchored to the contour.
    geometries=[]
    for t,v,cat in zip(times,vals,categories):
        x,y=float(xp(v)),float(yp(t))
        if t==times[0]:
            line(d,[(x+5,y-6),(x+29,y-34),(x+49,y-34)],RUST,1.8)
            lx,ly=x+59,y-35
        else:
            line(d,[(x+12,y),(x+36,y)],RUST,1.8)
            lx,ly=x+47,y
        # Paper-white halo keeps the markers unambiguous over dark outline.
        ellipse(d,(x,y),9.5,PAPER,GREEN,2)
        ellipse(d,(x,y),4.3,GREEN)
        text(d,(lx,ly),str(data['series'][0]['values'][categories.index(cat)]),35,'number_bold',RUST,'lm')
        geometries.append({'category':cat,'value':float(v),'x':x,'y':y,'label_x':lx,'label_y':ly})

    # A brief reading key uses only the new data's mapping, not a trend claim.
    text(d,(1144,524),'横向宽度 = 当年手术数',25,'sans',MUTED,'mm')
    text(d,((x0+xb)/2,1625),'手术数（百万例 / 所示年度）',32,'sans',INK,'mm')
    text(d,((x0+xb)/2,1672),data['series'][0]['unit'],25,'italic',MUTED,'mm')

    line(d,[(146,1752),(1450,1752)],(128,147,144),1.4)
    ellipse(d,(159,1800),6,GREEN)
    text(d,(181,1800),'5 个观测节点；曲线仅作插值。',26,'sans',INK,'lm')
    text(d,(181,1847),'从零线读到右侧轮廓；总面积不编码，眼睛与鼻孔不计入数据。',25,'sans',MUTED,'lm')
    text(d,(1450,1890),'数据：本案合成输入 · 非真实统计',21,'sans',MUTED,'rm')

    out=im.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
    out.save(ROOT/'final.png',compress_level=7)
    (ROOT/'artifacts').mkdir(exist_ok=True)
    fullmask.resize((W,H),Image.Resampling.LANCZOS).save(ROOT/'artifacts/data_mask.png')
    geometry={'canvas':[W,H],'x0':x0,'x4':xb,'y_min_year':yb,'y_max_year':yt,
              'pixels_per_million':(xb-x0)/(vmax-vmin),'pixels_per_year':(yb-yt)/(times[-1]-times[0]),
              'interpolation':'monotone piecewise cubic Hermite (PCHIP); five observations only',
              'source_y_anchors':sy_anchors.tolist(),'destination_y_anchors':ty_anchors.tolist(),
              'eye_rect':[410,620,eye_w,eye_h],'nodes':geometries}
    (ROOT/'artifacts/render_geometry.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':
    main()
