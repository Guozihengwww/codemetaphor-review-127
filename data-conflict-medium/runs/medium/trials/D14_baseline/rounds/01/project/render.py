#!/usr/bin/env python3
"""Reproduce the infographic locally. Run through capture_render.py in this study."""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import map_coordinates, gaussian_filter

ROOT = Path(__file__).resolve().parent
SCALE = 2
W = H = 2000
PAPER = (246, 240, 225)
INK = '#343c3f'
MUTED = '#58696c'
PATH_COLOR = '#26475b'
POINT_COLOR = '#a74832'
FONT_DIR = ROOT / 'assets/fonts'
TEXT_INVENTORY = []


def font(name, size):
    return ImageFont.truetype(str(FONT_DIR / name), round(size*SCALE))


def pt(x, y):
    return (round(float(x)*SCALE), round(float(y)*SCALE))


def line(draw, points, fill, width=1):
    draw.line([pt(x,y) for x,y in points], fill=fill, width=max(1,round(width*SCALE)), joint='curve')


def text(draw, xy, string, size=30, color=INK, face='Sans-Regular.ttf', anchor='lt', stroke=0):
    f = font(face,size)
    xy2=pt(*xy)
    draw.text(xy2,string,font=f,fill=color,anchor=anchor,
              stroke_width=round(stroke*SCALE),stroke_fill=PAPER)
    b=draw.textbbox(xy2,string,font=f,anchor=anchor,stroke_width=round(stroke*SCALE))
    TEXT_INVENTORY.append({'text':string,'bbox':[round(v/SCALE,2) for v in b]})


def ellipse(draw, x, y, radius, fill, outline=None, width=1):
    draw.ellipse([pt(x-radius,y-radius),pt(x+radius,y+radius)],fill=fill,
                 outline=outline,width=max(1,round(width*SCALE)))


def dotted(draw, p0, p1, fill, width=1, dash=5, gap=8):
    a=np.asarray(p0,dtype=float);b=np.asarray(p1,dtype=float)
    length=np.linalg.norm(b-a)
    if length==0:return
    for off in np.arange(0,length,dash+gap):
        u=a+(b-a)*off/length;v=a+(b-a)*min(off+dash,length)/length
        line(draw,[u,v],fill,width)


def paper_background():
    rng=np.random.default_rng(471)
    grain=rng.normal(0,0.95,(H,W))
    soft=gaussian_filter(rng.normal(0,1,(H,W)),6)*6
    yy,xx=np.mgrid[:H,:W]
    vignette=1.2*((xx-W/2)**2+(yy-H/2)**2)/(W/2)**2
    a=np.clip(np.array(PAPER)[None,None,:]+grain[:,:,None]+soft[:,:,None]-vignette[:,:,None],0,255)
    return Image.fromarray(a.astype(np.uint8)).resize((W*SCALE,H*SCALE),Image.Resampling.BICUBIC).convert('RGBA')


def diver_layer(target):
    """One monotone x map plus a vertical centerline warp, shared by all anatomy."""
    source=np.asarray([[126,865],[320,560],[469,420],[620,270],[766,168],
                       [945,260],[1120,413],[1273,620],[1383,792],[1470,923]],float)
    raw=np.asarray(Image.open(ROOT/'assets/diver_cutout.png').convert('RGBA'),dtype=np.float32)
    sx_map=PchipInterpolator(target[:,0],source[:,0],extrapolate=True)
    src_center=PchipInterpolator(source[:,0],source[:,1],extrapolate=True)
    dst_center=PchipInterpolator(target[:,0],target[:,1],extrapolate=True)
    # Vertical thickness is anatomical, independent of the numerical distance to zero.
    thickness_scale=1.12
    box=(205,430,1950,1585)
    left,top,right,bottom=box
    xs=np.arange(left*SCALE,right*SCALE,dtype=np.float32)/SCALE
    sx=sx_map(xs)
    sc=src_center(sx)
    dc=dst_center(xs)
    sh=(round((bottom-top)*SCALE),len(xs))
    out=np.zeros((sh[0],sh[1],4),dtype=np.uint8)
    # Premultiplied interpolation avoids dark outside-alpha fringes.
    alpha=raw[:,:,3]/255
    raw[:,:,:3]*=alpha[:,:,None]
    for row0 in range(0,sh[0],180):
        row1=min(row0+180,sh[0])
        ys=top+np.arange(row0,row1,dtype=np.float32)/SCALE
        sy=(ys[:,None]-dc[None,:])/thickness_scale+sc[None,:]
        xx=np.broadcast_to(sx[None,:],sy.shape)
        chunk=np.stack([map_coordinates(raw[:,:,ch],[sy,xx],order=1,mode='constant',cval=0,prefilter=False)
                        for ch in range(4)],axis=2)
        ca=chunk[:,:,3:4]/255
        chunk[:,:,:3]=np.divide(chunk[:,:,:3],ca,out=np.zeros_like(chunk[:,:,:3]),where=ca>0.001)
        out[row0:row1]=np.clip(chunk,0,255).astype(np.uint8)
    layer=Image.new('RGBA',(W*SCALE,H*SCALE),(0,0,0,0))
    layer.alpha_composite(Image.fromarray(out),(left*SCALE,top*SCALE))
    small=layer.resize((W,H),Image.Resampling.LANCZOS)
    small.save(ROOT/'assets/diver_warped.png')
    small.getchannel('A').save(ROOT/'artifacts/body_mask.png')
    return layer,source,thickness_scale


def water_layer(yzero, xleft, xright):
    height=165
    rng=np.random.default_rng(762)
    yy,xx=np.mgrid[:height,:W]
    t=yy/(height-1)
    col0=np.array([166,207,200]);col1=np.array([76,152,162])
    tex=gaussian_filter(rng.normal(0,3,(height,W)),0.7)+1.5*np.sin(xx/13+yy/50)
    c=col0[None,None,:]*(1-t[:,:,None])+col1[None,None,:]*t[:,:,None]+tex[:,:,None]
    a=np.zeros((height,W,4),np.uint8)
    a[:,:,:3]=np.clip(c,0,255).astype(np.uint8);a[:,int(xleft):int(xright),3]=255
    band=Image.fromarray(a).resize((W*SCALE,height*SCALE),Image.Resampling.BICUBIC)
    layer=Image.new('RGBA',(W*SCALE,H*SCALE),(0,0,0,0))
    layer.alpha_composite(band,(0,int(yzero*SCALE)))
    d=ImageDraw.Draw(layer)
    # Fine dry-pencil ripples stay below the exact horizontal zero line.
    for i in range(57):
        x=float(rng.uniform(xleft+10,xright-55));y=float(rng.uniform(yzero+82,yzero+155))
        length=float(rng.uniform(10,58))
        points=[(x+z,y+1.5*math.sin(z/15+i)) for z in np.linspace(0,length,14)]
        line(d,points,(28,108,127,70),1.2)
    return layer


def main():
    data=json.loads((ROOT/'data.json').read_text(encoding='utf-8'))
    brief=json.loads((ROOT/'brief.json').read_text(encoding='utf-8'))
    assert (brief['width'],brief['height'])==(W,H)
    values=np.asarray(data['series'][0]['values'],float)
    years=np.asarray(data['time_points'],float)
    assert len(values)==len(years)==len(data['categories'])==10
    amin=float(data['value_axis']['minimum']);amax=float(data['value_axis']['maximum'])
    assert amin==0 and data['value_axis']['scale']=='linear'
    xfirst,xlast=300.,1875.
    yzero,ytop=1690.,465.
    xx=xfirst+(years-years[0])/(years[-1]-years[0])*(xlast-xfirst)
    yy=yzero-(values-amin)/(amax-amin)*(yzero-ytop)
    points=np.column_stack([xx,yy])
    canvas=paper_background();draw=ImageDraw.Draw(canvas)

    # Header and restrained editorial typography.
    text(draw,(135,76),data['visible_notice'],27,POINT_COLOR,'Sans-Bold.ttf')
    line(draw,[(135,118),(415,118)],POINT_COLOR,2)
    text(draw,(129,149),data['title'],86,INK,'Title-CJK.otf')
    text(draw,(135,274),'GDP per capita',48,INK,'Serif-Italic.ttf')
    text(draw,(1818,289),'2000–2018',30,MUTED,anchor='rt')

    # Graph paper: one common grid for years and values. Each 43.75px cell is
    # half a year horizontally and 1,250 USD per person vertically.
    grid_left,grid_right=230.,1920.
    grid_top,grid_bottom=421.,1855.
    for gx in np.arange(xfirst-2*43.75,grid_right+1,43.75):
        line(draw,[(gx,grid_top),(gx,grid_bottom)],'#dfe0d2',0.7)
    for gy in np.arange(yzero-29*43.75,grid_bottom+1,43.75):
        if gy>=grid_top:
            line(draw,[(grid_left,gy),(grid_right,gy)],'#dfe0d2',0.7)
    for val in np.arange(0,amax+1,5000):
        py=yzero-(val-amin)/(amax-amin)*(yzero-ytop)
        line(draw,[(grid_left,py),(grid_right,py)],'#ccd6cc',1.0)
    for x in xx:
        line(draw,[(x,grid_top),(x,grid_bottom)],'#d3dace',1.0)

    # Soft projections make the chart scale readable without competing with the body.
    for x,y in points:
        dotted(draw,(x,y+14),(x,yzero-12),'#aebcae',1.1,4,8)

    body,source,thickness_scale=diver_layer(points)
    canvas.alpha_composite(body)
    canvas.alpha_composite(water_layer(yzero,grid_left,grid_right))
    draw=ImageDraw.Draw(canvas)

    # Axes and exact common zero, which is also the water's surface.
    axisx=230
    line(draw,[(axisx,443),(axisx,yzero)],MUTED,2.0)
    line(draw,[(axisx-6,452),(axisx,442),(axisx+6,452)],MUTED,1.8)
    for val in np.arange(0,amax+1,5000):
        py=yzero-(val-amin)/(amax-amin)*(yzero-ytop)
        line(draw,[(axisx-9,py),(axisx+1,py)],MUTED,1.8)
        text(draw,(axisx-23,py),f'{int(val):,}',27,MUTED,anchor='rm',stroke=1)
    text(draw,(230,378),data['series'][0]['unit'],29,MUTED)
    line(draw,[(grid_left,yzero),(grid_right,yzero)],'#296e79',3.5)
    for x,cat in zip(xx,data['categories']):
        line(draw,[(x,yzero-10),(x,yzero+10)],'#2b6870',2)
        text(draw,(x,yzero+42),cat,32,'#254d57','Sans-Bold.ttf',anchor='mm')
    text(draw,(1915,1814),'年份 →',25,'#234f5b','Note-CJK.otf',anchor='rt')

    # Straight segments are the sole quantitative channel. Anatomy is subordinate.
    line(draw,points,'#fff5db',13)
    line(draw,points,PATH_COLOR,5)
    for x,y in points:
        ellipse(draw,x,y,12,'#fff6df',PATH_COLOR,2.5)
        ellipse(draw,x,y,6,POINT_COLOR)

    # Values remain bound to individual nodes through short non-data leaders.
    label_offsets=[(77,112),(88,121),(55,129),(3,176),(0,-144),
                   (-30,178),(-101,177),(-106,151),(-108,138),(-28,97)]
    leaders=[]
    for i,((x,y),v,(dx,dy)) in enumerate(zip(points,values,label_offsets)):
        lx,ly=x+dx,y+dy
        radius=17
        endy=ly-29 if dy>0 else ly+30
        start=np.array([x,y],float)
        end=np.array([lx,endy],float)
        delta=end-start
        start=start+delta/np.linalg.norm(delta)*radius
        # Thin beige-brown guides have no nodes and no arrowheads.
        line(draw,[start,end],'#907e64',1.4)
        line(draw,[(lx-15,endy),(lx+15,endy)],'#907e64',1.2)
        text(draw,(lx,ly),f'{int(v):,}',36,INK,'Sans-Bold.ttf',anchor='mm',stroke=4)
        leaders.append({'category':data['categories'][i],'value':int(v),'label_center':[lx,ly]})

    # Small integrated key in open graph space.
    keyx,keyy=835,1378
    line(draw,[(keyx,keyy),(keyx+73,keyy)],'#fff5db',10)
    line(draw,[(keyx,keyy),(keyx+73,keyy)],PATH_COLOR,4.5)
    ellipse(draw,keyx+36,keyy,8,'#fff6df',PATH_COLOR,2)
    ellipse(draw,keyx+36,keyy,4,POINT_COLOR)
    text(draw,(keyx+94,keyy),'GDP per capita',29,MUTED,anchor='lm')
    text(draw,(835,1420),'水面 = 0 USD per person',25,MUTED,'Note-CJK.otf')

    line(draw,[(135,1900),(1915,1900)],'#b3b6a9',1)
    text(draw,(135,1930),'读图：沿脚至手读取；节点到水面的垂直距离对应数值。人体厚度不编码数值。',25,MUTED,'Note-CJK.otf')

    final=canvas.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
    final.save(ROOT/'final.png')
    geometry={
        'canvas':[W,H],'data_source':'data.json','category_count':len(years),
        'value_domain':[amin,amax],'y_zero':yzero,'y_maximum':ytop,
        'x_first':xfirst,'x_last':xlast,'scale_px_per_unit':(yzero-ytop)/(amax-amin),
        'source_anchors':source.tolist(),'target_nodes':points.tolist(),
        'body_transform':'monotone PCHIP x map; PCHIP centerline vertical warp; straight data path',
        'anatomical_thickness_multiplier':thickness_scale,
        'labels':leaders,'text_inventory':TEXT_INVENTORY,
        'node_core_rgb':[167,72,50],
    }
    (ROOT/'artifacts/geometry.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Rendered final.png at 2000×2000 from one generated diver and local numerical data.')


if __name__=='__main__':
    main()
