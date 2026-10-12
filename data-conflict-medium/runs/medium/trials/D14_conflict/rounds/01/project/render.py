#!/usr/bin/env python3
"""Rebuild the infographic from local frozen data and generated artwork."""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import map_coordinates, gaussian_filter

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / 'data.json').read_text(encoding='utf-8'))
BRIEF = json.loads((ROOT / 'brief.json').read_text(encoding='utf-8'))
W, H = BRIEF['width'], BRIEF['height']
S = 2
INK = '#44392f'
TEAL = '#17585e'
RED = '#b33932'
CREAM = '#f7f0e1'
PAPER = (247, 242, 229)
BASELINE = 1660.0
TOP = 370.0
AXIS_X = 196.0
X_FIRST, X_LAST = 270.0, 1782.0
MINIMUM = float(DATA['value_axis']['minimum'])
MAXIMUM = float(DATA['value_axis']['maximum'])
VALUES = np.asarray(DATA['series'][0]['values'], dtype=float)
TIMES = np.asarray(DATA['time_points'], dtype=float)
XS = X_FIRST + (TIMES-TIMES[0])/(TIMES[-1]-TIMES[0])*(X_LAST-X_FIRST)
YS = BASELINE - (VALUES-MINIMUM)/(MAXIMUM-MINIMUM)*(BASELINE-TOP)
POINTS = np.column_stack((XS, YS))
FONT_DIR = ROOT / 'assets/fonts'

def font(size, name='sans'):
    filename = {'sans':'sans.ttf', 'bold':'sans_bold.ttf', 'title':'title.ttc', 'italic':'serif_italic.ttf'}[name]
    return ImageFont.truetype(str(FONT_DIR / filename), round(size*S))

def line(draw, points, color, width=1):
    draw.line([(round(x*S),round(y*S)) for x,y in points], fill=color, width=round(width*S), joint='curve')

def text(draw, xy, value, size=30, face='sans', fill=INK, anchor='lt'):
    draw.text((round(xy[0]*S),round(xy[1]*S)), str(value), font=font(size, face), fill=fill, anchor=anchor)

def circle(draw, xy, r, fill, outline=None, width=1):
    x,y=xy
    draw.ellipse(((x-r)*S,(y-r)*S,(x+r)*S,(y+r)*S),fill=fill,outline=outline,width=round(width*S))

def extrap_interp(x, xp, fp):
    y=np.interp(x,xp,fp)
    y=np.where(x<xp[0],fp[0]+(x-xp[0])*(fp[1]-fp[0])/(xp[1]-xp[0]),y)
    return np.where(x>xp[-1],fp[-1]+(x-xp[-1])*(fp[-1]-fp[-2])/(xp[-1]-xp[-2]),y)

def warp_diver():
    """One continuous coordinate map; centerline maps exactly to the data polyline."""
    settings=json.loads((ROOT/'assets/diver_controls.json').read_text(encoding='utf-8'))
    source=np.asarray(settings['source_centerline'],dtype=float)
    raw=np.asarray(Image.open(ROOT/'assets/raw_01.png').convert('RGBA')).astype(np.float32)
    # Remove faint transparent-background residue, without changing the raw file.
    raw[:,:,3]=np.where(raw[:,:,3]<24,0,raw[:,:,3])
    h,w=raw.shape[:2]
    sx,sy=source.T
    source_slopes=np.gradient(sy,sx)
    target_slopes=np.gradient(YS,XS)
    thickness=settings['transverse_scale']*np.sqrt(1+target_slopes**2)/np.sqrt(1+source_slopes**2)
    dest_x=np.arange(W,dtype=float)
    source_x=np.interp(dest_x,np.r_[X_FIRST-70,XS,X_LAST+55],np.r_[0,sx,w-1])
    source_y_center=extrap_interp(source_x,sx,sy)
    target_y_center=extrap_interp(dest_x,XS,YS)
    scale=np.interp(dest_x,XS,thickness)
    source_y=source_y_center[None,:]+(np.arange(H)[:,None]-target_y_center[None,:])/scale[None,:]
    source_x_full=np.broadcast_to(source_x[None,:],(H,W))
    valid_x=(dest_x>=X_FIRST-70)&(dest_x<=X_LAST+55)
    # Premultiply before resampling so transparent edges keep their ink color.
    alpha=raw[:,:,3]/255
    channels=[raw[:,:,c]*alpha for c in range(3)]+[raw[:,:,3]]
    warped=np.stack([map_coordinates(c,[source_y,source_x_full],order=1,mode='constant',cval=0) for c in channels],axis=2)
    warped[:,:,3]*=valid_x[None,:]
    out_alpha=warped[:,:,3]/255
    warped[:,:,:3]=np.divide(warped[:,:,:3],out_alpha[:,:,None],out=np.zeros_like(warped[:,:,:3]),where=out_alpha[:,:,None]>0)
    im=Image.fromarray(np.uint8(np.clip(warped,0,255)),'RGBA')
    im.save(ROOT/'assets/diver_warped.png')
    return im

def main():
    (ROOT/'artifacts').mkdir(exist_ok=True)
    rng=np.random.default_rng(614)
    # Warm, lightly grained graph paper, created locally; no reference pixels.
    fine=rng.normal(0,.64,(H,W))
    coarse=gaussian_filter(rng.normal(0,1,(H//4,W//4)),sigma=9)
    coarse=np.asarray(Image.fromarray(np.float32(coarse),'F').resize((W,H),Image.Resampling.BILINEAR))
    paper=np.stack([np.clip(c+fine+coarse*5,0,255) for c in PAPER],axis=2).astype('uint8')
    im=Image.fromarray(paper,'RGB').resize((W*S,H*S),Image.Resampling.BICUBIC).convert('RGBA')
    d=ImageDraw.Draw(im)
    # Typography intentionally lives outside the illustrative data shape.
    circle(d,(108,78),5,RED)
    text(d,(128,63),DATA['visible_notice'],25,'bold',RED)
    text(d,(1886,63),'2000 — 2018',26,'sans',INK,'rt')
    text(d,(102,126),DATA['title'],78,'title')
    text(d,(106,244),DATA['series'][0]['label'],42,'italic')
    text(d,(1884,252),DATA['series'][0]['unit'],30,'sans',INK,'rt')
    line(d,[(106,316),(1894,316)],'#b9ae99',1.2)

    grid=Image.new('RGBA',im.size,(0,0,0,0));gd=ImageDraw.Draw(grid)
    for x in np.arange(78,1924,42):
        line(gd,[(x,348),(x,1822)],(109,113,120,43),.8)
    for y in np.arange(358,1822,42):
        line(gd,[(78,y),(1924,y)],(109,113,120,43),.8)
    for value in range(0,35001,5000):
        y=BASELINE-value/(MAXIMUM-MINIMUM)*(BASELINE-TOP)
        line(gd,[(AXIS_X,y),(1898,y)],(99,102,104,60),1.1)
    im.alpha_composite(grid)
    d=ImageDraw.Draw(im)
    # Major axes and year guides share the one numerical coordinate map.
    for value in range(0,35001,5000):
        y=BASELINE-value/(MAXIMUM-MINIMUM)*(BASELINE-TOP)
        text(d,(175,y),f'{value:,}',27,'sans',INK,'rm')
        line(d,[(AXIS_X-7,y),(AXIS_X,y)],INK,1.4)
    line(d,[(AXIS_X,BASELINE),(AXIS_X,TOP-10)],'#62594d',1.8)
    line(d,[(AXIS_X-7,TOP+2),(AXIS_X,TOP-10),(AXIS_X+7,TOP+2)],'#62594d',1.8)
    guide=Image.new('RGBA',im.size);gg=ImageDraw.Draw(guide)
    for x,y in POINTS:
        for yy in np.arange(y+13,BASELINE-4,17):
            line(gg,[(x,yy),(x,min(yy+6,BASELINE-4))],(35,96,101,64),1.2)
    im.alpha_composite(guide)

    # Water starts precisely on the common zero line; texture stays below it.
    wx0,wx1=78,1924
    wy0,wy1=round(BASELINE),1822
    yy,xx=np.mgrid[0:wy1-wy0,0:wx1-wx0]
    wash=(np.sin(xx/72+yy/22)*2.5+np.sin(xx/170-yy/15)*2+rng.normal(0,1.6,xx.shape))
    base=np.array([74,162,172])[None,None,:]
    grad=yy[:,:,None]/(wy1-wy0)*np.array([-29,-33,-20])[None,None,:]
    water=np.uint8(np.clip(base+grad+wash[:,:,None],0,255))
    water_im=Image.fromarray(water,'RGB').resize(((wx1-wx0)*S,(wy1-wy0)*S),Image.Resampling.BICUBIC)
    im.paste(water_im,(wx0*S,wy0*S))
    waterlines=Image.new('RGBA',im.size);wd=ImageDraw.Draw(waterlines)
    for y in [1678,1745,1787,1810]:
        pts=[(x,y+1.8*math.sin(x/40+y/17)) for x in range(wx0,wx1,3)]
        line(wd,pts,(214,236,222,45),1.4)
    for k in range(20):
        x=111+k*89+float(rng.uniform(-15,15));y=float(rng.choice([1753,1796]))
        line(wd,[(x-12,y+5),(x,y-3),(x+13,y+2)],(15,96,112,72),1.2)
    im.alpha_composite(waterlines)
    d=ImageDraw.Draw(im)
    line(d,[(78,BASELINE),(1924,BASELINE)],TEAL,2.5)
    for cat,x in zip(DATA['categories'],XS):
        line(d,[(x,BASELINE),(x,BASELINE+13)],TEAL,2)
        text(d,(x,1697),cat,32,'bold','#123f47','mt')
    text(d,(1898,1781),'水面 = 0',24,'title','#e4f0df','rt')

    diver=warp_diver()
    im.alpha_composite(diver.resize((W*S,H*S),Image.Resampling.LANCZOS))
    d=ImageDraw.Draw(im)
    # Exactly straight segments. Neither the body outline nor limb thickness is a value channel.
    pts=[tuple(p) for p in POINTS]
    line(d,pts,CREAM,11)
    line(d,pts,TEAL,5)
    for p in pts:
        circle(d,p,12.5,CREAM,TEAL,2)
        circle(d,p,6,RED)

    # Direct labels, kept outside the figure with unambiguous short leaders.
    offsets=[(20,-105),(84,-75),(95,-69),(94,-68),(0,145),(155,76),(130,76),(95,91),(36,126),(0,-94)]
    label_boxes=[]
    for i,((x,y),(dx,dy),v) in enumerate(zip(pts,offsets,VALUES)):
        tx,ty=x+dx,y+dy
        value=f'{int(v):,}'
        f=font(37,'bold')
        bb=d.textbbox((tx*S,ty*S),value,font=f,anchor='mm')
        # Leaders terminate just outside the text, with a paper-colored text bed.
        x0,y0,x1,y1=np.asarray(bb,dtype=float)/S
        pad=7
        endpoint=(tx, y1+6 if dy<0 else y0-6)
        vec=np.array(endpoint)-np.array([x,y]);ln=np.linalg.norm(vec)
        start=np.array([x,y])+vec/ln*17
        line(d,[tuple(start),endpoint],'#766c5d',1.3)
        d.rounded_rectangle(((x0-pad)*S,(y0-4)*S,(x1+pad)*S,(y1+4)*S),radius=4*S,fill=CREAM)
        text(d,(tx,ty),value,37,'bold',TEAL,'mm')
        label_boxes.append({'category':DATA['categories'][i],'value':int(v),'bbox':[x0-pad,y0-4,x1+pad,y1+4],'center':[tx,ty]})

    # A compact key in the open part of the pike makes the visual encoding explicit.
    line(d,[(774,469),(841,469)],CREAM,11)
    line(d,[(774,469),(841,469)],TEAL,5)
    circle(d,(807,469),10,CREAM,TEAL,1.8)
    circle(d,(807,469),4.8,RED)
    text(d,(861,449),'高度读取金额',26,'title',TEAL)
    text(d,(774,495),'逐期直线连接 · 线性量尺',23,'title','#72685b')

    text(d,(105,1871),'读法：沿脚 → 髋 → 手的折线读取数值；高度以水面 0 为基准。',28,'title',INK)
    text(d,(105,1922),'合成数据 · 0–35,000 USD per person · 身体外缘与厚度不编码数值',24,'title','#776e61')
    result=im.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
    result.save(ROOT/'final.png',compress_level=7)
    geom={'canvas':[W,H],'axis':{'minimum':MINIMUM,'maximum':MAXIMUM,'scale':'linear','top_px':TOP,'zero_y_px':BASELINE,'x_first_px':X_FIRST,'x_last_px':X_LAST},
          'points':[{'category':c,'value':int(v),'x':float(x),'y':float(y)} for c,v,(x,y) in zip(DATA['categories'],VALUES,POINTS)],'label_boxes':label_boxes,
          'generated_asset':'assets/raw_01.png','transformed_asset':'assets/diver_warped.png'}
    (ROOT/'artifacts/render_geometry.json').write_text(json.dumps(geom,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Rendered final.png:',W,H,'; data nodes:',len(POINTS))

if __name__=='__main__':
    main()
