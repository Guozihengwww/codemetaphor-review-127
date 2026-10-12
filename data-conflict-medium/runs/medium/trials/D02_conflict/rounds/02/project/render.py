#!/usr/bin/env python3
"""Regenerate final.png from frozen data and local generated wood materials.
Run complete candidates only through capture_render.py. No network/model calls.
"""
from pathlib import Path
from decimal import Decimal
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT/'data.json').read_text(encoding='utf-8'), parse_float=Decimal)
BRIEF = json.loads((ROOT/'brief.json').read_text(encoding='utf-8'))
W,H = BRIEF['width'],BRIEF['height']
S = 2
CX,CY,R,Y_SCALE = 842.5,1050.0,535.0,0.96
START_DEG = -90.0
values_dec = DATA['series'][0]['values']
values = np.array([float(v) for v in values_dec])
assert sum(values_dec)==100
names = DATA['categories']
assert len(names)==len(values)
edges = np.r_[0.0,np.cumsum(values)]*3.6
mids = np.deg2rad(START_DEG+(edges[:-1]+edges[1:])/2)

PALETTE = ['#477F73','#CAB578','#719DA9','#A9B87C','#D0A455',
 '#C87F5C','#7E8AA6','#919D63','#DDA57B','#56958F','#C8BD82',
 '#A08BAC','#769FB5','#768F64','#C58776','#CAA55E','#5F919D',
 '#A8B781','#B97954','#A28D9F','#598274','#D0A677','#80A49F',
 '#A6AD65','#6B879B','#BF8470','#C7A575']
palette=np.array([[int(h[k:k+2],16) for k in (1,3,5)] for h in PALETTE],dtype=np.float32)
INK=(29,52,45)
MUTED=(71,92,82)
LEADER=(64,85,73)
BG=(220,234,225)

def font(size,style='regular'):
    paths={'regular':'sans.ttf','bold':'sans_bold.ttf','title':'serif_cjk.ttc','cjk':'sans_cjk.ttc'}
    return ImageFont.truetype(str(ROOT/'assets/fonts'/paths[style]),round(size*S),index=2 if style in ('title','cjk') else 0)

def xy(p): return tuple(round(x*S) for x in p)

def point(radius,angle):
    return (CX+radius*math.cos(angle),CY+Y_SCALE*radius*math.sin(angle))

def main():
    rng=np.random.default_rng(3201)
    grain=rng.normal(0,0.48,(H*S,W*S,1)).astype(np.float32)
    bg=np.clip(np.array(BG,dtype=np.float32)+grain,0,255).astype('uint8')
    im=Image.fromarray(bg,'RGB').convert('RGBA')
    del bg,grain

    # A common material image is resampled once. Its exterior bark is never a sector.
    material=Image.open(ROOT/'assets/wood_normalized.png').convert('RGBA')
    half=670*R/600
    bounds=(round((CX-half)*S),round((CY-half*Y_SCALE)*S))
    outsize=(round(2*half*S),round(2*half*Y_SCALE*S))
    material=material.resize(outsize,Image.Resampling.LANCZOS)
    texture=np.array(material,dtype=np.float32)
    hh,ww=texture.shape[:2]
    py=(np.arange(hh,dtype=np.float32)+bounds[1]+0.5)/S
    px=(np.arange(ww,dtype=np.float32)+bounds[0]+0.5)/S
    ux=(px[None,:]-CX)
    uy=(py[:,None]-CY)/Y_SCALE
    radial=np.sqrt(ux*ux+uy*uy)
    phase=np.mod(np.degrees(np.arctan2(uy,ux))-START_DEG,360.0)
    ids=np.minimum(np.searchsorted(edges[1:],phase,side='right'),len(values)-1)
    domain=radial<=R
    luminance=texture[:,:,:3]@np.array([0.2126,0.7152,0.0722],dtype=np.float32)
    shade=np.power(np.maximum(luminance,1)/(0.69*255),1.12)
    chroma=texture[:,:,:3]-luminance[:,:,None]
    tint=np.clip(palette[ids]*shade[:,:,None]+0.06*chroma,0,255)
    texture[:,:,:3][domain]=tint[domain]
    texture[:,:,3][domain]=255
    mat=Image.fromarray(np.round(texture).astype('uint8'),'RGBA')

    shadow_layer=Image.new('L',im.size,0)
    shadow_layer.paste(mat.getchannel('A'),(bounds[0]+12*S,bounds[1]+20*S))
    shadow_layer=shadow_layer.filter(ImageFilter.GaussianBlur(23*S))
    shadow_layer=shadow_layer.point(lambda p: round(p*0.25))
    sh=Image.new('RGBA',im.size,(25,49,41,0));sh.putalpha(shadow_layer)
    im=Image.alpha_composite(im,sh)
    im.alpha_composite(mat,bounds)
    draw=ImageDraw.Draw(im)

    # Very fine ink seams clarify the common boundaries; no sector is exploded.
    seam=Image.new('RGBA',im.size,(0,0,0,0))
    sd=ImageDraw.Draw(seam)
    for edge in edges[:-1]:
        p=point(R,math.radians(START_DEG+float(edge)))
        sd.line([xy((CX,CY)),xy(p)],fill=(34,49,39,95),width=1)
    im=Image.alpha_composite(im,seam)
    draw=ImageDraw.Draw(im)

    # Publication header.
    draw.rounded_rectangle(xy((90,88))+xy((350,132)),radius=8*S,fill=INK)
    draw.text(xy((109,96)),DATA['visible_notice'],font=font(21,'bold'),fill=(241,246,235))
    draw.text(xy((90,164)),DATA['title'],font=font(82,'title'),fill=INK)
    draw.text(xy((94,282)),DATA['series'][0]['label'],font=font(32),fill=INK)
    draw.text(xy((95,338)),'Share (%)  ·  Clockwise in input order',font=font(24),fill=MUTED)
    draw.line([xy((90,393)),xy((1595,393))],fill=(123,150,133),width=2)
    draw.text(xy((1593,179)),str(len(names)),font=font(67,'title'),fill=INK,anchor='ra')
    draw.text(xy((1592,280)),'GROUPS',font=font(21,'bold'),fill=MUTED,anchor='ra')
    draw.text(xy((1592,338)),'TOTAL 100%',font=font(24,'bold'),fill=MUTED,anchor='ra')

    # Solve ordered label rows independently on each side; this only moves text,
    # never sector angles, radii, source material coordinates, or data areas.
    label_y={}
    for side in (1,-1):
        sel=np.where(np.cos(mids)*side>0)[0]
        sel=sel[np.argsort(np.sin(mids[sel]))]
        target=CY+Y_SCALE*585*np.sin(mids[sel])
        sol=minimize(lambda ys:np.sum((ys-target)**2),np.linspace(460,1730,len(sel)),
            constraints={'type':'ineq','fun':lambda ys:np.diff(ys)-79},
            bounds=[(460,1730)]*len(sel),method='SLSQP',options={'ftol':1e-7,'maxiter':600})
        for i,y in zip(sel,sol.x):label_y[int(i)]=float(y)

    records=[]
    label_boxes=[]
    for i,(name,val,ang) in enumerate(zip(names,values_dec,mids)):
        right=math.cos(float(ang))>0
        ly=label_y[i]
        p0=point(R*0.985,float(ang))
        p1=point(585,float(ang))
        elbow=(1455 if right else 230,ly)
        end=(1460 if right else 222,ly)
        pts=[p0,p1,elbow,end]
        # Short light under-stroke keeps the material-side segment traceable.
        draw.line([xy(p0),xy(p1)],fill=(235,239,218),width=5)
        draw.line([xy(p) for p in pts],fill=LEADER,width=3,joint='curve')
        dot=3.1
        draw.ellipse(xy((p0[0]-dot,p0[1]-dot))+xy((p0[0]+dot,p0[1]+dot)),fill=INK)
        tx=1478 if right else 207
        anchor='la' if right else 'ra'
        name_pos=xy((tx,ly-30))
        value_pos=xy((tx,ly+6))
        exact=str(val)+'%'
        draw.text(name_pos,name,font=font(27),fill=INK,anchor=anchor)
        draw.text(value_pos,exact,font=font(29,'bold'),fill=INK,anchor=anchor)
        for pos,textv,f in [(name_pos,name,font(27)),(value_pos,exact,font(29,'bold'))]:
            b=draw.textbbox(pos,textv,font=f,anchor=anchor)
            label_boxes.append({'category':name,'text':textv,'bbox_px':[n/S for n in b]})
        records.append({'category':name,'value_exact':str(val),'label_exact':exact,
          'start_deg_unprojected':START_DEG+float(edges[i]),
          'end_deg_unprojected':START_DEG+float(edges[i+1]),
          'mid_deg_unprojected':float(np.degrees(ang)),
          'palette_hex':PALETTE[i],'label_y':ly,'leader_points_px':[list(p) for p in pts]})

    # Small, explicit encoding note; the bark is contextual material only.
    draw.line([xy((90,1823)),xy((1595,1823))],fill=(123,150,133),width=2)
    draw.text(xy((93,1860)),'面积代表份额 · 木面总计 100% · 树皮不计入数据域',font=font(26,'cjk'),fill=INK)
    draw.text(xy((94,1912)),'Synthetic values  /  27 categories  /  unit: percent',font=font(21),fill=MUTED)
    im=im.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
    im.save(ROOT/'final.png',compress_level=6)

    # Pixel-domain checks use the same visible surface mapping at render resolution.
    counts=np.bincount(ids[domain].ravel(),minlength=len(values))
    area_pct=counts/counts.sum()*100
    geometry={
       'canvas_px':[W,H],'supersampling':S,'data_center_px':[CX,CY],
       'unprojected_radius_px':R,'projection_y_scale':Y_SCALE,
       'data_domain_area_px':float(counts.sum()/S**2),
       'total_value_exact':str(sum(values_dec)),
       'sector_count':len(values),'data_order':names,
       'render_mask_area_percent':[float(x) for x in area_pct],
       'maximum_mask_area_error_percentage_points':float(np.max(abs(area_pct-values))),
       'smallest_unprojected_outer_arc_px':float(R*math.radians(float(min(values))*3.6)),
       'labels':label_boxes,'sectors':records,
       'wood_asset':'assets/wood_normalized.png','raw_asset':'assets/raw_01.png',
       'seam_width_render_px':1,'seam_width_final_px':0.5,
       'all_geometry_bound_to_data_json':True
    }
    (ROOT/'artifacts').mkdir(exist_ok=True)
    (ROOT/'artifacts/render_geometry.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Rendered final.png',W,H,'sectors',len(values),'max mask-area error pp',geometry['maximum_mask_area_error_percentage_points'])

if __name__=='__main__':main()
