"""Regenerate the food infographic from frozen data and local generated assets.
Run with no arguments. No network or model calls. All coordinates share one mapping.
"""
from pathlib import Path
import json
import math
import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'assets'
ART = ROOT / 'artifacts'
S = 2


def font(size, bold=False, cjk=False):
    name = ('NotoSansCJK-Bold.ttc' if bold else 'NotoSansCJK-Regular.ttc') if cjk else ('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')
    return ImageFont.truetype(str(ASSETS / 'fonts' / name), round(size*S))


def mask_polygon(points, size):
    m = Image.new('L', (size[0]*S, size[1]*S))
    ImageDraw.Draw(m).polygon([(round(x*S), round(y*S)) for x,y in points], fill=255)
    return m.resize(size, Image.Resampling.LANCZOS)


def shadow(canvas, alpha, offset=(3,7), blur=5, opacity=.28, color=(65,31,18)):
    a = alpha.filter(ImageFilter.GaussianBlur(blur)).point(lambda v: round(v*opacity))
    lay = Image.new('RGBA', canvas.size, (*color,0)); lay.putalpha(a)
    out = Image.new('RGBA', canvas.size); out.alpha_composite(lay, offset)
    return Image.alpha_composite(canvas, out)


def main():
    data = json.loads((ROOT/'data.json').read_text(encoding='utf-8'))
    brief = json.loads((ROOT/'brief.json').read_text(encoding='utf-8'))
    W,H = brief['width'], brief['height']
    size=(W,H)
    ART.mkdir(exist_ok=True)
    xmin,xmax=335.0,1585.0
    ymin,ymax=344.0,1058.0
    vmin,vmax=data['value_axis']['minimum'],data['value_axis']['maximum']
    times=np.array(data['time_points'],float)
    X=lambda t: xmin+(np.asarray(t)-times[0])/(times[-1]-times[0])*(xmax-xmin)
    Y=lambda v: ymax-(np.asarray(v)-vmin)/(vmax-vmin)*(ymax-ymin)
    xs=X(times)
    ser={s['id']:s for s in data['series']}
    av=np.array(ser['A']['values'],float); bv=np.array(ser['B']['values'],float)
    ya,yb=Y(av),Y(bv)
    ptsA=np.column_stack([xs,ya]); ptsB=np.column_stack([xs,yb])

    # Fine, non-directional tabletop grain; deterministic and data-independent.
    rng=np.random.default_rng(3190)
    yy,xx=np.mgrid[0:H,0:W]
    glow=np.exp(-(((xx-650)/1700)**2+((yy-280)/1600)**2))
    coarse=ndimage.gaussian_filter(rng.normal(size=(H,W)),12)
    coarse=coarse/(coarse.std()+1e-8)
    fine=rng.normal(size=(H,W))
    base=np.array([170.0, 43.0, 46.0])
    rgb=base[None,None,:]+glow[:,:,None]*np.array([33,24,20])+coarse[:,:,None]*1.1+fine[:,:,None]*1.25
    canvas=Image.fromarray(np.uint8(np.clip(rgb,0,255))).convert('RGBA')
    cream=(255,246,224,255); muted=(246,214,190,255)

    # Text and rules are rendered at 2x for consistent clean letterforms.
    ui=Image.new('RGBA',(W*S,H*S))
    dr=ImageDraw.Draw(ui)
    def text(p,s,size=28,fill=cream,bold=False,cjk=False,anchor='lt'):
        dr.text((p[0]*S,p[1]*S),s,font=font(size,bold,cjk),fill=fill,anchor=anchor)
    def line(points,fill=cream,width=1):
        dr.line([(round(x*S),round(y*S)) for x,y in points],fill=fill,width=round(width*S),joint='curve')
    def rect(box,fill,radius=0,outline=None,width=1):
        box=tuple(round(x*S) for x in box)
        dr.rounded_rectangle(box,radius=radius*S,fill=fill,outline=outline,width=width*S)

    text((112,91),data['title'],62,bold=True,cjk=True)
    text((116,190),'葱茎与奶酪，把两组变化放在同一把尺上',29,cjk=True,fill=muted)
    rect((1600,100,1885,151),(255,241,220,255),radius=25)
    text((1742.5,125.5),data['visible_notice'],24,fill=(151,43,42,255),bold=True,anchor='mm')
    text((1882,183),'1990—2009  /  单位：%',25,cjk=True,fill=muted,anchor='rt')
    line([(113,250),(1885,250)],(255,235,216,75),1)

    # Shared linear axis, drawn behind the food, never on a second scale.
    text((142,296),'占比 (%)',26,cjk=True,fill=cream)
    line([(223,ymin),(223,ymax)],(255,238,214,140),1)
    for v in [15,20,25,30,35,40,42]:
        y=float(Y(v))
        text((203,y),str(v),25,fill=cream,anchor='rm')
        line([(216,y),(233,y)],(255,238,214,185),1)
        line([(245,y),(1679,y)],(255,238,214,37),1)
    canvas=Image.alpha_composite(canvas,ui.resize(size,Image.Resampling.LANCZOS))
    ui=Image.new('RGBA',(W*S,H*S)); dr=ImageDraw.Draw(ui)

    # Generative bread and butter remain a large visible, photographic substrate.
    bread=Image.open(ASSETS/'bread_butter.png').convert('RGBA').resize((1414,723),Image.Resampling.LANCZOS)
    bread_layer=Image.new('RGBA',size); bread_layer.alpha_composite(bread,(269,364))
    canvas=shadow(canvas,bread_layer.getchannel('A'),(5,13),14,.42,(51,17,17))
    canvas=Image.alpha_composite(canvas,bread_layer)

    # B is the actual bottom cut of this cheese, not an overlaid graph line.
    cheese_top=600.0
    outline=[(xmin,cheese_top),(xmax,cheese_top)]+[tuple(p) for p in ptsB[::-1]]
    cheese_mask=mask_polygon(outline,size)
    cheese_mask.save(ART/'cheese_mask.png')
    cheese=Image.new('RGBA',size)
    tex=Image.open(ASSETS/'cheese_texture.png').convert('RGB').resize((int(xmax-xmin)+1,390),Image.Resampling.LANCZOS)
    cheese.paste(tex,(int(xmin),int(cheese_top)))
    cheese.putalpha(cheese_mask)
    # Inward edge bevel: no added geometry below the data boundary.
    ca=np.array(cheese).astype(float)
    distance=ndimage.distance_transform_edt(np.array(cheese_mask)>127)
    inside=np.array(cheese_mask)>0
    bevel=np.where(inside,np.exp(-distance/1.4),0)
    ca[:,:,:3]*=(1-.115*bevel[:,:,None])
    cheese=Image.fromarray(np.clip(ca,0,255).astype('uint8'))
    canvas=shadow(canvas,cheese_mask,(2,6),4,.38)
    canvas=Image.alpha_composite(canvas,cheese)

    # A is a narrow physical chive. Nearest-segment UV mapping keeps its axis
    # exactly on piecewise-linear data, preserving the generated plant surface.
    radius=8.0
    pad=math.ceil(radius+3)
    box=(int(xs.min())-pad,int(ya.min())-pad,int(xs.max())+pad+1,int(ya.max())+pad+1)
    gx,gy=np.meshgrid(np.arange(box[0],box[2]),np.arange(box[1],box[3]))
    best=np.full(gx.shape,np.inf); along=np.zeros(gx.shape); normal=np.zeros(gx.shape)
    seglen=np.linalg.norm(np.diff(ptsA,axis=0),axis=1)
    cum=np.r_[0,np.cumsum(seglen)]
    for i,(p,q) in enumerate(zip(ptsA[:-1],ptsA[1:])):
        vec=q-p; L=seglen[i]
        t=np.clip(((gx-p[0])*vec[0]+(gy-p[1])*vec[1])/(L*L),0,1)
        cx=p[0]+t*vec[0]; cy=p[1]+t*vec[1]
        dist=np.hypot(gx-cx,gy-cy)
        change=dist<best
        signed=((gx-cx)*(-vec[1])+(gy-cy)*vec[0])/L
        best[change]=dist[change]
        normal[change]=signed[change]
        along[change]=(cum[i]+t*L)[change]
    st=np.array(Image.open(ASSETS/'chive_texture.png').convert('RGB'))
    u=np.clip(along/cum[-1]*(st.shape[1]-1),0,st.shape[1]-1)
    v=np.clip((normal/radius+1)/2*(st.shape[0]-1),0,st.shape[0]-1)
    sample=np.stack([ndimage.map_coordinates(st[:,:,k].astype(float),[v,u],order=1,mode='nearest') for k in range(3)],axis=2)
    alpha=np.clip(radius+.5-best,0,1)*255
    piece=Image.fromarray(np.dstack([np.clip(sample,0,255),alpha]).astype('uint8'))
    stem=Image.new('RGBA',size); stem.alpha_composite(piece,box[:2])
    stem.getchannel('A').save(ART/'stem_mask.png')
    canvas=shadow(canvas,stem.getchannel('A'),(1,3),2,.46)
    canvas=Image.alpha_composite(canvas,stem)

    # Value capsules have short leaders to the mapped coordinates.
    def dot(x,y,color,r=3.4):
        dr.ellipse(((x-r)*S,(y-r)*S,(x+r)*S,(y+r)*S),fill=color)
    def pill(x,y,value,kind):
        bg=(66,102,43,255) if kind=='A' else (255,246,217,250)
        fg=(255,251,227,255) if kind=='A' else (67,51,27,255)
        edge=(230,237,191,110) if kind=='A' else (123,99,55,75)
        rect((x-49,y-23,x+49,y+23),bg,23,edge,1)
        text((x,y-1),f'{value:.1f}',29,fg,bold=False,anchor='mm')
    label_info=[]
    for i,(x,y,val) in enumerate(zip(xs,ya,av)):
        px,py=float(x),float(y)+(48 if i==0 else -48)
        line([(x,y),(px,py-23 if i==0 else py+23)],(241,249,208,225),1.5)
        dot(x,y,(251,249,216,255),3)
        pill(px,py,val,'A')
        label_info.append({'series':'A','year':data['categories'][i],'text':f'{val:.1f}','point':[float(x),float(y)],'label_center':[px,py]})
    for i,(x,y,val) in enumerate(zip(xs,yb,bv)):
        px=float(x)+(45 if i==0 else (-42 if i==len(xs)-1 else 0))
        py=float(y)-39
        line([(x,y),(px,py+23)],(103,91,49,235),1.4)
        dot(x,y,(103,83,37,255),3.2)
        pill(px,py,val,'B')
        label_info.append({'series':'B','year':data['categories'][i],'text':f'{val:.1f}','point':[float(x),float(y)],'label_center':[px,py]})

    # Short callouts keep series/category labels separate from food texture.
    endA=float(ya[-1]); endB=float(yb[-1])
    line([(xmax+13,endA),(1688,endA),(1721,endA-22)],(244,234,204,215),1.5)
    text((1734,endA-48),ser['A']['label'],29,bold=True)
    text((1734,endA-6),'葱茎中心线',22,cjk=True,fill=muted)
    line([(xmax+10,endB),(1688,endB),(1721,endB+23)],(244,234,204,215),1.5)
    text((1734,endB+4),ser['B']['label'],29,bold=True)
    text((1734,endB+46),'奶酪下边缘',22,cjk=True,fill=muted)

    for x,year in zip(xs,data['categories']):
        line([(x,1114),(x,1126)],(255,236,212,160),1.5)
        text((x,1147),year,30,fill=cream,anchor='mt')
    text((1695,1149),'年份',24,cjk=True,fill=muted)
    line([(113,1210),(1885,1210)],(255,235,216,75),1)
    text((116,1240),'合成数据 · 年份按真实间距 · 共用 15–42% 线性轴',24,cjk=True,fill=cream)
    text((1885,1240),'奶酪上缘、食材面积与厚度不编码',23,cjk=True,fill=muted,anchor='rt')
    canvas=Image.alpha_composite(canvas,ui.resize(size,Image.Resampling.LANCZOS))
    canvas.convert('RGB').save(ROOT/'final.png')

    # Measure rasterized, visible component boundaries rather than target echoes.
    ma=np.array(cheese_mask); ms=np.array(stem.getchannel('A'))
    cheese_samples=[]; stem_samples=[]
    for xi in range(math.ceil(xmin)+2,math.floor(xmax)-1):
        cys=np.where(ma[:,xi]>=128)[0]
        sys=np.where(ms[:,xi]>=128)[0]
        if cys.size:
            actual=float(cys.max())+.5
            expected=float(np.interp(xi,xs,yb))
            cheese_samples.append((xi,actual,expected,abs(actual-expected)))
        if sys.size:
            actual=float(sys.min()+sys.max())/2
            expected=float(np.interp(xi,xs,ya))
            stem_samples.append((xi,actual,expected,abs(actual-expected)))
    b_err=max(row[3] for row in cheese_samples)
    a_err=max(row[3] for row in stem_samples)
    crossings=[]
    for i in range(len(times)-1):
        d0=av[i]-bv[i]; d1=av[i+1]-bv[i+1]
        if d0*d1<0:
            t=-d0/(d1-d0); year=float(times[i]+t*(times[i+1]-times[i]))
            val=float(av[i]+t*(av[i+1]-av[i]))
            cx,cy=float(X(year)),float(Y(val))
            crossings.append({'year':year,'percent':val,'pixel':[cx,cy],
                'stem_visible_at_crossing':bool(ms[round(cy),round(cx)]>128),
                'layer_order':'table < bread and butter < cheese < chive < numeric labels'})
    def node_measures(points,mask,kind):
        out=[]
        for i,(x,y) in enumerate(points):
            xi=round(x); ys=np.where(mask[:,xi]>=128)[0]
            actual=float(ys.max())+.5 if kind=='B' else float(ys.min()+ys.max())/2
            out.append({'year':data['categories'][i],'x_px':xi,'measured_y_px':actual,'expected_y_px':float(y),'error_px':abs(actual-y)})
        return out
    checks={
        'canvas':[W,H],'source':'data.json','mapping':{'x_range':[xmin,xmax],'y_range':[ymin,ymax],'value_range':[vmin,vmax],'x_scale':'true year spacing','y_scale':'linear'},
        'years':data['categories'],'x_coordinates':xs.tolist(),'A_nodes':node_measures(ptsA,ms,'A'),'B_nodes':node_measures(ptsB,ma,'B'),
        'dense_raster_measurements':{'A_columns':len(stem_samples),'B_columns':len(cheese_samples),'A_max_centerline_error_px':a_err,'B_max_lower_edge_error_px':b_err,'A_error_percent_full_scale':100*a_err/(ymax-ymin),'B_error_percent_full_scale':100*b_err/(ymax-ymin)},
        'crossings':crossings,'numeric_labels':label_info,
        'cheese_top_y_px':cheese_top,'positive_cheese_thickness_min_px':float(yb.min()-cheese_top),
        'geometry_pass':a_err<(ymax-ymin)*.03 and b_err<(ymax-ymin)*.03,
        'mask_paths':['artifacts/cheese_mask.png','artifacts/stem_mask.png']
    }
    (ART/'geometry_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'canvas':[W,H],'geometry_pass':checks['geometry_pass'],'A_max_error_px':a_err,'B_max_error_px':b_err,'crossings':crossings},ensure_ascii=False))


if __name__=='__main__':
    main()
