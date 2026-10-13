#!/usr/bin/env python3
"""Offline melon-area infographic. Full renders must use capture_render.py.

A single linear area mapping is used: 50 response percentage points per full
circle; all top surfaces have vertical scale 0.96. The four detached sectors are
translations of the same source disk, without rotation, scaling, or minimum size.
"""
from pathlib import Path
import json
import math
import hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageChops

ROOT=Path(__file__).resolve().parent
S=3
R=294.0
Y_SCALE=0.96
MARGIN=49
LOCAL=int(2*(R+MARGIN))
INK=(48,57,42)
MUTED=(102,109,88)
LINE=(151,158,137)
GOLD=(152,112,25)
BG=(241,242,232)

def read_json(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))

def write_json(name, value):
    path=ROOT/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    config=read_json('input.json')
    brief=read_json(config['brief'])
    data=read_json(config['data'])
    core=read_json(config['critical_metaphor_contract'])
    assert core['visible_title']==data['title']
    assert data['synthetic'] is True
    values=[float(v) for v in data['series'][0]['values']]
    categories=data['categories']
    assert len(values)==len(categories)==6
    assert abs(sum(values)-100)<1e-9
    assert values[0]==data['left_capacity']==data['right_capacity']==50
    assert abs(sum(values[1:])-data['right_capacity'])<1e-9
    W,H=brief['width'],brief['height']
    assert (W,H)==(1486,2100)
    canvas=Image.new('RGB',(W*S,H*S),BG)
    draw=ImageDraw.Draw(canvas)
    regular=ROOT/'assets/fonts/NimbusSans-Regular.otf'
    bold=ROOT/'assets/fonts/NimbusSans-Bold.otf'
    fonts={}
    labels=[]

    def font(size,weight='regular'):
        key=(size,weight)
        if key not in fonts:
            fonts[key]=ImageFont.truetype(str(bold if weight=='bold' else regular),round(size*S))
        return fonts[key]

    def text(x,y,s,size=28,color=INK,weight='regular',align='left',role=None):
        f=font(size,weight)
        width=draw.textlength(s,font=f)/S
        if align=='center': x-=width/2
        elif align=='right': x-=width
        box=draw.textbbox((round(x*S),round(y*S)),s,font=f,anchor='lt')
        draw.text((round(x*S),round(y*S)),s,font=f,fill=color,anchor='lt')
        labels.append({'text':s,'role':role or 'annotation','bbox_px':[round(k/S,3) for k in box]})
        return width

    def line(points,fill=LINE,width=1.2):
        draw.line([(round(x*S),round(y*S)) for x,y in points],fill=fill,width=max(1,round(width*S)),joint='curve')

    def percent(x,y,value,size,center=False,role='value'):
        number=format(value,'.12g')
        f=font(size,'regular'); pf=font(round(size*.49),'regular')
        gap=size*.038
        width=draw.textlength(number,font=f)/S+gap+draw.textlength('%',font=pf)/S
        if center: x-=width/2
        a=text(x,y,number,size,role=role)
        text(x+a+gap,y+size*.47,'%',round(size*.49),role=role)
        return width

    # Editorial heading and unmistakable synthetic-data notice.
    text(88,103,'SIMULATED RESPONSE DISTRIBUTION',21,MUTED,weight='bold')
    draw.rounded_rectangle((1013*S,87*S,1398*S,141*S),radius=7*S,fill=(224,229,210))
    text(1205.5,103,data['visible_notice'],25,INK,weight='bold',align='center',role='synthetic_notice')
    text(88,202,'Reasons for fewer',112,weight='bold',role='title')
    text(88,319,'women on panels',112,weight='bold',role='title')
    text(90,458,'Percent of simulated responses',30,MUTED,role='unit')
    line([(88,514),(1398,514)],fill=(187,192,173),width=1)

    # Major category labels have short leaders to their corresponding whole faces.
    centers=[(382.0,1020.0),(1104.0,1020.0)]
    for k,(cx,cy) in enumerate(centers):
        percent(cx,561,values[k],118,center=True,role='category_value_'+str(k))
        text(cx,688,categories[k],32,align='center',role='category_'+str(k))
        line([(cx,724),(cx,cy-R*Y_SCALE-6)],width=1.3)

    rind=Image.open(ROOT/'assets/melon_rind.png').convert('RGB')
    flesh=Image.open(ROOT/'assets/flesh_strip.png').convert('RGB')
    # Local metric coordinates are the single source for all six masks.
    n=LOCAL*S
    center=R+MARGIN
    yy,xx=np.mgrid[0:n,0:n].astype(np.float32)
    px=(xx+.5)/S-center
    py=((yy+.5)/S-center)/Y_SCALE
    rho2=px*px+py*py
    theta=(np.degrees(np.arctan2(py,px))+360)%360
    inside=rho2<=R*R
    baseline_area=float(inside.sum())/(S*S)
    analytic_area=math.pi*R*R*Y_SCALE
    # Material is mapped into the same uncut coordinate disk before sector masking.
    rw=rind.width
    affine=(rw/(2*R*S),0,rw*(R-center)/(2*R),
            0,rw/(2*R*Y_SCALE*S),rw*(R-center/Y_SCALE)/(2*R))
    texture=rind.transform((n,n),Image.Transform.AFFINE,affine,resample=Image.Resampling.BICUBIC)
    flesh_tex=ImageEnhance.Color(flesh.resize((n,n),Image.Resampling.BICUBIC)).enhance(.86)
    flesh_tex=ImageEnhance.Brightness(flesh_tex).enhance(.86)
    (ROOT/'artifacts/masks').mkdir(parents=True,exist_ok=True)
    pieces=[]
    gap_start=5.0
    b_angle=360*values[1]/data['right_capacity']
    gap_end=gap_start+360-b_angle
    specs=[{'index':0,'start':0.0,'sweep':360.0,'center':centers[0]},
           {'index':1,'start':gap_end,'sweep':b_angle,'center':centers[1]}]
    angle=gap_start
    row_ys=[1432.0,1538.0,1644.0,1750.0]
    for i,row in zip(range(2,6),row_ys):
        sweep=360*values[i]/data['right_capacity']
        mid=math.radians(angle+sweep/2)
        # The sector centroid aligns with its label row. All movement is translation.
        cx=950.0
        cy=round(row-(2/3)*R*Y_SCALE*math.sin(mid))
        specs.append({'index':i,'start':angle,'sweep':sweep,'center':(cx,float(cy))})
        angle+=sweep
    assert abs(angle-gap_end)<1e-9

    for spec in specs:
        i=spec['index']; start=spec['start']; sweep=spec['sweep']
        if sweep==360:
            mask_bool=inside.copy()
        else:
            relative=(theta-start+360)%360
            mask_bool=inside & (relative<sweep)
        mask=Image.fromarray(mask_bool.astype('uint8')*255,'L')
        # Identical 6 px extrusion. Its flesh texture is a non-data sidewall.
        depth=6*S
        union=mask.copy()
        for dy in range(1,depth+1):
            shifted=Image.new('L',(n,n),0)
            shifted.paste(mask,(0,dy))
            union=ImageChops.lighter(union,shifted)
        wall_alpha=ImageChops.subtract(union,mask)
        wall=flesh_tex.copy().convert('RGBA')
        wall.putalpha(wall_alpha)
        radius=10 if i<2 else 3
        offset=(2*S,10*S) if i<2 else (1*S,5*S)
        shadow_mask=Image.new('L',(n,n),0)
        shadow_mask.paste(union,offset)
        shadow_mask=shadow_mask.filter(ImageFilter.GaussianBlur(radius*S)).point(lambda p:round(p*.19))
        piece=Image.new('RGBA',(n,n),(44,47,26,0))
        piece.putalpha(shadow_mask)
        piece.alpha_composite(wall)
        top=texture.copy().convert('RGBA')
        top.putalpha(mask)
        piece.alpha_composite(top)
        cx,cy=spec['center']
        x0=int(round(cx-center)); y0=int(round(cy-center))
        canvas.paste(piece,(x0*S,y0*S),piece)
        # Save exact compositing masks for independent raster-area audit.
        low_mask=mask.resize((LOCAL,LOCAL),Image.Resampling.LANCZOS)
        low_mask.save(ROOT/f'artifacts/masks/top_{i+1:02d}.png')
        ar=np.asarray(low_mask,dtype=np.float64)/255
        area=float(ar.sum())
        ys,xs=np.nonzero(ar>.5)
        origin=(center,center)
        # Measure occupied boundary radii in the actual raster mask.
        rr=np.sqrt(((xs+.5-center)/R)**2+(((ys+.5-center)/Y_SCALE)/R)**2)
        radius95=np.percentile(rr,95)
        if i>=2:
            band=(rr>.94)&(rr<.985)
            angles=(np.degrees(np.arctan2((ys+.5-center)/Y_SCALE,xs+.5-center))+360)%360
            edge_angles=angles[band]
            observed_sweep=float(edge_angles.max()-edge_angles.min()) if len(edge_angles) else None
        else:
            observed_sweep=None
        pieces.append({
            'index':i,'category':categories[i],'value_percent':values[i],
            'capacity_percent':50,'start_deg':start,'sweep_deg':sweep,
            'center_px':[cx,cy],'source_center_px':list(centers[0] if i==0 else centers[1]),
            'translation_px':[cx-(centers[0] if i==0 else centers[1])[0],cy-(centers[0] if i==0 else centers[1])[1]],
            'radius_px':R,'vertical_scale':Y_SCALE,'mask_path':f'artifacts/masks/top_{i+1:02d}.png',
            'mask_origin_px':[x0,y0],
            'actual_alpha_area_px2':area,'recovered_percent_from_alpha':area/(2*baseline_area)*100,
            'hard_mask_area_px2':int((ar>.5).sum()),
            'observed_outer_band_angular_span_deg':observed_sweep,
            'theoretical_outer_arc_px':R*math.radians(sweep),
            'occupied_top_bbox_px':[int(x0+xs.min()),int(y0+ys.min()),int(x0+xs.max()+1),int(y0+ys.max()+1)]
        })

    # Rows are the four original right-face sectors, physically moved below it.
    text(88,1368,'FOUR EXPLODED SLICES',21,MUTED,weight='bold')
    text(1398,1368,'1% combined · true scale',23,MUTED,align='right')
    for i,row in zip(range(2,6),row_ys):
        percent(88,row-29,values[i],59,role='category_value_'+str(i))
        text(305,row-12,categories[i],31,role='category_'+str(i))
        cx,cy=specs[i]['center']
        line([(818,row),(894,row),(cx-13,cy)],fill=(164,170,150),width=1)

    # Reading key and response model are part of the visible statistical framing.
    line([(88,1850),(1398,1850)],fill=(179,186,163),width=1)
    text(88,1882,'READING THE MELON FACES',21,MUTED,weight='bold')
    text(88,1924,'Each complete face = 50%. Together = 100%.',28)
    text(88,1965,'Top-face area encodes share; gaps, edges',25,MUTED)
    text(88,2000,'and shadows do not. Tiny slices are not enlarged.',25,MUTED)
    text(877,1882,'RESPONSE MODEL',21,MUTED,weight='bold')
    text(877,1924,'One mutually exclusive reason',26)
    text(877,1962,'per participant. Simulated values;',26)
    text(877,2000,'not a real survey.',26,weight='bold')

    canvas.resize((W,H),Image.Resampling.LANCZOS).save(ROOT/'final.png')
    write_json('artifacts/geometry.json',{
        'canvas_px':[W,H],'supersampling':S,'mapping':'percentage = top-face alpha area / area of two equal complete faces * 100',
        'baseline_radius_px':R,'vertical_scale':Y_SCALE,'sidewall_depth_px':6,
        'raster_baseline_area_px2':baseline_area,'analytic_baseline_area_px2':analytic_area,
        'right_total_sweep_deg':sum(p['sweep_deg'] for p in pieces[1:]),
        'pieces':pieces,'labels':labels,
        'inputs_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['input.json','brief.json','data.json','core.json','reference.png']}
    })
    print('Rendered final.png from local material and frozen data; geometry audit saved.')

if __name__=='__main__':
    main()
