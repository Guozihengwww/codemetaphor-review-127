#!/usr/bin/env python3
"""Rebuild final.png from frozen data and local generated assets, offline.

All complete invocations in this experiment must run under capture_render.py.
Quantitative regions use a capacity-fitted warped power diagram plus an
analytic-area organic interior. No image-model call is made here.
"""
from pathlib import Path
import json, hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from geometry import solve, classify, diagnostics

ROOT=Path(__file__).resolve().parent
S=2
W,H=1440,2100
CX,CY,R=720,840,470
INK='#4b302a'
PAPER='#faf2d0'
PALETTE=['#b9493f','#e2aa68','#da8668','#efcaa1','#b45f49','#e5a57f',
         '#cc8670','#f0b59b','#b96c60','#e6ad90','#c4816b','#e5a08a','#f6ddbd']


def rgb(c):return tuple(bytes.fromhex(c.lstrip('#')))
def font(name,size):return ImageFont.truetype(str(ROOT/'assets/fonts'/name),round(size*S))

def main():
    data=json.loads((ROOT/'data.json').read_text(encoding='utf-8'))
    brief=json.loads((ROOT/'brief.json').read_text(encoding='utf-8'))
    assert (brief['width'],brief['height'])==(W,H)
    values=data['series'][0]['values']; cats=data['categories']; total=data['total']
    assert sum(values)==total and len(values)==13
    sites,weights,edges,outer,interior,solver=solve(data)
    ids_diag,anchors,check=diagnostics(data,sites,weights,edges,outer,interior)
    assert check['outer_order']==[cats[i] for i in outer]
    assert not check['interior_touches_rim']
    assert all(p['connected_components']==1 for p in check['areas'])
    assert check['max_area_error_percentage_points']<.05
    # A quiet, deterministic paper grain, shared by the chart and page.
    rng=np.random.default_rng(731)
    noise=rng.normal(0,.48,(H,W,1)).astype(np.float32)
    yy,xx=np.mgrid[0:H,0:W].astype(np.float32)
    shading=((xx-W*.48)**2/(W*W)+(yy-H*.45)**2/(H*H))[:,:,None]
    paper=np.array(rgb(PAPER),np.float32)[None,None,:]+noise-1.9*shading
    im=Image.fromarray(np.uint8(np.clip(paper,0,255)),'RGB').resize((W*S,H*S),Image.Resampling.BICUBIC).convert('RGBA')
    draw=ImageDraw.Draw(im)
    draw.rectangle((32*S,32*S,(W-32)*S,(H-32)*S),outline='#d5bf94',width=S)

    # Supersampled label map is the single geometry used for coloring, strokes,
    # labels, measurements and topology checks.
    pad=6; left=CX-R-pad; top=CY-R-pad; N=2*(R+pad)*S
    z=(np.arange(N,dtype=np.float32)+.5)/S-pad
    x,y=np.meshgrid((z-R)/R,(z-R)/R)
    disk=x*x+y*y<=1
    labels=classify(x,y,sites,weights,interior,values[interior]/total)
    labels[~disk]=255
    colors=np.array([rgb(c) for c in PALETTE]+[(0,0,0)]*243,np.float32)
    fill=colors[labels]
    grain=rng.normal(0,.72,(N,N)).astype(np.float32)
    slow=ndimage.gaussian_filter(rng.normal(0,1,(N,N)).astype(np.float32),16)
    slow=slow/(slow.std()+1e-8)*.32
    fill=np.clip(fill+(grain+slow)[:,:,None],0,255)
    # Thin, centrally placed seam ink. Region proportions are measured both
    # before the seams and on the remaining visible colored surface.
    edge=np.zeros((N,N),bool)
    v=(labels[1:]!=labels[:-1]) & disk[1:] & disk[:-1]
    edge[1:]|=v;edge[:-1]|=v
    v=(labels[:,1:]!=labels[:,:-1]) & disk[:,1:] & disk[:,:-1]
    edge[:,1:]|=v;edge[:,:-1]|=v
    dist=ndimage.distance_transform_edt(~edge)/S
    ink_alpha=np.clip(1.6-dist,0,1)*disk
    # Circle contour marks the exact quantitative extent D, under the rope rim.
    rr=np.sqrt(x*x+y*y)*R
    rim_alpha=np.clip(1.35-(R-rr),0,1)*disk
    ink_alpha=np.maximum(ink_alpha,rim_alpha)
    fill=fill*(1-ink_alpha[:,:,None])+np.array(rgb('#75463d'))*ink_alpha[:,:,None]
    rgba=np.zeros((N,N,4),np.uint8);rgba[:,:,:3]=np.uint8(fill);rgba[:,:,3]=disk.astype(np.uint8)*255
    layer=Image.fromarray(rgba,'RGBA')
    im.alpha_composite(layer,(left*S,top*S))

    # The generated frame supplies the actual wrapped hoop, knot, suspension
    # cord and the three overlapping feather groups. Its opening is measured
    # and registered by prepare_assets.py to the data disk.
    frame=Image.open(ROOT/'assets/frame_01.png').convert('RGBA')
    frame_np=np.asarray(frame)
    yf,xf=np.mgrid[0:H,0:W]
    frame_inside=int(frame_np[((xf-CX)**2+(yf-CY)**2<(R-1)**2),3].max())
    assert frame_inside==0
    im.alpha_composite(frame.resize((W*S,H*S),Image.Resampling.LANCZOS))
    draw=ImageDraw.Draw(im)

    def centered(text,cx,cy,f,color=INK):
        b=draw.textbbox((0,0),text,font=f)
        tw=b[2]-b[0];th=b[3]-b[1]
        px=cx*S-tw/2-b[0];py=cy*S-th/2-b[1]
        draw.text((round(px),round(py)),text,font=f,fill=color)
        return [cx-tw/(2*S),cy-th/(2*S),cx+tw/(2*S),cy+th/(2*S)]
    def tracked(text,cx,cy,size,tracking,color):
        f=font('sans.ttf',size)
        widths=[draw.textlength(c,font=f) for c in text]
        totalw=sum(widths)+(len(text)-1)*tracking*S
        xpos=cx*S-totalw/2
        for c,cw in zip(text,widths):
            b=draw.textbbox((0,0),c,font=f)
            draw.text((round(xpos),round(cy*S-(b[1]+b[3])/2)),c,font=f,fill=color)
            xpos+=cw+tracking*S
    tracked(data['visible_notice'],421,78,18,3.3,'#944f3d')
    tracked(f'{len(cats)} GROUPS',1020,78,18,3.3,'#944f3d')
    centered('捕梦网',421,146,font('title.ttc',90),'#6b382e')
    centered('曲边区域',1020,147,font('title.ttc',68),'#6b382e')
    centered(data['series'][0]['label'],421,218,font('sans.ttf',22),'#775648')
    centered(f'总计 {total} '+data['series'][0]['unit'],1020,218,font('title.ttc',24),'#775648')

    label_records=[]
    def data_label(i,cx,cy,external=False):
        prominent=(i==0)
        col='#fff3d5' if i in (0,4) else '#422c27'
        if external: col=INK
        fs=28 if prominent else 22
        cat_box=centered(cats[i],cx,cy-(47 if prominent else 28),font('sans.ttf',fs),col)
        countfont=font('serif.ttf',70 if prominent else 46)
        denomfont=font('serif.ttf',35 if prominent else 27)
        v=str(values[i]); den=' / '+str(total)
        a=draw.textlength(v,font=countfont);b=draw.textlength(den,font=denomfont)
        tx=cx*S-(a+b)/2;baseline=(cy+32)*S
        draw.text((round(tx),round(baseline)),v,font=countfont,fill=col,anchor='ls')
        draw.text((round(tx+a),round(baseline-2*S)),den,font=denomfont,fill=col,anchor='ls')
        vb=draw.textbbox((round(tx),round(baseline)),v,font=countfont,anchor='ls')
        db=draw.textbbox((round(tx+a),round(baseline-2*S)),den,font=denomfont,anchor='ls')
        box=[min(cat_box[0],vb[0]/S,db[0]/S),min(cat_box[1],vb[1]/S,db[1]/S),
             max(cat_box[2],vb[2]/S,db[2]/S),max(cat_box[3],vb[3]/S,db[3]/S)]
        fraction=None
        if not external:
            x0,y0,x1,y1=box
            qx,qy=np.meshgrid(np.linspace(x0,x1,70),np.linspace(y0,y1,45))
            ids=classify((qx-CX)/R,(qy-CY)/R,sites,weights,interior,values[interior]/total)
            fraction=float(np.mean(ids==i))
        label_records.append({'category':cats[i],'text':f'{cats[i]}: {values[i]} / {total}',
            'external':external,'center':[cx,cy],'bbox':box,'bbox_fraction_in_own_region':fraction})
    for i,a in enumerate(anchors):
        if i==1:continue
        cx=CX+a[0]*R;cy=CY+a[1]*R
        if i==0:cy+=6
        data_label(i,cx,cy)
    # The 6-record cell keeps its true area. Its category/value are outside,
    # attached by a single leader terminating well within the correct region.
    a=anchors[1]; px=CX+a[0]*R;py=CY+a[1]*R
    p2=(1139,515);p3=(1337,515)
    draw.line([(round(px*S),round(py*S)),(p2[0]*S,p2[1]*S),(p3[0]*S,p3[1]*S)],fill='#6b4235',width=2*S)
    dot=3.6*S
    draw.ellipse((px*S-dot,py*S-dot,px*S+dot,py*S+dot),fill='#6b4235')
    data_label(1,1254,461,True)
    # Compact key states the area encoding and the only quantity unit.
    draw.line((94*S,1978*S,1346*S,1978*S),fill='#c3a77d',width=S)
    centered('区域面积 = 计数 / 301  ·  单位：records（条）',720,2008,font('title.ttc',22),'#644536')
    centered('外围类别按 01–12 顺时针排列；Group 13 为内部区域。',720,2043,font('title.ttc',20),'#795c48')

    # Record measurements from the displayed supersampled label map, not from
    # the target vector. Border/text ink does not redefine a region's boundary.
    n_disk=int(disk.sum());bare=np.bincount(labels[disk],minlength=len(values))
    visible=np.array([np.sum((labels==i)*(1-ink_alpha)) for i in range(len(values))])
    per=[]
    for i in range(len(values)):
        target=values[i]/total
        per.append({'category':cats[i],'value':values[i],'denominator':total,
                    'target_share':target,'rendered_region_pixels':int(bare[i]),
                    'rendered_share':float(bare[i]/n_disk),
                    'area_error_percentage_points':float(100*(bare[i]/n_disk-target)),
                    'visible_fill_share_after_seams':float(visible[i]/visible.sum()),
                    'visible_fill_error_percentage_points':float(100*(visible[i]/visible.sum()-target))})
    check.update({'canvas':[W,H],'supersample_factor':S,'data_disk_center':[CX,CY],'data_disk_radius':R,
        'rendered_disk_pixels':n_disk,'rendered_areas':per,'generated_frame_max_alpha_inside_disk':frame_inside,
        'all_values_sum_to_total':sum(values)==total,'category_count':len(cats),
        'labels':label_records,'small_label_leader_anchor':[px,py],
        'small_label_leader_anchor_category':cats[int(classify(np.array([(px-CX)/R]),np.array([(py-CY)/R]),sites,weights,interior,values[interior]/total)[0])],
        'solver':solver,'source_data_sha256':hashlib.sha256((ROOT/'data.json').read_bytes()).hexdigest()})
    (ROOT/'artifacts').mkdir(exist_ok=True)
    (ROOT/'artifacts/render_checks.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (ROOT/'artifacts/geometry_used.json').write_text(json.dumps({'sites':sites.tolist(),'weights':weights.tolist(),'anchors':anchors},indent=2)+'\n',encoding='utf-8')
    im.convert('RGB').resize((W,H),Image.Resampling.LANCZOS).save(ROOT/'final.png')
    print(json.dumps({'file':'final.png','size':[W,H],
         'max_area_error_pp':max(abs(q['area_error_percentage_points']) for q in per),
         'max_visible_fill_error_pp':max(abs(q['visible_fill_error_percentage_points']) for q in per),
         'label_box_containment':{q['category']:q['bbox_fraction_in_own_region'] for q in label_records}},ensure_ascii=False))

if __name__=='__main__':main()
