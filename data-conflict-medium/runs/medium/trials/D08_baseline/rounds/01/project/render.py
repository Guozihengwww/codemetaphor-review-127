"""Regenerate final.png with local generated artwork and frozen numerical data.
Run through capture_render.py for every complete chart render in this experiment.
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt, minimum_filter, label, gaussian_filter
from geometry import build, classify, from_power

ROOT=Path(__file__).resolve().parent
S=2
W,H=1440,2100
CX,CY,R=720,905,520
INK='#472C29'
PALETTE=['#BB594B','#E9C981','#D88F68','#F1C5A1','#E8A583','#D47765','#F7DAA9']

def font(name,size):
    ext='ttc' if name=='chinese' else 'otf'
    return ImageFont.truetype(str(ROOT/f'assets/fonts/{name}.{ext}'),round(size*S))

def xy(p):return tuple(round(t*S) for t in p)

def text(draw,position,txt,f,fill=INK,anchor='lt'):
    draw.text(xy(position),txt,font=f,fill=fill,anchor=anchor)

def tracked(draw,txt,pos,f,tracking=2.3,fill=INK,align='left'):
    widths=[draw.textlength(c,font=f)/S for c in txt]
    width=sum(widths)+(len(txt)-1)*tracking
    xx=pos[0]-(width/2 if align=='center' else width if align=='right' else 0)
    for c,ww in zip(txt,widths):
        text(draw,(xx,pos[1]),c,f,fill);xx+=ww+tracking
    return width

def densify(poly,max_step=.006):
    out=[]
    for a,b in zip(poly,np.roll(poly,-1,axis=0)):
        n=max(1,int(np.ceil(np.linalg.norm(b-a)/max_step)))
        out.append(a+(b-a)*np.arange(n)[:,None]/n)
    return np.vstack(out)

def composite_asset(canvas,name,left,top,scale=1.,occlusion=None):
    im=Image.open(ROOT/f'assets/{name}.png').convert('RGBA')
    im=im.resize((round(im.width*scale*S),round(im.height*scale*S)),Image.Resampling.LANCZOS)
    position=xy((left,top))
    canvas.alpha_composite(im,position)
    if occlusion is not None:
        occlusion.paste(im.getchannel('A'),position,im.getchannel('A'))
    return [left,top,im.width/S,im.height/S]

def main():
    brief=json.loads((ROOT/'brief.json').read_text(encoding='utf-8'))
    assert (brief['width'],brief['height'])==(W,H)
    data,sites,weights,polygons,geom=build()
    categories=data['categories'];values=data['series'][0]['values'];total=data['total']
    assert sum(values)==total and len(categories)==len(values)
    rng=np.random.default_rng(8127)
    # Subtle cream paper grain, deterministic and independent of the data.
    grain=rng.normal(0,.8,(H,W)).astype(np.float32)
    cloud=gaussian_filter(rng.normal(0,1,(H,W)).astype(np.float32),32)
    cloud/=max(cloud.std(),1e-8)
    paper=np.clip(np.array([250.,244.,215.])[None,None,:]+(grain+cloud*.36)[:,:,None],0,255).astype('uint8')
    canvas=Image.fromarray(paper,'RGB').resize((W*S,H*S),Image.Resampling.BICUBIC).convert('RGBA')
    draw=ImageDraw.Draw(canvas)
    draw.rectangle(xy((35,34,1405,2066)),outline='#A68A68',width=S)
    draw.line([xy((72,102)),xy((1368,102))],fill='#CBB99B',width=S)
    tracked(draw,data['visible_notice'],(76,63),font('labels',22),2.9)
    tracked(draw,f'{total} RECORDS  /  {len(categories)} REGIONS',(1364,63),font('body',22),1.5,align='right')
    title_font=font('title',169)
    title='Dream Catcher'
    tw=draw.textlength(title,font=title_font)/S
    text(draw,((W-tw)/2-6,129),title,title_font,'#A84E42')
    text(draw,(W/2,268),data['title'],font('chinese',27),'#664D3D',anchor='mt')
    # The generated hanging parts sit behind the generated hoop.
    placements={}
    placements['feathers_left']=composite_asset(canvas,'feathers_left',144.7,1228.2,.87)
    placements['feathers_right']=composite_asset(canvas,'feathers_right',1022.1,1228.1,.87)
    placements['feathers_center']=composite_asset(canvas,'feathers_center',600.1,1431.7,.74)
    # Pixel-center classification is the actual rendered data geometry.
    n=2*R*S
    grid=(np.arange(n,dtype=np.float32)+.5-R*S)/(R*S)
    gx,gy=np.meshgrid(grid,grid)
    ids=classify(gx,gy,sites,weights)
    inside=ids>=0
    rgb=np.zeros((n,n,4),dtype=np.uint8)
    pigment=rng.normal(0,.80,(n,n)).astype(np.float32)
    large=gaussian_filter(rng.normal(0,1,(n,n)).astype(np.float32),15)
    large/=max(large.std(),1e-8)
    pigment+=large*.85
    colors=[tuple(bytes.fromhex(c[1:])) for c in PALETTE]
    for i,color in enumerate(colors):
        m=ids==i
        for channel in range(3):rgb[:,:,channel][m]=np.clip(color[channel]+pigment[m],0,255)
    rgb[:,:,3]=inside.astype('uint8')*255
    disk=Image.fromarray(rgb,'RGBA')
    canvas.alpha_composite(disk,xy((CX-R,CY-R)))
    draw=ImageDraw.Draw(canvas)
    path_points=[]
    for poly in polygons:
        pp=from_power(densify(poly))*R+np.array([CX,CY])
        path_points.append(pp)
        draw.line([xy(p) for p in np.vstack([pp,pp[0]])],fill='#674137',width=round(2.7*S),joint='curve')
    # Original generated rope, remapped once onto the exact circular data boundary.
    occlusion=Image.new('L',canvas.size,0)
    placements['rope_ring']=composite_asset(canvas,'rope_ring',CX-599.5,CY-599.5,1.,occlusion)
    # Top knot: trim only the excess suspension cord, preserving the generated knot.
    knot=Image.open(ROOT/'assets/top_knot.png').convert('RGBA').crop((0,35,80,125))
    knot=knot.resize((96*S,108*S),Image.Resampling.LANCZOS)
    kp=xy((CX-48,311))
    canvas.alpha_composite(knot,kp);occlusion.paste(knot.getchannel('A'),kp,knot.getchannel('A'))
    placements['top_knot']=[CX-48,311,96,108]
    draw=ImageDraw.Draw(canvas)
    # Derive label anchors from the same final-scale masks used by the renderer.
    g=(np.arange(2*R,dtype=np.float32)+.5-R)/R
    xx,yy=np.meshgrid(g,g)
    labels=classify(xx,yy,sites,weights)
    label_meta=[]
    name_font=font('labels',28);num_font=font('numbers',76);den_font=font('body',25)
    for i,(category,value) in enumerate(zip(categories,values)):
        m=labels==i
        distance=distance_transform_edt(m)
        name_width=draw.textlength(category,font=name_font)/S
        num_width=draw.textlength(str(value),font=num_font)/S
        denom=f'/ {total}'
        denom_width=draw.textlength(denom,font=den_font)/S
        bw=max(name_width,num_width+10+denom_width)
        bh=96
        external=category=='Dream G'
        if external:
            anchor=np.array([1166.,397.])
        else:
            clearance=minimum_filter(m.astype('uint8'),size=(bh+5,int(np.ceil(bw))+12),mode='constant')
            score=distance*clearance
            if score.max()<=0:
                score=distance
            ay,ax=np.unravel_index(score.argmax(),score.shape)
            anchor=np.array([CX-R+ax+.5,CY-R+ay+.5])
        color='#FFF3D7' if i==0 else INK
        text(draw,(anchor[0],anchor[1]-23),category,name_font,color,anchor='ms')
        start=anchor[0]-(num_width+10+denom_width)/2
        text(draw,(start,anchor[1]+46),str(value),num_font,color,anchor='ls')
        text(draw,(start+num_width+10,anchor[1]+43),denom,den_font,color,anchor='ls')
        # Actual ink boxes, expressed in native output pixels, for collision checks.
        items=[(category,name_font,(anchor[0],anchor[1]-23),'ms'),(str(value),num_font,(start,anchor[1]+46),'ls'),(denom,den_font,(start+num_width+10,anchor[1]+43),'ls')]
        boxes=[]
        for txt,ff,pos,aa in items:
            box=draw.textbbox(xy(pos),txt,font=ff,anchor=aa)
            boxes.append([float(z/S) for z in box])
        label_meta.append({'category':category,'value':value,'denominator':total,'anchor':anchor.tolist(),'external':external,'ink_boxes':boxes})
        if external:
            # Leader terminates inside the genuinely small G cell.
            ay,ax=np.unravel_index(distance.argmax(),distance.shape)
            tip=np.array([CX-R+ax+.5,CY-R+ay+.5])
            points=[tip,np.array([1022.,393.]),np.array([1074.,393.])]
            draw.line([xy(p) for p in points],fill=INK,width=2*S,joint='curve')
            draw.ellipse(xy((tip[0]-4,tip[1]-4,tip[0]+4,tip[1]+4)),fill=INK)
            label_meta[-1]['leader_tip']=tip.tolist()
    # A quiet reading key uses the actual area encoding and the supplied unit.
    draw.line([xy((265,2007)),xy((1175,2007))],fill='#BEAA88',width=S)
    text(draw,(720,2020),'Area of each region = its count / 301',font('body',25),'#654E3D',anchor='mt')
    # Compact unit/series caption near the feathers, without attaching data to feathers.
    text(draw,(78,1948),'Synthetic records',font('labels',22),'#765941')
    text(draw,(78,1978),'Unit: records',font('body',21),'#765941')
    final=canvas.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
    final.save(ROOT/'final.png')
    # Checks measure raster geometry and actual occlusion, not just target labels.
    base_counts=np.bincount(labels[labels>=0],minlength=len(categories))
    occ=np.array(occlusion.resize((W,H),Image.Resampling.LANCZOS))[CY-R:CY+R,CX-R:CX+R]
    visible=(labels>=0)&(occ<128)
    visible_counts=np.bincount(labels[visible],minlength=len(categories))
    errors=[]
    for i,entry in enumerate(label_meta):
        if entry['external']:continue
        mismatch=0;checked=0
        for box in entry['ink_boxes']:
            x0,y0,x1,y1=box
            ix0=max(0,int(np.floor(x0-(CX-R))));ix1=min(2*R,int(np.ceil(x1-(CX-R))))
            iy0=max(0,int(np.floor(y0-(CY-R))));iy1=min(2*R,int(np.ceil(y1-(CY-R))))
            sub=labels[iy0:iy1,ix0:ix1]
            mismatch+=int((sub!=i).sum());checked+=sub.size
        entry['bbox_pixels_outside_own_region']=mismatch
        entry['bbox_pixels_checked']=checked
        if mismatch:errors.append(entry['category'])
    topology=[]
    edge=(xx*xx+yy*yy>=.993**2)&(labels>=0)
    for i,category in enumerate(categories):
        components=label(labels==i,np.ones((3,3)))[1]
        topology.append({'category':category,'components_8_connected':int(components),'touches_disk_rim':bool(np.any(edge&(labels==i)))})
    rows=[]
    for i,c in enumerate(categories):
        observed=float(base_counts[i]/base_counts.sum())
        exposed=float(visible_counts[i]/visible_counts.sum())
        target=values[i]/total
        rows.append({'category':c,'value':values[i],'target_share':target,'rendered_pixel_area':int(base_counts[i]),'rendered_share':observed,'absolute_error_percentage_points':abs(observed-target)*100,'visible_area_after_rope_occlusion':int(visible_counts[i]),'visible_share_after_rope_occlusion':exposed,'visible_error_percentage_points':abs(exposed-target)*100})
    report={'canvas':[W,H],'data_disk':{'cx':CX,'cy':CY,'radius':R,'pixel_area':int(base_counts.sum())},'continuous_geometry':geom,'measured_regions':rows,'topology':topology,'label_boxes_crossing_regions':errors,'labels':label_meta,'generated_asset_placements':placements,'frame_occluded_disk_pixels':int(((labels>=0)&~visible).sum()),'area_transform':'polar angle rotation theta -> theta + 0.86(1-r^2) + 0.30(1-r^2)^2; r unchanged; Jacobian 1','check_kind':'local computational self-check, not independent or human evaluation'}
    (ROOT/'artifacts/geometry_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    Image.fromarray((labels+1).astype('uint8'),'L').save(ROOT/'artifacts/region_ids.png')
    print(json.dumps({'size':[W,H],'maximum_raster_error_pp':max(q['absolute_error_percentage_points'] for q in rows),'maximum_visible_error_pp':max(q['visible_error_percentage_points'] for q in rows),'label_boxes_crossing_regions':errors,'frame_occluded_disk_pixels':report['frame_occluded_disk_pixels']},ensure_ascii=False))

if __name__=='__main__':main()
