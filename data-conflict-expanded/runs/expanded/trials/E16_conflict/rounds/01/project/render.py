"""Regenerate the infographic from frozen local data and generated material.

Run from this directory: PYTHON render.py
Experiment render attempts must be launched through capture_render.py.
No network calls or model calls are made here.
"""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'artifacts'
FONTS = ROOT / 'assets/fonts'
BG = '#f3f0e9'
INK = '#222c2d'
MUTED = '#63706f'
RULE = '#c5c8bf'
PALETTE = ['#43566e','#6c8ba0','#92abc4','#d08b77','#af6556','#91a38a',
           '#47685d','#779685','#734951','#a77980','#c68151','#a44735']

def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def rgb(color):
    return np.array([int(color[i:i+2],16) for i in (1,3,5)],dtype=float)

def font(size, kind='sans'):
    return ImageFont.truetype(str(FONTS / (kind+'.ttc' if kind=='cjk' else kind+'.otf')),size)

def main():
    ART.mkdir(exist_ok=True)
    inputs = json.loads((ROOT/'input.json').read_text(encoding='utf-8'))
    brief = json.loads((ROOT/inputs['brief']).read_text(encoding='utf-8'))
    data = json.loads((ROOT/inputs['data']).read_text(encoding='utf-8'))
    core = json.loads((ROOT/inputs['critical_metaphor_contract']).read_text(encoding='utf-8'))
    assert (ROOT/inputs['reference']).is_file()  # available reference; never sampled
    assert data['topic_zh'] == core['topic_zh']
    W,H = brief['width'],brief['height']
    values = np.asarray(data['series'][0]['values'],dtype=float)
    categories = data['categories']
    assert len(categories)==len(values)==len(PALETTE)
    assert abs(values.sum()-100)<1e-7
    assert data['value_axis']=={'minimum':0,'maximum':100,'scale':'linear'}

    # One projection, applied identically to the photographic material and masks.
    projected_size=(900,1350)
    origin=(510,415)
    image = Image.open(ROOT/'assets/garment.png').convert('RGBA').resize(projected_size,Image.Resampling.LANCZOS)
    selector = Image.open(ROOT/'assets/fabric_selector.png').convert('L').resize(projected_size,Image.Resampling.LANCZOS)
    photo = np.asarray(image).astype(float)
    alpha = photo[:,:,3]/255
    fabric_selection = np.asarray(selector).astype(float)/255
    visible = alpha*fabric_selection
    ph,pw = visible.shape
    yy,xx = np.indices((ph,pw),dtype=float)

    # A gently undulating seam field. Its values are ordered top to bottom.
    # Weighted quantiles of final projected visible pixels, NOT equal band heights,
    # allocate the union of sleeves and torso with every visible pixel counted once.
    seam_field=yy+7*np.sin(xx/88)+3*np.sin(xx/25)
    valid = visible.ravel()>0
    sorted_indices=np.flatnonzero(valid)[np.argsort(seam_field.ravel()[valid],kind='stable')]
    weights=visible.ravel()[sorted_indices]
    cumulative=np.cumsum(weights)
    total=float(cumulative[-1])
    target=np.cumsum(values)[:-1]/100*total
    cuts=np.searchsorted(cumulative,target)
    thresholds=seam_field.ravel()[sorted_indices[cuts]]
    ids=np.searchsorted(thresholds,seam_field,side='right').astype('uint8')

    # Color is applied to the generated cotton material, retaining actual fabric
    # shadows, weave, stitching, buttons, cuffs, placket, overlaps and hem.
    lum=photo[:,:,:3]@np.array([0.2126,0.7152,0.0722])
    shade=np.clip(lum/174,0.10,1.50)
    colored=np.zeros((ph,pw,3),dtype=float)
    for k,color in enumerate(PALETTE):
        pattern=np.ones_like(lum)
        curved_x=xx+2.5*np.sin(yy/77)+2*np.sin(xx/41)
        if k in (1,2,3,4):
            period={1:9,2:7,3:10,4:8}[k]
            stripe=np.maximum(0,1-np.abs((curved_x+period/2)%period-period/2)/0.75)
            pattern+=stripe*({1:.15,2:.22,3:.24,4:.18}[k])-.025
        if k in (6,8,9):
            v=np.exp(-(((curved_x+14)%28-14)/.9)**2)
            h=np.exp(-(((yy+12)%24-12)/.9)**2)
            pattern+=.15*v+.13*h-.018
        if k in (0,5,7,10,11):
            pattern+=.015*np.sin((xx+yy)*1.6)
        factor=shade*pattern
        colored[ids==k]=(rgb(color)[None,None,:]*factor[:,:,None])[ids==k]
    for threshold in thresholds:
        distance=seam_field-threshold
        # Fine tonal stitching remains a part of the same category's surface.
        seam_factor=1-.20*np.exp(-((distance-.50)/.56)**2)+.075*np.exp(-((distance-1.6)/.48)**2)
        colored*=seam_factor[:,:,None]
    colored=np.clip(colored,0,255)
    recolored=photo.copy()
    recolored[:,:,:3]=photo[:,:,:3]*(1-fabric_selection[:,:,None])+colored*fabric_selection[:,:,None]
    garment=Image.fromarray(np.rint(recolored).astype('uint8'),'RGBA')

    canvas=Image.new('RGB',(W,H),BG)
    # A restrained, offset contact shadow is non-data and does not occlude fabric.
    shadow=Image.new('L',(W,H),0)
    shadow.paste(Image.fromarray(np.uint8(alpha*29)),(origin[0]+14,origin[1]+10))
    shadow=shadow.filter(ImageFilter.GaussianBlur(15))
    shadow_layer=Image.new('RGB',(W,H),'#323c38')
    canvas.paste(shadow_layer,(0,0),shadow)
    canvas.paste(garment,origin,garment)
    draw=ImageDraw.Draw(canvas)
    text_boxes=[]

    def text(x,y,value,size=24,kind='sans',fill=INK,anchor='lt',role='text'):
        f=font(size,kind)
        draw.text((x,y),value,font=f,fill=fill,anchor=anchor,stroke_width=0)
        box=draw.textbbox((x,y),value,font=f,anchor=anchor)
        text_boxes.append({'text':value,'role':role,'bbox':list(box)})
        return box

    def tracked(x,y,value,size=16,spacing=2,fill=MUTED):
        f=font(size,'sans_bold')
        left=x
        for ch in value:
            draw.text((x,y),ch,font=f,fill=fill,anchor='lt')
            x+=draw.textlength(ch,font=f)+spacing
        text_boxes.append({'text':value,'role':'heading','bbox':[left,y,x,y+size]})

    # Editorial title and conspicuous synthetic-data notice.
    draw.rounded_rectangle((78,64,336,108),radius=5,fill=INK)
    text(98,77,data['visible_notice'],19,'sans_bold','#f3f0e9',role='synthetic_notice')
    tracked(1108,79,'12 CATEGORIES',15,1.8)
    text(78,146,'Occupational composition',91,'serif',role='title')
    text(81,261,data['topic_zh'],28,'cjk',role='series_topic')
    text(82,312,'Population composition  /  Share of total (%)',24,'sans',MUTED,role='metric_unit')
    draw.line((80,371,1380,371),fill=RULE,width=1)
    tracked(82,420,'OCCUPATIONAL CATEGORY',15,1.7)
    tracked(432,420,'SHARE',15,1.7)
    draw.line((82,449,505,449),fill=RULE,width=1)
    text(82,478,'One garment, twelve parts.',27,'serif_italic',INK)
    text(82,521,'Panel area shows each category’s',21,'sans',MUTED)
    text(82,550,'share of the 100% total.',21,'sans',MUTED)

    # Locate leaders on actual visible color masks; use area-weighted medians.
    label_y=[]
    for k in range(len(values)):
        rowweights=(visible*(ids==k)).sum(axis=1)
        label_y.append(float(np.searchsorted(np.cumsum(rowweights),rowweights.sum()/2)+origin[1]))
    ideal=np.array(label_y)
    min_gap=80.0
    constraints=[{'type':'ineq','fun':lambda z:np.diff(z)-min_gap}]
    solution=minimize(lambda z:np.square(z-ideal).sum(),np.linspace(ideal[0],1724,len(ideal)),
                      method='SLSQP',bounds=[(660,1730)]*len(ideal),constraints=constraints,
                      options={'ftol':1e-8,'maxiter':1000})
    centers=solution.x if solution.success else np.linspace(720,1715,12)
    label_records=[]
    for k,(name,v,color,cy) in enumerate(zip(categories,values,PALETTE,centers)):
        parent,*sub=name.split(' / ')
        wrapped={
            'Higher managerial and professional':['Higher managerial','and professional'],
            'Lower managerial and professional':['Lower managerial','and professional'],
            'Intermediate occupations':['Intermediate occupations'],
            'Small employers and own account':['Small employers','and own account'],
            'Lower supervisory and technical':['Lower supervisory','and technical'],
            'Semi routine occupations':['Semi routine occupations'],
            'Routine occupations':['Routine occupations'],
            'Long term unemployed':['Long term unemployed'],
        }[parent]
        total_h=len(wrapped)*25+(22 if sub else 0)
        ty=round(cy-total_h/2+2)
        draw.rounded_rectangle((82,ty+1,88,ty+total_h-5),radius=2,fill=color)
        for j,line in enumerate(wrapped):
            text(102,ty+j*25,line,23,'sans',INK,role='category')
        if sub:
            text(103,ty+len(wrapped)*25+2,sub[0],17,'sans',MUTED,role='subcategory')
        label=f'{v:g}%'
        text(501,round(cy+1),label,31,'serif',INK,anchor='rm',role='value')
        ay=int(round(ideal[k]-origin[1]))
        in_band=(ids[ay]==k)&(visible[ay]>.98)
        xs=np.flatnonzero(in_band)
        if not len(xs):
            raise ValueError('No visible fabric for leader '+str(k))
        ax=int(xs[0])+origin[0]+15
        ay_canvas=ay+origin[1]
        # All targets follow input order, so the ordered leaders never cross.
        elbow=min(ax-18,554)
        draw.line([(515,round(cy)),(elbow,round(cy)),(ax-12,ay_canvas),(ax,ay_canvas)],fill='#89918b',width=1)
        draw.ellipse((ax-3,ay_canvas-3,ax+3,ay_canvas+3),fill=color,outline=BG,width=1)
        label_records.append({'index':k,'category':name,'display_value':label,
                              'label_y':round(float(cy),3),'leader_target':[ax,ay_canvas],
                              'target_is_category':int(ids[ay,ax-origin[0]])==k})

    # Reading key; no secondary chart or invented statistical provenance.
    draw.line((80,1796,1380,1796),fill=RULE,width=1)
    tracked(82,1821,'READ THE FABRIC',15,1.8)
    text(82,1855,'Area, not stripe height, represents the percentage.',21,'sans',INK)
    text(82,1885,'Collar and hanger are excluded. Overlaps count once.',20,'sans',MUTED)
    tracked(855,1821,'SYNTHETIC PARTITIONS',15,1.8)
    text(855,1855,'Subcategories are mutually exclusive',20,'sans',INK)
    text(855,1885,'parts of their named parent. Not a survey.',20,'sans',MUTED)
    draw.line((80,1930,1380,1930),fill=RULE,width=1)
    text(82,1951,'SOURCE  Local synthetic dataset',16,'sans',MUTED)
    text(1380,1951,'12 categories  ·  Total 100%',16,'sans',MUTED,anchor='rt')

    canvas.save(ROOT/'final.png',optimize=True)
    index_path=ROOT/'rounds/index.json'
    if index_path.exists():
        index=json.loads(index_path.read_text(encoding='utf-8'))
        if index:
            canvas.save(ART/f"candidate_{index[-1]['number']:02d}.png",optimize=True)

    # Save actual projected support masks for post-render area and binding checks.
    full_visible=np.zeros((H,W),dtype=np.uint8)
    full_ids=np.zeros((H,W),dtype=np.uint8)
    oy,ox=origin[1],origin[0]
    full_visible[oy:oy+ph,ox:ox+pw]=np.rint(visible*255).astype('uint8')
    full_ids[oy:oy+ph,ox:ox+pw]=np.where(visible>0,ids+1,0)
    Image.fromarray(full_visible).save(ART/'visible_fabric_mask.png')
    Image.fromarray(full_ids).save(ART/'category_id_mask.png')
    measured=[]
    for k,v in enumerate(values):
        actual=float(visible[ids==k].sum())
        ys,xs=np.where((ids==k)&(visible>.5))
        measured.append({'category':categories[k],'target_percent':float(v),
                         'visible_area_px':round(actual,4),'measured_percent':round(100*actual/total,6),
                         'absolute_error_percentage_points':round(abs(100*actual/total-v),6),
                         'visible_bbox':[int(xs.min()+ox),int(ys.min()+oy),int(xs.max()+ox),int(ys.max()+oy)]})
    save_json(ART/'render_geometry.json',{
        'canvas':[W,H],'projected_size':list(projected_size),'origin':list(origin),
        'visible_union_area_px':round(total,6),'area_basis':'final projected alpha-weighted visible fabric union',
        'thresholds_local':thresholds.tolist(),'categories':measured,
        'label_records':label_records,'text_boxes':text_boxes,
        'label_minimum_spacing_px':float(np.min(np.diff(centers))),
        'layout_optimizer_success':bool(solution.success),
        'generated_material':'assets/raw_01.png','reference_pixels_used':False})
    print(json.dumps({'canvas':[W,H],'visible_fabric_area_px':round(total,2),
                      'max_area_error_pp':max(r['absolute_error_percentage_points'] for r in measured),
                      'all_leaders_bound':all(r['target_is_category'] for r in label_records)},ensure_ascii=False))

if __name__=='__main__':
    main()
