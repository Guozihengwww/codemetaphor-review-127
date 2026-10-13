#!/usr/bin/env python3
"""Offline audit-fee infographic. All data geometry and type are code controlled.
Run complete renders only through capture_render.py, as required by the case.
"""
from pathlib import Path
import json, math, io
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cairosvg

ROOT=Path(__file__).resolve().parent
S=2
INK='#142e46'
PALETTE=['#eba78d','#eac14f','#bce0c7','#99c9e8','#153b59']
CX,CY,R=770.,1080.,595.
Q=.98
INNER=.45
FONTS={'body':'body.ttf','bold':'bold.ttf','display':'display.otf','ledger':'ledger.otf'}

def load_json(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))

def font(size,kind='body'):
    return ImageFont.truetype(str(ROOT/'assets/fonts'/FONTS[kind]),round(size*S))

def write_json(name,obj):
    (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def px(v):return round(v*S)

def text(draw,xy,s,size,fill=INK,kind='body',anchor='lt'):
    draw.text((xy[0]*S,xy[1]*S),s,font=font(size,kind),fill=fill,anchor=anchor,stroke_width=0)

def line(draw,pts,fill,width=1):
    draw.line([(px(x),px(y)) for x,y in pts],fill=fill,width=max(1,px(width)),joint='curve')

def point(r,deg):
    a=math.radians(deg)
    return (r*math.cos(a),r*math.sin(a))

def sector_path(a,b):
    p0,p1=point(R,a),point(R,b)
    q1,q0=point(R*INNER,b),point(R*INNER,a)
    large=int(b-a>180)
    return f'M {p0[0]:.9f},{p0[1]:.9f} A {R},{R} 0 {large} 1 {p1[0]:.9f},{p1[1]:.9f} L {q1[0]:.9f},{q1[1]:.9f} A {R*INNER},{R*INNER} 0 {large} 0 {q0[0]:.9f},{q0[1]:.9f} Z'

def main():
    brief=load_json('brief.json');data=load_json('data.json');core=load_json('core.json');inp=load_json('input.json')
    # Keep the contract and all frozen inputs as local dependencies.
    assert inp['brief']=='brief.json' and inp['data']=='data.json'
    assert (ROOT/inp['reference']).is_file()
    assert '0.45' in core['data_carrying_relationship_zh'] and '0.98' in core['data_carrying_relationship_zh']
    assert data['synthetic'] and len(data['series'])==1
    cats=data['categories'];series=data['series'][0];values=series['values']
    assert len(cats)==len(values)==len(PALETTE)
    total=sum(values)
    assert total==data['series_totals'][series['id']]
    assert data['value_axis']['minimum']==0 and data['value_axis']['maximum']==total
    W,H=brief['width'],brief['height']
    (ROOT/'artifacts').mkdir(exist_ok=True)

    # Paper texture is deterministic. Background ledger figures use only this
    # case's synthetic numbers; there is no reference-image pixel dependency.
    yy,xx=np.mgrid[:H,:W]
    rng=np.random.default_rng(20261014)
    grain=rng.normal(0,.60,(H,W))
    shade=.7*np.sin(xx/290.)+.9*np.cos(yy/390.)
    base=np.empty((H,W,3),dtype=np.uint8)
    for k,col in enumerate((210,229,240)):
        base[:,:,k]=np.clip(col+grain+shade,0,255)
    paper=Image.fromarray(base,'RGB').resize((W*S,H*S),Image.Resampling.BICUBIC).convert('RGBA')
    ledger=Image.new('RGBA',paper.size)
    ld=ImageDraw.Draw(ledger)
    for j,y in enumerate(range(-30,H+100,103)):
        line(ld,[(-80,y),(W+100,y-28)],(39,87,116,84),1.5)
        row=(j+len(cats))%len(cats)
        for col,x in enumerate((80,466,860,1253)):
            val=values[(row+col)%len(values)]
            text(ld,(x,y+28-28*x/W),f'{val:,.2f}',35,(32,66,91,82),'ledger')
    for x in (-45,382,808,1228,1660):
        line(ld,[(x,-60),(x+32,H+30)],(39,87,116,64),1.1)
    paper.alpha_composite(ledger)

    # A dark photographic-style editorial header fades into the accounting sheet.
    z=np.clip((yy-210)/420,0,1)
    opacity=(1-z)**1.30*.985
    header=np.zeros((H,W,4),np.uint8)
    header[:,:,:3]=(9,33,57);header[:,:,3]=(255*opacity).astype(np.uint8)
    canvas=paper.copy()
    canvas.alpha_composite(Image.fromarray(header,'RGBA').resize(canvas.size,Image.Resampling.BICUBIC))

    # Soft contact shadow beneath the lens; frame/handle detail and texture are
    # generated pixels, while the lens's circular data boundary is exact.
    shadow=Image.new('RGBA',canvas.size)
    shd=ImageDraw.Draw(shadow)
    shd.ellipse((px(CX-R-25+17),px(CY-Q*(R+25)+29),px(CX+R+25+17),px(CY+Q*(R+25)+29)),fill=(10,38,57,95))
    shadow=shadow.filter(ImageFilter.GaussianBlur(px(21)))
    canvas.alpha_composite(shadow)

    # Enlarged, subdued ledger visible through the glass and the centre hole.
    lens=paper.resize((round(W*S*1.16),round(H*S*1.16)),Image.Resampling.BICUBIC)
    offx=round(CX*S*.16);offy=round(CY*S*.16)
    lens=lens.crop((offx,offy,offx+W*S,offy+H*S))
    wash=Image.new('RGBA',canvas.size,(226,240,246,176));lens.alpha_composite(wash)
    lensmask=Image.new('L',canvas.size)
    md=ImageDraw.Draw(lensmask)
    md.ellipse((px(CX-R),px(CY-Q*R),px(CX+R),px(CY+Q*R)),fill=255)
    canvas.paste(lens,(0,0),lensmask)

    records=[];paths=[];start=-90.
    for cat,v,color in zip(cats,values,PALETTE):
        sweep=360.*v/total;end=start+sweep
        path=sector_path(start,end)
        paths.append(f'<path d="{path}" fill="{color}"/>')
        records.append({'category':cat,'value':v,'unit':series['unit'],'share_percent':100.*v/total,'start_angle_deg':start,'end_angle_deg':end,'sweep_deg':sweep,'color':color,'path':path})
        start=end
    # No separating strokes: the 0.4% segment must retain its actual 1.44 degrees.
    svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><g transform="translate({CX} {CY}) scale(1 {Q})">'+''.join(paths)+'</g></svg>'
    (ROOT/'chart_geometry.svg').write_text(svg+'\n',encoding='utf-8')
    chart=Image.open(io.BytesIO(cairosvg.svg2png(bytestring=svg.encode('utf-8'),output_width=W*S,output_height=H*S))).convert('RGBA')
    canvas.alpha_composite(chart)

    # Subtle common optical shading, confined to the ring, never resizing slices.
    nx=(xx-CX)/R;ny=(yy-CY)/(Q*R);rr=np.hypot(nx,ny)
    aa=(np.clip((rr-.80)/.20,0,1)**2*20).astype(np.uint8)
    aa[(rr>=1)|(rr<=INNER)]=0
    opt=np.zeros((H,W,4),np.uint8);opt[:,:,:3]=(17,51,74);opt[:,:,3]=aa
    canvas.alpha_composite(Image.fromarray(opt,'RGBA').resize(canvas.size,Image.Resampling.BICUBIC))

    # Affine placement uses exactly the same 0.98 y projection as the chart.
    asset=Image.open(ROOT/'assets/magnifier_normalized.png').convert('RGBA')
    geo=load_json('assets/asset_geometry.json')['normalized_aperture']
    scale=R/geo['radius']
    A=1/(scale*S);B=1/(scale*Q*S)
    mapping=(A,0,geo['cx']-CX/scale,0,B,geo['cy']-CY/(scale*Q))
    object_layer=asset.transform(canvas.size,Image.Transform.AFFINE,mapping,Image.Resampling.BICUBIC)
    # Guard against any interpolation alpha reaching the quantitative region.
    # This full-resolution exact ellipse removes frame pixels inside the lens.
    oalpha=np.asarray(object_layer.getchannel('A')).copy()
    for y in range(canvas.height):
        vy=((y+.5)/S-CY)/(Q*R)
        if abs(vy)<1:
            half=R*math.sqrt(1-vy*vy)
            a0=max(0,math.ceil((CX-half)*S-.5));a1=min(W*S,math.floor((CX+half)*S-.5)+1)
            oalpha[y,a0:a1]=0
    object_layer.putalpha(Image.fromarray(oalpha,'L'))
    canvas.alpha_composite(object_layer)

    draw=ImageDraw.Draw(canvas)
    # Header, all exact semantic labels are code-rendered.
    notice=data['visible_notice']
    draw.rounded_rectangle((px(96),px(65),px(431),px(123)),radius=px(7),fill='#dbeafb')
    text(draw,(263.5,94),notice,25,'#0c2c46','bold','mm')
    text(draw,(1498,95),'AUDIT FIRMS',23,'#bcd0e2','bold','rm')
    title=data['title'].removeprefix('SYNTHETIC ')
    text(draw,(94,171),title,124,'#ffffff','display')
    text(draw,(99,329),f"{series['label']}  /  {series['unit']}",35,'#e5edf5','body')
    line(draw,[(98,390),(1500,390)],(192,217,234,150),1.2)

    label_evidence=[]
    for i,record in enumerate(records[:-1]):
        angle=(record['start_angle_deg']+record['end_angle_deg'])/2
        lr=R*.752
        dx,dy=point(lr,angle)
        tx,ty=CX+dx,CY+Q*dy
        # Maintain a three-line hierarchy within each data-bearing arc.
        text(draw,(tx,ty-68),record['category'],44,INK,'bold','mm')
        text(draw,(tx,ty+2),f"{record['value']:,}",81,INK,'display','mm')
        text(draw,(tx,ty+69),f"{record['share_percent']:.1f}%",38,INK,'body','mm')
        label_evidence.append({'category':record['category'],'anchor_x':tx,'anchor_y':ty,'angle_deg':angle,'radial_fraction':.752})

    small=records[-1]
    angle=(small['start_angle_deg']+small['end_angle_deg'])/2
    ux,uy=point(R*INNER+1,angle)
    # A neutral hairline identifies the actual inner tip without enlarging it.
    ex,ey=CX+ux,CY+Q*uy
    line(draw,[(ex,ey+1),(ex,876),(CX,876)],INK,1.7)
    text(draw,(CX,911),small['category'],31,INK,'bold','mm')
    text(draw,(CX,958),f"{small['value']:,}  /  {small['share_percent']:.1f}%",34,INK,'body','mm')
    line(draw,[(CX-125,1008),(CX+125,1008)],'#a1b8c7',1.25)
    text(draw,(CX,1063),'TOTAL AUDIT FEES',23,'#4c6579','bold','mm')
    text(draw,(CX,1142),f'{total:,}',94,INK,'display','mm')
    text(draw,(CX,1212),series['unit'],28,'#4c6579','body','mm')

    # Footer leaves the handle unobstructed and makes provenance explicit.
    footer=Image.new('RGBA',canvas.size)
    fd=ImageDraw.Draw(footer)
    fd.rounded_rectangle((px(76),px(1807),px(1122),px(1942)),radius=px(12),fill=(229,240,246,225))
    text(fd,(99,1833),'Each arc shows its share of total audit fees.',28,INK,'body')
    text(fd,(99,1885),'Source: supplied synthetic dataset. Firm labels are illustrative.',23,'#496477','body')
    canvas.alpha_composite(footer)

    canvas.convert('RGB').resize((W,H),Image.Resampling.LANCZOS).save(ROOT/'final.png')
    write_json('artifacts/geometry.json',{'canvas':[W,H],'centre':[CX,CY],'unprojected_outer_radius':R,'unprojected_inner_radius':R*INNER,'inner_outer_ratio':INNER,'vertical_projection':Q,'projected_outer_radii':[R,Q*R],'projected_inner_radii':[R*INNER,Q*R*INNER],'total':total,'series':series['label'],'unit':series['unit'],'direction':'clockwise','start_angle_deg':-90.,'segments':records,'label_positions':label_evidence,'small_segment_annotation':{'category':small['category'],'tip':[ex,ey],'label_centre':[CX,934]},'supersampling':S,'separating_stroke_width':0,'generated_asset':'assets/magnifier_normalized.png','asset_scale_x':scale,'asset_scale_y':scale*Q})
    print(f'Rendered {W}x{H}; {len(values)} categories; total {total:,} {series["unit"]}.')

if __name__=='__main__':main()
