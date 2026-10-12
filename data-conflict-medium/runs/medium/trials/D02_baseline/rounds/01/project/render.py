"""Rebuild final.png from the frozen data and local generated wood material.
Run from this project: PYTHON render.py. No network or model calls.
"""
from pathlib import Path
import json, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT=Path(__file__).resolve().parent
S=2
W,H=1685,2000
CX,CY,R,SY=685.,1040.,575.,.96
START=-190.
PALETTE=['#8980bd','#8ca363','#e79961','#8d96b7','#cb754f',
         '#506d95','#88b8c4','#e1988c','#86a85b','#bf6d55',
         '#dfb35f','#879b75','#e6c57f','#c48b98','#528c88',
         '#b4c778','#bc996b']
INK='#173c38'; MUTED='#526c65'; LINE='#6d847a'
LABEL_BOXES=[]; LEADERS=[]

def rgb(h):return tuple(int(h[i:i+2],16) for i in (1,3,5))
def font(name,size):return ImageFont.truetype(str(ROOT/'assets/fonts'/name),round(size*S))
F={
 'title':font('Title-Bold.otf',101),
 'cn':font('Title-Chinese.otf',90),
 'sub':font('Label-Regular.ttf',29),
 'small':font('Label-Regular.ttf',23),
 'kicker':font('Label-Bold.ttf',23),
 'name':font('Label-Regular.ttf',29),
 'value':font('Label-Bold.ttf',35),
 'bigvalue':font('Label-Bold.ttf',42),
 'foot':font('Label-Regular.ttf',24),
}

def txt(draw,xy,text,key,fill=INK,anchor='lt',record=None):
    xy2=tuple(v*S for v in xy)
    draw.text(xy2,text,font=F[key],fill=fill,anchor=anchor)
    b=draw.textbbox(xy2,text,font=F[key],anchor=anchor)
    if record is not None:LABEL_BOXES.append({'category':record,'text':text,'bbox':[v/S for v in b]})

def point(angle,r=R):
    a=math.radians(angle)
    return (CX+r*math.cos(a),CY+SY*r*math.sin(a))

def bezier(p0,p1,p2,p3,n=140):
    t=np.linspace(0,1,n)[:,None]
    return (1-t)**3*np.array(p0)+3*(1-t)**2*t*np.array(p1)+3*(1-t)*t*t*np.array(p2)+t**3*np.array(p3)

def leader(draw,i,mid,endpoint,controls=None):
    p0=point(mid,R-4)
    if controls is None:
        controls=(point(mid,R+112),(endpoint[0]-24,endpoint[1]))
    path=bezier(p0,*controls,endpoint)
    # A fine pale halo keeps lines readable over the rough bark.
    pp=[(round(x*S),round(y*S)) for x,y in path]
    draw.line(pp,fill='#dae9df',width=round(4*S),joint='curve')
    c=tuple(round(v*.52+52*.48) for v in rgb(PALETTE[i]))
    draw.line(pp,fill=c,width=round(1.6*S),joint='curve')
    x,y=p0
    draw.ellipse(((x-4.2)*S,(y-4.2)*S,(x+4.2)*S,(y+4.2)*S),fill=PALETTE[i],outline='#eef1e5',width=S)
    x,y=endpoint
    draw.ellipse(((x-2.5)*S,(y-2.5)*S,(x+2.5)*S,(y+2.5)*S),fill=c)
    LEADERS.append({'category_index':i,'mid_angle_unprojected_deg':mid,
                    'anchor':list(p0),'endpoint':list(endpoint),'path':path.round(4).tolist()})

def main():
    data=json.loads((ROOT/'data.json').read_text(encoding='utf-8'))
    brief=json.loads((ROOT/'brief.json').read_text(encoding='utf-8'))
    assert (W,H)==(brief['width'],brief['height'])
    cats=data['categories'];values=np.array(data['series'][0]['values'],dtype=float)
    assert len(cats)==len(values)==len(PALETTE) and abs(values.sum()-100)<1e-8
    edges=np.r_[0,np.cumsum(values)]*3.6
    mids=START+(edges[:-1]+edges[1:])/2
    rng=np.random.default_rng(812)
    yy,xx=np.mgrid[:H,:W]
    glow=np.exp(-(((xx-W*.37)/(W*.78))**2+((yy-H*.46)/(H*.7))**2))
    grain=rng.normal(0,.85,(H,W))
    base=np.array([207,228,220])[None,None,:]+glow[:,:,None]*np.array([17,13,12])+grain[:,:,None]
    im=Image.fromarray(np.uint8(np.clip(base,0,255)),'RGB').resize((W*S,H*S),Image.Resampling.BICUBIC).convert('RGBA')
    # One normalized material, one face radius, one affine projection.
    material=Image.open(ROOT/'assets/wood_normalized.png').convert('RGBA')
    side=round(material.width*R/650*S)
    mh=round(side*SY)
    body=material.resize((side,mh),Image.Resampling.LANCZOS)
    bx=round(CX*S-side/2);by=round(CY*S-mh/2)
    arr=np.array(body,dtype=np.float32)
    py,px=np.mgrid[:mh,:side].astype(np.float32)
    dx=(px+.5+bx)/S-CX;dy=((py+.5+by)/S-CY)/SY
    radius=np.hypot(dx,dy)
    theta=np.mod(np.degrees(np.arctan2(dy,dx))-START,360.)
    sector=np.searchsorted(edges[1:],theta,side='right').clip(0,len(values)-1)
    face=radius<=R
    luminance=arr[:,:,0]*.2126+arr[:,:,1]*.7152+arr[:,:,2]*.0722
    shade=np.clip((luminance/201.)**1.4,.24,1.2)
    colors=np.array([rgb(c) for c in PALETTE],dtype=np.float32)
    stained=np.clip(colors[sector]*shade[:,:,None],0,255)
    arr[face,:3]=stained[face]
    arr[face,3]=255
    body=Image.fromarray(np.uint8(arr),'RGBA')
    # The shadow uses only the object's alpha, and never contributes to data area.
    shadow=Image.new('RGBA',im.size,(0,0,0,0))
    shadow_alpha=Image.new('L',im.size,0)
    shadow_alpha.paste(body.getchannel('A'),(bx+10*S,by+18*S))
    shadow_alpha=shadow_alpha.filter(ImageFilter.GaussianBlur(18*S)).point(lambda a:round(a*.24))
    shadow.putalpha(shadow_alpha)
    im=Image.alpha_composite(im,shadow)
    im.alpha_composite(body,(bx,by))
    draw=ImageDraw.Draw(im)
    # Header hierarchy follows the visual reference; content follows data.json.
    txt(draw,(86,75),data['visible_notice'],'kicker')
    txt(draw,(1599,75),'FOREST AREA  /  SHARE','small',anchor='rt')
    draw.line((86*S,119*S,1599*S,119*S),fill='#8ca49a',width=S)
    txt(draw,(83,150),'Synthetic','title')
    title_en_width=draw.textlength('Synthetic',font=F['title'])/S
    txt(draw,(83+title_en_width+29,157),'木材切面','cn')
    txt(draw,(88,277),data['series'][0]['label'],'sub')
    txt(draw,(1599,280),'Unit: percent (%)','small',anchor='rt')
    # Small boundaries preserve continuous rings. No exploded slices or gaps.
    for angle in START+edges[:-1]:
        p=point(float(angle),R-.3)
        draw.line((CX*S,CY*S,p[0]*S,p[1]*S),fill=(56,67,53,92),width=1)
    for i in range(len(cats)):
        if 3<=i<=15:
            ey=820+(i-3)*73
            leader(draw,i,float(mids[i]),(1395,ey))
            txt(draw,(1425,ey-28),cats[i],'name',record=cats[i])
            txt(draw,(1425,ey+10),f'{values[i]:.1f}%','value',record=cats[i])
    # Broad sectors receive airy peripheral callouts above and below the wood.
    leader(draw,0,float(mids[0]),(92,735),controls=(point(float(mids[0]),R+62),(112,739)))
    txt(draw,(84,637),cats[0],'name',record=cats[0])
    txt(draw,(82,682),f'{values[0]:.1f}%','bigvalue',record=cats[0])
    leader(draw,1,float(mids[1]),(648,434),controls=(point(float(mids[1]),R+37),(648,452)))
    txt(draw,(648,344),cats[1],'name',anchor='mt',record=cats[1])
    txt(draw,(648,385),f'{values[1]:.1f}%','bigvalue',anchor='mt',record=cats[1])
    leader(draw,2,float(mids[2]),(1106,458),controls=(point(float(mids[2]),R+68),(1088,479)))
    txt(draw,(1106,366),cats[2],'name',anchor='mt',record=cats[2])
    txt(draw,(1106,407),f'{values[2]:.1f}%','bigvalue',anchor='mt',record=cats[2])
    leader(draw,16,float(mids[16]),(281,1665),controls=(point(float(mids[16]),R+72),(306,1648)))
    txt(draw,(277,1700),cats[16],'name',anchor='mt',record=cats[16])
    txt(draw,(277,1744),f'{values[16]:.1f}%','bigvalue',anchor='mt',record=cats[16])
    # A quiet methodology footer; it is not an additional data series.
    draw.line((86*S,1844*S,1599*S,1844*S),fill='#8ca49a',width=S)
    txt(draw,(88,1872),'17 categories  ·  100.0% total','foot')
    txt(draw,(1599,1872),'Clockwise in the supplied order','foot',anchor='rt')
    txt(draw,(88,1922),'Synthetic values. Colored wood area represents share; bark is excluded.','small',fill=MUTED)
    im=im.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
    im.save(ROOT/'final.png')
    # Geometry and label evidence records the raster actually used for compositing.
    (ROOT/'artifacts').mkdir(exist_ok=True)
    counts=np.bincount(sector[face].ravel(),minlength=len(values))
    sectors=[]
    for i,c in enumerate(cats):
        sectors.append({'category':c,'value':float(values[i]),'label':f'{values[i]:.1f}%',
          'start_unprojected_degrees':float(START+edges[i]),'end_unprojected_degrees':float(START+edges[i+1]),
          'raster_face_pixels':int(counts[i]),'raster_share_percent':float(counts[i]/counts.sum()*100),
          'color':PALETTE[i]})
    geometry={'canvas':[W,H],'center':[CX,CY],'radius_unprojected':R,'vertical_scale':SY,
        'data_semiaxes':[R,R*SY],'direction':'clockwise','start_angle_degrees':START,
        'surface_source':'assets/wood_normalized.png','sectors':sectors,
        'label_boxes':LABEL_BOXES,'leaders':LEADERS}
    (ROOT/'artifacts/render_geometry.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Saved final.png at 1685 x 2000; 17 stained wood sectors and 17 peripheral labels.')

if __name__=='__main__':main()
