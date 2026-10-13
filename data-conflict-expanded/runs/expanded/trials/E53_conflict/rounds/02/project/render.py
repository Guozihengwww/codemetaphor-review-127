"""Offline renderer. Run only through capture_render.py for experiment renders."""
from pathlib import Path
from functools import lru_cache
import json, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent
S = 3
R0, RMAX = 24.0, 166.0
INK = (23, 24, 24, 255)
RING = (82, 84, 84, 205)
CENTERS = [(505,640),(1270,670),(900,965),(335,1240),(1440,1270),(960,1492),(540,1700)]
HAND_XY, HAND_WIDTH = (770,1775), 565

@lru_cache(None)
def font(name, size):
    return ImageFont.truetype(str(ROOT / 'fonts' / (name+'.otf')), round(size*S))

def text(layer, xy, txt, size, face='sans', fill=INK, anchor='mm'):
    ImageDraw.Draw(layer).text((xy[0]*S,xy[1]*S),txt,font=font(face,size),fill=fill,anchor=anchor)

def line(layer, pts, color=INK, width=1):
    ImageDraw.Draw(layer).line([(round(x*S),round(y*S)) for x,y in pts],fill=color,width=max(1,round(width*S)))

def bezier(p0,p1,p2,p3,n=120):
    t=np.linspace(0,1,n)[:,None]
    return ((1-t)**3*np.array(p0)+3*(1-t)**2*t*np.array(p1)+3*(1-t)*t*t*np.array(p2)+t**3*np.array(p3)).tolist()

def path(layer, start, segments, color, width=1.45):
    points=[start]
    for p1,p2,p3 in segments:
        points+=bezier(points[-1],p1,p2,p3)[1:]
    line(layer,points,color,width)

def circle(layer, xy, r, fill=None, outline=None, width=1):
    x,y=xy
    ImageDraw.Draw(layer).ellipse(((x-r)*S,(y-r)*S,(x+r)*S,(y+r)*S),fill=fill,outline=outline,width=max(1,round(width*S)))

def curved_label(layer, center, label, radius=189, size=32):
    f=font('serif',size)
    # Character advances along the circular baseline; each glyph is tangent.
    advances=[f.getlength(c)/S for c in label]
    total=sum(advances)
    offset=-total/2
    for char, advance in zip(label,advances):
        theta=(offset+advance/2)/radius
        x=center[0]+radius*math.sin(theta)
        y=center[1]-radius*math.cos(theta)
        tile=Image.new('RGBA',(round(3*size*S),round(3*size*S)))
        d=ImageDraw.Draw(tile)
        d.text((tile.width/2,tile.height/2),char,font=f,fill=INK,anchor='ms')
        tile=tile.rotate(-math.degrees(theta),resample=Image.Resampling.BICUBIC)
        layer.alpha_composite(tile,(round(x*S-tile.width/2),round(y*S-tile.height/2)))
        offset+=advance

def paper(width,height):
    rng=np.random.default_rng(53)
    noise=rng.normal(0,0.90,(height,width,1))
    low=rng.normal(0,1.8,(60,50))
    lowim=Image.fromarray(np.clip(low+128,0,255).astype('uint8')).resize((width,height),Image.Resampling.BICUBIC)
    grain=(np.array(lowim).astype(float)-128)[:,:,None]*0.45
    arr=np.clip(np.array([249.0,248.0,244.5])[None,None,:]+noise+grain,0,255).astype('uint8')
    im=Image.fromarray(arr).convert('RGBA').resize((width*S,height*S),Image.Resampling.BICUBIC)
    d=ImageDraw.Draw(im)
    # Subtle fibers and a few pigment flecks along the paper edge.
    for _ in range(900):
        x=float(rng.uniform(0,width)); y=float(rng.uniform(0,height))
        length=float(rng.uniform(0.3,1.7)); angle=float(rng.uniform(0,math.pi))
        shade=int(rng.integers(198,235))
        d.line((x*S,y*S,(x+length*math.cos(angle))*S,(y+length*math.sin(angle))*S),fill=(shade,shade,shade-3,255),width=1)
    for side,color in [('left',(47,69,193,190)),('right',(237,140,25,190))]:
        for _ in range(150):
            x=float(min(rng.exponential(14),53))
            if side=='right': x=width-x
            y=float(rng.uniform(0,height)); r=float(rng.uniform(.25,1.25))
            d.ellipse(((x-r)*S,(y-r)*S,(x+r)*S,(y+r)*S),fill=color)
    return im

def main():
    inp=json.loads((ROOT/'input.json').read_text(encoding='utf-8'))
    brief=json.loads((ROOT/inp['brief']).read_text(encoding='utf-8'))
    data=json.loads((ROOT/inp['data']).read_text(encoding='utf-8'))
    core=json.loads((ROOT/inp['critical_metaphor_contract']).read_text(encoding='utf-8'))
    assert (ROOT/inp['reference']).is_file(), 'Frozen visual reference must remain available.'
    w,h=brief['width'],brief['height']
    cats=data['categories']; series=data['series'][0]; values=series['values']
    assert len(cats)==len(values)==len(CENTERS)==7
    assert data['synthetic'] and data['value_axis']=={'minimum':0,'maximum':50,'scale':'linear'}
    assert core['visible_title']==data['title']
    assert series['unit']=='percent'
    maxvalue=float(data['value_axis']['maximum'])
    radius=lambda value:R0+(RMAX-R0)*float(value)/maxvalue
    im=paper(w,h)
    d=ImageDraw.Draw(im)

    # Editorial title, explicit provenance and the exact metric role.
    d.rounded_rectangle((747*S,65*S,1053*S,105*S),radius=2*S,fill=INK)
    text(im,(900,85),data['visible_notice'],21,'sans-bold',(255,255,251,255))
    text(im,(900,175),'Increases in',91,'serif-bold')
    text(im,(900,270),'positive feelings',109,'serif-bold')
    text(im,(900,349),series['label']+' (%)',28,'sans')
    text(im,(900,390),'Multiple selections allowed · Shares need not sum to 100%',22,'sans',(83,85,83,255))

    hand=Image.open(ROOT/'assets/hand.png').convert('RGBA')
    scale=HAND_WIDTH/hand.width
    grip=(HAND_XY[0]+1278*scale,HAND_XY[1]+52*scale)
    starts=[(cx,cy+RMAX+67) for cx,cy in CENTERS]
    strings=[
        [((505,1030),(700,1025),(658,1240)),((610,1440),(680,1600),(790,1665)),((870,1674),(1007,1677),(1090,1671)),((1182,1707),(1240,1747),grip)],
        [((1270,1080),(1138,1180),(1170,1385)),((1192,1570),(1190,1740),grip)],
        [((900,1280),(1170,1200),(1200,1505)),((1220,1650),(1230,1740),grip)],
        [((330,1580),(210,1680),(293,1815)),((328,1865),(404,1876),(440,1877)),((528,1878),(620,1878),(720,1875)),((900,1870),(1040,1775),(1140,1748)),((1210,1725),(1265,1740),grip)],
        [((1455,1570),(1372,1660),grip)],
        [((960,1770),(1100,1720),grip)],
        [((530,2070),(815,1990),(982,1846)),((1121,1700),(1205,1705),grip)]
    ]
    for start,segments in zip(starts,strings):
        path(im,start,segments,(56,58,57,255),1.55)

    # Each group has the very same zero kernel, outer radius and five-point rings.
    geometries=[]
    for idx,(cat,value,center) in enumerate(zip(cats,values,CENTERS),1):
        cx,cy=center
        rr=radius(value)
        for tick in range(0,51,5):
            circle(im,center,radius(tick),outline=RING,width=1.05 if tick<50 else 1.3)
        curved_label(im,center,cat)
        # A restrained physical shadow is separate from the colored data disk.
        margin=math.ceil(rr*0.16+8)
        side=math.ceil(2*(rr+margin)*S)
        shadow=Image.new('RGBA',(side,side))
        sx=side/2+max(1.5,rr*.033)*S
        sy=side/2+max(2,rr*.044)*S
        ImageDraw.Draw(shadow).ellipse((sx-rr*S,sy-rr*S,sx+rr*S,sy+rr*S),fill=(22,23,27,65))
        shadow=shadow.filter(ImageFilter.GaussianBlur(max(1.3,rr*.031)*S))
        im.alpha_composite(shadow,(round(cx*S-side/2),round(cy*S-side/2)))

        # Analytical pixel coverage imposes the supplied radial mapping without
        # minimum-size adjustments, generated alpha assumptions, or integer radii.
        left=math.floor((cx-rr)*S)-2; top=math.floor((cy-rr)*S)-2
        right=math.ceil((cx+rr)*S)+2; bottom=math.ceil((cy+rr)*S)+2
        xx=(np.arange(left,right)+.5)/S-cx
        yy=(np.arange(top,bottom)+.5)/S-cy
        dist=np.hypot(xx[None,:],yy[:,None])
        coverage=np.clip((rr-dist)*S+.5,0,1)
        mask=Image.fromarray(np.round(coverage*255).astype('uint8'))
        texture=Image.open(ROOT/f'assets/sphere_{idx:02d}.png').convert('RGB')
        # Map normalized texture coordinates to the exact fractional-radius box.
        a=texture.width/(2*rr*S)
        tex=texture.transform((right-left,bottom-top),Image.Transform.AFFINE,
                (a,0,(left-(cx-rr)*S)*a,0,a,(top-(cy-rr)*S)*a),Image.Resampling.BICUBIC)
        tex.putalpha(mask)
        im.alpha_composite(tex,(left,top))
        # Small diagnostic coverage masks, not complete chart previews.
        # Store actual rendered coverage in its own tightly cropped pixel grid.
        mask.save(ROOT/f'artifacts/coverage_{idx:02d}.png')

        tagw=121 if float(value).is_integer() else 134
        tagtop=cy+RMAX+22
        d=ImageDraw.Draw(im)
        d.rectangle(((cx-tagw/2)*S,tagtop*S,(cx+tagw/2)*S,(tagtop+43)*S),fill=INK)
        # A tiny tapered attachment, as on the reference's hanging black labels.
        d.polygon([(cx-7)*S,(tagtop+43)*S,(cx+7)*S,(tagtop+43)*S,cx*S,(tagtop+51)*S],fill=INK)
        label=f'{value:g}%' if isinstance(value,(float,int)) else str(value)+'%'
        text(im,(cx,tagtop+22),label,29,'sans',(255,255,251,255))
        geometries.append({'index':idx,'category':cat,'value_percent':value,
              'center':list(center),'radius_px':rr,'core_radius_px':R0,
              'data_increment_px':rr-R0,'outer_scale_radius_px':RMAX,
              'tag':label,'coverage_path':f'artifacts/coverage_{idx:02d}.png',
              'coverage_scale':S,'coverage_canvas_origin':[left/S,top/S],
              'actual_alpha_area_radius_px':math.sqrt(float(coverage.sum())/(math.pi*S*S))})

    # Foreground hand hides the very ends of the threads at the pinch.
    hand=hand.resize((round(HAND_WIDTH*S),round(hand.height*scale*S)),Image.Resampling.LANCZOS)
    im.alpha_composite(hand,(round(HAND_XY[0]*S),round(HAND_XY[1]*S)))

    # A literal same-size quarter-circle key with the identical mapping.
    ox,oy=1482,1940
    text(im,(1530,1715),'RADIAL SCALE',19,'sans-bold')
    ld=ImageDraw.Draw(im)
    ld.pieslice(((ox-R0)*S,(oy-R0)*S,(ox+R0)*S,(oy+R0)*S),270,360,fill=(213,211,204,255))
    for tick in range(0,51,5):
        r=radius(tick)
        ld.arc(((ox-r)*S,(oy-r)*S,(ox+r)*S,(oy+r)*S),270,360,fill=RING,width=round(1.1*S))
        if tick%10==0:
            line(im,[(ox-7,oy-r),(ox,oy-r)],(85,87,83,255),1)
            text(im,(ox-13,oy-r),f'{tick}%',18,'sans',anchor='rm')
    line(im,[(ox,oy),(ox+RMAX,oy)],(150,151,145,255),.6)
    text(im,(1390,1980),'Equal 5-percentage-point steps.',18,'sans',anchor='lm')
    text(im,(1390,2010),'0% includes a fixed core.',18,'sans',anchor='lm')
    text(im,(1390,2040),'Radius encodes share; area does not.',18,'sans',anchor='lm')

    text(im,(90,2055),'A SHARED SCALE, SEVEN FEELINGS',18,'sans-bold',anchor='lm')
    text(im,(90,2086),'Source: supplied synthetic data.',18,'sans',fill=(83,85,83,255),anchor='lm')
    text(im,(90,2115),'Illustrative values, not population estimates.',18,'sans',fill=(83,85,83,255),anchor='lm')
    im=im.convert('RGB').resize((w,h),Image.Resampling.LANCZOS)
    im.save(ROOT/'final.png')
    (ROOT/'artifacts/render_geometry.json').write_text(json.dumps({
        'canvas':[w,h],'mapping':{'r0':R0,'R':RMAX,'value_min':0,'value_max':maxvalue,'formula':'r = r0 + (R-r0) * value / 50','area_encoding':False},
        'ring_ticks_percent':list(range(0,51,5)),'legend_origin':[ox,oy],
        'legend_radii_px':[radius(t) for t in range(0,51,5)],
        'labels':cats,'string_count':len(strings),'string_end':list(grip),
        'hand_origin':list(HAND_XY),'hand_width':HAND_WIDTH,'marks':geometries,
        'visible_text':[data['visible_notice'],'Increases in positive feelings',series['label']+' (%)','Multiple selections allowed · Shares need not sum to 100%']
    },ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':
    main()
