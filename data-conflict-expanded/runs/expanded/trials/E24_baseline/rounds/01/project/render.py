#!/usr/bin/env python3
"""Offline raster composition; all data lengths share the same linear scale.
Run only through capture_render.py during the experiment.
"""
from pathlib import Path
import json
from decimal import Decimal
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent
S = 2
W, H = 1600, 2000
BASE = 1570.0
SCALE = 34.0
X0, STEP, TREE_W = 170.0, 60.0, 86.0
INK = (29, 37, 37)
MUTED = (69, 78, 76)
PAPER = (238, 236, 225)
ORANGE = (202, 77, 21)


def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))


def font(size, face='sans'):
    names={'sans':'Sans-Regular.otf','bold':'Sans-Bold.otf','serif':'Serif-Regular.otf','italic':'Serif-Italic.otf'}
    return ImageFont.truetype(str(ROOT/'assets/fonts'/names[face]),round(size*S))


def text(draw, xy, content, size=24, fill=INK, face='sans', anchor='lt', spacing=1):
    x,y=xy
    f=font(size,face)
    if '\n' not in content:
        draw.text((round(x*S),round(y*S)),content,font=f,fill=fill,anchor=anchor)
    else:
        for line in content.split('\n'):
            draw.text((round(x*S),round(y*S)),line,font=f,fill=fill,anchor=anchor)
            y+=size*spacing


def line(draw,coords,fill,width=1):
    draw.line([(round(x*S),round(y*S)) for x,y in coords],fill=fill,width=max(1,round(width*S)))


def rectangle(draw, box, fill, outline=None,width=1):
    draw.rectangle(tuple(round(v*S) for v in box),fill=fill,outline=outline,width=round(width*S))


def dashline(draw,y,x0=116,x1=1534):
    for x in np.arange(x0,x1,12):
        line(draw,[(x,y),(min(x+5,x1),y)],(105,114,110,85),.85)


def fmt(v):
    return f'{float(v):.1f}'


def make_background():
    smoke=Image.open(ROOT/'assets/raw_02.png').convert('RGB').resize((W,1640),Image.Resampling.LANCZOS)
    p=np.asarray(smoke).astype(np.float32)
    lum=p@np.array([.2126,.7152,.0722],dtype=np.float32)
    # Preserve generated smoke structure while keeping the data silhouette readable.
    y=np.linspace(0,1,1640)[:,None,None]
    tone=np.array([224,225,216])[None,None,:]*(1-y**1.2)+np.array([127,136,131])[None,None,:]*(y**1.2)
    texture=(lum[:,:,None]-155)*(.33+.19*y)
    bg=np.clip(tone+texture,0,255).astype(np.uint8)
    out=Image.new('RGB',(W,H),(31,41,41))
    out.paste(Image.fromarray(bg),(0,0))
    return out.resize((W*S,H*S),Image.Resampling.LANCZOS).convert('RGBA')


def make_tree(asset, width, height, fire_h, variant):
    src=Image.open(ROOT/asset).convert('RGBA')
    sprite=src.resize((width,height),Image.Resampling.LANCZOS)
    p=np.asarray(sprite).astype(np.float32)
    alpha=p[:,:,3]
    # Strengthen fractional edge coverage in width only; never add a data-length cap.
    alpha=np.clip(alpha*1.7,0,255)
    lum=p[:,:,:3]@np.array([.2126,.7152,.0722],dtype=np.float32)
    bark=np.clip(lum/100,0,1)
    rgb=np.zeros_like(p[:,:,:3])
    rgb[:]=np.array([11.,17.,17.])+(bark[:,:,None]*np.array([24.,25.,23.]))
    cut=height-fire_h
    fire=Image.open(ROOT/'assets/raw_02.png').convert('RGB')
    fw,fh=fire.size
    fx=int((variant*.173%1)*max(1,fw-240))
    fire=fire.crop((fx,int(fh*.63),min(fx+240,fw),fh)).resize((width,fire_h),Image.Resampling.LANCZOS)
    f=np.asarray(fire).astype(np.float32)
    # Warm material remains confined to the exact fire-related segment.
    u=np.linspace(0,1,fire_h)[:,None,None]
    ramp=np.array([244.,108.,20.])[None,None,:]*(1-u)+np.array([255.,228.,107.])[None,None,:]*u
    warmth=.56*f+.44*ramp
    warmth=warmth*(.80+.20*bark[cut:,:,None])
    rgb[cut:]=np.clip(warmth,0,255)
    out=np.dstack((np.clip(rgb,0,255),alpha)).astype(np.uint8)
    return Image.fromarray(out,'RGBA'),cut


def main():
    config=read('input.json')
    brief=read(config['brief']); data=read(config['data']); core=read(config['critical_metaphor_contract'])
    assert (brief['width'],brief['height'])==(W,H)
    assert data['value_axis']=={'minimum':0,'maximum':30,'scale':'linear'}
    assert 'Fire-related loss (million hectares)' in core['semantic_roles_zh']
    cats=data['categories']; ser={s['id']:s for s in data['series']}
    fire=ser['fire_related']['values']; other=ser['non_fire']['values']
    totals=[float(Decimal(str(a))+Decimal(str(b))) for a,b in zip(fire,other)]
    assert len(cats)==len(fire)==len(other)==23 and max(totals)<=30
    im=make_background();d=ImageDraw.Draw(im)
    # Editorial heading and visible synthetic-data status.
    rectangle(d,(83,57,316,95),ORANGE)
    text(d,(199.5,77),'SYNTHETIC DATA',21,(255,246,221),'bold','mm')
    text(d,(1514,64),f'{cats[0]}—{cats[-1]}',26,INK,'sans','rt')
    text(d,(1514,103),'Annual forest loss',22,MUTED,'sans','rt')
    text(d,(78,125),'Forest loss',130,INK,'serif')
    text(d,(82,258),'by cause',125,INK,'italic')
    text(d,(89,418),'Two sources of loss. One shared scale.',28,MUTED)
    text(d,(949,209),'Each tree represents one year.',26,INK,'bold')
    text(d,(949,252),'Tree height = total forest loss.\nThe two colored sections add up\nto the annual total.',25,MUTED,spacing=1.4)
    rectangle(d,(949,388,975,414),(20,25,25))
    text(d,(992,388),ser['non_fire']['label'],25,INK)
    rectangle(d,(949,433,975,459),(244,138,37))
    text(d,(992,433),ser['fire_related']['label'],25,INK)
    # One linear axis. All chart data occupy this scale without added crown height.
    text(d,(84,508),'MILLION HECTARES',20,MUTED,'bold')
    grid=Image.new('RGBA',im.size,(0,0,0,0));gd=ImageDraw.Draw(grid)
    for val in range(0,31,5):
        yy=BASE-SCALE*val
        if val:
            dashline(gd,yy)
        text(d,(96,yy),str(val),25,INK if val>=15 else PAPER,'sans','rm')
    im=Image.alpha_composite(im,grid)
    # Generated material itself carries both data segments.
    geoms=[]
    markdir=ROOT/'artifacts/marks';markdir.mkdir(parents=True,exist_ok=True)
    for i,(year,fv,nv,tv) in enumerate(zip(cats,fire,other,totals)):
        asset=f'assets/tree_{i%5+1:02d}.png'
        cx=X0+i*STEP
        ytop=BASE-SCALE*tv; boundary=BASE-SCALE*fv
        top=round(ytop*S); bottom=round(BASE*S); cut=round(boundary*S)
        left=round((cx-TREE_W/2)*S); width=round(TREE_W*S)
        tree,localcut=make_tree(asset,width,bottom-top,bottom-cut,i)
        # Alternate the five source trees; reversal changes appearance, never height.
        if i%2:tree=ImageOps.mirror(tree)
        im.alpha_composite(tree,(left,top))
        # Diagnostic source-sized mark; no separate complete-chart preview.
        small=tree.resize((round(TREE_W),round((bottom-top)/S)),Image.Resampling.LANCZOS)
        small.save(markdir/f'{year}.png')
        arr=np.asarray(small)
        mask=arr[:,:,3]>=96
        yy,xx=np.where(mask)
        geoms.append({'year':year,'asset':asset,'fire':fv,'non_fire':nv,'total':tv,
            'center_x':cx,'left_x':left/S,'width_px':width/S,
            'top_y':top/S,'boundary_y':cut/S,'baseline_y':bottom/S,
            'total_height_px':(bottom-top)/S,'fire_height_px':(bottom-cut)/S,
            'non_fire_height_px':(cut-top)/S,
            'sprite_alpha96_bbox':[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],
            'sprite_file':f'artifacts/marks/{year}.png'})
    d=ImageDraw.Draw(im)
    # Total labels are attached to actual tree tips, with a consistent short leader.
    for i in (0,3,11,15,18,22):
        g=geoms[i]; cx=g['center_x']; yy=g['top_y']
        line(d,[(cx,yy-8),(cx,yy-23)],INK,.9)
        text(d,(cx,yy-29),fmt(g['total']),25,INK,'bold','mb')
    # Peak callout: derived from current data, not from the design reference.
    peak=int(np.argmax(totals))
    text(d,(X0+peak*STEP,481),'HIGHEST TOTAL',17,MUTED,'bold','mt')
    text(d,(X0+peak*STEP,509),cats[peak],18,MUTED,'sans','mt')
    # Reading aid in the open region over the shorter trees.
    line(d,[(170,715),(206,715)],ORANGE,3)
    text(d,(170,738),'A forest of annual totals',39,INK,'serif')
    text(d,(170,794),'Read the height, not the width.\nOrange below = fire-related loss.\nCharcoal above = non-fire loss.',24,MUTED,spacing=1.45)
    # Axis baseline, category ticks and selected year labels.
    line(d,[(116,BASE),(1534,BASE)],PAPER,1.5)
    for i,c in enumerate(cats):
        cx=X0+i*STEP
        major=i in (0,4,9,14,19,22)
        line(d,[(cx,BASE),(cx,BASE+(13 if major else 6))],PAPER,1)
        if major:text(d,(cx,BASE+24),c,25,PAPER,'sans','mt')
    # Native numeric table: every year and both supplied series remain legible.
    rectangle(d,(0,1640,W,H),(31,41,41))
    text(d,(83,1665),'ANNUAL VALUES',19,(220,224,211),'bold')
    text(d,(1514,1665),'million hectares',20,(195,205,196),'sans','rt')
    y_year,y_fire,y_non,y_total=1715,1760,1805,1850
    for label,yy,color in [('YEAR',y_year,(171,185,177)),('FIRE',y_fire,(255,176,80)),('NON-FIRE',y_non,(220,224,211)),('TOTAL',y_total,(243,238,220))]:
        text(d,(83,yy),label,15,color,'bold','lm')
    for i,(year,fv,nv,tv) in enumerate(zip(cats,fire,other,totals)):
        cx=X0+i*STEP
        if i%2==0:
            rectangle(d,(cx-27,1697,cx+27,1872),(37,48,47))
        for content,yy,sz,color,face in [(year,y_year,18,(171,185,177),'sans'),(fmt(fv),y_fire,21,(255,176,80),'bold'),(fmt(nv),y_non,21,(220,224,211),'sans'),(fmt(tv),y_total,21,(243,238,220),'bold')]:
            text(d,(cx,yy),content,sz,color,face,'mm')
    line(d,[(83,1895),(1514,1895)],(80,95,87),1)
    text(d,(83,1921),'Source: supplied synthetic study data · 23 years · values are not real-world estimates.',20,(203,212,201))
    text(d,(83,1955),'Encoding: shared 0–30 linear scale; total = fire-related + non-fire loss. All values shown to 0.1 million hectares.',18,(171,185,177))
    im.convert('RGB').resize((W,H),Image.Resampling.LANCZOS).save(ROOT/'final.png')
    payload={'canvas':[W,H],'supersampling':S,'axis_minimum':0,'axis_maximum':30,'pixels_per_million_hectares':SCALE,'baseline':BASE,'marks':geoms,'data_labels':{'years':cats,'fire_related':[fmt(v) for v in fire],'non_fire':[fmt(v) for v in other],'total':[fmt(v) for v in totals]},'reference_pixels_used':False}
    (ROOT/'artifacts/geometry.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Rendered final.png (1600 × 2000); 23 generated-tree marks on one linear scale.')

if __name__=='__main__':main()
