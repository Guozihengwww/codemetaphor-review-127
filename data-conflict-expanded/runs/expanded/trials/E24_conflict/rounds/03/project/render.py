#!/usr/bin/env python3
"""Offline data-driven forest-loss infographic. Run through capture_render.py."""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.ndimage import maximum_filter1d

ROOT = Path(__file__).resolve().parent
S = 2

def read_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))

def font(name, size):
    return ImageFont.truetype(str(ROOT / 'assets/fonts' / name), round(size*S))

FONTS = {
    'sans': 'Sans-Regular.ttf', 'bold': 'Sans-Bold.ttf',
    'serif': 'Serif-Regular.otf', 'italic': 'Serif-Italic.otf'
}

def main():
    inputs = read_json('input.json')
    brief = read_json(inputs['brief'])
    data = read_json(inputs['data'])
    core = read_json(inputs['critical_metaphor_contract'])
    assert data['synthetic'] and core['semantic_roles_zh']
    W, H = brief['width'], brief['height']
    assert (W, H) == (1600, 2000)
    years = data['categories']
    series = {x['id']: x for x in data['series']}
    fire = np.asarray(series['fire_related']['values'], dtype=float)
    nonfire = np.asarray(series['non_fire']['values'], dtype=float)
    totals = fire + nonfire
    times = np.asarray(data['time_points'])
    assert len(years) == len(fire) == len(nonfire) == len(times)
    assert len(years) == 23 and all(v >= 0 for v in fire) and all(v >= 0 for v in nonfire)
    unit = series['fire_related']['unit']
    assert unit == series['non_fire']['unit']
    axis = data['value_axis']
    assert axis['minimum'] == 0 and axis['scale'] == 'linear'
    assert max(totals) <= axis['maximum']

    ink = (29, 35, 34)
    muted = (88, 96, 91)
    paper = (236, 235, 227)
    orange = (185, 63, 19)
    background = Image.open(ROOT/'assets/smoke.png').convert('RGB').resize((W*S,H*S),Image.Resampling.LANCZOS)
    background = Image.blend(Image.new('RGB', background.size, paper), background, .34)
    # Fade the atmospheric texture into a quiet lower data table.
    a = np.asarray(background).astype(float)
    yy = np.arange(H*S)[:,None,None]/S
    footer_mix = np.clip((yy-1530)/180,0,1)*.95
    a = a*(1-footer_mix) + np.asarray(paper)[None,None,:]*footer_mix
    im = Image.fromarray(np.uint8(np.clip(a,0,255)), 'RGB').convert('RGBA')
    del a
    draw = ImageDraw.Draw(im)
    text_records = []

    def txt(x,y,text,size=24,face='sans',fill=ink,anchor='la'):
        f=font(FONTS[face],size)
        # lt uses visible glyph top for all fonts, independent of ascender metrics.
        anc={'la':'lt','ma':'mt','ra':'rt'}[anchor]
        pos=(round(x*S),round(y*S))
        draw.text(pos,str(text),font=f,fill=fill,anchor=anc,stroke_width=0)
        b=draw.textbbox(pos,str(text),font=f,anchor=anc)
        text_records.append({'text':str(text),'box':[round(v/S,2) for v in b]})

    def line(coords, fill, width=1):
        draw.line([round(v*S) for v in coords],fill=fill,width=max(1,round(width*S)))

    def rect(box,fill):
        draw.rectangle([round(v*S) for v in box],fill=fill)

    def dashed(x0,y,x1,fill,width=1,dash=7,gap=10):
        x=x0
        while x<x1:
            line((x,y,min(x+dash,x1),y),fill,width)
            x+=dash+gap

    def tracking(x,y,text,size=18,space=2,fill=muted):
        f=font(FONTS['bold'],size)
        for ch in text:
            txt(x,y,ch,size,'bold',fill)
            x+=draw.textlength(ch,font=f)/S+space

    # Editorial title and data context, all rendered from text by code.
    rect((80,62,348,108),orange)
    tracking(98,75,data['visible_notice'],18,2.0,(255,248,229))
    tracking(1060,77,'ANNUAL SERIES / 2001–2023',16,1.2,muted)
    txt(76,141,'Forest loss',116,'serif')
    txt(80,259,'by cause',116,'italic')
    txt(83,390,'Annual forest loss, split into two sources',27,'sans',muted)

    txt(1016,174,f'{years[0]} to {years[-1]}',22,'bold',muted)
    txt(1010,220,f'{totals[0]:.1f} → {totals[-1]:.1f}',58,'serif',ink)
    txt(1016,289,unit+' in annual loss',21,'sans',muted)
    rect((1018,349,1035,372),ink)
    txt(1050,348,series['non_fire']['label'],24,'sans',ink)
    rect((1018,396,1035,419),(236,114,17))
    txt(1050,395,series['fire_related']['label'],24,'sans',ink)
    line((80,450,1520,450),(152,157,150),1)
    tracking(80,479,'LOSS / MILLION HECTARES',17,1.3,muted)
    txt(1518,478,'One tree = one year · Tree height = total loss',21,'sans',muted,'ra')

    # One linear transform controls tree apex, orange junction and zero baseline.
    chart_top, baseline = 540.0, 1590.0
    scale = (baseline-chart_top)/(axis['maximum']-axis['minimum'])
    x0,x1 = 210.,1464.
    xs = x0+(times-times[0])/(times[-1]-times[0])*(x1-x0)
    grid_left, grid_right = 160.,1518.
    for v in range(int(axis['minimum']),int(axis['maximum'])+1,5):
        y=baseline-(v-axis['minimum'])*scale
        if v:
            dashed(grid_left,y,grid_right,(158,164,157),.85)
        else:
            # Draw zero line behind the marks so it cannot occlude their bases.
            line((grid_left,baseline,grid_right,baseline),ink,1.5)
        txt(137,y-12,f'{v}',22,'sans',muted,'ra')

    tree_sources = [Image.open(ROOT/f'assets/tree_{i:02d}.png').convert('RGBA') for i in range(1,5)]
    flame_plate = Image.open(ROOT/'assets/flames.png').convert('RGB')
    actual_geometry = []
    pixel_masks = []
    variant_order = [0,1,3,2,1,0,3,1,2,0,3,2,0,1,3,0,2,1,3,2,0,3,1]
    for i,(year,xc,f,n,total) in enumerate(zip(years,xs,fire,nonfire,totals)):
        width = 69
        width_px = width*S
        total_px = int(round(total*scale*S))
        fire_px = int(round(f*scale*S))
        top_px = round(baseline*S)-total_px
        junction_px = round(baseline*S)-fire_px
        left_px = round(xc*S-width_px/2)
        source = tree_sources[variant_order[i]]
        if i%3==1:
            source=ImageOps.mirror(source)
        tree = source.resize((width_px,total_px),Image.Resampling.LANCZOS)
        rgba = np.asarray(tree).copy()
        # Increase twig stroke weight only horizontally: no vertical padding,
        # no minimum data heights, and no marks outside the encoded extent.
        alpha = maximum_filter1d(rgba[:,:,3],size=3,axis=1)
        # Make the generated apex readable at its encoded height. This small
        # stem follows the asset's own top centerline and stays inside the data
        # bounding box; it changes stroke weight, never quantitative length.
        tip_h=min(18*S,total_px)
        tip_points=[]
        for yy_tip in range(tip_h):
            row=alpha[yy_tip].astype(float)
            weights=np.maximum(row-8,0)
            tx=float((np.arange(width_px)*weights).sum()/weights.sum()) if weights.sum() else width_px/2
            tip_points.append((int(round(tx)),yy_tip))
        tip_mask=Image.fromarray(alpha,'L')
        ImageDraw.Draw(tip_mask).line(tip_points,fill=255,width=3)
        alpha=np.asarray(tip_mask).copy()
        base_weight=alpha[-12*S:].sum(axis=0).astype(float)
        base_center=float((np.arange(width_px)*base_weight).sum()/base_weight.sum())
        left_px=round(xc*S-base_center)
        apex_x=(left_px+tip_points[0][0])/S
        grey = rgba[:,:,:3].mean(axis=2)
        dark = np.clip(12+grey*.17,12,35)
        rgb = np.stack([dark*.85,dark*.98,dark*.94],axis=-1)
        crop_x = (i*137)%(flame_plate.width-96)
        tex = flame_plate.crop((crop_x,0,crop_x+96,flame_plate.height)).resize((width_px,fire_px),Image.Resampling.LANCZOS)
        flame = np.asarray(tex).astype(float)
        t=np.linspace(0,1,fire_px)[:,None,None]
        ramp=np.array([244.,100.,9.])[None,None,:]*(1-t)+np.array([255.,211.,51.])[None,None,:]*t
        flame=.62*flame+.38*ramp
        rgb[-fire_px:,:,:]=flame
        rgba[:,:,:3]=np.uint8(np.clip(rgb,0,255))
        rgba[:,:,3]=alpha
        mark = Image.fromarray(rgba,'RGBA')
        im.alpha_composite(mark,(left_px,top_px))
        # Native-resolution isolated data marks, used for pixel-level checks.
        native_mark=mark.resize((width,round(total_px/S)),Image.Resampling.LANCZOS)
        ar=np.asarray(native_mark)
        visible=ar[:,:,3]>=64
        fy,fx=np.where(visible & (ar[:,:,0]>145) & (ar[:,:,0]>ar[:,:,2]+60))
        vy,vx=np.where(visible)
        by,bx=np.where(visible & (ar[:,:,0]<70) & (ar[:,:,1]<70))
        actual_geometry.append({
            'year':year,'tree_asset':f'assets/tree_{variant_order[i]+1:02d}.png',
            'center_x':round(float(xc),3),'apex_x':apex_x,'target_fire':float(f),'target_non_fire':float(n),'target_total':round(float(total),1),
            'target_top_y':baseline-total*scale,'target_junction_y':baseline-f*scale,'baseline_y':baseline,
            'composite_top_y':top_px/S,'composite_junction_y':junction_px/S,
            'visible_bbox_in_mark':[int(vx.min()),int(vy.min()),int(vx.max())+1,int(vy.max())+1],
            'visible_height_pixels':int(vy.max()-vy.min()+1),
            'visible_fire_height_pixels':int(fy.max()-fy.min()+1),
            'visible_nonfire_height_pixels':int(by.max()-by.min()+1),
            'alpha_threshold':64,
            'mask_visible_pixels':int(visible.sum()),
            'left_x':left_px/S,'width':width,
        })
        pixel_masks.append((left_px/S,top_px/S,ar))

    draw=ImageDraw.Draw(im)
    # Numerical apex labels sit above, never extend the tree's data geometry.
    peak=int(totals.argmax())
    for i,(xc,total) in enumerate(zip(xs,totals)):
        top=baseline-total*scale
        txt(float(xc),top-31,f'{total:.1f}',19,'bold',orange if i==peak else ink,'ma')
    # A single restrained callout occupies open space above the later years.
    cx=float(xs[peak]); cy=baseline-float(totals[peak])*scale
    txt(820,602,f'{years[peak]} / HIGHEST TOTAL',19,'bold',orange)
    txt(820,636,f'{totals[peak]:.1f} million hectares',29,'serif',ink)
    line((805,641,742,641),(115,122,114),1.2)
    line((742,641,cx+11,cy+36),(115,122,114),1.2)

    # Annual labels align with both the trees and the exact-value register.
    txt(81,1617,'YEAR',17,'bold',muted)
    for xc,year in zip(xs,years):
        line((float(xc),baseline,float(xc),baseline+10),ink,1)
        txt(float(xc),1615,year,19,'sans',ink,'ma')
    line((80,1657,1520,1657),(166,170,162),1)
    # Every supplied value is retained to one decimal place.
    rows=[('TOTAL',totals,1685,ink,'bold'),('NON-FIRE',nonfire,1735,ink,'sans'),('FIRE',fire,1785,orange,'sans')]
    for label,vals,y,color,face in rows:
        txt(81,y+3,label,17,'bold',color)
        for xc,val in zip(xs,vals):
            txt(float(xc),y,f'{val:.1f}',20,face,color,'ma')
    line((80,1835,1520,1835),(166,170,162),1)
    txt(80,1860,'READING THE TREES',17,'bold',orange)
    txt(320,1857,'Orange base: fire-related loss. Black upper section: non-fire loss.',22,'sans',ink)
    txt(320,1891,'Both heights share the 0–30 million-hectare scale; all annual values are shown above.',21,'sans',muted)
    txt(80,1950,'SYNTHETIC DATA · Illustrative values, not observed forest-loss statistics.',19,'sans',muted)
    txt(1520,1950,'Source: supplied data.json',19,'sans',muted,'ra')

    out=im.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
    out.save(ROOT/'final.png',optimize=True)
    # Actual final raster measurements: locally search for the tree's dark apex
    # and orange segment in its narrow trunk region, below all apex labels.
    final_arr=np.asarray(out).astype(np.int16)
    for r in actual_geometry:
        x=int(round(r['center_x']))
        top=int(math.floor(r['target_top_y']))
        # Avoid branches of neighbors by locating stem at the lower central band.
        lo=max(0,int(r['left_x'])); hi=min(W,int(r['left_x']+r['width']))
        near_bottom=final_arr[int(baseline)-15:int(baseline)-4,lo:hi,:]
        warm=(near_bottom[:,:,0]>160)&(near_bottom[:,:,0]>near_bottom[:,:,2]+65)&(near_bottom[:,:,1]>65)
        candidates=np.where(warm.sum(axis=0)>5)[0]
        stem_x=lo+int(np.median(candidates)) if len(candidates) else x
        strip=final_arr[top:int(baseline),max(0,stem_x-2):stem_x+3,:]
        orange_pixels=(strip[:,:,0]>150)&(strip[:,:,0]>strip[:,:,2]+65)&(strip[:,:,1]>60)
        orange_rows=np.where(orange_pixels.any(axis=1))[0]
        r['final_pixel_fire_top_y']=int(top+orange_rows.min()) if len(orange_rows) else None
        r['final_pixel_fire_height']=int(baseline-r['final_pixel_fire_top_y']) if len(orange_rows) else None
        # Dark apex is searched in the object's uppermost 16 px; chart labels
        # are 31 px above the mathematical apex and cannot satisfy this search.
        apex_x=int(round(r['apex_x']))
        window=final_arr[top:min(top+16,int(baseline)),apex_x-3:apex_x+4,:]
        black=(window[:,:,0]<70)&(window[:,:,1]<70)&(window[:,:,2]<70)
        black_rows=np.where(black.any(axis=1))[0]
        r['final_pixel_apex_y']=int(top+black_rows.min()) if len(black_rows) else None
        r['final_pixel_total_height']=int(baseline-r['final_pixel_apex_y']) if len(black_rows) else None
        r['final_scan_stem_x']=stem_x
    artifact=ROOT/'artifacts'
    artifact.mkdir(exist_ok=True)
    (artifact/'geometry_checks.json').write_text(json.dumps({
        'canvas':[W,H],'scale_pixels_per_million_hectares':scale,'plot_top':chart_top,'baseline':baseline,
        'method':'Native mark alpha and RGB bounds, then independent dark-apex and warm-trunk scans in final.png.',
        'trees':actual_geometry,
        'all_text_boxes_inside_canvas':all(0<=r['box'][0]<=r['box'][2]<=W and 0<=r['box'][1]<=r['box'][3]<=H for r in text_records),
        'text_labels':text_records,
    },ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Rendered {W} × {H}; {len(years)} generated tree marks; scale {scale:g} px / million hectares.')

if __name__=='__main__':
    main()
