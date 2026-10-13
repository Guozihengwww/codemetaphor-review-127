#!/usr/bin/env python3
"""Offline, deterministic coal-pile infographic. Run with no arguments.

Coal body height = (share / 30) * 360 px. All bodies share y=798.
Widths are layout settings only. Shadows are composed in a separate layer.
The prepared photographic assets and fonts are local dependencies.
"""
from pathlib import Path
from decimal import Decimal
import json
import hashlib
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
BRIEF = json.loads((ROOT/'brief.json').read_text(encoding='utf-8'))
DATA = json.loads((ROOT/'data.json').read_text(encoding='utf-8'))
CORE = json.loads((ROOT/'core.json').read_text(encoding='utf-8'))
INPUT = json.loads((ROOT/'input.json').read_text(encoding='utf-8'))
W, H = BRIEF['width'], BRIEF['height']
SS = 3
BASELINE = 798.0
PLOT_HEIGHT = 360.0
VMIN, VMAX = DATA['value_axis']['minimum'], DATA['value_axis']['maximum']
PX_PER_UNIT = PLOT_HEIGHT / (VMAX-VMIN)
NAVY = '#153E56'
SECONDARY = '#416B7D'
MUTED = '#567F8D'
FONT_PATHS = {
    'regular': ROOT/'assets/fonts/NimbusSans-Regular.otf',
    'bold': ROOT/'assets/fonts/NimbusSans-Bold.otf',
    'number': ROOT/'assets/fonts/NimbusSansNarrow-Regular.otf',
}
TEXT_RECORDS = []

def font(size, style='regular'):
    return ImageFont.truetype(str(FONT_PATHS[style]), round(size*SS))

def line(draw, coords, fill, width=1):
    draw.line([round(v*SS) for v in coords], fill=fill, width=max(1,round(width*SS)))

def text(draw, xy, s, size, fill=NAVY, style='regular', align='left', track=0, role='annotation'):
    f = font(size, style)
    x,y = xy
    length = draw.textlength(s, font=f)/SS + max(0,len(s)-1)*track
    if align=='center': x -= length/2
    if align=='right': x -= length
    if track:
        xx = x*SS
        for ch in s:
            draw.text((round(xx),round(y*SS)),ch,font=f,fill=fill,anchor='lt')
            xx += draw.textlength(ch,font=f) + track*SS
    else:
        draw.text((round(x*SS),round(y*SS)),s,font=f,fill=fill,anchor='lt')
    # The text engine's ink bounds are saved as evidence, not inferred font boxes.
    b = draw.textbbox((round(x*SS),round(y*SS)),s,font=f,anchor='lt')
    box = [b[0]/SS,b[1]/SS,(b[2]+max(0,len(s)-1)*track*SS)/SS,b[3]/SS]
    TEXT_RECORDS.append({'text':s,'role':role,'ink_bbox_px':box})
    return length


def fmt(v):
    return format(Decimal(str(v)).normalize(),'f')


def background():
    # A cool paper-like ground, with deterministic subpixel grain.
    yy,xx = np.mgrid[0:H,0:W]
    t = .44*(xx/(W-1)) + .56*(yy/(H-1))
    a,b = np.array([222,239,238.]), np.array([176,213,219.])
    rgb = a[None,None,:]*(1-t[:,:,None]) + b[None,None,:]*t[:,:,None]
    glow = np.exp(-(((xx-550)/1350)**2 + ((yy-740)/680)**2))
    rgb += glow[:,:,None]*np.array([2.5,3.5,3.5])
    grain = np.random.default_rng(1978).normal(0,.36,(H,W,1))
    return Image.fromarray(np.uint8(np.clip(rgb+grain,0,255))).resize((W*SS,H*SS),Image.Resampling.BICUBIC).convert('RGBA')


def shadow(width):
    # Non-data floor shadow. Height/depth does not depend on the share value.
    sw,sh = round((width+42)*SS), 62*SS
    yy,xx = np.mgrid[0:sh,0:sw]
    xx = (xx-sw/2)/SS
    yy = (yy-25*SS)/SS
    a = (29*np.exp(-(xx/(width*.50))**2 -(yy/11)**2)
         + 42*np.exp(-(xx/(width*.40))**2 -(yy/3.3)**2))
    rgba = np.zeros((sh,sw,4),dtype='uint8')
    rgba[:,:,:3] = [29,70,78]
    rgba[:,:,3] = np.clip(a,0,255).astype('uint8')
    return Image.fromarray(rgba)


def main():
    assert DATA['synthetic'] and DATA['value_axis']['scale']=='linear'
    assert VMIN==0 and VMAX==30
    series=DATA['series'][0]
    values=series['values']
    assert len(DATA['series'])==1 and len(values)==9
    assert sum(Decimal(str(v)) for v in values)==Decimal(str(DATA['series_totals']['S1']))
    im=background()
    draw=ImageDraw.Draw(im)
    # A single editorial header, with a high-contrast synthetic-data notice.
    draw.rounded_rectangle([96*SS,64*SS,365*SS,108*SS],radius=7*SS,fill='#F3D48D')
    text(draw,(115,76),DATA['visible_notice'],22,style='bold',track=1.4,role='synthetic_notice')
    text(draw,(96,153),'Solid fuel import shares',84,style='bold',role='title')
    text(draw,(100,254),series['label']+' ('+series['unit']+')',31,role='metric')
    text(draw,(100,303),'By synthetic partner and subcategory',25,SECONDARY,role='scope')
    # This figure is the supplied series total, not an additional dataset.
    text(draw,(1904,90),'TOTAL ALLOCATION',17,SECONDARY,style='bold',align='right',track=1.65)
    text(draw,(1904,130),fmt(DATA['series_totals']['S1'])+'%',76,style='bold',align='right',role='series_total')
    text(draw,(1904,221),'of imported solid-fuel mass',23,SECONDARY,align='right')

    text(draw,(96,384),'SHARE (%)',17,SECONDARY,style='bold',track=1.1,role='axis_title')
    # Common zero-ground and a visible, uniform 0–30% linear scale.
    for tick in range(0,31,5):
        y=BASELINE-tick*PX_PER_UNIT
        col='#ACCBD0' if tick else '#6D9FAA'
        line(draw,(144,y,1918,y),col,1 if tick else 1.25)
        text(draw,(126,y-10),str(tick),20,MUTED,align='right',role='axis_tick')
    line(draw,(144,BASELINE-PLOT_HEIGHT,144,BASELINE),'#7BA4AF',1)

    # Fixed layout footprints, not an additional data encoding.
    widths=[350,160,160,165,165,165,145,155,155]
    x=154.0
    geometry=[]
    masks=[]
    for i,(cat,v,width) in enumerate(zip(DATA['categories'],values,widths)):
        height=float(Decimal(str(v))*Decimal(str(PX_PER_UNIT)))
        center=x+width/2
        obj=Image.open(ROOT/f'assets/coal_{i+1:02}.png').convert('RGBA')
        target=(round(width*SS),round(height*SS))
        obj=obj.resize(target,Image.Resampling.LANCZOS)
        y=BASELINE-target[1]/SS
        # Separate projected floor shadow, composed before the coal body.
        sh=shadow(width)
        im.alpha_composite(sh,(round((center-(width+42)/2)*SS),round((BASELINE-25)*SS)))
        pos=(round(x*SS),round(y*SS))
        im.alpha_composite(obj,pos)
        aa=np.asarray(obj)[:,:,3]
        ys,xs=np.where(aa>=64)
        body_bbox=[(pos[0]+int(xs.min()))/SS,(pos[1]+int(ys.min()))/SS,
                   (pos[0]+int(xs.max())+1)/SS,(pos[1]+int(ys.max())+1)/SS]
        # Each value sits above its own pile, linked by a short vertical stem.
        peak_columns=np.where(aa[:max(1,round(2*SS))].max(axis=0)>=64)[0]
        peak_x = x+float(np.median(peak_columns))/SS if len(peak_columns) else center
        label_x = max(x+53,min(x+width-53,peak_x))
        draw=ImageDraw.Draw(im)
        text(draw,(label_x,y-64),fmt(v)+'%',43,NAVY,style='number',align='center',role='value')
        line(draw,(label_x,y-24,label_x,y-8),'#31677F',1.4)
        parts=cat.split(' / ',1)
        text(draw,(center,855),parts[0],27,NAVY,style='bold',align='center',role='category_parent')
        if len(parts)>1:
            text(draw,(center,898),parts[1],22,SECONDARY,align='center',role='subcategory')
        geometry.append({'index':i,'category':cat,'value':v,'unit':series['unit'],
                         'asset':f'assets/coal_{i+1:02}.png','layout_left_px':x,
                         'layout_width_px':width,'shared_ground_y_px':BASELINE,
                         'target_height_px':height,'raster_height_before_downsample_px':target[1]/SS,
                         'supersample_alpha64_bbox_px':body_bbox,
                         'supersample_alpha64_height_px':body_bbox[3]-body_bbox[1],
                         'label_x_px':label_x})
        x += width+18

    draw=ImageDraw.Draw(im)
    line(draw,(96,985,1904,985),'#94BCC5',1)
    text(draw,(96,1015),'READING THE PILES',17,SECONDARY,style='bold',track=1.6)
    text(draw,(96,1052),'Pile height follows one linear scale, from 0 to 30%.',23,NAVY)
    text(draw,(96,1088),'Footprint, chunk count and shadow do not encode values.',21,SECONDARY)
    text(draw,(1110,1015),'SYNTHETIC DATA NOTES',17,SECONDARY,style='bold',track=1.6)
    text(draw,(1110,1052),'Illustrative partner labels; not observed trade statistics.',21,SECONDARY)
    text(draw,(1110,1088),'Subcategories are mutually exclusive parts of their named partner.',21,SECONDARY)

    final=im.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
    final.save(ROOT/'final.png',optimize=True)
    (ROOT/'artifacts').mkdir(exist_ok=True)
    # Saved post-render measurements include actual final-image pixels below.
    pix=np.array(final).astype(float)
    chroma=pix.max(axis=2)-pix.min(axis=2)
    lum=pix[:,:,0]*.2126+pix[:,:,1]*.7152+pix[:,:,2]*.0722
    coal=(chroma<32)&(lum<170)
    for g in geometry:
        # Dark near-neutral coal is distinct from the blue labels, scale and shadows.
        x0=int(math.floor(g['layout_left_px']))
        x1=int(math.ceil(g['layout_left_px']+g['layout_width_px']))
        roi=coal[int(BASELINE-PLOT_HEIGHT):int(BASELINE)+2,x0:x1]
        ys,xs=np.where(roi)
        if len(ys):
            top=int(ys.min()+BASELINE-PLOT_HEIGHT)
            bottom=int(ys.max()+BASELINE-PLOT_HEIGHT+1)
            g['final_visible_coal_bbox_px']=[int(xs.min()+x0),top,int(xs.max()+x0+1),bottom]
            g['final_visible_height_px']=bottom-top
            g['final_visible_height_error_px']=(bottom-top)-g['target_height_px']
            g['final_visible_height_error_percentage_points']=g['final_visible_height_error_px']/PX_PER_UNIT
            g['final_visible_ground_error_px']=bottom-BASELINE
            g['final_visible_height_relative_error_pct']=100*g['final_visible_height_error_px']/g['target_height_px']
    evidence={'canvas':[W,H],'mapping':{'minimum':VMIN,'maximum':VMAX,'plot_height_px':PLOT_HEIGHT,
                'pixels_per_percentage_point':PX_PER_UNIT,'ground_y_px':BASELINE},
              'pixel_measurement_rule':'Final RGB PNG; within each coal footprint and y=438..799: luminance <170 and RGB chroma <32. Coal is near-neutral; labels and shadows are blue.',
              'geometry':geometry,'texts':TEXT_RECORDS,
              'input_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                              for name in ['brief.json','data.json','core.json','input.json','reference.png']}}
    (ROOT/'artifacts/geometry_measurements.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Saved final.png:',W,H,'; coal body heights measured in final RGB pixels.')

if __name__=='__main__':
    main()
