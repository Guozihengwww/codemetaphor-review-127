"""Regenerate the infographic from frozen local data and generated shell material.

Run through capture_render.py for all experiment rendering attempts.
The quantitative geometry is the alpha-weighted area of the displayed egg.
"""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.ndimage import label as connected_components

ROOT = Path(__file__).resolve().parent
SCALE = 2
BG = (242, 248, 242)
INK = (39, 55, 46)
MUTED = (95, 111, 99)
LINE = (102, 115, 102)
# Shell pigments, listed in the original category order.
PALETTE = np.array([
    [104, 62, 38],
    [221, 114, 45],
    [241, 182, 74],
    [180, 132, 68],
    [248, 222, 171],
    [199, 105, 62],
], dtype=np.float32) / 255.0
TEXT_BOXES = []


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def font(kind, size):
    name = {'regular': 'regular.ttf', 'bold': 'bold.ttf', 'title': 'title.ttc'}[kind]
    return ImageFont.truetype(str(ROOT / 'assets' / 'fonts' / name), round(size * SCALE))


def text(draw, xy, value, size, kind='regular', fill=INK, anchor='lt', record=True):
    pos = tuple(round(v*SCALE) for v in xy)
    f = font(kind, size)
    draw.text(pos, value, font=f, fill=fill, anchor=anchor)
    if record:
        box = draw.textbbox(pos, value, font=f, anchor=anchor)
        TEXT_BOXES.append({'text': value, 'bbox': [round(x/SCALE, 2) for x in box]})


def polyline(draw, points, color=LINE, width=1.7):
    ps = [(round(x*SCALE), round(y*SCALE)) for x,y in points]
    draw.line(ps, fill=color, width=round(width*SCALE), joint='curve')


def geometry(shell, values, origin):
    """Partition actual generated silhouette with shared-hub straight rays.

    Weighted angular quantiles integrate the screen-space area, not circle angles.
    Every subpixel is assigned once, and alpha coverage is shared consistently.
    """
    a = np.asarray(shell)[:, :, 3].astype(np.float32) / 255.0
    h, w = a.shape
    cx, cy = w*.51, h*.54
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    theta = np.mod(np.arctan2(xx + .5 - cx, cy - yy - .5), 2*np.pi)
    inside = a > 0
    bins = 1048576
    slot = np.minimum((theta[inside]/(2*np.pi)*bins).astype(np.int32), bins-1)
    hist = np.bincount(slot, weights=a[inside], minlength=bins)
    cumulative = np.cumsum(hist)
    targets = np.cumsum(values)[:-1] / sum(values) * cumulative[-1]
    at = np.searchsorted(cumulative, targets)
    prev = cumulative[np.maximum(at-1, 0)]
    frac = (targets - prev) / np.maximum(hist[at], 1e-12)
    cuts = np.r_[0, (at+frac)/bins*(2*np.pi), 2*np.pi]
    ids = np.searchsorted(cuts[1:-1], theta, side='right').astype(np.uint8)
    actual = np.bincount(ids[inside], weights=a[inside], minlength=len(values))
    areas = actual/actual.sum()*100
    anchors = []
    for i in range(len(values)):
        t = (cuts[i]+cuts[i+1])/2
        rr = np.arange(0, max(w,h)*1.5, .25, dtype=np.float32)
        rx = (cx + rr*np.sin(t)).astype(np.int32)
        ry = (cy - rr*np.cos(t)).astype(np.int32)
        ok = (rx>=0)&(rx<w)&(ry>=0)&(ry<h)
        rr, rx, ry = rr[ok], rx[ok], ry[ok]
        r = float(rr[a[ry,rx]>.5][-1])
        anchors.append((origin[0]+(cx+r*np.sin(t))/SCALE,
                        origin[1]+(cy-r*np.cos(t))/SCALE))
    return a, ids, cuts, areas, anchors, (origin[0]+cx/SCALE, origin[1]+cy/SCALE)


def make_shell(shell, a, ids):
    rgb = np.asarray(shell)[:,:,:3].astype(np.float32)/255
    # The generated raster supplies all spatial light, dark, pores and grain.
    lum = rgb @ np.array([.2126,.7152,.0722],np.float32)
    pigment = PALETTE[ids]
    shadow = np.minimum(lum/.64, 1.0)**.86
    highlight = np.maximum((lum-.64)/.36,0)**1.40
    highlight = np.clip(highlight,0,1)
    warm_white = np.array([1.0,.978,.929],np.float32)
    out = pigment * shadow[:,:,None]
    out = out*(1-highlight[:,:,None]) + warm_white*highlight[:,:,None]
    # Keep the source's slight chromatic shell variation along with its pores.
    source_chroma = rgb - lum[:,:,None]
    out += source_chroma*.13
    out = np.clip(out*255,0,255).astype(np.uint8)
    return Image.fromarray(np.dstack((out,np.round(a*255).astype(np.uint8))), 'RGBA')


def draw_label(draw, category, value, x, y, index, major=False):
    dot = tuple(np.round(PALETTE[index]*255).astype(int))
    draw.rounded_rectangle((round(x*SCALE),round((y+5)*SCALE),round((x+7)*SCALE),round((y+28)*SCALE)), radius=3*SCALE, fill=dot)
    text(draw,(x+23,y),category,30,'regular')
    text(draw,(x+19,y+45),f'{value:.1f}%',90 if major else 62,'bold', fill=INK)


def main():
    TEXT_BOXES.clear()
    data=json.loads((ROOT/'data.json').read_text(encoding='utf-8'))
    brief=json.loads((ROOT/'brief.json').read_text(encoding='utf-8'))
    cats=data['categories']; values=np.array(data['series'][0]['values'],float)
    assert len(cats)==6 and len(values)==6 and np.isclose(values.sum(),100)
    width,height=brief['width'],brief['height']
    im=Image.new('RGB',(width*SCALE,height*SCALE),BG)
    draw=ImageDraw.Draw(im)

    # Editorial heading and exact frozen title.
    text(draw,(130,67),'COMPOSITION STUDY',24,'bold',MUTED)
    text(draw,(130,124),data['title'],66,'title')
    text(draw,(133,224),data['series'][0]['label'],28,'regular',MUTED)
    text(draw,(1870,231),'Unit: '+data['series'][0]['unit'],24,'regular',MUTED,anchor='rt')
    draw.rounded_rectangle((1490*SCALE,67*SCALE,1870*SCALE,119*SCALE),radius=26*SCALE,outline=(131,147,131),width=2*SCALE)
    text(draw,(1680,92),data['visible_notice'],24,'bold',INK,anchor='mm')
    polyline(draw,[(130,274),(1870,274)],(204,215,202),1)

    # Soft cast shadow is outside the data mask and carries no quantity.
    shadow=Image.new('RGBA',im.size,(0,0,0,0))
    sd=ImageDraw.Draw(shadow)
    sd.ellipse((682*SCALE,1578*SCALE,1340*SCALE,1630*SCALE),fill=(44,58,39,36))
    shadow=shadow.filter(ImageFilter.GaussianBlur(17*SCALE))
    im=Image.alpha_composite(im.convert('RGBA'),shadow)

    original=Image.open(ROOT/'assets/eggshell.png').convert('RGBA')
    egg_h=1260; egg_w=round(egg_h*original.width/original.height)
    origin=(550,350)
    shell=original.resize((egg_w*SCALE,egg_h*SCALE),Image.Resampling.LANCZOS)
    a,ids,cuts,areas,anchors,hub=geometry(shell,values,origin)
    colored=make_shell(shell,a,ids)
    im.alpha_composite(colored,(origin[0]*SCALE,origin[1]*SCALE))
    draw=ImageDraw.Draw(im)

    # Fine radial seams terminate at the common point; seam width is non-data.
    seams=Image.new('RGBA',shell.size,(0,0,0,0)); ss=ImageDraw.Draw(seams)
    cx,cy=(hub[0]-origin[0])*SCALE,(hub[1]-origin[1])*SCALE
    for t in cuts[:-1]:
        end=(cx+shell.height*2*np.sin(t),cy-shell.height*2*np.cos(t))
        ss.line([(cx,cy),end],fill=(83,51,30,105),width=2)
    s_arr=np.array(seams)
    s_arr[:,:,3]=np.round(s_arr[:,:,3]*a).astype(np.uint8)
    im.alpha_composite(Image.fromarray(s_arr),(origin[0]*SCALE,origin[1]*SCALE))
    draw=ImageDraw.Draw(im)

    # Leaders start on their own region's contour. Outer tracks are nested,
    # preserving visible one-to-one bindings without crossing the egg body.
    paths=[
        [anchors[0],(anchors[0][0],320),(790,320),(790,445),(754,445)],
        [anchors[1],(1340,anchors[1][1])],
        [anchors[2],(1205,421),(1492,421),(1492,607),(1520,607)],
        [anchors[3],(1467,anchors[3][1]),(1467,852),(1520,852)],
        [anchors[4],(1449,anchors[4][1]),(1449,1132),(1520,1132)],
        [anchors[5],(520,anchors[5][1]),(520,1231),(457,1231)],
    ]
    for i,points in enumerate(paths):
        polyline(draw,points,width=1.6)
        x,y=points[0]
        draw.ellipse(((x-2.3)*SCALE,(y-2.3)*SCALE,(x+2.3)*SCALE,(y+2.3)*SCALE),fill=LINE)
    specs=[(530,386),(1374,294),(1553,560),(1553,805),(1553,1085),(165,1151)]
    for i,((x,y),cat,val) in enumerate(zip(specs,cats,values)):
        draw_label(draw,cat,val,x,y,i,major=(i==5))

    polyline(draw,[(130,1664),(1870,1664)],(204,215,202),1)
    text(draw,(130,1702),'Each region’s area is its share of the whole egg.',27,'regular')
    text(draw,(1870,1702),'TOTAL  100%',27,'bold',anchor='rt')
    text(draw,(130,1751),'Clockwise: Component A → Component F   /   Area-calibrated partitions',22,'regular',MUTED)

    # Area evidence is calculated again after downsampling individual region
    # coverage to the actual output pixel grid, rather than only from target cuts.
    coverage=[]
    components=[]
    for i in range(len(cats)):
        high=np.round(a*(ids==i)*255).astype(np.uint8)
        cov=np.array(Image.fromarray(high).resize((egg_w,egg_h),Image.Resampling.BOX),dtype=np.float64)/255
        coverage.append(cov)
        # Count connected area at native output resolution (discard subpixel fringes).
        labels,n=connected_components(cov>.5)
        sizes=np.bincount(labels.ravel())[1:]
        components.append({'components_over_2px':int((sizes>2).sum()),'pixel_area_over_50pct_coverage':int((cov>.5).sum())})
    screen_areas=np.array([c.sum() for c in coverage]); screen_pct=screen_areas/screen_areas.sum()*100
    combined=np.sum(coverage,axis=0)
    full=np.array(Image.fromarray(np.round(a*255).astype(np.uint8)).resize((egg_w,egg_h),Image.Resampling.BOX),dtype=float)/255
    # Diagnostic only: categorical mask at native output resolution.
    mask_id=np.argmax(np.stack(coverage),axis=0).astype(np.uint8)+1
    mask_id[full<.5]=0
    Image.fromarray(mask_id,'L').save(ROOT/'artifacts/region_ids.png')
    Image.fromarray(np.round(full*255).astype(np.uint8)).save(ROOT/'artifacts/egg_mask.png')
    metrics={
        'canvas_px':[width,height], 'supersampling':SCALE,
        'mask_bbox_px':[origin[0],origin[1],origin[0]+egg_w,origin[1]+egg_h],
        'shared_hub_px':list(hub),'clockwise_from_top':True,
        'boundary_angles_degrees':np.degrees(cuts).tolist(),
        'area_definition':'alpha-weighted final screen-space egg area; not central angle',
        'regions':[{'category':c,'label':f'{v:.1f}%','target_percent':float(v),
                    'high_resolution_area_percent':float(areas[i]),
                    'screen_area_percent':float(screen_pct[i]),
                    'absolute_error_percentage_points':float(abs(screen_pct[i]-v)),
                    'contour_anchor_px':list(anchors[i]),**components[i]}
                   for i,(c,v) in enumerate(zip(cats,values))],
        'max_area_error_pp':float(np.max(abs(screen_pct-values))),
        'coverage_discrepancy_area_px':float(np.abs(combined-full).sum()),
        'screen_mask_area_px':float(full.sum()),
        'all_subpixels_assigned_exactly_once':True,
        'color_seam_width_px':1.0,'category_label_bboxes':TEXT_BOXES,
        'leader_paths_px':paths,
        'generated_material_visible_over_percent_of_data_mask':100.0,
        'generated_image_role':'all spatial shell lighting, texture and silhouette; code recolors and partitions',
        'reference_pixels_used':False,
    }
    write_json(ROOT/'artifacts/geometry_check.json',metrics)
    im.convert('RGB').resize((width,height),Image.Resampling.LANCZOS).save(ROOT/'final.png')
    print(json.dumps({'canvas':[width,height],'actual_area_percent':screen_pct.tolist(),'max_error_pp':metrics['max_area_error_pp']},ensure_ascii=False))


if __name__=='__main__':
    main()
