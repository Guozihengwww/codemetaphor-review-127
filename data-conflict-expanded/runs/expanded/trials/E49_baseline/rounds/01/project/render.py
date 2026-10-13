"""Offline infographic renderer. Run through capture_render.py for every chart render."""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent
S = 2
BG = (12, 25, 43)
CREAM = (255, 237, 186)
WHITE = (242, 246, 247)
MUTED = (172, 192, 211)


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def font(size, bold=False):
    name = 'NimbusSans-Bold.otf' if bold else 'NimbusSans-Regular.otf'
    return ImageFont.truetype(str(ROOT / 'assets/fonts' / name), round(size*S))


def main():
    inputs = read_json('input.json')
    brief = read_json(inputs['brief'])
    data = read_json(inputs['data'])
    core = read_json(inputs['critical_metaphor_contract'])
    assert (ROOT / inputs['reference']).is_file()
    assert data['synthetic'] is True and data['value_axis']['scale'] == 'linear'
    assert core['semantic_roles_zh'] == ['Synthetic market capitalization (USD billion)']
    width, height = brief['width'], brief['height']
    assert (width, height) == (2000, 2000)
    series = data['series'][0]
    vals = series['values']
    cats = data['categories']
    assert len(cats) == len(vals) == 7
    assert data['value_axis']['minimum'] == 0
    max_axis = data['value_axis']['maximum']
    x0, xmax = 430.0, 1770.0
    px_per_unit = (xmax-x0) / max_axis
    axis_y = 514
    plot_bottom = 1760
    row_ys = [632 + 173*i for i in range(len(vals))]

    canvas = Image.open(ROOT / 'assets/raw_02.png').convert('RGBA').resize((width*S,height*S),Image.Resampling.LANCZOS)
    canvas = Image.alpha_composite(canvas,Image.new('RGBA',canvas.size,(*BG,85)))
    draw = ImageDraw.Draw(canvas)

    def txt(x, y, text, size, color=WHITE, bold=False, anchor='la'):
        draw.text((round(x*S),round(y*S)),text,font=font(size,bold),fill=color,anchor=anchor)

    def spaced(x, y, text, size, spacing, color=WHITE, bold=False, center=False):
        ff = font(size,bold)
        total = sum(ff.getlength(c) for c in text) + spacing*S*(len(text)-1)
        at = x*S-total/2 if center else x*S
        for c in text:
            draw.text((round(at),round(y*S)),c,font=ff,fill=color,anchor='lt')
            at += ff.getlength(c)+spacing*S

    def line(points, fill, w=1):
        draw.line([(round(x*S),round(y*S)) for x,y in points],fill=fill,width=max(1,round(w*S)))

    # Editorial heading: the synthetic notice is part of the main visual hierarchy.
    line([(96,80),(690,80)],(72,93,115),1)
    line([(1310,80),(1904,80)],(72,93,115),1)
    draw.rounded_rectangle((735*S,52*S,1265*S,108*S),radius=28*S,fill=(42,54,68),outline=(177,157,108),width=1*S)
    spaced(1000,69,data['visible_notice'],28,3.0,CREAM,True,True)
    spaced(1000,157,'AIRLINE MARKET',126,2.0,CREAM,True,True)
    spaced(1000,283,'CAPITALIZATION',126,2.0,CREAM,True,True)
    txt(1000,420,f"{series['label']}  ·  {series['unit']}",31,MUTED,anchor='mm')

    # Axis and unobtrusive map guides. The aircraft nose, not its bounding-box center, is the endpoint.
    guides = Image.new('RGBA',canvas.size,(0,0,0,0))
    gd = ImageDraw.Draw(guides)
    for tick in range(0,int(max_axis)+1,5):
        x = x0 + tick*px_per_unit
        if tick == 0:
            gd.line((round(x*S),axis_y*S,round(x*S),plot_bottom*S),fill=(223,232,238,88),width=2*S)
        else:
            for y in range(axis_y+14,plot_bottom,18):
                gd.line((round(x*S),y*S,round(x*S),(y+6)*S),fill=(197,215,232,32),width=S)
    canvas.alpha_composite(guides)
    draw = ImageDraw.Draw(canvas)
    spaced(111,476,'AIRLINE',25,2.0,MUTED,True)
    line([(x0,axis_y),(xmax,axis_y)],(103,124,145),1)
    for tick in range(0,int(max_axis)+1,5):
        x = x0 + tick*px_per_unit
        txt(x,482,str(tick),29,CREAM if tick==0 else MUTED,True,anchor='mm')
        line([(x,axis_y-4),(x,axis_y+7)],(178,196,211),1.5)

    aircraft = Image.open(ROOT / 'assets/aircraft.png').convert('RGBA')
    spec = read_json('assets/aircraft_geometry.json')
    nx, ny = spec['nose_xy']
    source_span = nx-spec['solid_x_min']
    base_span = 302.0
    vmax = max(vals)
    placements = []
    for i,(cat,value,y) in enumerate(zip(cats,vals,row_ys)):
        # A single uniform scale for both axes makes the entity area proportional to value.
        s = base_span/source_span * math.sqrt(value/vmax)
        nose = x0+value*px_per_unit
        leftf = nose*S - nx*s*S
        topf = y*S - ny*s*S
        left, top = math.floor(leftf), math.floor(topf)
        right = math.ceil(leftf+aircraft.width*s*S)
        bottom = math.ceil(topf+aircraft.height*s*S)
        sprite = aircraft.transform((right-left,bottom-top),Image.Transform.AFFINE,
            (1/(s*S),0,(left-leftf)/(s*S),0,1/(s*S),(top-topf)/(s*S)),
            Image.Resampling.BICUBIC)
        placements.append(dict(index=i,category=cat,value=value,y=y,nose=nose,scale=s,
                               left=left,top=top,sprite=sprite))

    # Fading light ribbons connect the identity blocks to the rear of each airplane.
    # Their common width is not an area encoding; only aircraft entity pixels carry area.
    for p in placements:
        y, nose, value = p['y'],p['nose'],p['value']
        body_span = base_span*math.sqrt(value/vmax)
        end = nose - 0.40*body_span
        tw = max(1,math.ceil((end-x0)*S))
        th = 88*S
        u = np.linspace(0,1,tw)[None,:]
        v = np.linspace(-1,1,th)[:,None]
        horizontal = (1-u)**0.9
        vertical = np.clip((1-np.abs(v))*14,0,1)
        rgba = np.empty((th,tw,4),dtype=np.uint8)
        rgba[:,:,:3] = (237,243,247)
        rgba[:,:,3] = np.rint(238*horizontal*vertical).astype('uint8')
        canvas.alpha_composite(Image.fromarray(rgba),(round(x0*S),round((y-44)*S)))

    draw = ImageDraw.Draw(canvas)
    for i,(cat,y) in enumerate(zip(cats,row_ys)):
        # The square letter block takes the semantic role of the reference's identity flag.
        draw.rectangle((96*S,(y-43)*S,430*S,(y+43)*S),fill=(236,241,243))
        draw.rectangle((96*S,(y-43)*S,184*S,(y+43)*S),fill=(40,68,91))
        draw.rectangle((96*S,(y-43)*S,102*S,(y+43)*S),fill=CREAM)
        txt(141,y+1,cat.rsplit(' ',1)[-1],47,CREAM,True,'mm')
        txt(207,y+1,cat,39,(24,49,70),True,'lm')

    # All aircraft are layered above ribbons; shadows do not participate in the area calculation.
    for p in placements:
        spr = p['sprite']
        shadow = Image.new('RGBA',spr.size,(0,0,0,0))
        shadow.putalpha(spr.getchannel('A').point(lambda a:round(a*0.38)))
        pad = 12*S
        padded = Image.new('RGBA',(spr.width+2*pad,spr.height+2*pad),(0,0,0,0))
        padded.alpha_composite(shadow,(pad,pad))
        padded = padded.filter(ImageFilter.GaussianBlur(5*S))
        canvas.alpha_composite(padded,(p['left']-pad+2*S,p['top']-pad+5*S))
        canvas.alpha_composite(spr,(p['left'],p['top']))

    draw = ImageDraw.Draw(canvas)
    label_boxes = []
    for p in placements:
        label = f"${p['value']:.1f} bn"
        lx, ly = p['nose']+31,p['y']+2
        txt(lx+1,ly+2,label,58,(3,13,23),True,'lm')
        txt(lx,ly,label,58,CREAM,True,'lm')
        bb = draw.textbbox((round(lx*S),round(ly*S)),label,font=font(58,True),anchor='lm')
        label_boxes.append([round(a/S,2) for a in bb])

    # Encoding explanation and source disclosure remain separate from data labels.
    line([(96,1818),(1904,1818)],(74,97,120),1)
    spaced(96,1860,'READ THE NOSE',25,1.6,CREAM,True)
    txt(431,1873,'Distance from 0 and aircraft area encode the same market value.',28,WHITE,anchor='lm')
    txt(96,1941,'Source: local synthetic dataset  ·  Illustrative airline categories and values.',25,MUTED,anchor='lm')
    txt(1904,1941,'0–35 linear scale',25,MUTED,anchor='rm')

    final = canvas.convert('RGB').resize((width,height),Image.Resampling.LANCZOS)
    final.save(ROOT / 'final.png')

    # Measure the rasterized entity masks and actual bright nose pixels in final.png.
    # These checks use the same transformed artwork pixels, not just nominal dimensions.
    outdir = ROOT / 'artifacts/masks'
    outdir.mkdir(parents=True,exist_ok=True)
    final_array = np.asarray(final).astype(float)
    luma = final_array @ np.array([.2126,.7152,.0722])
    union = np.zeros((height,width),dtype=np.uint8)
    geometry = []
    overlap_pixels = 0
    areas = []
    for p in placements:
        fullmask = Image.new('L',canvas.size,0)
        fullmask.paste(p['sprite'].getchannel('A'),(p['left'],p['top']))
        mask = fullmask.resize((width,height),Image.Resampling.LANCZOS)
        ma = np.asarray(mask)
        solid = ma >= 128
        overlap_pixels += int(np.count_nonzero(solid & (union>0)))
        union[solid] = 1
        yy,xx = np.where(solid)
        bbox = [int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1]
        area = int(solid.sum())
        areas.append(area)
        mpath = f"artifacts/masks/aircraft_{p['index']+1:02d}.png"
        mask.crop(tuple(bbox)).save(ROOT / mpath)
        # The nose neighborhood excludes amount labels (which begin 31 pixels after the nose).
        rx0,rx1 = math.floor(p['nose']-20),math.ceil(p['nose']+14)
        ry0,ry1 = p['y']-9,p['y']+10
        fy,fx = np.where(luma[ry0:ry1,rx0:rx1] >= 165)
        visible_nose = int(rx0+fx.max()) if len(fx) else None
        observed_value = (visible_nose-x0)/px_per_unit if visible_nose is not None else None
        geometry.append({
            'category':p['category'],'value_usd_billion':p['value'],
            'target_nose_x':round(p['nose'],6),'row_center_y':p['y'],
            'mask_rightmost_x':int(xx.max()),'visible_nose_x_luminance165':visible_nose,
            'observed_value_from_final_nose':round(observed_value,6) if observed_value is not None else None,
            'nose_error_pixels':round(visible_nose-p['nose'],6) if visible_nose is not None else None,
            'uniform_source_scale':p['scale'],'solid_area_pixels_alpha128':area,
            'alpha_integral_pixels':round(float(ma.sum()/255),6),
            'mask_bbox_xyxy':bbox,'mask_path':mpath,
            'value_label':f"${p['value']:.1f} bn",'value_label_bbox_xyxy':label_boxes[p['index']]
        })
    for g in geometry:
        expected = g['value_usd_billion']/vmax
        actual = g['solid_area_pixels_alpha128']/areas[0]
        g['expected_area_ratio_to_largest'] = round(expected,8)
        g['measured_area_ratio_to_largest'] = round(actual,8)
        g['area_ratio_relative_error_percent'] = round(100*(actual/expected-1),6)
    measurement = {
        'canvas_pixels':[width,height],
        'data_file':'data.json','axis_minimum':0,'axis_maximum':max_axis,
        'common_zero_x':x0,'axis_maximum_x':xmax,'pixels_per_usd_billion':px_per_unit,
        'source_aircraft':'assets/aircraft.png',
        'area_measurement':'Count of actual final-resolution transformed entity alpha >= 128; trails and shadows excluded.',
        'visible_nose_measurement':'Rightmost final.png pixel of luminance >= 165 in a local nose ROI that excludes numeric labels.',
        'aircraft_overlap_pixels':overlap_pixels,
        'all_label_boxes_inside_canvas':all(0<=b[0]<b[2]<=width and 0<=b[1]<b[3]<=height for b in label_boxes),
        'rows':geometry
    }
    (ROOT/'artifacts/geometry_checks.json').write_text(json.dumps(measurement,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'canvas':[width,height],'aircraft_count':len(geometry),
        'maximum_nose_error_pixels':max(abs(g['nose_error_pixels']) for g in geometry),
        'maximum_area_ratio_relative_error_percent':max(abs(g['area_ratio_relative_error_percent']) for g in geometry),
        'aircraft_overlap_pixels':overlap_pixels},ensure_ascii=False))


if __name__ == '__main__':
    main()
