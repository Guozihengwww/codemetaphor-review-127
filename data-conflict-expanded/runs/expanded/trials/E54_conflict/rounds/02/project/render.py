#!/usr/bin/env python3
"""Offline, deterministic infographic renderer. Run with the frozen capture wrapper."""
from pathlib import Path
import json, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent
S = 3
INK = '#263F4C'
MUTED = '#607985'
PALETTE = ['#D7513B', '#DFAE28', '#4799C3', '#79A547', '#E88935']


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def rgb(hex_color):
    return tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))


def main():
    inp = read_json('input.json')
    brief = read_json(inp['brief'])
    data = read_json(inp['data'])
    core = read_json(inp['critical_metaphor_contract'])
    # All frozen input dependencies remain local, including the design reference.
    with Image.open(ROOT / inp['reference']) as reference:
        reference_size = list(reference.size)
    assert data['synthetic'] and core['topic_zh'] == data['topic_zh']
    W, H = brief['width'], brief['height']
    categories = data['categories']
    series = {s['id']: s for s in data['series']}
    ceo = series['ceo_pay']['values']
    worker = series['worker_median_pay']['values']
    assert len(categories) == len(ceo) == len(worker) == 5
    assert data['worker_axis'] == {'minimum': 0, 'maximum': 80000}
    assert data['value_axis']['scale'] == 'linear'
    assert data['value_axis']['maximum'] == data['ceo_area_reference']

    # A restrained pale sky with a fixed seed, not an image-generation dependency.
    yy, xx = np.mgrid[0:H, 0:W]
    top = np.array([250., 251., 247.])
    bottom = np.array([228., 242., 243.])
    t = np.clip((yy - 290) / 1410, 0, 1)[..., None]
    sky = top[None, None, :] * (1-t) + bottom[None, None, :] * t
    noise = np.random.default_rng(54).normal(0, 0.36, (H, W, 1))
    sky = np.uint8(np.clip(sky + noise, 0, 255))
    canvas = Image.fromarray(sky, 'RGB').convert('RGBA').resize((W*S, H*S), Image.Resampling.BICUBIC)
    draw = ImageDraw.Draw(canvas)
    fonts = {}
    text_boxes = []

    def font(size, weight='regular'):
        key = (size, weight)
        if key not in fonts:
            names = {'regular':'Sans-Regular.otf', 'bold':'Sans-Bold.otf', 'italic':'Sans-Italic.otf'}
            fonts[key] = ImageFont.truetype(str(ROOT / 'assets/fonts' / names[weight]), round(size*S))
        return fonts[key]

    def text(x, y, value, size=28, color=INK, weight='regular', align='left', tracking=0):
        f = font(size, weight)
        if tracking:
            width = sum(draw.textlength(c, font=f) for c in value) + tracking*S*max(0,len(value)-1)
        else:
            width = draw.textlength(value, font=f)
        x0 = x*S - (width/2 if align=='center' else width if align=='right' else 0)
        y0 = y*S
        if tracking:
            at = x0
            for c in value:
                draw.text((at,y0), c, font=f, fill=color, anchor='lt')
                at += draw.textlength(c, font=f)+tracking*S
        else:
            draw.text((x0,y0), value, font=f, fill=color, anchor='lt')
        bb = draw.textbbox((x0,y0), value, font=f, anchor='lt')
        text_boxes.append({'text':value,'bbox':[round(bb[0]/S,2),round(bb[1]/S,2),round((x0+width)/S,2),round(bb[3]/S,2)]})
        return width/S

    def line(points, fill, width=1):
        draw.line([(round(x*S),round(y*S)) for x,y in points], fill=fill, width=max(1,round(width*S)))

    def rect(box, fill, outline=None, width=1, radius=0):
        box = tuple(round(v*S) for v in box)
        if radius:
            draw.rounded_rectangle(box, radius=round(radius*S), fill=fill, outline=outline, width=round(width*S))
        else:
            draw.rectangle(box, fill=fill, outline=outline, width=round(width*S))

    def ellipse(box, fill, outline=None, width=1):
        draw.ellipse(tuple(round(v*S) for v in box), fill=fill, outline=outline, width=max(1,round(width*S)))

    def polygon(points, fill):
        draw.polygon([(round(x*S),round(y*S)) for x,y in points], fill=fill)

    def dashed(x1,y1,x2,y2,color,width=1,dash=7,gap=9):
        dx,dy=x2-x1,y2-y1
        length=math.hypot(dx,dy)
        if length==0:return
        for distance in np.arange(0,length,dash+gap):
            end=min(distance+dash,length)
            line([(x1+dx*distance/length,y1+dy*distance/length),(x1+dx*end/length,y1+dy*end/length)],color,width)

    def cloud(cx, cy, width, height, opacity=88):
        layer=Image.new('RGBA',canvas.size,(0,0,0,0)); cd=ImageDraw.Draw(layer)
        fill=(255,255,252,opacity)
        for dx,dy,rx,ry in [(-.24,.05,.23,.37),(0,-.13,.24,.56),(.26,.04,.26,.41),(.03,.19,.39,.32)]:
            cd.ellipse(tuple(round(v*S) for v in [cx+(dx-rx)*width,cy+(dy-ry)*height,cx+(dx+rx)*width,cy+(dy+ry)*height]),fill=fill)
        canvas.alpha_composite(layer.filter(ImageFilter.GaussianBlur(1.2*S)))

    # Header.
    rect((88,62,388,108),'#C94E36',radius=8)
    text(238,73,data['visible_notice'],24,'#FFFFFF','bold','center',1.15)
    text(1892,77,'FIVE COMPANIES  /  TWO PAY MEASURES',22,MUTED,'bold','right',1.0)
    text(88,148,'Executive and worker pay',91,INK,'bold')
    text(92,262,'Annual CEO compensation and median worker salary  •  USD/year',31,MUTED)
    line([(90,311),(1890,311)],'#CAD9D8',1.5)

    # Shared sky decorations are never quantitative.
    cloud(834,506,180,72,80)
    cloud(1410,764,126,50,75)
    cloud(303,1253,185,73,125)
    cloud(1146,1375,154,61,104)
    cloud(1768,941,132,51,100)

    # Legend: clearly separate the area and height channels, without extra data objects.
    xg=1078
    text(xg,368,'READING THE BALLOONS',23,MUTED,'bold',tracking=1.5)
    for i,c in enumerate(PALETTE):
        rect((xg+598+i*31,367,xg+620+i*31,374),c,radius=3)
    text(xg,424,'01',24,'#BF684C','bold')
    text(xg+62,416,'Envelope area',35,INK,'bold')
    text(xg+62,465,'CEO compensation (USD/year)',28,INK)
    text(xg+62,507,'Area is proportional; $22,000,000 is the reference.',25,MUTED)
    text(xg,572,'02',24,'#508893','bold')
    text(xg+62,564,'Basket height',35,INK,'bold')
    text(xg+62,613,'Median worker salary (USD/year)',28,INK)
    text(xg+62,655,'Linear scale from $0 to $80,000; equal-size baskets.',25,MUTED)
    text(xg+62,696,'Ropes and baskets are not part of the area measure.',23,MUTED,'italic')

    zero_y=1648.0
    salary_span=640.0
    salary_min=float(data['worker_axis']['minimum'])
    salary_max=float(data['worker_axis']['maximum'])
    salary_y=lambda value: zero_y-(value-salary_min)/(salary_max-salary_min)*salary_span
    axis_x=185.0
    plot_right=1912.0
    centers=[520.,1015.,1325.,1580.,1815.]
    text(74,918,'MEDIAN WORKER',20,MUTED,'bold',tracking=.6)
    text(74,949,'SALARY · USD/YEAR',19,MUTED)
    for value in (0,20000,40000,60000,80000):
        y=salary_y(value)
        if value:
            dashed(axis_x,y,plot_right,y,'#B8D0D3',1,5,11)
        line([(axis_x-8,y),(axis_x,y)],'#70909A',1.5)
        text(axis_x-20,y-11,f'${value:,}',23,MUTED,align='right')
    line([(axis_x,salary_y(80000)),(axis_x,zero_y)],'#8DA8AE',1.4)

    # The ground's upper edge is the common zero line. Landscape thickness is non-data.
    rect((185,zero_y,1912,1785),'#E2EBCD')
    polygon([(185,1724),(430,1698),(744,1741),(1090,1704),(1420,1729),(1674,1695),(1912,1727),(1912,1785),(185,1785)],'#D4E1B9')
    polygon([(185,1763),(405,1746),(662,1767),(1000,1741),(1345,1766),(1718,1743),(1912,1766),(1912,1785),(185,1785)],'#CBDCAF')
    line([(185,zero_y),(1912,zero_y)],'#7F9C6A',2.3)

    def tree(x,y,h):
        ellipse((x-16,y-2,x+22,y+5),'#BFCE9F')
        ellipse((x-h*.29,y-h*.94,x+h*.29,y-h*.25),'#AEC66B')
        ellipse((x-h*.20,y-h*1.13,x+h*.17,y-h*.42),'#BCD779')
        line([(x,y),(x,y-h*.75)],'#718644',2)
        line([(x,y-h*.36),(x-h*.18,y-h*.56)],'#718644',1.8)
        line([(x,y-h*.56),(x+h*.15,y-h*.76)],'#718644',1.6)

    tree(266,1711,47)
    tree(1436,1703,38)
    tree(1900,1718,51)

    # One texture and one silhouette. Every data object is transformed isotropically.
    template=Image.open(ROOT/'assets/envelope_template.png').convert('RGBA')
    source=np.asarray(template)
    mask=Image.open(ROOT/'assets/envelope_mask.png').convert('L')
    assert template.size==mask.size
    source_area=float(np.asarray(mask,dtype=float).sum()/255)
    sw,sh=template.size
    ref_h=680.0
    ref_scale=ref_h/sh
    reference_area=source_area*ref_scale**2
    luminance=(.2126*source[:,:,0]+.7152*source[:,:,1]+.0722*source[:,:,2])/255
    # Preserve the generated seams and illumination when applying the category colors.
    shade=np.clip((luminance-.22)/.73,0,1)
    geometry=[]
    envelope_layers=[]
    for i,(category,cp,wp,cx,color) in enumerate(zip(categories,ceo,worker,centers,PALETTE)):
        by=salary_y(wp)
        unit_scale=math.sqrt(float(cp)/float(data['ceo_area_reference']))
        k=ref_scale*unit_scale
        ew,eh=sw*k,sh*k
        left=cx-ew/2
        bottom=by-42.0
        top=bottom-eh
        # Draw the height guide BEFORE the object, terminating precisely at its basket center.
        dashed(cx,by,cx,zero_y,'#8BBEC6',1.6,4,7)
        ellipse((cx-21,zero_y+13,cx+21,zero_y+21),'#BDCF9C')
        ellipse((cx-5,zero_y-5,cx+5,zero_y+5),'#F8FAF0',color,1.8)
        text(cx,1735,category,28,INK,'bold','center')

        # Thin suspension ropes are independent of envelope area.
        # Attach to the tapered fabric near its bottom and to the fixed basket corners.
        attach_y=top+eh*.925
        half_attach=ew*.116
        basket_top=by-10
        line([(cx-half_attach,attach_y),(cx-10.5,basket_top)],'#7B745A',1.5)
        line([(cx+half_attach,attach_y),(cx+10.5,basket_top)],'#7B745A',1.5)
        line([(cx-ew*.052,bottom-eh*.011),(cx-5,basket_top)],'#A19775',.8)
        line([(cx+ew*.052,bottom-eh*.011),(cx+5,basket_top)],'#A19775',.8)

        base=np.asarray(rgb(color),dtype=float)
        col=base[None,None,:]*(.50+.60*shade[:,:,None])+30*shade[:,:,None]**5
        colored=np.dstack([np.uint8(np.clip(col,0,255)),source[:,:,3]])
        tex=Image.fromarray(colored,'RGBA')
        # Padding permits fractional, isotropic transforms without clipping AA edges.
        x0=math.floor(left)-3; y0=math.floor(top)-3
        patch_w=math.ceil(ew)+6; patch_h=math.ceil(eh)+6
        affine=(1/(S*k),0,(x0-left)/k,0,1/(S*k),(y0-top)/k)
        patch=tex.transform((patch_w*S,patch_h*S),Image.Transform.AFFINE,affine,Image.Resampling.BICUBIC)
        projected=mask.transform((patch_w*S,patch_h*S),Image.Transform.AFFINE,affine,Image.Resampling.BICUBIC)
        patch.putalpha(projected)
        canvas.alpha_composite(patch,(x0*S,y0*S))
        draw=ImageDraw.Draw(canvas)
        final_mask=projected.resize((patch_w,patch_h),Image.Resampling.LANCZOS)
        final_mask.save(ROOT/f'artifacts/envelope_coverage_{i+1:02d}.png')
        ma=np.asarray(final_mask,dtype=float)/255
        visible_area=float(ma.sum())
        binary=ma>=.5
        nonzero=np.argwhere(binary)
        visible_bbox=[int(nonzero[:,1].min()+x0),int(nonzero[:,0].min()+y0),int(nonzero[:,1].max()+x0+1),int(nonzero[:,0].max()+y0+1)]
        expected_area=reference_area*float(cp)/data['ceo_area_reference']
        # Actual basket drawing: every category uses the same 28 by 20 px body.
        # The body is centrally symmetric in y, so its center is the encoded anchor.
        rect((cx-14,by-10,cx+14,by+10),'#907445',outline='#5E553E',width=1.2,radius=2)
        for dy in (-5,0,5):
            line([(cx-12,by+dy),(cx+12,by+dy)],'#BCA170',.7)
        for dx in (-8,0,8):
            line([(cx+dx,by-8),(cx+dx,by+8)],'#6A5B3E',.7)
        line([(cx-13.5,by-9),(cx+13.5,by-9)],'#DAC598',1.7)
        # Small center dot makes the salary anchor unambiguous.
        ellipse((cx-2,by-2,cx+2,by+2),'#FBFAF2')
        if i<3:
            label_x=cx+31
            line([(cx+15,by),(cx+24,by)],'#6D8E94',1.4)
            text(label_x,by-15,f'${wp:,}',33,INK,'bold')
            text(label_x,by+24,'median worker',21,MUTED)
        else:
            label_x=cx-31
            line([(cx-24,by),(cx-15,by)],'#6D8E94',1.4)
            text(label_x,by-15,f'${wp:,}',33,INK,'bold','right')
            text(label_x,by+24,'median worker',21,MUTED,align='right')

        geometry.append({
            'category':category,'ceo_usd_year':cp,'worker_usd_year':wp,
            'isotropic_scale_from_template':k,'scale_relative_to_22000000':unit_scale,
            'expected_area_px2':expected_area,'raster_alpha_area_px2':visible_area,
            'area_relative_error_percent':100*(visible_area/expected_area-1),
            'area_half_alpha_pixel_count':int(binary.sum()),
            'visible_bbox':visible_bbox,'continuous_bbox':[left,top,left+ew,bottom],
            'basket_center':[cx,by],'basket_body_bounds':[cx-14,by-10,cx+14,by+10],
            'height_above_zero_px':zero_y-by,
            'envelope_mask_path':f'artifacts/envelope_coverage_{i+1:02d}.png',
            'mask_origin':[x0,y0],
            'all_envelope_pixels_within_canvas':0<=x0 and 0<=y0 and x0+patch_w<=W and y0+patch_h<=H,
        })
        envelope_layers.append((top,ew,eh,bottom))

    # Quantitative CEO labels, added by code rather than generated into the texture.
    text(520,536,categories[0],34,'#FFF9E9','bold','center')
    text(520,590,f'${ceo[0]:,}',51,'#FFFDF2','bold','center')
    text(520,659,'CEO compensation',26,'#FFF4DE',align='center')
    text(1015,877,categories[1],29,'#3E432A','bold','center')
    text(1015,923,f'${ceo[1]:,}',38,'#343C24','bold','center')
    text(1015,971,'CEO compensation',22,'#3E432A',align='center')
    # The three small envelopes keep their true area; external callouts preserve legibility.
    line([(1393,867),(1427,847),(1445,847)],'#81A7B6',1.5)
    text(1454,812,categories[2],30,INK,'bold')
    text(1454,855,f'CEO  ${ceo[2]:,}',29,'#307393','bold')
    top_d=envelope_layers[3][0]
    text(1580,top_d-99,categories[3],30,INK,'bold','center')
    text(1580,top_d-57,f'CEO  ${ceo[3]:,}',27,'#5D7E36','bold','center')
    line([(1580,top_d-16),(1580,top_d-6)],'#8CA770',1.2)
    top_e=envelope_layers[4][0]
    text(1815,top_e-101,categories[4],30,INK,'bold','center')
    text(1815,top_e-58,f'CEO  ${ceo[4]:,}',27,'#A95F28','bold','center')
    line([(1815,top_e-17),(1815,top_e-5)],'#BE916A',1.2)

    # A short, data-grounded reading note occupies quiet sky below the main envelopes.
    rect((694,1422,1348,1566),'#F6FAF5',outline='#D5E2D9',width=1,radius=12)
    rect((694,1436,699,1551),'#4E8F9B',radius=2)
    text(721,1444,'A smaller envelope, a higher salary.',28,INK,'bold')
    highest_worker=int(np.argmax(worker))
    assert highest_worker==2  # frozen case: prose below is checked against this input
    text(721,1485,f'{categories[highest_worker]} has the highest median worker pay:',24,MUTED)
    text(721,1518,f'${worker[highest_worker]:,} per year, with ${ceo[highest_worker]:,} in CEO compensation.',24,MUTED)

    # Footer explains the scale honestly, including very small visible marks.
    text(187,1817,'One landscape. Two independent scales.',33,INK,'bold')
    text(187,1868,'Envelope area uses one shared shape, scaled without a minimum size. Basket centers locate worker salaries.',25,MUTED)
    text(187,1908,'Synthetic data for illustration only. Company labels are fictional; all values are annual US dollars.',24,MUTED)
    text(1890,1956,'SOURCE  LOCAL SYNTHETIC DATA',18,MUTED,'bold','right',.8)

    final=canvas.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
    final.save(ROOT/'final.png',optimize=True)
    out={
        'canvas':[W,H],'supersampling':S,'reference_size':reference_size,
        'worker_axis':{'minimum':salary_min,'maximum':salary_max,'zero_y':zero_y,'max_y':salary_y(salary_max),'span_px':salary_span},
        'ceo_reference_value':data['ceo_area_reference'],'template_pixel_area':source_area,
        'reference_rendered_area_px2':reference_area,'template_reference_height':ref_h,
        'categories':geometry,'text_bounds':text_boxes,
        'fonts_local':True,'network_or_model_calls_in_renderer':False,
    }
    (ROOT/'artifacts/geometry.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'canvas':[W,H],'ceo_area_measurements':[{k:v for k,v in g.items() if k in ['category','raster_alpha_area_px2','area_relative_error_percent','visible_bbox','basket_center']} for g in geometry]},ensure_ascii=False))

if __name__=='__main__':
    main()
