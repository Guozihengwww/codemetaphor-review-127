"""Offline renderer. Run only through capture_render.py for recorded candidates."""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy import ndimage
from geometry import build_regions, DROP_BOX, SUPERSAMPLE, START_DEGREES

ROOT = Path(__file__).resolve().parent
INK = '#163B55'
MUTED = '#567078'
PALETTE = ['#168CAA', '#67A58A', '#D85D6E', '#DE9A36', '#C9BA46', '#294579']

def rgb(value):
    return tuple(int(value[i:i+2], 16) for i in (1,3,5))

def save_json(path, obj):
    (ROOT / path).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def main():
    inputs = json.loads((ROOT/'input.json').read_text(encoding='utf-8'))
    brief = json.loads((ROOT/inputs['brief']).read_text(encoding='utf-8'))
    data = json.loads((ROOT/inputs['data']).read_text(encoding='utf-8'))
    core = json.loads((ROOT/inputs['critical_metaphor_contract']).read_text(encoding='utf-8'))
    assert core['topic_zh'] == data['topic_zh']
    assert len(data['series']) == 1 and len(data['categories']) == 6
    assert data['synthetic'] and data['series'][0]['unit'] == '% of synthetic total'
    values = data['series'][0]['values']
    assert abs(sum(values)-100) < 1e-8
    width, height = brief['width'], brief['height']
    s = SUPERSAMPLE

    # The generated paper sample is mirrored into a seamless repeating sheet.
    tile = Image.open(ROOT/'assets/paper_tile.png').convert('RGB')
    tw, th = tile.size
    sheet = Image.new('RGB', (tw*2, th*2))
    sheet.paste(tile,(0,0)); sheet.paste(ImageOps.mirror(tile),(tw,0))
    sheet.paste(ImageOps.flip(tile),(0,th)); sheet.paste(ImageOps.flip(ImageOps.mirror(tile)),(tw,th))
    paper = Image.new('RGB',(width,height))
    for y in range(0,height,th*2):
        for x in range(0,width,tw*2):
            paper.paste(sheet,(x,y))
    paper = Image.blend(paper, Image.new('RGB',paper.size,'#F5F0E5'), .36)
    canvas = paper.resize((width*s,height*s),Image.Resampling.BICUBIC)

    regions, mask, stats, pivot = build_regions(ROOT, values)
    x0,y0,x1,y1 = DROP_BOX
    w,h=x1-x0,y1-y0
    density = np.asarray(Image.open(ROOT/'assets/drop_print_density.png').resize((w*s,h*s),Image.Resampling.BICUBIC),dtype=np.float32)/255
    texture_tint = .025 + .48*density
    palette = np.array([rgb(c) for c in PALETTE]+[(0,0,0)],dtype=np.float32)
    safe_region = np.where(mask, regions, 6)
    flat = palette[safe_region]
    local_paper = np.asarray(canvas.crop((x0*s,y0*s,x1*s,y1*s)),dtype=np.float32)
    drop = np.rint(flat*(1-texture_tint[...,None])+local_paper*texture_tint[...,None]).astype(np.uint8)
    local_paper = np.uint8(local_paper)
    local_paper[mask] = drop[mask]
    canvas.paste(Image.fromarray(local_paper),(x0*s,y0*s))

    draw = ImageDraw.Draw(canvas)
    fonts = {}
    text_boxes = []
    def font(size, bold=False):
        key=(size,bold)
        if key not in fonts:
            name='NimbusSans-Bold.otf' if bold else 'NimbusSans-Regular.otf'
            fonts[key]=ImageFont.truetype(str(ROOT/'assets/fonts'/name),round(size*s))
        return fonts[key]

    def text(label,xy,size=30,fill=INK,bold=False,align='left',tracking=0,role='annotation'):
        f=font(size,bold)
        advances=[draw.textlength(ch,font=f) for ch in label]
        length=(sum(advances)+(len(label)-1)*tracking*s) if tracking else draw.textlength(label,font=f)
        x,y=xy[0]*s,xy[1]*s
        if align=='center':x-=length/2
        elif align=='right':x-=length
        start=x
        if tracking:
            baseline=y-draw.textbbox((0,0),label,font=f,anchor='ls')[1]
            for ch,advance in zip(label,advances):
                draw.text((x,baseline),ch,font=f,fill=fill,anchor='ls')
                x+=advance+tracking*s
        else:draw.text((x,y),label,font=f,fill=fill,anchor='lt')
        bbox=draw.textbbox((start,y),label,font=f,anchor='lt')
        text_boxes.append({'text':label,'role':role,'bbox_xyxy':[round(start/s,2),round(bbox[1]/s,2),round((start+length)/s,2),round(bbox[3]/s,2)]})
        return length/s

    def line(points,fill=INK,weight=1.65):
        draw.line([(round(x*s),round(y*s)) for x,y in points],fill=fill,width=max(1,round(weight*s)),joint='curve')

    # All wording, numbers and leaders are code-controlled, separate from imagery.
    draw.rounded_rectangle((572*s,67*s,928*s,119*s),radius=3*s,fill=INK)
    text(data['visible_notice'],(750,82),24,fill='#F8F2E6',bold=True,align='center',tracking=2.3,role='synthetic_notice')
    text('HOUSEHOLD',(750,178),43,bold=True,align='center',tracking=9,role='title')
    text('WATER USE',(750,250),150,align='center',tracking=-2,role='title')
    text('COMPOSITION',(750,421),29,bold=True,align='center',tracking=5.2,role='title')
    text(data['series'][0]['unit'],(750,475),25,align='center',tracking=.7,fill=MUTED,role='unit')

    # A chosen outer ray for Toilet keeps its label clear of the pointed tip.
    def edge_at(deg):
        px,py=pivot[0]-x0,pivot[1]-y0
        angle=math.radians(deg)
        rr=np.linspace(0,max(w,h)*1.6,20000)
        xx=np.rint((px+np.cos(angle)*rr)*s).astype(int)
        yy=np.rint((py+np.sin(angle)*rr)*s).astype(int)
        good=(xx>=0)&(xx<w*s)&(yy>=0)&(yy<h*s)
        valid=np.zeros(len(rr),bool);valid[good]=mask[yy[good],xx[good]]
        r=rr[np.flatnonzero(valid)[-1]]
        return [x0+px+np.cos(angle)*r,y0+py+np.sin(angle)*r]

    anchors=[edge_at(-72),stats[1]['edge_xy'],stats[2]['edge_xy'],stats[3]['edge_xy'],stats[4]['edge_xy'],stats[5]['edge_xy']]
    layout=[
        {'xy':(1050,762),'lines':['Toilet'],'number_y':811,'leader':[(983,862),(1025,862)]},
        {'xy':(1222,1138),'lines':['Clothes','Washer'],'number_y':1233,'leader':[(1202,1243)]},
        {'xy':(1124,1603),'lines':['Shower'],'number_y':1652,'leader':[(1050,1708),(1100,1708)]},
        {'xy':(104,1603),'lines':['Faucet'],'number_y':1652,'leader':[(367,1708),(318,1708)]},
        {'xy':(96,1281),'lines':['Leaks'],'number_y':1330,'leader':[(288,1385),(271,1385)]},
        {'xy':(132,984),'lines':['Other'],'number_y':1033,'leader':[(348,1092),(310,1092)]},
    ]
    bindings=[]
    for i,(category,value,spec,anchor) in enumerate(zip(data['categories'],values,layout,anchors)):
        assert ' '.join(spec['lines'])==category
        points=[anchor]+spec['leader']
        line(points)
        r=3.0*s
        ax,ay=anchor[0]*s,anchor[1]*s
        draw.ellipse((ax-r,ay-r,ax+r,ay+r),fill=INK)
        tx,ty=spec['xy']
        for k,label in enumerate(spec['lines']):
            text(label,(tx,ty+39*k),32,bold=True,role='category')
        n=f'{value:.1f}'
        advance=text(n,(tx,spec['number_y']),68,tracking=-1.1,role='value')
        text('%',(tx+advance+4,spec['number_y']+10),37,role='value_unit')
        bindings.append({'category':category,'value':value,'display':n+'%',
                         'region_index':i,'leader_xy':points,'colour':PALETTE[i]})

    line([(96,1810),(1404,1810)],fill='#ADB4AB',weight=1)
    text('ONE DROP, SIX USES',(96,1849),23,bold=True,tracking=2.4)
    text('Area of each coloured region = share of the synthetic total.',(96,1898),27)
    text('Illustrative composition; not observed household statistics.',(96,1944),23,fill=MUTED)
    text(f'{sum(values):.1f}%',(1404,1840),48,align='right',tracking=-.8,role='total')
    text('TOTAL',(1404,1901),18,align='right',tracking=2.6,fill=MUTED)

    # Preserve exact per-region coverage for audit, independent of label content.
    coverage=np.stack([(regions==i).reshape(h,s,w,s).sum(axis=(1,3)) for i in range(len(values))]).astype(np.uint8)
    np.savez_compressed(ROOT/'artifacts/region_coverage.npz',counts=coverage,box=np.array(DROP_BOX),supersample=s,palette=np.array([rgb(c) for c in PALETTE],dtype=np.uint8))
    total_pixels=int(mask.sum())
    mask_components=ndimage.label(mask)[1]
    mask_holes=int((ndimage.binary_fill_holes(mask)&~mask).sum())
    for st,category in zip(stats,data['categories']): st['category']=category
    save_json('artifacts/geometry_check.json',{
        'method':'Angular boundaries solved from cumulative area of the actual supersampled droplet silhouette, not circular pie angles.',
        'category_order_clockwise':data['categories'],'start_degrees':START_DEGREES,
        'shared_pivot_xy':list(pivot),'drop_bbox_xyxy':list(DROP_BOX),'supersample':s,
        'mask_area_pixels':total_pixels,'mask_components':int(mask_components),'mask_holes':mask_holes,
        'unassigned_inside_pixels':int(((regions==255)&mask).sum()),
        'multiple_assignment_pixels':0,'max_area_error_percentage_points':max(abs(st['share_pct']-v) for st,v in zip(stats,values)),
        'regions':stats,'label_bindings':bindings,'text_boxes':text_boxes,
        'visible_generated_material_zh':'生图的纸张材质铺满背景，生图水滴轮廓与印刷颗粒用于全部六个面积分区。'
    })
    canvas.resize((width,height),Image.Resampling.LANCZOS).save(ROOT/'final.png')
    print(json.dumps({'size':[width,height],'shares_pct':[round(v['share_pct'],6) for v in stats],
                     'maximum_error_pp':max(abs(st['share_pct']-v) for st,v in zip(stats,values)),
                     'mask_holes':mask_holes},ensure_ascii=False))

if __name__=='__main__': main()
