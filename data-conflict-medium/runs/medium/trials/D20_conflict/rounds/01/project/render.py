"""Reproduce this single infographic using only local data and generated assets.

Run: PYTHON render.py
The experiment launches complete renders via capture_render.py.
All airplane transforms use one scale and one nose anchor. No size floor is used.
"""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT/'data.json').read_text(encoding='utf-8'))
BRIEF = json.loads((ROOT/'brief.json').read_text(encoding='utf-8'))
META = json.loads((ROOT/'assets/aircraft_geometry.json').read_text(encoding='utf-8'))
W, H = BRIEF['width'], BRIEF['height']
S = 3
X0, X1 = 490.0, 1810.0
VMIN, VMAX = DATA['value_axis']['minimum'], DATA['value_axis']['maximum']
VALUES = DATA['series'][0]['values']
ROW_Y = [600 + 170*i for i in range(len(VALUES))]
MAX_PLANE_LENGTH = 380.0
PAPER = '#F4EDD5'
WHITE = '#F4F7F9'
MUTED = '#95ACBE'
GRID = '#426077'
FONT_DIR = ROOT/'assets/fonts'
FONTS = {}

def font(size, bold=False, cjk=False):
    name = ('NotoSansCJK-Bold.ttc' if bold else 'NotoSansCJK-Regular.ttc') if cjk else ('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')
    key=(name,size)
    if key not in FONTS:
        FONTS[key]=ImageFont.truetype(str(FONT_DIR/name),round(size*S),index=2 if cjk else 0)
    return FONTS[key]

def pos(v):
    return int(round(v*S))

def coords(box):
    return tuple(pos(v) for v in box)

def text(draw,xy,label,size,fill=WHITE,bold=False,cjk=False,anchor='lt'):
    draw.text((xy[0]*S,xy[1]*S),str(label),font=font(size,bold,cjk),fill=fill,anchor=anchor)

def line(draw,points,fill,width=1):
    draw.line([(pos(x),pos(y)) for x,y in points],fill=fill,width=max(1,pos(width)))

def rect(draw,box,fill,outline=None,radius=0,width=1):
    if radius:
        draw.rounded_rectangle(coords(box),radius=pos(radius),fill=fill,outline=outline,width=pos(width))
    else:
        draw.rectangle(coords(box),fill=fill,outline=outline,width=pos(width))

def rgba(hexcolor,a=255):
    return tuple(int(hexcolor[i:i+2],16) for i in (1,3,5))+(a,)

def num(v):
    return str(v)  # Never rounds a positive supplied value to zero.

def main():
    assert W==2000 and H==2000 and len(DATA['categories'])==len(VALUES)
    assert DATA['value_axis']['scale']=='linear'
    out = ROOT/'artifacts'
    geomdir=out/'geometry'
    geomdir.mkdir(parents=True,exist_ok=True)
    # A generated, subdued world map fills the background. No reference pixels.
    background=Image.open(ROOT/'assets/raw_02.png').convert('RGB').resize((W*S,H*S),Image.Resampling.LANCZOS)
    background=Image.blend(Image.new('RGB',background.size,'#101F30'),background,0.69)
    im=background.convert('RGBA')
    draw=ImageDraw.Draw(im)
    # Header.
    rect(draw,(110,68,426,114),'#24384A',outline='#506577',radius=10,width=1)
    text(draw,(130,78),DATA['visible_notice'],25,PAPER,True)
    text(draw,(1890,83),'AVIATION / 07',22,MUTED,anchor='rt')
    text(draw,(108,150),DATA['title'],76,WHITE,True,True)
    text(draw,(113,263),'AIRLINE MARKET CAPITALIZATION',29,MUTED,True)
    text(draw,(1890,268),'USD billion',29,PAPER,anchor='rt')
    line(draw,[(110,332),(1890,332)],'#465869',1.3)
    # Shared linear scale. Guides are subdued and sit below every object.
    text(draw,(110,370),'AIRLINE',23,MUTED,True)
    text(draw,(490,367),'机鼻位置',23,MUTED,cjk=True)
    text(draw,(1890,369),'0—35  /  LINEAR SCALE',20,MUTED,anchor='rt')
    grid_layer=Image.new('RGBA',im.size)
    gd=ImageDraw.Draw(grid_layer)
    for tick in range(0,36,5):
        x=X0+(tick-VMIN)/(VMAX-VMIN)*(X1-X0)
        line(gd,[(x,451),(x,1695)],rgba('#7795AA',110 if tick==0 else 42),1.6 if tick==0 else 0.9)
        line(gd,[(x,426),(x,441)],rgba('#A9BFCD',180),1.3)
        text(gd,(x,419),tick,24,PAPER if tick==0 else MUTED,anchor='mb')
    line(gd,[(X0,439),(X1,439)],rgba('#7993A6',105),1)
    for y in ROW_Y:
        line(gd,[(110,y+83),(1890,y+83)],rgba('#829AAE',32),0.8)
    im=Image.alpha_composite(im,grid_layer)
    # Aircraft template: the same solid-body mask and material for all values.
    master=Image.open(ROOT/'assets/aircraft_master.png').convert('RGBA')
    mw,mh=master.size
    anchors=META['nose_anchor_master_px']
    records=[]
    for i,(category,value,y) in enumerate(zip(DATA['categories'],VALUES,ROW_Y)):
        endpoint=X0+(value-VMIN)/(VMAX-VMIN)*(X1-X0)
        scale=MAX_PLANE_LENGTH/mw*math.sqrt(value/max(VALUES))
        length=mw*scale
        left=endpoint-anchors[0]*scale
        top=y-anchors[1]*scale
        # The fade is a non-quantitative wake. The nose, NOT the wake start,
        # encodes horizontal position. At small values the body crosses zero.
        wake_start=min(X0,left-0.32*length)
        wake_end=left+0.15*length
        wake_half_height=0.065*length
        wx0=int(math.floor(wake_start))-2
        wx1=int(math.ceil(wake_end))+2
        wy0=int(math.floor(y-wake_half_height))-2
        wy1=int(math.ceil(y+wake_half_height))+2
        xs=(np.arange((wx1-wx0)*S)+0.5)/S+wx0
        ys=(np.arange((wy1-wy0)*S)+0.5)/S+wy0
        along=np.clip((xs-wake_start)/max(1e-8,wake_end-wake_start),0,1)
        across=np.clip(1-np.abs(ys-y)/wake_half_height,0,1)
        alpha=186.0*(along[None,:]**0.75)*(across[:,None]**0.45)
        alpha[:,(xs<wake_start)|(xs>wake_end)]=0
        wake=np.zeros((len(ys),len(xs),4),dtype=np.uint8)
        wake[:,:,:3]=(225,238,247)
        wake[:,:,3]=alpha.astype('uint8')
        im.alpha_composite(Image.fromarray(wake),(pos(wx0),pos(wy0)))
        # Uniform affine sampling avoids independent width/height rounding.
        bx=math.floor(left)-3; by=math.floor(top)-3
        br=math.ceil(endpoint)+3; bb=math.ceil(top+mh*scale)+3
        bw=br-bx; bh=bb-by
        inv=1/(scale*S)
        sampled=master.transform((bw*S,bh*S),Image.Transform.AFFINE,
            (inv,0,(bx-left)/scale,0,inv,(by-top)/scale),Image.Resampling.BICUBIC)
        before=im.crop((pos(bx),pos(by),pos(br),pos(bb)))
        before.resize((bw,bh),Image.Resampling.LANCZOS).convert('RGB').save(geomdir/f'{i+1:02d}_underlay.png')
        im.alpha_composite(sampled,(pos(bx),pos(by)))
        small=sampled.resize((bw,bh),Image.Resampling.LANCZOS)
        small.save(geomdir/f'{i+1:02d}_aircraft.png')
        alpha_final=np.asarray(small)[:,:,3]
        solid=alpha_final>=128
        yy,xx=np.where(solid)
        area=float(alpha_final.sum()/255.0)
        record={
          'category':category,'value':value,'unit':DATA['series'][0]['unit'],
          'value_label':'$'+num(value)+' Bn',
          'row_y_px':y,'origin_x_px':X0,'target_nose_x_px':endpoint,
          'uniform_master_scale':scale,'relative_linear_scale':math.sqrt(value/max(VALUES)),
          'raster_crop_xywh':[bx,by,bw,bh],
          'raster_body_area_alpha_px2':round(area,6),
          'raster_body_area_solid_px2':int(solid.sum()),
          'raster_rightmost_solid_boundary_px':int(bx+xx.max()+1),
          'raster_visible_bounds_half_alpha':[int(bx+xx.min()),int(by+yy.min()),int(bx+xx.max()+1),int(by+yy.max()+1)],
          'label_start_x_px':endpoint+30,
          'area_excludes':['wake','underlay','bounding rectangle'],
        }
        records.append(record)
        # Category identification blocks: generic synthetic IDs, no real logos.
        draw=ImageDraw.Draw(im)
        rect(draw,(110,y-37,385,y+37),'#162C40',outline='#3A5368',radius=8,width=1)
        rect(draw,(110,y-37,177,y+37),'#2C485F',radius=8)
        rect(draw,(168,y-36,177,y+36),'#2C485F')
        text(draw,(143.5,y),chr(65+i),33,PAPER,True,anchor='mm')
        text(draw,(197,y),category,29,WHITE,True,anchor='lm')
        # All amounts are external and to the right of the measurement nose.
        tx=endpoint+30
        text(draw,(tx,y-2),'$'+num(value),49,PAPER,True,anchor='lm')
        number_width=draw.textlength('$'+num(value),font=font(49,True))/S
        text(draw,(tx+number_width+12,y+1),'Bn',25,PAPER,anchor='lm')
        label_width=number_width+12+draw.textlength('Bn',font=font(25))/S
        record['label_bbox_px']=[round(tx,3),y-34,round(tx+label_width,3),y+34]
    # Compact reading guide is part of the graphic, not a decorative mini-chart.
    draw=ImageDraw.Draw(im)
    line(draw,[(110,1750),(1890,1750)],'#465869',1.3)
    text(draw,(110,1785),'01',22,PAPER,True)
    text(draw,(165,1781),'位置看机鼻',29,WHITE,True,True)
    text(draw,(165,1827),'共用零点，0–35 线性刻度。',24,MUTED,cjk=True)
    text(draw,(1035,1785),'02',22,PAPER,True)
    text(draw,(1090,1781),'面积看机体',29,WHITE,True,True)
    text(draw,(1090,1827),'同一机型，机体面积与市值成正比。',24,MUTED,cjk=True)
    text(draw,(110,1893),'小值机体可跨过零线；以机鼻读数。尾迹不计入面积。',22,MUTED,cjk=True)
    text(draw,(1890,1896),'SOURCE: LOCAL SYNTHETIC DATA',19,MUTED,anchor='rt')
    # One final export. Every complete execution is captured by the wrapper.
    final=im.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
    final.save(ROOT/'final.png',compress_level=7)
    # Check actual rendered mask geometry, and visibility in the exported image.
    arr=np.array(final).astype(np.int16)
    max_area=records[0]['raster_body_area_alpha_px2']
    for i,r in enumerate(records):
        bx,by,bw,bh=r['raster_crop_xywh']
        a=np.array(Image.open(geomdir/f'{i+1:02d}_aircraft.png'))[:,:,3]
        under=np.array(Image.open(geomdir/f'{i+1:02d}_underlay.png')).astype(np.int16)
        patch=arr[by:by+bh,bx:bx+bw]
        delta=np.max(np.abs(patch-under),axis=2)
        visible=(a>=128)&(delta>=18)
        ys,xs=np.where(visible)
        r['visible_body_pixels_with_contrast_ge_18']=int(visible.sum())
        r['solid_body_pixels_visibility_fraction']=round(float(visible.sum()/np.count_nonzero(a>=128)),6)
        r['visible_rightmost_boundary_in_final_px']=int(bx+xs.max()+1)
        r['nose_error_full_scale_percent']=round(abs(r['visible_rightmost_boundary_in_final_px']-r['target_nose_x_px'])/(X1-X0)*100,6)
        r['observed_area_ratio_to_A']=round(r['raster_body_area_alpha_px2']/max_area,8)
        r['expected_area_ratio_to_A']=value_ratio=VALUES[i]/max(VALUES)
        r['area_ratio_relative_error_percent']=round(abs(r['observed_area_ratio_to_A']/value_ratio-1)*100,5)
        r['label_gap_from_visible_nose_px']=round(r['label_start_x_px']-r['visible_rightmost_boundary_in_final_px'],3)
    report={
      'canvas_px':[W,H], 'supersampling':S,
      'data_file':'data.json',
      'mapping':{'origin_x_px':X0,'maximum_x_px':X1,'minimum':VMIN,'maximum':VMAX,'scale':'linear',
                 'nose_formula':'x = 490 + value / 35 * 1320',
                 'body_scale_formula':'(380 / 1274) * sqrt(value / 31.2)',
                 'minimum_size_clamp':False},
      'measurement':'Coverage-weighted body mask after the actual affine transform and final downsample; visibility measured against the rendered local underlay.',
      'rows':records
    }
    (out/'geometry_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'canvas':final.size,'rows':len(records),
      'max_nose_error_full_scale_percent':max(r['nose_error_full_scale_percent'] for r in records),
      'max_area_relative_error_percent':max(r['area_ratio_relative_error_percent'] for r in records)},ensure_ascii=False))

if __name__=='__main__':
    main()
