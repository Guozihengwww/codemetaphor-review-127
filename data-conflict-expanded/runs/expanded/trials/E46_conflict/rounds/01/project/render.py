"""Offline deterministic infographic renderer. Run only through capture_render.py.

Generated object parts are composited into code-controlled data geometry.
All project paths are relative to this file. No model or network calls.
"""
from pathlib import Path
from decimal import Decimal
import json
import math
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
S = 4
BG = (252, 253, 253)
INK = (20, 29, 32)
MUTED = (102, 116, 121)
LIGHT = (212, 222, 225)
FAINT = (234, 240, 241)
CYAN = (17, 157, 183)


def read_json(name, decimal=False):
    text = (ROOT/name).read_text(encoding='utf-8')
    return json.loads(text, parse_float=Decimal) if decimal else json.loads(text)


@lru_cache(maxsize=None)
def font(size, bold=False):
    return ImageFont.truetype(str(ROOT/'assets'/'fonts'/('Sans-Bold.otf' if bold else 'Sans-Regular.otf')), round(size*S))


def main():
    inputs = read_json('input.json')
    brief = read_json(inputs['brief'])
    data = read_json(inputs['data'], decimal=True)
    core = read_json(inputs['critical_metaphor_contract'])
    assert (ROOT/inputs['reference']).is_file()
    assert core['visible_title'] == data['title']
    assert data['value_axis']['scale'] == 'linear'
    assert data['value_axis']['minimum'] == 0
    assert len(data['series']) == 1
    series = data['series'][0]
    categories = data['categories']
    values = series['values']
    assert len(categories) == len(values) == 5
    assert sum(values) == data['series_totals'][series['id']]
    W,H = brief['width'], brief['height']
    canvas = Image.new('RGB',(W*S,H*S),BG)
    draw = ImageDraw.Draw(canvas)
    text_boxes = []

    def txt(x,y,text,size=32,fill=INK,bold=False,anchor='lt',track=None):
        f=font(size,bold)
        if track is None:
            draw.text((round(x*S),round(y*S)),text,font=f,fill=fill,anchor=anchor)
            b=draw.textbbox((round(x*S),round(y*S)),text,font=f,anchor=anchor)
            text_boxes.append({'text':text,'bbox':[round(v/S,3) for v in b]})
        else:
            xp=x
            for ch in text:
                draw.text((round(xp*S),round(y*S)),ch,font=f,fill=fill,anchor='lt')
                xp+=draw.textlength(ch,font=f)/S+track
            text_boxes.append({'text':text,'bbox':[x,y,xp,y+size]})

    def line(points, fill=LIGHT, width=1):
        draw.line([(round(x*S),round(y*S)) for x,y in points],fill=fill,width=max(1,round(width*S)))

    def rect(box,fill):
        draw.rectangle(tuple(round(v*S) for v in box),fill=fill)

    def formatted(v):
        return format(v,',f')

    # Title and synthetic-data notice are code typeset, never raster text.
    rect((96,82,414,130),INK)
    txt(115,95,data['visible_notice'],25,(255,255,255),True,track=1.6)
    txt(1698,95,'WATER  /  ANNUAL VOLUME',23,MUTED,anchor='rt')
    txt(91,181,'Annual renewable',106,INK,True)
    txt(91,295,'water resources',106,INK,True)
    txt(98,430,'Five synthetic regions  ·  One common linear scale',30,MUTED)
    txt(1698,271,series['unit'],40,CYAN,True,anchor='rt')
    txt(1698,322,'billion cubic metres',27,MUTED,anchor='rt')
    txt(1698,358,'per year',27,MUTED,anchor='rt')
    line([(96,479),(1704,479)],LIGHT,1.2)

    ORIGIN = 770.0
    SPAN = 1240.0
    MAXIMUM = float(data['value_axis']['maximum'])
    SCALE = SPAN/MAXIMUM
    SPOUTS = np.linspace(348.0,1628.0,len(categories)).tolist()
    TAP_WIDTH = 238.0
    STREAM_WIDTH = 42.0
    tap = Image.open(ROOT/'assets/tap.png').convert('RGBA')
    splash = Image.open(ROOT/'assets/splash.png').convert('RGBA')
    texture = Image.open(ROOT/'assets/water_texture.png').convert('RGB')
    asset_meta = read_json('assets/component_geometry.json')
    tap_scale = TAP_WIDTH/tap.width
    tw = round(TAP_WIDTH*S)
    th = round(tap.height*tap_scale*S)
    tap_r = tap.resize((tw,th),Image.Resampling.LANCZOS)
    tap_cx = asset_meta['tap']['outlet_center_x']*tap_scale

    # A sparse, neutral ruler shares the real zero at the outlet lower edge.
    AXIS_X=145
    line([(AXIS_X,ORIGIN),(AXIS_X,ORIGIN+SPAN)],LIGHT,1.3)
    for val in range(0,9001,1000):
        yy=ORIGIN+val*SCALE
        major=(val%3000==0)
        line([(AXIS_X-8,yy),(AXIS_X+8,yy)],MUTED if major else LIGHT,1.2)
        if major:
            txt(122,yy-11,format(val,','),22,MUTED,anchor='rt')
            if val>0:
                # Short scale rules avoid turning the water metaphor into a grid.
                line([(AXIS_X+13,yy),(240,yy)],FAINT,1)
    txt(96,719,series['unit'],21,MUTED)
    # Zero guide stays pale and is occluded by the water at all five outlets.
    for xx in range(163,1703,15):
        line([(xx,ORIGIN),(min(xx+6,1703),ORIGIN)],LIGHT,.8)

    geometry=[]
    for idx,(category,v,spout_x) in enumerate(zip(categories,values,SPOUTS)):
        body_center=spout_x-tap_cx+TAP_WIDTH/2
        txt(body_center,512,category,36,INK,True,anchor='mt')
        txt(body_center,558,formatted(v),30,CYAN,False,anchor='mt')
        length=float(v)*SCALE
        endpoint=ORIGIN+length
        splash_h=min(82.0,length*.80)
        splash_w=min(150.0,STREAM_WIDTH+1.3*splash_h)
        # Local high-resolution layer: its last row uses fractional coverage.
        # This clipping ensures all splash and stream pixels remain above the
        # data endpoint. No minimum data length is introduced.
        local_w=math.ceil(max(splash_w,STREAM_WIDTH)*S)+16
        local_h=math.ceil(length*S)
        water=Image.new('RGBA',(local_w,local_h),(0,0,0,0))
        strip=texture.resize((round(STREAM_WIDTH*S),local_h),Image.Resampling.BICUBIC)
        sa=np.array(strip,dtype=float)
        # Controlled broad cyan tonality with visible generated water streaks.
        sa=.58*sa+.42*np.array([108,213,230],dtype=float)
        strip=Image.fromarray(np.uint8(np.clip(sa,0,255)),'RGB').convert('RGBA')
        water.alpha_composite(strip,((local_w-strip.width)//2,0))
        # Splash keeps generated droplets and highlights; its fit is decorative.
        sh=max(1,round(splash_h*S))
        sw=max(1,round(splash_w*S))
        spl=splash.resize((sw,sh),Image.Resampling.LANCZOS)
        sy=local_h-sh
        water.alpha_composite(spl,((local_w-sw)//2,sy))
        ar=np.array(water)
        coverage=np.clip(length*S-np.arange(local_h),0,1)
        ar[:,:,3]=np.rint(ar[:,:,3]*coverage[:,None]).astype(np.uint8)
        water=Image.fromarray(ar,'RGBA')
        wx=round(spout_x*S-local_w/2)
        wy=round(ORIGIN*S)
        canvas.paste(water,(wx,wy),water)
        # Identical tap geometry; the true lower edge aligns exactly to zero.
        tap_x=round((spout_x-tap_cx)*S)
        tap_y=round(ORIGIN*S)-th
        canvas.paste(tap_r,(tap_x,tap_y),tap_r)
        geometry.append({
            'category':category,'value':str(v),'formatted_label':formatted(v),
            'unit':series['unit'],'spout_x':spout_x,'outlet_y':ORIGIN,
            'data_length_px':length,'endpoint_y':endpoint,
            'stream_width_px':STREAM_WIDTH,
            'tap_bbox_px':[tap_x/S,tap_y/S,(tap_x+tw)/S,(tap_y+th)/S],
            'splash_bbox_px':[spout_x-splash_w/2,endpoint-splash_h,spout_x+splash_w/2,endpoint],
            'supersample_length_rows':local_h,
            'fractional_final_row_coverage':float(coverage[-1]),
        })

    # A directly computed comparison occupies the white space opened by the data.
    ratio=values[0]/values[-1]
    assert ratio == Decimal('500')
    txt(751,1195,format(ratio,'.0f')+'×',184,CYAN)
    txt(760,1407,categories[0]+' / '+categories[-1],43,INK,True)
    txt(760,1475,'The annual volume in '+categories[0],32,MUTED)
    txt(760,1519,'is '+format(ratio,'.0f')+' times that in '+categories[-1]+'.',32,MUTED)
    line([(760,1600),(1664,1600)],LIGHT,1.2)
    txt(760,1646,'HOW TO READ THE WATER',23,INK,True,track=1.8)
    txt(760,1698,'Length runs from the outlet to the splash base.',29,MUTED)
    txt(760,1740,'Every stream uses the same 0–9,000 linear scale.',29,MUTED)
    txt(760,1782,'Tap size, water width and splash area carry no value.',29,MUTED)

    # Quiet endpoint callout makes the longest stream easy to connect to its label.
    end_a=geometry[0]['endpoint_y']
    line([(439,end_a),(534,end_a)],LIGHT,1.2)
    txt(553,end_a-35,categories[0],24,MUTED)
    txt(552,end_a+1,formatted(values[0])+' '+series['unit'],32,INK,True)

    line([(96,2064),(1704,2064)],LIGHT,1.2)
    txt(96,2100,'Synthetic regional labels and values; not observed statistics.',25,MUTED)
    txt(1698,2100,'1 Gm³ = 1 billion m³',25,MUTED,anchor='rt')
    txt(96,2143,'Annual renewable water resources  ·  Values shown in full precision',22,MUTED)
    txt(1698,2143,'LINEAR SCALE  /  ZERO AT OUTLET',20,MUTED,anchor='rt')

    # Downsample once. This is the sole complete chart write.
    canvas=canvas.resize((W,H),Image.Resampling.LANCZOS)
    canvas.save(ROOT/'final.png',compress_level=7)
    artifact=ROOT/'artifacts';artifact.mkdir(exist_ok=True)
    meta={
      'canvas':[W,H],'supersampling':S,'linear_scale_px_per_Gm3_per_year':SCALE,
      'zero_y':ORIGIN,'full_scale_length_px':SPAN,'axis_maximum':MAXIMUM,
      'categories':geometry,'derived_ratio':str(ratio),'text_boxes':text_boxes,
      'dependencies':['input.json','brief.json','data.json','core.json','reference.png',
                      'assets/tap.png','assets/splash.png','assets/water_texture.png',
                      'assets/component_geometry.json','assets/fonts/Sans-Regular.otf',
                      'assets/fonts/Sans-Bold.otf'],
    }
    (artifact/'render_geometry.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Rendered final.png: '+str(W)+' x '+str(H)+'; '+str(len(values))+' categories.')


if __name__=='__main__':
    main()
