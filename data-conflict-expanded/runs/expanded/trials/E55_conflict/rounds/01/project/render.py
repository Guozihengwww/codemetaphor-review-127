#!/usr/bin/env python3
"""Offline data-bound poster. Run with Python from the supplied environment."""
from pathlib import Path
from decimal import Decimal
import json
import math
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parent
S = 4

def load_json(name, decimals=False):
    with (ROOT / name).open(encoding='utf-8') as f:
        return json.load(f, parse_float=Decimal) if decimals else json.load(f)


def dump(name, value):
    p = ROOT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def font(name, size):
    return ImageFont.truetype(str(ROOT / 'fonts' / name), round(size * S))


FONTS = {
    'serif': 'NotoSerif-Regular.ttf',
    'italic': 'NotoSerif-Italic.ttf',
    'sans': 'NotoSans-Regular.ttf',
    'bold': 'NotoSans-Bold.ttf',
}


def main():
    inputs = load_json('input.json')
    brief = load_json(inputs['brief'])
    core = load_json(inputs['critical_metaphor_contract'])
    data = load_json(inputs['data'], decimals=True)
    # The original visual reference remains an offline dependency, never pasted.
    assert (ROOT / inputs['reference']).is_file()
    assert core['visible_title'] == data['title']
    assert data['synthetic'] is True
    assert len(data['series']) == 1
    series = data['series'][0]
    assert series['unit'] == 'years'
    assert data['value_axis']['scale'] == 'linear'
    vmin, vmax = [Decimal(str(data['value_axis'][k])) for k in ('minimum', 'maximum')]
    values = [Decimal(str(v)) for v in series['values']]
    categories = data['categories']
    assert len(categories) == len(values) == 7
    assert vmin == 0 and vmax == 25
    W, H = brief['width'], brief['height']
    assert (W, H) == (1620, 2004)
    (ROOT / 'artifacts').mkdir(exist_ok=True)

    # Exact mapping shared by all data objects. No lower bound is put on length.
    X0, X1 = Decimal('444'), Decimal('1476')
    px_per_year = (X1-X0) / (vmax-vmin)
    row_y = [577 + i * 168 for i in range(len(values))]
    axis_y = 1699
    ink, muted, warm = '#302c27', '#6c675e', '#9a4d35'
    paper = (247, 244, 234)

    # A deterministic paper surface. Texture does not alter any data geometry.
    rng = np.random.default_rng(5514)
    grain = rng.normal(0, 0.44, (H, W, 1))
    base = np.clip(np.array(paper)[None,None,:]+grain, 0, 255).astype('uint8')
    page = Image.fromarray(base).resize((W*S,H*S), Image.Resampling.BILINEAR).convert('RGBA')
    draw = ImageDraw.Draw(page)
    text_bounds = []

    def tx(x,y,s,size=24,face='sans',fill=ink,anchor='lt',track=None):
        f = font(FONTS[face], size)
        if track is None:
            draw.text((x*S,y*S),s,font=f,fill=fill,anchor=anchor,stroke_width=0)
            b = draw.textbbox((x*S,y*S),s,font=f,anchor=anchor)
            text_bounds.append({'text':s,'bbox_px':[round(q/S,3) for q in b]})
        else:
            assert anchor == 'lt'
            xstart=x
            for c in s:
                draw.text((x*S,y*S),c,font=f,fill=fill,anchor='lt')
                x += draw.textlength(c,font=f)/S + track
            text_bounds.append({'text':s,'bbox_px':[xstart,y,x,y+size*1.4]})

    def line(xa,ya,xb,yb,fill,width=1):
        draw.line((round(xa*S),round(ya*S),round(xb*S),round(yb*S)),fill=fill,width=max(1,round(width*S)))

    def rect(xa,ya,xb,yb,fill,outline=None,width=1):
        draw.rectangle((round(xa*S),round(ya*S),round(xb*S),round(yb*S)),fill=fill,outline=outline,width=round(width*S))

    # Editorial header: synthetic status is unmistakable and the topic is literal.
    rect(94,74,380,121,warm)
    tx(115,85,data['visible_notice'],23,'bold',fill='#fffaf0',track=0.8)
    tx(1526,95,'FICTIONAL COUNTRIES  /  YEARS',20,fill=muted,anchor='rm')
    visible_title = data['title'].removeprefix('SYNTHETIC ')
    assert visible_title == 'Years with a female head of state'
    tx(89,155,'Years with a female',83,'serif')
    tx(89,251,'head of state',91,'italic')
    tx(95,371,'Years in office across fictional countries',27,fill=muted)
    line(94,431,1526,431,'#a49b8e',1.15)
    tx(94,467,'COUNTRY / GROUP',20,'bold',fill=muted,track=1)
    tx(444,467,series['label'],23,fill=muted)
    tx(1476,479,'LINEAR SCALE',18,'bold',fill=muted,anchor='rm')

    # Square graph paper, with every ten minor squares equal to five years.
    step = float(px_per_year) / 2
    gx0, gx1, gy0, gy1 = 73,1547,511,1699
    for k in range(-30,70):
        x = float(X0)+k*step
        if gx0 <= x <= gx1:
            line(x,gy0,x,gy1,'#d9dcd3' if k%5 else '#c9cfc5',0.55 if k%5 else 0.75)
    for k in range(80):
        y = gy1-k*step
        if gy0 <= y <= gy1:
            line(gx0,y,gx1,y,'#d9dcd3' if k%5 else '#c9cfc5',0.55 if k%5 else 0.75)
    line(float(X0),517,float(X0),axis_y,'#8a8173',1.0)

    source = Image.open(ROOT/'assets/forearm_fist.png').convert('RGBA')
    src = np.asarray(source).astype(np.float32)
    sh, sw = src.shape[:2]
    split = float(load_json('assets/preparation.json')['wrist_split_x'])
    geometry = []

    def mapped_arm(length,height,tone):
        """Continuous x map with a fixed anatomical wrist split; all parts join.

        Source skin/ink supplies every object pixel. The shaft adapts separately
        from the hand. Outer source support maps precisely to [0, length].
        """
        pad = 3
        w = math.ceil(length)+2*pad
        h = math.ceil(height)+2*pad
        xx = (np.arange(w*S,dtype=np.float32)+.5)/S-pad
        yy = (np.arange(h*S,dtype=np.float32)+.5)/S-pad
        hand_w = min(height*(sw-split)/sh, length*.55)
        shaft_w = length-hand_w
        sx = np.where(xx < shaft_w, xx/shaft_w*split,
                      split+(xx-shaft_w)/hand_w*(sw-split))
        # A slimmer forearm while retaining the generated wrist and fist.
        # This is non-quantitative thickness; it does not change x geometry.
        t = np.clip(sx/split,0,1)
        thickness = np.where(sx<split, .73+.27*(t*t*(3-2*t)), 1.0)
        target_center = height*.58
        sy = ((yy[:,None]-target_center)/thickness[None,:]+target_center)/height*sh-.5
        sxgrid = np.broadcast_to(sx[None,:]-.5,sy.shape)
        coords = np.array([sy,sxgrid])
        out = np.stack([map_coordinates(src[:,:,c], coords, order=1, mode='constant',cval=0) for c in range(4)],axis=2)
        # Antialiased exact horizontal clipping, rather than integer length rounding.
        coverage = np.clip(np.minimum(xx+.5/S,length)-np.maximum(xx-.5/S,0),0,1/S)*S
        out[:,:,3] *= coverage[None,:]
        out[:,:,:3] *= np.array(tone)[None,None,:]
        return Image.fromarray(np.clip(out,0,255).astype(np.uint8),'RGBA'),pad

    tones = [(1,1,1),(.86,.81,.75),(.65,.62,.61),(1,.91,.76),(.73,.68,.64),(.77,.70,.66),(1,1,1)]
    label_bindings = []
    for i,(category,value,cy) in enumerate(zip(categories,values,row_y)):
        length_dec = (value-vmin)*px_per_year
        length = float(length_dec)
        end = float(X0+length_dec)
        value_text = format(value, 'f')
        if value == 0: value_text='0'
        # Category roles remain explicit, including grouped synthetic countries.
        tx(94,cy,category,28,'sans',anchor='lm')
        mark = {'category':category,'value':value_text,'target_length_px':length,
                'target_endpoint_x':end,'zero_x':float(X0),'center_y':cy,
                'zero_mark_omitted':value==0}
        if value > 0:
            height = min(148., length*.60)
            obj,pad=mapped_arm(length,height,tones[i])
            ox = float(X0)-pad
            oy = cy-height*.58-pad
            # Placement at subpixel coordinates on the same supersampled canvas.
            ix, iy = round(ox*S), round(oy*S)
            page.alpha_composite(obj,(ix,iy))
            # Measure the actual raster alpha after the same final downsample.
            # Per-object crops are diagnostics, not extra chart renders.
            object_at_origin = Image.new('L',(obj.width+2*S,obj.height+2*S))
            object_at_origin.paste(obj.getchannel('A'),(S,S))
            object_small = object_at_origin.resize((object_at_origin.width//S,object_at_origin.height//S),Image.Resampling.LANCZOS)
            object_small.save(ROOT/f'artifacts/mark_{i+1:02d}_alpha.png')
            alpha = np.asarray(object_small)
            col = alpha.max(axis=0)
            active32=np.flatnonzero(col>=32)
            active128=np.flatnonzero(col>=128)
            mask_origin_x=ix/S-1
            mark.update(height_px=height,diagnostic_mask=f'artifacts/mark_{i+1:02d}_alpha.png',
                mask_origin_x=mask_origin_x, mask_origin_y=iy/S-1,
                alpha32_left_x=float(mask_origin_x+active32[0]) if len(active32) else None,
                alpha32_right_x=float(mask_origin_x+active32[-1]+1) if len(active32) else None,
                alpha128_left_x=float(mask_origin_x+active128[0]) if len(active128) else None,
                alpha128_right_x=float(mask_origin_x+active128[-1]+1) if len(active128) else None,
                max_alpha=int(alpha.max()), nonzero_alpha_pixels=int(np.count_nonzero(alpha)),
                integrated_column_max_px=round(float(col.sum())/255.,6))
        else:
            mark.update(height_px=0.,max_alpha=0,nonzero_alpha_pixels=0)
        geometry.append(mark)
        # Text is never used to extend the quantitative object.
        number_x=max(end+27,float(X0)+49)
        tx(number_x,cy-10,value_text,38 if i==0 else 33,'bold',anchor='lm')
        unit_text='years each' if category.startswith('Countries ') else 'years'
        tx(number_x,cy+22,unit_text,20,fill=muted)
        label_bindings.append({'category':category,'display_value':value_text,'unit_text':unit_text,'number_x':number_x,'row_y':cy})

    # A sparse annotation occupies otherwise empty graph paper, outside all marks.
    # It explicitly explains the object-based quantitative encoding.
    rect(965,988,1478,1205,tuple(paper)+(236,))
    line(992,1018,1062,1018,warm,2.4)
    tx(992,1044,'One shared starting point.',27,'italic',fill=ink)
    tx(992,1093,'Read from 0 to the outer edge',23,fill=muted)
    tx(992,1128,'of each fist. Very small values',23,fill=muted)
    tx(992,1163,'stay small on the same scale.',23,fill=muted)

    # Linear years axis. No data symbols are added to zero rows.
    line(float(X0),axis_y,float(X1),axis_y,ink,1.6)
    ticks=[]
    for tick in range(int(vmin),int(vmax)+1,5):
        x=float(X0+Decimal(tick)*px_per_year)
        line(x,axis_y,x,axis_y+12,ink,1.5)
        tx(x,axis_y+29,str(tick),27,anchor='mt')
        ticks.append({'value':tick,'x':x})
    tx(float(X1),1771,'YEARS',20,'bold',fill=muted,anchor='rt',track=None)
    tx(float(X0),1775,'Length encodes years; thickness is illustrative.',22,fill=muted)
    line(94,1834,1526,1834,'#b6aea1',1)
    tx(94,1862,'Grouped entries: each named country has the same value; years are not added together.',23,fill=muted)
    tx(94,1904,'Synthetic values for fictional countries. Category order follows the supplied data.',23,fill=muted)

    final = page.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
    final.save(ROOT/'final.png',optimize=True)
    index_path=ROOT/'rounds/index.json'
    if index_path.is_file():
        index=load_json('rounds/index.json')
        if index and index[-1]['status']=='running':
            shutil.copy2(ROOT/'final.png',ROOT/f"artifacts/candidate_{index[-1]['number']:02d}.png")
    dump('artifacts/geometry.json',{
        'canvas':[W,H], 'scale_type':'linear', 'axis_minimum':str(vmin),'axis_maximum':str(vmax),
        'zero_x':float(X0),'maximum_x':float(X1),'pixels_per_year':str(px_per_year),
        'marks':geometry,'ticks':ticks,'label_bindings':label_bindings,
        'source_asset':'assets/forearm_fist.png','all_categories_in_input_order':categories,
        'text_bounds':text_bounds})
    print(json.dumps({'output':'final.png','canvas':[W,H],'positive_objects':sum(v>0 for v in values),'zero_objects':sum(v==0 for v in values)},ensure_ascii=False))


if __name__ == '__main__':
    main()
