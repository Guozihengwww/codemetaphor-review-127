#!/usr/bin/env python3
"""Rebuild the local train infographic without network or model calls.

All quantitative positions use x(value); generated raster trains are adapted
with one three-part horizontal transform, retaining the nose and rear proportions.
"""
from pathlib import Path
import json
import math
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
S = 2
PAPER = (247, 245, 239)
INK = (24, 44, 54)
MUTED = (93, 109, 113)
GRID = (217, 219, 211)
ACCENT = (176, 67, 47)
X0, X1 = 280.0, 1870.0
AXIS_Y, DIST_Y = 414.0, 1625.0
BODY_HEIGHT, FIRST_WHEEL_Y, ROW_PITCH = 82.0, 560.0, 145.0


@lru_cache(maxsize=64)
def font(size, bold=False):
    name = 'Bold' if bold else 'Regular'
    return ImageFont.truetype(str(ROOT / 'assets' / 'fonts' / f'ChartSans-{name}.otf'), round(size * S))


def main():
    data = json.loads((ROOT / 'data.json').read_text(encoding='utf-8'))
    brief = json.loads((ROOT / 'brief.json').read_text(encoding='utf-8'))
    parts = json.loads((ROOT / 'assets/train_parts.json').read_text(encoding='utf-8'))
    width, height = brief['width'], brief['height']
    categories = data['categories']
    values = data['series'][0]['values']
    vmin, vmax = data['value_axis']['minimum'], data['value_axis']['maximum']
    conversion = data['conversion']['km_per_mile']
    hours = data['conversion']['distance_comparison_hours']
    assert len(categories) == len(values) == len(parts) == 8
    assert vmin == 0 and vmax == 600
    assert data['value_axis']['scale'] == 'linear'
    assert [p['category'] for p in parts] == categories

    def x(value):
        return X0 + (value - vmin) / (vmax - vmin) * (X1 - X0)

    # A faint, deterministic paper grain, unrelated to the data encoding.
    rng = np.random.default_rng(18)
    grain = rng.normal(0, 0.40, (height * S, width * S, 1))
    rgb = np.clip(np.asarray(PAPER)[None, None, :] + grain, 0, 255).astype('uint8')
    canvas = Image.fromarray(rgb, 'RGB').convert('RGBA')
    draw = ImageDraw.Draw(canvas)
    text_inventory = []

    def line(coords, fill=INK, stroke=1):
        draw.line(tuple(round(c * S) for c in coords), fill=fill, width=max(1, round(stroke * S)))

    def rect(box, fill, radius=0, outline=None, stroke=1):
        box = tuple(round(c * S) for c in box)
        if radius:
            draw.rounded_rectangle(box, radius=round(radius*S), fill=fill, outline=outline, width=max(1,round(stroke*S)))
        else:
            draw.rectangle(box, fill=fill, outline=outline, width=max(1,round(stroke*S)))

    def circle(cx, cy, r, fill, outline=None, stroke=1):
        draw.ellipse(tuple(round(c*S) for c in (cx-r, cy-r, cx+r, cy+r)),fill=fill,outline=outline,width=max(1,round(stroke*S)))

    def text(tx, ty, label, size=28, bold=False, fill=INK, anchor='lt'):
        f = font(size,bold)
        pos = (round(tx*S), round(ty*S))
        draw.text(pos, label, font=f, fill=fill, anchor=anchor)
        bbox=draw.textbbox(pos,label,font=f,anchor=anchor)
        text_inventory.append({'text':label,'bbox':[round(c/S,2) for c in bbox], 'size_px':size})
        return draw.textlength(label,font=f)/S

    def measure(label,size=28,bold=False):
        return draw.textlength(label,font=font(size,bold))/S

    # Editorial heading: neutral fictional identities, with the synthetic notice prominent.
    rect((0,0,width,14),INK)
    rect((80,0,204,14),ACCENT)
    text(80,59,'速度可视化 / 八组模拟数据',24,fill=MUTED)
    rect((1530,47,1920,100),INK,radius=10)
    text(1725,74,data['visible_notice'],25,True,fill=PAPER,anchor='mm')
    text(77,126,data['title'],88,True)
    text(80,243,'从同一零点到鼻尖，比较八辆列车的合成峰值速度。',30)
    text(80,291,'长度表示速度 · 造型为示意 · 按输入顺序排列',25,fill=MUTED)

    # Exactly the same mapping is used for both top units and the bottom distance ruler.
    for tick in range(0,601,100):
        gx=x(tick)
        line((gx,AXIS_Y,gx,DIST_Y),GRID,1)
    line((X0,AXIS_Y,X0,DIST_Y),(157,169,167),1.5)
    line((X0,AXIS_Y,X1,AXIS_Y),INK,1.7)
    text(80,349,'km/h',28,True)
    text(80,388,'mph',23,fill=MUTED)
    for tick in range(0,601,20):
        gx=x(tick)
        line((gx,AXIS_Y,gx,AXIS_Y+(13 if tick%100==0 else 6)),INK,1.4 if tick%100==0 else 0.8)
        if tick%100==0:
            text(gx,350,str(tick),29,True,anchor='mt')
            text(gx,389,f'{tick/conversion:.1f}',23,fill=MUTED,anchor='mt')

    row_colors=[(37,70,105),(42,107,111),(159,49,52),(39,48,93),
                (168,97,52),(39,76,150),(37,104,81),(165,49,47)]
    geometry=[]
    # Save masks of physical trains (not shadows, captions, or grid lines) for checks.
    mask_canvas=Image.new('L',(width*S,height*S),0)

    for i,(category,value,part) in enumerate(zip(categories,values,parts)):
        wheel_y=FIRST_WHEEL_Y+i*ROW_PITCH
        roof_y=wheel_y-BODY_HEIGHT
        label_y=wheel_y-121
        end=x(value)
        target_length=end-X0
        source=Image.open(ROOT/part['path']).convert('RGBA')
        sw,sh=source.size
        scale_y=BODY_HEIGHT/(part['wheel_bottom_y']-part['roof_y'])
        out_h=round(sh*scale_y*S)
        out_w=round(target_length*S)
        # The cab nose and flat rear retain their aspect ratio; the carriage middle
        # changes length. Window count, wheel size, and height do not encode data.
        rear_src,head_src=90,320
        rear_w=round(rear_src*scale_y*S)
        head_w=round(head_src*scale_y*S)
        middle_w=out_w-rear_w-head_w
        if middle_w<=0:
            raise ValueError('Data length cannot contain the fixed train ends')
        train=Image.new('RGBA',(out_w,out_h),(0,0,0,0))
        slices=[(0,rear_src,rear_w),(rear_src,sw-head_src,middle_w),(sw-head_src,sw,head_w)]
        dx=0
        for left,right,pixels in slices:
            tile=source.crop((left,0,right,sh)).resize((pixels,out_h),Image.Resampling.LANCZOS)
            train.alpha_composite(tile,(dx,0))
            dx+=pixels
        paste_x=round(X0*S)
        paste_y=round((wheel_y-part['wheel_bottom_y']*scale_y)*S)
        canvas.alpha_composite(train,(paste_x,paste_y))
        mask_canvas.paste(train.getchannel('A'),(paste_x,paste_y))

        # Category and exact original value are attached to each respective train row.
        circle(89,roof_y+40,6,row_colors[i])
        text(110,roof_y+22,category,31,True)
        primary=f'{value} km/h'
        secondary=f'{value/conversion:.1f} mph'
        gap=23
        primary_w=measure(primary,36,True)
        secondary_w=measure(secondary,25)
        left_label=end-3-primary_w-gap-secondary_w
        # A paper knockout keeps labels clear of the vertical grid without changing
        # the physical train silhouette or its data-bearing extent.
        rect((left_label-10,label_y-6,end+8,label_y+42),PAPER,radius=3)
        text(left_label,label_y,primary,36,True)
        text(left_label+primary_w+gap,label_y+9,secondary,25,fill=MUTED)

        # Measure actual transformed alpha support, not just target coordinates.
        arr=np.asarray(train.getchannel('A'))
        yy,xx=np.where(arr>=128)
        if len(xx)==0:
            raise ValueError('Empty transformed physical train')
        actual_left=(paste_x+int(xx.min()))/S
        actual_right=(paste_x+int(xx.max())+1)/S
        rightmost=arr[:,int(xx.max())]
        tip_ys=np.where(rightmost>=128)[0]
        geometry.append({
            'category':category,'km_per_hour':value,'mph':f'{value/conversion:.1f}',
            'one_hour_km':value*hours,'one_hour_mi':f'{value*hours/conversion:.1f}',
            'target_zero_x':X0,'target_tip_x':end,'target_length_px':target_length,
            'visible_alpha_left_x':actual_left,'visible_alpha_tip_x':actual_right,
            'visible_tip_y_range':[(paste_y+int(tip_ys.min()))/S,(paste_y+int(tip_ys.max())+1)/S],
            'tip_error_px':actual_right-end,'left_error_px':actual_left-X0,
            'visible_length_px':actual_right-actual_left,
            'inferred_km_per_hour':(actual_right-actual_left)/(X1-X0)*600,
            'physical_body_height_px':BODY_HEIGHT,
            'carriage_middle_scale_x':middle_w/S/(sw-head_src-rear_src),
            'cab_scale_x':head_w/S/head_src,
            'scale_y':scale_y,
            'paste_bbox':[paste_x/S,paste_y/S,(paste_x+out_w)/S,(paste_y+out_h)/S]
        })

    line((X0,DIST_Y,X1,DIST_Y),INK,1.7)
    text(80,1601,'1 小时',29,True)
    text(80,1643,'行程 / km',22,fill=MUTED)
    for tick in range(0,601,20):
        gx=x(tick)
        line((gx,DIST_Y-(13 if tick%100==0 else 6),gx,DIST_Y),INK,1.4 if tick%100==0 else 0.8)
        if tick%100==0:
            text(gx,DIST_Y+18,f'{tick*hours:g}',28,True,anchor='mt')

    # One-hour distance comparison strip; its arrow ends at the same Train A tip.
    band=(237,239,232)
    rect((65,1707,1935,1859),band,radius=14)
    text(80,1729,'1h',40,True)
    text(80,1795,'示例',23,fill=MUTED)
    example=values[0]
    right=x(example)
    arrow_y=1799
    line((X0,arrow_y,right-1,arrow_y),INK,1.8)
    circle(X0,arrow_y,5,INK)
    draw.polygon([(round(right*S),round(arrow_y*S)),(round((right-13)*S),round((arrow_y-6)*S)),(round((right-13)*S),round((arrow_y+6)*S))],fill=INK)
    text(X0,1731,f'{categories[0]}',29,True)
    example_label=f'{example*hours:g} km / {example*hours/conversion:.1f} mi'
    text(right,1728,example_label,36,True,anchor='rt')
    text((X0+right)/2,1820,'保持合成峰值速度行驶 1 小时',23,fill=MUTED,anchor='mt')
    line((80,1890,1920,1890),(205,211,205),1)
    text(80,1912,'换算：mph = km/h ÷ 1.609344；行程（km）= 速度（km/h）× 1 h。',23,fill=MUTED)
    text(80,1955,'来源：本地合成数据 · 列车造型仅用于视觉编码',21,fill=MUTED)

    # Final export occurs only when this script is launched via capture_render.py.
    final=canvas.convert('RGB').resize((width,height),Image.Resampling.LANCZOS)
    final.save(ROOT/'final.png')
    artifact=ROOT/'artifacts'
    artifact.mkdir(exist_ok=True)
    mask_canvas.resize((width,height),Image.Resampling.LANCZOS).save(artifact/'physical_train_mask.png')
    (artifact/'geometry.json').write_text(json.dumps({
        'canvas':[width,height],'x_zero':X0,'x_max':X1,'scale_px_per_km_per_hour':(X1-X0)/600,
        'range':[vmin,vmax],'scale':'linear','units':{'km_per_mile':conversion,'hours':hours},
        'axis_tick_pairs':[{'km_per_hour':t,'mph':f'{t/conversion:.1f}','x':x(t),'one_hour_km':t*hours} for t in range(0,601,100)],
        'rows':geometry,'text_inventory':text_inventory,
        'asset_method':'Opaque train raster alpha cropped to physical silhouette; aspect-preserved cab and rear; resized carriage middle.'
    },ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Rendered {width}x{height}; eight trains; scale {(X1-X0)/600:g} px per km/h.')


if __name__=='__main__':
    main()
