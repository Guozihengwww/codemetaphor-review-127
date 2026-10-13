"""Offline train infographic. Run with the frozen capture_render.py wrapper.

All quantitative coordinates use Axis.x(value); train parts are resized to that
endpoint. The main range is never compressed or given a minimum data length.
"""
from pathlib import Path
import json
import math
from dataclasses import dataclass
from decimal import Decimal
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
S = 3
BG = '#F7F5EF'
INK = '#183542'
MUTED = '#596E74'
GRID = '#D8DFDB'
TEAL = '#0B6469'
RUST = '#AD452C'
PANEL = '#EDF1EB'
FONT = {'regular': ROOT/'assets/font_regular.otf', 'bold': ROOT/'assets/font_bold.otf'}

@dataclass(frozen=True)
class Axis:
    left: float
    width: float
    minimum: float
    maximum: float
    def x(self,value):
        return self.left+(float(value)-self.minimum)/(self.maximum-self.minimum)*self.width
    @property
    def scale(self):
        return self.width/(self.maximum-self.minimum)


def read_json(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))


def exact(v):
    return str(v)


def main():
    deps=read_json('input.json')
    brief=read_json(deps['brief'])
    data=read_json(deps['data'])
    contract=read_json(deps['critical_metaphor_contract'])
    assert (ROOT/deps['reference']).is_file()
    assert data['synthetic'] and data['value_axis']['scale']=='linear'
    assert len(data['categories'])==len(data['series'][0]['values'])==8
    assert contract['semantic_roles_zh']==['Synthetic peak speed (km/h)']
    width,height=brief['width'],brief['height']
    im=Image.new('RGBA',(width*S,height*S),BG)
    d=ImageDraw.Draw(im)
    fonts={}
    text_records=[]
    mark_records=[]
    markdir=ROOT/'artifacts/marks'
    markdir.mkdir(parents=True,exist_ok=True)

    def font(size,bold=False):
        key=(size,bold)
        if key not in fonts:
            fonts[key]=ImageFont.truetype(str(FONT['bold' if bold else 'regular']),round(size*S))
        return fonts[key]
    def text(x,y,value,size=26,fill=INK,bold=False,anchor='lt',tag=None):
        ft=font(size,bold)
        bbox=d.textbbox((round(x*S),round(y*S)),value,font=ft,anchor=anchor)
        d.text((round(x*S),round(y*S)),value,font=ft,fill=fill,anchor=anchor)
        text_records.append({'text':value,'tag':tag,'bbox':[round(v/S,3) for v in bbox]})
    def line(coords,fill=GRID,weight=1):
        d.line([round(c*S) for c in coords],fill=fill,width=max(1,round(weight*S)))
    def rect(coords,fill,outline=None,radius=0,weight=1):
        box=tuple(round(c*S) for c in coords)
        if radius:
            d.rounded_rectangle(box,round(radius*S),fill=fill,outline=outline,width=round(weight*S))
        else:
            d.rectangle(box,fill=fill,outline=outline,width=round(weight*S))

    # Header: only topic, metric and synthetic provenance, without real-world claims.
    rect((72,61,357,105),RUST,radius=7)
    text(91,72,data['visible_notice'],23,'#FFFFFF',True)
    text(1927,72,'EIGHT HYPOTHETICAL TRAINS',22,MUTED,anchor='rt')
    text(72,140,'Train speed & one-hour distance',91,INK,True,tag='title')
    text(76,249,'Synthetic peak speed, compared with the distance covered in one hour.',29,INK)
    text(76,294,'Each body starts at zero. The physical nose marks the value.',25,MUTED)
    line((72,342,1927,342),INK,1.2)

    axis=Axis(350,1560,data['value_axis']['minimum'],data['value_axis']['maximum'])
    zoom=Axis(940,936,0,20)
    km_per_mile=data['conversion']['km_per_mile']
    hours=data['conversion']['distance_comparison_hours']
    axis_y=443
    text(72,375,'PEAK SPEED',22,INK,True)
    text(308,374,'km/h',22,INK,True,anchor='rt')
    text(308,409,'mph',21,MUTED,anchor='rt')
    ticks=list(range(0,601,100))
    for v in ticks:
        x=axis.x(v)
        text(x,373,str(v),26,INK,True,anchor='mt',tag='main_kmh_tick')
        text(x,410,f'{v/km_per_mile:.1f}',22,MUTED,anchor='mt',tag='main_mph_tick')
        line((x,axis_y,x,1660),GRID,1)
        line((x,axis_y,x,axis_y+12),INK,1.4)
    line((axis.left,axis_y,axis.x(600),axis_y),INK,1.4)
    # Zero has the same geometric location for every row; faint grid does not
    # create a minimum-sized mark for the smallest values.
    for v in range(20,600,20):
        if v%100:
            x=axis.x(v)
            line((x,axis_y,x,axis_y+6),MUTED,.8)

    # Build a train from generated carriages plus a generated nose/cab.
    # At short lengths, reduce illustrative height to avoid a vertical sliver.
    # Height is not an additional quantitative scale.
    def train_sprite(idx,target_w,max_h):
        src=Image.open(ROOT/f'assets/train_{chr(97+idx)}.png').convert('RGBA')
        w,h=src.size
        joint=round(w*.617)
        split=round(w*.306)
        front=src.crop((joint,0,w,h))
        arr=np.array(src)
        roof_samples=[]
        for xx in range(150,min(390,w)):
            ys=np.flatnonzero(arr[:,xx,3]>128)
            if len(ys): roof_samples.append(int(ys[0]))
        roof=int(np.median(roof_samples))
        # Transfer the generated roof pantograph to the cab-only short version.
        panto=src.crop((round(w*.303),0,round(w*.416),roof+5))
        front.alpha_composite(panto,(round(front.width*.10),0))
        tile=src.crop((split,0,joint,h))
        use_h=min(max_h,target_w/3.8)
        H=max(1,round(use_h*S))
        W=max(1,round(target_w*S))
        natural_front=front.width/h*use_h
        if target_w < natural_front*1.22:
            result=front.resize((W,H),Image.Resampling.LANCZOS)
            parts=1
        else:
            fwidth=round(natural_front*S)
            rem=W-fwidth
            natural_tile=tile.width/h*use_h*S
            count=max(1,round(rem/natural_tile))
            result=Image.new('RGBA',(W,H),(0,0,0,0))
            edges=[round(i*rem/count) for i in range(count+1)]
            for i in range(count):
                ww=edges[i+1]-edges[i]
                result.alpha_composite(tile.resize((ww,H),Image.Resampling.LANCZOS),(edges[i],0))
            result.alpha_composite(front.resize((fwidth,H),Image.Resampling.LANCZOS),(rem,0))
            parts=count+1
        return result,use_h,parts

    def add_train(idx,value,axis,bottom,max_h,where):
        target_w=axis.x(value)-axis.x(0)
        sprite,used_h,parts=train_sprite(idx,target_w,max_h)
        x0=round(axis.x(0)*S)
        y0=round(bottom*S)-sprite.height
        im.alpha_composite(sprite,(x0,y0))
        # Retain a native-resolution alpha diagnostic for the same sprite with
        # its fractional placement, including filter support outside the edges.
        pad=4*S
        small=Image.new('RGBA',(sprite.width+2*pad,sprite.height+2*pad),(0,0,0,0))
        small.alpha_composite(sprite,(pad,pad))
        offx=x0//S-4
        offy=y0//S-4
        # Align diagnostic sampling to the final canvas pixel grid.
        diagnostic=Image.new('RGBA',((math.ceil((x0+sprite.width)/S)+4-offx)*S,(math.ceil((y0+sprite.height)/S)+4-offy)*S),(0,0,0,0))
        diagnostic.alpha_composite(sprite,(x0-offx*S,y0-offy*S))
        diagnostic=diagnostic.resize((diagnostic.width//S,diagnostic.height//S),Image.Resampling.LANCZOS)
        path=f'artifacts/marks/{where}_{chr(97+idx)}.png'
        diagnostic.save(ROOT/path)
        alpha=np.array(diagnostic.getchannel('A'))
        observed={}
        for threshold in (16,32,64,128):
            ys,xs=np.where(alpha>=threshold)
            observed[str(threshold)]={
                'left_px': int(xs.min())+offx if len(xs) else None,
                'right_exclusive_px':int(xs.max()+1)+offx if len(xs) else None,
                'top_px':int(ys.min())+offy if len(ys) else None,
                'bottom_exclusive_px':int(ys.max()+1)+offy if len(ys) else None,
                'occupied_pixels':len(xs)}
        mark_records.append({'category':data['categories'][idx],'view':where,
            'value_kmh':value,'mph_display':f'{value/km_per_mile:.1f}',
            'distance_km_in_one_hour':value*hours,
            'axis_range':[axis.minimum,axis.maximum],'pixels_per_kmh':axis.scale,
            'zero_x_px':axis.x(0),'target_nose_x_px':axis.x(value),
            'target_length_px':target_w,'raster_placement_x_px':x0/S,
            'raster_width_px':sprite.width/S,'illustrative_height_px':used_h,
            'assembled_sections':parts,'alpha_bounds_by_threshold':observed,
            'diagnostic':path,'diagnostic_origin':[offx,offy]})

    values=data['series'][0]['values']
    bottoms=[565+150*i for i in range(8)]
    for i,(category,value,bottom) in enumerate(zip(data['categories'],values,bottoms)):
        text(76,bottom-72,category,31,INK,True,tag='category')
        value_text=exact(value)+' km/h'
        text(76,bottom-30,value_text,27,INK,True,tag='kmh_value')
        text(77,bottom+7,f'{value/km_per_mile:.1f} mph',23,MUTED,tag='mph_value')
        add_train(i,value,axis,bottom,89,'main')

    # A distinctly bounded additional scale improves visibility without changing
    # any primary-chart length or substituting icons for the data-bearing trains.
    rect((842,1040,1927,1660),PANEL,outline='#BCCDC4',radius=18,weight=1.2)
    text(871,1064,'SMALL VALUES, MAGNIFIED',27,TEAL,True)
    text(871,1102,'0–20 km/h  ·  18× the main horizontal scale',22,MUTED)
    for v in range(0,21,5):
        x=zoom.x(v)
        text(x,1143,str(v),20,TEAL,True,anchor='mt')
        line((x,1175,x,1624),'#CBD8D0',1)
        line((x,1175,x,1183),TEAL,1)
    line((zoom.left,1175,zoom.x(20),1175),TEAL,1)
    for i,bottom in zip(range(4,8),[1260,1380,1500,1620]):
        text(863,bottom-34,data['categories'][i],20,TEAL,True)
        add_train(i,values[i],zoom,bottom,62,'detail')

    text(435,1478,'Small marks keep',23,MUTED)
    text(435,1507,'their true lengths.',23,MUTED)
    text(435,1551,'See the detail scale →',22,TEAL)

    # One-hour distance reference shares the SAME origin and pixels per unit.
    distance_y=1735
    text(72,1695,'IN ONE HOUR',22,INK,True)
    text(72,1735,'Distance at the',23,MUTED)
    text(72,1765,'stated speed',23,MUTED)
    line((axis.left,distance_y,axis.x(600),distance_y),INK,1.4)
    for v in ticks:
        x=axis.x(v)
        line((x,distance_y-12,x,distance_y),INK,1.4)
        text(x,distance_y+14,str(v*hours),25,INK,True,anchor='mt',tag='distance_tick')
    text(1910,1801,'km travelled',22,MUTED,anchor='rt')
    text(350,1801,'0–600 km, on the same horizontal scale',22,MUTED)

    line((72,1854,1927,1854),'#C9D1CA',1)
    text(76,1881,'SPEED × 1 HOUR = DISTANCE',24,INK,True)
    text(1927,1881,'mph = km/h ÷ 1.609344',24,INK,anchor='rt')
    text(76,1929,'All figures are synthetic. Distance assumes one hour at a constant peak speed.',23,MUTED)
    text(76,1960,'Only nose position encodes speed; vehicle height, colour and window count are illustrative.',21,MUTED)

    output=im.convert('RGB').resize((width,height),Image.Resampling.LANCZOS)
    output.save(ROOT/'final.png',compress_level=6)
    (ROOT/'artifacts/render_geometry.json').write_text(json.dumps({
        'canvas':[width,height],'supersampling':S,
        'main_axis':vars(axis),'detail_axis':vars(zoom),
        'detail_horizontal_magnification':zoom.scale/axis.scale,
        'marks':mark_records,'text':text_records,
        'data_file':'data.json','input_dependencies':deps,
    },ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Rendered 2000 × 2000 infographic using local generated train parts.')

if __name__=='__main__':
    main()
