"""Offline, data-bound rendering of the frozen antique-book composition.

Run through capture_render.py in this experiment. No network/model calls.
All paths needed by this renderer are relative to the project directory.
"""
from pathlib import Path
from decimal import Decimal
import hashlib
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent
SCALE = 3
INK = (39, 42, 36)
MUTED = (103, 104, 94)
PAPER = (242, 241, 232)
TEXT_BOXES = []

def load_json(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))

def save_json(path, value):
    p = ROOT / path
    p.parent.mkdir(exist_ok=True, parents=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def font(size, face='P052-Roman.otf'):
    return ImageFont.truetype(str(ROOT / 'assets/fonts' / face), round(size*SCALE))

def text(draw, content, x, y, size, face='P052-Roman.otf', color=INK, tracking=0, role='text'):
    f = font(size, face)
    bounds = f.getbbox(content)
    height = (bounds[3]-bounds[1])/SCALE
    width = (f.getlength(content)/SCALE + max(0, len(content)-1)*tracking)
    xx = (x-width/2)*SCALE
    yy = y*SCALE-bounds[1]
    if tracking:
        for c in content:
            draw.text((xx, yy), c, font=f, fill=color)
            xx += f.getlength(c)+tracking*SCALE
    else:
        draw.text((xx, yy), content, font=f, fill=color)
    TEXT_BOXES.append({'role':role,'text':content,'bbox':[round(x-width/2,3),y,round(x+width/2,3),round(y+height,3)]})
    return width, height

def vertical_title(canvas, title, center, cy, band_width, size, color):
    if band_width < 12:
        return
    f = font(size)
    box = f.getbbox(title)
    w, h = box[2]-box[0], box[3]-box[1]
    tile = Image.new('RGBA', (w+12*SCALE,h+12*SCALE), (0,0,0,0))
    td = ImageDraw.Draw(tile)
    px, py = 6*SCALE-box[0], 6*SCALE-box[1]
    # Very shallow, inset-looking gilded type, entirely within this data band.
    td.text((px,py+SCALE*.7), title, font=f, fill=(15,12,8,100))
    td.text((px,py), title, font=f, fill=(*color,245))
    tile = tile.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    assert tile.width/SCALE < band_width, (title, tile.width/SCALE, band_width)
    pos = (round(center*SCALE-tile.width/2),round(cy*SCALE-tile.height/2))
    canvas.paste(tile, pos, tile)

def draw_ornament(draw, cx, cy):
    c = (121,123,107)
    def line(points,width=1):
        draw.line([(round(x*SCALE),round(y*SCALE)) for x,y in points], fill=c, width=round(width*SCALE))
    line([(cx-133,cy),(cx-35,cy)])
    line([(cx+35,cy),(cx+133,cy)])
    for direction in [-1,1]:
        pts=[]
        for t in np.linspace(0,math.pi*2,81):
            pts.append((cx+direction*(13+11*math.cos(t)),cy+4*math.sin(t)))
        line(pts)
    line([(cx,cy-5),(cx+5,cy),(cx,cy+5),(cx-5,cy),(cx,cy-5)],1.1)

def compose_books(data, x, y, width, height):
    values = [Decimal(str(v)) for v in data['series'][0]['values']]
    categories = data['categories']
    assert sum(values) == Decimal('100')
    keys = ['hindu','folk','muslim','unaffiliated','christian','jewish','buddhist']
    assert len(categories) == len(keys)
    sw, sh = width*SCALE, height*SCALE
    row = np.zeros((sh,sw,3),dtype=np.float32)
    total_coverage = np.zeros(sw,dtype=np.float64)
    records=[]
    cumulative=Decimal('0')
    for idx,(name,key,value) in enumerate(zip(categories,keys,values)):
        left_d=Decimal(width)*cumulative/Decimal('100')
        right_d=Decimal(width)*(cumulative+value)/Decimal('100')
        left,right=float(left_d),float(right_d)
        a,b=left*SCALE,right*SCALE
        i,j=math.floor(a),math.ceil(b)
        coverage=np.maximum(0,np.minimum(np.arange(i,j)+1,b)-np.maximum(np.arange(i,j),a))
        source=Image.open(ROOT / f'assets/spine_{key}.png').convert('RGB')
        material=np.array(source.resize((j-i,sh),Image.Resampling.LANCZOS),dtype=np.float32)
        # Any supplementary binding-edge shade stays strictly inside this band.
        pos=(np.arange(i,j)+.5)/SCALE-left
        falloff=min(3.0,(right-left)*.018)
        edge_shade=1-.06*np.exp(-np.maximum(pos,0)/falloff)-.10*np.exp(-np.maximum(right-left-pos,0)/falloff)
        material*=edge_shade[None,:,None]
        row[:,i:j,:]+=material*coverage[None,:,None]
        total_coverage[i:j]+=coverage
        records.append({'index':idx,'category':name,'value':float(value),'value_label':format(value,'f')+'%',
                        'x_left':x+left,'x_right':x+right,'width_px':right-left,
                        'center_x':x+(left+right)/2,'top':y,'bottom':y+height,
                        'supersample_coverage_area_px':float(coverage.sum()/SCALE),
                        'material_path':f'assets/spine_{key}.png'})
        cumulative+=value
    assert np.max(np.abs(total_coverage-1))<1e-9
    return Image.fromarray(np.clip(np.rint(row),0,255).astype(np.uint8)),records

def main():
    inputs=load_json('input.json')
    brief=load_json(inputs['brief'])
    data=load_json(inputs['data'])
    core=load_json(inputs['critical_metaphor_contract'])
    # Reference stays available as a frozen design dependency; its pixels are never composited.
    assert (ROOT / inputs['reference']).is_file()
    assert core['visible_title'] == data['title']
    assert data['series'][0]['unit'] == 'percent of population'
    assert data['value_axis'] == {'minimum':0,'maximum':100,'scale':'linear'}
    W,H=brief['width'],brief['height']
    assert (W,H)==(2000,1392)
    canvas=Image.new('RGB',(W*SCALE,H*SCALE),PAPER)
    draw=ImageDraw.Draw(canvas)
    # A small typographic masthead keeps the synthetic status unambiguous.
    text(draw,data['visible_notice'],W/2,78,22,'NimbusSans-Bold.otf',(119,58,43),tracking=3,role='synthetic_notice')
    for x0,x1 in [(120,784),(1216,1880)]:
        draw.line((x0*SCALE,88*SCALE,x1*SCALE,88*SCALE),fill=(193,189,171),width=2)
    title=data['title'].removeprefix('SYNTHETIC ')
    text(draw,title,W/2,140,86,role='topic_title')
    total=format(Decimal(str(data['series_totals']['share'])).normalize(),'f')
    text(draw,f'Share of population (%)  ·  Total {total}%',W/2,246,27,'NimbusSans-Regular.otf',MUTED,role='unit_and_total')
    draw_ornament(draw,1000,315)

    bx,by,bw,bh=120,372,1760,668
    # Shared floor shadow, not an additional quantitative dimension.
    shadow=Image.new('RGBA',canvas.size,(0,0,0,0))
    sd=ImageDraw.Draw(shadow)
    sd.rounded_rectangle((117*SCALE,1024*SCALE,1883*SCALE,1054*SCALE),radius=14*SCALE,fill=(32,27,18,75))
    shadow=shadow.filter(ImageFilter.GaussianBlur(10*SCALE))
    canvas.paste(shadow,(0,0),shadow)
    books,geometry=compose_books(data,bx,by,bw,bh)
    canvas.paste(books,(bx*SCALE,by*SCALE))
    sizes=[49,22,51,45,55,0,28]
    gold=[(235,211,145),(48,41,25),(238,225,178),(240,221,163),(237,231,207),(235,213,150),(243,216,163)]
    for rec,size,color in zip(geometry,sizes,gold):
        vertical_title(canvas,rec['category'],rec['center_x'],by+bh/2,rec['width_px'],size,color)

    draw=ImageDraw.Draw(canvas)
    label_plan=[]
    for idx,rec in enumerate(geometry):
        center=rec['center_x']
        lower=idx in [1,3,5]
        label_x=1712.0 if idx==5 else (1850.0 if idx==6 else center)
        value_y=1225 if lower else 1119
        label_y=value_y+54
        end_y=value_y-16
        if idx==5:
            points=[(center,1049),(center,1087),(label_x,1180),(label_x,end_y)]
        elif idx==6:
            points=[(center,1049),(center,1082),(label_x,1090),(label_x,end_y)]
        else:
            points=[(center,1049),(label_x,end_y)]
        draw.line([(round(a*SCALE),round(b*SCALE)) for a,b in points],fill=(77,80,68),width=3)
        r=1.9
        draw.ellipse(((center-r)*SCALE,(1049-r)*SCALE,(center+r)*SCALE,(1049+r)*SCALE),fill=(66,70,59))
        text(draw,rec['value_label'],label_x,value_y,43,'P052-Bold.otf',role=f'value_{idx}')
        text(draw,rec['category'],label_x,label_y,27,role=f'category_{idx}')
        label_plan.append({'index':idx,'category':rec['category'],'value':rec['value_label'],'anchor_x':center,
                           'points':points,'value_xy':[label_x,value_y],'category_xy':[label_x,label_y]})
    text(draw,'Book width = population share. Heights are equal. Synthetic study values, not population estimates.',
         W/2,1350,20,'NimbusSans-Regular.otf',MUTED,role='encoding_and_provenance')
    final=canvas.resize((W,H),Image.Resampling.LANCZOS)
    final.save(ROOT / 'final.png')
    save_json('artifacts/render_geometry.json',{'canvas':[W,H],'scale':SCALE,'band':[bx,by,bw,bh],
              'mapping':'width = 1760 × percentage / 100; input order; exact fractional edge coverage; no minimum width',
              'categories':geometry,'leaders':label_plan})
    save_json('artifacts/text_boxes.json',TEXT_BOXES)
    print(json.dumps({'saved':'final.png','size':[W,H],'book_widths_px':[r['width_px'] for r in geometry]}))

if __name__ == '__main__':
    main()
