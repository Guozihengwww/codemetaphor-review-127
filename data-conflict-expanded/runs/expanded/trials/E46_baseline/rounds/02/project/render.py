#!/usr/bin/env python3
"""Offline infographic renderer. Run through capture_render.py for a full candidate."""
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from scipy.ndimage import map_coordinates

ROOT=Path(__file__).resolve().parent

def read_json(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))

inputs=read_json('input.json')
brief=read_json(inputs['brief'])
data=read_json(inputs['data'])
core=read_json(inputs['critical_metaphor_contract'])
assert data['synthetic'] is True
assert data['value_axis']['scale']=='linear'
assert data['value_axis']['minimum']==0
assert len(data['series'])==1
series=data['series'][0]
categories=data['categories']
values=series['values']
assert len(categories)==len(values)==5
assert sum(values)==data['series_totals'][series['id']]
assert 'Annual renewable water resources' in core['visible_title']

W,H=brief['width'],brief['height']
S=2
BACKGROUND='#FFFFFF'
INK='#102D35'
MUTED='#61777D'
PALE='#E0EBED'
CYAN='#00AECB'
canvas=Image.new('RGBA',(W*S,H*S),BACKGROUND)
draw=ImageDraw.Draw(canvas)
font_root=ROOT/'assets/fonts'

def font(size,bold=False):
    return ImageFont.truetype(str(font_root/('NimbusSans-Bold.otf' if bold else 'NimbusSans-Regular.otf')),round(size*S))

def text(x,y,s,size=32,fill=INK,bold=False,anchor='lt'):
    draw.text((round(x*S),round(y*S)),s,font=font(size,bold),fill=fill,anchor=anchor)

def tracked(x,y,s,size=24,spacing=2.2,fill=MUTED,bold=False):
    f=font(size,bold)
    for c in s:
        draw.text((round(x*S),round(y*S)),c,font=f,fill=fill,anchor='lt')
        x+=draw.textlength(c,font=f)/S+spacing

def line(points,fill,width=1):
    draw.line([(round(x*S),round(y*S)) for x,y in points],fill=fill,width=max(1,round(width*S)))

def rect(box,fill):
    draw.rectangle(tuple(round(v*S) for v in box),fill=fill)

# Editorial title and unambiguous notice.
draw.rounded_rectangle((108*S,66*S,415*S,118*S),radius=26*S,fill='#DBF6FA')
text(132,80,data['visible_notice'],26,CYAN,True)
tracked(1298,83,'FIVE REGIONS',24,2.7,MUTED)
text(108,164,'Annual renewable',108,INK,True)
text(108,280,'water resources',108,INK,True)
text(112,405,'Annual volume · Gm³/year  /  billion cubic metres per year',30,MUTED)
line([(108,462),(1690,462)],PALE,1.5)

# A single linear mapping is used for all positions and asset transforms.
outlet_y=763.0
full_scale_px=1180.0
vmax=float(data['value_axis']['maximum'])
scale=full_scale_px/vmax
stream_centers=[390.0+310.0*i for i in range(len(categories))]
stream_width=32.0
faucet_width=246.0
faucet_src=Image.open(ROOT/'assets/faucet.png').convert('RGBA')
faucet_height=faucet_width*faucet_src.height/faucet_src.width
faucet_outlet_x=668.0/faucet_src.width*faucet_width
faucet_resized=faucet_src.resize((round(faucet_width*S),round(faucet_height*S)),Image.Resampling.LANCZOS)
water_src=Image.open(ROOT/'assets/water_texture.png').convert('RGBA')
splash_src=np.asarray(Image.open(ROOT/'assets/splash.png').convert('RGBA'),dtype=np.float32)
water_mask=Image.new('L',(W*S,H*S),0)
wm_draw=ImageDraw.Draw(water_mask)

# Light, labelled ruler. The thin guides do not carry an additional quantity.
axis_x=127.0
line([(axis_x,outlet_y),(axis_x,outlet_y+full_scale_px)],'#C9DADD',1.2)
for v in (0,3000,6000,9000):
    y=outlet_y+v*scale
    line([(axis_x-5,y),(axis_x+11,y)],'#A5BCC2',1.5)
    text(axis_x-16,y,f'{v:,}',25,MUTED,anchor='rm')
    # Sparse guide segments keep the object silhouettes dominant.
    for x in range(156,1700,16):
        line([(x,y),(min(x+4,1695),y)],'#E4EEF0',0.8)

# Fit the splash's generated central jet to the exact uniform stream width.
# The outer splash is compressed independently; neither width nor area encodes data.
def resized_splash(width,height):
    ow=max(1,round(width*S)); oh=max(1,round(height*S))
    h,w=splash_src.shape[:2]
    target_jet_width=stream_width*S
    target_x=np.array([0,(ow-target_jet_width)/2,(ow+target_jet_width)/2,ow-1])
    source_x=np.array([0,309,354,w-1])
    xs=np.interp(np.arange(ow),target_x,source_x)
    ys=np.linspace(0,h-1,oh)
    gy,gx=np.meshgrid(ys,xs,indexing='ij')
    out=np.stack([map_coordinates(splash_src[:,:,c],[gy,gx],order=1,mode='nearest') for c in range(4)],axis=2)
    # Blend the incoming jet into the column over its upper 30 px.
    fade_rows=min(oh,round(30*S))
    out[:fade_rows,:,3]*=np.linspace(0,1,fade_rows)[:,None]
    return Image.fromarray(np.clip(out,0,255).astype('uint8'),'RGBA')

geometry=[]
for i,(category,value,cx) in enumerate(zip(categories,values,stream_centers)):
    length=value*scale
    endpoint=outlet_y+length
    # Pixel edges, not object centres, define the data-bearing interval.
    start_px=round(outlet_y*S)
    end_px=round(endpoint*S)
    x_px=round((cx-stream_width/2)*S)
    col=water_src.resize((round(stream_width*S),end_px-start_px),Image.Resampling.LANCZOS)
    canvas.alpha_composite(col,(x_px,start_px))
    water_mask.paste(255,(x_px,start_px,x_px+col.width,end_px))
    splash_h=min(122.0,length*0.32)
    splash_w=min(190.0,190.0*min(1.0,length/150.0))
    splash=resized_splash(splash_w,splash_h)
    spx=round((cx-splash_w/2)*S)
    spy=end_px-splash.height
    canvas.alpha_composite(splash,(spx,spy))
    water_mask.paste(splash.getchannel('A'),(spx,spy),splash.getchannel('A'))
    # Equal-scale faucet overlays the jet at its lower lip.
    fx=round((cx-faucet_outlet_x)*S)
    fy=start_px-faucet_resized.height
    canvas.alpha_composite(faucet_resized,(fx,fy))
    label_x=cx-faucet_outlet_x+faucet_width/2
    text(label_x,515,category,37,INK,True,'mt')
    text(cx,endpoint+26,f'{value:,}',49,INK,True,'mt')
    text(cx,endpoint+83,series['unit'],25,MUTED,False,'mt')
    geometry.append({
        'category':category,'value':value,'unit':series['unit'],
        'outlet_center_x':cx,'outlet_lower_edge_y':outlet_y,
        'target_length_px':length,'target_splash_bottom_edge_y':endpoint,
        'raster_water_top_edge_y':start_px/S,'raster_water_bottom_edge_y':end_px/S,
        'faucet_box_xywh':[fx/S,fy/S,faucet_resized.width/S,faucet_resized.height/S],
        'splash_box_xywh':[spx/S,spy/S,splash.width/S,splash.height/S],
        'stream_width_px':stream_width,
        'value_label':f'{value:,}','category_label':category,
    })

# A data-derived total occupies the clear lower-right area.
rect((704,1482,1720,1850),BACKGROUND)
line([(725,1518),(725,1780)],CYAN,4)
tracked(764,1517,'COMBINED ANNUAL VOLUME',25,2.0,MUTED)
text(754,1582,f'{sum(values):,}',143,INK,True)
text(764,1737,'Gm³/year across the five regions',33,INK)
text(764,1800,'Illustrative regional values; no real-world ranking.',27,MUTED)

# Compact reading key and the exact unit definition.
line([(108,2020),(1690,2020)],PALE,1.5)
tracked(108,2054,'HOW TO READ',22,1.8,MUTED,True)
text(108,2097,'Longer outlet-to-splash distance = more annual water.',27,INK)
text(108,2136,'Faucet size, stream width and splash area are not quantities.',24,MUTED)
tracked(1192,2054,'UNIT',22,1.8,MUTED,True)
text(1192,2097,'1 Gm³/year = 1 billion m³/year',27,INK)
text(1192,2136,'Scale: 0–9,000 · linear',24,MUTED)

out=canvas.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
out.save(ROOT/'final.png')
water_mask.resize((W,H),Image.Resampling.LANCZOS).save(ROOT/'artifacts/water_geometry_mask.png')
(ROOT/'artifacts/geometry.json').write_text(json.dumps({
    'canvas':[W,H],'value_axis':data['value_axis'],'mapping_px_per_Gm3_year':scale,
    'supersampling':S,'origin_y':outlet_y,'full_scale_px':full_scale_px,
    'all_series_values_used':values,'total':sum(values),'marks':geometry,
    'generated_material_roles':['five faucets','all five water columns','all five endpoint splashes'],
    'reference_pixels_used':False
},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(f'Rendered {W}x{H}; {len(geometry)} data-bearing faucet streams.')

if __name__=='__main__':
    pass
