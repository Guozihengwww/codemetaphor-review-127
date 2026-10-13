"""Offline renderer: generated print material, area-calibrated radial partitions.

Run from this project with the frozen capture_render.py wrapper. No model/network
calls occur here. All numerical/category text and all partition geometry come
from the frozen local dataset. The reference is never copied into the image.
"""
from pathlib import Path
import json
import hashlib
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent
S = 3

def read_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))

def write_json(name, obj):
    p = ROOT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

inputs = read_json('input.json')
brief = read_json(inputs['brief'])
data = read_json(inputs['data'])
core = read_json(inputs['critical_metaphor_contract'])
assert (ROOT / inputs['reference']).is_file()
assert core['topic_zh'] == data['topic_zh']
W, H = brief['width'], brief['height']
assert (W, H) == (1500, 2007)
values = np.array(data['series'][0]['values'], dtype=float)
assert len(values) == len(data['categories']) == len(data['category_parent']) == 9
assert np.isclose(values.sum(), data['series_totals']['S1'])
total = float(values.sum())

# Related hues preserve the three synthetic parent/subcategory relationships.
palette_hex = ['#168BAC', '#328D76', '#83BBA2', '#BE4866', '#E58E99',
               '#D39025', '#F0B94D', '#C6C334', '#293F70']
palette = np.array([tuple(bytes.fromhex(c[1:])) for c in palette_hex], dtype=np.float32)
navy = '#18354F'
body = '#293E4A'
muted = '#62716F'
paper_rgb = np.array([249, 246, 237], dtype=np.float32)

# Fill the whole page with the generated unprinted paper, reflected at joins.
# Its ink material is used, separately, across 100% of the drop surface below.
paper = np.array(Image.open(ROOT/'assets/paper_strip.png').convert('RGB'))
paper = np.concatenate([paper, paper[::-1]], axis=0)
paper = np.concatenate([paper, paper[:, ::-1]], axis=1)
paper = np.tile(paper, (math.ceil(H/paper.shape[0]), math.ceil(W/paper.shape[1]), 1))[:H,:W]
paper = np.clip(paper.astype(float)*0.38 + paper_rgb*0.62, 0, 255).astype('uint8')
canvas = Image.fromarray(paper).resize((W*S, H*S), Image.Resampling.BICUBIC)
draw = ImageDraw.Draw(canvas)

fonts = {}
def font(size, bold=False):
    key = (size, bold)
    if key not in fonts:
        fonts[key] = ImageFont.truetype(str(ROOT/'fonts'/('NimbusSans-Bold.otf' if bold else 'NimbusSans-Regular.otf')), round(size*S))
    return fonts[key]

text_records = []
def text(txt, x, y, size, color=body, bold=False, align='left', tracking=0, role='label'):
    f = font(size, bold)
    if tracking:
        width = sum(draw.textlength(c, font=f) for c in txt) + (len(txt)-1)*tracking*S
    else:
        width = draw.textlength(txt, font=f)
    x0 = x*S - (width/2 if align == 'center' else width if align == 'right' else 0)
    y0 = y*S
    if tracking:
        cursor = x0
        for c in txt:
            draw.text((cursor, y0), c, font=f, fill=color, anchor='lt')
            cursor += draw.textlength(c, font=f) + tracking*S
    else:
        draw.text((x0, y0), txt, font=f, fill=color, anchor='lt')
    bb = draw.textbbox((x0,y0), txt, font=f, anchor='lt')
    record = {'text': txt, 'role': role, 'bbox': [round(x0/S,2),round(y,2),round((x0+width)/S,2),round(bb[3]/S,2)]}
    text_records.append(record)
    return record

def line(points, color=muted, width=1.6):
    draw.line([(round(x*S),round(y*S)) for x,y in points], fill=color, width=max(1,round(width*S)), joint='curve')

# Title hierarchy, deliberately distinct from the measurement annotations.
badge = (566, 78, 934, 126)
draw.rectangle(tuple(round(v*S) for v in badge), fill=navy)
text(data['visible_notice'], 750, 90, 25, '#FBF8EF', True, 'center', 2.2, role='synthetic_notice')
text('HOUSEHOLD', 750, 190, 63, navy, False, 'center', 7.0, role='title')
text('WATER USE', 750, 273, 137, navy, True, 'center', 1.2, role='title')
text('Composition  /  ' + data['series'][0]['unit'], 750, 430, 27, muted, False, 'center', role='metric_and_unit')
line([(95,490),(1405,490)], '#A9B2AD', 1.3)

# The generated silhouette is the quantitative data mask. Its aspect and overall
# position are non-data design choices; the partitions are solved after scaling.
DX, DY, DW, DH = 370, 560, 760, 1068
mask_img = Image.open(ROOT/'assets/drop_mask.png').convert('L').resize((DW*S,DH*S), Image.Resampling.LANCZOS)
mask = np.asarray(mask_img) >= 128
assert ndi.label(mask)[1] == 1
assert np.array_equal(ndi.binary_fill_holes(mask), mask)
yy, xx = np.indices(mask.shape, dtype=np.float32)
hub = np.array([DW*S*0.5, DH*S*0.64], dtype=np.float32)
theta = np.arctan2(xx + 0.5 - hub[0], -(yy + 0.5 - hub[1]))
# Place the first category around the tip. Solve all subsequent radial boundaries
# using the empirical area distribution of the scaled non-circular silhouette.
half = float(np.quantile(np.abs(theta[mask]), values[0]/total))
start = -half
phase = (theta-start) % (2*np.pi)
cumulative = np.cumsum(values)[:-1] / total
sorted_phase = np.sort(phase[mask])
cuts = []
for fraction in cumulative:
    k = int(round(len(sorted_phase)*fraction))
    cuts.append(float((sorted_phase[k-1]+sorted_phase[k])/2))
cuts = np.array(cuts)
region = np.searchsorted(cuts, phase, side='right').astype(np.uint8)
region[~mask] = 255
boundaries = np.r_[0.0, cuts, 2*np.pi]
counts = np.bincount(region[mask], minlength=9)
shares = counts/counts.sum()*100

# Recolour the image-generated ink while retaining its real scanned grain,
# mottling and paper fibres. Even the lightest fibres remain coloured, not holes.
ink = np.asarray(Image.open(ROOT/'assets/drop_ink.png').convert('RGB').resize((DW*S,DH*S), Image.Resampling.BICUBIC)).astype(np.float32)
red = ink[:,:,0]
low = float(np.quantile(red[mask], .03))
coverage = np.clip((245-red)/(245-low), 0, 1)
coverage = .60 + .40*coverage
colors = palette[np.minimum(region,8)]
surface = colors*coverage[:,:,None] + paper_rgb*(1-coverage[:,:,None])
surface = np.clip(surface,0,255).astype(np.uint8)
alpha = Image.fromarray((mask*255).astype(np.uint8))
canvas.paste(Image.fromarray(surface), (DX*S,DY*S), alpha)
draw = ImageDraw.Draw(canvas)

def edge_on_ray(angle):
    ux, uy = math.sin(angle), -math.cos(angle)
    rr = np.arange(0, 2*DH*S, .4)
    x = np.floor(hub[0]+ux*rr).astype(int)
    y = np.floor(hub[1]+uy*rr).astype(int)
    inside = (x>=0)&(x<DW*S)&(y>=0)&(y<DH*S)
    x,y=x[inside],y[inside]
    hit=np.flatnonzero(mask[y,x])
    at=hit[-1]
    return (DX+(x[at]+.5)/S, DY+(y[at]+.5)/S)

anchors=[]
for i in range(9):
    mid=(boundaries[i]+boundaries[i+1])/2+start
    anchors.append(edge_on_ray(mid))
# The first label attaches to the upper right shoulder within Toilet, avoiding
# a leader at the sharp tip. Its position has no quantitative role.
row=round(244*S)
right=np.flatnonzero(mask[row])[-1]
anchors[0]=(DX+(right+.5)/S, DY+(row+.5)/S)

label_specs = {
    0: {'x':1166,'y':741, 'side':'right','end_y':857,'elbow':1074},
    1: {'x':1166,'y':1004,'side':'right','end_y':1126,'elbow':1143},
    2: {'x':1166,'y':1221,'side':'right','end_y':1343,'elbow':1143},
    3: {'x':1166,'y':1460,'side':'right','end_y':1582,'elbow':1111},
    4: {'x':750, 'y':1698,'side':'bottom'},
    5: {'x':95,  'y':1518,'side':'left','end_y':1640,'elbow':458},
    6: {'x':95,  'y':1300,'side':'left','end_y':1422,'elbow':346},
    7: {'x':95,  'y':1100,'side':'left','end_y':1216,'elbow':346},
    8: {'x':95,  'y':891, 'side':'left','end_y':1007,'elbow':426},
}

leaders=[]
for i,(category,value) in enumerate(zip(data['categories'],data['series'][0]['values'])):
    spec=label_specs[i]
    x,y=spec['x'],spec['y']
    parent=data['category_parent'][i]
    sub=category.split(' / ',1)[1] if ' / ' in category else ''
    value_text=str(value)+'%'
    side=spec['side']
    align='center' if side=='bottom' else 'left'
    text(parent,x,y,31,body,True,align,role=f'category_{i}')
    if sub:
        text(sub,x,y+38,24,muted,False,align,role=f'subcategory_{i}')
        num_y=y+73
    else:
        num_y=y+52
    text(value_text,x,num_y,57 if i==0 else 50,navy,True,align,role=f'value_{i}')
    ax,ay=anchors[i]
    if side=='bottom':
        pts=[(ax,ay),(ax,1669),(750,1669),(750,1682)]
    else:
        end_x=1405 if side=='right' else 95
        pts=[(ax,ay),(spec['elbow'],spec['end_y']),(end_x,spec['end_y'])]
    line(pts,'#64716E',1.65)
    # A small coloured endpoint is entirely outside the data-bearing surface.
    if side!='bottom':
        ex=1405 if side=='right' else 95
        ey=spec['end_y']
        draw.ellipse(((ex-3)*S,(ey-3)*S,(ex+3)*S,(ey+3)*S),fill=palette_hex[i])
    leaders.append({'category':category,'points':[[round(a,2),round(b,2)] for a,b in pts]})

# A short reading note explicitly describes the area encoding and synthetic scope.
line([(95,1869),(1405,1869)],'#A9B2AD',1.3)
text('READING THE DROP',95,1900,21,navy,True,tracking=1.8,role='reading_note')
text('Coloured area = share of the total. Read clockwise from Toilet.',380,1899,23,body,role='encoding')
text('Subcategories are mutually exclusive parts of their named parent. Synthetic values; not a survey.',95,1946,22,muted,role='scope')

# These masks are diagnostics, not an alternative chart rendering.
(ROOT/'artifacts').mkdir(exist_ok=True)
Image.fromarray(region).save(ROOT/'artifacts/region_ids_supersampled.png')
Image.fromarray((mask*255).astype('uint8')).resize((DW,DH),Image.Resampling.LANCZOS).save(ROOT/'artifacts/drop_coverage.png')
output=canvas.resize((W,H),Image.Resampling.LANCZOS)
output.save(ROOT/'final.png', optimize=True)
geometry={
    'canvas':[W,H], 'supersampling':S, 'drop_bbox_xywh':[DX,DY,DW,DH],
    'hub_xy':[round(DX+float(hub[0])/S,3),round(DY+float(hub[1])/S,3)],
    'start_angle_degrees_clockwise_from_up':math.degrees(start),
    'boundaries_degrees_clockwise_from_up':[float(math.degrees(a+start)) for a in boundaries],
    'mask_components':int(ndi.label(mask)[1]),'mask_holes':int((ndi.binary_fill_holes(mask)&~mask).sum()),
    'data_mask_pixel_count':int(mask.sum()),'unassigned_mask_pixels':int(np.sum(mask&(region>8))),
    'overlapping_assignment_pixels':0,
    'regions':[{'category':c,'input_percent':float(v),'measured_supersampled_pixels':int(n),
                'measured_area_percent':float(s),'error_percentage_points':float(s-v),
                'ink_color':color} for c,v,n,s,color in zip(data['categories'],values,counts,shares,palette_hex)],
    'palette':palette_hex,'paper_rgb':paper_rgb.astype(int).tolist(),
    'texture_coverage_min_max':[float(coverage[mask].min()),float(coverage[mask].max())],
    'leaders':leaders,'text':text_records,
    'frozen_input_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['input.json','brief.json','data.json','core.json','reference.png']},
}
write_json('artifacts/geometry.json',geometry)
print(json.dumps({'canvas':[W,H], 'area_percentages':shares.tolist(), 'maximum_absolute_area_error_pp':float(np.max(np.abs(shares-values)))},ensure_ascii=False))
