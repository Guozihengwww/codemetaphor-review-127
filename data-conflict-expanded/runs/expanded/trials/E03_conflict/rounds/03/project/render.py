"""Offline, data-bound wood-sector infographic. Run via capture_render.py.

All sector angles use values / 100 in one unprojected disk. The single
screen transform is (x, y) -> (x, 0.96*y). Bark is outside that disk.
"""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))

inputs = read('input.json')
brief, data, core = read(inputs['brief']), read(inputs['data']), read(inputs['critical_metaphor_contract'])
assert (ROOT/inputs['reference']).is_file()  # Retain design reference; never copy its pixels.
assert data['synthetic'] and data['series'][0]['unit'] == 'percent'
assert len(data['series']) == 1
values = np.asarray(data['series'][0]['values'], dtype=float)
cats = data['categories']
assert len(cats) == len(values) == len(data['category_parent'])
assert abs(values.sum()-100.) < 1e-8
W, H = brief['width'], brief['height']
SS = 2
CX, CY, R, YS = 842.5, 1050., 490., .96
START = -96.0
BG = (228, 237, 231)
INK = (27, 47, 40)
MUTED = (86, 106, 95)
LINE = (94, 114, 103)

palette_hex = ['#39705B','#609882','#8EBBA1','#B9D8B8',
               '#B87939','#D7A55A','#917342','#E6C584',
               '#437E96','#91BEC8','#716187','#AA9BBA',
               '#A6584E','#D58E72','#607848','#A4B66D',
               '#456F74','#C6923E','#607699','#BE6A60',
               '#7F976F','#B49C6A','#756C8A','#A0BBB1',
               '#8E6847','#D3AF7B']
assert len(palette_hex) == len(values)
palette = np.asarray([[int(h[i:i+2],16) for i in (1,3,5)] for h in palette_hex], dtype=np.float32)
edges = START + np.r_[0., np.cumsum(values)*3.6]
mid = (edges[:-1]+edges[1:])/2

FONTDIR = ROOT/'assets/fonts'
def font(kind, size):
    return ImageFont.truetype(str(FONTDIR/ {'serif':'display.otf','sans':'sans.otf','bold':'sans-bold.otf'}[kind]), int(round(size*SS)))
canvas = Image.new('RGB', (W*SS,H*SS), BG)
draw = ImageDraw.Draw(canvas)
text_boxes = []
def text(x,y,s,size=26,kind='sans',fill=INK,align='left',tag=None):
    f=font(kind,size)
    width=draw.textlength(s,font=f)/SS
    if align=='right': x-=width
    if align=='center': x-=width/2
    draw.text((round(x*SS),round(y*SS)),s,font=f,fill=fill,anchor='lt',stroke_width=0)
    bbox=draw.textbbox((round(x*SS),round(y*SS)),s,font=f,anchor='lt')
    text_boxes.append({'text':s,'tag':tag or 'editorial','bbox':[round(z/SS,2) for z in bbox]})
    return width
def line(points,fill=LINE,width=1):
    draw.line([(round(x*SS),round(y*SS)) for x,y in points],fill=fill,width=max(1,round(width*SS)),joint='curve')

# Editorial header: synthetic status, topic, metric and total.
draw.rounded_rectangle((74*SS,62*SS,332*SS,106*SS),radius=4*SS,fill=INK)
text(203,74,data['visible_notice'],21,'bold',(239,245,239),'center')
text(76,140,'Forest area shares',95,'serif')
text(79,266,'Synthetic forest-area share · percent (%)',29,'sans',MUTED)
text(80,311,'Mutually exclusive categories, in the supplied order',23,'sans',MUTED)
text(1608,146,'100%',68,'serif',INK,'right')
text(1607,226,'TOTAL FOREST AREA',20,'bold',MUTED,'right')
text(1607,272,f'{len(cats)} synthetic categories',23,'sans',MUTED,'right')
line([(76,368),(1609,368)],(169,186,173),1)
text(80,399,'READ CLOCKWISE',18,'bold',MUTED)
text(1608,399,'ALL VALUES IN %',18,'bold',MUTED,'right')

# Warp the generated material once. Face radius is exactly R; all data use
# the same ellipse. The original irregular bark extends outside R.
mat = np.asarray(Image.open(ROOT/'assets/tree_normalized_v2.png').convert('RGBA')).astype(np.float32)
mg = read('assets/material_geometry_v2.json')
mr = float(mg['wood_radius']); mx,my = mg['center_xy']
pad = 85
x0,y0 = math.floor(CX-R-pad), math.floor(CY-R*YS-pad)
x1,y1 = math.ceil(CX+R+pad), math.ceil(CY+R*YS+pad)
px = (np.arange(x0*SS,x1*SS,dtype=np.float32)+.5)/SS
py = (np.arange(y0*SS,y1*SS,dtype=np.float32)+.5)/SS
xx,yy = np.meshgrid(px,py)
dx,dy = xx-CX,(yy-CY)/YS
rad = np.hypot(dx,dy)
theta = np.mod(np.degrees(np.arctan2(dy,dx))-START,360.)
idx = np.searchsorted(np.cumsum(values)*3.6,theta,side='right')
idx = np.minimum(idx,len(values)-1)
sx,sy = mx+dx/R*mr, my+dy/R*mr
material = np.stack([map_coordinates(mat[:,:,k],[sy,sx],order=1,mode='constant',cval=0) for k in range(4)],axis=-1)
wood = rad <= R

# The stain remains translucent to fine photographic wood detail. It does
# not introduce any category-specific scale, offset, exploded slice or hole.
rgb = material[:,:,:3].copy()
lum = rgb @ np.asarray([.2126,.7152,.0722],dtype=np.float32)
gain = np.clip(lum/171.,.13,1.5)**.88
stained = palette[idx]*gain[:,:,None]*.91 + rgb*.09
stained[idx==len(values)-1] = rgb[idx==len(values)-1]*np.asarray([1.0,.995,.98])
material[:,:,:3] = np.where(wood[:,:,None],np.clip(stained,0,255),rgb)
material[:,:,3] = np.where(wood,255,material[:,:,3])
patch = Image.fromarray(np.clip(material,0,255).astype(np.uint8),'RGBA')

# Soft ground shadow, derived from the generated silhouette.
alpha = patch.getchannel('A')
shadow = Image.new('RGBA',canvas.size,(0,0,0,0))
shadow_mask=Image.new('L',canvas.size,0)
shadow_mask.paste(alpha,(x0*SS+7*SS,y0*SS+22*SS))
shadow_mask=shadow_mask.filter(ImageFilter.GaussianBlur(19*SS)).point(lambda v:round(v*.22))
shadow.paste((40,55,43,255),(0,0,W*SS,H*SS),shadow_mask)
canvas=Image.alpha_composite(canvas.convert('RGBA'),shadow)
canvas.alpha_composite(patch,(x0*SS,y0*SS))
canvas=canvas.convert('RGB'); draw=ImageDraw.Draw(canvas)

def pos(a,r=1.):
    t=math.radians(a)
    return (CX+R*r*math.cos(t),CY+R*YS*r*math.sin(t))

# Fine seams clarify even the 1% sectors, without a fake minimum width.
for a in edges[:-1]:
    line([(CX,CY),pos(a)],(227,224,194),.5)

# Callouts are kept in angular order on each side. Optimized only in label
# space; moving a label never changes its sector or share.
labels=[]
for side in ('left','right'):
    indices=[i for i,a in enumerate(mid) if ('right' if math.cos(math.radians(a))>=0 else 'left')==side]
    indices.sort(key=lambda i:pos(mid[i])[1])
    desired=np.array([pos(mid[i],1.16)[1] for i in indices])
    gap=np.array([73. if (' / ' in cats[indices[j]] or ' / ' in cats[indices[j+1]]) else 62. for j in range(len(indices)-1)])
    cons=[{'type':'ineq','fun':lambda y,j=j:y[j+1]-y[j]-gap[j]} for j in range(len(gap))]
    fit=minimize(lambda y:np.sum((y-desired)**2),np.linspace(500,1630,len(indices)),method='SLSQP',constraints=cons,bounds=[(474,1640)]*len(indices),options={'maxiter':400,'ftol':1e-7})
    assert fit.success or np.min(np.diff(fit.x)-gap)>-.01, str(fit.message)
    for i, ly in zip(indices,fit.x):
        start=pos(mid[i],.985)
        ex=303. if side=='left' else 1382.
        terminal=292. if side=='left' else 1393.
        points=[start,(ex,float(ly)),(terminal,float(ly))]
        # A quiet pale under-stroke preserves line visibility over rough bark.
        line(points,(236,240,229),2.2)
        line(points,LINE,1.05)
        px0,py0=start
        draw.ellipse(((px0-3)*SS,(py0-3)*SS,(px0+3)*SS,(py0+3)*SS),fill=(240,239,218),outline=INK,width=1*SS)
        label_left,label_right=(62.,279.) if side=='left' else (1406.,1623.)
        raw_value=data['series'][0]['values'][i]
        value=(str(raw_value).rstrip('0').rstrip('.') if isinstance(raw_value,float) and raw_value%1 else str(raw_value))+'%'
        if raw_value==int(raw_value): value=f'{int(raw_value)}%'
        if ' / ' in cats[i]:
            parent,sub=cats[i].split(' / ',1)
            text(label_left,ly-25,parent,26,'bold',INK,tag=f'category:{i}')
            text(label_left,ly+8,'/ '+sub,21,'sans',MUTED,tag=f'category:{i}')
            text(label_right,ly-25,value,25,'bold',INK,'right',tag=f'value:{i}')
        else:
            text(label_left,ly-12,cats[i],26,'bold',INK,tag=f'category:{i}')
            text(label_right,ly-12,value,25,'bold',INK,'right',tag=f'value:{i}')
        labels.append({'index':i,'category':cats[i],'value_text':value,'side':side,'label_center_y':float(ly),'anchor':list(start),'line_points':[list(p) for p in points]})

# A start cue references the actual first radial boundary.
start_point=pos(START,1.17)
cue=[pos(START,1.003),(start_point[0],start_point[1]-31)]
line(cue,(236,240,229),2.2)
line(cue,MUTED,1)
text(start_point[0],start_point[1]-58,'START',17,'bold',MUTED,'center')

line([(76,1732),(1609,1732)],(169,186,173),1)
text(78,1772,'One wood face. The full 100%.',39,'serif')
text(79,1834,'Sector area shows each share; bark is excluded from the total.',25,'sans',MUTED)
text(79,1877,'Subcategories are mutually exclusive parts of their named region.',23,'sans',MUTED)
text(79,1927,'Illustrative synthetic data — not a statistical survey.',22,'sans',MUTED)
text(1608,1930,'S1  /  PERCENT',18,'bold',MUTED,'right')

out=canvas.resize((W,H),Image.Resampling.LANCZOS)
out.save(ROOT/'final.png',optimize=True)

# Save literal raster evidence and layout metadata for offline validation.
mask=np.where(wood,idx+1,0).astype(np.uint8)
Image.fromarray(mask,'L').save(ROOT/'artifacts/sector_mask_2x.png')
Image.fromarray(np.clip(rgb,0,255).astype(np.uint8),'RGB').resize((x1-x0,y1-y0),Image.Resampling.LANCZOS).save(ROOT/'artifacts/projected_material.png')
counts=np.bincount(mask[wood].ravel(),minlength=len(values)+1)[1:]
geometry={
    'canvas':[W,H], 'center':[CX,CY], 'radius':R,'y_scale':YS,'start_angle_deg':START,
    'data_bbox':[CX-R,CY-R*YS,CX+R,CY+R*YS], 'material_patch_bbox':[x0,y0,x1,y1],
    'supersampling':SS,'palette':palette_hex,'seam_width_px':.5,
    'sector_mask_pixel_count':int(counts.sum()),
    'sectors':[{'index':i,'category':cats[i],'value':float(values[i]),'start_angle_deg':float(edges[i]),'end_angle_deg':float(edges[i+1]),'raster_area_percent':float(counts[i]/counts.sum()*100)} for i in range(len(values))],
    'labels':sorted(labels,key=lambda d:d['index']), 'text_boxes':text_boxes,
    'material_zh':'内置生图原始素材经径向校正后覆盖完整数据木面与树皮；按数据扇区着色保留年轮、裂缝和木材纹理。'
}
(ROOT/'artifacts/render_geometry.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'canvas':[W,H],'categories':len(cats),'total':float(values.sum()),'max_raster_share_error_pp':float(np.max(np.abs(counts/counts.sum()*100-values)))},ensure_ascii=False))
