#!/usr/bin/env python3
"""Offline, deterministic object-length infographic. Run from the project directory."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
S = 2
INK = '#303c35'
SECONDARY = '#596158'
TERRA = '#98533d'
PAPER = '#f7f4e9'

def read_json(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))

def save_json(relative, value):
    p = ROOT / relative
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

INPUT = read_json('input.json')
BRIEF = read_json(INPUT['brief'])
DATA = read_json(INPUT['data'])
CORE = read_json(INPUT['critical_metaphor_contract'])
W, H = BRIEF['width'], BRIEF['height']
assert (W, H) == (1620, 2004)
assert DATA['synthetic'] is True and DATA['value_axis']['scale'] == 'linear'
assert len(DATA['series']) == 1
SERIES = DATA['series'][0]
CATEGORIES, VALUES = DATA['categories'], SERIES['values']
assert len(CATEGORIES) == len(VALUES) == 7
assert CORE['visible_title'] == DATA['title']
assert (ROOT / INPUT['reference']).is_file()
VMIN, VMAX = DATA['value_axis']['minimum'], DATA['value_axis']['maximum']
assert VMIN == 0 and all(VMIN <= v <= VMAX for v in VALUES)

FONT_NAMES = {
    'sans': 'NimbusSans-Regular.otf',
    'bold': 'NimbusSans-Bold.otf',
    'serif': 'NimbusRoman-Regular.otf',
    'italic': 'NimbusRoman-Italic.otf',
    'mono': 'NimbusMonoPS-Regular.otf',
}
FONT_CACHE = {}

def font(kind, size):
    key = kind, size
    if key not in FONT_CACHE:
        FONT_CACHE[key] = ImageFont.truetype(str(ROOT/'assets/fonts'/FONT_NAMES[kind]), round(size*S))
    return FONT_CACHE[key]

# All position and quantity calculations use this one linear mapping.
X0, XMAX = 470.0, 1490.0
PX_PER_YEAR = (XMAX-X0)/(VMAX-VMIN)
GRID_TOP, AXIS_Y = 514.0, 1704.0
ROWS = [594.0 + 166.0*i for i in range(len(VALUES))]
ARM_HEIGHT = 132.0

def x_value(v):
    return X0 + (v-VMIN)*PX_PER_YEAR

# Neutral paper grain, synthesized in code rather than copied from the reference.
rng = np.random.default_rng(83)
base = np.zeros((H, W, 3), dtype=np.float32)
base[:] = (235, 234, 223)
yy, xx = np.ogrid[:H, :W]
page = (xx >= 38) & (xx < W-38) & (yy >= 28) & (yy < H-28)
noise = rng.normal(0, 0.50, (H,W,1))
base[page] = (247,244,233)
base += noise
image = Image.fromarray(np.clip(base,0,255).astype(np.uint8)).resize((W*S,H*S), Image.Resampling.BICUBIC).convert('RGBA')
draw = ImageDraw.Draw(image)
label_boxes = []

def line(coords, fill, width=1):
    draw.line(tuple(round(c*S) for c in coords), fill=fill, width=max(1,round(width*S)))

def text(pos, message, size, kind='sans', fill=INK, anchor='lt', track=None, record=None):
    x,y = pos
    f = font(kind,size)
    if track is None:
        draw.text((round(x*S),round(y*S)), message, font=f, fill=fill, anchor=anchor)
        bbox = draw.textbbox((round(x*S),round(y*S)), message, font=f, anchor=anchor)
    else:
        widths = [draw.textlength(c,font=f)/S for c in message]
        total = sum(widths)+track*max(0,len(message)-1)
        if anchor.startswith('m'): x-=total/2
        elif anchor.startswith('r'): x-=total
        start=x
        boxes=[]
        for c,cw in zip(message,widths):
            p=(round(x*S),round(y*S))
            draw.text(p,c,font=f,fill=fill,anchor='lt')
            boxes.append(draw.textbbox(p,c,font=f,anchor='lt'))
            x+=cw+track
        bbox=(round(start*S),min(b[1] for b in boxes),round((start+total)*S),max(b[3] for b in boxes))
    if record:
        label_boxes.append({'role':record,'text':message,'bbox':[round(v/S,2) for v in bbox]})

# Type-led editorial header.
line((108,83,1512,83),'#a5aa9b',1)
text((110,106),DATA['visible_notice'],25,'bold',TERRA,track=2.2,record='synthetic_notice')
text((1512,108),'FICTIONAL COUNTRIES',22,'sans',SECONDARY,anchor='rt',track=1.5)
text((106,172),'Years with a female',104,'serif',record='title_line_1')
text((110,275),'head of state',111,'italic',record='title_line_2')
text((111,407),SERIES['label']+'  /  '+SERIES['unit'],29,'sans',SECONDARY,record='series_and_unit')
line((108,457,1512,457),'#a5aa9b',1)
text((110,483),'FICTIONAL COUNTRY',21,'bold',SECONDARY,track=1.5)
text((X0,483),'LENGTH REPRESENTS YEARS',21,'bold',SECONDARY,track=1.5)

# Square graph paper, with half-year small squares and more visible 5-year guides.
step = PX_PER_YEAR/2
left, right = 94.0, 1516.0
for k in range(-30, 53):
    x = X0+k*step
    if left <= x <= right:
        if k % 10 == 0 and x >= X0:
            color, lw = '#bfc5b7', 1.0
        elif k % 2 == 0:
            color, lw = '#d8dccd', 0.7
        else:
            color, lw = '#e4e5d8', 0.5
        line((x,GRID_TOP,x,AXIS_Y),color,lw)
for k in range(60):
    y = AXIS_Y-k*step
    if y >= GRID_TOP:
        line((left,y,right,y),'#e1e3d5',0.6)
line((X0,GRID_TOP,X0,AXIS_Y), '#646d5e', 2.2)
line((X0,AXIS_Y,XMAX,AXIS_Y), '#646d5e', 2.2)
for v in range(int(VMIN),int(VMAX)+1):
    x=x_value(v)
    is_major=v%5==0
    line((x,AXIS_Y,x,AXIS_Y+(13 if is_major else 6)), '#646d5e', 1.8 if is_major else 1)
    if is_major:
        text((x,1733),str(v),30,'sans',anchor='mt',record='axis_tick')
text(((X0+XMAX)/2,1782),SERIES['unit'].upper(),23,'bold',SECONDARY,anchor='mt',track=3,record='axis_unit')

measurements=[]
for i,(category,value,y) in enumerate(zip(CATEGORIES,VALUES,ROWS)):
    endpoint=x_value(value)
    arm_record={'category':category,'value':value,'unit':SERIES['unit'],'target_length_px':round(endpoint-X0,5),'target_endpoint_x':round(endpoint,5),'row_center_y':y,'has_arm':value>0}
    if value>0:
        source=Image.open(ROOT/f'assets/arm_{i+1:02d}.png').convert('RGBA')
        sw,sh=source.size
        target_w=round((endpoint-X0)*S)
        target_h=round(ARM_HEIGHT*S)
        # A continuous piecewise affine warp: preserve wrist+fist aspect ratio,
        # and fit only the forearm to the remaining data-controlled length.
        split=round(sw*0.79)
        head_source_w=sw-split
        fixed_w=round(head_source_w*target_h/sh)
        if fixed_w>=target_w:
            # General small-value policy: shrink the WHOLE object; never clamp length.
            scale=target_w/sw
            target_h=max(1,round(sh*scale))
            sprite=source.resize((target_w,target_h),Image.Resampling.LANCZOS)
            mode='uniform_for_small_value'
        else:
            sprite=Image.new('RGBA',(target_w,target_h))
            left_part=source.crop((0,0,split,sh)).resize((target_w-fixed_w,target_h),Image.Resampling.LANCZOS)
            right_part=source.crop((split,0,sw,sh)).resize((fixed_w,target_h),Image.Resampling.LANCZOS)
            sprite.paste(left_part,(0,0))
            sprite.paste(right_part,(target_w-fixed_w,0))
            mode='forearm_affine_wrist_fist_rigid_scale'
        px=round(X0*S)
        py=round(y*S-target_h/2)
        image.alpha_composite(sprite,(px,py))
        # Measure the actual transformed raster alpha, not just the target width.
        a=np.asarray(sprite.getchannel('A'))
        ay,ax=np.where(a>=128)
        arm_record.update({
            'transform':mode,'source_path':f'assets/arm_{i+1:02d}.png',
            'split_source_x':split,'wrist_fist_width_px':fixed_w/S,
            'raster_opaque_bbox':[float((px+ax.min())/S),float((py+ay.min())/S),float((px+ax.max()+1)/S),float((py+ay.max()+1)/S)],
            'raster_alpha_end_x':float((px+ax.max()+1)/S),
            'raster_alpha_length_px':float((ax.max()-ax.min()+1)/S),
            'raster_covered_area_px':round(float((a>=128).sum()/S**2),2)
        })
    else:
        arm_record.update({'transform':'none_zero_value','raster_alpha_length_px':0,'raster_alpha_end_x':X0,'raster_covered_area_px':0})
    # Label every row and supplied value exactly, including the true zero.
    text((111,y),category,34,'sans',anchor='lm',record='category')
    text((endpoint+19,y),f'{value:.1f}',37,'bold',anchor='lm',record='value')
    measurements.append(arm_record)

# Clear reading instructions and group semantics, set outside the quantitative field.
line((108,1836,1512,1836),'#a5aa9b',1)
text((110,1860),'From zero to fist',27,'bold')
text((341,1860),' — the full arm length shows years in office.',27,'sans',SECONDARY)
text((110,1902),'Each country in a grouped row has the displayed value; years are not summed.',26,'sans',SECONDARY,record='group_semantics')
text((110,1939),'All countries and values are synthetic. A zero value is shown without an arm.',24,'sans',SECONDARY,record='synthetic_scope_zero')

final=image.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
final.save(ROOT/'final.png')

# Inspect the colored object geometry in the saved-resolution final raster.
arr=np.asarray(final).astype(np.int16)
for r in measurements:
    y=r['row_center_y']
    xlo=int(X0)-4
    ylo=round(y-ARM_HEIGHT/2)-5
    region=arr[ylo:round(y+ARM_HEIGHT/2)+6,xlo:1520]
    colored=((region.max(axis=2)-region.min(axis=2))>32)&(region.min(axis=2)<224)
    cy,cx=np.where(colored)
    if len(cx):
        left=xlo+int(cx.min())
        right=xlo+int(cx.max())+1
        r.update({'visible_color_bbox':[left,ylo+int(cy.min()),right,ylo+int(cy.max())+1],
                  'visible_color_length_px':right-left,
                  'visible_color_endpoint_x':right,
                  'visible_implied_years':round((right-X0)/PX_PER_YEAR,4),
                  'visible_endpoint_error_px':round(right-r['target_endpoint_x'],4)})
    else:
        r.update({'visible_color_bbox':None,'visible_color_length_px':0,'visible_color_endpoint_x':X0,'visible_implied_years':0,'visible_endpoint_error_px':0})
save_json('artifacts/geometry.json',{
    'canvas':[W,H],'linear_axis':{'minimum':VMIN,'maximum':VMAX,'zero_x':X0,'maximum_x':XMAX,'pixels_per_year':PX_PER_YEAR},
    'measurement_method_zh':'先量变换后alpha≥128的实体边界，再从最终PNG逐行提取有色像素（通道差>32且最小通道<224）核对可见端点；色彩法可能略排除深色外轮廓。',
    'rows':measurements,'text_bounds':label_boxes,
})
print('Rendered final.png:',W,H)
print('Visible lengths:',[r['visible_color_length_px'] for r in measurements])
