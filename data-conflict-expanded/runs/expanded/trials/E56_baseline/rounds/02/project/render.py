#!/usr/bin/env python3
"""Offline, data-driven harvest cylinders. Run only through capture_render.py.

Data height = 960 * value / 72000 screen pixels. The 17 degree view changes
ellipse aspect only; height is never multiplied by a projection coefficient.
The clear glass base is an additional 32 screen pixels, excluded from data.
"""
from pathlib import Path
import json
import math
import hashlib
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parent
inputs = json.loads((ROOT / 'input.json').read_text(encoding='utf-8'))
data = json.loads((ROOT / inputs['data']).read_text(encoding='utf-8'))
brief = json.loads((ROOT / inputs['brief']).read_text(encoding='utf-8'))
core = json.loads((ROOT / inputs['critical_metaphor_contract']).read_text(encoding='utf-8'))
assert (ROOT / inputs['reference']).is_file()
assert data['value_axis'] == {'minimum': 0, 'maximum': 72000, 'scale': 'linear'}
assert len(data['series']) == 1
series = data['series'][0]
values = series['values']
categories = data['categories']
assert len(values) == len(categories) == 5
W, H = brief['width'], brief['height']
SS = 2
WW, HH = W * SS, H * SS
ANGLE = 17.0
SIN = math.sin(math.radians(ANGLE))
BASELINE = 1320.0
FULL_HEIGHT = 960.0
BASE = 32.0
RX = 134.0
RY = RX * SIN
INNER_RX = 126.0
INNER_RY = INNER_RX * SIN
CENTERS = np.linspace(390, 1755, len(categories)).tolist()
NAMES = ['legumes', 'vegetables', 'potatoes', 'cereals', 'forage']
INK = '#173727'
GREEN = '#246A34'
MUTED = '#63716A'
GRID = '#DFE5DF'

@lru_cache(None)
def font(size, bold=False):
    return ImageFont.truetype(str(ROOT / 'assets/fonts' / ('bold.otf' if bold else 'regular.otf')), round(size * SS))

# Pale photographic studio backdrop, stronger towards the horizon.
yy, xx = np.mgrid[0:HH, 0:WW].astype(np.float32)
xn, yn = xx / WW, yy / HH
radial = np.clip(((xn - .50) / .80) ** 2 + ((yn - .54) / 1.05) ** 2, 0, 1)
floor = np.exp(-((yn - .827) / .087) ** 2)
bg = np.stack([248 - 10 * radial - 7 * floor,
               249 - 8 * radial - 5 * floor,
               243 - 4 * radial - 4 * floor], axis=-1)
canvas = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8), 'RGB').convert('RGBA')
del yy, xx, xn, yn, radial, floor, bg

def layer():
    return Image.new('RGBA', (WW, HH))

def composite(im):
    global canvas
    canvas = Image.alpha_composite(canvas, im)

def text(s, x, y, size, color=INK, bold=False, align='left', track=0):
    d = ImageDraw.Draw(canvas)
    f = font(size, bold)
    if track:
        widths = [d.textlength(c, font=f) for c in s]
        total = sum(widths) + (len(s) - 1) * track * SS
        start = x * SS - (total if align == 'right' else total / 2 if align == 'center' else 0)
        for c, wi in zip(s, widths):
            d.text((start, y * SS), c, fill=color, font=f, anchor='lt')
            start += wi + track * SS
    else:
        anchor = {'left': 'lt', 'right': 'rt', 'center': 'mt'}[align]
        d.text((x * SS, y * SS), s, fill=color, font=f, anchor=anchor)

def line(points, fill, width=1):
    ImageDraw.Draw(canvas).line([(round(x * SS), round(y * SS)) for x, y in points], fill=fill, width=max(1, round(width * SS)))

def ellipse_on(im, cx, cy, rx, ry, fill=None, outline=None, width=1):
    ImageDraw.Draw(im).ellipse(((cx-rx)*SS, (cy-ry)*SS, (cx+rx)*SS, (cy+ry)*SS), fill=fill, outline=outline, width=max(1,round(width*SS)))

def ellipse_arc(im, cx, cy, rx, ry, a0, a1, color, width):
    angles = np.linspace(math.radians(a0), math.radians(a1), 180)
    pts = [((cx+rx*math.cos(a))*SS, (cy+ry*math.sin(a))*SS) for a in angles]
    ImageDraw.Draw(im).line(pts, fill=color, width=max(1,round(width*SS)), joint='curve')

def paste_patch(rgb, alpha, x0, y0):
    rgba = np.concatenate([np.clip(rgb,0,255).astype(np.uint8), np.clip(alpha[...,None],0,255).astype(np.uint8)], axis=-1)
    im = Image.fromarray(rgba, 'RGBA')
    canvas.alpha_composite(im, (round(x0*SS), round(y0*SS)))

@lru_cache(None)
def material(name):
    path = 'assets/raw_02.png' if name == 'forage' else f'assets/{name}_material_v2.png'
    return np.asarray(Image.open(ROOT / path).convert('RGB'), dtype=np.float32)

def sample_material(tex, u, v, scale=1.0):
    # Continuous photographic coordinates, no repetition or mirror synthesis.
    # The coordinate extent is bounded before sampling; out-of-mask corners
    # may clamp to the border but are never visible in the object.
    coords = [v * scale, u * scale]
    return np.stack([map_coordinates(tex[:,:,ch], coords, order=1, mode='nearest') for ch in range(3)], axis=-1)

# Header.
pill = layer()
ImageDraw.Draw(pill).rounded_rectangle((110*SS, 69*SS, 414*SS, 115*SS), radius=23*SS, fill='#E3ECDD')
composite(pill)
text(data['visible_notice'], 262, 81, 23, GREEN, True, 'center', 1.5)
text(data['title'].removeprefix('SYNTHETIC '), 109, 151, 86, INK, True)
text(f"{series['label']}  /  {series['unit']}", 114, 256, 29, MUTED)
text('ANNUAL TOTAL', 1888, 87, 23, MUTED, True, 'right', 2.4)
total = sum(values)
assert total == data['series_totals'][series['id']]
text(f'{total:,}', 1890, 135, 109, GREEN, True, 'right')
text(series['unit'], 1887, 252, 29, MUTED, False, 'right')
line([(110, 318), (1890,318)], '#D3DDD2', 1.2)

# Unbroken common zero datum and shared linear scale.
for tick in (0, 18000, 36000, 54000, 72000):
    y = BASELINE - FULL_HEIGHT * tick / data['value_axis']['maximum']
    text(f'{tick:,}', 206, y-10, 22, '#869088', False, 'right')
    if tick:
        # Fine guide lines are deliberately quiet behind the material objects.
        line([(232,y),(1890,y)], GRID, .8)
    else:
        line([(232,y),(1890,y)], '#9CAA9C', 1.1)
    line([(221,y),(232,y)], '#A6B4A5', 1)

# Contact shadows below all five glass bases.
shadows = layer()
for cx in CENTERS:
    ellipse_on(shadows, cx+14, BASELINE+BASE+28, RX+20, 29, fill=(49,68,49,39))
shadows = shadows.filter(ImageFilter.GaussianBlur(12*SS))
composite(shadows)
contact = layer()
for cx in CENTERS:
    ellipse_on(contact, cx, BASELINE+BASE+32, RX-6, 13, fill=(44,65,42,30))
composite(contact.filter(ImageFilter.GaussianBlur(5*SS)))

geometry = []
for i, (category, val, cx, name) in enumerate(zip(categories, values, CENTERS, NAMES)):
    h = FULL_HEIGHT * val / data['value_axis']['maximum']
    top = BASELINE - h
    bottom = BASELINE + BASE
    # The clear base is rendered first. Its ellipse centers are exactly 32 px apart.
    # Transparent/glossy glass body plus thick lower glass edge.
    base_layer = layer()
    ellipse_on(base_layer, cx, bottom, RX, RY, fill=(181,195,190,155), outline=(121,144,136,165), width=1.4)
    # Front curved base surface.
    x0, y0 = math.floor(cx-RX-2), math.floor(BASELINE-RY-1)
    pw, ph = math.ceil(2*RX+4)*SS, math.ceil(BASE+2*RY+5)*SS
    gy, gx = np.mgrid[0:ph, 0:pw].astype(np.float32)
    x = x0 + (gx+.5)/SS
    y = y0 + (gy+.5)/SS
    u = (x-cx)/RX
    curve = RY*np.sqrt(np.clip(1-u*u,0,1))
    t = (y-(BASELINE+curve))/BASE
    mask = (np.abs(u)<=1)&(t>=0)&(t<=1)
    shade = 213 + 22*np.cos((t-.25)*2*math.pi) + 11*np.cos(u*math.pi/2)
    brgb = np.stack([shade-4, shade+2, shade],axis=-1)
    # A solid glass floor is mostly translucent neutral grey, never crop-colored.
    rgba = np.concatenate([np.clip(brgb,0,255).astype(np.uint8), np.where(mask,205,0)[...,None].astype(np.uint8)],axis=-1)
    patch = Image.fromarray(rgba,'RGBA')
    base_layer.alpha_composite(patch,(x0*SS,y0*SS))
    ellipse_arc(base_layer, cx, bottom-4, RX-2, RY-1, 0,180,(255,255,255,225),2.6)
    ellipse_arc(base_layer, cx, bottom, RX, RY, 0,180,(108,134,126,180),1.2)
    ellipse_arc(base_layer, cx, bottom-11, RX-3, RY-1, 5,175,(247,253,248,215),1.8)
    composite(base_layer)

    # Photograph mapped to the curved side of the data-carrying crop volume.
    x0, y0 = math.floor(cx-RX-3), math.floor(top-RY-4)
    pw = math.ceil(2*RX+6)*SS
    ph = math.ceil(h+2*RY+9)*SS
    gy, gx = np.mgrid[0:ph,0:pw].astype(np.float32)
    x = x0+(gx+.5)/SS
    y = y0+(gy+.5)/SS
    u = (x-cx)/INNER_RX
    front = INNER_RY*np.sqrt(np.clip(1-u*u,0,1))
    side_mask = (np.abs(u)<=1)&(y>=top+front)&(y<=BASELINE+front)
    tex = material(name)
    # Crop-scale factors control particle size only, never data height.
    scales = [1.5, 1.5, 1.5, 1.5, 2.0]
    scale = scales[i]
    tx = INNER_RX*np.arcsin(np.clip(u,-1,1))*scale + tex.shape[1]*.50
    ty = (y-top-front)*scale + 100
    assert tx[side_mask].min() >= 0 and tx[side_mask].max() < tex.shape[1]-1
    assert ty[side_mask].min() >= 0 and ty[side_mask].max() < tex.shape[0]-1
    rgb = sample_material(tex,tx,ty)
    diffuse = .83 + .13*np.sqrt(np.clip(1-u*u,0,1)) - .10*u
    rgb *= diffuse[...,None]
    paste_patch(rgb,np.where(side_mask,255,0),x0,y0)

    # Disk-shaped top material, projected at exactly 17 degrees.
    top_mask = u*u + ((y-top)/INNER_RY)**2 <= 1
    top_u = (x-cx)*scale + tex.shape[1]*.5
    top_v = (y-top)/SIN*scale + tex.shape[0]*.5
    assert top_u[top_mask].min() >= 0 and top_u[top_mask].max() < tex.shape[1]-1
    assert top_v[top_mask].min() >= 0 and top_v[top_mask].max() < tex.shape[0]-1
    rgb_top = sample_material(tex,top_u,top_v)
    rgb_top *= np.clip(1.04-.055*u, .94, 1.11)[...,None]
    paste_patch(rgb_top,np.where(top_mask,255,0),x0,y0)

    # Fine crop floor boundary: same 0 datum for every vessel.
    crop_edge = layer()
    ellipse_arc(crop_edge,cx,BASELINE,INNER_RX,INNER_RY,0,180,(66,80,59,145),1.1)
    composite(crop_edge)

    # Glass is a surface around the crop, not a separate decorative icon.
    # All wall/rim points share the data mapping above.
    glass = layer()
    gd = ImageDraw.Draw(glass)
    # Two refractive edges and narrow broad softbox reflections.
    for sign in (-1,1):
        ex = cx+sign*RX
        gd.line([(ex*SS,top*SS),(ex*SS,bottom*SS)],fill=(133,157,147,205),width=round(1.4*SS))
        gd.line([((ex-sign*2.3)*SS,top*SS),((ex-sign*2.3)*SS,bottom*SS)],fill=(255,255,255,245),width=round(2.2*SS))
    # Neutral translucent glass reflection, follows the front curvature.
    for center, half, opacity in [(-.78,.065,.36),(-.64,.018,.48),(.77,.083,.24),(.91,.021,.47)]:
        du = (u-center)/half
        strength = opacity*np.exp(-.5*du*du)
        outer_u = (x-cx)/RX
        arc = RY*np.sqrt(np.clip(1-outer_u*outer_u,0,1))
        refl_mask=(np.abs(outer_u)<.985)&(y>=top+arc+1)&(y<=BASELINE+front)
        alpha=np.where(refl_mask,strength*255,0)
        paste_patch(np.full_like(rgb_top,255),alpha,x0,y0)
    # The cup mouth stays a single plane: no particles float above it.
    ellipse_arc(glass,cx,top,RX,RY,180,360,(125,150,138,180),1.6)
    ellipse_arc(glass,cx,top,RX-2,RY-1,180,360,(255,255,255,245),1.7)
    ellipse_arc(glass,cx,top,RX,RY,0,180,(255,255,255,248),3.0)
    ellipse_arc(glass,cx,top,RX-1,RY-1,0,180,(131,159,145,185),1.0)
    ellipse_arc(glass,cx,top,INNER_RX+2,INNER_RY+.5,0,180,(246,254,239,210),1.3)
    composite(glass)

    # Numerical labels and semantic labels are bound only to frozen data.
    share = 100*val/total
    text(f'{share:.1f}%',cx,top-160,72,GREEN,True,'center')
    text(f'{val:,} {series["unit"]}',cx,top-82,29,INK,False,'center')
    text(category,cx,1434,32,INK,True,'center')
    # Small datum marks expose the data start and end at the side tangency.
    # They do not extend the data-carrying shape.
    for cy in (top,BASELINE):
        line([(cx-RX-8,cy),(cx-RX-1,cy)],'#8EAA94',1.15)

    # Component masks are cropped diagnostics, never another complete chart.
    diag = ROOT / 'artifacts/masks'
    diag.mkdir(parents=True,exist_ok=True)
    silhouette = Image.fromarray(np.where(top_mask|side_mask,255,0).astype(np.uint8),'L')
    silhouette.resize((round(pw/SS),round(ph/SS)),Image.Resampling.LANCZOS).save(diag/f'{name}_crop.png')
    geometry.append({'category':category,'value':val,'share_percent':share,
                     'cx':cx,'top_center_y':top,'crop_bottom_center_y':BASELINE,
                     'glass_bottom_center_y':bottom,'screen_height_px':h,
                     'inner_rx':INNER_RX,'inner_ry':INNER_RY,'glass_rx':RX,'glass_ry':RY,
                     'crop_roi':[x0,y0,pw/SS,ph/SS],
                     'mask_path':f'artifacts/masks/{name}_crop.png'})

# Readable explanation of the statistical encoding.
line([(110,1521),(1890,1521)],'#D3DDD2',1.2)
text('HOW TO READ',111,1555,20,GREEN,True,track=1.8)
text('Rim-to-crop-base center distance = harvest.  Percentages = share of the total.',329,1551,25,INK)
text('Shared linear scale: 0–72,000 t/year  •  17° view  •  32 px clear glass bases are excluded.',329,1590,22,MUTED)

canvas.convert('RGB').resize((W,H),Image.Resampling.LANCZOS).save(ROOT/'final.png')
geometry_report={'canvas':[W,H],'full_scale_screen_height_px':FULL_HEIGHT,
                 'view_angle_degrees':ANGLE,'nondata_base_px':BASE,
                 'data_sha256':hashlib.sha256((ROOT/inputs['data']).read_bytes()).hexdigest(),
                 'objects':geometry}
(ROOT/'artifacts/geometry.json').write_text(json.dumps(geometry_report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'canvas':[W,H], 'categories':categories, 'total':total,
                  'data_heights_px':[round(g['screen_height_px'],4) for g in geometry]}))
