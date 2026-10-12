"""Regenerate the data-bound fork infographic from local assets and data.
Run from the case directory: PYTHON render.py
No model, network, or external image calls are made here.
"""
from pathlib import Path
import json, math, hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT/'data.json').read_text(encoding='utf-8'))
BRIEF = json.loads((ROOT/'brief.json').read_text(encoding='utf-8'))
W,H = BRIEF['width'],BRIEF['height']
S = 2
ANGLE = 38.0
C,SN = math.cos(math.radians(ANGLE)),math.sin(math.radians(ANGLE))
PX_PER_PERCENT = 22.0
BODY_W,BODY_H = 448,820
TINE_W,STEP = 30,66
ORIGINS = [(605,980),(1305,830)]
COLORS = [(12,68,133),(178,14,46)]
LOCAL_ORIGIN = (350,740)
LOCAL_SIZE = (750,1590)
FONT_DIR = Path('/usr/share/fonts/truetype/dejavu')
CJK_DIR = Path('/usr/share/fonts/opentype/noto')
FONTS = {
    'regular':FONT_DIR/'DejaVuSans.ttf',
    'bold':FONT_DIR/'DejaVuSans-Bold.ttf',
    'cjk':CJK_DIR/'NotoSansCJK-Regular.ttc',
    'cjk_bold':CJK_DIR/'NotoSansCJK-Bold.ttc',
}

def font(size,kind='regular'):
    return ImageFont.truetype(str(FONTS[kind]),round(size*S))

def txt(draw,xy,text,size,fill,kind='regular',anchor='lt'):
    draw.text((round(xy[0]*S),round(xy[1]*S)),text,font=font(size,kind),fill=fill,anchor=anchor)

def screen(u,v,origin):
    return origin[0]+C*u+SN*v, origin[1]-SN*u+C*v

def rotate_local(local,origin):
    ox,oy=LOCAL_ORIGIN
    tx,ty=origin
    inverse=(C,-SN,S*(ox-C*tx+SN*ty),SN,C,S*(oy-SN*tx-C*ty))
    return local.transform((W*S,H*S),Image.Transform.AFFINE,inverse,Image.Resampling.BICUBIC)

def local_paste(layer,im,u,v):
    ox,oy=LOCAL_ORIGIN
    layer.alpha_composite(im,(round((u+ox)*S),round((v+oy)*S)))

def text_strip(parts,size=30):
    widths=[font(sz,k).getlength(t) for t,sz,k,col in parts]
    strip=Image.new('RGBA',(math.ceil(sum(widths))+6*S,52*S))
    d=ImageDraw.Draw(strip)
    x=0
    for (t,sz,k,col),width in zip(parts,widths):
        d.text((round(x),38*S),t,font=font(sz,k),fill=col,anchor='ls')
        x+=width
    box=strip.getbbox()
    return strip.crop(box) if box else strip

def rotated_label(local,u,tip,category,value,color):
    strip=text_strip([(category+' ',30,'regular',(65,72,78,255)),(value+'%',32,'bold',color+(255,))])
    strip=strip.transpose(Image.Transpose.ROTATE_270)
    # The last value glyph ends 13 pixels before the physical tine tip/root.
    local_paste(local,strip,u-strip.width/(2*S),tip-13-strip.height/S)
    return {'category':category,'value_label':value+'%', 'label_gap_px':13,
            'local_tip':[u,tip], 'local_label_bounds':[u-strip.width/(2*S),tip-13-strip.height/S,u+strip.width/(2*S),tip-13]}

canvas=Image.new('RGBA',(W*S,H*S),(255,255,255,255))
d=ImageDraw.Draw(canvas)
BLACK=(22,29,36,255)
GRAY=(91,100,108,255)
LIGHT=(207,214,220,255)
# Quiet editorial header: the synthetic status and the quantitative rule are explicit.
txt(d,(92,63),DATA['visible_notice'],26,BLACK,'bold')
txt(d,(86,117),DATA['title'],106,BLACK,'cjk_bold')
txt(d,(95,265),'两组类别，沿叉齿读取百分比',30,GRAY,'cjk')
txt(d,(1900,98),'7 类别 × 2 组',29,BLACK,'cjk',anchor='rt')
txt(d,(1900,151),'共用 0–20% 线性尺度',26,GRAY,'cjk',anchor='rt')
for x,series,color in zip([96,500],DATA['series'],COLORS):
    d.rounded_rectangle((x*S,327*S,(x+13)*S,357*S),radius=3*S,fill=color+(255,))
    txt(d,(x+29,324),series['label'],26,color+(255,),'bold')

measurements=[]
for series_index,(series,origin,color,key) in enumerate(zip(DATA['series'],ORIGINS,COLORS,['blue','red'])):
    local=Image.new('RGBA',(LOCAL_SIZE[0]*S,LOCAL_SIZE[1]*S))
    ld=ImageDraw.Draw(local)
    ox,oy=LOCAL_ORIGIN
    material=Image.open(ROOT/'assets'/f'{key}_tine_material.png').convert('RGBA')
    # One continuous material coordinate system, independent of the value.
    material=material.resize((round(TINE_W*S),round(20*PX_PER_PERCENT*S)),Image.Resampling.LANCZOS)
    # The asset is a material sample. Its geometry is overwritten by a rectangular
    # data mask, with no minimum length or rounding of positive values to zero.
    material.putalpha(255)
    for i,(cat,val) in enumerate(zip(DATA['categories'],series['values'])):
        u=(i-(len(DATA['categories'])-1)/2)*STEP
        length=float(val)*PX_PER_PERCENT
        if length>0:
            ph=math.ceil(length*S)
            piece=material.crop((0,material.height-ph,material.width,material.height))
            # Fractional coverage makes the 8.0/8.2 distinction persist on export.
            frac=length*S-(ph-1)
            if frac<1:
                a=np.full((ph,material.width),255,dtype=np.uint8)
                a[0,:]=round(255*frac)
                piece.putalpha(Image.fromarray(a))
            local_paste(local,piece,u-TINE_W/2,-ph/S)
            # A restrained gloss at the square tip remains inside the data mask.
            px=(u-TINE_W/2+ox)*S; py=(oy-ph/S)*S
            ld.line((round(px+2*S),round(py+.6*S),round(px+(TINE_W-2)*S),round(py+.6*S)),fill=(220,234,255,110),width=1)
        meta=rotated_label(local,u,-length,cat,str(val),color)
        meta.update(series_id=series['id'],value=val,unit=series['unit'],local_x=u,
                    target_length_px=length,tip_screen=list(screen(u,-length,origin)),
                    root_screen=list(screen(u,0,origin)))
        measurements.append(meta)
    # A full generated lower body connects all seven data positions at v=0.
    body=Image.open(ROOT/'assets'/f'{key}_body.png').convert('RGBA').resize((BODY_W*S,BODY_H*S),Image.Resampling.LANCZOS)
    local_paste(local,body,-BODY_W/2,0)
    # Root reference line is inside the head, never above the quantitative zero.
    for u0 in np.arange(-214,215,12):
        ld.line(((ox+u0)*S,(oy+2.8)*S,(ox+min(u0+6,214))*S,(oy+2.8)*S),fill=(255,255,255,150),width=2)
    txt(ld,(ox+237,oy-4),'0% baseline',18,(112,122,132,255),'regular',anchor='lm')
    # Series engraving belongs to the actual handle, with the same rigid transform.
    engrave=text_strip([(series['label'],20,'bold',(255,255,255,232))]).transpose(Image.Transpose.ROTATE_270)
    local_paste(local,engrave,-engrave.width/(2*S),474)
    canvas.alpha_composite(rotate_local(local,origin))

# A single ruler uses the identical 22 px/% scale and identical rotation.
d=ImageDraw.Draw(canvas)
scale_origin=(420,1550)
txt(d,(94,1095),'共享齿长尺度',27,BLACK,'cjk')
txt(d,(94,1140),'0–20% · LINEAR',21,GRAY,'regular')
p0=screen(0,0,scale_origin);p1=screen(0,-20*PX_PER_PERCENT,scale_origin)
d.line((p0[0]*S,p0[1]*S,p1[0]*S,p1[1]*S),fill=(146,157,168,255),width=2*S)
for val in [0,5,10,15,20]:
    p=screen(0,-val*PX_PER_PERCENT,scale_origin)
    a=screen(-8,-val*PX_PER_PERCENT,scale_origin);b=screen(8,-val*PX_PER_PERCENT,scale_origin)
    d.line((a[0]*S,a[1]*S,b[0]*S,b[1]*S),fill=(116,129,142,255),width=2*S)
    txt(d,(p[0]-20,p[1]),str(val)+'%',25,GRAY,'regular',anchor='rm')
txt(d,(94,1624),'零值保留位置，不伸出叉齿。',23,GRAY,'cjk')
# Footer identifies the data and the category-order convention without adding a claim.
d.line((94*S,1711*S,1906*S,1711*S),fill=(223,228,232,255),width=S)
txt(d,(94,1740),'齿长 = 数值（%）  /  类别按 Group 01 → Group 07 排列',24,GRAY,'cjk')
txt(d,(1906,1789),'SOURCE: SUPPLIED SYNTHETIC DATA',19,GRAY,'regular',anchor='rs')

final=canvas.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
final.save(ROOT/'final.png',optimize=True)
# Evidence is measured from the rendered RGB image, along each actual tine axis.
pixels=np.asarray(final).astype(float)
def sample_rgb(x,y):
    # Bilinear sampling gives a consistent subpixel boundary estimate.
    x0,y0=math.floor(x),math.floor(y)
    if x0<0 or y0<0 or x0+1>=W or y0+1>=H:return np.array([255.,255.,255.])
    dx,dy=x-x0,y-y0
    return ((1-dx)*(1-dy)*pixels[y0,x0]+dx*(1-dy)*pixels[y0,x0+1]+
            (1-dx)*dy*pixels[y0+1,x0]+dx*dy*pixels[y0+1,x0+1])
def colored(rgb,key):
    r,g,b=rgb
    return (b-r>24 and b-g>14 and r<205) if key=='blue' else (r-g>28 and r-b>22 and g<210)
for item in measurements:
    si=0 if item['series_id']==DATA['series'][0]['id'] else 1
    key=['blue','red'][si];origin=ORIGINS[si];u=item['local_x']
    # Search only the continuous colored run attached to the root. The label gap
    # prevents numerical text from being counted as material.
    distances=np.arange(0.75,20*PX_PER_PERCENT+6,0.25)
    flags=[colored(sample_rgb(*screen(u,-float(v),origin)),key) for v in distances]
    if item['value']==0:
        visible=sum(flags[i] for i,v in enumerate(distances) if v<=4)
        measured=0.0 if visible==0 else max([float(v) for v,f in zip(distances,flags) if f and v<8],default=0.0)
    else:
        initial=next((i for i,f in enumerate(flags[:16]) if f),None)
        if initial is None:measured=None
        else:
            end=next((i for i in range(initial+1,len(flags)) if not flags[i] and not flags[min(i+1,len(flags)-1)]),len(flags)-1)
            measured=float(distances[end-1])+0.125
    item['visible_centerline_length_px']=measured
    item['error_full_scale_percent']=None if measured is None else round(abs(measured-item['target_length_px'])/(20*PX_PER_PERCENT)*100,4)
    item['in_frame_tip']=0<=item['tip_screen'][0]<W and 0<=item['tip_screen'][1]<H
record={
    'canvas_px':[W,H], 'angle_degrees':ANGLE,'px_per_percent':PX_PER_PERCENT,
    'value_axis':DATA['value_axis'],'body_size_px':[BODY_W,BODY_H],
    'rigid_transform':'x = tx + cos(theta)*u + sin(theta)*v; y = ty - sin(theta)*u + cos(theta)*v',
    'origins_px':ORIGINS,'tine_width_px':TINE_W,'tine_step_px':STEP,
    'measurement_method_zh':'从最终RGB图沿每个齿的中心轴双线性采样，测量与零线连通的系列色像素长度；白色间隔隔开标签。',
    'measurements':measurements,
    'final_sha256':hashlib.sha256((ROOT/'final.png').read_bytes()).hexdigest(),
}
(ROOT/'artifacts').mkdir(exist_ok=True)
(ROOT/'artifacts'/'geometry_check.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'canvas':[W,H],'items':len(measurements),'maximum_full_scale_error_percent':max(m['error_full_scale_percent'] or 0 for m in measurements)},ensure_ascii=False))
