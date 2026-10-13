"""Offline, data-bound portrait infographic. Run with the supplied Python interpreter."""
from pathlib import Path
import json, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import map_coordinates

ROOT=Path(__file__).resolve().parent
A=ROOT/'assets'
# Keep all frozen input dependencies local and available to the renderer.
inputs=json.loads((ROOT/'input.json').read_text(encoding='utf-8'))
brief=json.loads((ROOT/inputs['brief']).read_text(encoding='utf-8'))
data=json.loads((ROOT/inputs['data']).read_text(encoding='utf-8'))
contract=json.loads((ROOT/inputs['critical_metaphor_contract']).read_text(encoding='utf-8'))
with Image.open(ROOT/inputs['reference']) as reference:
    reference_size=reference.size  # Reference pixels are never composited.
W,H=brief['width'],brief['height']
S=2
BG='#F5F0E7';INK='#292923';MUTED='#746B60';ACCENT='#AD482F'
CX,CY,R=880.,1030.,500.
values=np.asarray(data['series'][0]['values'],dtype=float)
categories=data['categories']
assert len(values)==len(categories)==8
assert np.isclose(values.sum(),100.) and np.all(values>0)
angles=np.concatenate(([0.],np.cumsum(values)*3.6))
colors=['#E9BBA5','#F0CEB2','#BA7961','#A85743','#D59E78','#965B46','#DFC3A2','#E9D4B9']
canvas=Image.new('RGB',(W*S,H*S),BG)
draw=ImageDraw.Draw(canvas)
font_paths={'regular':A/'fonts/Text-Regular.otf','bold':A/'fonts/Display-Bold.otf','mono':A/'fonts/Numbers-Mono.ttf'}
font_cache={}
text_records=[]

def font(size,kind='regular'):
    key=(size,kind)
    if key not in font_cache:font_cache[key]=ImageFont.truetype(str(font_paths[kind]),round(size*S))
    return font_cache[key]

def text(txt,x,y,size=30,kind='regular',fill=INK,align='left'):
    f=font(size,kind)
    width=draw.textlength(txt,font=f)/S
    if align=='center':x-=width/2
    elif align=='right':x-=width
    # y denotes actual visible text top, for stable alignment across fonts.
    bb=draw.textbbox((0,0),txt,font=f)
    draw.text((round(x*S),round(y*S)-bb[1]),txt,font=f,fill=fill)
    text_records.append({'text':txt,'bbox':[round(x,2),round(y,2),round(x+width,2),round(y+(bb[3]-bb[1])/S,2)]})
    return width

def line(points,fill=INK,width=2):
    draw.line([(round(x*S),round(y*S)) for x,y in points],fill=fill,width=max(1,round(width*S)),joint='curve')

def circle(cx,cy,r,fill=None,outline=None,width=1):
    draw.ellipse(((cx-r)*S,(cy-r)*S,(cx+r)*S,(cy+r)*S),fill=fill,outline=outline,width=round(width*S))

def paste_asset(name,x,y,w,h):
    im=Image.open(A/name).convert('RGBA').resize((round(w*S),round(h*S)),Image.Resampling.LANCZOS)
    if name=='eye_right.png':
        ar=np.array(im)
        ar[:,-round(w*S*0.045):,3]=0  # exclude the adjacent hair fragment from the eye cutout
        im=Image.fromarray(ar)
    canvas.paste(im,(round(x*S),round(y*S)),im)

# Quiet editorial frame and hierarchy.
line([(54,54),(1746,54),(1746,1986),(54,1986),(54,54)],'#B8AD9E',1.2)
draw.rounded_rectangle((99*S,94*S,388*S,137*S),radius=3*S,fill=ACCENT)
text(data['visible_notice'],116,104,25,'bold',BG)
text('COMPOSITION  /  %',1696,107,22,'mono',MUTED,'right')
text('Portrait of',97,170,104,'bold')
text('unemployment',95,276,104,'bold')
text('Share of the unemployed by age, gender and population group',101,389,29,'regular',MUTED)
line([(101,441),(1698,441)],'#C9BFB1',1.2)

# The ear is illustrative and stays outside the measured disk.
paste_asset('ear_left.png',CX-R-104,CY-83,128,262)

# Generated strand textures are mapped to two non-data hair bands.
# Their masks never change the exact circular denominator.
x0,y0=int(CX-R-155),int(CY-R-153)
x1,y1=int(CX+R+153),int(CY+R+110)
w,h=(x1-x0)*S,(y1-y0)*S
yy,xx=np.mgrid[0:h,0:w].astype(np.float32)
dx=(xx+.5)/S+x0-CX;dy=(yy+.5)/S+y0-CY
rr=np.hypot(dx,dy)
aa=np.degrees(np.arctan2(dx,-dy))%360
for name,start,end in [('hair_left_strip.png',276,360),('hair_right_strip.png',0,104)]:
    u=(aa-start)/(end-start)
    if start==276:
        outer=R+21+52*np.sin(np.clip(u,0,1)*np.pi/2)+3*np.sin(u*31*np.pi)**2
        su=.12+.88*u
    else:
        outer=R+77+25*(.5+.5*np.cos(u*12*np.pi))
        su=u
    valid=(aa>=start)&(aa<=end)&(rr>=R-2)&(rr<=outer)
    t=np.clip((rr-R)/(outer-R),0,1)
    strip=np.array(Image.open(A/name).convert('RGB'))
    col=np.stack([map_coordinates(strip[:,:,c].astype(float),[(1-t)*(strip.shape[0]-1),np.clip(su,0,1)*(strip.shape[1]-1)],order=1,mode='nearest') for c in range(3)],axis=-1)
    rgba=np.concatenate((np.clip(col,0,255).astype('uint8'),(valid*255).astype('uint8')[:,:,None]),axis=-1)
    hair=Image.fromarray(rgba)
    canvas.paste(hair,(x0*S,y0*S),hair)
    ts=np.linspace(start,end,600)
    uv=(ts-start)/(end-start)
    if start==276:rad=R+21+52*np.sin(uv*np.pi/2)+3*np.sin(uv*31*np.pi)**2
    else:rad=R+77+25*(.5+.5*np.cos(uv*12*np.pi))
    pts=[(CX+q*math.sin(math.radians(t)),CY-q*math.cos(math.radians(t))) for t,q in zip(ts,rad)]
    line(pts,INK,3.1)

# Quantitative face disk, exactly 360 degrees * share, clockwise from 12 o'clock.
bbox=(round((CX-R)*S),round((CY-R)*S))
N=round(2*R*S)
yy,xx=np.mgrid[0:N,0:N].astype(np.float32)
fx=(xx+.5)/S-R;fy=(yy+.5)/S-R
phi=np.degrees(np.arctan2(fx,-fy))%360
sid=np.searchsorted(angles[1:],phi,side='right').clip(0,len(values)-1)
mask=(fx*fx+fy*fy)<=R*R
rgb=np.array([[int(c[i:i+2],16) for i in (1,3,5)] for c in colors],dtype=np.float32)[sid]
# A restrained generated paper texture spans the face; it never alters geometry.
tex=Image.open(A/'face_paper_texture.png').convert('L').resize((N,N),Image.Resampling.BICUBIC)
t=np.array(tex,dtype=np.float32);grain=(t-t.mean())*.14
rgb=np.clip(rgb+grain[:,:,None],0,255).astype('uint8')
rgba=np.concatenate((rgb,(mask*255).astype('uint8')[:,:,None]),axis=-1)
face=Image.fromarray(rgba)
canvas.paste(face,bbox,face)

# Both eyes, the center nose and lips are substantive parts of the generated portrait.
paste_asset('eye_left.png',CX-374,CY-352,285,231)
paste_asset('eye_right.png',CX+91,CY-352,271,241)
paste_asset('nose.png',CX-88,CY-267,179,314)
paste_asset('lips.png',CX-156,CY+199,312,154)

# All true sector radii remain traceable even where they pass over facial features.
for deg in angles[:-1]:
    ar=math.radians(float(deg))
    line([(CX,CY),(CX+R*math.sin(ar),CY-R*math.cos(ar))],'#483A30',2.3)
circle(CX,CY,R,outline=INK,width=3.0)
circle(CX,CY,3.1,fill=INK)
# Short 12 o'clock start indicator, explicitly not an extra slice.
line([(CX,CY-R-7),(CX,CY-R-26)],INK,2)

# Text and leader locations adapt to this frozen distribution. Anchors are always
# derived from the actual category midpoint rather than from the reference art.
layouts=[
 {'x':1456,'y':565,'align':'left','tail':[(1384,632),(1434,632)]},
 {'x':1470,'y':1101,'align':'left','tail':[(1413,1185),(1448,1185)]},
 {'x':1470,'y':1300,'align':'left','tail':[(1400,1380),(1448,1380)]},
 {'x':1300,'y':1530,'align':'left','tail':[(1274,1500),(1274,1594)]},
 {'x':1135,'y':1710,'align':'center','tail':[(1135,1658),(1135,1687)]},
 {'x':813,'y':1734,'align':'center','tail':[(849,1676),(813,1711)]},
 {'x':417,'y':1585,'align':'center','tail':[(516,1532),(464,1562)]},
 {'x':110,'y':684,'align':'left','tail':[(354,781),(315,757)]},
]
geometry=[]
for i,(cat,v,lay) in enumerate(zip(categories,values,layouts)):
    mid=float((angles[i]+angles[i+1])/2)
    ar=math.radians(mid)
    anchor=(CX+(R+2)*math.sin(ar),CY-(R+2)*math.cos(ar))
    line([anchor]+lay['tail'],'#6F6153',1.7)
    circle(*anchor,3.6,fill=BG,outline='#584B40',width=1.4)
    group,role=cat.split(' ',1)
    text(group,lay['x'],lay['y'],32,'bold',INK,lay['align'])
    text(role,lay['x'],lay['y']+39,30,'regular',INK,lay['align'])
    label=f'{v:g}%'
    text(label,lay['x'],lay['y']+81,54,'bold',ACCENT,lay['align'])
    geometry.append({'category':cat,'value':float(v),'start_deg':float(angles[i]),'end_deg':float(angles[i+1]),'angle_deg':float(angles[i+1]-angles[i]),'mid_deg':mid,'anchor_xy':list(anchor),'label':label,'color':colors[i]})

# Method note separates composition shares from an unemployment rate.
line([(101,1898),(1698,1898)],'#C9BFB1',1.2)
text('Clockwise from 12 o’clock; the nose is the center. Only the circular face encodes shares.',101,1920,24,'regular',MUTED)
text('Hair and ear are illustrative. Total = 100%. Synthetic composition, not observed statistics.',101,1952,24,'regular',MUTED)

canvas.resize((W,H),Image.Resampling.LANCZOS).save(ROOT/'final.png')
(ROOT/'artifacts').mkdir(exist_ok=True)
(ROOT/'artifacts/geometry.json').write_text(json.dumps({'canvas':[W,H],'center':[CX,CY],'radius':R,'clockwise_origin':'12 oclock','categories':geometry,'text_boxes':text_records,'frozen_reference_size':list(reference_size)},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(f'Rendered {W}×{H}; {len(values)} exact sectors; total {values.sum():g}%.')
