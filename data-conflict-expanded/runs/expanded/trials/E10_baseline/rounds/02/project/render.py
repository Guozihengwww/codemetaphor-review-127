#!/usr/bin/env python3
"""Offline deterministic renderer. Full candidates must be captured by capture_render.py."""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parent
S = 2
INK = '#232B2B'
MUTED = '#6C7169'
PAPER = '#F4EFE4'

def read_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))

def font(size, kind='regular'):
    names = {'regular':'NimbusSans-Regular.otf','bold':'NimbusSans-Bold.otf',
             'italic':'NimbusRoman-Italic.otf','mono':'DejaVuSansMono.ttf'}
    return ImageFont.truetype(str(ROOT/'assets/fonts'/names[kind]), round(size*S))

def main():
    inputs = read_json('input.json')
    brief = read_json(inputs['brief'])
    data = read_json(inputs['data'])
    core = read_json(inputs['critical_metaphor_contract'])
    assert core['visible_title'] == data['title']
    assert (ROOT/inputs['reference']).is_file()
    width, height = brief['width'], brief['height']
    cats = data['categories']
    series = data['series'][0]
    vals = np.asarray(series['values'], dtype=float)
    total = float(vals.sum())
    assert total == data['series_totals'][series['id']] == 100
    assert len(cats) == len(vals) == 7

    # Paper grain is decorative; it never changes data coordinates.
    rng = np.random.default_rng(418)
    base = np.array([244,239,228], dtype=float)
    grain = rng.normal(0, .65, (height, width, 1))
    rgb = np.clip(base[None,None,:] + grain, 0,255).astype(np.uint8)
    canvas = Image.fromarray(rgb).resize((width*S,height*S), Image.Resampling.BICUBIC).convert('RGBA')
    draw = ImageDraw.Draw(canvas)

    def line(points, fill=INK, stroke=1):
        draw.line([(round(x*S),round(y*S)) for x,y in points], fill=fill,
                  width=max(1,round(stroke*S)), joint='curve')
    def text(x,y,txt,size,kind='regular',fill=INK,anchor='lt'):
        draw.text((x*S,y*S),txt,font=font(size,kind),fill=fill,anchor=anchor)
    def tracked(x,y,txt,size=22,tracking=3,fill=MUTED):
        f=font(size,'bold')
        for c in txt:
            draw.text((x*S,y*S),c,font=f,fill=fill,anchor='lt')
            x += draw.textlength(c,font=f)/S + tracking
    def paste_asset(name, box):
        x,y,w,h=box
        im=Image.open(ROOT/'assets'/name).convert('RGBA')
        im=im.resize((round(w*S),round(h*S)),Image.Resampling.LANCZOS)
        canvas.alpha_composite(im,(round(x*S),round(y*S)))

    # Editorial heading and semantic parent labels.
    tracked(112,65,'ENERGY / EXPORT FLOWS',22,3.1)
    draw.rounded_rectangle((1537*S,51*S,1936*S,107*S),radius=6*S,fill=INK)
    text(1737,79,data['visible_notice'],25,'bold',PAPER,'mm')
    text(108,143,'Oil export destinations',112,'bold')
    text(112,278,series['label']+' ('+series['unit']+')',34)
    text(1936,284,'7 markets  /  100% total',27,'regular',MUTED,'rt')
    line([(112,338),(1936,338)],'#AEB1A7',1.4)

    # A common coordinate system for all seven front surfaces.
    x0, x1 = 428.0, 1104.0
    cx, rx = (x0+x1)/2, (x1-x0)/2
    y0, body_h, ry = 618.0, 800.0, 54.0
    y1 = y0+body_h
    bounds = y0+np.r_[0,np.cumsum(vals)]/total*body_h
    colors = ['#B34E37','#E9D5B8','#C98A4E','#EEE6D6','#C56448','#B8BFAD','#54716A']
    xs = np.linspace(x0,x1,677)
    arc = ry*np.sqrt(np.maximum(0,1-((xs-cx)/rx)**2))
    def arc_y(x):
        return ry*math.sqrt(max(0,1-((x-cx)/rx)**2))

    # The generated robe supports the cylindrical face and the white beard.
    paste_asset('shoulders.png',(160,1397,1132,468))

    # Generated black turban, fitted to the barrel. Any cloth inside the data
    # face is removed at its exact top front boundary, so no layer is hidden.
    tx,ty,tw,th = 195,367,1000,731
    turban = Image.open(ROOT/'assets/turban.png').convert('RGBA').resize((tw*S,th*S),Image.Resampling.LANCZOS)
    tarr = np.array(turban)
    xx = tx+(np.arange(tw*S)+.5)/S
    yy = ty+(np.arange(th*S)+.5)/S
    tf = ry*np.sqrt(np.maximum(0,1-((xx-cx)/rx)**2))
    hide = ((xx[None,:]>=x0)&(xx[None,:]<=x1)&(yy[:,None]>=y0+tf[None,:]-5))
    tarr[:,:,3][hide]=0
    canvas.alpha_composite(Image.fromarray(tarr),(tx*S,ty*S))

    # A non-data metallic top lid. The front rim shares the top interface.
    rim_box=(x0*S,(y0-ry)*S,x1*S,(y0+ry)*S)
    draw.ellipse(rim_box,fill='#757A70',outline=INK,width=3*S)
    draw.ellipse(((x0+12)*S,(y0-ry+9)*S,(x1-12)*S,(y0+ry-9)*S),
                 fill='#A9ACA0',outline='#D9D8C9',width=3*S)
    # Brushed highlight strokes on the lid, behind its front rim.
    for dx in np.linspace(-.87,.87,110):
        px=cx+rx*dx
        radius_y=(ry-13)*math.sqrt(max(0,1-dx*dx))
        line([(px,y0-radius_y),(px,y0+radius_y)],'#B5B7AA',.4)

    # Exact axial heights. Every interface is a vertical translation of the
    # same front half ellipse. No per-layer padding or minimum height exists.
    lft,top = int(x0),int(y0)
    bw,bh = int(x1-x0)+1,math.ceil(body_h+ry)+1
    gx = lft+(np.arange(bw*S)+.5)/S
    gy = top+(np.arange(bh*S)+.5)/S
    t=(gx-cx)/rx
    f=ry*np.sqrt(np.maximum(0,1-t*t))
    z=gy[:,None]-y0-f[None,:]
    bands=np.searchsorted(np.cumsum(vals)/total*body_h,z,side='right')
    inside=(z>=0)&(z<body_h)&(np.abs(t[None,:])<=1)
    palette=np.asarray([tuple(bytes.fromhex(c[1:])) for c in colors],dtype=float)
    pixels=palette[np.minimum(bands,6)]
    shade=.64+.30*np.sqrt(np.maximum(0,1-t*t))+.12*(1-t)/2
    shade+=.040*np.exp(-((t+.48)/.16)**2)
    texture=rng.normal(0,.95,(bh*S,bw*S,1))
    pixels=np.clip(pixels*shade[None,:,None]+texture,0,255).astype(np.uint8)
    surface=np.zeros((bh*S,bw*S,4),dtype=np.uint8)
    surface[:,:,:3]=pixels
    surface[:,:,3]=inside*255
    canvas.alpha_composite(Image.fromarray(surface),(lft*S,top*S))

    # Crisp black seams and tiny adjacent glints preserve the drum identity
    # without adding height to any of the seven layers.
    for j,y in enumerate(bounds):
        path=list(zip(xs,y+arc))
        line(path,INK,2.3 if j in (0,7) else 1.8)
        if j not in (0,7):
            line(list(zip(xs[3:-3],y+arc[3:-3]+1.9)), '#F3EAD7',.8)
    line([(x0,y0),(x0,y1)],INK,2.5)
    line([(x1,y0),(x1,y1)],INK,2.5)
    # Bottom metal lip lies outside the data body, like the top lid.
    line(list(zip(xs,y1+arc+5)),INK,3.4)
    line(list(zip(xs[4:-4],y1+arc[4:-4]+2.7)),'#E0DDCA',1.4)

    # Warp only the generated beard. Its top contacts the translated bottom
    # ellipse; its hair remains strictly outside the quantitative face.
    beard=np.array(Image.open(ROOT/'assets/beard.png').convert('RGBA'))
    ba=beard[:,:,3]
    valid=ba>18
    src_top=np.argmax(valid,axis=0).astype(float)
    src_top[~valid.any(axis=0)]=beard.shape[0]
    bx,bw2=496,540
    by,bh2=1408,486
    dx=bx+(np.arange(bw2*S)+.5)/S
    syx=(np.arange(bw2*S)+.5)/(bw2*S)*(beard.shape[1]-1)
    top_source=np.interp(syx,np.arange(beard.shape[1]),src_top)
    contact=y1+ry*np.sqrt(np.maximum(0,1-((dx-cx)/rx)**2))+7
    dest_y=by+(np.arange(bh2*S)+.5)/S
    source_y=(dest_y[:,None]-contact[None,:])/.72+top_source[None,:]
    source_x=np.broadcast_to(syx,source_y.shape)
    warped=np.stack([map_coordinates(beard[:,:,c].astype(float),[source_y,source_x],order=1,mode='constant',cval=0) for c in range(4)],axis=-1)
    warped[:,:,3][dest_y[:,None]<contact[None,:]]=0
    canvas.alpha_composite(Image.fromarray(np.clip(warped,0,255).astype(np.uint8)),(bx*S,by*S))

    # Mandatory labels share precisely the same ordered data transform.
    tracked(1300,548,'EXPORT DESTINATION',21,2.0)
    tracked(1779,548,'SHARE (%)',21,1.7)
    line([(1300,594),(1936,594)],'#AEB1A7',1.2)
    rows=[]
    sample_x=x1-40
    for i,(cat,val,col) in enumerate(zip(cats,vals,colors)):
        cy=(bounds[i]+bounds[i+1])/2+arc_y(sample_x)
        # Attachment marker sits on the appropriate material surface.
        line([(sample_x,cy),(1268,cy)],'#798076',1.45)
        draw.ellipse(((sample_x-3)*S,(cy-3)*S,(sample_x+3)*S,(cy+3)*S),fill=INK)
        draw.rectangle((1300*S,(cy-12)*S,1324*S,(cy+12)*S),fill=col,outline=INK,width=S)
        text(1350,cy+1,cat,42,'regular',INK,'lm')
        value_text=f'{val:g}%'
        text(1936,cy+2,value_text,52,'bold',INK,'rm')
        rows.append({'category':cat,'value':float(val),'color':col,'axis_start_y':float(bounds[i]),
                     'axis_end_y':float(bounds[i+1]),'axial_height_px':float(bounds[i+1]-bounds[i]),
                     'label_center_y':float(cy),'attachment_x':sample_x})

    # A concise, explicitly non-data explanatory caption.
    text(1300,1598,'One barrel. Seven markets.',35,'italic')
    text(1300,1664,'Layer height represents share',27,'regular',MUTED)
    text(1300,1703,'of total synthetic oil exports.',27,'regular',MUTED)
    text(1300,1756,'All seven layers sum to 100%.',27,'regular',MUTED)
    line([(112,1904),(1936,1904)],'#AEB1A7',1.2)
    text(112,1937,'SYNTHETIC DATA  /  Illustrative markets; not observed trade statistics.',24,'regular',MUTED)
    text(1936,1937,'Height = share  •  Top → bottom: A–G',24,'regular',MUTED,'rt')

    canvas.convert('RGB').resize((width,height),Image.Resampling.LANCZOS).save(ROOT/'final.png')
    metadata={'canvas':[width,height],'barrel':{'x0':x0,'x1':x1,'top_axis_y':y0,'height':body_h,
                'ellipse_radius_y':ry,'bottom_axis_y':y1},'rows':rows,'category_order':cats,
                'total':total,'data_label_unit':series['unit'],'image_generation_calls':1,
                'asset_placement':{'turban':[tx,ty,tw,th],'shoulders':[160,1397,1132,468],
                                   'beard':[bx,by,bw2,bh2]}}
    (ROOT/'artifacts').mkdir(exist_ok=True)
    (ROOT/'artifacts/geometry.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Rendered final.png at',width,'x',height,'with seven proportional barrel layers.')

if __name__=='__main__':
    main()
