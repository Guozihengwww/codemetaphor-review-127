"""Rebuild final.png locally. Run complete renders through capture_render.py.

All seven book fronts share one affine percentage-to-x mapping. Textures and
edge shading are confined to their own half-open intervals. There is no minimum
width, inter-book gap, or overlap. No model or network calls are used.
"""
from pathlib import Path
from decimal import Decimal
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent
S = 4
FONTS = ROOT/'assets/fonts'
COL = {'ink':'#262923', 'muted':'#676b60', 'rule':'#aaa793', 'lead':'#72756a', 'gold':'#9b8455'}

def font(name, size):
    files={'sans':'NotoSans-Regular.ttf', 'serif':'NimbusRoman-Regular.otf',
           'bold':'NimbusRoman-Bold.otf', 'italic':'NimbusRoman-Italic.otf',
           'cjk':'NotoSerifCJK-Regular.ttc'}
    return ImageFont.truetype(str(FONTS/files[name]), round(size*S), index=2 if name=='cjk' else 0)

def main():
    data = json.loads((ROOT/'data.json').read_text(encoding='utf-8'))
    brief = json.loads((ROOT/'brief.json').read_text(encoding='utf-8'))
    width, height = brief['width'], brief['height']
    values = [Decimal(str(v)) for v in data['series'][0]['values']]
    names = data['categories']
    assert len(names)==len(values)==7 and sum(values)==100
    assert data['value_axis']=={'minimum':0,'maximum':100,'scale':'linear'}
    x0, top, total_w, book_h = 120., 399., 1760., 640.
    bottom = top+book_h
    cum = [Decimal(0)]
    for value in values:
        cum.append(cum[-1]+value)
    edges = [x0 + total_w*float(t)/100 for t in cum]
    # Pixel centers implement one shared, gapless ownership map at 4x resolution.
    q = [math.ceil(x*S - 0.5) for x in edges]
    bounds = []
    text_boxes = []

    yy, xx = np.mgrid[0:height,0:width]
    lighting = 1.2*np.exp(-((xx-width*.5)/(width*.6))**2-((yy-height*.3)/(height*.9))**2)
    grain = np.random.default_rng(21).normal(0,.23,(height,width))
    bg = np.clip(np.array([244,242,233])[None,None,:]+lighting[:,:,None]+grain[:,:,None],0,255).astype('uint8')
    canvas = Image.fromarray(bg).resize((width*S,height*S),Image.Resampling.BICUBIC).convert('RGB')
    draw = ImageDraw.Draw(canvas)

    def line(points, fill=COL['rule'], w=1):
        draw.line([(round(x*S),round(y*S)) for x,y in points], fill=fill, width=max(1,round(w*S)), joint='curve')

    def text(x,y,content,size=24,face='sans',fill=COL['ink'],anchor='lt',tag=None):
        f=font(face,size)
        xy=(round(x*S),round(y*S))
        draw.text(xy,content,font=f,fill=fill,anchor=anchor)
        bb=draw.textbbox(xy,content,font=f,anchor=anchor)
        text_boxes.append({'text':content,'bbox':[round(v/S,3) for v in bb], 'tag':tag or 'annotation'})
        return (bb[2]-bb[0])/S

    def spaced(center,y,content,size,tracking,fill):
        f=font('sans',size)
        widths=[draw.textlength(c,font=f)/S for c in content]
        full=sum(widths)+tracking*(len(content)-1)
        x=center-full/2
        for c,w in zip(content,widths):
            draw.text((round(x*S),round(y*S)),c,font=f,fill=fill,anchor='lt')
            x+=w+tracking
        text_boxes.append({'text':content,'bbox':[center-full/2,y,center+full/2,y+size*1.3], 'tag':'notice'})

    # Quiet editorial heading, with the frozen title reproduced by code.
    spaced(1000,63,data['visible_notice'],18,4.1,COL['muted'])
    line([(614,73),(791,73)],COL['rule'],.8)
    line([(1209,73),(1386,73)],COL['rule'],.8)
    title = data['title']
    latin, chinese = title.split(' ',1)
    f_latin, f_cjk = font('italic',96), font('cjk',83)
    title_w = draw.textlength(latin,font=f_latin)/S+26+draw.textlength(chinese,font=f_cjk)/S
    tx=(width-title_w)/2
    text(tx,133,latin,96,'italic',tag='title')
    tx+=draw.textlength(latin,font=f_latin)/S+26
    text(tx,132,chinese,83,'cjk',tag='title')
    text(1000,248,'Equal height. Continuous width. A full collection at 100%.',25,'serif',COL['muted'],'mt')

    text(x0,314,data['series'][0]['label']+' (%)',22,'sans',COL['ink'])
    text(x0+total_w,317,'7 CATEGORIES  /  LINEAR WIDTH SCALE',16,'sans',COL['muted'],'rt')
    axis_y=373
    line([(x0,axis_y),(x0+total_w,axis_y)],COL['rule'],.8)
    for tick in range(0,101,25):
        x=x0+total_w*tick/100
        line([(x,axis_y-6),(x,axis_y+3)],COL['rule'],.8)
        text(x,345,f'{tick}%',17,'sans',COL['muted'],'mt',tag='scale')

    # Shadows are beneath the data band; no shadow can cover a neighbouring strip.
    sh = Image.new('L',(width*S,height*S),0)
    sd=ImageDraw.Draw(sh)
    sd.ellipse((int((x0-11)*S),int((bottom-4)*S),int((x0+total_w+18)*S),int((bottom+23)*S)),fill=58)
    sh=sh.filter(ImageFilter.GaussianBlur(10*S))
    canvas.paste((48,45,37),mask=sh)
    sh2=Image.new('L',canvas.size,0)
    sd=ImageDraw.Draw(sh2)
    sd.rectangle((int(x0*S),int((bottom-2)*S),int((x0+total_w)*S),int((bottom+2)*S)),fill=75)
    sh2=sh2.filter(ImageFilter.GaussianBlur(2*S))
    canvas.paste((40,37,30),mask=sh2)

    # Map generated front-face material to the exact code-owned data intervals.
    book_mask = Image.new('L',(width*S,round(book_h*S)),0)
    md=ImageDraw.Draw(book_mask)
    for i,(name,value) in enumerate(zip(names,values)):
        key=chr(65+i)
        asset=Image.open(ROOT/f'assets/spine_{key}.png').convert('RGB')
        pw=q[i+1]-q[i]
        skin=asset.resize((pw,round(book_h*S)),Image.Resampling.LANCZOS)
        # Additional softly shaded binding edges stay within the existing strip.
        a=np.asarray(skin).astype(np.float32)
        xpos=(np.arange(pw)+.5)/S
        exact_width=edges[i+1]-edges[i]
        edge_px=min(4.8,exact_width*.045)
        shade=1-.12*np.exp(-xpos/max(edge_px,.01))-.20*np.exp(-(pw/S-xpos)/max(edge_px,.01))
        a*=shade[None,:,None]
        skin=Image.fromarray(np.clip(a,0,255).astype('uint8'))
        canvas.paste(skin,(q[i],round(top*S)))
        md.rectangle((q[i],0,q[i+1]-1,round(book_h*S)-1),fill=i+1)
        bounds.append({'category':name,'value_percent':str(value),'asset':f'assets/spine_{key}.png',
                       'left':edges[i],'right':edges[i+1], 'width':exact_width,
                       'raster_left_4x':q[i],'raster_right_4x_exclusive':q[i+1],
                       'raster_width_final_px':pw/S,'height':book_h})

    draw=ImageDraw.Draw(canvas)
    # All mandatory values and categories are external, staggered, and code-owned.
    label_x=[(edges[i]+edges[i+1])/2 for i in range(7)]
    label_x[5]=1641
    label_y=[1108,1217,1108,1217,1108,1217,1108]
    leaders=[]
    for i,(name,value) in enumerate(zip(names,values)):
        center=(edges[i]+edges[i+1])/2
        ly=label_y[i]
        anchor=(center,bottom-6)
        if i==5:
            # The thin anchor remains wholly inside F's actual 2.112 px strip.
            pts=[anchor,(center,1080),(label_x[i],1183),(label_x[i],ly-17)]
            leader_color='#365a7d'
            line(pts,leader_color,.8)
            dot_r=.36
        else:
            pts=[anchor,(center,ly-17)]
            leader_color=COL['lead']
            line(pts,leader_color,.9)
            dot_r=2.2
        draw.ellipse(((center-dot_r)*S,(anchor[1]-dot_r)*S,(center+dot_r)*S,(anchor[1]+dot_r)*S),fill=leader_color)
        # Separate percent sign stays at the same type size for exact legibility.
        number=format(value,'f').rstrip('0').rstrip('.') if '.' in format(value,'f') else format(value,'f')
        text(label_x[i],ly,number+'%',49,'bold',COL['ink'],'mt',tag=name+' value')
        text(label_x[i],ly+55,name,23,'sans',COL['muted'],'mt',tag=name+' category')
        leaders.append({'category':name,'anchor_xy':list(anchor),'path_xy':[list(p) for p in pts],
                        'stroke_px':.8 if i==5 else .9,
                        'anchor_inside_own_band':edges[i]<center<edges[i+1]})

    line([(120,1336),(1880,1336)],COL['rule'],.7)
    text(120,1355,'Width encodes share; all seven book spines have equal height.',17,'sans',COL['muted'])
    text(1880,1355,'Source: supplied synthetic data',17,'sans',COL['muted'],'rt')
    final=canvas.resize((width,height),Image.Resampling.LANCZOS)
    final.save(ROOT/'final.png',optimize=True)
    (ROOT/'artifacts').mkdir(exist_ok=True)
    # Diagnostic object-ownership map is a narrow mask, not a complete chart.
    book_mask.resize((width,round(book_h)),Image.Resampling.NEAREST).save(ROOT/'artifacts/band_ownership.png')
    geometry={'canvas':[width,height],'scale':S,'data_band':{'x':x0,'y':top,'width':total_w,'height':book_h},
              'cumulative_percent':[str(v) for v in cum], 'bands':bounds,'leaders':leaders,
              'text_boxes':text_boxes,'minimum_width_applied':False,
              'series_label':data['series'][0]['label'],'unit':'percent'}
    (ROOT/'artifacts/render_geometry.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Rendered final.png: 2000 x 1392; seven adjacent data-owned book fronts.')

if __name__=='__main__':
    main()
