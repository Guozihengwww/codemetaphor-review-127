#!/usr/bin/env python3
"""Offline renderer. Run complete renders only via capture_render.py.

The only quantitative mapping is y = zero_y - value * plot_height / 4000.
All book widths and all below-zero portions are identical. Generated book
materials are clipped to that mapping; no mark has a minimum data height.
"""
from pathlib import Path
import json
import math
from decimal import Decimal
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import numpy as np

ROOT = Path(__file__).resolve().parent
S = 3
FONTS = {
    'sans': 'NimbusSans-Regular.otf',
    'bold': 'NimbusSans-Bold.otf',
    'serif': 'P052-Roman.otf',
    'serif_bold': 'P052-Bold.otf',
    'italic': 'P052-Italic.otf',
}
BG = '#eab53d'
INK = '#29271f'
CREAM = '#fff5d8'
GOLD = '#d6b671'

def font(size, family='sans'):
    return ImageFont.truetype(str(ROOT/'assets/fonts'/FONTS[family]), round(size*S))

def number(v):
    # Decimal places in the supplied JSON are retained (including 1933.0).
    t = str(v)
    if '.' in t:
        whole, fraction = t.split('.')
        return f'{int(whole):,}.' + fraction
    return f'{int(t):,}'

def main():
    config = json.loads((ROOT/'input.json').read_text(encoding='utf-8'))
    brief = json.loads((ROOT/config['brief']).read_text(encoding='utf-8'))
    data = json.loads((ROOT/config['data']).read_text(encoding='utf-8'))
    core = json.loads((ROOT/config['critical_metaphor_contract']).read_text(encoding='utf-8'))
    if not (ROOT/config['reference']).is_file():
        raise FileNotFoundError('Frozen visual reference must remain available.')
    assert data['value_axis'] == {'minimum':0,'maximum':4000,'scale':'linear'}
    assert core['visible_title'] == data['title']
    assert data['synthetic'] is True
    W,H = brief['width'], brief['height']
    assert (W,H)==(2000,1869)
    cats = data['categories']
    vals = data['series'][0]['values']
    assert len(cats)==len(vals)==15
    assert all(0<=v<=4000 for v in vals)
    im = Image.new('RGB',(W*S,H*S),BG)
    d=ImageDraw.Draw(im)
    labels=[]
    def line(points,fill,width=1):
        d.line([(round(x*S),round(y*S))for x,y in points],fill=fill,width=max(1,round(width*S)))
    def rect(box,fill,outline=None,width=1):
        d.rectangle(tuple(round(x*S)for x in box),fill=fill,outline=outline,width=max(1,round(width*S)))
    def text(x,y,t,size=24,family='sans',fill=INK,align='left',track=None,role=None):
        f=font(size,family)
        if track is None:
            tw=d.textlength(t,font=f)/S
        else:
            tw=sum(d.textlength(c,font=f)/S for c in t)+track*max(0,len(t)-1)
        x0=x-tw/2 if align=='center' else x-tw if align=='right' else x
        if track is None:
            d.text((round(x0*S),round(y*S)),t,font=f,fill=fill,anchor='lt')
        else:
            xp=x0
            for c in t:
                d.text((round(xp*S),round(y*S)),c,font=f,fill=fill,anchor='lt')
                xp+=d.textlength(c,font=f)/S+track
        b=d.textbbox((round(x0*S),round(y*S)),t,font=f,anchor='lt')
        labels.append({'text':t,'role':role,'bbox':[x0,y,x0+tw,y+(b[3]-b[1])/S]})
        return tw

    # Quiet editorial masthead. The synthetic notice is deliberately prominent.
    text(80,73,'THE SYNTHETIC LIBRARY',23,'bold',track=2.4)
    rect((1540,58,1928,111),INK)
    text(1734,73,data['visible_notice'],24,'bold',CREAM,'center',track=1.5,role='synthetic_notice')
    line([(80,143),(1928,143)],'#a47d2b',1)
    text(80,167,'BOOKS  /  A DATA PORTRAIT',17,'sans',fill='#72541f',track=1.5)

    # Single shared linear scale. Lower cover sections do not encode values.
    zero=1220.0
    plot_height=950.0
    px_per_value=plot_height/(data['value_axis']['maximum']-data['value_axis']['minimum'])
    book_bottom=1610.0
    x_start=180.0
    book_width=106.0
    gap=12.0
    rect((0,zero,W,book_bottom+24),'#dca630')
    for tick in range(0,4001,500):
        y=zero-tick*px_per_value
        text(128,y-11,number(tick),25,'sans',fill=INK,align='right',role='axis_tick')
        line([(142,y),(164,y)],INK if tick==0 else '#896923',1.2)
    text(63,220,'MILLION',15,'bold',track=0.8)
    text(63,241,'COPIES',15,'bold',track=0.8)

    # Title is placed in the large unused middle of this dataset's silhouette.
    text(484,330,'SYNTHETIC',32,'bold',fill='#72541f',track=5.1,role='title')
    text(475,391,'Book sales',144,'serif_bold',role='title')
    text(486,581,data['series'][0]['label'],32,'sans',role='series')
    text(486,627,'('+data['series'][0]['unit']+')',27,'sans',fill='#6b5225',role='unit')
    line([(486,686),(1275,686)],'#ac842d',1.2)
    text(486,715,'Only the spine above zero measures sales.',26,'sans')
    text(486,761,'15 titles  ·  Linear scale  ·  Original category order',23,'sans',fill='#6b5225')

    palettes=['charcoal','vermilion','olive','petrol','ochre','oxblood','charcoal','olive','petrol','vermilion','oxblood','ochre','petrol','olive','vermilion']
    geometry=[]
    for i,(cat,v,material) in enumerate(zip(cats,vals,palettes)):
        x=x_start+i*(book_width+gap)
        xc=x+book_width/2
        top=zero-v*px_per_value
        # Supersampled edge coordinates are the only raster quantization.
        left_px=round(x*S)
        top_px=round(top*S)
        width_px=round(book_width*S)
        total_px=round(book_bottom*S)-top_px
        source=Image.open(ROOT/f'assets/{material}_body.png').convert('RGB')
        # Preserve fine grain and vertical wrap edges: width-normalize then
        # mirror-tile only along y, rather than distorting the cloth arbitrarily.
        tile_h=round(source.height*width_px/source.width)
        tile=source.resize((width_px,tile_h),Image.Resampling.LANCZOS)
        if material=='ochre':
            tile=ImageEnhance.Brightness(tile).enhance(0.77)
        book=Image.new('RGB',(width_px,total_px))
        offset=(i*47*S)%tile_h
        cursor=-offset
        k=0
        while cursor<total_px:
            section=tile if k%2==0 else tile.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            book.paste(section,(0,cursor))
            cursor+=tile_h
            k+=1
        # Generated bottom binding remains inside the non-data portion.
        foot=Image.open(ROOT/f'assets/{material}_foot.png').convert('RGB').resize((width_px,round(38*S)),Image.Resampling.LANCZOS)
        if material=='ochre':
            foot=ImageEnhance.Brightness(foot).enhance(0.77)
        book.paste(foot,(0,total_px-foot.height))
        bd=ImageDraw.Draw(book)
        # No shadow or cap protrudes above the exact data endpoint.
        bd.line([(0,0),(width_px-1,0)],fill='#26231d',width=round(1.1*S))
        bd.line([(round(4*S),round(2.3*S)),(width_px-round(4*S),round(2.3*S))],fill='#cbb27b',width=max(1,round(.55*S)))
        bd.line([(1,0),(1,total_px)],fill='#25241d',width=round(1*S))
        bd.line([(width_px-round(2*S),0),(width_px-round(2*S),total_px)],fill='#232018',width=round(1.2*S))
        im.paste(book,(left_px,top_px))
        # A complete bound book continues below zero, where identity lives.
        frame=(x+12,zero+39,x+book_width-12,zero+287)
        rect(frame,None,GOLD,.65)
        line([(x+12,zero+44),(x+book_width-12,zero+44)],GOLD,.6)
        # All required category/title labels are code-rendered.
        title_size=38
        while font(title_size,'serif_bold').getlength(cat)/S > 214:
            title_size-=.5
        tf=font(title_size,'serif_bold')
        box=tf.getbbox(cat,anchor='lt')
        tw=math.ceil(tf.getlength(cat))+12*S
        th=box[3]-box[1]+12*S
        tiletext=Image.new('RGBA',(tw,th),(0,0,0,0))
        td=ImageDraw.Draw(tiletext)
        td.text((6*S,6*S),cat,font=tf,fill=CREAM,anchor='lt')
        tiletext=tiletext.rotate(-90,expand=True)
        tx=round(xc*S-tiletext.width/2)
        ty=round((zero+164)*S-tiletext.height/2)
        im.paste(tiletext,(tx,ty),tiletext)
        labels.append({'text':cat,'role':'category','bbox':[tx/S,ty/S,(tx+tiletext.width)/S,(ty+tiletext.height)/S]})
        text(xc,zero+310,'AUTHOR',12,'sans',CREAM,'center',track=1.0,role='fictional_author')
        author_id=cat.replace('Book ','').replace('added ','')
        text(xc,zero+331,author_id,21,'serif_bold',CREAM,'center',role='fictional_author')
        value_label=number(v)
        fs=30
        # Long exact decimals have room because their tops occur at different y.
        text(xc,top-44,value_label,fs,'bold',INK,'center',role='value')
        geometry.append({'category':cat,'value':v,'formatted_value':value_label,'x':x,'width':book_width,'top_y':top,'zero_y':zero,'encoded_height_px':zero-top,'raster_top_y':top_px/S,'bottom_y':book_bottom,'material':material,'author_credit':f'Author {author_id} (fictional)'})

    # White zero line is an annotation, not a positive-height minimum.
    line([(164,zero),(1945,zero)],CREAM,1.3)
    text(72,1259,'ZERO',14,'bold',track=1)
    text(72,1282,'LINE',14,'bold',track=1)
    line([(131,1267),(154,1233)],'#71561f',1)
    # Distinguish the fixed identity area without attaching another chart.
    line([(145,1328),(145,1568)],'#8e6822',1)
    line([(137,1328),(153,1328)],'#8e6822',1)
    line([(137,1568),(153,1568)],'#8e6822',1)
    # Small rotated description in the unused left margin.
    side='DECORATIVE SPINE'
    sf=font(16,'bold')
    sb=sf.getbbox(side,anchor='lt')
    si=Image.new('RGBA',(math.ceil(sf.getlength(side))+8*S,sb[3]-sb[1]+8*S),(0,0,0,0))
    ImageDraw.Draw(si).text((4*S,4*S),side,font=sf,fill='#78551e',anchor='lt')
    si=si.rotate(-90,expand=True)
    im.paste(si,(round(111*S-si.width/2),round(1448*S-si.height/2)),si)

    # Shelf and concise interpretation note.
    rect((0,book_bottom,W,book_bottom+8),'#745027')
    rect((0,book_bottom+8,W,book_bottom+27),'#9a6831')
    line([(0,book_bottom+9),(W,book_bottom+9)],'#c68f43',1)
    rect((0,book_bottom+27,W,H),'#f7eedb')
    text(80,1676,'READING THE SHELF',20,'bold',track=2.1)
    text(80,1721,'Height above zero = synthetic copies sold.',30,'serif_bold')
    text(80,1765,'The shared lower section is decorative; total book height is not the value.',24,'sans')
    text(80,1820,'Book labels and author credits are fictional.',20,'sans',fill='#766d59')
    line([(1410,1677),(1410,1827)],'#c8bca3',1)
    total=data['series_totals'][data['series'][0]['id']]
    text(1480,1677,'15 TITLES  /  TOTAL',19,'bold',track=1.8)
    text(1475,1718,number(int(total)),61,'serif_bold')
    text(1480,1789,'million copies in this synthetic set',19,'sans',fill='#665f50')

    # Single final resampling. Nothing modifies final.png after it is captured.
    im=im.resize((W,H),Image.Resampling.LANCZOS)
    im.save(ROOT/'final.png',format='PNG',optimize=True)
    layout={'canvas':[W,H],'value_axis':data['value_axis'],'plot_height_px':plot_height,'pixels_per_million':px_per_value,'baseline_y':zero,'decorative_height_px':book_bottom-zero,'book_width_px':book_width,'gap_px':gap,'background_hex':BG,'supersampling':S,'books':geometry,'text_labels':labels,'generated_asset_uses':len(geometry),'reference_pixels_used':False}
    (ROOT/'artifacts').mkdir(exist_ok=True)
    (ROOT/'artifacts/layout.json').write_text(json.dumps(layout,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Rendered 2000 × 1869, 15 equal-width complete spines; all heights share one linear mapping.')

if __name__=='__main__':
    main()
