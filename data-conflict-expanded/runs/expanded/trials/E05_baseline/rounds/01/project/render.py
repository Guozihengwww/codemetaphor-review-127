"""Offline book-sales infographic. Run with the supplied capture_render.py wrapper.

The single quantitative mapping is y = zero_y - value * plot_height / axis_max.
Only the spine extension above zero carries data; the lower bindings are constant.
Generated bookcloth, caps and tail bands are mapped into these exact rectangles.
"""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
S = 3
FONTS = ROOT / 'assets/fonts'
INK = '#283128'
IVORY = '#fff6db'
MUTED = '#66532e'
GOLD = '#ddc185'


def load(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))


def font(size, style='sans'):
    fn={'sans':'Sans.otf','bold':'Sans-Bold.otf','serif':'EditorialSerif.otf','serifbold':'EditorialSerif-Bold.otf'}[style]
    return ImageFont.truetype(str(FONTS/fn), round(size*S))


def render():
    inp=load('input.json')
    brief=load(inp['brief'])
    data=load(inp['data'])
    core=load(inp['critical_metaphor_contract'])
    assert (ROOT/inp['reference']).is_file(), 'Keep the frozen visual-reference dependency.'
    assert core['visible_title']==data['title']
    assert data['value_axis']=={'minimum':0,'maximum':4000,'scale':'linear'}
    W,H=brief['width'],brief['height']
    values=data['series'][0]['values']
    cats=data['categories']
    assert len(values)==len(cats)==10
    assert sum(values)==data['series_totals'][data['series'][0]['id']]

    # Warm paper, deterministic and deliberately quiet behind the data.
    rng=np.random.default_rng(512)
    paper=np.empty((H,W,3),dtype=np.uint8)
    noise=rng.normal(0,0.65,(H,W))
    yy=np.arange(H)[:,None]/H
    base=np.array([236,184,60])
    for k in range(3):
        paper[:,:,k]=np.clip(base[k]+noise+(1-yy)*2,0,255).astype(np.uint8)
    im=Image.fromarray(paper).resize((W*S,H*S),Image.Resampling.BICUBIC)
    d=ImageDraw.Draw(im)

    def line(coords,fill=INK,width=1):
        d.line([(round(x*S),round(y*S)) for x,y in coords],fill=fill,width=max(1,round(width*S)))
    def rect(box,fill,outline=None,width=1):
        d.rectangle(tuple(round(x*S) for x in box),fill=fill,outline=outline,width=round(width*S))
    def txt(x,y,t,size=30,style='sans',fill=INK,align='left'):
        f=font(size,style)
        bb=d.textbbox((0,0),t,font=f)
        width=(bb[2]-bb[0])/S
        if align=='center': x-=width/2
        elif align=='right': x-=width
        d.text((round(x*S)-bb[0],round(y*S)-bb[1]),t,font=f,fill=fill)
        return width,(bb[3]-bb[1])/S
    def track(x,y,t,size=26,spacing=3,fill=INK):
        for ch in t:
            f=font(size,'bold')
            bb=d.textbbox((0,0),ch,font=f)
            d.text((round(x*S)-bb[0],round(y*S)-bb[1]),ch,font=f,fill=fill)
            x+=d.textlength(ch,font=f)/S+spacing
    def rotated(x,y,t,size=26,style='sans',fill=INK,angle=90):
        f=font(size,style)
        bb=f.getbbox(t)
        layer=Image.new('RGBA',(bb[2]-bb[0]+8*S,bb[3]-bb[1]+8*S))
        q=ImageDraw.Draw(layer)
        q.text((4*S-bb[0],4*S-bb[1]),t,font=f,fill=fill)
        layer=layer.rotate(angle,expand=True,resample=Image.Resampling.BICUBIC)
        im.paste(layer,(round(x*S-layer.width/2),round(y*S-layer.height/2)),layer)

    # Editorial hierarchy occupies the negative space created by the current data.
    track(80,57,'BOOKS / A–J',21,2.3)
    bx,by,bw,bh=1510,48,405,63
    rect((bx,by,bx+bw,by+bh),INK)
    txt(bx+bw/2,by+20,data['visible_notice'],28,'bold',IVORY,'center')

    track(667,232,'SYNTHETIC',48,7.0)
    txt(653,312,'Book sales',154,'serifbold')
    txt(668,500,data['series'][0]['label'],40,'sans')
    txt(668,554,'Unit: '+data['series'][0]['unit'],30,'sans',MUTED)
    line([(668,625),(1877,625)],INK,1.8)

    track(670,676,'TOTAL COPIES SOLD',22,2.4)
    txt(660,721,f'{sum(values):,}',102,'serifbold')
    txt(1036,752,'million copies',37,'serif')
    txt(1038,804,'across ten synthetic books',28,'sans',MUTED)

    # Explain the unusual baseline before the eye reaches the short marks.
    line([(668,901),(732,901)],IVORY,4)
    txt(754,881,'Read upward from the white zero line.',31,'bold')
    txt(754,930,'The equal lower bindings are decorative.',29,'sans',MUTED)
    txt(754,974,'Linear scale · 0–4,000 million copies',26,'sans',MUTED)

    # Book geometry: common width, fixed order, common decorative depth.
    x_start,book_w,gap=265.0,145.0,18.0
    zero_y,plot_h,bottom=1147.0,1020.0,1725.0
    top_y=zero_y-plot_h
    scale=plot_h/data['value_axis']['maximum']
    rect((72,zero_y,W-72,bottom+20),'#dfaa40')

    # Full-range tick marks, without grids that would compete with the bookcloth.
    for value in range(0,4001,500):
        y=zero_y-value*scale
        line([(204,y),(238,y)],INK,1.35)
        txt(184,y-12,f'{value:,}',26,'sans',INK,'right')
    rotated(73,641,'MILLION COPIES',23,'bold',INK,90)

    geometries=[]
    for i,(cat,value) in enumerate(zip(cats,values)):
        x=x_start+i*(book_w+gap)
        y=zero_y-value*scale
        x0,x1=round(x*S),round((x+book_w)*S)
        y0,y1=round(y*S),round(bottom*S)
        tex=Image.open(ROOT/f'assets/spine_{i+1:02d}.png').convert('RGB')
        tw,th=tex.size
        width_px=x1-x0
        height_px=y1-y0
        cap_h=round(24*S)
        tail_h=round(27*S)
        # Slice the generated cap/tail separately so no decorative band is stretched
        # into a spurious quantitative mark. The middle is still generated bookcloth.
        spine=Image.new('RGB',(width_px,height_px))
        head=tex.crop((0,0,tw,43)).resize((width_px,cap_h),Image.Resampling.LANCZOS)
        body=tex.crop((0,43,tw,th-51)).resize((width_px,height_px-cap_h-tail_h),Image.Resampling.LANCZOS)
        tail=tex.crop((0,th-51,tw,th)).resize((width_px,tail_h),Image.Resampling.LANCZOS)
        spine.paste(head,(0,0));spine.paste(body,(0,cap_h));spine.paste(tail,(0,height_px-tail_h))
        im.paste(spine,(x0,y0))
        # Integral generated rails remain visible. Only identity lettering is added.
        center=x+book_w/2
        txt(center,y-65,f'{value:,}',48,'bold',INK,'center')
        # A tiny cream ornament is part of the fixed, non-data binding.
        cy=zero_y+72
        line([(center-35,cy),(center-8,cy)],GOLD,1.2)
        line([(center+8,cy),(center+35,cy)],GOLD,1.2)
        diamond=[(center,cy-5),(center+5,cy),(center,cy+5),(center-5,cy),(center,cy-5)]
        line(diamond,GOLD,1.4)
        rotated(center,zero_y+261,cat,58,'serifbold',IVORY,-90)
        line([(x+15,zero_y+433),(x+book_w-15,zero_y+433)],GOLD,1.1)
        line([(x+15,zero_y+440),(x+book_w-15,zero_y+440)],GOLD,0.7)
        txt(center,zero_y+467,'AUTHOR',17,'bold',IVORY,'center')
        txt(center,zero_y+497,'unspecified',18,'serif',IVORY,'center')
        geometries.append({'category':cat,'value':value,'unit':data['series'][0]['unit'],
            'x_left':x,'x_right':x+book_w,'y_top':y,'y_zero':zero_y,'y_bottom':bottom,
            'height_above_zero':zero_y-y,'rendered_supersampled_top':y0/S,
            'title':cat,'author_display':'AUTHOR / unspecified',
            'texture':f'assets/spine_{i+1:02d}.png'})

    # The zero line sits wholly in the non-data region; it never consumes a small mark.
    line([(202,zero_y+1),(1898,zero_y+1)],IVORY,2.0)
    txt(177,zero_y+17,'ZERO',18,'bold',INK,'right')

    # Bracket explicitly marks the common, non-quantitative lower part of every book.
    line([(223,zero_y+28),(211,zero_y+28),(211,bottom-4),(223,bottom-4)],MUTED,1.25)
    rotated(155,(zero_y+bottom)/2+12,'DECORATIVE SECTION · NO DATA',20,'bold',MUTED,90)

    # A restrained shelf grounds all complete books at the same physical level.
    rect((72,bottom+2,1928,bottom+13),'#654627')
    rect((72,bottom+13,1928,bottom+22),'#ad7d37')
    txt(80,1787,'Synthetic titles; author names are unspecified.',25,'sans',INK)
    txt(80,1824,'Only height above zero encodes sales. Full book height is not a value.',24,'sans',MUTED)
    txt(1916,1787,'10 BOOKS  /  1 SERIES',23,'bold',INK,'right')
    txt(1916,1824,'SOURCE: SUPPLIED SYNTHETIC DATA',19,'sans',MUTED,'right')

    # The final resample preserves subpixel marks without imposing a minimum height.
    im.resize((W,H),Image.Resampling.LANCZOS).save(ROOT/'final.png')
    info={'canvas':[W,H],'scale_px_per_million':scale,'axis_min':0,'axis_max':4000,
          'zero_y':zero_y,'plot_top_y':top_y,'decorative_bottom_y':bottom,
          'equal_book_width':book_w,'gap':gap,'supersampling':S,
          'minimum_size_override':False,'reference_pixels_used':False,
          'series':data['series'][0]['label'],'unit':data['series'][0]['unit'],
          'total':sum(values),'books':geometries}
    (ROOT/'artifacts/geometry.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Rendered final.png, 2000 × 1869; 10 generated-texture book spines; single linear mapping.')


if __name__=='__main__':
    render()
