"""Offline renderer. Run from this directory with the supplied Python interpreter.

Every complete experimental rendering is launched by capture_render.py.
The flag is a generated textile clipped by the complete, unsmoothed data polygon.
"""
from pathlib import Path
import json
import math
import hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
S = 2


def read_json(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8'))


def main():
    inputs=read_json('input.json')
    brief=read_json(inputs['brief'])
    data=read_json(inputs['data'])
    core=read_json(inputs['critical_metaphor_contract'])
    assert (ROOT/inputs['reference']).is_file()
    assert data['synthetic'] is True and data['title']==core['visible_title']
    W,H=brief['width'],brief['height']
    times=np.asarray(data['time_points'],dtype=float)
    vals=np.asarray(data['series'][0]['values'],dtype=float)
    assert len(times)==len(vals)==len(data['categories'])
    assert [str(int(t)) for t in times]==data['categories']
    assert np.all(np.diff(times)>0)
    assert len(data['series'])==1 and data['value_axis']['scale']=='linear'
    lo,hi=data['value_axis']['minimum'],data['value_axis']['maximum']
    assert lo==0 and hi==130 and np.all((vals>=lo)&(vals<=hi))
    boundary=float(data['forecast_boundary'])
    x0,x1,y0,y1=150.,1490.,650.,1735.

    def x(t): return x0+(float(t)-times[0])/(times[-1]-times[0])*(x1-x0)
    def y(v): return y1-(float(v)-lo)/(hi-lo)*(y1-y0)
    pts=[(x(t),y(v)) for t,v in zip(times,vals)]
    lookup={int(t):(p,float(v)) for t,v,p in zip(times,vals,pts)}
    im=Image.open(ROOT/'assets/navy_weave.png').convert('RGB').resize((W*S,H*S),Image.Resampling.LANCZOS)
    draw=ImageDraw.Draw(im,'RGBA')
    fonts={}
    paper=(249,243,224,255)
    muted=(166,194,213,255)
    pale=(118,155,183,255)
    coral=(253,111,105,255)
    forecast=(255,237,187,255)

    def font(size,bold=False):
        key=(size,bold)
        if key not in fonts:
            name='NimbusSans-Bold.otf' if bold else 'NimbusSans-Regular.otf'
            fonts[key]=ImageFont.truetype(str(ROOT/'assets/fonts'/name),round(size*S))
        return fonts[key]

    def txt(text,px,py,size=28,bold=False,fill=paper,anchor='ls',tracking=None):
        if tracking is None:
            draw.text((round(px*S),round(py*S)),str(text),font=font(size,bold),fill=fill,anchor=anchor)
        else:
            total=sum(draw.textlength(c,font=font(size,bold))/S for c in str(text))+max(0,len(str(text))-1)*tracking
            if anchor.startswith('r'): px-=total
            if anchor.startswith('m'): px-=total/2
            for c in str(text):
                draw.text((round(px*S),round(py*S)),c,font=font(size,bold),fill=fill,anchor='ls')
                px+=draw.textlength(c,font=font(size,bold))/S+tracking

    def line(points,fill,width=1):
        draw.line([(round(px*S),round(py*S)) for px,py in points],fill=fill,width=max(1,round(width*S)),joint='curve')

    def rect(box,fill):
        draw.rectangle(tuple(round(v*S) for v in box),fill=fill)

    def dash(points,fill,width=2,dashlen=8,gap=5):
        phase=0.
        cycle=dashlen+gap
        for a,b in zip(points[:-1],points[1:]):
            dx,dy=b[0]-a[0],b[1]-a[1]
            length=math.hypot(dx,dy)
            if length==0: continue
            pos=0.
            while pos<length-1e-9:
                on=phase<dashlen
                remain=(dashlen-phase) if on else (cycle-phase)
                step=min(remain,length-pos)
                if on and step>1e-8:
                    line([(a[0]+dx*pos/length,a[1]+dy*pos/length),(a[0]+dx*(pos+step)/length,a[1]+dy*(pos+step)/length)],fill,width)
                pos+=step
                phase=(phase+step)%cycle
                if cycle-phase<1e-8: phase=0.

    def dot(px,py,r=3,fill=coral):
        draw.ellipse(((px-r)*S,(py-r)*S,(px+r)*S,(py+r)*S),fill=fill)

    # Editorial masthead; all text is drawn by code.
    line([(90,51),(1510,51)],(137,168,190,125),1)
    rect((90,77,412,126),(190,51,66,255))
    txt(data['visible_notice'],111,110,27,True,tracking=1.1)
    txt(f'ANNUAL SERIES  /  {int(times[0])}–{int(times[-1])}',1510,110,25,False,muted,'rs',tracking=.5)
    txt('Debt relative',84,274,164,True)
    txt('to output',84,432,164,True)
    line([(1088,254),(1088,361)],(191,59,73,255),4)
    txt('Debt as a share',1118,279,29)
    txt('of synthetic',1118,317,29)
    txt('economic output.',1118,355,29)
    txt(data['series'][0]['label'],91,498,31,True)
    txt(data['series'][0]['unit'],422,498,28,False,muted)
    line([(92,545),(140,545)],coral,3.3)
    txt(f'Historical · {int(times[0])}–{int(boundary)-1}',158,553,26,False,paper)
    dash([(650,545),(709,545)],forecast,3.3,9,5)
    txt(f'Forecast · {int(boundary)}–{int(times[-1])}',728,553,26,False,paper)
    line([(90,585),(1510,585)],(137,168,190,115),1)

    # Shared linear chart. Texture is decoration within this single series.
    rect((x(boundary),y0,x1,y1),(159,188,208,12))
    for v in [0,20,40,60,80,100,120,130]:
        py=y(v)
        line([(x0,py),(x1,py)],(137,176,203,72 if v!=100 else 112),1 if v!=100 else 1.25)
        txt(f'{v}%',124,py+8,24,False,muted,'rs')

    polygon=pts+[(x1,y1),(x0,y1)]
    mask=Image.new('L',(W*S,H*S),0)
    md=ImageDraw.Draw(mask)
    md.polygon([(round(px*S),round(py*S)) for px,py in polygon],fill=255)
    material=Image.open(ROOT/'assets/flag_surface.png').convert('RGB').resize((round((x1-x0)*S),round((y1-y0)*S)),Image.Resampling.LANCZOS)
    material_layer=Image.new('RGB',(W*S,H*S))
    material_layer.paste(material,(round(x0*S),round(y0*S)))
    im.paste(material_layer,(0,0),mask)
    draw=ImageDraw.Draw(im,'RGBA')

    # No smoothing, amplitude warping or minimum-height floor is applied.
    split=int(np.searchsorted(times,boundary))
    history=pts[:split]
    prediction=pts[split-1:]
    line(history,(6,24,40,235),6.0)
    line(history,coral,3.2)
    # The 2024→2025 bridge is included in the dashed forecast connection.
    dash(prediction,(6,24,40,235),5.5,8,5)
    dash(prediction,forecast,3.0,8,5)
    dash([(x(boundary),y0),(x(boundary),y1)],(187,207,219,200),1.4,5,7)
    txt('FORECAST',x1,614,21,True,muted,'rs',tracking=.8)
    txt(f'{int(boundary)}–{int(times[-1])}',x1,641,21,False,paper,'rs')
    line([(x0,y1),(x1,y1)],(196,210,219,220),1.5)
    for t in times:
        line([(x(t),y1),(x(t),y1+6)],(176,199,215,125),.7)
    tick_years=list(range(int(times[0]),int(times[-1]),10))
    if tick_years[-1]+7>times[-1]: tick_years=tick_years[:-1]
    tick_years += [int(times[-1])]
    for t in tick_years:
        line([(x(t),y1),(x(t),y1+12)],(193,212,226,230),1.0)
        anchor='ls' if t==times[0] else ('rs' if t==times[-1] else 'ms')
        txt(str(t),x(t),y1+46,25,False,paper,anchor)
    txt('YEAR',90,1813,19,True,muted,tracking=1.7)

    # Key values computed from the current series, never from the reference.
    max_i=int(np.argmax(vals)); max_t=int(times[max_i]); max_p=pts[max_i]
    txt(f'{max_t} · SERIES HIGH',195,658,22,True,muted,tracking=.6)
    txt(f'{vals[max_i]:.1f}%',194,713,53,True,coral)
    line([max_p,(172,715),(180,715)],(215,224,227,230),1.3)
    dot(*max_p,3.2)

    # Select the largest interior local maximum from the supplied values.
    local_max=[i for i in range(1,len(vals)-1) if vals[i]>vals[i-1] and vals[i]>=vals[i+1]]
    peak_i=max(local_max,key=lambda i:vals[i])
    px,py=pts[peak_i]
    txt(str(int(times[peak_i])),px,743,24,True,muted,'ms')
    txt(f'{vals[peak_i]:.1f}%',px,798,52,True,coral,'ms')
    line([(px,810),(px,py-6)],(215,224,227,220),1.3)
    dot(px,py,3.2)

    # Nearby boundary values are stated explicitly because the true marks are small.
    bt=int(boundary)
    txt(f'{bt-1} → {bt}',1122,1120,25,True,paper)
    txt(f'{lookup[bt-1][1]:.1f}% → {lookup[bt][1]:.1f}%',1122,1163,34,True,paper)
    txt('Historical → forecast',1122,1197,22,False,muted)

    min_i=int(np.argmin(vals)); min_p=pts[min_i]
    txt(f'{int(times[min_i])} · SERIES LOW',1333,1391,22,True,muted,'rs')
    txt(f'{vals[min_i]:.1f}%',1333,1448,52,True,coral,'rs')
    line([(min_p[0],1465),(min_p[0],min_p[1]-7)],(187,207,219,200),1.2)
    dot(*min_p,2.4)
    last_p=pts[-1]
    txt(f'{int(times[-1])}',1490,1504,24,True,muted,'rs')
    txt(f'{vals[-1]:.1f}%',1490,1560,51,True,forecast,'rs')
    line([(last_p[0],1577),(last_p[0],last_p[1]-7)],(208,220,221,210),1.2)
    dot(*last_p,2.8,forecast)

    # Clear topic and provenance without importing real-world statistical claims.
    line([(90,1833),(1510,1833)],(137,168,190,125),1)
    txt('Source: supplied synthetic series. Not observed economic statistics.',90,1873,26,False,paper)
    txt(f'All {len(vals)} annual values plotted. Flag edge = ratio; textile pattern carries no extra data.',90,1910,25,False,muted)
    txt('Linear time and value axes · 0–130% scale',90,1960,23,False,muted)
    txt('LOCAL RESEARCH STUDY',1510,1960,20,True,muted,'rs',tracking=1.2)

    im=im.resize((W,H),Image.Resampling.LANCZOS)
    im.save(ROOT/'final.png')
    # Diagnostic mask only, not another complete render.
    (ROOT/'artifacts').mkdir(exist_ok=True)
    mask.resize((W,H),Image.Resampling.LANCZOS).save(ROOT/'artifacts/area_mask.png')
    geometry={
        'canvas':[W,H], 'plot_bounds':[x0,y0,x1,y1],
        'x_mapping':{'first_year':float(times[0]),'last_year':float(times[-1]),'pixels_per_year':(x1-x0)/(times[-1]-times[0])},
        'y_mapping':{'minimum':lo,'maximum':hi,'pixels_per_percentage_point':(y1-y0)/(hi-lo)},
        'point_count':len(vals),'historical_points':split,'forecast_points':len(vals)-split,
        'forecast_boundary_year':bt,'forecast_boundary_x':x(boundary),
        'history_stroke':'solid','forecast_stroke':'dashed; includes 2024-to-2025 connector',
        'vertices':[{'year':int(t),'value':float(v),'x':px,'y':py} for t,v,(px,py) in zip(times,vals,pts)],
        'selected_labels':{'series_max_year':max_t,'interior_peak_year':int(times[peak_i]),'series_min_year':int(times[min_i]),'final_year':int(times[-1])},
        'frozen_inputs_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in [inputs['brief'],inputs['data'],inputs['critical_metaphor_contract'],inputs['reference']]}
    }
    (ROOT/'artifacts/data_geometry.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Rendered {W}×{H}; {len(vals)} points, {split} historical and {len(vals)-split} forecast.')


if __name__=='__main__':
    main()
