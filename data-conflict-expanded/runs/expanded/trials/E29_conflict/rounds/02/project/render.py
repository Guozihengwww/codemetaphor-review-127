#!/usr/bin/env python3
"""Offline infographic compositor. Run only through the frozen capture wrapper.

The generated hand supplies all hand imagery. A row-wise texture warp makes the
horizontal silhouette midpoint follow a monotone cubic interpolation of the
supplied birth-cohort rates. Width is a constant, non-data visual parameter.
"""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parent
S = 2
INK = '#342b30'
PLUM = '#743a64'
MUTED = '#6d706b'
PAPER = '#faf8f0'
GRID = '#e5e7df'
X0, X1 = 286.0, 1322.0
YT, YB = 590.0, 1480.0
HALF = 62.0
SOURCE_TOP, SOURCE_BOTTOM = 245.0, 810.0
TIP_Y, PALM_BOTTOM = 425.0, 1932.0


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def save_json(path, value):
    (ROOT / path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def font(name, size):
    suffix = '.otf' if name.startswith('serif') else '.ttf'
    return ImageFont.truetype(str(ROOT / 'assets' / 'fonts' / (name + suffix)), round(size * S))


def main():
    inputs = read_json('input.json')
    brief = read_json(inputs['brief'])
    data = read_json(inputs['data'])
    core = read_json(inputs['critical_metaphor_contract'])
    assert (ROOT / inputs['reference']).is_file()
    assert core['semantic_roles_zh'] == ['Synthetic left-handedness rate (percent)']
    assert len(data['series']) == 1 and data['series'][0]['unit'] == 'percent'
    assert data['value_axis'] == {'minimum': 0, 'maximum': 14, 'scale': 'linear'}
    width, height = brief['width'], brief['height']
    assert (width, height) == (1560, 2000)
    years = np.asarray(data['time_points'], dtype=float)
    values = np.asarray(data['series'][0]['values'], dtype=float)
    assert len(years) == len(values) == len(data['categories'])
    assert np.all(np.diff(years) > 0)
    value_scale = (X1 - X0) / 14
    year_scale = (YB - YT) / (years[-1] - years[0])
    interp = PchipInterpolator(years, values, extrapolate=False)

    def x_for_y(y):
        y = np.asarray(y)
        t = years[0] + (YB - y) / year_scale
        return X0 + interp(np.clip(t, years[0], years[-1])) * value_scale

    def y_for_year(t):
        return YB - (t - years[0]) * year_scale

    im = Image.new('RGBA', (width * S, height * S), '#e8e7e2')
    shadow = Image.new('RGBA', im.size)
    ImageDraw.Draw(shadow).rectangle((46*S, 54*S, 1521*S, 1961*S), fill=(57, 52, 42, 30))
    im.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(10*S)))
    d = ImageDraw.Draw(im)

    def line(points, fill, w=1):
        d.line([(round(x*S), round(y*S)) for x,y in points], fill=fill, width=max(1,round(w*S)), joint='curve')

    def rect(box, fill, outline=None, w=1):
        d.rectangle(tuple(round(v*S) for v in box), fill=fill, outline=outline, width=round(w*S))

    def text(x, y, label, size=26, family='sans', fill=INK, anchor='lt'):
        d.text((round(x*S),round(y*S)), str(label), font=font(family,size), fill=fill, anchor=anchor)

    def tracked(x, y, label, size=18, spacing=2, fill=INK, family='sans_bold'):
        f = font(family,size)
        for ch in label:
            d.text((round(x*S),round(y*S)),ch,font=f,fill=fill,anchor='lt')
            x += d.textlength(ch,font=f)/S + spacing

    rect((45,45,1515,1955), PAPER, '#d6d8d0', .8)
    # The square paper grid is decorative; chart guides below carry the axes.
    for x in range(75,1487,20):
        line([(x,350),(x,1929)], '#d8ddd4' if (x-75)%100==0 else GRID, .75 if (x-75)%100==0 else .45)
    for y in range(350,1930,20):
        line([(75,y),(1485,y)], '#d8ddd4' if (y-350)%100==0 else GRID, .75 if (y-350)%100==0 else .45)

    # Editorial heading and explicit synthetic-data notice.
    d.rounded_rectangle((110*S,98*S,386*S,140*S), radius=5*S, fill=PLUM)
    tracked(128,109,data['visible_notice'],18,1.7,'#fff9f0')
    text(108,157,'Left-handedness',88,'serif')
    text(112,249,'by birth year',64,'serif_italic')
    text(1378,171,f"{data['categories'][0]}–{data['categories'][-1]}",29,'sans',anchor='rt')
    text(1378,216,'BIRTH-YEAR COHORTS',16,'sans_bold',MUTED,'rt')
    text(1378,257,'Synthetic illustration',21,'sans',MUTED,'rt')
    tracked(X0,353,'LEFT-HANDEDNESS RATE (%)',20,1.1,INK)

    # Shared 0–14 percent axis, above the non-data fingertip.
    for v in range(0,15,2):
        x = X0 + v*value_scale
        text(x,387,f'{v}%',24,'sans',anchor='mt')
        line([(x,420),(x,YB)], '#ccd3ca', .8)
        line([(x,415),(x,427)], INK,1.4)
    line([(X0,420),(X1,420)],INK,1.5)
    line([(X0,420),(X0,YB)],INK,1.5)
    for year,label in zip(years,data['categories']):
        y = float(y_for_year(year))
        line([(X0,y),(X1,y)],'#bac4b9',1.0)
        line([(X0-7,y),(X0+7,y)],INK,1.5)
        text(X0-25,y,label,29,'sans',anchor='rm')

    # Rotated semantic category-axis title.
    yl = Image.new('RGBA',(760,90))
    yd = ImageDraw.Draw(yl)
    yd.text((380,45),'BIRTH YEAR COHORT',font=font('sans_bold',19),fill=INK,anchor='mm')
    yl = yl.rotate(90,expand=True)
    im.alpha_composite(yl,(round(114*S),round((1035-yl.height/(2*S))*S)))

    # Extract actual generated silhouette spans; no reference pixels are read.
    source = np.array(Image.open(ROOT/'assets/hand_clean.png').convert('RGBA'))
    alpha = source[:,:,3]
    rows = np.arange(source.shape[0])
    left = np.zeros(len(rows),float)
    right = np.zeros(len(rows),float)
    for r in rows:
        xs = np.flatnonzero(alpha[r] >= 128)
        if len(xs):
            left[r],right[r] = xs[0],xs[-1]
        else:
            left[r],right[r] = 500,514

    def sample_warp(xs, ys, sx, sy, new_alpha=None):
        sx,sy = np.broadcast_arrays(sx,sy)
        coords = np.array([sy,sx],dtype=np.float32)
        out = np.empty(sx.shape+(4,),dtype=np.uint8)
        for ch in range(4):
            out[:,:,ch] = map_coordinates(source[:,:,ch],coords,order=1,mode='constant',cval=0,prefilter=False)
        if new_alpha is not None:
            out[:,:,3] = new_alpha
        layer = Image.fromarray(out,'RGBA')
        im.alpha_composite(layer,(int(round(xs[0]*S)),int(round(ys[0]*S))))
        return out[:,:,3]

    # The palm begins strictly below the oldest data section.
    bottom_x = float(X0+values[0]*value_scale)
    sl,sr = left[int(SOURCE_BOTTOM)],right[int(SOURCE_BOTTOM)]
    palm_sx = (2*HALF)/(sr-sl)
    palm_sy = (PALM_BOTTOM-YB)/(1488-SOURCE_BOTTOM)
    xs = np.arange(940*S,1500*S)/S
    ys = np.arange(round(YB*S),round(PALM_BOTTOM*S)+1)/S
    sx = (sl+sr)/2 + (xs[None,:]-bottom_x)/palm_sx
    sy = SOURCE_BOTTOM + (ys[:,None]-YB)/palm_sy
    sample_warp(xs,ys,sx,sy)

    # Continuous finger shaft: horizontal bounds exactly c(y) ± HALF.
    xs = np.arange(480*S,1305*S)/S
    ys = np.arange(round(YT*S),round(YB*S)+1)/S
    center = x_for_y(ys)
    sy_vec = SOURCE_TOP + (ys-YT)/(YB-YT)*(SOURCE_BOTTOM-SOURCE_TOP)
    src_l = np.interp(sy_vec,rows,left)
    src_r = np.interp(sy_vec,rows,right)
    u = (xs[None,:]-center[:,None]+HALF)/(2*HALF)
    sx = src_l[:,None] + u*(src_r-src_l)[:,None]
    sy = sy_vec[:,None]
    coverage = np.minimum(xs[None,:]-(center[:,None]-HALF)+.25,
                          (center[:,None]+HALF)-xs[None,:]+.25)
    mask = np.clip(coverage*S,0,1)*255
    sample_warp(xs,ys,sx,sy,mask.astype(np.uint8))

    # Preserve the generated rounded cap and nail outside the observed interval.
    xs = np.arange(550*S,925*S)/S
    ys = np.arange(round(TIP_Y*S),round(YT*S))/S
    top_x = float(X0+values[-1]*value_scale)
    derivative_y = -float(interp.derivative()(years[-1]))*value_scale/year_scale
    dist = YT-ys
    cap_center = top_x - derivative_y*(dist-dist**2/(2*(YT-TIP_Y)))
    sl,sr = left[int(SOURCE_TOP)],right[int(SOURCE_TOP)]
    sx = (sl+sr)/2 + (xs[None,:]-cap_center[:,None])*(sr-sl)/(2*HALF)
    sy = SOURCE_TOP-(YT-ys[:,None])
    sample_warp(xs,ys,sx,sy)
    d = ImageDraw.Draw(im)

    # Reinforce the two precisely controlled outside contours, without a shadow.
    yy = np.linspace(YT,YB,1781)
    cc = x_for_y(yy)
    for sign in (-1,1):
        line(list(zip(cc+sign*HALF,yy)), '#573b2c', 1.6)

    # A subtle dashed centreline makes the nonstandard mapping discoverable.
    for ya in np.arange(YT,YB,17):
        yseg = np.linspace(ya,min(ya+7,YB),12)
        line(list(zip(x_for_y(yseg),yseg)),(108,61,91,140),1.3)

    geometry = []
    for year,label,value in zip(years,data['categories'],values):
        x = float(X0+value*value_scale)
        y = float(y_for_year(year))
        # Horizontal section, small midpoint marker, and an external value label.
        line([(x-HALF,y),(x+HALF,y)],(105,54,85,185),1.1)
        for end in [x-HALF,x+HALF]:
            line([(end,y-4),(end,y+4)],PLUM,1.3)
        d.ellipse(((x-5.2)*S,(y-5.2)*S,(x+5.2)*S,(y+5.2)*S),fill=PAPER,outline=PLUM,width=2*S)
        label_x=x+HALF+41
        number=f'{value:.1f}%'
        f=font('sans_bold',30)
        bbox=d.textbbox((round(label_x*S),round(y*S)),number,font=f,anchor='lm')
        d.rounded_rectangle((bbox[0]-7*S,bbox[1]-5*S,bbox[2]+7*S,bbox[3]+5*S),radius=3*S,fill=PAPER)
        line([(x+HALF+5,y),(label_x-13,y)],PLUM,1.2)
        text(label_x,y,number,30,'sans_bold',PLUM,'lm')
        geometry.append({'birth_year_cohort':label,'rate_percent':float(value),'y_px':y,'center_x_px':x,
                         'left_x_px':x-HALF,'right_x_px':x+HALF,
                         'label':number,'label_bbox_px':[round(v/S,2) for v in bbox]})

    # The reading key occupies the space left of the generated palm.
    rect((104,1613,928,1924),PAPER)
    line([(114,1621),(223,1621)],PLUM,3)
    text(114,1650,'How to read the hand',39,'serif')
    line([(119,1730),(244,1730)],PLUM,1.6)
    for x in [119,244]:
        line([(x,1722),(x,1738)],PLUM,1.6)
    d.ellipse((176*S,1725*S,186*S,1735*S),fill=PAPER,outline=PLUM,width=2*S)
    text(271,1730,'Finger midpoint = cohort rate',25,'sans',INK,'lm')
    text(114,1768,'Years identify birth cohorts.',23,'sans')
    text(114,1803,'Width and nail size are decorative.',23,'sans')
    text(114,1838,'Tip and palm lie outside the cohort range.',22,'sans')
    text(114,1874,'Synthetic values for illustration only.',22,'sans_bold',PLUM)
    text(114,1911,f'DATA  local synthetic dataset · {len(years)} cohorts · percent',17,'sans',MUTED)

    (ROOT/'artifacts').mkdir(exist_ok=True)
    # Numerical checks include dense interpolation, not just node equality.
    intervals=[]
    for a,b,va,vb in zip(years[:-1],years[1:],values[:-1],values[1:]):
        v=interp(np.linspace(a,b,1001))
        intervals.append({'birth_year_interval':[float(a),float(b)],'minimum':float(v.min()),'maximum':float(v.max()),
                          'within_neighbor_values':bool(v.min()>=min(va,vb)-1e-10 and v.max()<=max(va,vb)+1e-10)})
    save_json('artifacts/geometry.json',{'canvas':[width,height],'value_axis_x_px':[X0,X1],
        'value_axis_range_percent':[0,14],'years_bottom_to_top':[float(years[0]),float(years[-1])],
        'observation_y_px':[YT,YB],'horizontal_finger_width_px':2*HALF,'nodes':geometry,
        'interpolation':'PCHIP, no adjacent-value overshoot','interval_checks':intervals,
        'palm_begins_y_px':YB,'rounded_tip_y_range_px':[TIP_Y,YT],
        'generated_source':'assets/raw_01.png','processed_source':'assets/hand_clean.png'})
    im=im.convert('RGB').resize((width,height),Image.Resampling.LANCZOS)
    im.save(ROOT/'final.png',dpi=(160,160))
    print('Saved final.png at 1560 × 2000; seven cohort sections; offline generated-material warp.')


if __name__ == '__main__':
    main()
