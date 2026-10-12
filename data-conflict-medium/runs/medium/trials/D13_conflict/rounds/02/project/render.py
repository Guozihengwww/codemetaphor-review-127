"""Deterministic ocean infographic. Run with the supplied capture_render.py.

Every layer uses the same x positions, linear interpolation and downward scale.
The generated water texture is warped to total depth, then recolored and masked.
Only the data geometry controls the underwater silhouette and layer boundaries.
"""
from pathlib import Path
from functools import lru_cache
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
S = 3
W, H = 2200, 1552
X0, X1, Y0, DEPTH = 190, 2110, 415, 700
INK = '#142D3E'
MUTED = '#607580'
PAPER = '#FFFFFF'
PALE = '#E1E9EC'
COLORS = ['#1AAF91', '#347CEE', '#293F6C', '#9558BA']
PALETTES = [
    [[3, 74, 66], [20, 171, 137], [165, 253, 215]],
    [[8, 44, 128], [34, 101, 224], [158, 213, 255]],
    [[9, 21, 50], [34, 54, 100], [105, 142, 186]],
    [[55, 24, 88], [126, 67, 164], [213, 163, 234]],
]


@lru_cache(maxsize=None)
def font(size, face='sans'):
    files = {'sans': 'NotoSansCJK-Regular.ttc', 'serif': 'NotoSerifCJK-Bold.ttc',
             'latin': 'DejaVuSans.ttf', 'bold': 'DejaVuSans-Bold.ttf',
             'mono': 'DejaVuSansMono.ttf'}
    return ImageFont.truetype(str(ROOT / 'assets/fonts' / files[face]), round(size*S), index=2 if face in ('sans', 'serif') else 0)


def main():
    data = json.loads((ROOT / 'data.json').read_text(encoding='utf-8'))
    brief = json.loads((ROOT / 'brief.json').read_text(encoding='utf-8'))
    assert (brief['width'], brief['height']) == (W, H)
    vals = np.asarray([s['values'] for s in data['series']], dtype=float)
    cats = data['categories']
    assert vals.shape == (4, 16) and len(cats) == 16 and np.all(vals >= 0)
    assert data['value_axis'] == {'minimum': 0, 'maximum': 7200, 'scale': 'linear'}
    vmax = float(data['value_axis']['maximum'])
    xs = np.linspace(X0, X1, len(cats))
    cum = np.vstack([np.zeros(len(cats)), np.cumsum(vals, axis=0)])
    totals = cum[-1]
    scale = DEPTH/vmax
    ys = Y0 + cum*scale
    im = Image.new('RGB', (W*S, H*S), PAPER)
    d = ImageDraw.Draw(im)
    text_boxes = []

    def text(x, y, value, size=24, fill=INK, face='sans', anchor='lt', stroke=0):
        value = str(value)
        d.text((round(x*S), round(y*S)), value, font=font(size, face), fill=fill,
               anchor=anchor, stroke_width=round(stroke*S), stroke_fill='#164154')
        box = d.textbbox((round(x*S), round(y*S)), value, font=font(size, face), anchor=anchor,
                         stroke_width=round(stroke*S))
        text_boxes.append({'text': value, 'box_px': [round(v/S, 2) for v in box]})

    def line(points, fill=PALE, width=1):
        d.line([(round(x*S), round(y*S)) for x,y in points], fill=fill, width=max(1,round(width*S)))

    def rect(box, fill, outline=None, width=1, radius=0):
        box = tuple(round(v*S) for v in box)
        if radius:
            d.rounded_rectangle(box, radius=round(radius*S), fill=fill, outline=outline, width=round(width*S))
        else:
            d.rectangle(box, fill=fill, outline=outline, width=round(width*S))

    def dot(x, y, radius=3.5, fill=INK, outline='white', width=1):
        d.ellipse(((x-radius)*S,(y-radius)*S,(x+radius)*S,(y+radius)*S), fill=fill, outline=outline, width=round(width*S))

    # Editorial heading; no inherited claims from the reference.
    text(92, 43, 'O C E A N   P R O F I L E    /    1 6   P E R I O D S', 18, MUTED, 'latin')
    text(88, 78, data['title'], 62, INK, 'serif')
    text(94, 170, '四个来源，四层海水；每一层的厚度对应当期分项值。', 25, MUTED)
    rect((1712, 49, 2108, 94), '#E8F4EF', radius=6)
    text(1910, 72, data['visible_notice'], 23, '#176A57', 'bold', 'mm')
    text(2108, 123, '100 tonnes', 29, INK, 'bold', 'rt')
    text(2108, 166, 'per synthetic observation period', 18, MUTED, 'latin', 'rt')
    line([(92, 213), (2108, 213)], '#D5E1E4', 1)
    text(190, 227, '有序观测期', 18, MUTED)
    text(2110, 228, 'P01 → P16  /  原始顺序', 18, MUTED, 'sans', 'rt')

    # Axis geometry: 0 is the common waterline; increasing output goes down.
    for tick in range(0, 7201, 1200):
        y = Y0 + tick*scale
        line([(X0, y), (X1, y)], '#E0E8EB', .8)
        line([(X0-8, y), (X0, y)], '#879BA5', .8)
        text(X0-19, y, f'{tick:,}', 20, MUTED, 'latin', 'rm')
    line([(X0, Y0), (X0, Y0+DEPTH)], '#C6D4DA', 1)
    # The axis title is rasterized separately only as typography, not a preview.
    axis_text = '累计量  /  Cumulative output (100 tonnes)'
    af = font(20)
    box = af.getbbox(axis_text)
    label = Image.new('RGBA', (box[2]+12*S, (box[3]-box[1])+10*S))
    ImageDraw.Draw(label).text((6*S, 5*S-box[1]), axis_text, font=af, fill=MUTED)
    label = label.rotate(90, expand=True)
    im.paste(label, (round(43*S), round((Y0+DEPTH/2)*S-label.height/2)), label)
    for i, x in enumerate(xs):
        text(x, 272, cats[i], 23, INK, 'mono', 'mm')
        line([(x, 294), (x, Y0-16)], '#CEDADD', .8)

    # Warped generated water. This is the entire quantitative object, not a backdrop.
    pw, ph = (X1-X0)*S+1, DEPTH*S+1
    colx = X0 + np.arange(pw)/S
    bounds = np.vstack([np.interp(colx, xs, y) for y in ys])
    depths = bounds[-1]-Y0
    yy = np.arange(ph, dtype=np.float32)[:,None]/S
    source = np.asarray(Image.open(ROOT/'assets/water_material.png').convert('RGB'), dtype=np.float32)/255
    # One common spatial mapping across all four colors.
    sx = np.linspace(0, source.shape[1]-1, pw).astype(np.int32)
    sy = np.clip(yy/np.maximum(depths[None,:], .00001)*(source.shape[0]-1), 0, source.shape[0]-1).astype(np.int32)
    # Luminance retains caustics and shafts in the generated raster.
    lum_src = source[:,:,0]*.2126+source[:,:,1]*.7152+source[:,:,2]*.0722
    lum = lum_src[sy, sx[None,:]]
    lo, hi = np.percentile(lum_src, [1, 99])
    light = np.clip((lum-lo)/(hi-lo), 0, 1)**.88
    layer_img = np.zeros((ph,pw,3), dtype=np.uint8)
    owner = np.zeros((ph,pw), dtype=np.uint8)
    for k in range(4):
        mask = (Y0+yy >= bounds[k][None,:]) & (Y0+yy < bounds[k+1][None,:])
        palette = np.asarray(PALETTES[k], dtype=np.float32)
        t = light[:,:,None]
        low = palette[0]+np.minimum(t*2,1)*(palette[1]-palette[0])
        high = palette[1]+np.maximum((t-.5)*2,0)*(palette[2]-palette[1])
        mapped = np.where(t <= .5, low, high)
        layer_img[mask] = mapped[mask].astype('uint8')
        owner[mask] = k+1
    alpha = Image.fromarray((owner>0).astype('uint8')*255)
    water = Image.fromarray(layer_img)
    im.paste(water, (X0*S,Y0*S), alpha)
    del source, sy, light, lum, layer_img, water

    # Low contrast guides inside the water aid comparison without changing boundaries.
    guide = Image.new('RGBA', im.size)
    gd = ImageDraw.Draw(guide)
    for tick in range(1200,7201,1200):
        y = (Y0+tick*scale)*S
        gd.line([(X0*S,y),(X1*S,y)], fill=(235,250,255,42), width=2)
    for i,x in enumerate(xs):
        gd.line([(x*S,Y0*S),(x*S,ys[-1,i]*S)], fill=(242,253,255,47), width=2)
    # Restrict the guide overlay to the exact water footprint.
    guide_crop = guide.crop((X0*S,Y0*S,X0*S+pw,Y0*S+ph))
    ga = np.asarray(guide_crop.getchannel('A')).copy()
    ga[owner==0] = 0
    guide_crop.putalpha(Image.fromarray(ga))
    im.paste(guide_crop, (X0*S,Y0*S), guide_crop)
    d = ImageDraw.Draw(im)
    # Thin boundaries. They are centered on cumulative sums, with no added thickness.
    for k in range(1,5):
        pts = list(zip(xs,ys[k]))
        line(pts, ['#61DEC6','#8CB7F6','#516991','#724296'][k-1], .55 if k<4 else .8)
    # Small generated wave crests sit above 0. All measurements start at flat y=Y0.
    crest = Image.open(ROOT/'assets/surface_material.png').convert('RGBA').resize(((X1-X0)*S,14*S), Image.Resampling.LANCZOS)
    ca = np.asarray(crest).copy()
    cl = (ca[:,:,0]*.2126+ca[:,:,1]*.7152+ca[:,:,2]*.0722)/255
    ca[:,:,:3] = np.clip(np.array([10,125,110])[None,None,:]+cl[:,:,None]*np.array([116,125,102])[None,None,:],0,255).astype('uint8')
    crest = Image.fromarray(ca)
    im.paste(crest, (X0*S,(Y0-14)*S), crest)
    d = ImageDraw.Draw(im)
    line([(X0,Y0),(X1,Y0)], '#43B89E', .8)
    text(106, Y0-42, '水面', 18, MUTED)
    yacht = Image.open(ROOT/'assets/yacht.png').convert('RGBA')
    yacht_w = 360
    yacht_h = yacht_w*yacht.height/yacht.width
    yacht = yacht.resize((round(yacht_w*S),round(yacht_h*S)),Image.Resampling.LANCZOS)
    im.paste(yacht,(round((X1-yacht_w-4)*S),round((Y0-yacht_h-3)*S)),yacht)
    d = ImageDraw.Draw(im)

    # Series labels are inside the same texture regions as their data.
    label_x = 564
    for k, series in enumerate(data['series']):
        top = np.interp(label_x,xs,ys[k])
        bottom = np.interp(label_x,xs,ys[k+1])
        text(label_x,(top+bottom)/2,series['label'],22 if k!=2 else 28,'#F4FBFE','bold','mm',.35)

    # Total markers use the bottom boundary, never a fifth band.
    for idx in (7,11):
        x, y = xs[idx], ys[-1,idx]
        dot(x,y,3.8,'#9256B4','white',1.2)
        if idx == 11:
            line([(x,y+5),(x,y+33),(x+33,y+33)],'#A9B6BD',1)
            text(x+45,y+22,f'{cats[idx]} · {int(totals[idx]):,}',22,INK,'latin','lt')
        else:
            line([(x,y+5),(x,y+19)],'#A9B6BD',1)
            text(x,y+27,f'{cats[idx]} · {int(totals[idx]):,}',22,INK,'latin','mt')
    peak = int(np.argmax(totals))
    px, py = xs[peak],ys[-1,peak]
    dot(px,py,4.5,'#9256B4','white',1.2)
    line([(px,py+5),(px,1137),(472,1137)],'#9BAEB7',1.1)
    text(488,1121,f'{cats[peak]} · 最大合计 {int(totals[peak]):,}',24,INK,'sans')
    x,y = xs[-1],ys[-1,-1]
    dot(x,y,3.8,'#9256B4','white',1.1)
    line([(x,y+4),(x,492),(1958,492)],'#9BAEB7',1.1)
    text(1836,516,f'{cats[-1]} · 合计',22,MUTED)
    text(1834,552,f'{int(totals[-1]):,}',57,INK,'bold')
    text(1837,623,'100 tonnes',18,MUTED,'latin')

    # Reading key in the natural negative space below the shallowing water.
    rect((1560,792,2110,1090),'#FFFFFF')
    text(1584,806,'四层来源',28,INK,'serif')
    text(1584,850,'层厚 = 分项值',21,MUTED)
    for k,s in enumerate(data['series']):
        lx = 1584 + (k%2)*265
        ly = 900 + (k//2)*49
        rect((lx,ly,lx+27,ly+23),COLORS[k],radius=3)
        text(lx+42,ly-1,s['label'],23,INK,'latin')
    line([(1584,1002),(2085,1002)],'#DAE4E7',1)
    text(1584,1017,'底边 = A + B + C + D；累计量向下增大',21,MUTED)
    text(1584,1053,'量尺表示累计量，不表示实际海洋深度。',18,MUTED)

    # Three complete number tables, all labels and values sourced from data.json.
    text(92,1180,'完整数值',27,INK,'serif')
    text(236,1187,'/  ALL OBSERVATIONS',17,MUTED,'latin')
    text(2108,1189,data['series'][0]['unit']['values'],18,MUTED,'latin','rt')
    groups = [(0,6,92,730),(6,12,850,730),(12,16,1608,500)]
    value_cells = []
    for start,end,left,width in groups:
        top,bottom = 1225,1470
        rect((left,top,left+width,bottom),'#F5F8F8',radius=6)
        label_w = 110
        colw = (width-label_w)/(end-start)
        text(left+17,top+15,'来源',19,MUTED)
        for j,i in enumerate(range(start,end)):
            cx = left+label_w+(j+.5)*colw
            text(cx,top+24,cats[i],21,INK,'mono','mm')
        line([(left+14,top+49),(left+width-14,top+49)],'#D3DFE3',1)
        for k,s in enumerate(data['series']):
            cy = top+73+k*37
            rect((left+14,cy-7,left+23,cy+7),COLORS[k],radius=2)
            text(left+33,cy,s['label'],17,INK,'latin','lm')
            for j,i in enumerate(range(start,end)):
                cx = left+label_w+(j+.5)*colw
                value = s['values'][i]
                text(cx,cy,f'{value:,}',22,INK,'mono','mm')
                value_cells.append({'category':cats[i],'series':s['id'],'value':value,'center_px':[cx,cy]})
        line([(left+14,top+208),(left+width-14,top+208)],'#D3DFE3',1)
        text(left+16,top+228,'合计',19,INK,'sans','lm')
        for j,i in enumerate(range(start,end)):
            cx = left+label_w+(j+.5)*colw
            text(cx,top+228,f'{int(totals[i]):,}',22,INK,'bold','mm')
    text(92,1501,'分项读层厚，合计读底边。相邻观测期线性连接；船体与波纹不计入尺度。',18,MUTED)
    text(2108,1501,'本地合成数据 · 非真实统计',18,MUTED,'sans','rt')

    # Measure the actual supersampled material ownership, not only intended values.
    observations = []
    for i,x in enumerate(xs):
        col = int(round((x-X0)*S))
        for k in range(4):
            pix = np.flatnonzero(owner[:,col]==k+1)
            visible_px = len(pix)/S
            expected_px = float(vals[k,i]*scale)
            observations.append({'category':cats[i], 'series':data['series'][k]['id'],
                                 'expected_thickness_px':round(expected_px,5),
                                 'raster_thickness_px':round(visible_px,5),
                                 'error_px':round(visible_px-expected_px,5),
                                 'first_raster_y_px':round(Y0+int(pix[0])/S,5) if len(pix) else None,
                                 'last_raster_y_px':round(Y0+int(pix[-1])/S,5) if len(pix) else None})
    geometry = {'canvas':[W,H], 'plot':[X0,Y0,X1,Y0+DEPTH], 'supersampling':S,
                'x_positions_px':xs.tolist(), 'scale_px_per_unit':scale,
                'interpolation':'piecewise linear; same mapping for every layer',
                'material_covered_pixels_at_output_scale':int(np.count_nonzero(owner))/S**2,
                'measurements':observations,
                'max_abs_thickness_error_px':max(abs(m['error_px']) for m in observations),
                'minimum_raster_thickness_px':min(m['raster_thickness_px'] for m in observations),
                'source_values_cells':value_cells,'totals':totals.astype(int).tolist(),
                'text_boxes':text_boxes}
    (ROOT/'artifacts').mkdir(exist_ok=True)
    (ROOT/'artifacts/geometry_measurements.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    im = im.resize((W,H),Image.Resampling.LANCZOS)
    im.save(ROOT/'final.png')
    print(json.dumps({'canvas':[W,H],'measured_max_abs_error_px':geometry['max_abs_thickness_error_px'],
                      'minimum_thickness_px':geometry['minimum_raster_thickness_px'],'numeric_cells':len(value_cells)}))


if __name__ == '__main__':
    main()
