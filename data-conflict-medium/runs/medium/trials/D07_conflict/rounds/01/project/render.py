"""Render the frozen data as calibrated visible-area bands on a photographic shirt.

Run from this directory with the supplied Python interpreter. All inputs are local.
For the experiment, invoke this script only through capture_render.py.
"""
from pathlib import Path
import json
from decimal import Decimal
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT/'data.json').read_text(encoding='utf-8'))
BRIEF = json.loads((ROOT/'brief.json').read_text(encoding='utf-8'))
W,H = BRIEF['width'],BRIEF['height']
SS = 2
PALETTE = ['#526b91','#dbd4bf','#bc7764','#849aae','#879b8e','#bc9e69',
           '#495564','#c5999c','#adc0c0','#a7694e','#818970','#a8493d']
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONT_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
FONT_CJK = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONT_TITLE = '/usr/share/fonts/opentype/noto/NotoSerifCJK-Medium.ttc'
OUT = ROOT/'artifacts'
OUT.mkdir(exist_ok=True)

def hexrgb(h):
    return np.array([int(h[i:i+2],16) for i in (1,3,5)],dtype=np.float32)

def font(sz,bold=False,cjk=False,title=False):
    path=FONT_TITLE if title else FONT_CJK if cjk else FONT_BOLD if bold else FONT
    return ImageFont.truetype(path,int(round(sz*SS)))

def text(draw,xy,s,size,fill,**kw):
    f=font(size,kw.pop('bold',False),kw.pop('cjk',False),kw.pop('title',False))
    draw.text(tuple(int(v*SS) for v in xy),s,font=f,fill=fill,**kw)

def line(draw,xy,fill,width=1):
    draw.line(tuple(v*SS for v in xy),fill=fill,width=max(1,round(width*SS)))

def weighted_quantile(values,weights,probs):
    ids=np.argsort(values,kind='stable')
    vals=values[ids]; ws=weights[ids]
    acc=np.cumsum(ws,dtype=np.float64)
    indexes=np.searchsorted(acc,np.array(probs)*acc[-1],side='left')
    return vals[np.clip(indexes,0,len(vals)-1)]

def main():
    cats=DATA['categories']; vals=DATA['series'][0]['values']
    assert len(cats)==len(vals)==len(PALETTE)
    assert sum(Decimal(str(v)) for v in vals)==Decimal(100)
    assert DATA['value_axis']=={'minimum':0,'maximum':100,'scale':'linear'}
    # One uniform transform for the photograph, fabric mask and warp domain.
    left,top=356,304
    garment_w=1030
    src=Image.open(ROOT/'assets/garment_neutral.png').convert('RGBA')
    mask=Image.open(ROOT/'assets/fabric_mask.png').convert('L')
    garment_h=round(garment_w*src.height/src.width)
    gw,gh=garment_w*SS,garment_h*SS
    photo=np.asarray(src.resize((gw,gh),Image.Resampling.LANCZOS)).astype(np.float32)
    fabric=np.asarray(mask.resize((gw,gh),Image.Resampling.LANCZOS)).astype(np.float32)/255
    alpha=photo[:,:,3]/255
    fabric=np.minimum(fabric,alpha)
    yy,xx=np.mgrid[:gh,:gw].astype(np.float32)
    x=178+(xx+.5)*src.width/gw
    y=9+(yy+.5)*src.height/gh
    seam_l=288+.129*(y-320)
    seam_r=923-67*np.sin(np.pi*np.clip((y-325)/840,0,1))
    sleeve_l=1/(1+np.exp(np.clip((x-seam_l)/3.0,-30,30)))
    sleeve_r=1/(1+np.exp(np.clip((seam_r-x)/3.0,-30,30)))
    # Fold-aware material coordinate. Each horizontal level is monotone in y;
    # the same field controls every category, seam and textile pattern.
    lum=photo[:,:,:3] @ np.array([.2126,.7152,.0722],dtype=np.float32)
    smooth=gaussian_filter(lum,8*SS)
    warp=(10*np.sin((x-350)/104)+4*np.sin(x/29)
          +5*np.sin((x-580)/66)*np.clip((y-240)/950,0,1)
          +.045*(smooth-155)
          +sleeve_l*(16+.012*(y-650)+7*np.sin(x/38))
          +sleeve_r*(-12+.014*(y-650)))
    v=y+warp
    u=x+7*np.sin(x/46)+.022*(y-650)*np.sin(x/82)
    valid=fabric>.002
    cumulative=np.cumsum(vals,dtype=np.float64)[:-1]/100
    thresholds=weighted_quantile(v[valid],fabric[valid],cumulative)
    # 0.7 final pixel antialiasing at seams. The region masks used below are
    # retained and measured after reduction to the exact final output grid.
    aa=.65*src.height/garment_h
    prev=np.zeros((gh,gw),dtype=np.float32)
    color=np.zeros((gh,gw,3),dtype=np.float32)
    area_records=[]; label_positions=[]
    full_S=Image.new('L',(W,H),0)
    reduced_S=Image.fromarray(np.uint8(np.clip(fabric*255,0,255))).resize((garment_w,garment_h),Image.Resampling.LANCZOS)
    full_S.paste(reduced_S,(left,top)); full_S.save(OUT/'visible_fabric_mask.png')
    total_area=np.asarray(reduced_S,dtype=np.float64).sum()/255
    category_ids=np.zeros((garment_h,garment_w),dtype=np.uint8)
    max_coverage=np.zeros((garment_h,garment_w),dtype=np.float32)
    tiny_masks=[]
    for i,(cat,value,base) in enumerate(zip(cats,vals,PALETTE)):
        after=(np.clip((v-thresholds[i])/aa+.5,0,1) if i<len(thresholds)
               else np.ones_like(v))
        # CDF under a boundary: top category occupies the smallest material v.
        cdf=(1-after) if i<len(thresholds) else np.ones_like(v)
        if i==len(thresholds):
            coverage=1-prev
        else:
            coverage=cdf-prev
            prev=cdf
        coverage=np.maximum(coverage,0)
        base_rgb=hexrgb(base)
        textile=np.zeros((gh,gw),dtype=np.float32)
        phase=np.mod(u,7.3)
        if i==0:
            textile=np.clip(1-np.abs(phase-1.1)/.58,0,1)*.21-.022
        elif i==2:
            textile=(np.mod(u,11)<2.0).astype(np.float32)*.30-.038
        elif i==3:
            textile=(np.mod(u,5.5)<.65).astype(np.float32)*.18-.020
        elif i==6:
            textile=((np.mod(u,22)<1.2)|(np.mod(v,23)<1.0)).astype(np.float32)*.18
            textile-=((np.mod(u+8,22)<2)|(np.mod(v+7,23)<1.6)).astype(np.float32)*.08
        elif i==7:
            textile=(np.mod(u,8.0)<.8).astype(np.float32)*.12-.011
        elif i==9:
            textile=.018*np.sin((u+2*np.abs(np.mod(v,5)-2.5))*3)
        # Source weave, wrinkles, vertical seams, buttons, cuffs and hem remain.
        shade=np.clip(lum/167,.28,1.42)
        rgb=base_rgb[None,None,:]*shade[:,:,None]
        rgb=rgb+(245-rgb)*np.maximum(textile,0)[:,:,None]+rgb*np.minimum(textile,0)[:,:,None]
        color+=rgb*coverage[:,:,None]
        visible=coverage*fabric
        small=Image.fromarray(np.uint8(np.rint(visible*255))).resize((garment_w,garment_h),Image.Resampling.LANCZOS)
        small_arr=np.asarray(small,dtype=np.float64)/255
        small.save(OUT/f'band_{i+1:02d}_mask.png')
        actual_area=float(small_arr.sum())
        cy=int(np.searchsorted(np.cumsum(small_arr.sum(axis=1)),actual_area*.5))
        row=small_arr[cy]
        xx_visible=np.where(row>.72)[0]
        assert len(xx_visible),f'No visible label attachment: {cat}'
        ax=int(xx_visible[0]+min(22,len(xx_visible)//10))
        assert row[ax]>.45
        label_positions.append((left+ax,top+cy))
        category_ids[small_arr>max_coverage]=i+1
        max_coverage=np.maximum(max_coverage,small_arr)
        area_records.append({'category':cat,'label':str(value)+'%','target_percent':value,
            'final_visible_area_px':round(actual_area,4),
            'measured_percent':round(100*actual_area/total_area,6),
            'absolute_error_pp':round(abs(100*actual_area/total_area-value),6),
            'leader_endpoint_xy':[left+ax,top+cy],
            'leader_endpoint_region_coverage':round(float(row[ax]),6)})
    category_ids[np.asarray(reduced_S)<128]=0
    category_canvas=Image.new('L',(W,H),0)
    category_canvas.paste(Image.fromarray(category_ids),(left,top))
    category_canvas.save(OUT/'category_map.png')
    # Narrow seam shadow and stitch highlights belong to their current bands.
    seam=np.zeros((gh,gw),dtype=np.float32)
    stitch=np.zeros_like(seam)
    for threshold in thresholds:
        delta=v-threshold
        seam+=.15*np.exp(-((delta-.55)/.58)**2)
        stitch+=.07*np.exp(-((delta+1.45)/.34)**2)*(np.mod(u,3.8)<2.0)
    color=color*(1-seam[:,:,None])+(235-color)*stitch[:,:,None]
    # fabric is an absolute visible alpha; this ratio prevents double alpha.
    fraction=np.divide(fabric,alpha,out=np.zeros_like(fabric),where=alpha>0)
    photo[:,:,:3]=photo[:,:,:3]*(1-fraction[:,:,None])+color*fraction[:,:,None]
    garment=Image.fromarray(np.uint8(np.clip(photo,0,255)))
    # Warm editorial background, no secondary chart or decorative garment icon.
    gy,gx=np.mgrid[:H*SS,:W*SS].astype(np.float32)
    bg=hexrgb('#eeeae2')[None,None,:]+(5*np.exp(-((gx-650*SS)/(780*SS))**2-((gy-870*SS)/(1100*SS))**2))[:,:,None]
    canvas=Image.fromarray(np.uint8(np.clip(bg,0,255))).convert('RGBA')
    shadow=Image.new('RGBA',canvas.size,(0,0,0,0))
    sh=Image.fromarray(np.uint8(np.clip(alpha*32,0,255))).filter(ImageFilter.GaussianBlur(20*SS))
    shadow.paste((55,57,54,255),((left+17)*SS,(top+17)*SS),sh)
    canvas.alpha_composite(shadow)
    canvas.alpha_composite(garment,(left*SS,top*SS))
    d=ImageDraw.Draw(canvas)
    ink='#273336'; muted='#65716d'; rule='#bcbfb6'
    text(d,(84,56),'SYNTHETIC DATA',22,ink,bold=True)
    line(d,(84,98,1376,98),'#bdc1b8',1)
    text(d,(79,114),'衣服布面分层',72,ink,title=True)
    text(d,(87,221),'Share · 可见布料面积占比 (%)',27,muted,cjk=True)
    text(d,(1374,158),'12 GROUPS',22,ink,bold=True,anchor='ra')
    text(d,(1374,203),'TOTAL 100%',19,muted,anchor='ra')
    # The first region is deliberately larger. Every label uses the same format.
    header_y=label_positions[0][1]-88
    text(d,(87,header_y),'CATEGORY',17,muted,bold=True)
    text(d,(311,header_y),'SHARE (%)',17,muted,bold=True,anchor='ra')
    label_boxes=[]
    for i,((ax,ay),cat,value,clr) in enumerate(zip(label_positions,cats,vals,PALETTE)):
        # The connector ends inside the measured region on the visible sleeve.
        line(d,(333,ay,ax,ay),'#6e7a77',1.05)
        d.ellipse(((ax-3)*SS,(ay-3)*SS,(ax+3)*SS,(ay+3)*SS),fill=PALETTE[i],outline='#f5f1e9',width=2)
        d.rounded_rectangle((84*SS,(ay-11)*SS,90*SS,(ay+11)*SS),radius=2*SS,fill=clr)
        text(d,(103,ay),cat,22,ink,anchor='lm')
        text(d,(314,ay),str(value)+'%',27,ink,anchor='rm')
        label_boxes.append({'category':cat,'bbox':[84,ay-20,315,ay+20]})
    line(d,(84,1819,1376,1819),'#bdc1b8',1)
    text(d,(84,1852),'色层面积 = 类别份额',29,ink,cjk=True)
    text(d,(84,1903),'按可见布料面积计量；白领与衣架不计入，袖身重叠只计可见前层。',23,muted,cjk=True)
    text(d,(1376,1951),'12 CATEGORIES  /  1 SERIES  /  PERCENT',15,muted,anchor='ra')
    final=canvas.convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
    final.save(ROOT/'final.png')
    gaps=[label_positions[i+1][1]-label_positions[i][1] for i in range(len(cats)-1)]
    check={'canvas':[W,H],'series':DATA['series'][0]['label'],'sum_percent':float(sum(Decimal(str(v)) for v in vals)),
        'mapping':'linear cumulative share of final projected visible fabric alpha area',
        'fabric_mask':'artifacts/visible_fabric_mask.png','category_map':'artifacts/category_map.png',
        'final_visible_fabric_area_px':round(total_area,4),
        'mask_partition_residual_px':round(sum(r['final_visible_area_px'] for r in area_records)-total_area,4),
        'raw_material_thresholds':[round(float(x),6) for x in thresholds],
        'max_absolute_error_pp':max(r['absolute_error_pp'] for r in area_records),
        'transform':{'asset_crop_xyxy':[178,9,1069,1257],'final_bbox_xywh':[left,top,garment_w,garment_h],'supersampling':SS},
        'categories':area_records,'min_label_center_gap_px':min(gaps),'label_boxes':label_boxes,
        'generated_material_contribution_zh':'整件衣物的摄影轮廓、棉织纹理、自然褶皱、门襟、纽扣、袖口、下摆、白领、衣架和挂钩均来自 assets/raw_01.png；代码在同一布料遮罩内着色分区。'}
    (OUT/'geometry_check.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'canvas':check['canvas'],'max_error_pp':check['max_absolute_error_pp'],
                      'min_label_gap_px':check['min_label_center_gap_px'],'shares':area_records},ensure_ascii=False))

if __name__=='__main__':
    main()
