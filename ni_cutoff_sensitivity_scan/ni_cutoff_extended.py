#!/usr/bin/env python3
import importlib.util, itertools,collections
import numpy as np
import pandas as pd
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
p = SCRIPT_DIR / 'cutoff_scan.py'
spec=importlib.util.spec_from_file_location('cs',p)
cs=importlib.util.module_from_spec(spec)
spec.loader.exec_module(cs)
out=cs.OUT
exO=np.round(np.arange(2.2,2.401,.05),2)
exH=np.round(np.arange(2.1,2.301,.05),2)
refs={}
for fr in cs.frames:
    x=(cs.assign(fr['ds_o'],cs.BASE_O,cs.EXT_O),cs.assign(fr['ds_h'],cs.BASE_H,cs.EXT_H))
    refs[fr['cat'],fr['idx']]=x
rows=[]; trs=[]; detailed=[]
for eo,eh in itertools.product(exO,exH):
    cnt=collections.Counter()
    for fr in cs.frames:
        key=(fr['cat'],fr['idx'])
        ref=refs[key]
        now=(cs.assign(fr['ds_o'],cs.BASE_O,float(eo)),cs.assign(fr['ds_h'],cs.BASE_H,float(eh)))
        r_m=ref[0][0]+ref[1][0]
        n_m=now[0][0]+now[1][0]
        r_a=(ref[0][1],ref[1][1]); n_a=(now[0][1],now[1][1])
        cnt[fr['cat'],'n']+=1
        cnt[fr['cat'],'motif']+=int(r_m!=n_m)
        cnt[fr['cat'],'anchors']+=int(r_a!=n_a)
        cnt[fr['cat'],'O']+=int(r_a[0]!=n_a[0])
        cnt[fr['cat'],'H']+=int(r_a[1]!=n_a[1])
        cnt[fr['cat'],'unassigned']+=int('-' in n_m)
        if r_m!=n_m:
            trs.append(dict(group=fr['cat'],label=fr['label'],ext_O=eo,ext_H=eh,original_motif=r_m,new_motif=n_m,E_ads_eV=fr['energy']))
        if eo in [2.25,2.3,2.35] and eh in [2.15,2.2,2.25]:
            detailed.append(dict(group=fr['cat'],config_label=fr['label'],ext_O=eo,ext_H=eh,original_motif=r_m,new_motif=n_m,changed_motif=r_m!=n_m,changed_anchors=r_a!=n_a))
    for cat in cs.FILES:
        d=dict(group=cat,extended_O_cutoff=eo,extended_H_cutoff=eh,n=cnt[cat,'n'])
        for k in ['motif','anchors','O','H','unassigned']:
            d[k+'_changed' if k!='unassigned' else k]=cnt[cat,k]
            d[k+'_pct']=100*cnt[cat,k]/cnt[cat,'n']
        rows.append(d)
ss=pd.DataFrame(rows)
ss.to_csv(out/'extended_fourfold_cutoff_scan.csv',index=False)
pd.DataFrame(trs).to_csv(out/'extended_fourfold_motif_transitions.csv',index=False)
pd.DataFrame(detailed).to_csv(out/'extended_fourfold_3x3_assignments.csv',index=False)
for cat in cs.FILES:
    print('\n',cat,'Extended F cutoffs % changing motifs vs O2.30/H2.20')
    pp=ss[ss.group==cat].pivot(index='extended_O_cutoff',columns='extended_H_cutoff',values='motif_pct')
    print(pp.to_string(float_format=lambda x:f'{x:5.2f}'))
    mx=ss[ss.group==cat].sort_values('motif_pct',ascending=False).iloc[0]
    print('max',mx[['extended_O_cutoff','extended_H_cutoff','motif_changed','motif_pct','anchors_changed','anchors_pct']].to_dict())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7.4,5.4)); mat=pp.loc[exO,exH]
    im=plt.imshow(mat.values,aspect='auto',origin='lower',cmap='viridis',vmin=0)
    plt.xticks(range(5),[f'{v:.2f}' for v in exH]);plt.yticks(range(5),[f'{v:.2f}' for v in exO]);
    plt.xlabel('Extended fourfold Ni–H cutoff (Å)');plt.ylabel('Extended fourfold Ni–O cutoff (Å)')
    plt.title(f'{cat}: Motif changes vs extended F cutoffs 2.30/2.20 Å')
    plt.colorbar(im,label='Motif reassignment (%)')
    vmax=max(mat.values.ravel())
    for (i,j),z in np.ndenumerate(mat.values):
        plt.text(j,i,f'{z:.1f}',ha='center',va='center',color='white' if z<vmax*0.35 else 'black',fontsize=9)
    plt.tight_layout();plt.savefig(out/f'heatmap_{cat}_extended_fourfold.png',dpi=190);plt.close()
