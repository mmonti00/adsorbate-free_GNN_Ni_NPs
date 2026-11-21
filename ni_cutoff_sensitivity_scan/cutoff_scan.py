#!/usr/bin/env python3
"""Reproducible Ni-O/Ni-H final-site motif cutoff sensitivity scan for extxyz.

No ASE dependency. Handles triclinic periodic cells via exhaustive nearby lattice images.
Hybrid fourfold rule: exactly four within extended shell AND at least two within
short shell, otherwise ordinary nearest 1/2/3 within short shell. Inputs may
be unordered; unique O and hydroxyl H are identified geometrically.
"""
import re, itertools, collections
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent
FILES = {
    'NP': REPOSITORY_ROOT / 'datasets' / 'dataset_NPs_M6_energy_model.extxyz',
    'SLAB': REPOSITORY_ROOT / 'datasets' / 'dataset_SLABs_M6_energy_model.extxyz',
}
OUT = SCRIPT_DIR
OUT.mkdir(exist_ok=True)
O_GRID=np.round(np.arange(2.10,2.301,.05),2)
H_GRID=np.round(np.arange(1.90,2.101,.05),2)
BASE_O=2.20
BASE_H=2.00
EXT_O=2.30
EXT_H=2.20


def header_val(header,key):
    pat=re.compile(r'(?:^|\s)'+re.escape(key)+r'=([^\s]+|"[^"]*")')
    if key in ('Lattice','pbc'):
        m=re.search(r'(?<!\w)'+key+r'="([^"]+)"',header)
        return m.group(1) if m else None
    m=pat.search(header)
    return m.group(1).strip('"') if m else None


def get_distances(ni, ads, cell, periodic):
    d=ni-ads
    if not np.any(periodic): return np.linalg.norm(d,axis=1)
    # Full lattice min-image calculation including tilted cells.
    frac=d@np.linalg.inv(cell)
    frac[:,periodic]-=np.rint(frac[:,periodic])
    base=frac@cell
    choices=[(-1,0,1) if p else (0,) for p in periodic]
    shifts=np.array(list(itertools.product(*choices)),dtype=float)@cell
    vectors=base[:,None,:]+shifts[None,:,:]
    return np.sqrt(np.einsum('ijk,ijk->ij', vectors,vectors).min(axis=1))


def get_frames(path,cat):
    with open(path) as f:
        i=0
        while True:
            line=f.readline()
            if not line: return
            if not line.strip():continue
            n=int(line)
            hdr=f.readline().strip()
            latt=header_val(hdr,'Lattice')
            cell=np.array(list(map(float,latt.split()))).reshape(3,3) if latt else np.eye(3)*100
            pbc=header_val(hdr,'pbc')
            periodic=np.array([s.upper().startswith('T') for s in pbc.split()]) if pbc else np.array([False]*3)
            rows=[f.readline().split() for j in range(n)]
            syms=np.array([r[0] for r in rows]); pos=np.array([list(map(float,r[1:4])) for r in rows]);
            inds_ni=np.where(syms=='Ni')[0]
            inds_o=np.where(syms=='O')[0]
            inds_h=np.where(syms=='H')[0]
            if len(inds_o)!=1 or len(inds_h)!=2: raise ValueError((i,cat,len(inds_o),len(inds_h)))
            od=pos[inds_o[0]]
            dh=get_distances(pos[inds_h],od,cell,periodic)
            idxh=inds_h[np.argmax(dh)]
            # Both H are supplied (1 OH bound, 1 adsorbed H)
            en=header_val(hdr,'energy')
            label=header_val(hdr,'config_label') or f'{cat}_{i}'
            d_o=get_distances(pos[inds_ni],od,cell,periodic)
            d_h=get_distances(pos[inds_ni],pos[idxh],cell,periodic)
            yield dict(cat=cat,idx=i,label=label,energy=float(en) if en else np.nan,n_ni=len(inds_ni),ds_o=d_o,ds_h=d_h,ni_idx=inds_ni,oh_length=float(np.min(dh)))
            i+=1


def assign(ds, standard, extended):
    # 1e-8 buffer for float roundoff only.
    sh=np.where(ds < standard+1e-8)[0]
    ext=np.where(ds < extended+1e-8)[0]
    if len(ext)==4 and len(sh)>=2:
        return ('F',tuple(map(int,ext)))
    sh=sh[np.argsort(ds[sh], kind='stable')][:3]
    n=len(sh)
    return ({0:'-',1:'T',2:'B',3:'H'}[n],tuple(sorted(map(int,sh))))


def motif(fr, o,h):
    oa=assign(fr['ds_o'],o,EXT_O)
    ha=assign(fr['ds_h'],h,EXT_H)
    return oa[0]+ha[0],(oa[1],ha[1])

frames=[]
for cat,path in FILES.items():
    fs=list(get_frames(path,cat)); frames+=fs
    print(cat,len(fs),dict(collections.Counter(x['n_ni'] for x in fs)), 'OH distances range',min(x['oh_length'] for x in fs),max(x['oh_length'] for x in fs),flush=True)

assert len(frames)==2081
for hh in [1.9,2.0]:
    co=collections.Counter(motif(fr,2.2,hh)[0] for fr in frames)
    print('REFERENCE COUNTS',hh,sorted(co.items()),'total',sum(co.values()),flush=True)

# baseline motif and anchor sets
baseline={}
for fr in frames: baseline[(fr['cat'],fr['idx'])]=motif(fr,BASE_O,BASE_H)

records=[]
config_records=[]
trans_records=[]
for o,h in itertools.product(O_GRID,H_GRID):
    c=collections.Counter()
    for fr in frames:
        key=(fr['cat'],fr['idx'])
        bm,ba=baseline[key]
        mm,ma=motif(fr,float(o),float(h))
        c[(fr['cat'],'total')]+=1
        c[(fr['cat'],'changed_motif')]+=int(mm!=bm)
        c[(fr['cat'],'changed_anchors')]+=int(ma!=ba)
        c[(fr['cat'],'changed_O_anchors')]+=int(ma[0]!=ba[0])
        c[(fr['cat'],'changed_H_anchors')]+=int(ma[1]!=ba[1])
        c[(fr['cat'],'unassigned')]+=int('-' in mm)
        c[(fr['cat'],'motif_same_anchor_changed')]+=int(mm==bm and ma!=ba)
        config_records.append(dict(group=fr['cat'],config_label=fr['label'],frame_idx=fr['idx'],E_ads_eV=fr['energy'],N_Ni=fr['n_ni'],O_cutoff=float(o),H_cutoff=float(h),baseline_motif=bm,motif=mm,motif_changed=(mm!=bm),O_anchors_changed=(ma[0]!=ba[0]),H_anchors_changed=(ma[1]!=ba[1]),anchors_changed=(ma!=ba),baseline_O_anchors=str(ba[0]),baseline_H_anchors=str(ba[1]),O_anchors=str(ma[0]),H_anchors=str(ma[1])))
        if mm!=bm:
            trans_records.append(dict(group=fr['cat'],config_label=fr['label'],O_cutoff=float(o),H_cutoff=float(h),from_motif=bm,to_motif=mm,E_ads_eV=fr['energy']))
    for cat in ('NP','SLAB'):
        total=c[cat,'total']
        row=dict(group=cat, O_cutoff=float(o), H_cutoff=float(h),n=total)
        for k in ('changed_motif','changed_anchors','changed_O_anchors','changed_H_anchors','unassigned','motif_same_anchor_changed'):
            row[k]=c[cat,k]
            row[k+'_pct']=100*c[cat,k]/total
        records.append(row)

scan=pd.DataFrame(records)
scan.to_csv(OUT/'cutoff_scan_summary.csv',index=False)
conf=pd.DataFrame(config_records)
conf.to_csv(OUT/'all_configuration_assignments.csv',index=False)
trans=pd.DataFrame(trans_records)
trans.to_csv(OUT/'motif_transitions.csv',index=False)

# Compact derived tables used by REPORT.md.  Keep their construction here so
# every CSV in this directory can be regenerated from the released datasets.
np_energy = (
    conf[(conf['group'] == 'NP') & conf['motif'].isin(['BH', 'HH', 'TH'])]
    .groupby(['O_cutoff', 'H_cutoff', 'motif'], sort=True)['E_ads_eV']
    .agg(['count', 'mean', 'median'])
    .reset_index()
)
np_energy.to_csv(OUT/'NP_DFT_energy_by_motif_and_cutoff.csv', index=False)

conf['system'] = conf['config_label'].str.extract(r'^(Ni\d+)')[0]
per_system = (
    conf.groupby(['system', 'group', 'O_cutoff', 'H_cutoff'], sort=True)
    .agg(
        n=('motif', 'size'),
        changed=('motif_changed', 'sum'),
        changed_anchor=('anchors_changed', 'sum'),
    )
    .reset_index()
)
per_system['pct'] = 100.0 * per_system['changed'] / per_system['n']
per_system.to_csv(OUT/'per_system_cutoff_summary.csv', index=False)

# Motif counts per grid and group, including totals
counts=[]
for (cat,o,h), group in conf.groupby(['group','O_cutoff','H_cutoff']):
  cnt=group['motif'].value_counts()
  for mm,num in cnt.items():
   counts.append(dict(group=cat,O_cutoff=o,H_cutoff=h,motif=mm,count=int(num)))
pd.DataFrame(counts).to_csv(OUT/'motif_counts_per_cutoff.csv',index=False)

# Configurations with smallest distances from short-shell cutoff baseline
near=[]
for fr in frames:
    for what,ds,base in [('O',fr['ds_o'],BASE_O),('H',fr['ds_h'],BASE_H)]:
        nearest_idx=int(np.argmin(abs(ds-base)))
        near.append(dict(group=fr['cat'],label=fr['label'],element=what,nearest_distance_ang=float(ds[nearest_idx]),delta_cutoff_ang=float(ds[nearest_idx]-base),N_Ni=fr['n_ni'],E_ads_eV=fr['energy']))
pd.DataFrame(near).to_csv(OUT/'nearest_anchor_distance_to_cutoff.csv',index=False)

# Summary per baseline and around baseline
print('\nSUMMARY per group (rows O=2.20, H=1.90|1.95|2.00|2.05|2.10):',flush=True)
print(scan[(scan.O_cutoff==2.20)].to_string(index=False),flush=True)

# Plots with two panels? plotting only one figure per file. Default colors. Matplotlib allowed here internal output.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
for cat in ('NP','SLAB'):
    sub=scan[scan.group==cat]
    for key,textlabel,filebit in [('changed_motif_pct','Motif reassignment (%)','motif'),('changed_anchors_pct','Exact anchor-set reassignment (%)','anchors')]:
        mat=sub.pivot(index='O_cutoff',columns='H_cutoff',values=key).loc[O_GRID,H_GRID]
        plt.figure(figsize=(7.4,5.4))
        im=plt.imshow(mat.values,aspect='auto',origin='lower',cmap='viridis',vmin=0)
        plt.xticks(range(len(H_GRID)),[f'{v:.2f}' for v in H_GRID]); plt.yticks(range(len(O_GRID)),[f'{v:.2f}' for v in O_GRID])
        plt.xlabel('Standard Ni–H cutoff (Å)');plt.ylabel('Standard Ni–O cutoff (Å)')
        plt.title(f'{cat}: {textlabel}, relative to 2.20/2.00 Å')
        plt.colorbar(im,label=textlabel)
        for (i,j),z in np.ndenumerate(mat.values):
            plt.text(j,i,f'{z:.1f}',ha='center',va='center',color='white' if z<max(mat.values.ravel())*0.35 else 'black',fontsize=9)
        plt.tight_layout();plt.savefig(OUT/f'heatmap_{cat}_{filebit}.png',dpi=190);plt.close()

print('FILES',list(p.name for p in OUT.iterdir()),flush=True)
