"""Prepare the paper's four representations without computing anomaly scores.

Portable I/O around the archived selection, exclusion and interpolation rules.
All paths are supplied by the caller; existing output directories are refused.
"""
import argparse
import ast
import io
import json
import platform
import sys
import traceback
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.core import REPS, dump, interpolate, sha, stamp

FACTORY_SHA = '75e3dc82e50f2ec7bc1621b4355ccd213c530c6b9901e1bce6de9510cbb444ec'
METADATA_SHA = '6feab3cfd127a0cd9e1564c80159d6bd01d7309abb2ec25678fa8b4fe1f2b0e2'
UR3_SHA = '233334eeb5370e594ea621796ca2d693bd4980cc2bb24fa749b1b633a4993c8d'


def save_arrays(out, dataset, rows, raw, times, split, all_rows=None):
    ids = [i for part in split.values() for i in part]
    assert sorted(ids) == list(range(len(rows))), 'splits must be disjoint and complete'
    assert all(rows[i]['anomaly'] == 0 for part in ('fit', 'cal') for i in split[part])
    for kind, length in REPS:
        x = np.asarray([interpolate(a, t, length, kind) for a, t in zip(raw, times)])
        assert x.shape == (len(rows), length, 12) and np.isfinite(x).all()
        np.save(out / f'{dataset}_{kind}{length}.npy', x)
    dump(out / 'rows.json', rows)
    dump(out / 'split.json', split)
    dump(out / 'all_rows.json', all_rows if all_rows is not None else rows)


def ur3(source, out):
    import openpyxl
    assert sha(source) == UR3_SHA, 'UR3 source ZIP differs from the archived version'
    with zipfile.ZipFile(source) as z:
        names = [n for n in z.namelist() if n.endswith('.xlsx')]
        assert len(names) == 1
        wb = openpyxl.load_workbook(io.BytesIO(z.read(names[0])), read_only=True, data_only=True)
        table = list(wb.active.values)
        wb.close()
    d = pd.DataFrame([r[:24] for r in table[1:]], columns=[s.strip() for s in table[0][:24]])
    ts = pd.to_datetime(d.Timestamp.str.strip('"'), utc=True)
    t = (ts-ts.iloc[0]).dt.total_seconds().to_numpy()
    assert len(d) == 7409 and np.all(np.diff(t) > 0)
    names = [f'{s}_J{k}' for s in ['Current', 'Speed'] for k in range(6)]
    x = d[names].to_numpy(float)
    cy = d.cycle.to_numpy()
    cuts = np.r_[0, np.flatnonzero(cy[1:] != cy[:-1])+1, len(cy)]
    assert len(cuts) == 250
    rows, raw, times, all_rows = [], [], [], []
    for b, (lo, hi) in enumerate(zip(cuts[:-1], cuts[1:])):
        a, dt = x[lo:hi], d.iloc[lo:hi]
        gap = float(np.max(np.diff(t[lo:hi]))) if hi-lo > 1 else 0
        reasons = []
        if not np.isfinite(a).all(): reasons.append('missing_sensor')
        if dt[['Robot_ProtectiveStop', 'grip_lost']].isna().any().any(): reasons.append('missing_label')
        if gap > 10: reasons.append('time_gap_over_10s')
        if hi-lo < 2: reasons.append('too_short')
        if b in (0, len(cuts)-2): reasons.append('boundary_cycle')
        cat = int(bool(dt.Robot_ProtectiveStop.dropna().any())) + 2*int(bool(dt.grip_lost.dropna().any()))
        part = 'fit' if b < 99 else 'cal' if b < 174 else 'test'
        if part != 'test' and cat: reasons.append('nonnominal_fit_or_cal')
        r = dict(id=b, start=int(lo), end=int(hi), n_frames=int(hi-lo), category=cat,
                 anomaly=int(cat > 0), partition=part, exclusion_reasons=reasons, eligible=not reasons)
        all_rows.append(r)
        if not reasons:
            rows.append(r); raw.append(a); times.append(t[lo:hi])
    split = {k: [i for i,r in enumerate(rows) if r['partition']==k] for k in ['fit','cal','test']}
    assert {k:len(v) for k,v in split.items()} == {'fit':63,'cal':41,'test':43}
    save_arrays(out, 'ur3', rows, raw, times, split, all_rows)
    return [source]


def voraus(columns, out, seed):
    names = [f'{s}_{k}' for s in ['motor_torque','joint_velocity'] for k in range(1,7)]
    cols = {n: np.load(columns / f'{n}.npy', mmap_mode='r', allow_pickle=False)
            for n in names+['sample','time','anomaly','category','setting']}
    sid = cols['sample']; cuts = np.r_[0,np.flatnonzero(sid[1:]!=sid[:-1])+1,len(sid)]
    assert len(cuts)==2123 and len(np.unique(sid))==2122
    assert len({len(v) for v in cols.values()}) == 1
    rows, raw, times = [], [], []
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        assert all(np.all(cols[n][lo:hi] == cols[n][lo]) for n in ['anomaly','category','setting'])
        rows.append(dict(sample=int(sid[lo]),start=int(lo),end=int(hi),n_frames=int(hi-lo),
                         **{n:int(cols[n][lo]) for n in ['anomaly','category','setting']}))
        raw.append(np.column_stack([cols[n][lo:hi] for n in names]))
        times.append(np.asarray(cols['time'][lo:hi]))
    train = np.array([i for i,r in enumerate(rows) if r['setting']==72])
    assert len(train)==948 and all(rows[i]['anomaly']==0 for i in train)
    train = np.random.default_rng(seed).permutation(train)
    split = dict(fit=train[:568].tolist(),cal=train[568:758].tolist(),audit=train[758:].tolist(),
                 test=[i for i,r in enumerate(rows) if r['setting']!=72])
    assert sum(rows[i]['anomaly']==0 for i in split['test'])==419
    assert sum(rows[i]['anomaly']==1 for i in split['test'])==755
    save_arrays(out,'voraus',rows,raw,times,split)
    return [columns / f'{n}.npy' for n in cols]


def parse_dict(v):
    if v is None: return {}
    if isinstance(v,dict): return v
    result=ast.literal_eval(v)
    if not isinstance(result,dict): raise ValueError('metadata must be a literal dictionary')
    return result


def factory_records(metadata, available_ids):
    import pyarrow.parquet as pq
    assert sha(metadata)==METADATA_SHA, 'FactoryNet metadata version differs'
    rows=[]
    for r in pq.read_table(metadata).to_pylist():
        m=parse_dict(r['episode_metadata']); f=parse_dict(r['fault_metadata'])
        if r['id'] not in available_ids or r['flow_id']!='00000000-0000-0000-0000-000000000010' or r['counterfactual']!=0 or m.get('task')!='pick_and_place' or m.get('condition') not in ['normal','fault']: continue
        assert m['condition'] in ['normal','fault']
        anomaly=int(m['condition']=='fault'); date=str(r['created_at'])[:10]
        assert date in ['2026-04-06','2026-04-07','2026-04-08','2026-04-10','2026-04-11','2026-04-13']
        part='excluded' if date=='2026-04-06' else 'test' if date=='2026-04-07' else 'stress'
        if part=='stress': assert anomaly==1
        rows.append(dict(id=r['id'],created_at=str(r['created_at']),date=date,partition=part,anomaly=anomaly,
                         category=int(m.get('fault_id',f.get('fault_id'))) if anomaly else -1))
    rows.sort(key=lambda r:(r['created_at'],r['id']))
    assert len(rows)==4308 and len({r['id'] for r in rows})==4308
    normal6=[r for r in rows if r['date']=='2026-04-06' and not r['anomaly']]
    assert len(normal6)==377
    for i,r in enumerate(normal6):r['partition']='fit' if i<226 else 'cal'
    return rows


def factory(source, metadata, out):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    assert sha(source)==FACTORY_SHA, 'Requires the complete pinned Parquet, not a sparse carrier'
    pf=pq.ParquetFile(source)
    assert pf.metadata.num_rows==7129261 and pf.num_row_groups==59
    available_ids=set()
    for group in range(pf.num_row_groups):
        available_ids.update(pf.read_row_group(group,columns=['episode_id'],use_threads=False)['episode_id'].unique().to_pylist())
    rows=factory_records(metadata,available_ids);wanted={r['id'] for r in rows}
    signals=[f'{s}_{k}' for s in ['effort_current','feedback_vel'] for k in range(6)]
    # Group only required columns; typical selected arrays fit within 1 GiB.
    segments={k:[] for k in wanted}; ids=pa.array(sorted(wanted),type=pa.large_string())
    for i in range(pf.num_row_groups):
        tab=pf.read_row_group(i,columns=['episode_id','time_s']+signals,use_threads=False)
        tab=tab.filter(pc.is_in(tab['episode_id'],value_set=ids))
        if not len(tab):continue
        keys=np.asarray(tab['episode_id'].to_pylist(),dtype=object)
        values=np.column_stack([tab[n].to_numpy() for n in ['time_s']+signals])
        cuts=np.r_[0,np.flatnonzero(keys[1:]!=keys[:-1])+1,len(keys)]
        for lo,hi in zip(cuts[:-1],cuts[1:]):segments[keys[lo]].append(values[lo:hi])
    assert all(segments.values()), 'Selected UUID missing from source; stop before quality exclusions'
    eligible,raw,times=[],[],[]
    for r in rows:
        a=np.concatenate(segments[r['id']]);a=a[np.argsort(a[:,0],kind='stable')]
        t=(a[:,0]-a[0,0])*1000;x=a[:,1:];reasons=[]
        if len(x)<2:reasons.append('fewer_than_2_frames')
        if not np.isfinite(x).all():reasons.append('nonfinite_selected_sensor')
        if not np.isfinite(t).all():reasons.append('nonfinite_time')
        if len(t)>=2 and np.isfinite(t).all():
            if not np.all(np.diff(t)>0):reasons.append('nonmonotonic_or_duplicate_time')
            if np.diff(t).max()>1:reasons.append('max_gap_over_1_second')
        if r['partition']=='excluded':reasons.append('prespecified_first_day_fault_exclusion')
        r.update(eligible=not reasons,exclusion_reasons=reasons,n_frames=len(x))
        if not reasons:eligible.append(r);raw.append(x);times.append(t)
    split={k:[i for i,r in enumerate(eligible) if r['partition']==k] for k in ['fit','cal','test','stress']}
    assert {k:len(v) for k,v in split.items()}=={'fit':226,'cal':150,'test':1638,'stress':2219}
    save_arrays(out,'factory',eligible,raw,times,split,rows)
    return [source,metadata]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset',choices=['voraus','ur3','factory']);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--metadata',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=2027)
    a=p.parse_args()
    if a.dataset=='factory' and a.metadata is None:p.error('--metadata is required for FactoryNet')
    a.output.mkdir(parents=True,exist_ok=False)
    m=dict(state='running',started_at=stamp(),dataset=a.dataset,seed=a.seed,python=platform.python_version(),code_sha256=sha(__file__),command=sys.argv)
    dump(a.output/'manifest.json',m)
    try:
        inputs=ur3(a.source,a.output) if a.dataset=='ur3' else voraus(a.source,a.output,a.seed) if a.dataset=='voraus' else factory(a.source,a.metadata,a.output)
        m.update(state='completed',ended_at=stamp(),input_hashes={p.name:sha(p) for p in inputs},files={p.name:sha(p) for p in a.output.iterdir() if p.is_file() and p.name!='manifest.json'})
    except BaseException as e:
        m.update(state='failed',error=repr(e),traceback=traceback.format_exc());raise
    finally:dump(a.output/'manifest.json',m)

if __name__=='__main__':main()
