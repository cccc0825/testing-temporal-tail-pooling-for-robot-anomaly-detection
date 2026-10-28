"""Replay the paper's models from prepared arrays into a new output directory."""
import argparse
import json
import platform
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import joblib
from experiments import core


def replay(prepared, output, dataset, seed=2027):
    prepared,output=Path(prepared),Path(output)
    pm=json.loads((prepared/'manifest.json').read_text())
    assert pm['state']=='completed'
    for n,h in pm['files'].items():
        path=prepared/n
        assert path.is_relative_to(prepared) and '..' not in Path(n).parts
        assert core.sha(path)==h, f'Prepared file differs: {n}'
    rows=json.loads((prepared/'rows.json').read_text());sp=json.loads((prepared/'split.json').read_text())
    splits={k:np.asarray(v,dtype=int) for k,v in sp.items()}
    assert all(k in splits for k in ['fit','cal','test'])
    assert sorted(np.concatenate(list(splits.values())).tolist())==list(range(len(rows)))
    fit,cal=splits['fit'],splits['cal'];y=np.array([r['anomaly'] for r in rows]);cat=np.array([r['category'] for r in rows])
    assert len(fit)>1 and len(cal)>0 and set(np.unique(y)) <= {0,1}
    assert np.all(y[fit]==0) and np.all(y[cal]==0)
    output.mkdir(parents=True,exist_ok=False);begin=time.perf_counter()
    m=dict(state='running',started_at=core.stamp(),dataset=dataset,seed=seed,
           code_sha256=core.sha(__file__),kernel_sha256=core.sha(core.__file__),
           prepared_manifest_sha256=core.sha(prepared/'manifest.json'),python=platform.python_version(),command=sys.argv)
    core.dump(output/'manifest.json',m)
    try:
        def load(kind,length):
            x=np.load(prepared/f'{dataset}_{kind}{length}.npy',allow_pickle=False)
            assert x.shape==(len(rows),length,12) and np.isfinite(x).all()
            return x
        x=load('elapsed',128);scores,model,features=core.simple_features(x,fit)
        core.dump(output/'simple_model.json',model);cost=[]
        for timing,length,ref in core.CONFIGS:
            t0=time.perf_counter();x=load(timing,length);mean,std=core.fit_normalization(x,fit)
            vals,local,refs=core.nearest((x-mean)/std,fit,ref);cfg=f'{timing}{length}_{ref}'
            scores.update({f'nn_{cfg}_{name}':v for name,v in vals.items()})
            np.savez_compressed(output/f'{cfg}_model.npz',mean=mean,std=std,fit_index=fit,reference_index=refs,local_l1=local)
            cost.append(dict(config=cfg,fit_score_seconds=time.perf_counter()-t0))
        x=load('elapsed',128);mean,std=core.fit_normalization(x,fit);flat=((x-mean)/std).reshape(len(x),-1)
        t0=time.perf_counter();pca=PCA(n_components=.95,svd_solver='full').fit(flat[fit])
        scores['pca']=np.sqrt(np.mean((flat-pca.inverse_transform(pca.transform(flat)))**2,1))
        np.savez_compressed(output/'pca_model.npz',components=pca.components_,mean=pca.mean_,explained_variance=pca.explained_variance_,channel_mean=mean,channel_std=std)
        cost.append(dict(config='pca',fit_score_seconds=time.perf_counter()-t0))
        t0=time.perf_counter();iso=IsolationForest(n_estimators=200,max_samples=min(256,len(fit)),max_features=1.,n_jobs=2,random_state=seed).fit(flat[fit])
        scores['iforest']=-iso.score_samples(flat);joblib.dump(iso,output/'iforest_model.joblib')
        cost.append(dict(config='iforest',fit_score_seconds=time.perf_counter()-t0))
        assert len(scores)==36 and all(np.isfinite(v).all() for v in scores.values())
        np.savez_compressed(output/'scores.npz',**scores)
        metrics=[];decisions={}
        for name,s in scores.items():
            for alpha in core.ALPHAS:
                threshold=core.threshold(s[cal],alpha);pred=s>threshold
                decisions[f'{name}__{alpha}']=pred
                for co,ix in splits.items():
                    if co in ['fit','cal']:continue
                    metrics.append(dict(dataset=dataset,cohort=co,seed=seed,method=name,alpha=alpha,
                                        n_fit=len(fit),n_cal=len(cal),threshold=float(threshold) if np.isfinite(threshold) else None,
                                        infinite_threshold=bool(np.isinf(threshold)),**core.metrics(s[ix],y[ix],cat[ix],pred[ix])))
        np.savez_compressed(output/'predictions.npz',**decisions)
        core.dump(output/'metrics.json',metrics);core.dump(output/'cost.json',cost)
        core.dump(output/'rows.json',rows);core.dump(output/'split.json',sp)
        if dataset=='factory':
            a=scores['nn_elapsed128_l1_mean'];b=scores['nn_elapsed128_l1_tail_rms'];ix=splits['test'];yy=y[ix]
            delta=(b>core.threshold(b[cal],.05)).astype(int)[ix]-(a>core.threshold(a[cal],.05)).astype(int)[ix]
            boots=[]
            for length in [10,25,50]:
                rng=np.random.default_rng(3000+length);vals=[]
                for _ in range(2000):
                    starts=rng.integers(0,len(ix),int(np.ceil(len(ix)/length)))
                    jj=((starts[:,None]+np.arange(length))%len(ix)).ravel()[:len(ix)];q=yy[jj];d=delta[jj]
                    if not np.any(q==0) or not np.any(q==1):continue
                    vals.append([d[q==0].mean(),d[q==1].mean()])
                boots.append(dict(block_length=length,draws=2000,valid=len(vals),fpr_tpr_percentiles=np.quantile(vals,[.025,.975],axis=0,method='linear').tolist()))
            core.dump(output/'block_sensitivity.json',boots)
        m.update(state='completed',ended_at=core.stamp(),wall_seconds=time.perf_counter()-begin,
                 metric_records=len(metrics),files={p.name:core.sha(p) for p in output.iterdir() if p.is_file() and p.name!='manifest.json'})
    except BaseException as e:
        m.update(state='failed',error=repr(e),traceback=traceback.format_exc());raise
    finally:core.dump(output/'manifest.json',m)
    print(f'Completed {len(metrics)} metric records; outputs: {output}')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('dataset',choices=['voraus','ur3','factory'])
    p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=2027)
    a=p.parse_args();replay(a.prepared,a.output,a.dataset,a.seed)

if __name__=='__main__':main()
