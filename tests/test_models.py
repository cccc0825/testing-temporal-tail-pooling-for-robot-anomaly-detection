"""Synthetic correctness checks. Fixtures are not research observations."""
from pathlib import Path
import tempfile
import numpy as np
from experiments import core
from experiments.prepare import parse_dict
from experiments.run import replay


def main():
    local=np.vstack([np.arange(128),np.zeros(128)])
    values=core.pool(local)
    assert values['tail_mean'][0]==np.arange(121,128).mean()
    assert values['tail_rms'][0]==np.sqrt(np.mean(np.arange(121,128)**2))
    assert np.isinf(core.threshold(np.arange(41),.01))
    assert core.threshold(np.arange(41),.05)==39
    assert (np.array([39,40])>core.threshold(np.arange(41),.05)).tolist()==[False,True]
    raw=np.tile(np.array([0.,1.,9.])[:,None],(1,12));t=np.array([0.,1.,9.])
    assert np.allclose(core.interpolate(raw,t,128,'elapsed')[:,0],np.linspace(0,9,128))
    assert not np.array_equal(core.interpolate(raw,t,128,'elapsed'),core.interpolate(raw,t,128,'index'))
    for bad in [np.array([0.,0.,1.]),np.array([1.,0.,2.])]:
        try:core.interpolate(raw,bad,128,'elapsed')
        except AssertionError:pass
        else:raise AssertionError('Invalid time accepted')
    z=np.array([[[3.,0.]],[[2.,2.]],[[0.,0.]]]);fit=np.array([0,1])
    assert core.nearest(z,fit,'l1')[2][2]==0 and core.nearest(z,fit,'l2')[2][2]==1
    for text in ['[1,2]',"__import__('os').getcwd()"]:
        try:parse_dict(text)
        except (ValueError,SyntaxError):pass
        else:raise AssertionError('Unsafe/non-dictionary metadata accepted')
    with tempfile.TemporaryDirectory(prefix='pooling-synthetic-') as tmp:
        root=Path(tmp);prep=root/'prepared';prep.mkdir()
        rng=np.random.default_rng(123)
        for kind,length in core.REPS:np.save(prep/f'ur3_{kind}{length}.npy',rng.normal(size=(12,length,12)))
        rows=[dict(id=i,anomaly=int(i>=10),category=int(i>=10)) for i in range(12)]
        core.dump(prep/'rows.json',rows);core.dump(prep/'split.json',dict(fit=list(range(5)),cal=list(range(5,9)),test=list(range(9,12))))
        core.dump(prep/'manifest.json',dict(state='completed',files={p.name:core.sha(p) for p in prep.iterdir()}))
        replay(prep,root/'run','ur3')
        import json
        metrics=json.loads((root/'run/metrics.json').read_text());assert len(metrics)==108
        assert all(r['n_normal']==1 and r['n_fault']==2 for r in metrics)
        try:replay(prep,root/'run','ur3')
        except FileExistsError:pass
        else:raise AssertionError('Existing run was overwritten')
    print('Synthetic interpolation, pooling, calibration, selector and model CLI checks passed.')

if __name__=='__main__':main()
