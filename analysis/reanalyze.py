#!/usr/bin/env python3
"""Post-hoc descriptive frozen-score diagnostics; see POSTHOC_PROTOCOL.json."""
from pathlib import Path
import json,hashlib,datetime,sys,platform
import numpy as np
import pandas as pd
if '--compute-only' not in sys.argv:
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'analysis'; FIG=ROOT/'figures'; FIG.mkdir(exist_ok=True)
SRC=ROOT/'inputs'
POOLS=['mean','rms','max','tail_mean','tail_rms'];LABELS=['Mean','All RMS','Max','Tail mean','Tail RMS']
COLORS=['#0072B2','#009E73','#777777','#CC79A7','#D55E00'];STYLES=['-','--',':','-.',(0,(3,1,1,1))]
if '--compute-only' not in sys.argv:
 plt.rcParams.update({'font.size':9.5,'axes.titlesize':10,'axes.labelsize':9.5,'legend.fontsize':9.2,'xtick.labelsize':9.2,'ytick.labelsize':9.2,'pdf.fonttype':42,'ps.fonttype':42})
protocol=json.loads((OUT/'POSTHOC_PROTOCOL.json').read_text())
for f in protocol['inputs']:
 rel=f['path']
 assert hashlib.sha256((SRC/rel).read_bytes()).hexdigest()==f['sha256']

def roc(y,s):
 order=np.argsort(-s,kind='stable');ss=s[order];yy=y[order];ends=np.r_[np.flatnonzero(np.diff(ss)),len(ss)-1]
 tp=np.cumsum(yy)[ends];fp=np.cumsum(1-yy)[ends]
 return np.r_[0,fp/(1-y).sum()],np.r_[0,tp/y.sum()],np.r_[np.inf,ss[ends]]
def diagnostics(x,y,b):
 ux=np.unique(x);uy=np.array([max(y[x==z]) for z in ux]);area=0.
 for i,z in enumerate(ux):
  end=min(b,ux[i+1] if i+1<len(ux) else b)
  if end>z:area+=(end-z)*uy[i]
 # Linear area uses original ROC vertices, not upper-envelope collapsed ties.
 linear=0.
 for x0,x1,y0,y1 in zip(x[:-1],x[1:],y[:-1],y[1:]):
  if x1>x0 and x0<b:
   hi=min(x1,b);yh=y0+(y1-y0)*(hi-x0)/(x1-x0);linear+=(hi-x0)*(y0+yh)/2
 return max(y[x<=b]),area/b,linear/b,ux,uy
# Deterministic tie and FPR-grid convention checks.
x,y,_=roc(np.array([1,0]),np.array([1.,1.]));assert np.allclose(diagnostics(x,y,.1)[:3],(0.,0.,.05))
x,y,_=roc(np.array([1,0]),np.array([2.,1.]));assert np.allclose(diagnostics(x,y,.05)[:3],(1.,1.,1.))
allrows=[];curves=[];cats=[];normal=[];fitcheck=[];plots={};normalplot={}
for ds,title in [('voraus','voraus-AD'),('ur3','UR3 CobotOps'),('factory','FactoryNet')]:
 if ds=='factory': prep=SRC/'new_external/prepared';res=SRC/'new_external/results/factory_pickplace_v1';rf='rows.json';sf='split.json';scorefile='scores.npz'
 else:prep=SRC/'existing_probe/prepared';res=SRC/'existing_probe/results/existing_probe_v1';rf=f'{ds}_rows.json';sf=f'{ds}_split.json';scorefile=f'{ds}_scores.npz'
 rows=json.loads((prep/rf).read_text());split=json.loads((prep/sf).read_text());scores=np.load(res/scorefile);y=np.array([r['anomaly'] for r in rows]);cat=np.array([r['category'] for r in rows]);test=np.array(split['test']);cal=np.array(split['cal']);fit=np.array(split['fit']);assert not set(cal)&set(test);assert not y[cal].any();plots[ds]=[];normalplot[ds]=[]
 for pool,label in zip(POOLS,LABELS):
  s=scores[f'nn_elapsed128_l1_{pool}'];rank=int(np.ceil((len(cal)+1)*.95));th=np.sort(s[cal])[rank-1] if rank<=len(cal) else np.inf;pred=s[test]>th;yy=y[test];n0=int((yy==0).sum());n1=int(yy.sum());fa=int(pred[yy==0].sum());det=int(pred[yy==1].sum());x,t,thresholds=roc(yy,s[test]);r={'dataset':ds,'pool':pool,'n_normal':n0,'n_fault':n1,'threshold':th,'false_alarms':fa,'detected':det,'fpr':fa/n0,'tpr':det/n1,'auroc':float(np.trapezoid(t,x))}
  for b in [.05,.10]:
   m,step,linear,ux,ut=diagnostics(x,t,b);r[f'tpr_at_fpr_le_{b}']=m;r[f'pauc_step_normalized_{b}']=step;r[f'pauc_linear_normalized_{b}']=linear
  allrows.append(r);plots[ds].append((ux,ut,r));curves.extend({'dataset':ds,'pool':pool,'fpr':a,'tpr':b,'test_only_threshold_ge':float(c)} for a,b,c in zip(x,t,thresholds))
  fitcheck.append({'dataset':ds,'pool':pool,'fit_n':len(fit),'fit_zero':int((s[fit]==0).sum()),'comparison_use':'excluded: self-reference scores, not held-out nominal performance'})
  for stage,ix in [('cal',cal),('test_normal',test[yy==0])]:
   v=s[ix];qs=np.quantile(v,[0,.25,.5,.75,.9,.95,1]);normal.append(dict(dataset=ds,pool=pool,stage=stage,n=len(v),threshold=th,exceed=int((v>th).sum()),**{k:float(z) for k,z in zip(['min','q25','median','q75','q90','q95','max'],qs)}))
   if pool in ['mean','tail_rms']:normalplot[ds].append((pool,stage,np.sort(v)/th))
 if ds=='ur3':
  for key,label in [(f'nn_elapsed128_l1_{p}',l) for p,l in zip(POOLS,LABELS)]+[('motion_quiet_fraction','Quiet fraction')]:
   s=scores[key];th=np.sort(s[cal])[int(np.ceil((len(cal)+1)*.95))-1];pred=s[test]>th
   for c,name in [(1,'Protective stop only'),(2,'Grip loss only'),(3,'Both')]:
    sel=(cat[test]==c)&(y[test]==1);n=int(sel.sum());d=int(pred[sel].sum());cats.append(dict(method=label,category=name,support=n,detected=d,recall=d/n,false_alarms=int(pred[y[test]==0].sum()),normal_support=n0,source='frozen continuous scores'))
old=json.loads((SRC/'original/metrics.json').read_text())
for method,label in [('pca','PCA'),('iforest','Isolation forest')]:
 r=next(r for r in old if r['dataset']=='ur3' and r['cohort']=='test' and r['seed']==2027 and r['alpha']==.05 and r['method']==method)
 for c,name in [('1','Protective stop only'),('2','Grip loss only'),('3','Both')]:
  q=r['fault_categories'][c];cats.append(dict(method=label,category=name,support=q['n'],detected=q['detected'],recall=q['detected']/q['n'],false_alarms=r['false_alarms'],normal_support=r['n_normal'],source='original frozen category metrics'))
df=pd.DataFrame(allrows);df.to_csv(OUT/'pooling_metrics.csv',index=False);pd.DataFrame(curves).to_csv(OUT/'roc_vertices.csv',index=False);pd.DataFrame(cats).to_csv(OUT/'ur3_category_matrix.csv',index=False);pd.DataFrame(normal).to_csv(OUT/'normal_score_distributions.csv',index=False);pd.DataFrame(fitcheck).to_csv(OUT/'fitting_score_exclusion.csv',index=False)
contrasts=[]
for ds in ['voraus','ur3','factory']:
 d=df[df.dataset==ds].set_index('pool')
 for a,b in [('mean','rms'),('mean','tail_mean'),('tail_mean','tail_rms')]:
  contrasts.append(dict(dataset=ds,contrast=f'{b} minus {a}',**{f'delta_{k}':float(d.loc[b,k]-d.loc[a,k]) for k in ['auroc','fpr','tpr','pauc_step_normalized_0.05','pauc_step_normalized_0.1','pauc_linear_normalized_0.05','pauc_linear_normalized_0.1']}))
pd.DataFrame(contrasts).to_csv(OUT/'pooling_contrasts.csv',index=False)
if '--compute-only' not in sys.argv:
 fig,axes=plt.subplots(1,3,figsize=(7,2.55),sharey=True)
 for ax,(ds,title) in zip(axes,[('voraus','voraus-AD (419 normal)'),('ur3','UR3 (12 normal)'),('factory','FactoryNet (663 normal)')]):
  for j,(x,t,r) in enumerate(plots[ds]):
   ax.step(x,t,where='post',color=COLORS[j],linestyle=STYLES[j],lw=1.5,label=LABELS[j]);ax.scatter(r['fpr'],r['tpr'],marker='o',s=25,facecolors='none',edgecolors=COLORS[j],zorder=5)
  ax.axvline(.05,color='.7',lw=.6);ax.set(xlim=(-.002,.15),ylim=(0,1),xticks=[0,.05,.1,.15],xticklabels=['0','5','10','15'],xlabel='FPR budget (%)',title=title);ax.grid(axis='y',alpha=.16)
  if ds=='ur3':ax.axvline(1/12,color='.7',lw=.7,ls=':');ax.text(.09,.08,'1/12',fontsize=9.2)
 axes[0].set_ylabel('Recall');fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=5,frameon=False,columnspacing=1.1,handlelength=1.9);fig.subplots_adjust(left=.075,right=.99,top=.85,bottom=.32,wspace=.2);fig.savefig(FIG/'low_fpr_roc.pdf');fig.savefig(FIG/'low_fpr_roc.png',dpi=180);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(7,2.6),sharey=True)
 for ax,(ds,title) in zip(axes,[('voraus','voraus-AD'),('ur3','UR3'),('factory','FactoryNet')]):
  for pool,stage,v in normalplot[ds]:ax.step(v,np.arange(1,len(v)+1)/len(v),where='post',color=COLORS[0 if pool=='mean' else 4],ls='-' if stage=='cal' else '--',label=f'{"Mean" if pool=="mean" else "Tail RMS"} {stage.replace("_normal", "")}')
  ax.set(xscale='log',xlabel='Score / calibration threshold',title=title,ylim=(0,1));ax.axvline(1,color='.5',lw=.7)
 axes[0].set_ylabel('Normal-score ECDF');fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=4,frameon=False);fig.subplots_adjust(left=.08,right=.99,bottom=.33,top=.85,wspace=.2);fig.savefig(FIG/'normal_score_ecdf.pdf');plt.close(fig)
lines=[]
for method in dict.fromkeys(r['method'] for r in cats):
 rr=[r for r in cats if r['method']==method];lines.append(method+' & '+' & '.join(f"{r['detected']}/{r['support']}" for r in rr)+f" & {rr[0]['false_alarms']}/{rr[0]['normal_support']} \\\\")
(OUT/'ur3_category_rows.tex').write_text('\n'.join(lines)+'\n')
(OUT/'pooling_rows.tex').write_text('\n'.join(r['dataset']+' / '+r['pool'].replace('_',' ')+' & '+f"{r['auroc']:.3f} & {r['pauc_step_normalized_0.05']:.3f} & {r['pauc_step_normalized_0.1']:.3f} & {r['detected']}/{r['n_fault']} & {r['false_alarms']}/{r['n_normal']} \\\\" for r in allrows)+'\n')
manifest={'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'completed','command':sys.argv,'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__ if '--compute-only' not in sys.argv else 'not imported','protocol_sha256':hashlib.sha256((OUT/'POSTHOC_PROTOCOL.json').read_bytes()).hexdigest(),'outputs':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for folder in [OUT,FIG] for f in folder.iterdir() if f.is_file() and f.name != 'run_manifest.json' and f.suffix != '.log'}}
(OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2));print(df.to_string(index=False));print(pd.DataFrame(contrasts).to_string(index=False));print(pd.DataFrame(cats).to_string(index=False))
