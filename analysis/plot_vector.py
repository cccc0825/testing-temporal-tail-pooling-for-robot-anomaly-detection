"""Vector rendering of frozen-score diagnostics.
Uses reportlab, reading only generated frozen-score CSVs. Fonts 9.2-10 pt at 7in.
"""
from pathlib import Path
import csv,math,os
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor,Color
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
ROOT=Path(__file__).resolve().parent.parent;A=ROOT/'analysis';F=ROOT/'figures';F.mkdir(exist_ok=True)
# Use embedded locally available TrueType fonts.
candidates=[os.environ.get('DIAGNOSTIC_FONT',''),'/System/Library/Fonts/Supplemental/Arial.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','/usr/share/fonts/dejavu/DejaVuSans.ttf']
FONT=next((Path(p) for p in candidates if p and Path(p).is_file()),None)
if FONT is None: raise RuntimeError('Set DIAGNOSTIC_FONT to a readable TrueType font, e.g. DejaVuSans.ttf')
pdfmetrics.registerFont(TTFont('PlotSans',str(FONT)))
POOLS=['mean','rms','max','tail_mean','tail_rms'];LABELS=['Mean','All RMS','Max','Tail mean','Tail RMS'];COLORS=['#0072B2','#009E73','#777777','#CC79A7','#D55E00'];DASH=[[],[5,2],[1,2],[5,2,1,2],[3,1,1,1]]
metrics=list(csv.DictReader(open(A/'pooling_metrics.csv')));vertices=list(csv.DictReader(open(A/'roc_vertices.csv')))
c=canvas.Canvas(str(F/'low_fpr_roc.pdf'),pagesize=(504,181),invariant=1,initialFontName="PlotSans");c.setTitle('Observed recall versus FPR budget; frozen primary pooling scores')
def text(x,y,s,size=9.2,align='left'):
 c.setFillColorRGB(0,0,0);c.setFont('PlotSans',size);getattr(c,{'left':'drawString','center':'drawCentredString','right':'drawRightString'}[align])(x,y,s)
for panel,(ds,title) in enumerate([('voraus','voraus-AD (419 normal)'),('ur3','UR3 (12 normal)'),('factory','FactoryNet (663 normal)')]):
 left=34+panel*157;bottom=47;w=145;h=112
 X=lambda v:left+w*v/.15;Y=lambda v:bottom+h*v
 text(left+w/2,169,title,9.5,'center')
 c.setStrokeColorRGB(.8,.8,.8);c.setLineWidth(.35);c.setDash()
 for v in [0,.25,.5,.75,1]:
  c.line(left,Y(v),left+w,Y(v))
  if panel==0:text(left-4,Y(v)-3,f'{v:g}',9.2,'right')
 for v in [0,.05,.1,.15]:text(X(v),bottom-12,str(round(v*100)),9.2,'center')
 text(left+w/2,bottom-26,'FPR budget (%)',9.2,'center')
 c.setStrokeColorRGB(.65,.65,.65)
 for guide in [.05,.10]:c.line(X(guide),bottom,X(guide),bottom+h)
 if ds=='ur3':c.setDash([1,2]);c.line(X(1/12),bottom,X(1/12),bottom+h);text(X(1/12)+3,bottom+7,'1/12',9.2)
 c.setStrokeColorRGB(0,0,0);c.setDash();c.line(left,bottom,left,bottom+h);c.line(left,bottom,left+w,bottom)
 for j,pool in enumerate(POOLS):
  rr=[v for v in vertices if v['dataset']==ds and v['pool']==pool];d={}
  for r in rr:
   x=float(r['fpr']);d[x]=max(d.get(x,0),float(r['tpr']))
  xy=sorted(d.items());points=[]
  for k,(x,y) in enumerate(xy):
   if x>.15:break
   points.append((X(x),Y(y)));end=min(.15,xy[k+1][0] if k+1<len(xy) else .15);points.append((X(end),Y(y)))
  c.setStrokeColor(HexColor(COLORS[j]));c.setLineWidth(1.1);c.setDash(DASH[j]);path=c.beginPath();path.moveTo(*points[0])
  for z in points[1:]:path.lineTo(*z)
  c.drawPath(path);r=next(r for r in metrics if r['dataset']==ds and r['pool']==pool);c.setDash();c.circle(X(float(r['fpr'])),Y(float(r['tpr'])),2.6,stroke=1,fill=0)
c.saveState();c.translate(9,103);c.rotate(90);text(0,0,'Recall',9.5,'center');c.restoreState()
for j,label in enumerate(LABELS):
 x=30+j*96;c.setStrokeColor(HexColor(COLORS[j]));c.setDash(DASH[j]);c.setLineWidth(1.1);c.line(x,9,x+20,9);text(x+24,6,label,9.2)
c.save()
print('Wrote low_fpr_roc.pdf: 504 x 181 pt; embedded TrueType font; minimum font 9.2 pt.')
# Descriptive held-out nominal ECDFs; exclude fitting self-matches.
import numpy as np,json
SRC=ROOT/'inputs'
c=canvas.Canvas(str(F/'normal_score_ecdf.pdf'),pagesize=(504,181),invariant=1,initialFontName="PlotSans")
for panel,(ds,title) in enumerate([('voraus','voraus-AD'),('ur3','UR3'),('factory','FactoryNet')]):
 left=34+panel*157;bottom=47;w=145;h=112
 X=lambda v:left+w*(math.log10(v)+2)/4;Y=lambda v:bottom+h*v
 text(left+w/2,169,title,9.5,'center');c.setDash();c.setStrokeColorRGB(.8,.8,.8);c.setLineWidth(.35)
 for v in [0,.25,.5,.75,1]:
  c.line(left,Y(v),left+w,Y(v))
  if panel==0:text(left-4,Y(v)-3,f'{v:g}',9.2,'right')
 for v in [.01,.1,1,10,100]:text(X(v),bottom-12,str(v).removesuffix('.0'),9.2,'center')
 text(left+w/2,bottom-26,'Score / cal. threshold',9.2,'center');c.line(X(1),bottom,X(1),bottom+h)
 c.setStrokeColorRGB(0,0,0);c.line(left,bottom,left,bottom+h);c.line(left,bottom,left+w,bottom)
 if ds=='factory':prep=SRC/'new_external/prepared';res=SRC/'new_external/results/factory_pickplace_v1';rf='rows.json';sf='split.json';nf='scores.npz'
 else:prep=SRC/'existing_probe/prepared';res=SRC/'existing_probe/results/existing_probe_v1';rf=f'{ds}_rows.json';sf=f'{ds}_split.json';nf=f'{ds}_scores.npz'
 rows=json.loads((prep/rf).read_text());split=json.loads((prep/sf).read_text());scores=np.load(res/nf)
 for pool,color in [('mean',COLORS[0]),('tail_rms',COLORS[4])]:
  th=float(next(r['threshold'] for r in metrics if r['dataset']==ds and r['pool']==pool))
  for stage in ['cal','test']:
   ix=[i for i in split[stage] if rows[i]['anomaly']==0];v=np.sort(scores['nn_elapsed128_l1_'+pool][ix]/th);assert np.all((v>=.01)&(v<=100))
   c.setStrokeColor(HexColor(color));c.setDash([] if stage=='cal' else [4,2]);c.setLineWidth(1.1);path=c.beginPath();path.moveTo(X(.01),Y(0))
   for j,z in enumerate(v):path.lineTo(X(z),Y(j/len(v)));path.lineTo(X(z),Y((j+1)/len(v)))
   path.lineTo(X(100),Y(1));c.drawPath(path)
c.saveState();c.translate(9,103);c.rotate(90);text(0,0,'Normal-score ECDF',9.5,'center');c.restoreState()
for j,(label,color,dash) in enumerate([('Mean cal',COLORS[0],[]),('Mean test',COLORS[0],[4,2]),('Tail RMS cal',COLORS[4],[]),('Tail RMS test',COLORS[4],[4,2])]):
 x=22+j*124;c.setStrokeColor(HexColor(color));c.setDash(dash);c.line(x,9,x+18,9);text(x+22,6,label,9.2)
c.save();print('Wrote normal_score_ecdf.pdf: log range .01 to 100 contains every score.')
