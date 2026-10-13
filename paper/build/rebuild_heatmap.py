from pathlib import Path
import json,hashlib,os
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor,Color
from pdf2image import convert_from_path
R=Path(__file__).resolve().parents[2];P=R/"paper/figures"
# Prefer the fully recorded S26 grid; fall back to the historical aggregate grid.
D=R/"results/revision_20260925/grid_recorded/summary.json"
if D.exists():
 d=json.loads(D.read_text());cs={(c["pr_target"],c["n"]):{"rej_M":c["M"]["rate"],"rej_C":c["C"]["rate"]} for c in d["cells"].values()}
else:
 D=R/"results/overnight/e1_grid.json";d=json.loads(D.read_text());cs={(c["pr_target"],c["n"]):c for c in d["cells"].values()}
ps=sorted({p for p,n in cs});ns=sorted({n for p,n in cs})
c=canvas.Canvas(str(P/"Figure4.pdf"),pagesize=(960,450));a=HexColor("#FFF8FB");b=HexColor("#E7A8C2");ink=HexColor("#30272C");nom=HexColor("#806F78")
def col(v):return Color(a.red+(b.red-a.red)*v,a.green+(b.green-a.green)*v,a.blue+(b.blue-a.blue)*v)
def tx(x,y,t,z=11,al="left"):
 c.setFillColor(ink);c.setFont("Helvetica",z);getattr(c,{"left":"drawString","center":"drawCentredString","right":"drawRightString"}[al])(x,y,str(t))
g=[]
for k,arm in enumerate(("M","C")):
 x=65+k*435;y=80;cw=72;ch=49;v=[[cs[p,n]["rej_"+arm] for n in ns] for p in ps];g.append(v)
 tx(x,392,arm+" reference",17)
 for i,p in enumerate(ps):
  tx(x-9,y+(i+.5)*ch-4,f"{p:g}",12,"right")
  for j,n in enumerate(ns):
   c.setFillColor(col(v[i][j]));c.rect(x+j*cw,y+i*ch,cw,ch,fill=1,stroke=0);tx(x+(j+.5)*cw,y+(i+.5)*ch-4,f"{v[i][j]:.3f}",12,"center")
 for j,n in enumerate(ns):tx(x+(j+.5)*cw,y-18,n,12,"center")
 tx(x+2.5*cw,y-39,"sample size n",13,"center")
 c.setStrokeColor(nom);c.setLineWidth(2)
 for i in range(len(ps)-1):
  for j in range(len(ns)-1):
   q=[(j,i),(j+1,i),(j+1,i+1),(j,i+1)];e=[]
   for u,z in zip(q,q[1:]+q[:1]):
    f,h=v[u[1]][u[0]],v[z[1]][z[0]]
    if (f<.05)!=(h<.05):
     t=(.05-f)/(h-f);e.append((x+(u[0]+t*(z[0]-u[0])+.5)*cw,y+(u[1]+t*(z[1]-u[1])+.5)*ch))
   if len(e)==2:c.line(*e[0],*e[1])
   elif len(e)==4:c.line(*e[0],*e[1]);c.line(*e[2],*e[3])
 c.setStrokeColor(ink);c.setLineWidth(2);c.rect(x,y+ps.index(4.5)*ch,cw,ch,fill=0,stroke=1)
c.saveState();c.translate(19,227);c.rotate(90);tx(0,0,"target normalized participation ratio",13,"center");c.restoreState()
for i in range(200):c.setFillColor(col(i/199));c.rect(885,80+i*294/200,14,294/200+.1,stroke=0,fill=1)
for v in (0,.2,.4,.6,.8,1):tx(903,76+294*v,f"{v:.1f}",11)
c.saveState();c.translate(945,227);c.rotate(90);tx(0,0,"rejection rate",13,"center");c.restoreState()
c.setStrokeColor(nom);c.setLineWidth(2);c.line(188,15,212,15);tx(220,11,"nominal 0.05 contour",11)
c.setStrokeColor(ink);c.rect(383,9,18,12,fill=0,stroke=1);tx(410,11,"grid cell nearest the fibromyalgia embedding (target PR 4.5, n = 28)",11)
c.save();convert_from_path(P/"Figure4.pdf",dpi=240,poppler_path=os.environ.get("POPPLER_PATH") or None)[0].save(P/"Figure4.png")
(R/"reproducibility/audits/baby_heatmap.json").write_text(json.dumps({"source":D.relative_to(R).as_posix(),"sha256":hashlib.sha256(D.read_bytes()).hexdigest(),"pr":ps,"n":ns,"M":g[0],"C":g[1],"cells":60,"palette":["FFF8FB","E7A8C2"],"contour":.05},indent=2)+'\n')
print("Figure 4 rebuilt from",D.relative_to(R).as_posix())
