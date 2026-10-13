#Figs 3 to 5 at 180 mm width (text at least 8 pt), from the stored summaries
from pathlib import Path
import json,os
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from pdf2image import convert_from_path
R=Path(__file__).resolve().parents[2]
O=R/"paper/figures"
P=os.environ.get("POPPLER_PATH") or None
C=["#B87593","#D18EAA","#E7A8C2","#F4BCD1"]
INK="#222222";GRID="#E3E0E1";AXIS="#555555"
A=("W","D4","K","E")
CENTERED=("tri","triuneq","pairuneq","tetra")
W=510

def load(p):
    return json.loads((R/p).read_text(encoding="utf-8"))

def text(c,x,y,s,n=9,align="left"):
    c.setFillColor(HexColor(INK));c.setFont("Helvetica",n)
    getattr(c,{"left":"drawString","center":"drawCentredString","right":"drawRightString"}[align])(x,y,str(s))

def whisker(c,x,lo,hi,col=AXIS):
    c.setStrokeColor(HexColor(col));c.setLineWidth(1)
    c.line(x,lo,x,hi);c.line(x-2.5,lo,x+2.5,lo);c.line(x-2.5,hi,x+2.5,hi)

def frame(c,x,y,w,h,top,ticks,title,pct=True,nominal=True):
    text(c,x,y+h+8,title,9)
    for z in ticks:
        yy=y+h*z/top
        c.setStrokeColor(HexColor(GRID));c.setLineWidth(.5);c.line(x,yy,x+w,yy)
        text(c,x-4,yy-3,f"{100*z:g}" if pct else f"{z:g}",8,"right")
    c.setStrokeColor(HexColor(AXIS));c.setLineWidth(.8);c.line(x,y,x,y+h);c.line(x,y,x+w,y)
    if nominal:
        c.setDash(3,3);c.line(x,y+h*.05/top,x+w,y+h*.05/top);c.setDash()

def done(c,name):
    c.save()
    convert_from_path(O/f"{name}.pdf",dpi=300,poppler_path=P)[0].save(O/f"{name}.png")

def mechanism():
    s=load("results/revision_20260926/mechanism_control/summary.json")
    c=canvas.Canvas(str(O/"Main_Figure2.pdf"),pagesize=(W,250))
    arms=[("M_fit","M","fitted",3),("C_fit","C","fitted",1),("M_H","M","moment H",3),("C_H","C","moment H",1),("C_genlaw","C","generating law",0)]
    titles={"cell1":"Exponential spectrum, n = 28","cell3":"Two-block spectrum, n = 28","cell6":"Two-block spectrum, n = 180"}
    for j,cell in enumerate(("cell1","cell3","cell6")):
        x=34+j*161;y=62;w=142;h=150
        frame(c,x,y,w,h,1.,[0,.25,.5,.75,1],f"{chr(65+j)}  {titles[cell]}")
        for i,(a,lab,kind,col) in enumerate(arms):
            v=s["cells"][cell]["arms"][a];xx=x+16+i*27.5
            c.setFillColor(HexColor(C[col]));c.rect(xx-9,y,18,h*v["rate"],fill=1,stroke=0)
            whisker(c,xx,y+h*v["wilson95"][0],y+h*v["wilson95"][1])
            text(c,xx,y-11,lab,8,"center")
        text(c,x+16+27.5*.5,y-22,"fitted",8,"center");text(c,x+16+27.5*2.5,y-22,"moment H",8,"center");text(c,x+16+27.5*4,y-22,"law",8,"center")
    text(c,34,14,"Percent of null datasets rejected at p < 0.05; 400 datasets per cell; Wilson 95% intervals; dashed line 5%.",8)
    done(c,"Main_Figure2")

def embedding():
    h=load("results/revision_20260924/headline/summary.json")["summary"]
    c=canvas.Canvas(str(O/"Main_Figure3.pdf"),pagesize=(W,230))
    x,y,w,ht=40,50,190,140
    frame(c,x,y,w,ht,.25,[0,.05,.1,.15,.2,.25],"A  Rejection of null datasets (%)")
    for j,a in enumerate(("M","C")):
        v=h[a];xx=x+55+80*j
        c.setFillColor(HexColor(C[3-2*j]));c.rect(xx-18,y,36,ht*v["rate"]/.25,fill=1,stroke=0)
        whisker(c,xx,y+ht*v["wilson95"][0]/.25,y+ht*v["wilson95"][1]/.25)
        text(c,xx,y+ht*v["wilson95"][1]/.25+5,f"{100*v['rate']:.2f}%",8,"center");text(c,xx,y-11,a,9,"center")
    x=300;w=180
    top=max(2000,100*((h["M_only"]+99)//100))
    frame(c,x,y,w,ht,top,list(range(0,top+1,500)),"B  Paired discordant datasets",pct=False,nominal=False)
    for j,a in enumerate(("M_only","C_only")):
        v=h[a];xx=x+50+80*j
        c.setFillColor(HexColor(C[3-2*j]));c.rect(xx-18,y,36,ht*v/top,fill=1,stroke=0)
        text(c,xx,y+ht*v/top+5,f"{v:,}",8,"center");text(c,xx,y-11,a.replace("_"," "),9,"center")
    text(c,40,14,f"{h['n']:,} datasets; Wilson 95% intervals; dashed line 5%.",8)
    done(c,"Main_Figure3")

def by_n():
    src={28:"results/revision_20260924/scope_detection_n28",50:"results/revision_20260925/sample_size_n50",100:"results/revision_20260924/scope_detection_n100",200:"results/revision_20260925/sample_size_n200"}
    out={}
    for n,p in src.items():
        cells=load(p+"/summary.json")["cells"]
        for a in A:
            cen=max(((k,v) for k,v in cells.items() if k.split("_",2)[2][:-1] in CENTERED),key=lambda kv:kv[1]["arms"][a]["calibrated_rate"])
            sh=max(((k,v) for k,v in cells.items() if k.split("_",2)[2]=="shift4"),key=lambda kv:kv[1]["arms"][a]["calibrated_rate"])
            out[n,a]=dict(centered=cen[1]["arms"][a],centered_cell=cen[0],shift=sh[1]["arms"][a],shift_cell=sh[0])
    return out

def angular():
    b=by_n();ns=(28,50,100,200)
    c=canvas.Canvas(str(O/"Main_Figure4.pdf"),pagesize=(W,380))
    for k,a in enumerate(A):
        x=40+(k%2)*245;y=222-(k//2)*170;w=205;h=115
        frame(c,x,y,w,h,1.,[0,.25,.5,.75,1],f"{chr(65+k)}  {a}")
        xs=[x+22+i*54 for i in range(4)]
        for series,col,dx in (("shift",C[2],4),("centered",C[0],-4)):
            pts=[(xx+dx,y+h*b[n,a][series]["calibrated_rate"]) for xx,n in zip(xs,ns)]
            c.setStrokeColor(HexColor(col));c.setLineWidth(1.2)
            for p1,p2 in zip(pts,pts[1:]):
                c.line(*p1,*p2)
            for (xx,yy),n in zip(pts,ns):
                lo,hi=b[n,a][series]["two_stage_bootstrap95"]
                whisker(c,xx,y+h*lo,y+h*hi,col)
                c.setFillColor(HexColor(col))
                if series=="centered":
                    c.circle(xx,yy,3,fill=1,stroke=0)
                else:
                    c.rect(xx-3,yy-3,6,6,fill=1,stroke=0)
        for xx,n in zip(xs,ns):
            text(c,xx,y-11,f"n = {n}",8,"center")
    c.setFillColor(HexColor(C[0]));c.circle(46,20,3,fill=1,stroke=0);text(c,54,17,"largest detection of a centered mixture",8)
    c.setFillColor(HexColor(C[2]));c.rect(243,17,6,6,fill=1,stroke=0);text(c,254,17,"largest detection of the shift-only control",8)
    text(c,40,4,"Calibrated detection (%); two-stage bootstrap 95% intervals of the cell giving the maximum; dashed line 5%.",8)
    done(c,"Main_Figure4")

if __name__=="__main__":
    mechanism();embedding();angular()
