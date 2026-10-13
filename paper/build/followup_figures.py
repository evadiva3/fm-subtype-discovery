from pathlib import Path
import json,os
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from pdf2image import convert_from_path
R=Path(__file__).resolve().parents[2]
O=R/"paper/figures"
P=os.environ.get("POPPLER_PATH") or None
C=["#B87593","#D18EAA","#E7A8C2","#F4BCD1"]
A=("W","D4","K","E")

def text(c,x,y,s,n=10,align="left"):
    c.setFillColor(HexColor("#222222"))
    c.setFont("Helvetica",n)
    getattr(c,{"left":"drawString","center":"drawCentredString","right":"drawRightString"}[align])(x,y,str(s))

def dot(c,x,y,lo,hi,col):
    c.setStrokeColor(HexColor(col))
    c.setLineWidth(2)
    c.line(x,lo,x,hi)
    c.line(x-3,lo,x+3,lo)
    c.line(x-3,hi,x+3,hi)
    c.setFillColor(HexColor(col))
    k=C.index(col) if col in C else 0
    if k==0:
        c.circle(x,y,3,fill=1,stroke=0)
    elif k==1:
        c.rect(x-3,y-3,6,6,fill=1,stroke=0)
    else:
        p=c.beginPath()
        pts=[(x,y+4),(x-4,y-3),(x+4,y-3)] if k==2 else [(x,y+4),(x-4,y),(x,y-4),(x+4,y)]
        p.moveTo(*pts[0])
        for pt in pts[1:]:
            p.lineTo(*pt)
        p.close()
        c.drawPath(p,fill=1,stroke=0)

def axes(c,x,y,w,h,top,title,ticks):
    text(c,x,y+h+14,title,11)
    for z in ticks:
        yy=y+h*z/top
        c.setStrokeColor(HexColor("#E3E0E1"))
        c.setLineWidth(.5)
        c.line(x,yy,x+w,yy)
        text(c,x-7,yy-3,f"{100*z:g}",9,"right")
    c.setStrokeColor(HexColor("#555555"))
    c.line(x,y,x,y+h)
    c.line(x,y,x+w,y)
    c.setDash(3,3)
    c.line(x,y+.05*h/top,x+w,y+.05*h/top)
    c.setDash()

def done(c,i):
    c.save()
    convert_from_path(O/f"Figure{i}.pdf",dpi=200,poppler_path=P)[0].save(O/f"Figure{i}.png")

def legend(c,x,y):
    for i,a in enumerate(A):
        dot(c,x+75*i,y,y,y,C[i])
        text(c,x+10+75*i,y-3,a,11)

def fig3(h):
    c=canvas.Canvas(str(O/"Figure3.pdf"),pagesize=(900,340))
    axes(c,48,62,205,215,.25,"A  Embedding-geometry study",[0,.05,.1,.15,.2,.25])
    for j,a in enumerate(("M","C")):
        v=h[a]
        xx=100+100*j
        c.setFillColor(HexColor(C[3-j*2]))
        c.rect(xx-20,62,40,215*v["rate"]/.25,fill=1,stroke=0)
        dot(c,xx,62+215*v["rate"]/.25,62+215*v["wilson95"][0]/.25,62+215*v["wilson95"][1]/.25,"#555555")
        text(c,xx,62+215*v["wilson95"][1]/.25+9,f"{100*v["rate"]:.2f}%",10,"center")
        text(c,xx,43,a,11,"center")
    text(c,48,21,"10,000 datasets; 95% Wilson intervals",9)
    x,y,w,ht=330,62,200,215
    top=max(2000,100*((h["M_only"]+99)//100))
    text(c,x,y+ht+14,"B  Paired discordance",11)
    for v in range(0,top+1,500):
        yy=y+ht*v/top
        c.setStrokeColor(HexColor("#E3E0E1"))
        c.line(x,yy,x+w,yy)
        text(c,x-6,yy-3,str(v),9,"right")
    for j,a in enumerate(("M_only","C_only")):
        xx=375+j*90
        v=h[a]
        c.setFillColor(HexColor(C[3-j*2]))
        c.rect(xx-20,y,40,ht*v/top,fill=1,stroke=0)
        text(c,xx,y+ht*v/top+8,v,10,"center")
        text(c,xx,43,a.replace("_"," "),10,"center")
    axes(c,600,62,280,215,.55,"C  Centered-law control",[0,.1,.2,.3,.4,.5])
    vals=[(.1075,.0808,.1417,3,"M"),(.015,.0069,.0323,1,"C"),(.4275,.3799,.4764,3,"M"),(.25,.2101,.2947,1,"C"),(.0575,.0386,.0848,1,"C")]
    for j,(v,lo,hi,k,a) in enumerate(vals):
        xx=628+j*51
        c.setFillColor(HexColor(C[k]))
        c.rect(xx-15,62,30,215*v/.55,fill=1,stroke=0)
        dot(c,xx,62+215*v/.55,62+215*lo/.55,62+215*hi/.55,"#555555")
        text(c,xx,62+215*hi/.55+7,f"{100*v:.2f}",8,"center")
        text(c,xx,43,a,9,"center")
    for x,s in [(653,"Fitted"),(755,"Moment H"),(842,"Generating law")]:
        text(c,x,28,s,8,"center")
    text(c,600,12,"Percent rejected; 400 datasets; Table 8 cell 1",9)
    done(c,3)

def fig7(d):
    c=canvas.Canvas(str(O/"Figure7.pdf"),pagesize=(760,335))
    for s in range(2):
        x=52+s*370
        axes(c,x,74,310,210,.12,["A  Two-block spectrum","B  Exponential spectrum"][s],[0,.03,.05,.07,.1,.12])
        for j,q in enumerate((5,10)):
            cell=d["cells"][f"{s}_q{q}_null"]
            for k,a in enumerate(A):
                z=cell["arms"][a]
                xx=x+55+160*j+(k-1.5)*15
                lo,hi=z["conditional_wilson95"]
                dot(c,xx,74+210*z["calibrated_rate"]/.12,74+210*lo/.12,74+210*hi/.12,C[k])
            text(c,x+55+160*j,53,f"q = {q}",10,"center")
    legend(c,240,24)
    done(c,7)

#Panels A-D: S19 origin-symmetric mixtures. Panels E-H: S20 alternatives at half-separation 4.
def fig8(d,e=None):
    c=canvas.Canvas(str(O/"Figure8.pdf"),pagesize=(760,905))
    ph=140
    blocks=[(d,("leading2","leading4","minor2","minor4"),lambda i:("Leading " if i<2 else "Minor ")+str(2 if i%2==0 else 4),0,(715,520))]
    if e is not None:
        blocks.append((e,("antipodal4","tri4","pair4","shift4"),lambda i:("Antipodal","Three-group","Pair","Shift only")[i],4,(290,95)))
    for src,gs,lab,off,ys in blocks:
        # Scale S19 to its observed interval range so near-null rates remain readable.
        peak=max(src["cells"][f"{s}_q{q}_{g}"]["arms"][a]["two_stage_bootstrap95"][1] for s in range(2) for q in (5,10) for g in gs for a in A)
        top,ticks=(.2,[0,.05,.1,.15,.2]) if peak<=.2 else (1.,[0,.25,.5,.75,1])
        for s in range(2):
            for j,q in enumerate((5,10)):
                x=52+j*375
                y=ys[s]
                axes(c,x,y,305,ph,top,f"{chr(65+off+s*2+j)}  "+("Two-block" if s==0 else "Exponential")+f", q = {q}",ticks)
                for i,g in enumerate(gs):
                    cell=src["cells"][f"{s}_q{q}_{g}"]
                    for k,a in enumerate(A):
                        z=cell["arms"][a]
                        lo,hi=z["two_stage_bootstrap95"]
                        xx=x+35+i*77+(k-1.5)*10
                        dot(c,xx,y+ph*z["calibrated_rate"]/top,y+ph*lo/top,y+ph*hi/top,C[k])
                    text(c,x+35+i*77,y-17,lab(i),9,"center")
    text(c,52,885,"S19: origin-symmetric two-group mixtures",11)
    if e is not None:
        text(c,52,470,"S20: alternatives at half-separation 4",11)
    legend(c,240,24)
    done(c,8)

if __name__=="__main__":
    h=json.loads((R/"results/revision_20260924/headline/summary.json").read_text(encoding="utf-8"))["summary"]
    d=json.loads((R/"results/revision_20260924/calibrated_detection/summary.json").read_text(encoding="utf-8"))
    ep=R/"results/revision_20260924/asymmetric_detection/summary.json"
    e=json.loads(ep.read_text(encoding="utf-8")) if ep.exists() else None
    fig3(h)
    fig7(d)
    fig8(d,e)
