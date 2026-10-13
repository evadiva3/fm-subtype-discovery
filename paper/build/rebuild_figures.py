#rebuild Figure2, Figure3 and Figure9 from stored numbers, not reruns
from pathlib import Path
import math
import os
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor,black,white
from pdf2image import convert_from_path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"paper/figures";OUT.mkdir(parents=True,exist_ok=True)
POP=os.environ.get("POPPLER_PATH") or None
M=HexColor("#F4BCD1");C=HexColor("#D18EAA");GRID=HexColor("#DDDDDD")

def label(c,x,y,s,size=10,align="left"):
    c.setFillColor(black);c.setFont("Helvetica",size)
    getattr(c,{"left":"drawString","center":"drawCentredString","right":"drawRightString"}[align])(x,y,str(s))

def axes(c,x,y,w,h,ymax,ticks,title,nominal=None):
    for v in ticks:
        yy=y+h*v/ymax;c.setStrokeColor(GRID);c.setLineWidth(.5);c.line(x,yy,x+w,yy)
        label(c,x-6,yy-3,f"{v:g}",9,"right")
    c.setStrokeColor(black);c.line(x,y,x,y+h);c.line(x,y,x+w,y)
    label(c,x,y+h+14,title,12)
    if nominal is not None:
        c.setDash(4,3);c.setStrokeColor(HexColor("#555555"));c.line(x,y+h*nominal/ymax,x+w,y+h*nominal/ymax);c.setDash()

def bar(c,x,y,w,h,color):
    c.setFillColor(color);c.rect(x,y,w,h,fill=1,stroke=0)

def error(c,x,y,lo,hi,scale):
    c.setStrokeColor(black);c.setLineWidth(1);c.line(x,y+lo*scale,x,y+hi*scale)
    for v in (lo,hi):c.line(x-3,y+v*scale,x+3,y+v*scale)

def done(c,f):
    c.save();im=convert_from_path(str(OUT/f"Figure{f}.pdf"),dpi=180,poppler_path=POP)[0];im.save(OUT/f"Figure{f}.png")

def fig3():
    if (ROOT/"results/revision_20260924/headline/summary.json").exists():
        import json
        from followup_figures import fig3 as fresh
        fresh(json.loads((ROOT/"results/revision_20260924/headline/summary.json").read_text(encoding="utf-8"))["summary"])
        return
    c=canvas.Canvas(str(OUT/"Figure3.pdf"),pagesize=(900,355))
    axes(c,48,65,210,230,.23,[0,.05,.10,.15,.20],"A  Headline calibration",.05)
    for x,v,lo,hi,col,arm in [(80,.1937,.1861,.2016,M,"M"),(180,.0072,.0057,.0091,C,"C")]:
        bar(c,x,65,40,230*v/.23,col);error(c,x+20,65,lo,hi,230/.23);label(c,x+20,65+230*hi/.23+8,f"{v:.4f}",10,"center");label(c,x+20,48,arm,11,"center")
    label(c,48,24,"10,000 datasets; Wilson 95% intervals",9)
    axes(c,335,65,200,230,2000,[0,500,1000,1500,2000],"B  Paired discordance")
    for x,v,col,arm in [(365,1865,M,"M only"),(455,0,C,"C only")]:
        bar(c,x,65,40,230*v/2000,col);label(c,x+20,65+230*v/2000+8,str(v),10,"center");label(c,x+20,48,arm,10,"center")
    axes(c,605,65,275,230,.55,[0,.1,.2,.3,.4,.5],"C  Centered covariance control",.05)
    data=[(.1075,.0808,.1417,M,"M"),(.015,.0069,.0323,C,"C"),(.4275,.3799,.4764,M,"M"),(.25,.2101,.2947,C,"C"),(.0575,.0386,.0848,C,"C")]
    for i,(v,lo,hi,col,arm) in enumerate(data):
        x=619+i*51;bar(c,x,65,30,230*v/.55,col);error(c,x+15,65,lo,hi,230/.55);label(c,x+15,65+230*hi/.55+7,f"{v:.4f}",8,"center");label(c,x+15,48,arm,10,"center")
    for x,t in [(658,"Fitted"),(760,"Moment H"),(838,"Generating law")]:label(c,x,30,t,8,"center")
    label(c,605,14,"400 datasets; cell 1 of Table 8",9)
    done(c,3)

def fig9():
    c=canvas.Canvas(str(OUT/"Figure9.pdf"),pagesize=(850,340))
    axes(c,45,65,465,220,1,[0,.2,.4,.6,.8,1],"A  Monte Carlo reference p-values",.05)
    a=[.1685,.463,.4033,.7009,.9062];b=[.0508,.154,.2782,.3843,.8124]
    for i,arm in enumerate(["M","G","S","C","P"]):
        x=63+89*i;bar(c,x,65,28,a[i]*220,M);bar(c,x+29,65,28,b[i]*220,C);label(c,x+28,47,arm,11,"center")
    bar(c,55,21,12,9,M);label(c,73,21,"Fibromyalgia",10);bar(c,185,21,12,9,C);label(c,203,21,"Depression",10)
    x=610;y=80;w=195;h=180
    label(c,568,299,"B  Reported retraining variance share",12)
    c.setStrokeColor(GRID)
    for v in [0,.25,.5,.75,1]:c.line(x+w*v,y,x+w*v,y+h);label(c,x+w*v,y-15,f"{v:g}",9,"center")
    label(c,x-8,125,"FM",10,"right");label(c,x-8,215,"MDD",10,"right")
    bar(c,x,110,w*.8,35,M);label(c,x+w*.8+6,121,"0.80",10)
    label(c,x+3,212,"< 0.01",10)
    label(c,710,35,"Fraction of reference variance",10,"center")
    done(c,9)

def fig2():
    import json
    #e15 spike sweep, rounded as in the reported table
    cs=list(json.loads((ROOT/"results/overnight/e15_boundary.json").read_text())["cells"].values())
    hs=[e["spike_S"] for e in cs]
    shift=[round(e["shift_in_sd"],2) for e in cs]
    c=canvas.Canvas(str(OUT/"Figure2.pdf"),pagesize=(700,560))
    x=65;w=600
    def xp(v):return x+w*math.log(v)/math.log(160)
    def line(vals,y,h,low,high,color):
        c.setStrokeColor(color);c.setLineWidth(1.4)
        pts=[(xp(v),y+h*(z-low)/(high-low)) for v,z in zip(hs,vals)]
        for p,q in zip(pts,pts[1:]):c.line(*p,*q)
        c.setFillColor(color)
        for p in pts:c.circle(*p,2,stroke=0,fill=1)
    specs=[(385,120,-.2,.3,[-.2,0,.2],"A  Re-projection shift in surrogate standard deviations"),(215,120,0,.8,[0,.2,.4,.6,.8],"B  Rejection rate"),(45,120,0,90,[0,30,60,90],"C  Discordant datasets")]
    for y,h,low,high,ticks,title in specs:
        label(c,x,y+h+14,title,12)
        for v in ticks:
            yy=y+h*(v-low)/(high-low);c.setStrokeColor(GRID);c.line(x,yy,x+w,yy);label(c,x-8,yy-3,v,9,"right")
        for v in [1,2,4,10,40,160]:label(c,xp(v),y-14,v,9,"center")
    line(shift,385,120,-.2,.3,C)
    line([round(e["rej_M"],3) for e in cs],215,120,0,.8,M);line([round(e["rej_C"],3) for e in cs],215,120,0,.8,C)
    line([e["M_only"] for e in cs],45,120,0,90,M);line([e["C_only"] for e in cs],45,120,0,90,C)
    label(c,360,12,"Spike height h (logarithmic scale)",11,"center")
    label(c,490,360,"Light pink: M / M only     Dark pink: C / C only",9)
    done(c,2)

fig2();fig3();fig9()
print("Rebuilt figures 2, 3 and 9 from reported numerical tables.")
