
#entry
import argparse
import shutil
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DRIVERS=("shape_aware_at_article_geometry","shape_aware_detection_floor","shape_aware_minor_axis")

def run(*args):
    subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True)

def fresh_directory(path):
    path=path.resolve()
    if path.exists():
        raise SystemExit(f"Output directory exists: {path}. Choose a fresh path.")
    path.mkdir(parents=True)
    return path
def figures():
    import colorsys,json,os,runpy,types
    from pdf2image import convert_from_path
    from pypdf import PdfReader,PdfWriter
    from pypdf.generic import ContentStream,FloatObject,NameObject
    from reportlab.lib.colors import Color,HexColor
    from reportlab.pdfgen import canvas
    O=ROOT/"paper/figures"
    POP=os.environ.get("POPPLER_PATH") or None
    C=["#B87593","#D18EAA","#E7A8C2","#F4BCD1"]
    A=("W","D4","K","E")
    load=lambda p:json.loads((ROOT/p).read_text(encoding="utf-8"))
    def png(name,dpi):
        convert_from_path(O/f"{name}.pdf",dpi=dpi,poppler_path=POP)[0].save(O/f"{name}.png")
    def heatmap():
        d=load("results/recorded/s24_s29/grid_recorded/summary.json")
        cs={(z["pr_target"],z["n"]):z for z in d["cells"].values()}
        ps=sorted({p for p,n in cs});ns=sorted({n for p,n in cs})
        c=canvas.Canvas(str(O/"Figure4.pdf"),pagesize=(960,450))
        a=HexColor("#FFF8FB");b=HexColor("#E7A8C2");ink=HexColor("#30272C");nom=HexColor("#806F78")
        def col(v):
            return Color(a.red+(b.red-a.red)*v,a.green+(b.green-a.green)*v,a.blue+(b.blue-a.blue)*v)
        def tx(x,y,t,z=11,al="left"):
            c.setFillColor(ink);c.setFont("Helvetica",z)
            getattr(c,{"left":"drawString","center":"drawCentredString","right":"drawRightString"}[al])(x,y,str(t))
        for k,arm in enumerate(("M","C")):
            x=65+k*435;y=80;cw=72;ch=49
            v=[[cs[p,n][arm]["rate"] for n in ns] for p in ps]
            tx(x,392,arm+" reference",17)
            for i,p in enumerate(ps):
                tx(x-9,y+(i+.5)*ch-4,f"{p:g}",12,"right")
                for j,n in enumerate(ns):
                    c.setFillColor(col(v[i][j]));c.rect(x+j*cw,y+i*ch,cw,ch,fill=1,stroke=0)
                    tx(x+(j+.5)*cw,y+(i+.5)*ch-4,f"{v[i][j]:.3f}",12,"center")
            for j,n in enumerate(ns):
                tx(x+(j+.5)*cw,y-18,n,12,"center")
            tx(x+2.5*cw,y-39,"sample size n",13,"center")
            c.setStrokeColor(nom);c.setLineWidth(2)
            for i in range(len(ps)-1):
                for j in range(len(ns)-1):
                    q=[(j,i),(j+1,i),(j+1,i+1),(j,i+1)];e=[]
                    for u,z in zip(q,q[1:]+q[:1]):
                        f,h=v[u[1]][u[0]],v[z[1]][z[0]]
                        if (f<.05)!=(h<.05):
                            t=(.05-f)/(h-f);e.append((x+(u[0]+t*(z[0]-u[0])+.5)*cw,y+(u[1]+t*(z[1]-u[1])+.5)*ch))
                    if len(e)==2:
                        c.line(*e[0],*e[1])
                    elif len(e)==4:
                        c.line(*e[0],*e[1]);c.line(*e[2],*e[3])
            c.setStrokeColor(ink);c.setLineWidth(2);c.rect(x,y+ps.index(4.5)*ch,cw,ch,fill=0,stroke=1)
        c.saveState();c.translate(19,227);c.rotate(90);tx(0,0,"target normalized participation ratio",13,"center");c.restoreState()
        for i in range(200):
            c.setFillColor(col(i/199));c.rect(885,80+i*294/200,14,294/200+.1,stroke=0,fill=1)
        for v in (0,.2,.4,.6,.8,1):
            tx(903,76+294*v,f"{v:.1f}",11)
        c.saveState();c.translate(945,227);c.rotate(90);tx(0,0,"rejection rate",13,"center");c.restoreState()
        c.setStrokeColor(nom);c.setLineWidth(2);c.line(188,15,212,15);tx(220,11,"nominal 0.05 contour",11)
        c.setStrokeColor(ink);c.rect(383,9,18,12,fill=0,stroke=1);tx(410,11,"grid cell nearest the fibromyalgia embedding (target PR 4.5, n = 28)",11)
        c.save();png("Figure4",240)
    def recolor():
        pairs={(.87,.65,.65):"F4BCD1",(.72,.56,.55):"D18EAA"}
        def rgb(c):
            r,g,b=map(float,c)
            if max(c)-min(c)<.12:
                return (r,g,b)
            h,l,s=colorsys.rgb_to_hls(r,g,b)
            if l<.19:
                return (r,g,b)
            return colorsys.hls_to_rgb(.945,.58+.40*l,min(.72,max(.48,s)))
        def shade(v):
            h=pairs.get(tuple(round(float(x),2) for x in v))
            return [FloatObject(int(h[i:i+2],16)/255) for i in (0,2,4)] if h else [FloatObject(x) for x in rgb(v)]
        for n in (1,5,6):
            d=PdfReader(ROOT/f"archive/figure_inputs/Figure{n}.pdf");txt=d.pages[0].extract_text();w=PdfWriter()
            for pg in d.pages:
                c=ContentStream(pg.get_contents(),d);old=c.operations
                new=[(shade(v) if k in (b"rg",b"RG") else v,k) for v,k in old]
                assert [t for t in old if t[1] not in (b"rg",b"RG")]==[t for t in new if t[1] not in (b"rg",b"RG")]
                c.operations=new;pg[NameObject("/Contents")]=c
                for z in pg.get("/Resources",{}).get("/Pattern",{}).values():
                    z=z.get_object();u=ContentStream(z,d)
                    u.operations=[(shade(v) if k in (b"rg",b"RG") else v,k) for v,k in u.operations]
                    z.set_data(u.get_data())
                w.add_page(pg)
            q=O/f"Figure{n}.pdf"
            with q.open("wb") as f:
                w.write(f)
            assert txt==PdfReader(q).pages[0].extract_text()
            png(f"Figure{n}",240)
    def text(c,x,y,s,n=10,align="left"):
        c.setFillColor(HexColor("#222222"));c.setFont("Helvetica",n)
        getattr(c,{"left":"drawString","center":"drawCentredString","right":"drawRightString"}[align])(x,y,str(s))
    def dot(c,x,y,lo,hi,col):
        c.setStrokeColor(HexColor(col));c.setLineWidth(2)
        c.line(x,lo,x,hi);c.line(x-3,lo,x+3,lo);c.line(x-3,hi,x+3,hi)
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
            p.close();c.drawPath(p,fill=1,stroke=0)

    def axes(c,x,y,w,h,top,title,ticks):
        text(c,x,y+h+14,title,11)
        for z in ticks:
            yy=y+h*z/top
            c.setStrokeColor(HexColor("#E3E0E1"));c.setLineWidth(.5);c.line(x,yy,x+w,yy)
            text(c,x-7,yy-3,f"{100*z:g}",9,"right")
        c.setStrokeColor(HexColor("#555555"));c.line(x,y,x,y+h);c.line(x,y,x+w,y)
        c.setDash(3,3);c.line(x,y+.05*h/top,x+w,y+.05*h/top);c.setDash()
    def legend(c,x,y):
        for i,a in enumerate(A):
            dot(c,x+75*i,y,y,y,C[i]);text(c,x+10+75*i,y-3,a,11)
    #s7
    def fig3(h):
        c=canvas.Canvas(str(O/"Figure3.pdf"),pagesize=(900,340))
        axes(c,48,62,205,215,.25,"A  Embedding-geometry study",[0,.05,.1,.15,.2,.25])
        for j,a in enumerate(("M","C")):
            v=h[a];xx=100+100*j
            c.setFillColor(HexColor(C[3-j*2]));c.rect(xx-20,62,40,215*v["rate"]/.25,fill=1,stroke=0)
            dot(c,xx,62+215*v["rate"]/.25,62+215*v["wilson95"][0]/.25,62+215*v["wilson95"][1]/.25,"#555555")
            text(c,xx,62+215*v["wilson95"][1]/.25+9,f"{100*v['rate']:.2f}%",10,"center")
            text(c,xx,43,a,11,"center")
        text(c,48,21,"10,000 datasets; 95% Wilson intervals",9)
        x,y,w,ht=330,62,200,215
        top=max(2000,100*((h["M_only"]+99)//100))
        text(c,x,y+ht+14,"B  Paired discordance",11)
        for v in range(0,top+1,500):
            yy=y+ht*v/top
            c.setStrokeColor(HexColor("#E3E0E1"));c.line(x,yy,x+w,yy)
            text(c,x-6,yy-3,str(v),9,"right")
        for j,a in enumerate(("M_only","C_only")):
            xx=375+j*90;v=h[a]
            c.setFillColor(HexColor(C[3-j*2]));c.rect(xx-20,y,40,ht*v/top,fill=1,stroke=0)
            text(c,xx,y+ht*v/top+8,v,10,"center")
            text(c,xx,43,a.replace("_"," "),10,"center")
        axes(c,600,62,280,215,.55,"C  Centered-law control",[0,.1,.2,.3,.4,.5])
        vals=[(.1075,.0808,.1417,3,"M"),(.015,.0069,.0323,1,"C"),(.4275,.3799,.4764,3,"M"),(.25,.2101,.2947,1,"C"),(.0575,.0386,.0848,1,"C")]
        for j,(v,lo,hi,k,a) in enumerate(vals):
            xx=628+j*51
            c.setFillColor(HexColor(C[k]));c.rect(xx-15,62,30,215*v/.55,fill=1,stroke=0)
            dot(c,xx,62+215*v/.55,62+215*lo/.55,62+215*hi/.55,"#555555")
            text(c,xx,62+215*hi/.55+7,f"{100*v:.2f}",8,"center")
            text(c,xx,43,a,9,"center")
        for x,s in [(653,"Fitted"),(755,"Moment H"),(842,"Generating law")]:
            text(c,x,28,s,8,"center")
        text(c,600,12,"Percent rejected; 400 datasets; Table 8 cell 1",9)
        c.save();png("Figure3",200)
    #S5
    def fig7(d):
        c=canvas.Canvas(str(O/"Figure7.pdf"),pagesize=(760,335))
        for s in range(2):
            x=52+s*370
            axes(c,x,74,310,210,.12,["A  Two-block spectrum","B  Exponential spectrum"][s],[0,.03,.05,.07,.1,.12])
            for j,q in enumerate((5,10)):
                cell=d["cells"][f"{s}_q{q}_null"]
                for k,a in enumerate(A):
                    z=cell["arms"][a];xx=x+55+160*j+(k-1.5)*15
                    lo,hi=z["conditional_wilson95"]
                    dot(c,xx,74+210*z["calibrated_rate"]/.12,74+210*lo/.12,74+210*hi/.12,C[k])
                text(c,x+55+160*j,53,f"q = {q}",10,"center")
        legend(c,240,24)
        c.save();png("Figure7",200)
    #S6 panels A-D
    def fig8(d,e):
        c=canvas.Canvas(str(O/"Figure8.pdf"),pagesize=(760,905))
        ph=140
        blocks=[(d,("leading2","leading4","minor2","minor4"),lambda i:("Leading " if i<2 else "Minor ")+str(2 if i%2==0 else 4),0,(715,520)),
                (e,("antipodal4","tri4","pair4","shift4"),lambda i:("Antipodal","Three-group","Pair","Shift only")[i],4,(290,95))]
        for src,gs,lab,off,ys in blocks:
            peak=max(src["cells"][f"{s}_q{q}_{g}"]["arms"][a]["two_stage_bootstrap95"][1] for s in range(2) for q in (5,10) for g in gs for a in A)
            top,ticks=(.2,[0,.05,.1,.15,.2]) if peak<=.2 else (1.,[0,.25,.5,.75,1])
            for s in range(2):
                for j,q in enumerate((5,10)):
                    x=52+j*375;y=ys[s]
                    axes(c,x,y,305,ph,top,f"{chr(65+off+s*2+j)}  "+("Two-block" if s==0 else "Exponential")+f", q = {q}",ticks)
                    for i,g in enumerate(gs):
                        cell=src["cells"][f"{s}_q{q}_{g}"]
                        for k,a in enumerate(A):
                            z=cell["arms"][a];lo,hi=z["two_stage_bootstrap95"]
                            xx=x+35+i*77+(k-1.5)*10
                            dot(c,xx,y+ph*z["calibrated_rate"]/top,y+ph*lo/top,y+ph*hi/top,C[k])
                        text(c,x+35+i*77,y-17,lab(i),9,"center")
        text(c,52,885,"S19: origin-symmetric two-group mixtures",11)
        text(c,52,470,"S20: alternatives at half-separation 4",11)
        legend(c,240,24)
        c.save();png("Figure8",200)
    sys.modules["followup_figures"]=types.SimpleNamespace(fig3=fig3)
    runpy.run_path(str(ROOT/"paper/build/rebuild_figures.py"),run_name="__main__")
    heatmap()
    recolor()
    d=load("results/recorded/s18_s22/calibrated_detection/summary.json")
    fig7(d)
    fig8(d,load("results/recorded/s18_s22/asymmetric_detection/summary.json"))
    run(ROOT/"paper/build/main_figures.py")

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest="command",required=True)
    sub.add_parser("verify")
    sub.add_parser("test")
    sub.add_parser("figures")
    for name in ("power","validation","geometry","followup"):
        cmd=sub.add_parser(name)
        cmd.add_argument("--output",type=Path,required=True)
        cmd.add_argument("--workers",type=int,default=4)
    args=ap.parse_args()
    if args.command=="verify":
        run(ROOT/"analysis/verify_repository.py")
        run(ROOT/"reproducibility/s16/verify_results.py")
        run(ROOT/"analysis/verify_corrected_power.py")
        run(ROOT/"analysis/verify_geometry_surface.py")
        run(ROOT/"analysis/verify_consensus.py")
        run(ROOT/"analysis/verify_followup.py")
        run(ROOT/"analysis/verify_asymmetric.py")
        run(ROOT/"analysis/verify_scope.py")
        run(ROOT/"analysis/verify_s24_s26.py")
        run(ROOT/"analysis/verify_s29.py")
        run(ROOT/"analysis/verify_s31_s32.py")
    elif args.command=="test":
        run("-m","pytest","tests","packaging","-q","-p","no:cacheprovider")
    elif args.command=="figures":
        figures()
    else:
        if args.workers<1:
            ap.error("--workers must be positive")
        out=fresh_directory(args.output)
        if args.command=="power":
            for script in DRIVERS:
                run(ROOT/"analysis"/(script+".py"),"--workers",args.workers,"--output",out/(script+"_eigenvector_corrected.json"))
            run(ROOT/"analysis/verify_corrected_power.py","--directory",out)
        elif args.command=="geometry":
            run(ROOT/"analysis/geometry_surface.py","--workers",args.workers,"--output",out/"geometry_surface_fresh.json")
            run(ROOT/"analysis/verify_geometry_surface.py","--input",out/"geometry_surface_fresh.json")
        elif args.command=="followup":
            run(ROOT/"analysis/headline_recorded.py","--output",out/"headline","--workers",args.workers)
            run(ROOT/"analysis/calibrated_detection.py","--output",out/"calibrated_detection","--workers",args.workers)
            run(ROOT/"analysis/summarize_detection.py","--directory",out/"calibrated_detection")
            run(ROOT/"analysis/verify_followup.py","--directory",out)
            run(ROOT/"analysis/asymmetric_detection.py","--output",out/"asymmetric_detection","--workers",args.workers)
            run(ROOT/"analysis/summarize_detection.py","--directory",out/"asymmetric_detection")
            run(ROOT/"analysis/verify_asymmetric.py","--directory",out/"asymmetric_detection")
            for n in (28,100):
                sd=out/f"scope_detection_n{n}"
                run(ROOT/"analysis/scope_detection.py","--output",sd,"--n",n,"--workers",args.workers)
                run(ROOT/"analysis/summarize_detection.py","--directory",sd)
                run(ROOT/"analysis/verify_scope.py","--directory",sd)
            run(ROOT/"analysis/mc_calibrated_sensitivity.py","--output",out/"mc_calibrated_sensitivity","--workers",args.workers)
            run(ROOT/"analysis/summarize_detection.py","--directory",out/"mc_calibrated_sensitivity")
            for n in (50,200):
                sd=out/f"sample_size_n{n}"
                run(ROOT/"analysis/sample_size_detection.py","--output",sd,"--n",n,"--workers",args.workers)
                run(ROOT/"analysis/summarize_detection.py","--directory",sd)
            run(ROOT/"analysis/grid_recorded.py","--output",out/"grid_recorded","--workers",args.workers)
            run(ROOT/"analysis/robustness_detection.py","--part","b","--output",out/"robustness_mc","--workers",args.workers)
            run(ROOT/"analysis/robustness_detection.py","--part","b","--summarize","--output",out/"robustness_mc")
            for n in (28,100):
                sd=out/f"robustness_n{n}"
                run(ROOT/"analysis/robustness_detection.py","--part","a","--n",n,"--output",sd,"--workers",args.workers)
                run(ROOT/"analysis/robustness_detection.py","--part","a","--summarize","--output",sd)
            for st,name in (("S31","mechanism_control"),("S32","direction_reversal")):
                run(ROOT/"analysis/recorded_controls.py","--study",st,"--output",out/name,"--workers",args.workers)
                run(ROOT/"analysis/recorded_controls.py","--study",st,"--summarize","--output",out/name)
        else:
            s16=ROOT/"reproducibility/s16"
            shutil.copy2(s16/"run_validation.py",out/"run_validation.py")
            shutil.copy2(s16/"verify_results.py",out/"verify_results.py")
            shutil.copytree(s16/"modules",out/"modules",ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
            run(out/"run_validation.py","--datasets",200,"--draws",199,"--workers",args.workers)
            run(out/"verify_results.py")
if __name__=="__main__":
    main()
