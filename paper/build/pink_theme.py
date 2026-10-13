from pathlib import Path
import colorsys,re,json,hashlib
from pypdf import PdfReader,PdfWriter
from pypdf.generic import ContentStream,FloatObject,NameObject
import os

R=Path(__file__).resolve().parents[2]
P=R/"paper/figures"
B=R/"archive/pre_pink_20260923/paper/figures"

def rgb(c):
    r,g,b=map(float,c)
    if max(c)-min(c)<.12:return (r,g,b)
    h,l,s=colorsys.rgb_to_hls(r,g,b)
    if l<.19:return (r,g,b)
    return colorsys.hls_to_rgb(.945,.58+.40*l,min(.72,max(.48,s)))

def figs():
    from pdf2image import convert_from_path
    a=[]
    fresh=(R/"results/revision_20260924/calibrated_detection/summary.json").exists()
    for n in ((1,5,6) if fresh else (1,5,6,7,8)):
        p=B/f"Figure{n}.pdf";d=PdfReader(p);txt=d.pages[0].extract_text();w=PdfWriter()
        for pg in d.pages:
            c=ContentStream(pg.get_contents(),d);old=c.operations;new=[]
            for j,(v,k) in enumerate(old):
                if k in (b"rg",b"RG"):
                    h=({(.11,.42,.58):"D18EAA",(.68,.4,.23):"F4BCD1",(.33,.51,.26):"B87593",(.45,.48,.52):"E7A8C2"} if n==7 else{(.87,.65,.65):"F4BCD1",(.72,.56,.55):"D18EAA"}).get(tuple(round(float(x),2) for x in v))
                    v=[FloatObject(int(h[i:i+2],16)/255) for i in (0,2,4)] if h else [FloatObject(x) for x in rgb(v)]
                if n==8 and k==b"rg" and min(map(float,v))>.99 and j+1<len(old) and old[j+1][1]==b"BT":v=[FloatObject(.15)]*3
                new.append((v,k))
            assert [(v,k) for v,k in old if k not in (b"rg",b"RG")]==[(v,k) for v,k in new if k not in (b"rg",b"RG")]
            c.operations=new;pg[NameObject("/Contents")]=c
            for z in pg.get("/Resources",{}).get("/Pattern",{}).values():
                z=z.get_object();u=ContentStream(z,d);v=[]
                for x,k in u.operations:
                    if k in (b"rg",b"RG"):
                        h={ (.87,.65,.65):"F4BCD1",(.72,.56,.55):"D18EAA"}.get(tuple(round(float(t),2) for t in x))
                        x=[FloatObject(int(h[i:i+2],16)/255) for i in (0,2,4)] if h else[FloatObject(t) for t in rgb(x)]
                    v.append((x,k))
                u.operations=v;z.set_data(u.get_data())
            w.add_page(pg)
        q=P/f"Figure{n}.pdf"
        with q.open("wb")as f:w.write(f)
        assert txt==PdfReader(q).pages[0].extract_text()
        im=convert_from_path(q,dpi=240,poppler_path=os.environ.get("POPPLER_PATH") or None)[0];im.save(P/f"Figure{n}.png")
        a.append({"figure":n,"noncolor_pdf_operators_unchanged":True,"text_unchanged":True})
    for p in B.glob("*.svg"):
        if fresh:continue
        s=p.read_text(encoding="utf-8")
        def sub(m):
            h=m[0][1:];c=[int(h[i:i+2],16)/255 for i in (0,2,4)]
            return "#"+"".join(f"{round(x*255):02x}"for x in rgb(c))
        s=re.sub(r"#[0-9A-Fa-f]{6}",sub,s)
        s=re.sub(r'(<g id="text_[^"]+">\s*<!--.*?-->\s*<g style="fill: )#ffffff',r'\g<1>#262626',s,flags=re.S)
        (P/p.name).write_text(s,encoding="utf-8")
    (R/"reproducibility/audits/pink_figure_audit.json").write_text(json.dumps(a,indent=2))

if __name__=="__main__":figs()
