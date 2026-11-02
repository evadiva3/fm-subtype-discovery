#Regenerate every figure in the manuscript from its archived JSON artifact
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
ARTIFACT_DIRS=[Path("."),Path("results"),Path("results/overnight"),Path("results/calibration"),Path("../calibration"),Path("experiments_v9")];OUT=Path("figures")
C_M,C_C,C_G,C_S=("#DDA7A5","#B88E8C","#966B6D","#734F50");INK,MUTED,GRID=("#241a1e","#6f6268","#e6dade");NOMINAL="#9a8b91";HATCH_M,HATCH_C=("","///")
CMAP=LinearSegmentedColormap.from_list("nullcal_pink",["#fbf4f4","#DDA7A5","#966B6D","#3d2a2c"])
def style():
    plt.rcParams.update({"figure.dpi":150,"savefig.dpi":300,"font.size":8.5,"axes.titlesize":9,"axes.labelsize":8.5,"legend.fontsize":8,"xtick.labelsize":8,"ytick.labelsize":8,"axes.edgecolor":MUTED,"axes.linewidth":0.6,"axes.spines.top":False,"axes.spines.right":False,"text.color":INK,"axes.labelcolor":INK,"xtick.color":MUTED,"ytick.color":MUTED,"grid.color":GRID,"grid.linewidth":0.5,"legend.frameon":False,"figure.facecolor":"white","axes.facecolor":"white"})
def find(name,root=None):
    ds=[Path(root)]+[Path(root)/d for d in ARTIFACT_DIRS] if root else[]
    for d in ds+ARTIFACT_DIRS:
        p=d/name
        if p.exists():return p
    for b in([Path(root)] if root else[])+[Path(".")]:
        h=sorted(b.glob(f"results/phase*/*/{name}"))
        if h:return h[-1]
    return None
def load(name,root=None):
    p=find(name,root);return json.loads(p.read_text()) if p else None
ALIASES={"target_pr":"pr_target","S":"spike_S","effective_K":"eff_K_null"}
def cells_of(d):
    r=d.get("cells",d) if isinstance(d,dict) else d
    if isinstance(r,dict):o=[dict(v,cell=k) for k,v in r.items() if isinstance(v,dict)]
    else:o=[dict(c) for c in r]
    for c in o:
        for w,h in ALIASES.items():
            if w not in c and h in c:c[w]=c[h]
    return o
_REPO_CANDIDATES=[Path("."),Path(".."),Path("../.."),Path("../../..")]
def find_repo_file(rel,root=None):
    for c in([Path(root)] if root else[])+_REPO_CANDIDATES:
        p=c/rel
        if p.exists():return p
    return None
def _load_rel(rel,root=None):
    p=find_repo_file(rel,root);return json.loads(p.read_text()) if p else None
def _propagated_ladder(root):
    n={"fm_lad":"results/raw_feature_ladder.json","fm_prop":"results/handoff_20260802/propagated_null_summary.json","dep_lad":"results/mdd/canonical_single/null_progression_mdd_20000.json","dep_prop":"results/mdd/propagated_null/propagated_null_summary.json","fm_var":"results/handoff_20260802/reproducibility/reproducibility_summary.json","dep_var":"results/mdd/propagated_null/paired_rerun/paired_summary.json"}
    g={k:_load_rel(v,root) for k,v in n.items()};at=[n[k] for k,v in g.items() if not v]
    if at:return(None,", ".join(at)+" (Section 3.7 propagated null)")
    r=g["fm_lad"]["rungs"];fm={a:float(next((r["embedding_space_p"] for n,r in r.items() if n.startswith(p)))) for a,p in(("M","Misspecified"),("G","Geometry"),("S","Selection"),("C","Fully"))}
    fm["P"]=float(g["fm_prop"]["p"]);pb=g["dep_lad"]["p_by_construction"]
    d={"M":pb["misspecified"],"G":pb["geometry_only"],"S":pb["selection_only"],"C":pb["corrected"],"P":float(g["dep_prop"]["primary_canonical"]["p"])}
    return({"fibromyalgia":fm,"depression":d,"retraining_share":{"fibromyalgia":float(g["fm_var"]["var_training_share"]),"depression":float(g["dep_var"]["training_share"])}},None)
def save(fig,stem):
    OUT.mkdir(exist_ok=True)
    for e in("png","pdf"):fig.savefig(OUT/f"{stem}.{e}",bbox_inches="tight",facecolor="white")
    plt.close(fig);print(f"  wrote figures/{stem}.png and .pdf")
def nominal_line(ax,y=0.05,xloc=0.995,ha="right"):
    ax.axhline(y,color=NOMINAL,lw=0.9,ls=(0,(4,3)),zorder=1);ax.annotate("nominal 0.05",xy=(xloc,y),xycoords=("axes fraction","data"),ha=ha,va="bottom",fontsize=7,color=MUTED)
def errbars(ax,x,rate,ci,color,label,marker="o"):
    rate=np.asarray(rate,dtype=float);lo=rate-np.asarray([c[0] for c in ci],dtype=float);hi=np.asarray([c[1] for c in ci],dtype=float)-rate
    ax.errorbar(x,rate,yerr=[lo,hi],color=color,lw=2.0,marker=marker,ms=5,mew=0,capsize=0,elinewidth=1.2,label=label,zorder=3)
def fig1(root):
    a1=None;m=cy=None;ra=find("r1a_headline_recalibration.json",root)
    if ra:
        s=json.loads(Path(ra).read_text(encoding="utf-8"));a1={a:{"rejection":s[a]["rejection_rate"],"ci":s[a]["wilson95"]} for a in("M","C")};m=int(s["mcnemar"]["M_only"])
        cy=int(s["mcnemar"]["C_only"])
    if a1 is None:
        sm={a:load(f"calibration_summary_{t}.json",root) for a,t in(("M","misspecified"),("C","corrected"))}
        if not all(sm.values()):
            ms=[t for a,t in(("M","misspecified"),("C","corrected")) if not sm[a]];return "r1a_headline_recalibration.json, or calibration_summary_{"+",".join(ms)+"}.json (Section 3.1)"
        a1={a:{"rejection":s["rejection_rate_05"],"ci":s["rejection_rate_05_ci95"]} for a,s in sm.items()};pv={}
        for a,t in(("M","misspecified"),("C","corrected")):
            p=find(f"calibration_pvalues_{t}.npy",root)
            if p:pv[a]=np.load(p)
        if len(pv)==2 and pv["M"].shape==pv["C"].shape:
            rm,rc=(pv["M"]<0.05,pv["C"]<0.05);m=int((rm&~rc).sum());cy=int((rc&~rm).sum())
    f,(a,b,c)=plt.subplots(1,3,figsize=(7.1,3.0),gridspec_kw={"width_ratios":[0.95,0.8,1.5],"wspace":0.42});ls=["M","C"];rs=[a1["M"]["rejection"],a1["C"]["rejection"]];cs=[a1["M"]["ci"],a1["C"]["ci"]]
    for i,(l,r,ci,cl,h) in enumerate(zip(ls,rs,cs,[C_M,C_C],[HATCH_M,HATCH_C])):
        a.bar(i,r,width=0.5,color=cl,hatch=h,edgecolor="white",linewidth=0.8,zorder=3);a.errorbar(i,r,yerr=[[r-ci[0]],[ci[1]-r]],color=INK,lw=1.0,capsize=0,zorder=4)
        a.annotate(f"{r:.3f}",(i,ci[1]),ha="center",va="bottom",fontsize=8,color=INK,xytext=(0,3),textcoords="offset points")
    nominal_line(a,xloc=0.02,ha="left");a.set_xticks(range(2));a.set_xticklabels(ls);a.set_ylabel("realized Type I error");a.set_xlim(-0.6,1.6);a.yaxis.grid(True);a.set_axisbelow(True)
    a.set_title("95 % Wilson",loc="left",color=INK)
    if m is None:
        b.axis("off");b.annotate('paired discordance unavailable:\nper-dataset p-values not found',(0.5,0.5),xycoords="axes fraction",ha="center",va="center",fontsize=7.5,color=MUTED)
    else:
        b.barh([1,0],[m,cy],height=0.5,color=[C_M,C_C],hatch=[HATCH_M,HATCH_C],edgecolor="white",linewidth=0.8,zorder=3)
        for y,v in zip([1,0],[m,cy]):b.annotate(f"{v}",(v,y),ha="left",va="center",fontsize=8,color=INK,xytext=(4,0),textcoords="offset points")
        b.set_yticks([1,0]);b.set_yticklabels(["M only","C only"]);b.set_ylim(-0.6,1.6);b.set_xlim(0,max(m,cy)*1.34+1);b.set_xlabel("discordant datasets");b.xaxis.grid(True);b.set_axisbelow(True)
        b.set_title("paired discordance",loc="left",color=INK)
    _fig1_popcov_panel(c,root);save(f,"fig1_type1_paired")
POPCOV_CELL="cell1_exp_n28"
def _fig1_popcov_panel(ax,root):
    d=load("popcov_control.json",root);c=(d or{}).get("cells",{}).get(POPCOV_CELL)
    if not c:
        ax.axis("off")
        ax.annotate(f'population-covariance control unavailable:\npopcov_control.json, cell {POPCOV_CELL}',(0.5,0.5),xycoords="axes fraction",ha="center",va="center",fontsize=7.5,color=MUTED);return
    g=[("covariance fitted","est",0.0),("known Σ","known",2.3)];w=0.8
    for l,t,x0 in g:
        for j,(a,cl,h) in enumerate([("M",C_M,HATCH_M),("C",C_C,HATCH_C)]):
            a_=c["arms"][f"{a}_{t}"];r=float(a_["rejection_rate"]);lo,hi=(float(v) for v in a_["wilson95"]);x=x0+j*w;ax.bar(x,r,width=w*0.86,color=cl,hatch=h,edgecolor="white",linewidth=0.8,zorder=3)
            ax.errorbar(x,r,yerr=[[r-lo],[hi-r]],color=INK,lw=1.0,capsize=0,zorder=4)
            ax.annotate(f"{r:.4f}",(x,hi),ha="center",va="bottom",fontsize=7.5,color=INK,xytext=(0,3),textcoords="offset points")
    nominal_line(ax,xloc=0.585,ha="center");ts=[x0+j*w for _,_,x0 in g for j in(0,1)];ax.set_xticks(ts);ax.set_xticklabels(["M","C","M","C"]);ax.set_xlim(-w*0.8,g[1][2]+w*1.8)
    for l,_,x0 in g:ax.annotate(l,xy=(x0+w/2,-0.15),xycoords=("data","axes fraction"),ha="center",va="top",fontsize=8,color=INK)
    tp=max((float(c["arms"][f"{a_}_{t}"]["wilson95"][1]) for a_ in("M","C") for t in("est","known")));ax.set_ylim(0,tp*1.3);ax.yaxis.grid(True);ax.set_axisbelow(True)
    ax.set_ylabel("realized Type I error");ax.set_title(f"fitted against population Σ, n = {c["n"]}, d = {c["d"]}",loc="left",color=INK)
def fig2(root):
    d=load("e1_grid.json",root)
    if not d:return "e1_grid.json (Section 3.2 grid)"
    cs={(c["pr_target"],c["n"]):c for c in cells_of(d)};ps=sorted({k[0] for k in cs});ns=sorted({k[1] for k in cs});f,a1=plt.subplots(1,2,figsize=(7.6,3.2));im=None
    for ax,a,t in[(a1[0],"M","M, unmatched"),(a1[1],"C","C, fully matched")]:
        g=np.array([[cs[p,n][f"rej_{a}"] for n in ns] for p in ps]);im=ax.imshow(g,origin="lower",cmap=CMAP,vmin=0.0,vmax=1.0,aspect="auto")
        for i in range(len(ps)):
            for j in range(len(ns)):ax.annotate(f"{g[i, j]:.3f}",(j,i),ha="center",va="center",fontsize=6.4,color="white" if g[i,j]>0.55 else INK)
        ax.contour(np.arange(len(ns)),np.arange(len(ps)),g,levels=[0.05],colors=[NOMINAL],linewidths=1.6);ax.set_xticks(range(len(ns)));ax.set_xticklabels([str(n) for n in ns])
        ax.set_yticks(range(len(ps)));ax.set_yticklabels([f"{p:g}"for p in ps]);ax.set_xlabel("sample size n");ax.set_title(t,loc="left",color=INK);ax.tick_params(length=0)
        for s in ax.spines.values():s.set_visible(False)
    a1[0].set_ylabel("target participation ratio")
    if 4.5 in ps and 28 in ns:
        xf,yf=(ns.index(28),ps.index(4.5))
        for ax in a1:ax.add_patch(Rectangle((xf-0.5,yf-0.5),1,1,fill=False,edgecolor=INK,linewidth=1.6,zorder=5))
        f.legend(handles=[Line2D([],[],color=NOMINAL,lw=1.6,label="nominal 0.05"),Rectangle((0,0),1,1,fill=False,edgecolor=INK,linewidth=1.6,label="fibromyalgia geometry (PR 4.5, n = 28)")],loc="lower center",bbox_to_anchor=(0.44,-0.1),ncol=2,columnspacing=1.8,handlelength=1.4)
    cb=f.colorbar(im,ax=a1,fraction=0.03,pad=0.02);cb.set_label("realized Type I error",fontsize=8);cb.outline.set_visible(False);cb.ax.tick_params(length=2);save(f,"fig2_geometry_surface")
def fig3(root):
    d=load("e2_spectrum_shape.json",root)
    if not d:return "e2_spectrum_shape.json (Section 3.3 spectrum families)"
    cs=cells_of(d);fs=sorted({c["family"] for c in cs});p=sorted({c["target_pr"] for c in cs});fg,a1=plt.subplots(1,len(p),figsize=(2.3*len(p),2.9),sharey=True);a1=np.atleast_1d(a1);w=0.38
    for ax,pr in zip(a1,p):
        s={c["family"]:c for c in cs if c["target_pr"]==pr};xs=np.arange(len(fs))
        for o,am,cl,h,l in[(-w/2,"M",C_M,HATCH_M,"M, unmatched"),(w/2,"C",C_C,HATCH_C,"C, fully matched")]:
            r=[s[f][f"rej_{am}"] if f in s else np.nan for f in fs];ci=[s[f][f"ci_{am}"] if f in s else[np.nan,np.nan] for f in fs];lo=[max(a-c[0],0) for a,c in zip(r,ci)]
            hi=[max(c[1]-a,0) for a,c in zip(r,ci)];ax.bar(xs+o,r,width=w,color=cl,hatch=h,edgecolor="white",linewidth=0.6,label=l,zorder=3)
            ax.errorbar(xs+o,r,yerr=[lo,hi],fmt="none",ecolor=INK,elinewidth=0.9,capsize=0,zorder=4)
        nominal_line(ax);ax.set_xticks(xs);ax.set_xticklabels([f.replace("_"," ") for f in fs],rotation=45,ha="right");ax.set_title(f"PR {pr}",loc="left",color=INK);ax.yaxis.grid(True)
        ax.set_axisbelow(True)
    a1[0].set_ylabel("realized Type I error");a1[-1].legend(loc="upper right");save(fg,"fig3_spectra")
def fig4(root):
    d=load("e15_boundary.json",root)
    if not d:return "e15_boundary.json (Section 3.4 sign change)"
    cs=sorted(cells_of(d),key=lambda c:c["S"]);h=[c["S"] for c in cs];xs=np.arange(len(h));f,a=plt.subplots(3,1,figsize=(4.4,5.8),sharex=True,gridspec_kw={"hspace":0.45})
    a[0].axhline(0,color=NOMINAL,lw=0.9,ls=(0,(4,3)));a[0].plot(xs,[c["shift"] for c in cs],color=C_G,lw=2.0,marker="o",ms=5,mew=0);a[0].set_ylabel("re-projection shift")
    a[0].set_title("re-projection shift",loc="left",color=INK);errbars(a[1],xs,[c["rej_M"] for c in cs],[c["ci_M"] for c in cs],C_M,"M, unmatched")
    errbars(a[1],xs,[c["rej_C"] for c in cs],[c["ci_C"] for c in cs],C_C,"C, fully matched",marker="s");nominal_line(a[1]);a[1].set_ylabel("Type I error");a[1].legend(loc="upper left")
    a[1].set_title("realized Type I error",loc="left",color=INK);w=0.4;a[2].bar(xs-w/2,[c["M_only"] for c in cs],width=w,color=C_M,label="M only",zorder=3)
    a[2].bar(xs+w/2,[c["C_only"] for c in cs],width=w,color=C_C,label="C only",zorder=3);a[2].set_xticks(xs);a[2].set_xticklabels([f"{v:g}"for v in h]);a[2].set_ylabel("discordant datasets")
    a[2].set_xlabel("spike height $h$");a[2].legend(loc="upper left");a[2].set_title("paired discordance by direction",loc="left",color=INK)
    for ax in a:
        ax.yaxis.grid(True);ax.set_axisbelow(True)
    save(f,"fig4_signchange")
def fig5(root):
    ds=load("e6_statistics.json",root);da=load("e7_algorithms.json",root)
    if not ds or not da:
        m=[n for n,x in[("e6_statistics.json",ds),("e7_algorithms.json",da)] if not x];return " and ".join(m)+" (Section 3.5)"
    f,a1=plt.subplots(1,2,figsize=(6.8,3.0),sharey=True)
    for ax,d,k,t in[(a1[0],ds,"statistic","statistic varying, k-means held"),(a1[1],da,"algorithm","algorithm varying, silhouette held")]:
        cs=cells_of(d);ns=[c[k] for c in cs];xs=np.arange(len(ns));w=0.38
        for o,am,cl,h,l in[(-w/2,"M",C_M,HATCH_M,"M, unmatched"),(w/2,"C",C_C,HATCH_C,"C, fully matched")]:
            r=[c[f"rej_{am}"] for c in cs];ci=[c[f"ci_{am}"] for c in cs];ax.bar(xs+o,r,width=w,color=cl,hatch=h,edgecolor="white",linewidth=0.6,label=l,zorder=3)
            ax.errorbar(xs+o,r,fmt="none",ecolor=INK,elinewidth=0.9,yerr=[[max(a-c[0],0) for a,c in zip(r,ci)],[max(c[1]-a,0) for a,c in zip(r,ci)]],capsize=0,zorder=4)
        nominal_line(ax);ax.set_xticks(xs);ax.set_xticklabels([n.replace("_"," ") for n in ns],rotation=30,ha="right");ax.set_title(t,loc="left",color=INK);ax.yaxis.grid(True);ax.set_axisbelow(True)
    a1[0].set_ylabel("realized Type I error");a1[0].legend(loc="upper right");save(f,"fig5_statistic_algorithm")
def fig6(root):
    d=load("e4_k_family.json",root)
    if not d:return "e4_k_family.json (Section 3.6 k-search size)"
    cs=sorted(cells_of(d),key=lambda c:c["nominal_K"]);f,ax=plt.subplots(figsize=(4.6,3.0))
    for xk,ls,mk,l in[("effective_K","-","o","effective |K|"),("nominal_K",(0,(5,2)),"^","nominal |K|")]:
        x=[c[xk] for c in cs];ax.plot(x,[c["rej_M"] for c in cs],color=C_M,lw=2.0,ls=ls,marker=mk,ms=5,mew=0,label=f"M, {l}")
        ax.plot(x,[c["rej_C"] for c in cs],color=C_C,lw=2.0,ls=ls,marker=mk,ms=5,mew=0,label=f"C, {l}")
    nominal_line(ax);ax.set_xlabel("size of the k-family");ax.set_ylabel("realized Type I error");ax.yaxis.grid(True);ax.set_axisbelow(True);ax.legend(loc="center right");save(f,"fig6_k_family")
def fig7(root):
    d,m=_propagated_ladder(root)
    if not d:return m
    o1=["M","G","S","C","P"];cs={"M":C_M,"G":C_G,"S":C_S,"C":C_C,"P":INK};f,(a,b)=plt.subplots(1,2,figsize=(7.2,2.9),gridspec_kw={"width_ratios":[1.6,1],"wspace":0.42});w=0.38;xs=np.arange(len(o1))
    for o,c,h in[(-w/2,"fibromyalgia",None),(w/2,"depression","///")]:
        vs=[d[c][k] for k in o1];a.bar(xs+o,vs,width=w,color=[cs[k] for k in o1],edgecolor="white",linewidth=0.8,hatch=h,zorder=3,label=c)
    a.axhline(0.05,color=NOMINAL,lw=0.9,ls=(0,(4,3)));a.set_xticks(xs);a.set_xticklabels(o1);a.set_ylabel("permutation $p$");a.set_ylim(0,1.05);a.yaxis.grid(True);a.set_axisbelow(True)
    a.legend(loc="upper left");a.set_title("permutation $p$ by construction",loc="left",color=INK);s=d.get("retraining_share",{})
    if s:
        ks=list(s);b.barh(np.arange(len(ks)),[s[k] for k in ks],height=0.5,color=C_G,zorder=3)
        for y,k in enumerate(ks):
            v=s[k];b.annotate("< 0.01" if v<0.01 else f"{v:.2f}",(max(v,0),y),xytext=(4,0),textcoords="offset points",ha="left",va="center",fontsize=7.5,color=INK)
        b.set_yticks(np.arange(len(ks)));b.set_yticklabels(ks);b.set_xlim(0,1.12);b.set_xlabel("retraining share of null variance");b.xaxis.grid(True);b.set_axisbelow(True)
        b.set_title("variance decomposition",loc="left",color=INK)
    save(f,"fig7_ladder_variance")
def fig_constraint(root):
    d=load("e13_manifold.json",root) or load("manifold_test.json",root)
    if not d:return "e13_manifold.json or manifold_test.json (Section 3.6 constraints)"
    cs=d["cells"];n={"l2":'L2\nsphere',"l1":'L1\npolytope',"l4":'L4\nball',"linf":'L$\\infty$'+'\ncube',"ellip":'ellip-\nsoid',"none":'none\ncontrol'}
    o1=[c for c in["l2","l1","l4","linf","ellip","none"] if any((x.get("transform",x.get("key"))==c for x in cs))];g=lambda t:next((x for x in cs if x.get("transform",x.get("key"))==t))
    f,(a,b)=plt.subplots(1,2,figsize=(7.2,3.0),gridspec_kw={"width_ratios":[1.9,1],"wspace":0.32});xs=np.arange(len(o1));w=0.2
    for i,(am,cl,lb) in enumerate([("M",C_M,"M, unmatched"),("G",C_G,"G, geometry matched"),("S",C_S,"S, selection matched"),("C",C_C,"C, fully matched")]):
        o=(i-1.5)*w;a.bar(xs+o,[g(t)[f"rej_{am}"] for t in o1],width=w,color=cl,label=lb,zorder=3)
    nominal_line(a);a.set_xticks(xs);a.set_xticklabels([n[t] for t in o1]);a.set_ylabel("realized Type I error");a.yaxis.grid(True);a.set_axisbelow(True)
    tp=max((g(t)[f"rej_{am}"] for t in o1 for am in "MGSC"));a.set_ylim(0,tp*1.12);a.set_title("realized Type I error by constraint",loc="left",color=INK);sh=[g(t)["reproj_shift"] for t in o1]
    b.axhline(0,color=NOMINAL,lw=0.9,ls=(0,(4,3)));b.bar(xs,sh,width=0.5,color=[C_G if s>1e-09 else MUTED for s in sh],zorder=3);b.set_xticks(xs)
    st={"l2":"L2","l1":"L1","l4":"L4","linf":'L$\\infty$',"ellip":"ellip","none":"none"};b.set_xticklabels([st[t] for t in o1],fontsize=8);b.set_ylabel("re-projection shift");b.yaxis.grid(True)
    b.set_axisbelow(True);b.set_title("re-projection shift by constraint",loc="left",color=INK);h,l=a.get_legend_handles_labels()
    f.legend(h,l,loc="lower center",ncol=4,bbox_to_anchor=(0.5,-0.1),columnspacing=1.6,handlelength=1.3);save(f,"figS_constraints")
FIGURES={1:(fig1,"Realized Type I error of M and C, paired discordance"),2:(fig2,"Type I error over the PR x n grid, both constructions"),3:(fig3,"Rejection by spectrum family"),4:(fig4,"Shift, rejection and discordance against spike height"),5:(fig5,"Type I error by statistic and by algorithm"),6:(fig6,"Type I error against effective and nominal |K|"),7:(fig7,"All five constructions, and the variance decomposition"),8:(fig_constraint,"Supplementary: norm constraints (not one of the seven)")}
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",help="repository root to search for artifacts");ap.add_argument("--only",nargs="*",type=int,help="figure numbers")
    ap.add_argument("--list",action="store_true");a=ap.parse_args()
    if a.list:
        print("figure  status    needs")
        for n,(fn,d) in FIGURES.items():
            print(f"  {n}     {"":8} {d}")
        print('\nRun without --list to build; each missing artifact is named.');return
    style();w=a.only or list(FIGURES);b,mg=([],[]);print(f"building into {OUT}/ ...")
    for n in w:
        fn,d=FIGURES[n];print(f"figure {n}: {d}")
        try:ms=fn(a.root)
        except Exception as e:ms=f"{type(e).__name__}: {e}"
        if ms:
            mg.append((n,ms));print(f"  SKIPPED, needs {ms}")
        else:b.append(n)
    print(f"\nbuilt {len(b)} of {len(w)}: {b or "none"}")
    if mg:
        print('\nnot built, and what each needs:')
        for n,m in mg:print(f"  figure {n}: {m}")
        print('\nPoint --root at the repository, or copy the JSON artifacts next');print("to this script. Section S9 of the supplement maps every figure to")
        print("the script and artifact that produce it.")
if __name__=="__main__":main()
