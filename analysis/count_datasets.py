#Count the structureless datasets generated under the single-Gaussian null
import json
from pathlib import Path
_R=Path(__file__).resolve().parents[1];RESULTS=_R/"results";OVN=RESULTS/"overnight";INCLUDED=[];EXCLUDED=[]
SECTIONS=[("overnight","Overnight suite"),("calibration","Section 3.1 and the S4 calibration cells"),("phase1","Phase 1 revision experiments"),("phase2","Phase 2 mechanism experiments"),("popcov","Population-covariance control"),("unreported","Ran but not reported in the article")]
def rel(p):return Path(p).resolve().relative_to(_R).as_posix()
def load(p):return json.loads(Path(p).read_text())
def newest(root,name):
    h=sorted(Path(root).glob("*/"+name));return h[-1] if h else None
def add(section,key,label,artifact,cells,generated,analysed,rule):
    INCLUDED.append(dict(section=section,key=key,label=label,artifact=rel(artifact),cells=int(cells),generated=int(generated),analysed=int(analysed),rule=rule))
def drop(key,label,artifact,count,reason):EXCLUDED.append(dict(key=key,label=label,artifact=rel(artifact) if artifact else "",count=None if count is None else int(count),reason=reason))
def cell_totals(cells):
    it=cells.values() if isinstance(cells,dict) else cells;g=a=n=0
    for c in it:
        if not isinstance(c,dict) or c.get("failed") or "n_datasets" not in c:continue
        a+=c["n_datasets"];g+=c["n_datasets"]+c.get("n_skipped",0);n+=1
    return(g,a,n)
SPECTRUM_FIELDS=("n","d","pr_target","family","param_value","tau")
def spectrum_key(cell):return tuple((cell.get(f) for f in SPECTRUM_FIELDS))
def verification(path,key,label,section="overnight"):
    d=load(path);p=d["config"]["n_datasets"];s=set();g=a=0;rg=sd=0
    for c in d["cells"].values():
        if not isinstance(c,dict) or "replications" not in c:continue
        for r in c["replications"]:
            if "reproduces" in str(r.get("note","")):
                rg+=1;continue
            t=(spectrum_key(c),r.get("base"))
            if t in s:
                sd+=1;continue
            s.add(t);g+=p;a+=r["n_datasets"]
    add(section,key,label,path,len(s),g,a,"one count per distinct spectrum and seed base; dropped %d replication(s) reproducing an already-counted cell and %d sharing a dataset set with another cell"%(rg,sd))
def collect_overnight():
    p=OVN/"e1_grid.json";g,a,n=cell_totals(load(p)["cells"]);add("overnight","E1","E1 sample size by participation ratio grid",p,n,g,a,"sum over cells of n_datasets + n_skipped")
    verification(OVN/"e1_conly_check.json","E1v","E1 C-only verification, four seed bases");p=OVN/"e2_spectrum_shape.json";g,a,n=cell_totals(load(p)["cells"])
    add("overnight","E2","E2 spectrum families",p,n,g,a,"sum over cells of n_datasets + n_skipped");verification(OVN/"e2_conly_check.json","E2v","E2 reversal verification, four seed bases")
    p=OVN/"e3_radial_angular.json";d=load(p);c=d["cells"]
    add("overnight","E3","E3 radial and angular decomposition",p,1,c["n_datasets"],c["n_datasets"],"one dataset set, %d null families x %d rules scored on it"%(len(d["config"]["families"]),len(d["config"]["rules"])))
    p=OVN/"e4_k_family.json";g,a,n=cell_totals(load(p)["cells"]);add("overnight","E4","E4 k-family sweep",p,n,g,a,"fresh data per K setting, seed 600_000 + 1000*i + k");p=OVN/"e5_null_families.json"
    d=load(p);c=d["cells"];add("overnight","E5","E5 alternative null families",p,1,c["n_datasets"],c["n_datasets"],"one dataset set, %d arms scored on it"%len(d["config"]["arms"]))
    for k,fn,l,a1 in(("E6","e6_statistics.json","E6 statistics","statistics"),("E7","e7_algorithms.json","E7 algorithms","algorithms"),("E8","e8_normalization.json","E8 normalization schemes","schemes")):
        p=OVN/fn;d=load(p);ns=d["config"]["n_datasets"];add("overnight",k,l,p,1,ns,ns,"one dataset set, %d %s scored on it"%(len(d["config"][a1]),a1))
    p=OVN/"e9_calibration_seeds.json";d=load(p);g,a,n=cell_totals(d["cells"])
    add("overnight","E9","E9 seed replications",p,n,g,a,"%d independent seed bases x %d datasets"%(d["config"]["n_replications"],d["config"]["n_datasets"]));p=OVN/"e10_dimension.json"
    g,a,n=cell_totals(load(p)["cells"]);add("overnight","E10","E10 dimension at fixed PR and n/d",p,n,g,a,"sum over cells of n_datasets + n_skipped; guard skips separate the two columns here")
    p=OVN/"e13_mechanism.json";d=load(p);g,a,n=cell_totals(d["cells"])
    add("overnight","E13","E13 mechanism and shift",p,n,g,a,"one dataset set per spectrum, %d null families x %d rules on each"%(len(d["config"]["families"]),len(d["config"]["rules"])))
    p=OVN/"e14_whitening.json";nh=load(p)["cells"]["n_checked"];add("overnight","E14","E14 whitening degeneracy",p,1,nh,nh,"n_checked datasets, factorization inspected rather than tested")
    p=OVN/"e15_boundary.json";g,a,n=cell_totals(load(p)["cells"]);add("overnight","E15","E15 spike sweep",p,n,g,a,"sum over cells of n_datasets")
    verification(OVN/"e16_nwb_check.json","E16","E16 within-between and Davies-Bouldin verification");p=OVN/"e18_boundary_surface.json";d=load(p);g,a,n=cell_totals(d["cells"])
    pr=d["config"]["n_datasets"];add("overnight","E18","E18 sign-change surface",p,n,n*pr,a,"%d cells x %d datasets requested"%(n,pr))
def collect_calibration():
    p=RESULTS/"handoff_20260802"/"calibration_paired"/"calibration_paired_summary.json";d=load(p)
    add("calibration","C31","Section 3.1 paired calibration",p,1,d["n_datasets"],d["n_datasets"],"M and C paired on one dataset set");p=RESULTS/"calibration"/"calibration_summary_corrected.json"
    d=load(p);dm=load(RESULTS/"calibration"/"calibration_summary_misspecified.json");s=d["selected_k_counts"]==dm["selected_k_counts"]
    add("calibration","C31r","Section 3.1 independent re-execution",p,1,d["n_datasets"],d["n_datasets"],"corrected and misspecified share one dataset set, %s by identical selected_k_counts"%("confirmed" if s else "NOT confirmed"))
    p=RESULTS/"e2e_calibration.json";d=load(p);add("calibration","C45","S4.5 end-to-end calibration at n = 406",p,1,d["n_datasets"],d["n_datasets"],"single cell")
def collect_phase1():
    p=newest(RESULTS/"phase1","r1d_geometry_sweeps.json")
    if p:
        d=load(p);g=a=n=0
        for sw in("sweep_A","sweep_B"):
            gg,aa,nn=cell_totals(d[sw]);g,a,n=(g+gg,a+aa,n+nn)
        c=d["companion"];g+=c["n_datasets"]+c.get("n_skipped",0);a+=c["n_datasets"];n+=1
        add("phase1","R1D","R1D regenerated geometry sweeps plus companion",p,n,g,a,"sweep A + sweep B + the G and S companion cell")
    p=newest(RESULTS/"phase1","r1a_headline_recalibration.json")
    if p:
        d=load(p);h=d["header"]
        add("phase1","R1A","R1A headline recalibration",p,h["n_blocks"],h["n_datasets"],h["n_datasets"]-d.get("n_skipped",0),"%d blocks x %d datasets"%(h["n_blocks"],h["n_datasets_per_block"]))
    p=newest(RESULTS/"phase1","r1b_nongaussian.json")
    if p:
        d=load(p);cs=d["cells"];gs={k:c for k,c in cs.items() if k.endswith("_gaussian")};pd={k:c for k,c in cs.items() if "gmm" in k};o={k:c for k,c in cs.items() if k not in gs and k not in pd}
        g,a,n=cell_totals(gs);add("phase1","R1B","R1B non-Gaussian robustness, Gaussian arm",p,n,g,a,"only the gaussian family is drawn from a single Gaussian");og,_,on=cell_totals(o)
        drop("R1Bng","R1B robustness, heavy-tailed and skewed arms",p,og,"%d cells drawn from multivariate t and lognormal laws: structureless, but not the single-Gaussian null"%on)
        pg,_,pn=cell_totals(pd);drop("R1Bgmm","R1B robustness, mixture arm",p,pg,"%d cells carry a planted two-component mixture"%pn)
    p=newest(RESULTS/"phase1","r1b_gmm_n28_refit.json")
    if p:
        d=load(p);g,_,n=cell_totals(d.get("cells",{}))
        drop("R1Bgmm2","R1B mixture arm refit at n = 28",p,g,"%d cells of planted two-component mixture at separation ratio %s"%(n,d["header"].get("r_target")))
    rs=sorted((RESULTS/"phase1").glob("*/r1c_selective_inference.json"))
    if rs:
        rs.sort(key=lambda q:load(q)["header"]["n_datasets_per_cell"]);f,ps=(rs[-1],rs[:-1]);d=load(f);g=n=s=r=0
        for k,c in d["cells"].items():
            m=c.get("manifest",{})
            if m.get("n_datasets",0)<=1:
                r+=1;continue
            if m.get("normalized") and k.endswith("_l2"):
                s+=1;continue
            g+=m["n_datasets"];n+=1
        add("phase1","R1C","R1C selective-inference benchmark",f,n,g,g,"%d generating cells; %d L2 cells share their raw twin's datasets"%(n,s))
        drop("R1Creal","R1C real-embedding cells",f,r,"single realizations of the two real cohorts, not generated data")
        for q in ps:
            dd=load(q)
            drop("R1Cpilot","R1C pilot at %d datasets per cell"%dd["header"]["n_datasets_per_cell"],q,None,"same seed rule as the full run, so its datasets are a prefix of the run counted above")
    q=newest(RESULTS/"phase1","r1d_conly_check.json")
    if q:
        dd=load(q)
        drop("R1Dv","R1D C-only verification",q,0,"re-tests the %d discordant datasets of R1D at %d draws on their original seeds; generates no new data"%(dd["header"]["total_C_only"],dd["header"]["n_draws_deep"]))
def collect_phase2():
    p=newest(RESULTS/"phase2","r2a_shift_reduction.json")
    if p:
        d=load(p);h=d["header"];g=h["n_datasets"]*len(d["real"])
        add("phase2","R2A","R2A reduction validation, real arm",p,len(d["real"]),g,g,"%d spike heights x %d datasets"%(len(d["real"]),h["n_datasets"]))
        drop("R2Ared","R2A reduction validation, reduced arm",p,None,"%d one-dimensional reductions, scalars rather than datasets"%len(d["reduced"]))
    for fn,k,l in(("r2b_shift_reduction_v2.json","R2B","R2B corrected reduction"),("r2c_shift_reduction_v3.json","R2C","R2C reduction by PR")):
        q=newest(RESULTS/"phase2",fn)
        if q:
            dd=load(q);drop(k,l,q,None,"%d one-dimensional reduction cells at %d repetitions each; scalars rather than datasets"%(len(dd["cells"]),dd["header"]["n_reps"]))
    p=newest(RESULTS/"phase2","r2d_reduction_vs_truth.json")
    if p:
        d=load(p);h=d["header"];g=h["n_datasets"]*len(d["cells"])
        add("phase2","R2D","R2D reduction against truth, truth arm",p,len(d["cells"]),g,g,"%d spike heights x %d datasets"%(len(d["cells"]),h["n_datasets"]))
def collect_popcov():
    _popcov_artifact(RESULTS/"popcov_control.json","PCC","Population-covariance control, cells 1 to 4","PCCdeep")
    _popcov_artifact(RESULTS/"popcov_control_cells567.json","PCC567","Population-covariance control, cells 5 to 7","PCC567deep");_popcov_screens()
def _popcov_artifact(p,key,label,deep_key):
    if not p.exists():return
    d=load(p);pr=d["config"]["n_datasets"];g=a=n=e=0;dp=0
    for c in d["cells"].values():
        if c.get("failed"):continue
        n+=1;g+=c["n_datasets"]+c.get("n_skipped",0);a+=c["n_datasets"];v=c.get("verification") or{}
        if v.get("triggered"):
            rs=[r for r in v["replications"] if "reproduces" not in str(r.get("note",""))];e+=len(rs)*pr;g+=len(rs)*pr;a+=sum((r["n_datasets"] for r in rs));dp+=(v.get("deep") or{}).get("n_retested",0)
    add("popcov",key,label,p,n,g,a,"%d cells x %d datasets plus %d verification datasets at further seed bases"%(n,pr,e))
    drop(deep_key,label+", 2,000-draw re-tests",p,0,"re-tests %d already-counted discordant datasets on their original seeds; generates no new data"%dp)
def _popcov_screens():
    s=[]
    for ne in("popcov_control_pr9_screen.json","popcov_control_cells567_screen.json"):
        p=RESULTS/ne
        if p.exists():s.append((p,load(p).get("cells",{})))
    p=RESULTS/"popcov_control_pr9_screen_ladder.json"
    if p.exists():s.append((p,load(p).get("rungs",{})))
    if not s:return
    a,b=({},{});f=RESULTS/"popcov_control_cells567.json"
    if f.exists():
        for k,c in load(f)["cells"].items():
            if c.get("failed"):continue
            b[k]=c["cell_offset"];a[c["spectrum"],str(c["param_value"])]=c["cell_offset"]
    sn=set();nx=n=0;ax,aa=([],[])
    for ph,cs in s:
        for k,c in cs.items():
            if c.get("failed"):continue
            o=c.get("cell_offset")
            if o is None:o=b.get(k)
            sc=(c["spectrum"],str(c["param_value"]));sg=(o if o is not None else k,)+sc
            if sg in sn:continue
            sn.add(sg)
            if sc in a and o in(None,a[sc]):
                nx+=1;ax.append(ph)
            else:
                n+=c["n_datasets"]+c.get("n_skipped",0);aa.append(ph)
    if nx:
        drop("PCCscreenpre","Population-covariance control screens of adopted cells",ax[0],None,"%d screened conditionings were adopted as full cells; same offset and same seed rule, so their datasets are a prefix of the cells counted above"%nx)
    if n:
        drop("PCCscreen","Population-covariance control screens of cells not adopted",aa[0],n,"screening and ladder rungs at conditionings no full cell uses; distinct datasets that nothing else counts, held out of the ledger by decision")
def collect_unreported_and_exclusions():
    p=OVN/"e15_boundary_tune20.json"
    if p.exists():
        g,a,n=cell_totals(load(p)["cells"]);add("unreported","E15t","E15 spike sweep, 20-sample tuning variant",p,n,g,a,"re-tuned spectra, so its datasets differ from E15")
    for fn,k,rf in(("e6_statistics_n400.json","E6s","e6_statistics.json"),("e7_algorithms_n400.json","E7s","e7_algorithms.json")):
        q=OVN/fn
        if q.exists():
            dd=load(q)
            drop(k,"%s superseded %d-dataset run"%(dd["experiment"],dd["config"]["n_datasets"]),q,dd["config"]["n_datasets"],"same seed base as %s, so these datasets are a prefix of the larger run already counted"%rf)
    for fn,k,l,r in(("e11_consensus.json","E11","E11 consensus clustering","operates on stored runs of the real cohorts; no null data generated"),("e12_representations.json","E12","E12 representation ladder","one real or randomly projected embedding per cell, not a Type I cell"),("e17_consensus_null.json","E17","E17 consensus null","surrogates are resampled label matrices of real runs, not datasets drawn from a Gaussian")):
        q=OVN/fn
        if q.exists():drop(k,l,q,None,r)
    q=RESULTS/"handoff_20260802"/"audit_gap_fills"/"separation_ratio_by_delta_full.csv"
    if q.exists():
        rs=len(q.read_text().strip().splitlines())-1;drop("planted","Planted-cluster realizations, Table 10",q,rs,"structure by construction; power cells rather than Type I cells")
    for q,l in((RESULTS/"handoff_20260802"/"propagated_null_summary.json","fibromyalgia"),(RESULTS/"mdd"/"propagated_null"/"propagated_null_summary.json","depression")):
        if q.exists():
            dd=load(q);drop("prop","Propagated-null draws, %s"%l,q,dd["n_draws"],"surrogate cohorts pushed through the full pipeline, not datasets drawn under the null hypothesis under test")
    q=RESULTS/"mdd"/"propagated_null"/"paired_rerun"/"paired_summary.json"
    if q.exists():
        dd=load(q);drop("prop2","Propagated-null paired re-run",q,dd["n_pairs"],"re-training of surrogate cohorts already counted above")
    for q,l,r in((RESULTS/"handoff_20260802"/"audit_gap_fills"/"audit_gap_fills_summary.json","Canonical three-way null on the real embedding","permutation draws within one real dataset, not datasets"),(RESULTS/"fm_fixed_arch_control"/"fixed_arch_control_summary.json","Fixed-architecture control runs","retrainings on the real cohort, not generated data")):
        if q.exists():drop("real",l,q,None,r)
def report():
    b="="*92;print(b);print("STRUCTURELESS DATASETS GENERATED UNDER THE SINGLE-GAUSSIAN NULL");print(b);gg=g=rg=ra=0
    for s,t in SECTIONS:
        rs=[r for r in INCLUDED if r["section"]==s]
        if not rs:continue
        print('\n%s'%t);print("-"*92);print("%-5s %-48s %7s %10s %9s"%("","experiment","cells","generated","analysed"));sg=sa=0
        for r in rs:
            print("%-5s %-48s %7d %10d %9d"%(r["key"],r["label"][:48],r["cells"],r["generated"],r["analysed"]));print("      %s"%r["artifact"]);print("      rule: %s"%r["rule"]);sg+=r["generated"]
            sa+=r["analysed"]
        print("%-5s %-48s %7s %10d %9d"%("","subtotal","",sg,sa));gg+=sg;g+=sa
        if s!="unreported":
            rg+=sg;ra+=sa
    print('\n'+b);print("EXCLUSIONS");print(b)
    for r in EXCLUDED:
        c="-" if r["count"] is None else "%d"%r["count"];print("%-9s %-62s %10s"%(r["key"],r["label"][:62],c))
        if r["artifact"]:print("      %s"%r["artifact"])
        print("      reason: %s"%r["reason"])
    print('\n'+b);print("TOTALS");print(b);print("%-62s %10d %9d"%("reported Type I experiments",rg,ra));ug,ua=(gg-rg,g-ra)
    if ug:print("%-62s %10d %9d"%("ran but not reported",ug,ua))
    print("%-62s %10d %9d"%("everything generated under the single-Gaussian null",gg,g));print('\ngenerated counts every dataset drawn; analysed counts only those that passed')
    print("the minimum-cluster-size guard and reached a p-value.");return(gg,g,rg,ra)
def main():
    collect_overnight();collect_calibration();collect_phase1();collect_phase2();collect_popcov();collect_unreported_and_exclusions();return report()
if __name__=="__main__":main()
