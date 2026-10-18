"""Independent follow-up, not a reproduction of the missing FINAL spectrum run.

Run: python run_validation.py --datasets 200 --workers 4
Requires numpy, scipy and scikit-learn; records versions and per-dataset outcomes.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import argparse,hashlib,json,platform,sys,time,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import scipy,sklearn
from scipy.stats import binomtest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'modules'))
import angular_gof as gof
import angular_null as an
import nullcal_validation as nv
ROOT=20260922
D=119
N=28
GROUPS=('calibration_null','evaluation_null','leading_mixture')
PROJECTIONS=('nested5','nested10','oracle5')

def seed(*address):return nv.stream_seed(ROOT,61,*address)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,obj):path.write_text(json.dumps(obj,indent=2,sort_keys=True),encoding='utf-8')
def participation(x):
    x=x-x.mean(axis=0);e=np.linalg.svd(x,compute_uv=False)**2
    return float(e.sum()**2/(e@e))

def tune_spectrum():
    # Fresh independent pilot: one dominant eigenvalue, equal residual block.
    normals=[np.random.default_rng(seed(0,0,i,0)).normal(size=(N,D)) for i in range(100)]
    lo,hi=1e-5,1.
    for _ in range(24):
        eps=np.sqrt(lo*hi);lam=np.r_[1.,np.full(D-1,eps)]
        prs=[participation(nv.normalize(z*np.sqrt(lam))) for z in normals]
        if np.mean(prs)<4.5:lo=eps
        else:hi=eps
    return lam,dict(family='two_block',m=1,epsilon=float(eps),pilot_datasets=100,
                    target_centered_normalized_PR=4.5,achieved_pilot_PR=float(np.mean(prs)),
                    source='fresh tuning, not the missing FINAL spectrum artifact')

def projection(rotation,kind):
    fixed=np.linalg.qr(np.random.default_rng(seed(1,0,0,0)).normal(size=(D,10)))[0]
    if kind=='nested5':return fixed[:,:5]
    if kind=='nested10':return fixed
    # Simulation-only oracle: leading covariance eigenvector plus four nuisance axes.
    return rotation[:,:5]

def one(task):
    group,kind,index,lam,draws=task
    gid=GROUPS.index(group);pid=PROJECTIONS.index(kind)
    # Evaluation null and mixture share base noise, rotation and mixture signs.
    sid=0 if group=='calibration_null' else 1
    data_seed=seed(2,sid,index,0)
    rng=np.random.default_rng(data_seed)
    z=rng.normal(size=(N,D));rotation=np.linalg.qr(rng.normal(size=(D,D)))[0]
    x=(z*np.sqrt(lam))@rotation.T
    signs=rng.choice([-1.,1.],size=N)
    delta=4. if group=='leading_mixture' else 0.
    axis=rotation[:,0];x+=delta*np.sqrt(lam[0])*signs[:,None]*axis[None,:]
    p=projection(rotation,kind);xp=nv.normalize(nv.normalize(x)@p)
    covariance=(p.T@rotation*lam)@(rotation.T@p)
    v=p.T@axis
    row=dict(group=group,projection=kind,index=index,n=N,d=D,q=p.shape[1],delta=delta,
             source_seed=str(data_seed),source_data_sha256=hashlib.sha256(x.astype('<f8').tobytes()).hexdigest(),
             projected_sha256=hashlib.sha256(xp.astype('<f8').tobytes()).hexdigest(),
             retained_axis_length_squared=float(v@v),
             projected_squared_separation=float(delta**2*lam[0]*v@np.linalg.solve(covariance,v)))
    try:
        row.update(gof.evaluate(xp,draws=draws,seed=seed(3,gid,index,100+pid),
                                bootstrap_seed=seed(3,gid,index,200+pid),auxiliary=200))
    except Exception:row['error']=traceback.format_exc()
    return row

def interval(k,n):
    ci=binomtest(k,n).proportion_ci(method='exact');return [float(ci.low),float(ci.high)]

def rotation_audit():
    rows=[]
    for q in (5,10):
        for index in range(100):
            rng=np.random.default_rng(seed(4,q,index,0));x=nv.normalize(rng.normal(size=(N,q))*np.geomspace(1,.01,q))
            r=np.linalg.qr(rng.normal(size=(q,q)))[0]
            a,info=an.tyler_shape(x);b,info2=an.tyler_shape(x@r)
            wa=an.whiten(x,a);wb=an.whiten(x@r,b)
            sa,ka=gof.whitened_scores(wa);sb,kb=gof.whitened_scores(wb)
            rows.append(dict(q=q,index=index,shape_relative_error=float(np.linalg.norm(b-r.T@a@r)/np.linalg.norm(a)),
                             gram_max_error=float(np.max(np.abs(wa@wa.T-wb@wb.T))),
                             score_absolute_errors={k:abs(sa[k]-sb[k]) for k in sa},same_selected_k=ka==kb,
                             residual=max(info['whitened_moment_residual'],info2['whitened_moment_residual'])))
    return dict(n_comparisons=len(rows),rows=rows,max_shape_relative_error=max(r['shape_relative_error'] for r in rows),
                max_gram_error=max(r['gram_max_error'] for r in rows),
                max_score_errors={a:max(r['score_absolute_errors'][a] for r in rows) for a in ('W','D4','K')},
                different_selected_k=sum(not r['same_selected_k'] for r in rows))

def summarize(rows,config):
    out=dict(config=config,cells={},errors=[r for r in rows if 'error' in r],
             interpretation='Independent fresh-spectrum follow-up. Calibration-threshold estimates are exploratory; their uncertainty is not included in conditional power intervals.')
    for kind in PROJECTIONS:
        cal=[r for r in rows if r['projection']==kind and r['group']=='calibration_null']
        cutoffs={}
        for arm in gof.ARMS:
            valid=[r['p'][arm] for r in cal if 'error' not in r]
            candidates=sorted(set(valid))
            # Largest attainable observed p cutoff with empirical size <= 5%; ties retained.
            cutoffs[arm]=max([p for p in candidates if sum(x<=p for x in valid)<=.05*len(cal)],default=0.)
        for group in GROUPS:
            sub=[r for r in rows if r['projection']==kind and r['group']==group];n=len(sub)
            valid=[r for r in sub if 'error' not in r];arms={}
            for arm in gof.ARMS:
                count=sum(r['p'][arm]<=.05 for r in valid);adj=sum(r['p'][arm]<=cutoffs[arm] for r in valid)
                arms[arm]=dict(nominal_count=count,nominal_rate=count/n,nominal_exact95=interval(count,n),
                    independently_calibrated_p_cutoff=cutoffs[arm],calibrated_count=adj,calibrated_rate=adj/n,
                    calibrated_conditional_exact95=interval(adj,n))
            out['cells'][group+'_'+kind]=dict(n=n,numerical_failures=n-len(valid),arms=arms,
                observed_guard_failures=sum(r['guard_failed'] for r in valid),
                reference_guard_failures=sum(r['reference_guard_failures'] for r in valid),
                median_retained_axis_length=float(np.median([r['retained_axis_length_squared'] for r in sub])),
                median_projected_squared_separation=float(np.median([r['projected_squared_separation'] for r in sub])))
    # No omitted failures: rate bounds identify what unresolved errors could change.
    for cell in out['cells'].values():
        f=cell['numerical_failures'];n=cell['n']
        for a in cell['arms'].values():
            a['nominal_rate_bounds_if_failures_unresolved']=[a['nominal_count']/n,(a['nominal_count']+f)/n]
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--datasets',type=int,default=200);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--draws',type=int,default=199);args=ap.parse_args()
    out=HERE/'results';out.mkdir(exist_ok=True);records=out/'records';records.mkdir(exist_ok=True)
    lam,spectrum=tune_spectrum()
    cfg=dict(datasets_per_cell=args.datasets,draws=args.draws,auxiliary=200,spectrum=spectrum,root_seed=ROOT,
             versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),
             source_sha256={p.name:sha(p) for p in [Path(__file__),*(HERE/'modules').glob('*.py')]})
    prior=out/'config.json'
    if prior.exists() and json.loads(prior.read_text())!=cfg:raise RuntimeError('Configuration differs from existing records; use a separate output directory')
    save(prior,cfg);print('Configuration',spectrum,flush=True)
    if not (out/'rotation_audit.json').exists():save(out/'rotation_audit.json',rotation_audit())
    tasks=[];rows=[]
    for group in GROUPS:
        for kind in PROJECTIONS:
            for i in range(args.datasets):
                file=records/f'{group}_{kind}_{i:04}.json'
                if file.exists():rows.append(json.loads(file.read_text()))
                else:tasks.append((group,kind,i,lam,args.draws))
    start=time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(one,t) for t in tasks]
        for future in as_completed(futures):
            row=future.result();save(records/f"{row['group']}_{row['projection']}_{row['index']:04}.json",row);rows.append(row)
            if len(rows)%50==0:print(f'{len(rows)} / {args.datasets*9} completed; {time.time()-start:.0f}s',flush=True)
    result=summarize(rows,cfg);result['execution_wall_seconds']=time.time()-start;save(out/'summary.json',result)
    print('DONE; failures',len(result['errors']),flush=True)
if __name__=='__main__':main()
