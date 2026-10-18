"""S22: unequal group sizes, four groups and n = 100; see docs/FOLLOWUP_PROTOCOL_S22_20260924.md."""
import os
for v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[v]='1'
import argparse,hashlib,json,platform,sys,time,traceback
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import scipy,sklearn
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'reproducibility/s16/modules'))
import angular_gof as gof
import nullcal_validation as nv
from asymmetric_detection import PROJ,spectrum,sha
EXP=76
G=('calibration','null','tri4','triuneq4','pairuneq4','tetra4','shift4')
PROTOCOL=R/'docs/FOLLOWUP_PROTOCOL_S22_20260924.md'
TETRA=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])/np.sqrt(3)

def sd(root,n,*a):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,(root,EXP,n,*a))).encode()).digest()[:16],'big')

def sizes(fam,n):
    if fam=='tri':
        return [n-2*(n//3),n//3,n//3]
    if fam=='triuneq':
        return [n//2,n//4,n-n//2-n//4]
    if fam=='pairuneq':
        return [n-n//4,n//4]
    if fam=='tetra':
        return [n//4]*4
    return [n]

def means(name,rot,lam,perm):
    """Component means (K x d) and row labels; centered families are recentered on realized sizes."""
    if name in ('calibration','null'):
        return None,None
    fam,de=name[:-1],float(name[-1])
    n=len(perm)
    m=np.sqrt(lam[:3])*rot[:,:3]
    if fam=='shift':
        return (de*(m[:,0]+m[:,1])/2)[None,:],np.zeros(n,dtype=int)
    if fam in ('tri','triuneq'):
        c=np.array([[np.cos(t),np.sin(t),0.] for t in (0.,2*np.pi/3,4*np.pi/3)])
    elif fam=='pairuneq':
        c=np.array([[1.,0.,0.],[-1.,0.,0.]])
    else:
        c=TETRA
    M=de*c@m.T
    sz=sizes(fam,n)
    assert sum(sz)==n and len(sz)==len(M)
    lab=np.repeat(np.arange(len(sz)),sz)[np.argsort(np.argsort(perm))]
    w=np.bincount(lab,minlength=len(M))/n
    return M-w@M,lab

def one(t):
    root,n,s,q,g,i,nd=t
    sid=0 if g==0 else 1
    ds=sd(root,n,0,s,sid,i)
    r=dict(n=n,spectrum=s,q=q,group=G[g],index=i,data_seed=str(ds))
    try:
        rng=np.random.default_rng(ds)
        z=rng.normal(size=(n,119))
        rot=np.linalg.qr(rng.normal(size=(119,119)))[0]
        perm=rng.permutation(n)
        lam=spectrum(s)
        x=(z*np.sqrt(lam))@rot.T
        M,lab=means(G[g],rot,lam,perm)
        p=PROJ[:,:q]
        cov=(p.T@rot*lam)@(rot.T@p)
        between=shift=0.
        if M is not None:
            x=x+M[lab]
            w=np.bincount(lab,minlength=len(M))/n
            pm=M@p
            mbar=w@pm
            dm=pm-mbar
            between=float(np.trace(np.linalg.solve(cov,(dm.T*w)@dm)))
            shift=float(mbar@np.linalg.solve(cov,mbar))
        xp=nv.normalize(nv.normalize(x)@p)
        rs=sd(root,n,2,s,q,g,i)
        bs=sd(root,n,3,s,q,g,i)
        u=float(np.random.default_rng(sd(root,n,4,s,q,g,i)).uniform())
        r.update(group_sizes=None if lab is None else np.bincount(lab).tolist(),source_sha256=hashlib.sha256(x.astype('<f8').tobytes()).hexdigest(),projected_sha256=hashlib.sha256(xp.astype('<f8').tobytes()).hexdigest(),projected_between_scatter=between,projected_mean_shift=shift,reference_seed=str(rs),bootstrap_seed=str(bs),boundary_uniform=u)
        r.update(gof.evaluate(xp,draws=nd,seed=rs,bootstrap_seed=bs,auxiliary=200))
    except Exception:
        r['error']=traceback.format_exc()
    return r

def config(root,n,ns):
    return dict(root_seed=root,experiment=EXP,n=n,d=119,qs=[5,10],spectra=['two_block_1_0.01','exponential_tau10'],groups=dict(zip(G,ns)),draws=199,auxiliary=200,bootstrap_resamples=2000,bootstrap_address=77,diagnostics=['projected_between_scatter','projected_mean_shift'],contrasts=[['shift4','tri4'],['shift4','triuneq4'],['shift4','pairuneq4'],['shift4','tetra4']],projection='S19 fixed nested projection (experiment 72 address)',protocol_sha256=sha(PROTOCOL),source_sha256={str(f.relative_to(R)).replace('\\','/'):sha(f) for f in [Path(__file__),R/'analysis/asymmetric_detection.py',R/'analysis/calibrated_detection.py',*(R/'reproducibility/s16/modules').glob('*.py')]},versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__))

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--output',type=Path,required=True)
    a.add_argument('--n',type=int,required=True,choices=(28,100))
    a.add_argument('--workers',type=int,default=16)
    a.add_argument('--pilot',action='store_true')
    a=a.parse_args()
    root=2026092403 if a.pilot else 20260924
    ns=[3]*len(G) if a.pilot else [1000,1000]+[250]*(len(G)-2)
    cfg=config(root,a.n,ns)
    a.output.mkdir(parents=True,exist_ok=True)
    cp=a.output/'config.json'
    if cp.exists():
        assert json.loads(cp.read_text(encoding='utf-8'))==cfg,'Configuration mismatch'
    else:
        cp.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    jp=a.output/'records.jsonl'
    rows=[json.loads(l) for l in jp.read_text(encoding='utf-8').splitlines()] if jp.exists() else []
    done={(r['spectrum'],r['q'],r['group'],r['index']) for r in rows}
    ts=[(root,a.n,s,q,g,i,199) for s in range(2) for q in (5,10) for g in range(len(G)) for i in range(ns[g]) if (s,q,G[g],i) not in done]
    t0=time.time()
    with Pool(a.workers) as pool,jp.open('a',encoding='utf-8') as j:
        for r in pool.imap_unordered(one,ts,chunksize=1):
            j.write(json.dumps(r)+'\n')
            j.flush()
            rows.append(r)
            if len(rows)%250==0:
                print(f'{len(rows)}/{sum(ns)*4}; {time.time()-t0:.0f}s',flush=True)
    end=dict(n=len(rows),errors=sum('error' in r for r in rows),records_sha256=sha(jp),execution_wall_s=time.time()-t0)
    (a.output/'execution.json').write_text(json.dumps(end,indent=2),encoding='utf-8')
    print(json.dumps(end),flush=True)
    if end['errors']:
        raise RuntimeError('Errors retained; do not summarize as completed')

if __name__=='__main__':
    main()
