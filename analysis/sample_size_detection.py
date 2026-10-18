"""S25: the S22 design at n = 50 and n = 200; see docs/FOLLOWUP_PROTOCOL_S24_S26_20260925.md.

Reuses analysis/scope_detection.py unchanged (its recorded S22 checksums stay valid). Only the four-group size
rule is extended to sample sizes not divisible by four, as the protocol specifies.
"""
import os
for v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[v]='1'
import argparse,json,platform,sys,time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import scipy,sklearn
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'analysis'))
import scope_detection as base
PROTOCOL=R/'docs/FOLLOWUP_PROTOCOL_S24_S26_20260925.md'
_sizes=base.sizes

def sizes(fam,n):
    if fam=='tetra':
        return [n-3*(n//4)]+[n//4]*3
    return _sizes(fam,n)
base.sizes=sizes

def config(root,n,ns):
    c=base.config(root,n,ns)
    c.update(bootstrap_address=81,protocol_sha256=base.sha(PROTOCOL),study='S25')
    c['source_sha256'][str(Path(__file__).relative_to(R)).replace('\\','/')]=base.sha(Path(__file__))
    return c

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--output',type=Path,required=True)
    a.add_argument('--n',type=int,required=True,choices=(50,200))
    a.add_argument('--workers',type=int,default=16)
    a.add_argument('--pilot',action='store_true')
    a=a.parse_args()
    root=2026092504 if a.pilot else 20260924
    ns=[3]*len(base.G) if a.pilot else [1000,1000]+[250]*(len(base.G)-2)
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
    ts=[(root,a.n,s,q,g,i,199) for s in range(2) for q in (5,10) for g in range(len(base.G)) for i in range(ns[g]) if (s,q,base.G[g],i) not in done]
    t0=time.time()
    with Pool(a.workers) as pool,jp.open('a',encoding='utf-8') as j:
        for r in pool.imap_unordered(base.one,ts,chunksize=1):
            j.write(json.dumps(r)+'\n');j.flush();rows.append(r)
            if len(rows)%250==0:
                print(f'{len(rows)}/{sum(ns)*4}; {time.time()-t0:.0f}s',flush=True)
    end=dict(n=len(rows),errors=sum('error' in r for r in rows),records_sha256=base.sha(jp),execution_wall_s=time.time()-t0)
    (a.output/'execution.json').write_text(json.dumps(end,indent=2),encoding='utf-8')
    print(json.dumps(end),flush=True)
    if end['errors']:
        raise RuntimeError('Errors retained; do not summarize as completed')

if __name__=='__main__':
    main()
