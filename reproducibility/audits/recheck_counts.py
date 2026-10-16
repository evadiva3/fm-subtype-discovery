from pathlib import Path
import math, json
import numpy as np

def paired_test(c_only, m_only, n):
    d = c_only + m_only
    p = min(1., 2 * sum(math.comb(d, i) for i in range(min(c_only, m_only)+1)) / 2**d)
    difference =(c_only - m_only) / n
    se = math.sqrt(max(0, (d/n-difference*difference)/(n-1)))
    return dict(c_only=c_only,m_only=m_only,n=n,difference_C_minus_M=difference,
                exact_two_sided_McNemar_p=p,paired_difference_normal95=[difference-1.95996398454*se,difference+1.95996398454*se])

cells =[
 ("exp_pr4.5",8,193),("power_pr3.0",42,211),("power_pr4.5",106,97),("power_pr6.0",63,35),
 ("one_spike_pr3.0",89,3),("one_spike_pr4.5",360,12),("one_spike_pr6.0",245,6),
 ("two_block_pr3.0",97,5),("two_block_pr4.5",341,12),("two_block_pr6.0",252,2),
 ("grid_pr3.0_n28",6,555),("grid_pr4.5_n28",1,187),("grid_pr9.0_n119",3,171),
 ("negative_within_between",72,30),("negative_Davies_Bouldin",35,10)]
paired ={k:paired_test(c,m,1600) for k,c,m in cells}
ordered=sorted(paired,key=lambda k:paired[k]["exact_two_sided_McNemar_p"])
running=0.
for rank,k in enumerate(ordered):
    running=max(running,min(1.,(len(cells)-rank)*paired[k]["exact_two_sided_McNemar_p"]))
    paired[k]["Holm_p_over_15_reported_verification_cells"]=running

def gaussian_eigen_mixture(n, eigenvalues, rng, *, delta=0., axis=0):
    lam=np.asarray(eigenvalues,float)
    z=rng.standard_normal((n,len(lam)))
    q=np.linalg.qr(rng.standard_normal((len(lam),len(lam))))[0]
    base=(z*np.sqrt(lam))@q.T
    labels=rng.choice([-1.,1.],size=n)
    shift=delta*np.sqrt(lam[axis])*labels[:,None]*q[:,axis][None,:]
    return base+shift,base,q,labels,shift

rng=np.random.default_rng(20260922)
lam=np.exp(-np.arange(12)/3)
x,base,q,labels,shift=gaussian_eigen_mixture(80,lam,rng,delta=4.,axis=3)
cov=(q*lam)@q.T
alignment=np.max(np.abs(shift@q[:,3]-4*np.sqrt(lam[3])*labels))
off=shift@q;off[:,3]=0
assert np.max(np.abs(off)) < 1e-12
assert np.allclose(cov@q[:,3],lam[3]*q[:,3])
assert np.allclose(x-base,shift)
old_alignment=float(abs(q[0,0]))
assert old_alignment < .999
proj=np.linalg.qr(rng.standard_normal((12,10)))[0]
assert np.linalg.norm(proj[:,:5].T@q[:,3])**2 <= np.linalg.norm(proj.T@q[:,3])**2+1e-12
checks=dict(eigenvector_residual=float(np.max(np.abs(cov@q[:,3]-lam[3]*q[:,3]))),
            orthogonal_shift_residual=float(np.max(np.abs(off))),
            shift_alignment_error=float(alignment),
            coordinate0_leading_eigenvector_absolute_dot=old_alignment,
            nested_projection_check="passed",
            scope="Deterministic implementation checks on a new synthetic example; not manuscript power results.")


Path(__file__).with_name("rechecked_results.json").write_text(json.dumps(dict(paired_comparisons=paired,implementation_checks=checks),indent=2))
print("All paired-count and geometry checks completed.")
