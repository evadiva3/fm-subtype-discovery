
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
        for script in ("rebuild_figures.py","rebuild_heatmap.py","pink_theme.py","followup_figures.py","main_figures.py"):
            run(ROOT/"paper/build"/script)
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
