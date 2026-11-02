import hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1]
MOVES={
    "results/revision_20260923":"results/recorded/s6_s9",
    "results/revision_20260924":"results/recorded/s18_s22",
    "results/revision_20260925":"results/recorded/s24_s29",
    "results/revision_20260926":"results/recorded/s31_s32",
    "results/revision_20261008":"results/recorded/s35_s36",
    "results/overnight":"results/exploratory/screens_e1_e18",
    "results/phase1":"results/exploratory/headline_recalibration",
    "results/phase2":"results/exploratory/mechanism_runs",
    "results/angular_v8":"results/angular/v8",
    "results/angular_v8_fourth":"results/angular/v8_fourth",
    "results/angular_v8_restarts":"results/angular/v8_restarts",
    "results/angular_v9":"results/angular/v9",
    "results/angular_v9_balanced":"results/angular/v9_balanced",
    "results/embeddings_v8":"results/angular/embeddings_v8",
    "results/embeddings_v9":"results/angular/embeddings_v9",
    "results/audit_v9_claims.json":"results/angular/audit_v9_claims.json",
    "results/validation_v7":"results/validation/v7",
    "results/validation_v7_resolution":"results/validation/v7_resolution",
    "results/handoff_20260802":"results/historical/handoff",
    "results/calibration":"results/historical/calibration",
    "results/mdd":"results/cohorts/mdd",
    "results/fm_fixed_arch_control":"results/cohorts/fm/fixed_arch_control",
    "results/clinical_validation_results.csv":"results/cohorts/fm/clinical_validation_results.csv",
    "results/stability_summary.csv":"results/cohorts/fm/stability_summary.csv",
    "results/subject_exclusions.csv":"results/cohorts/fm/subject_exclusions.csv",
}
for f in ("ambient_control.json","cell2_direction_check.json","e2e_calibration.json","e2e_calibration_pvalues.npy","grid_verification.json",
          "headline_embedding.json","popcov_control.csv","popcov_control.json","popcov_control_cells567.csv","popcov_control_cells567.json",
          "popcov_control_cells567_screen.json","popcov_control_pr9_screen.json","popcov_control_pr9_screen_ladder.json","raw_feature_ladder.json",
          "restart_sensitivity.json","reversal_verification.json","seed_overlap_audit.json","shape_aware_article_geometry.json",
          "shape_aware_detection_floor.json","shape_aware_minor_axis.json","sign_rule_audit.json","statistic_verification.json","two_error_decomposition.json"):
    MOVES["results/"+f]="results/exploratory/controls/"+f

def here(p):
    p=str(p).replace("\\","/")
    for a in sorted(MOVES,key=len,reverse=True):
        if p==a or p.startswith(a+"/"):return R/(MOVES[a]+p[len(a):])
    return R/p

def recorded(p):
    a=R/"archive/recorded_source"/str(p).replace("\\","/")
    return a if a.is_file() else here(p)

def same(p,digest):
    return any(hashlib.sha256(f.read_bytes()).hexdigest()==digest for f in {recorded(p),here(p)} if f.is_file())

def summary_ok(new,saved,src):
    new,saved=dict(new),dict(saved)
    return same(src,saved.pop("summary_source_sha256")) and new.pop("summary_source_sha256",None) is not None and new==saved
