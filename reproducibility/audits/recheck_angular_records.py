#recompute Tables S24 and S25 from the recovered p-values
from pathlib import Path
import csv,json
from collections import defaultdict
HERE=Path(__file__).resolve().parent
rows=list(csv.DictReader((HERE/"recovered_angular_records.csv").open(newline="",encoding="utf-8")))
expected=json.loads((HERE/"reported_angular_table_rates.json").read_text())
cells=defaultdict(list)
for r in rows:
    cells[f"{r["family"]}_n{r["n"]}_q{int(r["q"]):02}"].append(r)
for name,sub in sorted(cells.items()):
    rates=[100*sum((float(r[a])<=0.05 for r in sub))/len(sub) for a in ("W","D4","K","E")]
    assert rates==expected[name],(name,rates,expected[name])
assert len(rows)==7200
assert len({r["source_data_sha256"] for r in rows})==2400
print("PASS: all 36 cells match; 7,200 analyses; 2,400 distinct source datasets.")
