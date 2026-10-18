# Figures

The PDFs preserve vector text and shapes, and the PNGs are rendered from those PDFs. The filenames still use the earlier numbering. The table below shows where each file appears in the PLOS ONE submission.

| File | Figure | Built by |
|---|---|---|
| `Figure4.png` | Fig 2 | `rebuild_heatmap.py`, from the sixty saved grid rates |
| `Main_Figure2.png` | Fig 3 | `main_figures.py`, from the S31 summary |
| `Main_Figure3.png` | Fig 4 | `main_figures.py`, from the S18 summary |
| `Main_Figure4.png` | Fig 5 | `main_figures.py`, from the S19, S20, S22 and S25 summaries |
| `Figure9.png` | Fig 6 | `rebuild_figures.py`; Monte Carlo reference p-values, not permutation tests |
| `Figure1.png` | Figure S1 | original vector geometry, recolored by `pink_theme.py` |
| `Figure2.png` | Figure S2 | `rebuild_figures.py`, from `results/overnight/e15_boundary.json` |
| `Figure5.png`, `Figure6.png` | Figures S3 and S4 | original vector geometry, recolored by `pink_theme.py` |
| `Figure7.png`, `Figure8.png` | Figures S5 and S6 | `followup_figures.py`, from the recorded follow-ups in `results/revision_20260924/` |
| `Figure3.png` | Figure S7 | `followup_figures.py`; its third panel retains the historical centered-law control from Table S61 |

Fig 1 uses the closed-form result in Materials and methods and does not rely on stored data. The submitted Fig 2 and Fig 6 show the same data as `Figure4` and `Figure9`, with the layout set to the 7.5 in PLOS width.

Run `python reproduce.py figures` from the repository root to regenerate these files in the correct order.
