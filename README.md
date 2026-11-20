# Kinetic theory, DSMC and event-driven MD code for rough granular gases with velocity-dependent Coulomb–Amontons friction

Supplementary code for the article
*"Rotational freezing and history-dependent relaxation in a dilute granular gas with velocity-dependent
Coulomb–Amontons friction: a per-collision kinetic theory beyond mean field and its DSMC and molecular-dynamics tests"*
by T. A. Tazhimov, A. R. Sharden, I. T. Alishirov (Kazan National Research Technical University – KAI).

**Version 2.0** (revised manuscript). New in this version: event-driven molecular dynamics (`md_event.py`),
non-Gaussian diagnostics (`diag_nongauss.py`, `diag_bulk.py`), accuracy of the asymptotic formulas (`asym_check.py`),
sticking contribution and monotonicity scan (`sliding_check.py`), realisation checks (`check_CEG.py`), revised figure scripts.

Units: m = σ = k_B = 1, n σ² = 1, I = q m σ²/4.

## Requirements
Python ≥ 3.10, `pip install -r requirements.txt` (numpy, scipy, matplotlib; numba for the molecular dynamics).

## Files
| file | content | article |
|---|---|---|
| `theory.py` | `Model` class: Walton rule + Stribeck friction μ(g); 1-D quadratures F1, F2; two-temperature ODEs; Monte Carlo check of the collision integrals | Eqs. (13)–(17), Table 1 |
| `dsmc2.py` | homogeneous DSMC (Bird acceptance–rejection), optional stochastic thermostat | Sec. 6.1 |
| `md_event.py` | event-driven molecular dynamics of rough hard spheres, same collision rule (numba) | Sec. 6.3, Fig. 8 |
| `diag_nongauss.py` | cumulants, spin–velocity correlations and production rates measured on DSMC snapshots | Sec. 3.6, Table 2 |
| `diag_bulk.py` | cooling rate of the bulk in the frozen state | Sec. 3.6 |
| `walton_dsmc_check.py`, `run_const.py` | constant-friction runs (writes `res.pkl`) | Fig. 3 |
| `phase.py`, `asym_check.py` | phase diagram, exact zeros, first- and second-order threshold formulas | Fig. 2, Table 3 |
| `driven.py`, `run_driven.py`, `sliding_check.py` | heating curve, DSMC steady states, sticking contribution, 240-point monotonicity scan | Fig. 4, Table 4, Sec. 5.1 |
| `scan_extended.py` | second monotonicity scan, 384 sets varying e, β0, q (~10 min, 2 cores) | Sec. 5.1 (ii) |
| `run_cool.py` | Stribeck cooling: DSMC vs per-collision theory vs mean field (writes `cool.pkl`) | Fig. 5, Fig. 6 |
| `run_freeze.py`, `check_CEG.py` | history dependence of the frozen population; realisation checks | Table 5, Fig. 7 |
| `figs.py` | Figs. 1, 3, 6 (files `fig1_F.pdf`, `fig3_lock.pdf`, `fig5_spin.pdf`) | |
| `fig_phase.py`, `fig_driven.py`, `fig_cooling_rate.py`, `fig7.py`, `fig_md.py` | Figs. 2, 4, 5, 7, 8 | |

## Reproduce
```bash
python3 theory.py          # Table 1
python3 phase.py; python3 asym_check.py      # Fig. 2 data, Table 3
python3 run_const.py       # Fig. 3 data
python3 run_cool.py        # Figs. 5, 6 data
python3 diag_nongauss.py; python3 diag_bulk.py   # Table 2          (~10 min)
python3 driven.py; python3 run_driven.py; python3 sliding_check.py   # Fig. 4, Table 4, scan (~15 min)
python3 scan_extended.py                                            # extended scan, Sec. 5.1 (~10 min)
python3 run_freeze.py; python3 check_CEG.py  # Table 5, Fig. 7 data (~10 min)
python3 md_event.py abc; python3 md_event.py sB   # molecular dynamics (~40 min)
python3 figs.py; python3 fig_phase.py; python3 fig_driven.py; python3 fig_cooling_rate.py; python3 fig7.py; python3 fig_md.py
```
All random seeds are fixed.

## License
MIT (see `LICENSE`). Please cite the article and this repository (see `CITATION.cff`).
