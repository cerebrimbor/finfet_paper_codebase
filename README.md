# finfet-paper_model_datas

Code, data and figures for a manuscript on adjacent-dwell correlation in
random-telegraph noise (RTN): when the correlation between successive filled
and empty dwell times detects hidden trap sub-states, and when it misses them.

## Layout

| Path | Contents |
|---|---|
| `stochastic_ctmc/` | Python package: generators, Gillespie and SDE simulation, dwell-time laws, covariance identity, blind-spot construction, chi-squared test |
| `scripts/` | `generate_figures.py`, `reproduce_blind_spot_k33.py`, `verify_covariance_identity.py` (symbolic proof checks) |
| `verification/` | Independent checks of the paper's claims |
| `tests/` | Unit and statistical tests |
| `figures/`, `data/` | Paper figures and the numbers behind them |
| `device_provenance/` | BSIM-CMG model cards, ngspice decks and Id-Vg sweeps for the threshold-shift check |

## Setup

Python 3.10 or later.

```bash
git clone https://github.com/cerebrimbor/finfet-paper_model_datas.git
cd finfet-paper_model_datas
pip install -r requirements.txt
```

## Run

From the repository root:

```bash
pytest -m "not slow"                       
pytest                                     
python verification/run_all.py            
python scripts/verify_covariance_identity.py
python scripts/generate_figures.py         
```

Figure images are byte-identical only with the matplotlib version that made
them (3.10.9); the data in `data/` does not depend on it.

## License

MIT. See `LICENSE`. The BSIM-CMG model cards in `device_provenance/` are
Berkeley's published sample cards; see `device_provenance/README.md`.
