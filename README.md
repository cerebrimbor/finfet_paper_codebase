# finfet_paper_codebase

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.CONCEPT_ID.svg)](https://doi.org/10.5281/zenodo.23766567)

Code, data and figures for a manuscript on adjacent-dwell correlation in
random-telegraph noise (RTN): when the correlation between successive filled
and empty dwell times detects hidden trap sub-states, and when it misses them.

## Paper

Kaustubh Chaturvedi, *Detecting Hidden Trap States from Adjacent Dwell Times
in FinFET Random-Telegraph Noise*, preprint, Zenodo (2026).
https://doi.org/10.5281/zenodo.CONCEPT_ID

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
git clone https://github.com/cerebrimbor/finfet_paper_codebase.git
cd finfet_paper_codebase
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

## Citation

If you use this code, please cite the paper:

```bibtex
@misc{chaturvedi2026hiddentrap,
  author    = {Chaturvedi, Kaustubh},
  title     = {Detecting Hidden Trap States from Adjacent Dwell Times in
               {FinFET} Random-Telegraph Noise},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.CONCEPT_ID},
  note      = {Preprint}
}
```

## License

MIT. See `LICENSE`. The BSIM-CMG model cards in `device_provenance/` are
Berkeley's published sample cards; see `device_provenance/README.md`.
