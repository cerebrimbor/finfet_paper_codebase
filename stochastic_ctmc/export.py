# CSV and HDF5 export of the data behind the figures. h5py is optional.

from __future__ import annotations

import csv
from typing import Mapping

import numpy as np

__all__ = ["export_columns_csv", "hdf5_available", "export_bundle_hdf5", "read_bundle_hdf5"]


def export_columns_csv(path: str, columns: Mapping[str, np.ndarray]) -> str:
    names = list(columns.keys())
    arrays = [np.asarray(columns[n]).ravel() for n in names]
    if arrays:
        length = len(arrays[0])
        if any(len(a) != length for a in arrays):
            lengths = {n: len(a) for n, a in zip(names, arrays)}
            raise ValueError(f"columns must be equal length, got {lengths}")
    else:
        length = 0

    with open(path, "w", newline="", encoding="ascii") as f:
        writer = csv.writer(f)
        writer.writerow(names)
        for i in range(length):
            writer.writerow([f"{a[i]:.10g}" for a in arrays])
    return path


def hdf5_available() -> bool:
    try:
        import h5py  # noqa: F401
        return True
    except ImportError:
        return False


def export_bundle_hdf5(path: str, datasets: Mapping[str, np.ndarray],
                       attrs: Mapping[str, object] | None = None) -> str:
    if not hdf5_available():
        raise ImportError(
            "h5py is not installed; install it for HDF5 export, or use "
            "export_columns_csv for CSV output."
        )
    import h5py

    with h5py.File(path, "w") as f:
        for name, arr in datasets.items():
            f.create_dataset(name, data=np.asarray(arr), compression="gzip")
        for key, value in (attrs or {}).items():
            f.attrs[key] = value
    return path


def read_bundle_hdf5(path: str) -> dict[str, np.ndarray]:
    if not hdf5_available():
        raise ImportError("h5py is not installed; cannot read HDF5 bundle.")
    import h5py

    out: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        for name in f:
            out[name] = f[name][()]
    return out
