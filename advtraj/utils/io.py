"""
Utilities for trajectory io.
"""
import glob

import xarray as xr


def ds_save(ds, output_path, fmt: str = None) -> xr.Dataset:
    if fmt is None:
        fmt = "04d"
    file_index = ds.coords["time_index"].item()
    ref_index = ds.coords["ref_time_index"].item()
    output_file = f"{output_path}_{ref_index + 1:{fmt}}_{file_index + 1:{fmt}}.nc"

    d = ds.to_netcdf(output_file, unlimited_dims="time", mode="w", compute=False)

    d.compute()

    return ds


def gather_traj_files(traj_path: str, dim=None):

    if dim is None:
        dim = "time"
    files = glob.glob(traj_path)
    files.sort()

    da_list = []
    for file in files:
        da = xr.open_dataset(file)
        da_list.append(da)

    ds = xr.concat(da_list, dim=dim)

    return ds
