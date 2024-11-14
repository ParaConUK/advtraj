"""
Utilities for trajectory io.
"""
# import numpy as np
# import xarray as xr


def ds_save(ds, output_path):  # , out_prec="float32"):

    file_index = ds.coords["time_index"].item()

    output_file = f"{output_path}_{file_index+1:04d}.nc"

    d = ds.to_netcdf(output_file, unlimited_dims="time", mode="w", compute=False)
    d.compute()

    return ds
