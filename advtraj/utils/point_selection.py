# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 12:04:57 2022

@author: paclk
"""
import numpy as np
import xarray as xr


def mask_to_positions(mask: xr.DataArray) -> xr.Dataset:
    """
    Convert 3D logical mask to coordinate positions.

    Parameters
    ----------
    mask : xr.DataArray
        Evaluates True at required positions.

    Returns
    -------
    positions : xr.Dataset
        Contains data variables "x", "y", "z".
        Coordinates "pos_number" and any others (e.g. "time") in mask.

    """
    poi = (
        mask.where(mask, drop=True)
        .stack(pos_number=("x", "y", "z"))
        .dropna(dim="pos_number")
    )

    # print(f'{poi=}')
    # # now we'll turn this 1D dataset where (x, y, z) are coordinates into
    # # one where they are variables instead

    # p1 = poi.reset_index("pos_number")
    # print(f'{p1=}')

    # p2 = p1.assign_coords(pos_number=np.arange(poi.pos_number.size))
    # print(f'{p2=}')

    # p3 = p2.reset_coords(["x", "y", "z"])
    # print(f'{p3=}')

    # positions = p3[["x", "y", "z"]]

    # print(positions)

    positions = (
        poi.reset_index("pos_number")
        .assign_coords(pos_number=np.arange(poi.pos_number.size))
        .reset_coords(["x", "y", "z"])[["x", "y", "z"]]
    )

    return positions
