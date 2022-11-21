# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 16:37:07 2022

@author: paclk
"""
import time

import numpy as np

from ..integrate import integrate_trajectories


def chunk_starting_points(
    ds_starting_points,
    chunksize=None,
    ntasks=None,
    **kwargs,
):
    start_points = []

    if chunksize is None:
        if ntasks is None:
            raise ValueError("At least one of chunksize and ntasks must be set.")
        else:

            chunk = int(np.ceil(ds_starting_points.trajectory_number.size / ntasks))
    else:
        chunk = chunksize

    first = 0
    last = chunk
    for t in range(ntasks):
        print(f"Selecting {first}:{last}")
        kwargs_batch = kwargs.copy()
        kwargs_batch["ds_starting_points"] = ds_starting_points.isel(
            trajectory_number=slice(first, last)
        ).copy()
        start_points.append(kwargs_batch)
        first += chunk
        last += chunk
        last = min(last, ds_starting_points.trajectory_number.size)

    return start_points


def get_traj_chunk(kwargs):

    time1 = time.perf_counter()

    ds_traj = integrate_trajectories(**kwargs)

    time2 = time.perf_counter()

    delta_t = time2 - time1

    print(f"Elapsed time = {delta_t}")

    ds_traj.attrs["elapsed_time"] = delta_t

    return ds_traj
