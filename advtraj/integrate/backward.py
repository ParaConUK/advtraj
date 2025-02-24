"""
Functionality for computing trajectories backward from a set of starting points
at a single point in time using the position scalars.
"""
import math

import numpy as np
import xarray as xr
from tqdm import tqdm

from ..utils.data_to_traj import aux_coords_to_traj
from ..utils.grid_mapping import (
    estimate_3d_position_from_grid_indices,
    estimate_initial_grid_indices,
)
from ..utils.interpolation import interpolate_3d_fields
from ..utils.io import ds_save

ENTERED_W_BOUNDARY = 2**1
ENTERED_E_BOUNDARY = 2**2
ENTERED_S_BOUNDARY = 2**3
ENTERED_N_BOUNDARY = 2**4

ENTERED = (
    ENTERED_W_BOUNDARY | ENTERED_E_BOUNDARY | ENTERED_S_BOUNDARY | ENTERED_N_BOUNDARY
)


def calc_trajectory_previous_position(
    ds_position_scalars,
    ds_traj_posn,
    interp_order=5,
    interpolator=None,
):
    """
    The algorithm is as follows:

    1) for a trajectory position `(x,y,z)` at a time `t` interpolate the
    "position scalars" to find their value at `(x,y,z,t)`
    2) estimate the initial indices that the fluid at `(x,y,z,t)` came from by
    converting the "position scalars" back to position
    """
    # interpolate the position scalar values at the current trajectory
    # position

    ds_initial_position_scalar_locs = interpolate_3d_fields(
        ds=ds_position_scalars,
        ds_positions=ds_traj_posn,
        interpolator=interpolator,
        interp_order=interp_order,
        cyclic_boundaries="xy" if ds_position_scalars.xy_periodic else None,
    )

    # convert these position scalar values to grid positions so we can estimate
    # what grid positions the fluid was advected from

    ds_traj_init_grid_idxs = estimate_initial_grid_indices(
        ds_position_scalars=ds_initial_position_scalar_locs,
        N_grid=ds_position_scalars.sizes,
    )

    # interpolate these grid-positions from the position scalars so that we can
    # get an actual xyz-position
    ds_traj_posn_prev = estimate_3d_position_from_grid_indices(
        ds_grid=ds_position_scalars,
        i=ds_traj_init_grid_idxs.i,
        j=ds_traj_init_grid_idxs.j,
        k=ds_traj_init_grid_idxs.k,
    )

    return ds_traj_posn_prev


def backward(
    ds_position_scalars,
    ds_starting_point,
    da_times,
    interp_order=5,
    output_path=None,
    aux_coords: list = None,
):
    """
    Using the position scalars `ds_position_scalars` integrate backwards from
    `ds_starting_point` to the times in `da_times`
    """
    # create a list into which we will accumulate the trajectory points
    # while doing this we turn the time variable into a coordinate

    input_times = list(ds_position_scalars["time"].values)
    ref_time = da_times.values[-1]
    if ref_time not in input_times:
        raise ValueError(f"Reference time {ref_time} is not in dataset.")
    ref_index = input_times.index(ref_time)

    # ds_starting_point = ds_starting_point.assign_coords({"time": ref_time})
    ds_starting_point = ds_starting_point.assign_coords(time_index=ref_index)
    ds_starting_point = ds_starting_point.assign_coords(ref_time_index=ref_index)

    if "forecast_period" in ds_starting_point.coords:
        ds_starting_point = ds_starting_point.drop_vars(["forecast_period"])

    if aux_coords is not None:
        ds_starting_point = aux_coords_to_traj(
            ds_position_scalars.sel(time=ref_time),
            ds_starting_point,
            aux_coords,
            interp_order=1,
        )

    if output_path is not None:
        out_fmt = f"0{math.ceil(math.log10(len(input_times)))}"
        ds_starting_point = ds_save(ds_starting_point, output_path, fmt=out_fmt)

    file_index = ref_index - 1

    datasets = [ds_starting_point]

    # step back in time, `t_current` represents the time we're of the next
    # point (backwards) of the trajectory
    # start at 1 because this provides position for previous time.
    for t_current in tqdm(da_times.values[::-1], desc="backward"):

        ds_traj_posn_origin = datasets[-1].drop_vars("time")

        ds_position_scalars_current = ds_position_scalars.sel(time=t_current).drop_vars(
            "time"
        )

        ds_traj_posn_est = calc_trajectory_previous_position(
            ds_position_scalars=ds_position_scalars_current,
            ds_traj_posn=ds_traj_posn_origin,
            interp_order=interp_order,
        )
        # find the previous time so that we can construct a new dataset to contain
        # the trajectory position at the previous time
        time_to_now = ds_position_scalars.time.sel(time=slice(None, t_current))
        try:
            if time_to_now.size > 1:
                t_previous = time_to_now.isel(time=-2)
            else:
                if "forecast_reference_time" in ds_position_scalars.coords:
                    t_previous = ds_position_scalars.coords["forecast_reference_time"]
                else:
                    t_previous = time_to_now - (t_current - time_to_now)
        except IndexError:
            # this will happen if we're trying to integrate backwards from the
            # very first timestep, which we can't (and shouldn't). Just check
            # we have as many trajectory points as we're aiming for
            if len(datasets) == da_times.count():
                break
            else:
                raise

        ds_traj_posn_prev = ds_traj_posn_est.assign_coords({"time": t_previous.values})

        if "forecast_period" in ds_traj_posn_prev.coords:
            ds_traj_posn_prev = ds_traj_posn_prev.drop_vars(["forecast_period"])

        # Error in back trajectory is not quantifiable. Set to NaN.
        for c in "xyz":
            ds_traj_posn_prev[f"{c}_err"] = xr.full_like(
                ds_traj_posn_prev[c], -1, dtype=np.float32
            )

        flags = datasets[-1].flag.values & ENTERED

        if not ds_position_scalars.xy_periodic:
            x = ds_traj_posn_prev["x"].values
            flags[x < ds_position_scalars["x"].values[0]] |= ENTERED_W_BOUNDARY
            flags[x > ds_position_scalars["x"].values[-1]] |= ENTERED_E_BOUNDARY

            y = ds_traj_posn_prev["y"].values
            flags[y < ds_position_scalars["y"].values[0]] |= ENTERED_S_BOUNDARY
            flags[y > ds_position_scalars["y"].values[-1]] |= ENTERED_N_BOUNDARY

        ds_traj_posn_prev["flag"] = xr.DataArray(
            flags,
            coords={"trajectory_number": ds_traj_posn_prev.coords["trajectory_number"]},
        )

        ds_traj_posn_prev = ds_traj_posn_prev.assign_coords(time_index=file_index)

        if aux_coords is not None:
            ds_traj_posn_prev = aux_coords_to_traj(
                ds_position_scalars_current,
                ds_traj_posn_prev,
                aux_coords,
                interp_order=1,
            )

        if output_path is not None:
            ds_traj_posn_prev = ds_save(ds_traj_posn_prev, output_path, fmt=out_fmt)
        file_index -= 1

        datasets.append(ds_traj_posn_prev)

    ds_traj = xr.concat(datasets[::-1], dim="time")

    return ds_traj
