"""
Interpolate selected variables from input gridded dataset to trajectories.
"""
import time

import xarray as xr

from .interpolation import (
    interpolate_1d_field,
    interpolate_2d_field,
    interpolate_3d_field,
)


def data_to_traj(
    source_dataset: xr.Dataset,
    ds_traj: xr.Dataset,
    varlist: list,
    output_path: str,
    interp_order: int = 5,
    output_precision: str = "float32",
    write_sleeptime: int = 3,
) -> dict():
    """
    Interpolate 3D variables to trajectory points.

    Parameters
    ----------
    source_dataset : xr.Dataset
        Input variables on 3D grid at times matching trajectory times.
    ds_traj : xr.Dataset
        Trajectory positions, 'x', 'y' and 'z'.
    varlist : list(str)
        List of strings with variable names required from source_dataset.
    output_path : str
        Path to save output NetCDF file.
    interp_order : int, optional
        Order of polynomial interpolation. The default is 5.
    output_precision : str, optional
        Data type for output. The default is "float32".
    write_sleeptime : int, optional
        Pause after write. The default is 3.

    Returns
    -------
    dict
        'file': output_path.
        'ds'  : output xarray Dataset.

    """

    atts = source_dataset.attrs

    ds_out = xr.Dataset()
    for inc in atts:
        if isinstance(atts[inc], (dict, bool, type(None))):
            atts[inc] = str(atts[inc])

    ds_out.attrs = atts

    ds_out.to_netcdf(output_path, mode="w")

    for var_name in varlist:
        print(f"Mapping {var_name} onto trajectories.")

        da = source_dataset[var_name]

        rename_dict = {}
        for c in "xyz":
            nc = [d for d in da.dims if c in d][0]
            if nc != c:
                rename_dict[nc] = c
        if len(nc) > 0:
            da = da.rename(rename_dict)

        print(da)

        varout = []
        for traj_time in ds_traj.time:

            if traj_time.values in da.time.values:
                dat = da.sel(time=traj_time)

                ds_positions = ds_traj[["x", "y", "z"]].sel(time=traj_time)

                interp_data = interpolate_3d_field(
                    dat, ds_positions, interp_order=interp_order, cyclic_boundaries="xy"
                )

                varout.append(interp_data.astype(output_precision))
            else:
                print(f"No data for variable {var_name}, time {traj_time}.")

        ds_out[var_name] = xr.concat(varout, dim="time")

        encoding = {var_name: {"dtype": output_precision}}

        print(f"Saving {var_name}.")
        ds_out[var_name].to_netcdf(
            output_path, unlimited_dims="time", mode="a", encoding=encoding
        )

        # This wait seems to be needed to give i/o time to flush caches.
        time.sleep(write_sleeptime)

    return {"file": output_path, "ds": ds_out}


def aux_coords_to_traj(
    source_dataset: xr.Dataset,
    ds_traj: xr.Dataset,
    aux_coords: list,
    interp_order: int = 1,
    output_precision: str = "float32",
) -> xr.Dataset:
    """
    Interpolate auxiliary coordinates to trajectory points.

    Parameters
    ----------
    source_dataset : xr.Dataset
        Input variables on 3D grid at times matching trajectory times.
    ds_traj : xr.Dataset
        Trajectory positions, 'x', 'y' and 'z'.
    aux_coords : list(str)
        List of strings with variable names required from source_dataset.
    interp_order : int, optional
        Order of polynomial interpolation. The default is 5.
    output_precision : str, optional
        Data type for output. The default is "float32".

    Returns
    -------
    ds_traj : xr.Dataset
        Trajectory positions, 'x', 'y' and 'z' with additional coords.

    """

    for aux_coord in aux_coords:
        if aux_coord in source_dataset:
            da_coord = source_dataset.coords[aux_coord]
        else:
            cl = [c for c in source_dataset.coords if aux_coord in c]
            if cl:
                da_coord = source_dataset.coords[cl[0]]
            else:
                continue
        dims_ok = [d in "xyz" for d in da_coord.dims]
        if not all(dims_ok):
            print(
                f"Auxiliary coordinate {aux_coord} coordinates ",
                f"{da_coord.dims} are not all in xyz",
            )
            continue
        ndims = len(dims_ok)
        if ndims == 1:

            cyclic = source_dataset.attrs["xy_periodic"]

            output_points = ds_traj[da_coord.dims[0]].values

            out = interpolate_1d_field(
                da_coord, output_points, cyclic, interp_order=interp_order
            )

            # ds_traj[aux_coord] = out.astype(output_precision)
            ds_traj[aux_coord] = xr.DataArray(
                out.values.astype(output_precision),
                coords={"trajectory_number": ds_traj.coords["trajectory_number"]},
            )
        elif ndims == 2:

            cyclic = source_dataset.attrs["xy_periodic"]
            if cyclic:
                cyclic_boundaries = "xy"
            else:
                cyclic_boundaries = None

            output_points = ds_traj[list(da_coord.dims)]

            out = interpolate_2d_field(
                da_coord,
                output_points,
                cyclic_boundaries=cyclic_boundaries,
                interp_order=interp_order,
            )

            ds_traj[aux_coord] = out.astype(output_precision)

        elif ndims == 3:

            cyclic = source_dataset.attrs["xy_periodic"]
            if cyclic:
                cyclic_boundaries = "xy"
            else:
                cyclic_boundaries = None

            output_points = ds_traj[list(da_coord.dims)]

            out = interpolate_3d_field(
                da_coord,
                output_points,
                cyclic_boundaries=cyclic_boundaries,
                interp_order=interp_order,
            )

            ds_traj[aux_coord] = out.astype(output_precision)

    return ds_traj
