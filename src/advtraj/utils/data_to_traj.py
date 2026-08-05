"""
Interpolate selected variables from input gridded dataset to trajectories.
"""

import sys
import time

import xarray as xr
from loguru import logger
from tqdm import tqdm

from .interpolation import interp_gen_field

is_interactive = sys.stdout.isatty()


def rename_traj_coords(da, coords=None):
    if coords is None:
        coords = "xyz"
    rename_dict = {}
    for c in coords:
        nc = [d for d in da.dims if c in d][0]
        if nc != c:
            rename_dict[nc] = c
    if len(nc) > 0:
        da = da.rename(rename_dict)
    return da


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
        logger.info(f"Mapping {var_name} onto trajectories.")

        if var_name in source_dataset:
            da = source_dataset[var_name]
        else:
            logger.warning(f"{var_name} not in source_dataset.")
            continue

        grid_type = da.attrs.get("grid_type", None)

        if grid_type is None:
            source_dataset.attrs.get("grid_type", "xy_periodic")

        da = rename_traj_coords(da)

        logger.info(f"Source data:\n {da}")

        varout = []
        for traj_time in tqdm(
            ds_traj.time.values,
            desc="Trajectory Time",
            disable=not is_interactive,
            dynamic_ncols=is_interactive,
        ):

            # if traj_time.values in da.time.values:
            try:
                dat = da.sel(time=traj_time, method="nearest", tolerance=1.0)

            except KeyError:
                message = f"No data for variable {var_name}, time {traj_time}."
                if is_interactive:
                    tqdm.write(message)
                else:
                    logger.info(message)
            else:
                ds_positions = ds_traj[["x", "y", "z"]].sel(time=traj_time)

                interp_data = interp_gen_field(
                    dat, ds_positions, interp_order=interp_order, grid_type=grid_type
                )

                varout.append(interp_data.astype(output_precision))

        ds_out[var_name] = xr.concat(varout, dim="time")

        encoding = {var_name: {"dtype": output_precision}}

        logger.info(f"Saving {var_name}.")
        ds_out[var_name].to_netcdf(
            output_path, unlimited_dims="time", mode="a", encoding=encoding
        )

        # This wait seems to be needed to give i/o time to flush caches.
        time.sleep(write_sleeptime)

    return {"file": output_path, "ds": ds_out}


def find_aux_coord_in_dataset(source_dataset, aux_coord):
    if aux_coord in source_dataset:
        da_coord = source_dataset.coords[aux_coord]
    else:
        cl = [c for c in source_dataset.coords if aux_coord in c]
        sd_coord = None
        match len(cl):
            case 0:
                return None
            case 1:
                sd_coord = cl[0]
            case _:
                for c in cl:
                    if c.startswith(aux_coord):
                        sd_coord = c
                        break
                if sd_coord is None:
                    sd_coord = cl[0]
        da_coord = source_dataset.coords[sd_coord]
    return da_coord


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
    grid_type = source_dataset.attrs["grid_type"]
    for aux_coord in aux_coords:
        da_coord = find_aux_coord_in_dataset(source_dataset, aux_coord)
        if da_coord is None:
            continue

        dims_ok = [d in "xyz" for d in da_coord.dims]
        if not all(dims_ok):
            logger.warning(
                f"Auxiliary coordinate {aux_coord} coordinates ",
                f"{da_coord.dims} are not all in xyz",
            )
            continue
        ndims = len(dims_ok)

        if ndims == 1:
            output_points = ds_traj[da_coord.dims[0]]
        else:
            output_points = ds_traj[list(da_coord.dims)]

        out = interp_gen_field(da_coord, output_points, grid_type, interp_order)

        if ndims == 1:
            ds_traj[aux_coord] = xr.DataArray(
                out.values.astype(output_precision),
                coords={"trajectory_number": ds_traj.coords["trajectory_number"]},
            )
        else:
            ds_traj[aux_coord] = out.astype(output_precision)

    return ds_traj
