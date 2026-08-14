"""
Utilities for mapping to and from "position scalars" (which encode the position
as grid indices) to positions in real space
"""

import numpy as np
import xarray as xr
from loguru import logger

from .interpolation import map_1d_grid_index_to_position


def _calculate_phase(vr, vi, n_grid):
    """
    Function to convert real and imaginary points to location on grid
    size n_grid. Because we assume the grid indices refer to cell-centered
    positions we allow the grid-index to go down to -0.5 (which would
    correspond to the left-most of the domain). To support staggered variables
    this should be adapted.

    Note: assumes zeroth grid point has zero phase.

    Args:
        vr, vi  : real and imaginary parts of complex location.
        n_grid  : grid size

    Returns:
        Real position in [0,n)

    @author: Peter Clark, Leif Denby

    """

    vpos = ((np.arctan2(vi, vr) / (2.0 * np.pi)) * n_grid + 0.5) % n_grid - 0.5
    return vpos


def estimate_initial_grid_indices(ds_position_scalars, N_grid=dict()):
    """
    Using the position scalars `ds_position_scalars` estimate the original grid
    locations (ijk-indices) that the position scalars were advected from
    """

    if "grid_type" not in ds_position_scalars.attrs:
        logger.error("Attribute grid type not set.")
        raise Exception(
            "Set the `grid_type` attribute on the position scalars dataset"
            " `ds_position_scalars` to 'lam', 'xy_periodic' or 'global'"
        )
    else:
        grid_type = ds_position_scalars.attrs["grid_type"]

    grid_names = dict(x="i", y="j", z="k")
    da_indices = []
    match grid_type.lower():
        case "lam":
            for dim in "xy":
                grid_id = ds_position_scalars[f"traj_tracer_{dim}r"]
                grid_id.name = grid_names[dim]
                da_indices.append(grid_id)

        case "xy_periodic":
            if "x" not in N_grid or "y" not in N_grid:
                raise Exception(
                    "For xy_periodic domains you must provide the grid shape (as a"
                    " dictionary `N_grid=dict(x=<nx>, y=<ny>)`)"
                )
            for dim in "xy":
                da_vr = ds_position_scalars[f"traj_tracer_{dim}r"]
                da_vi = ds_position_scalars[f"traj_tracer_{dim}i"]
                grid_id = _calculate_phase(vr=da_vr, vi=da_vi, n_grid=N_grid.get(dim))
                grid_id.name = grid_names[dim]
                da_indices.append(grid_id)

        case "global":

            if "x" not in N_grid or "y" not in N_grid:
                raise Exception(
                    "For global domains you must provide the grid shape (as a"
                    " dictionary `N_grid=dict(x=<nx>, y=<ny>)`)"
                )
            # Assumptions:
            #    Longitude
            #    i=0...Nx-1
            #    i=0->0 radians, i=Nx-> 2 pi radians
            #    Data on p points 0.5, 1.5 ... Nx-0.5
            #    So 0 longitude is at grid point -0.5
            #
            #    Latitude
            #    j=0...Ny-1
            #    j=0->-pi/2 radians, i=Ny-> pi/2 radians
            #    Data on p points 0.5, 1.5 ... Ny-0.5
            #    So -90 degrees latitude is at grid point -0.5

            da_xr = ds_position_scalars["traj_tracer_xr"]
            da_xi = ds_position_scalars["traj_tracer_xi"]

            if np.any(np.isnan(da_xr)):
                raise Exception("Found nan in traj_tracer_xr.")
            if np.any(np.isnan(da_xi)):
                raise Exception("Found nan in traj_tracer_xi.")

            n_grid = N_grid.get("x")

            grid_id = (
                (np.arctan2(da_xi, da_xr) / (2.0 * np.pi) + 1) * n_grid
            ) % n_grid - 0.5

            grid_id.name = grid_names["x"]
            da_indices.append(grid_id)

            da_yr = ds_position_scalars["traj_tracer_yr"]
            if np.any(np.isnan(da_yr)):
                raise Exception("Found nan in traj_tracer_yr.")
            da_yr = np.clip(da_yr, -1, 1)
            n_grid = N_grid.get("y")
            grid_id = ((np.arcsin(da_yr) / np.pi + 0.5) * n_grid) % n_grid - 0.5
            grid_id.name = grid_names["y"]
            da_indices.append(grid_id)

    grid_id = ds_position_scalars["traj_tracer_zr"]
    if np.any(np.isnan(grid_id)):
        raise Exception("Found nan in traj_tracer_yr.")
    grid_id.name = grid_names["z"]
    da_indices.append(grid_id)

    return xr.merge(da_indices)


def estimate_3d_position_from_grid_indices(ds_grid, i, j, k, interp_order=1):
    """
    Using the 3D grid positions (in real units, not grid indices) defined in
    `ds_grid` (through coordinates `x`, `y` and `z`) interpolate the "grid
    indices" in `ds_grid_indices` (these may be fractional, i.e. they are
    not discrete integer grid indices) to the real x, y and z-positions.
    """

    x_pos = map_1d_grid_index_to_position(i, da_coord=ds_grid.x)
    y_pos = map_1d_grid_index_to_position(j, da_coord=ds_grid.y)
    z_pos = map_1d_grid_index_to_position(k, da_coord=ds_grid.z)

    if np.any(np.isnan([x_pos, y_pos, z_pos])):
        raise Exception("Found nan during interpolation")

    if isinstance(i, xr.DataArray):
        assert i.dims == j.dims == k.dims
        ds = xr.Dataset(coords=i.coords)
        ds["x"] = i.dims, x_pos
        ds["y"] = j.dims, y_pos
        ds["z"] = k.dims, z_pos
        return ds
    else:
        return [x_pos, y_pos, z_pos]


def grid_indices_to_position_scalars(i, j, k, nx, ny, nz, grid_type=None):
    """
    Based off `reinitialise_trajectories` in
    `components/tracers/src/tracers.F90` in the MONC model source code
    """
    pi = np.pi

    if grid_type is None:
        logger.warning("grid_type not set: set to xy_periodic")
        grid_type = "xy_periodic"

    ds = xr.Dataset()

    match grid_type.lower():
        case "xy_periodic":
            lam = 2.0 * pi * i / nx
            phi = 2.0 * pi * j / ny
            ds["traj_tracer_xr"] = np.cos(lam)
            ds["traj_tracer_xi"] = np.sin(lam)
            ds["traj_tracer_yr"] = np.cos(phi)
            ds["traj_tracer_yi"] = np.sin(phi)
        case "global":
            lam = 2.0 * pi * i / nx
            phi = pi * (j / ny - 0.5)
            cos_phi = np.cos(phi)
            ds["traj_tracer_xr"] = cos_phi * np.cos(lam)
            ds["traj_tracer_xi"] = cos_phi * np.sin(lam)
            ds["traj_tracer_yr"] = np.sin(phi)
        case "lam":
            ds["traj_tracer_xr"] = i
            ds["traj_tracer_yr"] = j

    ds["traj_tracer_zr"] = k
    ds.attrs[grid_type] = grid_type

    return ds


def grid_locations_to_position_scalars(ds_grid, ds_pts=None):

    nx = int(ds_grid.x.size)
    ny = int(ds_grid.y.size)
    nz = int(ds_grid.z.size)

    if ds_pts is None:
        i = np.arange(nx)
        j = np.arange(ny)
        k = np.arange(nz)

        i_, j_, k_ = np.meshgrid(i, j, k, indexing="ij")
        ds_indices = ds_grid.copy()
        ds_indices["i"] = ("x", "y", "z"), i_
        ds_indices["j"] = ("x", "y", "z"), j_
        ds_indices["k"] = ("x", "y", "z"), k_
    else:

        dx = ds_grid.x.dx
        dy = ds_grid.y.dy
        dz = ds_grid.z.dz

        i_ = (ds_pts.x - ds_grid.x.min()) / dx
        j_ = (ds_pts.y - ds_grid.y.min()) / dy
        k_ = (ds_pts.z - ds_grid.z.min()) / dz

        ds_indices = ds_pts.copy()
        ds_indices["i"] = i_
        ds_indices["j"] = j_
        ds_indices["k"] = k_

    grid_type = ds_grid.attrs["grid_type"]

    ds_position_scalars = grid_indices_to_position_scalars(
        i=ds_indices.i,
        j=ds_indices.j,
        k=ds_indices.k,
        nx=nx,
        ny=ny,
        nz=nz,
        grid_type=grid_type,
    )

    return ds_position_scalars
