"""
Utilities for mapping to and from "position scalars" (which encode the position
as grid indices) to positions in real space
"""

import numpy as np
import xarray as xr

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


def _estimate_dim_initial_grid_indices(ds_position_scalars, dim, xy_periodic, n_grid):
    """
    Using the position scalars in `ds_position_scalars` estimate the grid
    locations (in indices) where the fluid was initially located when the
    position scalars were initiated.

    Note: assumes zeroth grid point has zero phase.
    """
    if dim in ["x", "y"] and xy_periodic:
        da_vr = ds_position_scalars[f"traj_tracer_{dim}r"]
        da_vi = ds_position_scalars[f"traj_tracer_{dim}i"]
        grid_idx = _calculate_phase(vr=da_vr, vi=da_vi, n_grid=n_grid)
    else:
        grid_idx = ds_position_scalars[f"traj_tracer_{dim}r"]

    if isinstance(grid_idx, xr.DataArray):
        grid_idx.name = dict(x="i", y="j", z="k")[dim]
    return grid_idx


def estimate_initial_grid_indices(ds_position_scalars, N_grid=dict()):
    """
    Using the position scalars `ds_position_scalars` estimate the original grid
    locations (ijk-indices) that the position scalars were advected from
    """
    if "xy_periodic" not in ds_position_scalars.attrs:
        raise Exception(
            "Set the `xy_periodic` attribute on the position scalars dataset"
            " `ds_position_scalars` to indicate whether the the dataset"
            " has periodic boundary conditions in the xy-direction"
        )
    else:
        xy_periodic = ds_position_scalars.xy_periodic

    if xy_periodic and ("x" not in N_grid or "y" not in N_grid):
        raise Exception(
            "For xy-periodic domains you must provide the grid shape (as a"
            " dictionary `N_grid=dict(x=<nx>, y=<ny>)`)"
        )

    da_indices = []
    for dim in "xyz":

        da_indices.append(
            _estimate_dim_initial_grid_indices(
                ds_position_scalars=ds_position_scalars,
                dim=dim,
                xy_periodic=xy_periodic,
                n_grid=N_grid.get(dim),
            )
        )

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


def grid_indices_to_position_scalars(i, j, k, nx, ny, nz, xy_periodic):
    """
    Based off `reinitialise_trajectories` in
    `components/tracers/src/tracers.F90` in the MONC model source code
    """
    pi = np.pi

    ds = xr.Dataset()
    if xy_periodic:
        ds["traj_tracer_xr"] = np.cos(2.0 * pi * i / nx)
        ds["traj_tracer_xi"] = np.sin(2.0 * pi * i / nx)
        ds["traj_tracer_yr"] = np.cos(2.0 * pi * j / ny)
        ds["traj_tracer_yi"] = np.sin(2.0 * pi * j / ny)
    else:
        ds["traj_tracer_xr"] = i
        ds["traj_tracer_yr"] = j
    ds["traj_tracer_zr"] = k
    ds.attrs["xy_periodic"] = xy_periodic
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

    xy_periodic = ds_grid.xy_periodic

    ds_position_scalars = grid_indices_to_position_scalars(
        i=ds_indices.i,
        j=ds_indices.j,
        k=ds_indices.k,
        nx=nx,
        ny=ny,
        nz=nz,
        xy_periodic=xy_periodic,
    )

    return ds_position_scalars
