"""
Routines for interpolating 3D scalar fields to arbitrary positions in domains
with (optional) cyclic boundary conditions
"""

import numpy as np
import xarray as xr

import advtraj.lib.fast_interp as fast_interp


def interpolate_1d_field(da, ds_positions, cyclic, interp_order=1):

    if not cyclic:

        da_interpolated = da.interp(
            {da.dims[0]: ds_positions}, kwargs={"fill_value": "extrapolate"}
        )

    else:

        # if True:

        c = da.dims[0]
        c_min = da[c].values.min()
        c_max = da[c].values.max()

        # c_max = np.array([da[c].max() for c in da.dims])
        dX = da[c].values[1] - da[c].values[0]  # da[c].attrs[f"d{c}"]

        pad = False

        fn_e = fast_interp.interp1d(
            c_min, c_max, dX, da.values, c=pad, k=interp_order, p=True
        )

        pos = fn_e(ds_positions.values)

        # xr.DataArray(out.astype(output_precision),
        #                         coords={"trajectory_number":
        #                            ds_traj.coords['trajectory_number']})

        da_interpolated = xr.DataArray(
            pos,
            dims=ds_positions.dims,
            coords=ds_positions.coords,
            name=da.name,
        )
        # pos = xr.DataArray(
        #     pos,
        #     dims=da.dims,
        #     # coords=output_points,
        #     name=da.name,
        # )

    if np.any(np.isnan(da_interpolated)):
        raise Exception("Found nan during interpolation")

    return da_interpolated


def map_1d_grid_index_to_position(idx_grid, da_coord, cyclic=None):
    """
    Map indices `idx_grid` to the positions in grid defined by the
    cell-centered positions in `da_coord`.

    We assume that the grid-indices map to the cell-center positions, so
    that for a grid resolution `dx=25.0m` and a grid with two cells with
    a domain of length 50.0 we have the following:

        i:             0           1
        x:      0.0  12.5  25.0  23.5  50.0
                 |     x     |     x     |

    i_est:     -0.5    0    0.5    1    1.5

    We need to allow for the estimated grid indices `i_est` to map to grid
    positions up to the domain edges, which is outside of the cell-center
    positions. This is done by making the interpolation extend linearly outside
    the value range (by one index at either end)
    """

    N = da_coord.size
    # use linear interpolation because grid is assumed to be isotropic
    interp_order = 1
    fn_e = fast_interp.interp1d(0, N - 1, 1, da_coord.values, e=1, k=interp_order)
    pos = fn_e(np.array(idx_grid))

    if np.any(np.isnan(pos)):
        raise Exception("Found nan during interpolation")

    # Note - for a uniform grid (i.e. x, y) the following would be faster:

    # pos = (idx_grid.values * da_coord.attrs[f"d{da_coord.name}"]
    #        + da_coord.values[0])

    return pos


def interpolate_2d_field(da, ds_positions, interp_order=1, cyclic_boundaries=None):
    """
    Perform interpolation of xr.DataArray `da` at positions given by data
    variables in `ds_positions` with interpolation order `interp_order`. Cyclic
    boundary conditions are used by providing a `list` of the coordinates which
    have cyclic boundaries, e.g. (`cyclic_boundaries = 'xy'` or
    `cyclic_boundaries = ['x', 'y']`)
    """

    interpolator = gen_interpolator_2d_field(
        da, interp_order=interp_order, cyclic_boundaries=cyclic_boundaries
    )

    da_interpolated = interpolate_from_interpolator(da.name, ds_positions, interpolator)

    return da_interpolated


def interpolate_3d_field(da, ds_positions, interp_order=1, cyclic_boundaries=None):
    """
    Perform interpolation of xr.DataArray `da` at positions given by data
    variables in `ds_positions` with interpolation order `interp_order`. Cyclic
    boundary conditions are used by providing a `list` of the coordinates which
    have cyclic boundaries, e.g. (`cyclic_boundaries = 'xy'` or
    `cyclic_boundaries = ['x', 'y']`)
    """

    interpolator = gen_interpolator_3d_field(
        da, interp_order=interp_order, cyclic_boundaries=cyclic_boundaries
    )

    da_interpolated = interpolate_from_interpolator(da.name, ds_positions, interpolator)

    return da_interpolated


def interpolate_from_interpolator(v, ds_positions, interpolator):
    """
    Perform interpolation of variable named 'v' at positions given by data
    variables in `ds_positions` using interpolator fn.
    """

    fn = interpolator["fn"]

    dims = interpolator["dims"]

    vals = fn(*[ds_positions[c].values for c in dims])

    da_interpolated = xr.DataArray(
        vals,
        dims=ds_positions.dims,
        coords=ds_positions.coords,
        name=v,
    )

    return da_interpolated


def interpolate_3d_fields(
    ds, ds_positions, interpolator=None, interp_order=1, cyclic_boundaries=None
):
    """
    Perform interpolation of xr.DataSet `ds` at positions given by data
    variables in `ds_positions`.

    If interpolator provided, look in this for pre-generated fast_inter
    interpolator for each variable.
    Otherwise, interpolate with interpolation order `interp_order`. Cyclic
    boundary conditions are used by providing a `list` of the coordinates which
    have cyclic boundaries, e.g. (`cyclic_boundaries = 'xy'` or
    `cyclic_boundaries = ['x', 'y']`)
    """
    dataarrays = []

    for v in ds.data_vars:
        if interpolator is not None and v in interpolator:

            da_interpolated = interpolate_from_interpolator(
                v, ds_positions, interpolator[v]
            )

        else:

            da_interpolated = interpolate_3d_field(
                da=ds[v],
                ds_positions=ds_positions,
                interp_order=interp_order,
                cyclic_boundaries=cyclic_boundaries,
            )

        dataarrays.append(da_interpolated)

    ds_interpolated = xr.merge(dataarrays)
    ds_interpolated.attrs.update(ds.attrs)

    return ds_interpolated


def gen_interpolator_2d_field(da, interp_order=1, cyclic_boundaries=None):
    """
    Generate fast_interp interpolator for xr.DataArray `da` at positions with
    interpolation order `interp_order`.

    Cyclic boundary conditions are used by providing a `list` of the
    coordinates which have cyclic boundaries,
    e.g. (`cyclic_boundaries = 'xy'` or `cyclic_boundaries = ['x', 'y']`)
    """

    if cyclic_boundaries is None:
        cyclic_boundaries = []

    c_min = np.array([da[c].min().values for c in da.dims])
    c_max = np.array([da[c].max().values for c in da.dims])

    dX = np.array([da[c].values[1] - da[c].values[0] for c in da.dims])
    # [da[c].attrs[f"d{c}"]
    periodicity = [c in cyclic_boundaries for c in da.dims]

    pad = [not p for p in periodicity]
    padsize = [interp_order] * 2

    fn = fast_interp.interp2d(
        a=c_min,
        b=c_max,
        c=pad,
        e=padsize,
        h=dX,
        f=da.values,
        k=interp_order,
        p=periodicity,
    )
    return {"fn": fn, "dims": da.dims}


def gen_interpolator_3d_field(da, interp_order=1, cyclic_boundaries=None):
    """
    Generate fast_interp interpolator for xr.DataArray `da` at positions with
    interpolation order `interp_order`.

    Cyclic boundary conditions are used by providing a `list` of the
    coordinates which have cyclic boundaries,
    e.g. (`cyclic_boundaries = 'xy'` or `cyclic_boundaries = ['x', 'y']`)
    """

    if cyclic_boundaries is None:
        cyclic_boundaries = []

    c_min = np.array([da[c].min().values for c in da.dims])
    c_max = np.array([da[c].max().values for c in da.dims])
    dX = np.array([da[c].values[1] - da[c].values[0] for c in da.dims])
    # dX = np.array([da[c].attrs[f"d{c}"] for c in da.dims])
    periodicity = [c in cyclic_boundaries for c in da.dims]

    pad = [not p for p in periodicity]
    padsize = [interp_order] * 3

    fn = fast_interp.interp3d(
        a=c_min,
        b=c_max,
        c=pad,
        e=padsize,
        h=dX,
        f=da.values,
        k=interp_order,
        p=periodicity,
    )
    return {"fn": fn, "dims": da.dims}


def gen_interpolator_3d_fields(ds, interp_order=1, cyclic_boundaries=None) -> dict:
    """
    Generate fast_interp interpolators for xr.DataSet `ds` at positions with
    interpolation order `interp_order`.

    Cyclic boundary conditions are used by providing a `list` of the
    coordinates which have cyclic boundaries,
    e.g. (`cyclic_boundaries = 'xy'` or `cyclic_boundaries = ['x', 'y']`)
    """
    interpolators = {}
    for v in ds.data_vars:
        interpolators[v] = gen_interpolator_3d_field(
            da=ds[v],
            interp_order=interp_order,
            cyclic_boundaries=cyclic_boundaries,
        )

    return interpolators
