"""
Routines for interpolating 3D scalar fields to arbitrary positions in domains
with (optional) cyclic boundary conditions
"""

import numpy as np
import xarray as xr
from loguru import logger

import advtraj.lib.fast_interp_global as fast_interp

from ..utils.xarray_utils import get_dimorder


def interpolate_1d_field(da, ds_positions, grid_type=None, interp_order=1):

    if grid_type is None:
        grid_type = "xy_periodic"

    if grid_type == "lam" or "x" in da.dims[0] or "y" in da.dims[0]:

        da_interpolated = da.interp(
            {da.dims[0]: ds_positions}, kwargs={"fill_value": "extrapolate"}
        )
    else:

        c = da.dims[0]
        c_min = da[c].values.min()
        c_max = da[c].values.max()

        dX = da[c].values[1] - da[c].values[0]  # da[c].attrs[f"d{c}"]

        pad = False
        p = 0

        if grid_type == "xy_periodic" and ("x" in da.dims[0] or "y" in da.dims[0]):
            p = 1
        elif grid_type == "global" and "x" in da.dims[0]:
            p = 1
        elif grid_type == "global" and "y" in da.dims[0]:
            p = 2
        else:
            logger.warning(f"Unknown grid_type {grid_type} or dimension {da.dims[0]}")

        fn_e = fast_interp.interp1d(
            c_min, c_max, dX, da.values, c=pad, k=interp_order, p=p
        )

        pos = fn_e(ds_positions.values)

        da_interpolated = xr.DataArray(
            pos,
            dims=ds_positions.dims,
            coords=ds_positions.coords,
            name=da.name,
        )

    if np.any(np.isnan(da_interpolated)):
        raise Exception("Found nan during interpolation")

    return da_interpolated


def map_1d_grid_index_to_position(idx_grid, da_coord, grid_type=None):
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
    # print(f'{da_coord=}, {N=}')
    # print(f'{da_coord.values=}')
    # print(f'{idx_grid=}')
    fn_e = fast_interp.interp1d(0, N - 1, 1, da_coord.values, e=1, k=interp_order)
    pos = fn_e(np.array(idx_grid))

    if np.any(np.isnan(pos)):
        print(da_coord, da_coord.values)
        raise Exception("Found nan during interpolation")

    # Note - for a uniform grid (i.e. x, y) the following would be faster:

    # pos = (idx_grid.values * da_coord.attrs[f"d{da_coord.name}"]
    #        + da_coord.values[0])

    return pos


def interpolate_field(da, ds_positions, interp_order=1, grid_type=None):
    """
    Perform interpolation of xr.DataArray `da` at positions given by data
    variables in `ds_positions` with interpolation order `interp_order`.
    """
    # print(f'{grid_type=} {interp_order=}')

    interpolator = gen_interpolator_for_field(
        da, interp_order=interp_order, grid_type=grid_type
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

    # print(f'{v=}')

    # print(f'{ds_positions=}')

    # print(f'{dims=}')

    vals = fn(*[ds_positions[c].values for c in dims])

    # print(f'{vals=}')

    da_interpolated = xr.DataArray(
        vals,
        dims=ds_positions.dims,
        coords=ds_positions.coords,
        name=v,
    )

    return da_interpolated


def interpolate_fields(
    ds, ds_positions, interpolator=None, interp_order=1, grid_type=None
):
    """
    Perform interpolation of xr.DataSet `ds` at positions given by data
    variables in `ds_positions`.

    If interpolator provided, look in this for pre-generated fast_inter
    interpolator for each variable.
    Otherwise, interpolate with interpolation order `interp_order`.
    """
    dataarrays = []

    for v in ds.data_vars:
        if interpolator is not None and v in interpolator:

            da_interpolated = interpolate_from_interpolator(
                v, ds_positions, interpolator[v]
            )

        else:

            da_interpolated = interpolate_field(
                da=ds[v],
                ds_positions=ds_positions,
                interp_order=interp_order,
                grid_type=grid_type,
            )

        dataarrays.append(da_interpolated)

    ds_interpolated = xr.merge(dataarrays)
    ds_interpolated.attrs.update(ds.attrs)

    return ds_interpolated


def match_dims(dims, keys, vals):

    result = []
    for d in dims:
        for i, k in enumerate(keys):
            if d in k:
                result.append(vals[i])
                break
    return result


def gen_interpolator_for_field(da, interp_order=1, grid_type=None):
    """
    Generate fast_interp interpolator for xr.DataArray `da` at positions with
    interpolation order `interp_order`.

    grid_type can be one of 'lam', 'xy_periodic' or 'global'
    """
    logger.info(f"{grid_type=} {interp_order=}")

    if grid_type is None:
        grid_type = "xy_periodic"

    dimlist, dimorder = get_dimorder(da, req_order="xyz")

    # print(da)

    c_min = np.array([da[c].min().values for c in da.dims])
    c_max = np.array([da[c].max().values for c in da.dims])
    dX = np.array([da[c].values[1] - da[c].values[0] for c in da.dims])

    match len(da.dims):
        case 1:
            c_min = c_min[0]
            c_max = c_max[0]

            dX = dX[0]

            pad = False
            padsize = [interp_order] * 2

            if grid_type == "xy_periodic" and ("x" in da.dims[0] or "y" in da.dims[0]):
                periodicity = 1
            elif grid_type == "global" and "x" in da.dims[0]:
                periodicity = 1
            elif grid_type == "global" and "y" in da.dims[0]:
                periodicity = 2
            else:
                logger.warning(
                    f"Unknown grid_type {grid_type} or dimension {da.dims[0]}"
                )

            fn = fast_interp.interp1d(
                a=c_min,
                b=c_max,
                c=pad,
                e=padsize,
                h=dX,
                f=da.values,
                k=interp_order,
                p=periodicity,
            )

        case 2:
            match grid_type.lower():
                case "lam":
                    periodicity = [[0, 0][i] for i in dimorder]
                case "xy_periodic":
                    periodicity = [[1, 1][i] for i in dimorder]
                case "global":
                    periodicity = [[1, 2][i] for i in dimorder]
                case [p, q] if type(p) is int and type(q) is int:
                    periodicity = [p, q]
                case _:
                    raise ValueError("Unknown grid_type {grid_type}")

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

        case 3:
            match grid_type.lower():
                case "lam":
                    periodicity = [[0, 0, 0][i] for i in dimorder]
                case "xy_periodic":
                    periodicity = [[1, 1, 0][i] for i in dimorder]
                case "global":
                    periodicity = [[1, 2, 0][i] for i in dimorder]
                case [p, q, r] if type(p) is int and type(q) is int and type(r) is int:
                    periodicity = [p, q, r]
                case _:
                    raise ValueError("Unknown grid_type {grid_type}")

            # dX = np.array([da[c].attrs[f"d{c}"] for c in da.dims])
            # periodicity = [c in cyclic_boundaries for c in da.dims]

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


def gen_interpolator_fields(ds, interp_order=1, grid_type=None) -> dict:
    """
    Generate fast_interp interpolators for xr.DataSet `ds` at positions with
    interpolation order `interp_order`.

    Cyclic boundary conditions are used by providing a `list` of the
    coordinates which have cyclic boundaries,
    e.g. (`cyclic_boundaries = 'xy'` or `cyclic_boundaries = ['x', 'y']`)
    """
    if grid_type is None:
        grid_type = "xy_periodic"
    interpolators = {}
    for v in ds.data_vars:
        interpolators[v] = gen_interpolator_for_field(
            da=ds[v],
            interp_order=interp_order,
            grid_type=grid_type,
        )

    return interpolators


def interp_gen_field(field, output_points, grid_type, interp_order):

    # print(f'{grid_type=} {interp_order=}')
    dims_ok = [d in "xyz" for d in field.dims]
    if not all(dims_ok):
        logger.warning(
            f"Field {field} coordinates ",
            f"{field.dims} are not all in xyz",
        )
        return None

    ndims = len(field.dims)

    if ndims == 1:

        out = interpolate_1d_field(
            field, output_points, grid_type=grid_type, interp_order=interp_order
        )

    else:

        out = interpolate_field(
            field,
            output_points,
            grid_type=grid_type,
            interp_order=interp_order,
        )

    return out
