"""
    grid.py

    Utilities to deal with grid wrapping etc.

    Currently, caters for the following grid styles:

    "cell_centred"
    - p-point (cell centre) p[0, 0, 0]] at x = dx/2, y = dy/2, z = dz/2.
    - u-point (cell face) u[0, 0, 0] at p[0, 0, 0] - dx/2, i.e. x = 0.
    - v-point (cell face) v[0, 0, 0] at p[0, 0, 0] - dy/2, i.e. y = 0.
    - w-point (cell face) w[0, 0, 0] at p[0, 0, 0] - dz/2, i.e. z = 0.
    - Virtual p point at z = -dz/2.

    "monc"
    - Virtual p points at z = -dz/2 so:
    - p-point (cell centre) p[0, 0, 0]] at x = dx/2, y = dy/2, z = -dz/2.
    - u-point (cell face) u[0, 0, 0] at p[0, 0, 0] + dx/2, i.e. x = dx.
    - v-point (cell face) v[0, 0, 0] at p[0, 0, 0] + dy/2, i.e. y = dy.
    - w-point (cell face) w[0, 0, 0] at p[0, 0, 0] + dz/2, i.e. z = 0.

    ""
    - All points [0, 0, 0] at x = 0, y = 0, z = 0.
"""
import warnings

import numpy as np
import numpy.random as rnd
import xarray as xr


def wrap_posn(c, c_min, c_max):
    """
    Wrap coordinate positions `c` so that they lie inside the range [c_min, c_max[
    """
    lc = c_max - c_min
    c_ = c - c_min

    r = c_ / lc

    N_wrap = np.where(r >= 0.0, r.astype(int), r.astype(int) - 1.0)

    c_wrapped = c - lc * N_wrap

    return c_wrapped


def wrap_periodic_grid_coords(
    ds_grid, ds_posn, cyclic_coords=("x", "y"), cell_centered_coords=("x", "y")
):
    """
    ensure that positions given by `ds_posn` are inside the coordinates of
    `ds_grid`. Use `cell_centered_coords` to define which coords in `ds_grid`
    are using cell-centered
    """

    ds_posn_copy = ds_posn.copy()

    for c in cyclic_coords:
        da_coord = ds_grid[c]
        dc = find_coord_grid_spacing(da_coord=da_coord)

        c_min, c_max = da_coord.min().data, da_coord.max().data
        if c in cell_centered_coords:
            c_min -= dc / 2.0
            c_max += dc / 2.0
        else:
            c_max += dc

        # Now wrap the position where needed
        wrapped_c = wrap_posn(ds_posn_copy[c].values, c_min=c_min, c_max=c_max)

        # Now update variable c with wrapped values, including dim[0],
        # which should be 'trajectory_number', if available.
        # Do not include dim if it's not in the orriginal.

        ds_posn[c] = ds_posn_copy[c].copy(data=wrapped_c)

    return ds_posn_copy


def wrap_coords(ds_posn, ds_grid):
    """
    Wrapper for wrap_periodic_grid_coords

    Parameters
    ----------
    ds_posn : xarray Dataset
        Trajectory positions 'x', 'y', 'z'.
    ds_grid : xarray Dataset
        grid information.

    Returns
    -------
    xarray Dataset
        Wrapped version of ds_posn.

    """
    cyclic_coords = ("x", "y")
    cell_centered_coords = ("x", "y", "z")
    return wrap_periodic_grid_coords(
        ds_grid=ds_grid,
        ds_posn=ds_posn,
        cyclic_coords=cyclic_coords,
        cell_centered_coords=cell_centered_coords,
    )


def wrap_spherical_coords(ds_posn, ds_grid):
    """
    ensure that positions given by `ds_posn` are inside the domain of
    `ds_grid` on a spherical polar grid.
    """
    # ds_posn_copy = ds_posn.copy()

    Lx = ds_grid.x.attrs["Lx"]
    Ly = ds_grid.y.attrs["Ly"]

    wrapped_x = (ds_posn.x.values) % Lx

    # print(f'1: {wrapped_x.min()=} {wrapped_x.max()=}')

    wrapped_y = ds_posn.y.values

    if wrapped_x.shape == ():
        wrapped_x = np.array([wrapped_x])
    if wrapped_y.shape == ():
        wrapped_y = np.array([wrapped_y])

    lat_ge_Ly = wrapped_y >= Ly

    wrapped_x[lat_ge_Ly] = (wrapped_x[lat_ge_Ly] + Lx / 2) % Lx
    wrapped_y[lat_ge_Ly] = (2 * Ly - wrapped_y[lat_ge_Ly]) % Ly

    lat_lt_0 = wrapped_y < 0

    wrapped_x[lat_lt_0] = (wrapped_x[lat_lt_0] - Lx / 2) % Lx
    wrapped_y[lat_lt_0] = (-wrapped_y[lat_lt_0]) % Ly

    # print(f'2: {wrapped_x.min()=} {wrapped_x.max()=} {ds_posn_copy.x.dims=}')
    if ds_posn.x.shape == ():
        ds_posn["x"] = ds_posn.x.copy(data=wrapped_x[0])
    else:
        ds_posn["x"] = ds_posn.x.copy(data=wrapped_x)
    if ds_posn.y.shape == ():
        ds_posn["y"] = ds_posn.y.copy(data=wrapped_y[0])
    else:
        ds_posn["y"] = ds_posn.y.copy(data=wrapped_y)

    return ds_posn


def regularize_coords(ds_posn, ds_grid):

    match ds_grid.grid_type:
        case "xy_periodic":
            ds_posn = wrap_coords(ds_posn, ds_grid)
        case "global":
            ds_posn = wrap_spherical_coords(ds_posn, ds_grid)

    return ds_posn


def find_coord_grid_spacing(da_coord, show_warnings=True):
    grid_tol = 0.001

    v_name = f"d{da_coord.name}"
    if v_name in da_coord.attrs:
        return da_coord.attrs[v_name]
    else:
        if show_warnings:
            warnings.warn(
                f"The grid spacing isn't currently set for coordinate `{da_coord.name}`"
                f" to speed up calculations and ensure the grid-spacing is set correctly"
                f" set the `{v_name}` attribute of the `{da_coord.name}` coordinate"
                " to the value of the grid-spacing"
            )

    dx_all = np.diff(da_coord.values)

    if (np.max(dx_all) - np.min(dx_all)) / np.mean(dx_all) > grid_tol:
        raise Exception("Non-uniform grid")

    return np.mean(dx_all)


def find_grid_spacing(ds_grid, coords=("x", "y", "z")):
    return [find_coord_grid_spacing(ds_grid[c]) for c in coords]


def create_index_dims(ds, offsets: dict, dimlist: dict = None):
    """
    Construct x,y,z as 'grid' indices
    and make them the primary dimensions.

    Parameters
    ----------
    ds : xarray Dataset or DataArray
    offsets : dict
        Index offsets for x,y,z ( so first point of x is offsets['x'])
    dimlist : dict, optional
        Identifiers for dimensions corresponding to x,y,z. The default is None.
        If None,
        dimlist = {'x':'longitude',
                   'y':'latitude',
                   'z':'level_number'}

    Returns
    -------
    ds : xarray Dataset or DataArray
        Copy of input with new primary dimensions.

    """

    if dimlist is None:
        dimlist = {"x": "longitude", "y": "latitude", "z": "level_number"}

    swap_map = {}
    for outdim, indim in dimlist.items():
        for dim in ds.dims:
            if indim not in dim:
                continue
            coord_vals = np.arange(ds.sizes[dim], dtype="float32") + offsets[outdim]
            ds = ds.assign_coords({outdim: (dim, coord_vals)})
            swap_map[dim] = outdim

    ds = ds.swap_dims(swap_map)

    # Add in grid spacing and domain size for each dimension.

    for c in "xyz":
        ds[c].attrs[f"d{c}"] = 1.0
        ds[c].attrs[f"L{c}"] = np.float32(ds.dims[c] - 1)

    return ds


def great_circle_distance(lon1, lat1, lon2, lat2):

    ds = np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    ds = np.clip(ds, -1, 1)
    d = np.arccos(ds)
    return d


def confine_traj_bounds(ds_posn, ds_grid, vertical_boundary_option=1):
    """
    Confine trajectory position to domain.

    Parameters
    ----------
    pos : numpy array
        trajectory positions (n, 3).
    nx : int or float
        max value of x.
    ny : int or float
        max value of y.
    nz :  int or float
        max value of z.
    vertical_boundary_option : int, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    pos : TYPE
        DESCRIPTION.

    """
    ds_posn = regularize_coords(ds_posn, ds_grid)

    if vertical_boundary_option == 1:
        zmin = ds_grid.z.values.min()
        zmax = ds_grid.z.values.max()
        ds_posn["z"] = np.clip(ds_posn["z"], zmin, zmax)

    elif vertical_boundary_option == 2:

        ds_posn["z"] = np.clip(ds_posn["z"], 0, ds_grid.z.Lz)
        lam = 1.0 / 0.5
        k1 = ds_posn.z <= ds_grid.z.dz
        k2 = ds_posn.z >= (ds_grid.z.Lz - ds_grid.z.dz)

        ds_posn["z"] = xr.where(
            k1,
            ds_grid.z.dz * (1.0 + rnd.exponential(scale=lam, size=k1.shape)),
            ds_posn["z"],
        )
        ds_posn["z"] = xr.where(
            k2,
            ds_grid.z.Lz
            - ds_grid.z.dz * (1.0 + rnd.exponential(scale=lam, size=k2.shape)),
            ds_posn["z"],
        )

    return ds_posn


def pt_distance(ds_p1, ds_p2, ds_grid):

    # Deal with xy wraparound.

    dist_arr = np.array(
        [
            ds_p1.x.values - ds_p2.x.values,
            ds_p1.y.values - ds_p2.y.values,
            ds_p1.z.values - ds_p2.z.values,
        ]
    )

    grid_spacing = np.array([ds_grid[c].attrs[f"d{c}"] for c in "xyz"])

    if ds_grid["x"].ndim > 0:
        grid_spacing = grid_spacing[:, np.newaxis]

    match ds_grid.grid_type:
        case "xy_periodic":
            Lx = ds_grid.x.attrs["Lx"]
            Ly = ds_grid.y.attrs["Ly"]
            xerr = np.asarray(dist_arr[0])
            xerr[xerr > Lx / 2] -= Lx
            xerr[xerr < -Lx / 2] += Lx
            yerr = np.asarray(dist_arr[1])
            yerr[yerr > Ly / 2] -= Ly
            yerr[yerr < -Ly / 2] += Ly
            dist_arr = np.array(
                [
                    xerr,
                    yerr,
                    dist_arr[2],
                ]
            )
            in_domain = None
            # Normalise by grid spacing
            ndist = dist_arr / grid_spacing

        case "global":
            Lx = ds_grid.x.attrs["Lx"]
            Ly = ds_grid.y.attrs["Ly"]
            xerr = np.asarray(dist_arr[0])
            xerr[xerr > Lx / 2] -= Lx
            xerr[xerr < -Lx / 2] += Lx
            yerr = np.asarray(dist_arr[1])
            yerr[yerr > Ly / 2] -= Ly
            yerr[yerr < -Ly / 2] += Ly
            dist_arr = np.array(
                [
                    xerr,
                    yerr,
                    dist_arr[2],
                ]
            )
            in_domain = None

            ds_p1_c = confine_traj_bounds(
                ds_p1,
                ds_grid,
                vertical_boundary_option=1,
            )
            ds_p2_c = confine_traj_bounds(
                ds_p2,
                ds_grid,
                vertical_boundary_option=1,
            )

            lon1 = ds_p1_c.x.values.astype(np.float64) / Lx * 2 * np.pi
            lon2 = ds_p2_c.x.values.astype(np.float64) / Lx * 2 * np.pi
            lat1 = (ds_p1_c.y.values.astype(np.float64) / Ly - 0.5) * np.pi
            lat2 = (ds_p2_c.y.values.astype(np.float64) / Ly - 0.5) * np.pi

            gcd = great_circle_distance(lon1, lat1, lon2, lat2)
            horiz_dist = gcd * Lx / (2 * np.pi)

            vert_dist = dist_arr[2]

            ndist = np.stack(
                [horiz_dist / grid_spacing[0], vert_dist / grid_spacing[2]]
            )

        case "lam":
            ndist = dist_arr / grid_spacing
            in_domain = ds_p1.flag.values <= 1

    # Deal with single trajectory case
    if dist_arr.ndim == 1:
        dist_arr = np.expand_dims(dist_arr, 1)

    if ndist.ndim == 1:
        ndist = np.expand_dims(ndist, 1)

    return dist_arr, ndist, in_domain
