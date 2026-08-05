"""
Calculations for forward trajectory from a point in space and time using the
position scalars.
forward
    extrapolate_single_timestep
        gen_interpolator_fields
        --extrapolate first guess
        confine_traj_bounds
            regularize_coords
        -- if fixed_point_iteration
        backtrack_origin_point_iterate
            get_error_norm
                calc_backtrack_origin_dist
            pt_backtrack_origin_optimize
                calc_backtrack_origin_err
                    calc_backtrack_origin_dist
                        regularize_coords
                get_error_norm
                    calc_backtrack_origin_dist
             regularize_coords
        -- if optimize
        ds_backtrack_origin_optimize
            _pt_ds_to_arr
            pt_backtrack_origin_optimize
                calc_backtrack_origin_err
                    calc_backtrack_origin_dist
                        wrap_coords
                get_error_norm
                    calc_backtrack_origin_dist
             wrap_coords
        confine_traj_bounds
            regularize_coords
        -- set flags
        aux_coords_to_traj

"""

import math
import sys

import numpy as np
import scipy.optimize
import xarray as xr
from loguru import logger

# from tqdm.auto import tqdm
from tqdm import tqdm

from ..utils.data_to_traj import aux_coords_to_traj
from ..utils.grid import confine_traj_bounds, pt_distance
from ..utils.interpolation import gen_interpolator_fields
from ..utils.io import ds_save
from .backward import calc_trajectory_previous_position

NOT_CONVERGED = 1
LEFT_W_BOUNDARY = 2**5
LEFT_E_BOUNDARY = 2**6
LEFT_S_BOUNDARY = 2**7
LEFT_N_BOUNDARY = 2**8

LEFT = LEFT_W_BOUNDARY | LEFT_E_BOUNDARY | LEFT_S_BOUNDARY | LEFT_N_BOUNDARY

is_interactive = sys.stdout.isatty()


def calc_backtrack_origin_dist(
    ds_position_scalars,
    ds_traj_posn_org,
    ds_traj_posn,
    interpolator=None,
    interp_order=5,
):
    """
    Compute the distance between the true origin and the estimated origin
    (calculated by back-trajectory from `pt_traj_posn_next`)
    """
    ds_grid = ds_position_scalars[["x", "y", "z"]]

    # ds_traj_posn_org contains the 'current' (i.e. known) trajectory
    # positions.

    # ds_traj_posn contains a guess of the 'next' trajectory position.
    # Using this estimate of the next trajectory position use the position
    # scalars at that point to estimate where fluid originated from. If we've
    # made a good guess for the next position this estimated origin position
    # should be close to the true origin, i.e. the 'current' trajectory
    # position.
    ds_traj_posn_org_guess = calc_trajectory_previous_position(
        ds_position_scalars=ds_position_scalars,
        ds_traj_posn=ds_traj_posn,
        interpolator=interpolator,
        interp_order=interp_order,
    )

    dist_arr, ndist, in_domain = pt_distance(
        ds_traj_posn_org, ds_traj_posn_org_guess, ds_grid
    )

    return dist_arr, ndist, in_domain


def _pt_ds_to_arr(ds_pt):
    return np.array([ds_pt[c].data for c in "xyz"])


def _pt_arr_to_ds(arr_pt, ds=None):

    # Ensure arr_pt has 2 dimensions.
    arr_pt = arr_pt.reshape((3, -1))

    if ds is None:

        ds_pt = xr.Dataset()
        for n, c in enumerate("xyz"):
            ds_pt[c] = xr.DataArray(arr_pt[n, :], dims=("trajectory_number"))

    else:

        ds_pt = ds.copy()
        for n, c in enumerate("xyz"):
            ds_pt[c] = ds[c].copy(data=arr_pt[n, :])

    return ds_pt


def get_error_norm(
    scalars,
    pos_org,
    pos_est,
    grid_spacing,
    norm="max_abs_error",
    interpolator=None,
    interp_order=5,
):
    dist, ndist, in_domain = calc_backtrack_origin_dist(
        scalars,
        pos_org,
        pos_est,
        interpolator=interpolator,
        interp_order=interp_order,
    )

    if in_domain is None:
        ndist_valid = ndist
    else:
        ndist_valid = ndist[:, in_domain]

    if norm == "max_abs_error":
        # 	max(abs(x))
        err = np.linalg.norm(ndist_valid.flatten(), ord=np.inf)
    elif norm == "mean_abs_error":
        # mean(abs(x))
        err = np.mean(np.abs(ndist_valid.flatten()))
    else:
        # 2-norm(x.ravel) / N = sqrt(sum(abs(x)**2))/sqrt(N)
        # i.e. RMS
        err = np.linalg.norm(ndist_valid.flatten()) / np.sqrt(ndist_valid.size)

    return dist, in_domain, ndist, err


def backtrack_origin_point_iterate(
    ds_position_scalars,
    ds_traj_posn_org,
    ds_traj_posn_first_guess,
    interpolator,
    solver,
    interp_order=5,
    opt_one_step=False,
    maxiter=200,
    miniter=10,
    disp=False,
    relax=1.0,
    relax_reduce=0.95,
    tol=0.01,
    norm="max_abs_error",
    max_outer_loops=1,
    minim_kwargs=None,
):
    """
    Find the forward trajectory solutions as an Xarray DataSet.

    The solution is found when the the origin of air arriving
    at pt_traj_posn_next equals ds_traj_posn_org, with first guess
    pt_traj_posn_first_guess as a numpy array.

    This uses point iteration;
    optionally (solver='hybrid_hybrid_fixed_point_iterator') uses
    _pt_backtrack_origin_optimize to find solution for points that do not
    converge in the allowed iterations. Note that the optimizer often uses
    many hundreds of error evaluations, so it is worth letting the point
    iteration have lots of iterations before resorting to it.

    Point iteration can be defined as:
        x(n+1) = x(n) + relax * error(n)
    where error(n) is the distance between the required origin of the air and
    the origin at x(n).

    Parameters
    ----------
    ds_position_scalars : Xarray DataSet
        3D fields of trajectory tracer data.
    ds_traj_posn_org : Xarray DataSet
        Current trajectory positions.
    ds_traj_posn_first_guess : Xarray DataSet
        First guess of trajectory endpoints.
    interpolator : fast interp interpolator.
        Pre-calculated interpolator for fields in ds_position_scalars.
        If None, interpolator is calculated on each call to the interpolation.
    solver : str
        "fixed_point_iterator" or "hybrid_fixed_point_iterator".
    interp_order : int, optional
        Interpolation order. The default is 5.

    maxiter (int)        : Maximum number of iterations. Delault=500.
    miniter (int)        : Number of iterations to use before adjusting
                           relax. Default=10.
    relax (float)        : Initial relaxation factor. Default=0.8.
    relax_reduce (float) : Factor to mutiply relax after every miniter
                           iterations. Default=0.9.
    tol (float)          : Distance in terms of grid lengths defining
                           convergence. Default=0.01.
    opt_one_step (bool)  : If True, use single call to optimizer for
                           all trajectories. Default=False.
    disp (bool)          : Print information on convergence.
                           Default=False.
    norm (str)           : Norm to use in erro. Options are
                           'mean_abs_error', 'max_abs_error'
                           or RMS distance. Default='max_abs_error'
    max_outer_loops (int): Maximum number of outer loops of minimizer.
    minim_kwargs (dict)  : Options for minimiser.

    Returns
    -------
    Xarray DataSet, numpy array [3, n]
        New trajectory position and residual error vector.

    """

    grid_spacing = np.array([ds_position_scalars[c].attrs[f"d{c}"] for c in "xyz"])

    grid_spacing = grid_spacing[:, np.newaxis]

    # First find error in first guess.
    dist, in_domain, ndist, err = get_error_norm(
        ds_position_scalars,
        ds_traj_posn_org,
        ds_traj_posn_first_guess,
        grid_spacing,
        norm=norm,
        interpolator=interpolator,
        interp_order=interp_order,
    )

    ds_traj_posn_next = ds_traj_posn_first_guess

    # Now setup iteration
    niter = 0
    not_converged = True
    not_conv_mask = np.zeros(ds_traj_posn_org.sizes["trajectory_number"], dtype=bool)

    while not_converged:

        # We have found convergence is faster if we move slightly less
        # than the residual distance; this is the relax parameter,
        # generally <1.
        # Restrict change to no more than 1 grid box.

        delta = np.clip(dist * relax, -grid_spacing, grid_spacing)

        for i, c in enumerate("xyz"):
            if delta.shape[1] == 1:
                ds_traj_posn_next[c] += np.squeeze(delta[i])
            else:
                ds_traj_posn_next[c] += delta[i, ...]

        dist, in_domain, ndist, err = get_error_norm(
            ds_position_scalars,
            ds_traj_posn_org,
            ds_traj_posn_next,
            grid_spacing,
            norm=norm,
            interpolator=interpolator,
            interp_order=interp_order,
        )

        niter += 1
        not_converged = err > tol

        if niter >= maxiter:
            break
        # If we need a lot of iterations, reduce the relaxation factor.
        if niter % miniter == 0:
            relax *= relax_reduce

    # Are we converged?
    if niter >= maxiter:
        # Select out those trajectories that have not converged.
        ncm = np.abs(ndist) > tol
        not_conv_mask = ncm[0, :]
        for d in range(1, ncm.shape[0]):
            not_conv_mask = not_conv_mask | ncm[d, :]

        message = (
            f"Point Iteration failed to converge "
            f"in {niter:3d} iterations. "
            f"{np.sum(not_conv_mask)} trajectories did not converge. "
            f"Final error = {err}"
        )

    else:
        message = (
            f"Point Iteration converged "
            f"in {niter:3d} iterations. "
            f"Final error = {err}"
        )

    if solver == "hybrid_fixed_point_iterator" and niter >= maxiter:
        # Use optimization solution for un-converged trajectories.

        # Convert best estimate from point iteration to array.
        pt_traj_posn_next = _pt_ds_to_arr(ds_traj_posn_next)
        # Select unconverged trajectories.
        pt_traj_posn_next_not_conv = pt_traj_posn_next[:, not_conv_mask]

        message = (
            f"Optimising {pt_traj_posn_next_not_conv.shape[1]} "
            f"unconverged trajectories."
        )

        # Select out corresponding 'true' value.
        ds_traj_posn_org_subset = ds_traj_posn_org[["x", "y", "z"]].isel(
            trajectory_number=not_conv_mask
        )

        # Choice between optimising all the selected trajectories together
        # or one at a time.
        # First option is likely to be deprecated in future.
        if opt_one_step:
            pt_traj_posn_next_not_conv = pt_traj_posn_next_not_conv.flatten()
            pt_traj_posn_next_not_conv = pt_backtrack_origin_optimize(
                ds_position_scalars,
                ds_traj_posn_org_subset,
                pt_traj_posn_next_not_conv,
                interpolator,
                interp_order=interp_order,
                **minim_kwargs,
            )
        else:
            for itraj in tqdm(
                range(pt_traj_posn_next_not_conv.shape[1]),
                desc="Optimising",
                position=1,
                disable=not is_interactive,
                dynamic_ncols=is_interactive,
            ):

                tr_num = ds_traj_posn_org_subset.trajectory_number.values[itraj]
                # tqdm.write(f"Optimising unconverged trajectory {itraj}: {tr_num}.")
                # print(f"Optimising unconverged trajectory {itraj}: {tr_num}.")
                pt_traj_posn = pt_traj_posn_next_not_conv[:, itraj].flatten()
                ds_traj_posn_orig = ds_traj_posn_org_subset.sel(
                    trajectory_number=tr_num
                )
                pt_traj_posn = pt_backtrack_origin_optimize(
                    ds_position_scalars,
                    ds_traj_posn_orig,
                    pt_traj_posn,
                    interpolator,
                    interp_order=interp_order,
                    **minim_kwargs,
                )

                pt_traj_posn_next_not_conv[:, itraj] = pt_traj_posn

        # Put results back into main array.
        pt_traj_posn_next[:, not_conv_mask] = pt_traj_posn_next_not_conv.reshape(
            (3, -1)
        )
        # Convert to Dataset
        # ds_traj_posn_next_data = _pt_arr_to_ds(pt_traj_posn_next)

        for idim, c in enumerate("xyz"):
            ds_traj_posn_next.update(
                {c: ("trajectory_number", pt_traj_posn_next[idim, :])}
            )

        # Calculate final error.
        dist, in_domain, ndist, err = get_error_norm(
            ds_position_scalars,
            ds_traj_posn_org,
            ds_traj_posn_next,
            grid_spacing,
            norm=norm,
            interpolator=interpolator,
            interp_order=interp_order,
        )
        ncm = np.abs(ndist) > tol
        not_conv_mask = ncm[0, :]
        for d in range(1, ncm.shape[0]):
            not_conv_mask = not_conv_mask | ncm[d, :]

        message = f"{message} After minimization error={err}."

    if disp:
        tqdm.write(
            f"Point Iteration finished in {niter:3d} iterations. "
            f"Final error = {err}"
        )

    return ds_traj_posn_next, err, dist, ndist, in_domain, not_conv_mask, message


def pt_backtrack_origin_optimize(
    ds_position_scalars,
    ds_traj_posn_org,
    pt_traj_posn_first_guess,
    interpolator,
    interp_order=5,
    minimizer="BFGS",
    max_outer_loops=1,
    tol=0.01,
    minimize_options=None,
):
    """
    Find the forward trajectory solutions as a numpy array of points.

    The solution is found when the the origin of air arriving
    at pt_traj_posn_next equals ds_traj_posn_org, with first guess
    pt_traj_posn_first_guess as a numpy array.

    The minimizer scipy.optimize.minimize is called max_outer_loops times,
    with the previous best solution as the first guess, because, when they
    work, the optimization routines have been found to iterate to
    machine precision, which is much higher than practically needed.

    Parameters
    ----------
    ds_position_scalars : Xarray DataSet
        3D fields of trajectory tracer data.
    ds_traj_posn_org : Xarray DataSet
        Current trajectory positions.
    pt_traj_posn_first_guess : numpy array[3, n]
        First guess of trajectory endpoints.
    interpolator : fast interp interpolator.
        Pre-calculated interpolator for fields in ds_position_scalars.
        If None, interpolator is calculated on each call to the interpolation.
    minimizer : str, optional
        Methods supported by  scipy.optimize.minimize.
    interp_order : int, optional
        Interpolation order. The default is 5.
    max_outer_loops (int): Max number of calls to optimizer. Default=1.
    tol (float)          : Distance in terms of grid lengths defining
        convergence of outer loops. Default=0.01.
    minimize_options : dict, optional
        Options sent to scipy.optimize.minimize.

    Returns
    -------
    numpy array[3 * n]
        New trajectory position..

    """

    options = {
        "maxiter": 10,
        "disp": False,
    }

    if minimize_options is not None:
        options.update(minimize_options)

    def _calc_backtrack_origin_err(pt_traj_posn):

        ds_traj_posn = _pt_arr_to_ds(pt_traj_posn)

        dist_arr, ndist, in_domain = calc_backtrack_origin_dist(
            ds_position_scalars,
            ds_traj_posn_org,
            ds_traj_posn,
            interpolator=interpolator,
            interp_order=interp_order,
        )

        fndist = ndist.flatten()
        err = np.linalg.norm(fndist, ord=2) / np.sqrt(len(fndist))
        return err

    # for the minimization we will be using just a numpy-array
    # containing the (x,y,z) location

    pt_traj_posn_next = pt_traj_posn_first_guess
    niter = 0
    err_prev = pt_traj_posn_next.size * tol * 100
    while niter < max_outer_loops:
        sol = scipy.optimize.minimize(
            fun=_calc_backtrack_origin_err,
            x0=pt_traj_posn_next,
            method=minimizer,
            options=options,
        )
        pt_traj_posn_next = sol.x
        niter += 1
        err_now = sol.fun / np.sqrt(sol.x.size)
        if err_now <= tol or err_now == err_prev:  # grid_size * tol:
            break
        err_prev = err_now

    return pt_traj_posn_next


def ds_backtrack_origin_optimize(
    ds_position_scalars,
    ds_traj_posn_org,
    ds_traj_posn_first_guess,
    interpolator,
    minimization_method,
    interp_order=5,
    opt_one_step=False,
    kwargs=None,
):
    """
    Find the forward trajectory solutions as an Xarray DataSet.

    The solution is found when the the origin of air arriving
    at pt_traj_posn_next equals ds_traj_posn_org, with first guess
    pt_traj_posn_first_guess as a numpy array.

    This is a wrapper for pt_backtrack_origin_optimize which does the
    work finding the forward trajectory solutions as a numpy array of
    points. This pulls the array out of am xarray dataset then puts
    the result back in one, finally calculating the residual error.

    Parameters
    ----------
    ds_position_scalars : Xarray DataSet
        3D fields of trajectory tracer data.
    ds_traj_posn_org : Xarray DataSet
        Current trajectory positions.
    ds_traj_posn_first_guess : Xarray DataSet
        First guess of trajectory endpoints.
    interpolator : fast interp interpolator.
        If None, interpolator is calculated on each call to the interpolation.
        Pre-calculated interpolator for fields in ds_position_scalars.
    minimization_method : str
        Methods supported by  scipy.optimize.minimize.
    kwargs : dict
        Keyword arguments sent to  pt_backtrack_origin_optimize.

    Returns
    -------
    Xarray DataSet, numpy array [3, n]
        New trajectory position and residual error vector.

    """
    if kwargs is None:
        kwargs = {}

    pt_traj_posn_next = _pt_ds_to_arr(ds_traj_posn_first_guess).flatten()

    pt_traj_posn_next = pt_backtrack_origin_optimize(
        ds_position_scalars,
        ds_traj_posn_org,
        pt_traj_posn_next,
        interpolator,
        minimization_method,
        **kwargs,
    )

    ds_traj_posn_next = _pt_arr_to_ds(pt_traj_posn_next, ds=ds_traj_posn_org)

    dist, ndist, in_domain = calc_backtrack_origin_dist(
        ds_position_scalars,
        ds_traj_posn_org,
        ds_traj_posn_next,
        interpolator=interpolator,
        interp_order=interp_order,
    )

    not_conv_mask = np.zeros(ds_traj_posn_org.sizes["trajectory_number"], dtype=bool)

    err = np.linalg.norm(ndist.flatten(), ord=2)

    message = f"After minimization error={err}."

    return ds_traj_posn_next, dist, ndist, in_domain, not_conv_mask, message


def extrapolate_traj(ds_traj_posn_origin, ds_traj_posn_prev, ds_grid):

    # First guess - extrapolate from last two positions.
    # given points A and B the vector spanning from A to B is AB = B-A
    # let C = B + AB, then C = B + B - A = 2B - A

    # print(ds_grid)

    delta = xr.Dataset()
    for c in "xyz":
        delta[c] = ds_traj_posn_origin[c] - ds_traj_posn_prev[c]

    match ds_grid.grid_type:
        case "xy_cyclic":
            for c in "xy":
                L = ds_grid[c].attrs[f"L{c}"]
                delta[c] = delta[c].where(delta[c] < L / 2, delta[c] - L)
                delta[c] = delta[c].where(delta[c] > -L / 2, delta[c] + L)
        case "global":
            Lx = ds_grid["x"].attrs["Lx"]
            # Ly = ds_grid['y'].attrs['Ly']
            delta["x"] = delta["x"].where(delta["x"] < Lx / 2, delta["x"] - Lx)
            delta["x"] = delta["x"].where(delta["x"] > -Lx / 2, delta["x"] + Lx)

    ds_traj_posn_next_est = xr.Dataset()
    for c in "xyz":
        ds_traj_posn_next_est[c] = ds_traj_posn_origin[c] + delta[c]

    return ds_traj_posn_next_est


def extrapolate_single_timestep(
    ds_position_scalars_origin,
    ds_position_scalars_next,
    ds_traj_posn_prev,
    ds_traj_posn_origin,
    solver="hybrid_fixed_point_iterator",
    interp_order=5,
    opt_one_step=False,
    vertical_boundary_option=1,
    aux_coords=None,
    point_iter_kwargs=None,
    minim_kwargs=None,
):
    """
    Estimate from the trajectory position `ds_traj_posn_origin` to the next time.

    The algorithm is as follows:

    1) for a trajectory position `(x,y,z)` at a time `t` extrapolate a first
    guess for the next trajecotory position using the previous point in the
    trajectory
    2) find an optimal value for the estimated next trajectory point by
    minimizing the difference between the true origin and the point found when
    back-tracking from the current "next"-point estimate (using the position
    scalars)

    Args
    ----
    ds_position_scalars_origin: xarray DataArray
        3D gridded data at current time.
    ds_position_scalars_next: xarray DataArray
        3D gridded data at current time.
    ds_traj_posn_prev: xarray DataArray
        Trajectory positions at previous time step.
    ds_traj_posn_origin: xarray DataArray
        Trajectory positions at current time step.
    solver: Optional (default='hybrid_fixed_point_iterator').
        Method used by scipy.optimize.minimize
    interp_order: int
        Order of interpolation from grid to trajectory.
    """
    if point_iter_kwargs is None:
        point_iter_kwargs = {}
    if minim_kwargs is None:
        minim_kwargs = {}

    grid_type = ds_position_scalars_origin.attrs.get("grid_type", "xy_periodic")

    ds_grid = ds_position_scalars_origin[["x", "y", "z"]]

    # Generate interpolator for repeated interpolation of fields during
    # iteration.
    interpolator = gen_interpolator_fields(
        ds_position_scalars_next,
        interp_order=interp_order,
        grid_type=grid_type,
    )

    # traj_posn_next_est is our estimate of the trajectory positions at
    # the next time step.
    # We want the 'where from' at traj_posn_next_est to match
    # the current trajectory positions, traj_posn_origin

    # Let f(X) be the function which estimates the distance between the actual
    # origin and the point estimated by back-trajactory from the estimated next
    # point, i.e. f(X) -> X_err, we then want to minimize X_err.
    # We will use the Eucledian distance (L2-norm) so that we minimize the
    # magnitude of this error

    ds_traj_posn_next_est = extrapolate_traj(
        ds_traj_posn_origin, ds_traj_posn_prev, ds_grid
    )

    if "fixed_point_iterator" in solver:

        (
            ds_traj_posn_next,
            err,
            dist,
            ndist,
            in_domain,
            not_conv_mask,
            message,
        ) = backtrack_origin_point_iterate(
            ds_position_scalars_next,
            ds_traj_posn_origin,
            ds_traj_posn_next_est,
            interpolator,
            solver,
            interp_order=interp_order,
            opt_one_step=opt_one_step,
            minim_kwargs=minim_kwargs,
            **point_iter_kwargs,
        )

    else:

        (
            ds_traj_posn_next,
            dist,
            ndist,
            in_domain,
            not_conv_mask,
            message,
        ) = ds_backtrack_origin_optimize(
            ds_position_scalars_next,
            ds_traj_posn_origin,
            ds_traj_posn_next_est,
            interpolator,
            solver,
            interp_order=interp_order,
            opt_one_step=opt_one_step,
            kwargs=minim_kwargs,
        )

    for c in "xyz":
        ds_traj_posn_next[c] = ds_traj_posn_next[c].astype("float32")

    ds_traj_posn_next = confine_traj_bounds(
        ds_traj_posn_next, ds_grid, vertical_boundary_option=vertical_boundary_option
    )

    # Copy in final error measure for each trajectory.
    for i, c in enumerate("xyz"):
        derr = xr.DataArray(
            dist[i, :].astype(np.float32),
            coords={
                "trajectory_number": ds_traj_posn_next.coords[
                    "trajectory_number"
                ].values
            },
        )

        ds_traj_posn_next[f"{c}_err"] = derr

    flags = ds_traj_posn_origin.flag.values & LEFT

    flags[not_conv_mask] |= NOT_CONVERGED

    if ds_position_scalars_origin.grid_type.lower() == "lam":
        x = ds_traj_posn_next["x"].values
        flags[x < ds_position_scalars_origin["x"].values[0]] |= LEFT_W_BOUNDARY
        flags[x > ds_position_scalars_origin["x"].values[-1]] |= LEFT_E_BOUNDARY

        y = ds_traj_posn_next["y"].values
        flags[y < ds_position_scalars_origin["y"].values[0]] |= LEFT_S_BOUNDARY
        flags[y > ds_position_scalars_origin["y"].values[-1]] |= LEFT_N_BOUNDARY

    ds_traj_posn_next["flag"] = xr.DataArray(
        flags,
        coords={"trajectory_number": ds_traj_posn_prev.coords["trajectory_number"]},
    )

    # Set time coordinate.
    ds_traj_posn_next = ds_traj_posn_next.assign_coords(
        time=ds_position_scalars_next.time
    )

    if "forecast_period" in ds_traj_posn_next.coords:
        ds_traj_posn_next = ds_traj_posn_next.drop_vars(["forecast_period"])

    file_index = ds_traj_posn_origin.coords["time_index"].item() + 1
    ds_traj_posn_next = ds_traj_posn_next.assign_coords(time_index=file_index)

    if aux_coords is not None:
        # ds_grid = ds_position_scalars_origin[["x", "y", "z"]]

        # ds_traj_posn_next_conf = confine_traj_bounds(
        #     ds_traj_posn_next, ds_grid,
        #     vertical_boundary_option=vertical_boundary_option
        # )

        ds_traj_posn_next = aux_coords_to_traj(
            ds_position_scalars_origin, ds_traj_posn_next, aux_coords, interp_order=1
        )

    return ds_traj_posn_next, message


def forward(
    ds_position_scalars,
    ds_back_trajectory,
    da_times,
    interp_order=5,
    solver="fixed_point_iterator",
    vertical_boundary_option=1,
    opt_one_step=False,
    point_iter_kwargs=None,
    minim_kwargs=None,
    output_path=None,
    aux_coords=None,
):
    """
    Integrate trajectory forward one timestep.

    Using the position scalars `ds_position_scalars` integrate forwards from
    the last point in `ds_back_trajectory` to the times in `da_times`. The
    backward trajectory must contain at least two points in time as these are
    used for the initial guess for the forward extrapolation.

    The output dataset contains both the trajectory position and the
    residual error.

    Three solvers are available.

    The first uses the minimizer scipy.optimize.minimize to minimise the
    residual error. The keyword 'solver' should be set to one of the options
    for this function. The default (recommended) is 'BFGS'.

    See ds_backtrack_origin_optimize for available options.

    The second uses point iteration. Selected using keyword 'solver' set to
    "fixed_point_iterator". This is generally much faster than
    scipy.optimize.minimize, but some trajectories do not converge.

    See backtrack_origin_point_iterate for available options.

    The third is a combination of the first two, selected using keyword
    'solver' set to 'hybrid_fixed_point_iterator'.

    See backtrack_origin_point_iterate for available options.

    """
    # print(f'{ds_position_scalars=}')

    input_times = list(ds_position_scalars["time"].values)

    if ds_back_trajectory.time.count() < 2:
        raise Exception(
            "The back trajectory must contain at least two points for the forward"
            " extrapolation have an initial guess for the direction"
        )

    # create a copy of the backward trajectory so that we have a dataset into
    # which we will accumulate the full trajectory
    ds_traj = ds_back_trajectory.copy()

    if da_times.size == 0:
        logger.info("No forward trajectories requested")
        return ds_traj

    # step forward in time, `t_forward` represents the time we're of the next
    # point (forward) of the trajectory
    for t_next in tqdm(
        da_times,
        desc="forward",
        position=0,
        disable=not is_interactive,
        dynamic_ncols=is_interactive,
    ):

        ds_traj_posn_origin = ds_traj.isel(time=-1)

        t_origin = ds_traj_posn_origin.time

        ds_position_scalars_origin = ds_position_scalars.sel(time=t_origin)
        # print(f'{ds_position_scalars_origin=}')

        ds_position_scalars_next = ds_position_scalars.sel(time=t_next)
        # print(f'{ds_position_scalars_next=}')

        # for the original direction estimate we need *previous* position (i.e.
        # where we were before the "origin" point)
        ds_traj_posn_prev = ds_traj.isel(time=-2)
        # ds_traj_posn_prev = ds_traj.isel(time=[-3,-2])

        ds_traj_posn_est, message = extrapolate_single_timestep(
            ds_position_scalars_origin=ds_position_scalars_origin,
            ds_position_scalars_next=ds_position_scalars_next,
            ds_traj_posn_prev=ds_traj_posn_prev,
            ds_traj_posn_origin=ds_traj_posn_origin,
            interp_order=interp_order,
            solver=solver,
            vertical_boundary_option=vertical_boundary_option,
            opt_one_step=opt_one_step,
            aux_coords=aux_coords,
            point_iter_kwargs=point_iter_kwargs,
            minim_kwargs=minim_kwargs,
        )

        tqdm.write(message)

        if output_path is not None:
            out_fmt = f"0{math.ceil(math.log10(len(input_times)))}"
            ds_traj_posn_est = ds_save(ds_traj_posn_est, output_path, fmt=out_fmt)

        ds_traj = xr.concat([ds_traj, ds_traj_posn_est], dim="time")

        for c in "xyz":
            ds_traj[c] = ds_traj[c].astype("float32")

    return ds_traj
