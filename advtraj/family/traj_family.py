# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 10:29:03 2022

@author: Peter Clark

"""
import math
import os.path
import time
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import xarray as xr
from cohobj.object_tools import (
    box_bounds,
    box_overlap_with_wrap,
    get_object_labels,
    refine_object_overlap,
    refine_object_overlap_fast,
    tr_data_at_time,
    tr_data_obj,
    tr_objects_to_numpy,
    unsplit_objects,
)

from ..integrate import integrate_trajectories
from ..utils.data_to_traj import data_to_traj
from ..utils.point_selection import mask_to_positions


def traj_family(
    source,
    mask,
    output_path_base,
    get_objects=True,
    start_ref_time=None,
    end_ref_time=None,
    steps_backward=None,
    steps_forward=None,
    interp_order=5,
    forward_solver="fixed_point_iterator",
    options=None,
):
    """
    Generate a family of trajectories.

    Parameters
    ----------
    source : xr.Dataset
        Gridded input data.
    mask : xr.DataArray
        Mask defining startign points at each time in source.
    output_path_base : str
        Generic path for output family - an index nmber will be added.
    get_objects : bool, optional
        Find object ids at each reference time. The default is True.
    start_ref_time : float, optional
        First reference time in sourec data to use. The default is None.
    end_ref_time : float, optional
        Last reference time in sourec data to use.. The default is None.
    steps_backward : int, optional
        Number of timesteps back from the reference in each set of
        trajectories. The default is None.
    steps_forward : int, optional
        Number of timesteps back from the reference in each set of
        trajectories.. The default is None.
    interp_order : int, optional
        Interpolation order to use in trajectory calculations.
        The default is 5.
    forward_solver : str, optional
        Solver to use in forward trajectory calculation. The default is "fixed_point_iterator".
    options : dict, optional
        Options for trajectory calculation. The default is None.

    Returns
    -------
    traj_path_list : list(str)
        List of paths for trajectory family.

    """

    source_times = source.get_index("time")
    start_ref_index = source_times.get_loc(start_ref_time)
    end_ref_index = source_times.get_loc(end_ref_time)

    traj_path_list = []

    nsig = "_{" + f":0{math.ceil(math.log10(end_ref_index + 1))}d" + "}.nc"

    for i_ref_time in range(start_ref_index, end_ref_index + 1):

        mask_at_ref_time = mask.isel(time=i_ref_time)

        source_starting_points, olab = _get_starting_points(
            mask_at_ref_time, get_objects
        )

        time1 = time.perf_counter()

        traj = integrate_trajectories(
            ds_position_scalars=source,
            ds_starting_points=source_starting_points,
            steps_backward=steps_backward,
            steps_forward=steps_forward,
            interp_order=interp_order,
            forward_solver=forward_solver,
            point_iter_kwargs=options["pioptions"],
            minim_kwargs=options["minoptions"],
        )

        time2 = time.perf_counter()

        delta_t = time2 - time1

        traj.attrs["elapsed_time"] = delta_t

        if get_objects:
            traj = traj.assign_coords({"object_label": olab})
            Lx = source.x.Lx
            Ly = source.y.Ly

            traj = unsplit_objects(traj, Lx, Ly)

        output_path = output_path_base + nsig.format(i_ref_time)
        traj.to_netcdf(output_path)

        print(traj)
        print(f"Trajectories saved to {output_path}")

        traj.close()

        traj_path_list.append(output_path)

    return traj_path_list


def _get_starting_points(mask, get_objects=True):
    ds_starting_points = mask_to_positions(mask).rename(
        {"pos_number": "trajectory_number"}
    )

    for c in "xyz":
        ds_starting_points[f"{c}_err"] = xr.zeros_like(ds_starting_points[c])

    if get_objects:
        olab = (
            get_object_labels(mask)
            .rename({"pos_number": "trajectory_number"})
            .drop("time")
        )
    else:
        olab = None

    return ds_starting_points, olab


def traj_name_to_data_name(traj_file: str, opdir: str = None, append: bool = False):
    """
    Convert trajectory file name to data file name.

    Parameters
    ----------
    traj_file : str
        File path.
    opdir : str, optional
        New directory name. The default is None.
    append : bool, optional
        True: append '_data' to file name. The default is False.
        False: replace 'trajectories' with 'data'.
    Returns
    -------
    data_file_name : str
        Data file name.

    """

    if opdir is None:
        dir_name = os.path.dirname(traj_file)
    else:
        dir_name = opdir

    file_name = os.path.basename(traj_file)
    if append:
        data_file_name = dir_name + "/" + file_name.replace(".nc", "_data.nc")
    else:
        data_file_name = dir_name + "/" + file_name.replace("trajectories", "data")

    return data_file_name


def summarise_traj_family(family_times):
    """
    Generate string summary from trajectory family Dataset.

    Parameters
    ----------
    family_times : xr.Dataset
       Summary information generated by analyse_traj_family.

    Returns
    -------
    summary : list(str)
        Visual summary of data times.

    """

    # ntraj = family_times.time_start.size

    time_min = family_times.attrs["time_min"]
    time_max = family_times.attrs["time_max"]
    timestep = family_times.attrs["timestep"]

    ntimes = int(round((time_max - time_min) / timestep)) + 1

    summary = []
    times = np.linspace(time_min, time_max, ntimes)
    # print(times)
    for this_time in times:
        line = f"{this_time}: "
        for time_start, time_ref, time_end in zip(
            family_times.time_start.values,
            family_times.time_ref.values,
            family_times.time_end.values,
        ):
            # print(this_time, time_start[i], time_ref[i], time_end[i])
            c = "x"
            if this_time >= time_start and this_time <= time_end:
                c = "*"
            if this_time == time_ref:
                c = "R"
            line += c
        # print(line)
        summary.append(line)

    return summary


def analyse_traj_family(traj_files: list):
    """
    Extract summary information from list of trajectory family paths.

    Parameters
    ----------
    traj_files : list(str)
        Trajectory family paths.

    Returns
    -------
    family_times : xt.Dataset
        Summary information.

    """
    ntraj = len(traj_files)
    time_start = []
    time_end = []
    time_ref = []
    timestep = None
    for path in traj_files:
        # print(path)
        with xr.open_dataset(path) as ds:
            time_start.append(ds.time.min().item())
            time_end.append(ds.time.max().item())
            time_ref.append(ds.ref_time.item())
            if timestep is None:
                timestep = ds.attrs["trajectory timestep"]
            else:
                if ds.attrs["trajectory timestep"] != timestep:
                    print(
                        f"Warning: unequal timestep between files "
                        f"{timestep=} {ds.attrs['timestep']=}"
                    )

    time_min = np.squeeze(np.array(time_start).min())
    time_max = np.squeeze(np.array(time_end).max())

    for i in range(ntraj):
        print(f"{i:2d} {time_start[i]}, {time_ref[i]}, {time_end[i]}")

    family_times = xr.Dataset(
        {
            "time_start": ("itime", time_start),
            "time_ref": ("itime", time_ref),
            "time_end": ("itime", time_end),
            "path": ("itime", traj_files),
        },
        attrs={
            "timestep": timestep,
            "time_min": time_min,
            "time_max": time_max,
        },
    )

    return family_times


def data_to_traj_family(
    traj_files: list,
    source: xr.Dataset,
    varlist: list,
    odir: str,
    interp_order: int = 5,
    append: bool = False,
) -> list:
    """
    Interpolate gridded data to trajectory family.

    Parameters
    ----------
    traj_files : list(str)
        Input trajectory filenames.
    source : xr.DataSet
        Gridded input data.
    varlist : list(str)
        Variable names to interpolate.
    odir : str
        output directory name.
    interp_order : int, optional
        Order of Lagrange interpolation. The default is 5.
    append: bool
        Map trajectory file name to data file name by appending '_data' (True)
        or replacing 'trajectories' with 'data' (False). The default is False.

    Returns
    -------
    list(str)
        Output filenames corresponding to traj_files.
        traj_name_to_data_name is used to map one to the other.
    """
    pathlist = []
    for traj_file in traj_files:

        file_num = str(traj_file).split(".")[0].split("_")[-1]
        print(f"Processing file {file_num}")
        output_path_data = traj_name_to_data_name(traj_file, opdir=odir)

        ds_traj = xr.open_dataset(traj_file)

        #        print(ds_traj)
        ds_traj_data = data_to_traj(
            source,
            ds_traj,
            varlist,
            output_path_data,
            interp_order=interp_order,
        )
        print(ds_traj_data)

        pathlist.append(output_path_data)


def family_coords(family, coord_name):
    coords = [c[coord_name].values.squeeze() for c in family]
    return coords


def find_match_obj_at_time(
    traj_iobj: Union[xr.Dataset, dict],
    b_test,
    # traj_box: xr.Dataset,
    match_traj: Union[xr.Dataset, dict],
    match_traj_bounds: xr.Dataset,
    matching_objects_at_time: Union[dict, None],
    all_matching_objects: dict,
    match_time_back: float,
    fast: bool = True,
    use_numpy: bool = False,
) -> (Union[dict, None], dict):

    nx = int(round(match_traj_bounds.attrs["Lx"] / match_traj_bounds.attrs["dx"]))
    ny = int(round(match_traj_bounds.attrs["Ly"] / match_traj_bounds.attrs["dy"]))

    b_set = box_bounds(match_traj_bounds.sel(time=match_time_back))

    corr_box = box_overlap_with_wrap(b_test, b_set, nx, ny)

    if corr_box is None:
        # print(f"No matching objects master trajectory {iobj=} "
        #       f"{match_time_back=}")
        return matching_objects_at_time, all_matching_objects

    if matching_objects_at_time is not None:
        matching_objects_at_match_time = {}

    obj_labels = list(corr_box.object_label.values)

    # print(f"Matching object(s): {matching_objects_at_time[match_time_back]}")

    if fast:
        for o in obj_labels:
            if matching_objects_at_time is not None:
                matching_objects_at_match_time[o] = None
            all_matching_objects[o] = None
    else:
        if use_numpy:

            traj_iobj_time = tr_data_at_time(traj_iobj, match_time_back)
            # ind1 = np.where(traj_iobj['time'] == match_time_back)[0][0]
            # traj_iobj_time = {'xyz':traj_iobj['xyz'][:, ind1, :],
            #                   'mask': traj_iobj['mask'][ind1, :],
            #                   'object_label': traj_iobj['object_label'],
            #                   }

            match_traj_time = tr_data_at_time(match_traj, match_time_back)
            # ind2 = np.where(match_traj['time'] == match_time_back)[0][0]
            # match_traj_time = {'xyz': match_traj['xyz'][:, ind2, :],
            #                    'mask': match_traj['mask'][ind2, :],
            #                    'object_label': match_traj['object_label'],
            #                   }

            objs = find_refined_overlap_fast(
                traj_iobj_time, match_traj_time, obj_labels, nx, ny
            )

        else:
            traj_iobj_time = traj_iobj.sel(time=match_time_back)
            match_traj_time = match_traj.sel(time=match_time_back)

            objs = find_refined_overlap(traj_iobj_time, match_traj_time, obj_labels)

        for o, overlap in objs:
            if matching_objects_at_time is not None:
                matching_objects_at_match_time[o] = overlap
            if o in all_matching_objects:
                if overlap > all_matching_objects[o]:
                    all_matching_objects[o] = overlap
            else:
                all_matching_objects[o] = overlap

    if matching_objects_at_time is not None:
        matching_objects_at_time[match_time_back] = matching_objects_at_match_time

    return (matching_objects_at_time, all_matching_objects)


def find_matching_objects(
    iobj: int,
    traj_iobj: xr.Dataset,
    traj_iobj_bounds: xr.Dataset,
    match_traj: xr.Dataset,
    match_traj_bounds: xr.Dataset,
    ref_time_only: bool = True,
    fast: bool = True,
    use_numpy: bool = False,
) -> dict:
    """
    Find objects in one set of trajectories overlapping one object in another.

    Parameters
    ----------
    iobj : int
        Object number to match.
    traj_iobj : xr.Dataset
        trajectory dataset containing iobj.
        Not used if fast=True so can be None.
    traj_iobj_bounds : xr.Dataset
        bounds of active part of traj_obj.
    match_traj : xr.Dataset
        trajectory dataset containing objects to match iobj.
    match_traj_bounds : xr.Dataset
        bounds of active part of match_traj.
    fast : bool, optional
        Only use box overlap if True. Otherwise, find proportion of
        nearest grid point overlap. The default is True.

    Returns
    -------
    matching_objects_at_time : dict
        Dictionary.
        Keys equal to times in match_traj_bounds with matching objects
        Values equal to list of obj ids (fast=True) or tuples (object id, overlap)
        Special key 'All times' contains similar list for objects matched at any time.

    """

    traj_box = box_bounds(traj_iobj_bounds)
    traj_times = traj_iobj_bounds.time

    matching_objects_at_time = {}
    all_matching_objects = {}

    if ref_time_only:
        match_time_back = match_traj_bounds.ref_time.item()
        b_test = traj_box.sel(time=match_time_back)
        if np.isnan(b_test.x_min).item():
            # print(f"No points in master trajectory {iobj=} "
            #       f"{match_time_back=}")
            return matching_objects_at_time
        (dummy, all_matching_objects) = find_match_obj_at_time(
            traj_iobj,
            b_test,
            # traj_box,
            match_traj,
            match_traj_bounds,
            None,
            all_matching_objects,
            match_time_back,
            fast=fast,
            use_numpy=use_numpy,
        )

        if len(all_matching_objects) > 0:
            matching_objects_at_time["Ref time"] = all_matching_objects

    else:

        # print(f"Matching objects in trajectories reference time {match_time}")

        # Iterate backwards over all set times from master_ref to start.
        # for it_back in range(0, master_ref_time_index + 1) :
        for match_time_back in traj_times.values:

            # Time to make comparison
            # print(f"Matching objects at time {match_time_back}")
            if match_time_back not in match_traj.time:
                # print(f"Matching objects at time {match_time_back}")
                # print("Time not in trajectory.")
                continue

            b_test = traj_box.sel(time=match_time_back)
            if np.isnan(b_test.x_min).item():
                # print(f"No points in master trajectory {iobj=} "
                #       f"{match_time_back=}")
                continue

            (matching_objects_at_time, all_matching_objects) = find_match_obj_at_time(
                traj_iobj,
                b_test,
                traj_box,
                match_traj,
                match_traj_bounds,
                matching_objects_at_time,
                all_matching_objects,
                match_time_back,
                fast=fast,
                use_numpy=use_numpy,
            )

        if len(all_matching_objects) > 0:
            matching_objects_at_time["All times"] = all_matching_objects

    return matching_objects_at_time


def find_matching_objects_ref(
    traj_family_list: list,
    master_ref_time: float = None,
    select: list = None,
    ref_time_only: bool = True,
    forward: bool = True,
    adjacent_only: bool = True,
    fast: bool = True,
    use_numpy=False,
) -> dict:
    """
    Generate a dict of objects in a trajectory family matching selected objects
    at a given reference time all times they match.

    Parameters
    ----------
    traj_family_list : list
        List of tuples containing (trajectories:xr.Dataset, bounds:xr.Dataset)
    master_ref_time : float, optional
        Time to obtain reference objects to match. The default is None,
        which gives the last dataset in traj_family_list.
    select : list(int), optional
        List of object ids to match. The default is None, which translates to
        all in the trajectory dataset.
    fast : bool, optional
        Only use box overlap if True. Otherwise, find proportion of
        nearest grid point overlap. The default is True.

    Returns
    -------
    dict
        Keys:
              "master_ref"           : master_ref actually used.
              "master_ref_time"      : actual time of master_ref,
              "master_ref_time_index": index in trajectory data of
                                       master_ref_time,
              "matching_objects"     : dict.
                  Keys: Object ids in reference dataset matched.
                  Items: dict with keys the reference times of trajectories
                  with matching objects.

    @author: Peter Clark

    """
    traj_family = [t[0] for t in traj_family_list]

    ref_times = family_coords(traj_family, "ref_time")

    if master_ref_time is None:
        master_ref = len(traj_family_list) - 1
        master_ref_time = ref_times[-1]
    else:
        master_ref = ref_times.index(master_ref_time)

    if master_ref is None:
        master_ref = len(traj_family_list) - 1

    traj, traj_bounds = traj_family_list[master_ref]

    traj_times = traj.time.values

    if adjacent_only:
        ind = np.where(traj_times == master_ref_time)[0][0]
        traj_times = traj_times[ind - 1 : ind + 2]

    if not forward:
        traj_times = traj_times[::-1]

    master_ref_time = traj.ref_time.item()

    print(f"Reference time: {master_ref_time}")

    master_ref_times = traj.get_index("time")

    master_ref_time_index = master_ref_times.get_loc(master_ref_time)

    if select is None:
        select = traj_bounds.object_label.values

    if not fast and use_numpy:
        traj = tr_objects_to_numpy(traj, to_gridpoint=True)

    mol = {}

    # Iterate over objects in master_ref.
    for iobj in select:

        print(f"Object {iobj}")

        traj_iobj_bounds = traj_bounds.sel(object_label=iobj)
        if not fast:
            if use_numpy:
                traj_iobj = tr_data_obj(traj, iobj)
                # m = traj['object_label'] == iobj
                # traj_iobj = {'xyz': traj['xyz'][..., m],
                #              'mask': traj['mask'][..., m],
                #              'ref_time': traj['ref_time'],
                #              'time': traj['time'],
                #              'object_label':[iobj],
                #             }
            else:
                traj_iobj = traj.where(traj.object_label == iobj, drop=True)
        else:
            traj_iobj = None

        matching_objects = {}

        # Iterate over times in master_ref.
        for match_time in traj_times:

            if match_time == master_ref_time:
                continue
            # We are looking for objects in match_traj matching those in traj.

            if match_time not in ref_times:
                continue

            # print(
            # f"Matching objects in trajectories reference time {match_time}")

            match_index = ref_times.index(match_time)

            match_traj, match_traj_bounds = traj_family_list[match_index]

            # matching_objects[match_time] will be those objects in match_traj
            # which match traj at any time.

            if not fast and use_numpy:
                match_traj = tr_objects_to_numpy(match_traj, to_gridpoint=True)
            matching_objects_at_time = find_matching_objects(
                iobj,
                traj_iobj,
                traj_iobj_bounds,
                match_traj,
                match_traj_bounds,
                ref_time_only=ref_time_only,
                fast=fast,
                use_numpy=use_numpy,
            )
            if matching_objects_at_time:
                matching_objects[match_time] = matching_objects_at_time

        mol[iobj] = matching_objects

    ret_dict = {
        "master_ref": master_ref,
        "master_ref_time": master_ref_time,
        "master_ref_time_index": master_ref_time_index,
        "matching_objects": mol,
    }
    return ret_dict


def find_family_matching_objects(
    traj_family_list: List[Tuple[xr.Dataset, xr.Dataset]],
    ref_time_only: bool = True,
    forward: bool = True,
    adjacent_only: bool = True,
    fast: bool = True,
    use_numpy=False,
) -> dict:
    """
    Generate a dictionary of all objects in a family of trajectories
    matching all the objects in all reference times.

    Parameters
    ----------
    traj_family_list : List[Tuple[xr.Dataset,xr.Dataset]]
        List of tuples containing (trajectories:xr.Dataset, bounds:xr.Dataset)
    ref_time_only : bool, optional
        If true, restrict matching to objects at reference time.
        The default is True.
    forward : bool, optional
        If True choose reference times if increasing order.
        If False choose reference times if decreasing order.
        The default is True.
    adjacent_only : bool, optional
        If True only consider times adjacent to each other.
        The default is True.
    fast : bool, optional
        If True use estimate of fractional overlap, otherwise
        just use box overlap. The default is True.

    Returns
    -------
    dict
        Keys: reference times in trajectory family.
        Values: dicts of matching objects at different reference times.
        See `find_matching_objects_ref` for structure.

    """

    traj_family = [t[0] for t in traj_family_list]

    ref_times = family_coords(traj_family, "ref_time")

    if not forward:
        ref_times = ref_times[::-1]

    mol_family = {}

    for master_ref_time in ref_times:
        mol = find_matching_objects_ref(
            traj_family_list,
            float(master_ref_time),
            ref_time_only=ref_time_only,
            forward=forward,
            adjacent_only=adjacent_only,
            fast=fast,
            use_numpy=use_numpy,
        )
        mol_family[float(master_ref_time)] = mol

    return mol_family


def find_refined_overlap_fast(traj1, traj2, obj_labels, nx, ny):
    """
    Compute fractional overlap between object in traj1 and requested objects
    in traj2.

    Parameters
    ----------
    traj1 : xr.Dataset
        Trajectories from a single object.
    traj2 : xr.Dataset
       Trajectories containing objects with ids in obj_labels.
    obj_labels : list
       List of object ids required to find overlap with object in traj1.

    Returns
    -------
    list(tuple)
        List of (obj_labels, overlap).

    """
    over = []
    tr2ol = traj2["object_label"]

    for objm in obj_labels:

        # print(f"{b_set.sel(object_label=objm)=}")
        m = tr2ol == objm

        traj2_obj = {"xyz": traj2["xyz"][:, m], "mask": traj2["mask"][m]}

        inter = refine_object_overlap_fast(traj1, traj2_obj, nx, ny)

        over.append(inter)
        # over.append((objm, inter))

    return list(zip(obj_labels, over))
    # return over


def find_refined_overlap(traj1, traj2, obj_labels):
    """
    Compute fractional overlap between object in traj1 and requested objects
    in traj2.

    Parameters
    ----------
    traj1 : xr.Dataset
        Trajectories from a single object.
    traj2 : xr.Dataset
       Trajectories containing objects with ids in obj_labels.
    obj_labels : list
       List of object ids required to find overlap with object in traj1.

    Returns
    -------
    list(tuple)
        List of (obj_labels, overlap).

    """
    over = []
    for objm in obj_labels:

        # print(f"{b_set.sel(object_label=objm)=}")

        traj2_obj = traj2.where(traj2.object_label == objm, drop=True)

        inter = refine_object_overlap(traj1, traj2_obj)

        over.append(inter)

    return list(zip(obj_labels, over))


def print_matching_objects(
    matching_objects, select=None, ref_times_sel=None, full=True
):
    """
    Print matching object list.

    See find_matching_objects.

    """

    print(f' Master Reference File Index: {matching_objects["master_ref"]}')
    print(f' Master Reference Time: {matching_objects["master_ref_time"]}')

    if select is None:
        select = list(matching_objects["matching_objects"].keys())
    mol = matching_objects["matching_objects"]
    for iobj in select:
        print(f"Object {iobj}")
        for match_time, matches in mol[iobj].items():
            if ref_times_sel is not None and match_time not in ref_times_sel:
                continue
            print(f"  Reference time: {match_time}")

            for match_time_back, objs in matches.items():
                # rep = f'{objs}'
                rep = ""
                for obj, ov in objs.items():
                    if ov is None:
                        rep += f"{obj} "
                    else:
                        rep += f"{obj}: {ov:4.2f} "
                if len(rep) > 0:
                    if match_time_back == "All times":
                        print(f"        {match_time_back} matching obj: {rep}")
                    else:
                        if full:
                            print(f"    Time: {match_time_back} matching obj: {rep}")
    return


def matching_obj_to_nodelist(
    matching_objects: dict,
    iobj: int,
    ref_times_sel: Union[List[float], None] = None,
    overlap_thresh: float = None,
) -> list:
    """
    Generate list of nodes representing all objects in matching_objects
    that overlap iobj.

    Parameters
    ----------
    matching_objects : dict
        Matching objects. See `find_matching_objects_ref`.
    iobj : int
        Target object.
    ref_times_sel : Union[List[float], None], optional
        Only include reference times in this list. The default is None.
    overlap_thresh : float, optional
       Only include objects with overlap >= object_thresh. The default is None.

    Returns
    -------
    list
        List of nodes of type (time: float, object_number:int).

    """

    mol = matching_objects["matching_objects"]
    master_ref_time = matching_objects["master_ref_time"]
    nodelist = [(master_ref_time, iobj)]

    for match_time, matches in mol[iobj].items():
        if not matches:
            continue
        if ref_times_sel is not None and match_time not in ref_times_sel:
            continue
        if "All times" in matches.keys():
            objs = matches["All times"]
        else:
            objs = matches["Ref time"]
        for obj, ov in objs.items():
            if overlap_thresh is not None and ov < overlap_thresh:
                continue
            nodelist.append((match_time, obj))

    return nodelist


def _graph_matches(G, node, match_time, matches, ntype):

    ref_time = node[0]

    if "All times" in matches.keys():
        objs = matches["All times"]
    else:
        objs = matches["Ref time"]

    # print(f'{iobj=} {match_time=}')
    for obj, ov in objs.items():
        linked_node = (match_time, obj)
        if linked_node not in G.nodes:
            G.add_node(linked_node)
            # print(f'New node: {linked_node=}')

        if match_time < ref_time:
            E = (linked_node, node)
        else:
            E = (node, linked_node)
        if ntype == 1 and nx.is_simple_path(G, E):
            continue
        elif ntype == 2 and (
            E[1] in nx.descendants(G, E[0]) or E[0] in nx.descendants(G, E[1])
        ):
            continue
        G.add_edge(E[0], E[1], ntype=ntype)
        # print(f'New edge {E}')
        if ov is not None:
            G.edges[E[0], E[1]]["max_overlap"] = ov
    return G


def graph_matching_objects(
    mol_family: dict,
    include_types: Union[
        Tuple[
            int,
        ],
        None,
    ] = None,
) -> nx.DiGraph:
    """
    Generate acyclic directed graph of all the objects with ovelap.

    Parameters
    ----------
    mol_family : dict
        Matching objects in the trajectory family.
    include_types : Union[Tuple[int,], None], optional
        Overlap types to include.
        type 1: overlap on adjacent timesteps.
        type 2: only overlap on none-adjacent timesteps.
        The default is None which means all.

    Returns
    -------
    G : nx.DiGraph
        acyclic directed graph.

    """

    if include_types is None:
        include_types = (1, 2)

    ref_times = list(mol_family.keys())
    ref_times.sort()
    G = nx.DiGraph()

    # Pass 1: Adjacent times:

    for ntype in include_types:
        for it, master_ref_time in enumerate(ref_times[:-1]):

            # print(f'{master_ref_time=}')

            mol = mol_family[master_ref_time]
            next_time = ref_times[it + 1]
            if it == 0:
                prev_time = 0
            else:
                prev_time = ref_times[it - 1]

            for iobj, matching_objects in mol["matching_objects"].items():
                node = (master_ref_time, iobj)
                if node not in G.nodes:
                    # print(f'New node: {node=}')
                    G.add_node(node)

                for match_time, matches in matching_objects.items():
                    if not matches:
                        continue

                    if ntype == 1 and match_time != next_time:
                        continue
                    if ntype == 2 and match_time == next_time:
                        continue
                    if ntype == 2 and it > 0 and match_time == prev_time:
                        continue

                    G = _graph_matches(G, node, match_time, matches, ntype)

    return G


def start_objects(G: nx.DiGraph) -> list:
    """
    Generate list of all the starting nodes in a directed graph.

    Parameters
    ----------
    G : nx.DiGraph
        acyclic directed graph.

    Returns
    -------
    list
        List of nodes with no nodes connecting to.

    """
    start_obj = []
    for n in G.nodes:
        if G.in_degree(n) == 0:
            start_obj.append(n)
    return start_obj


def end_objects(G: nx.DiGraph) -> list:
    """
    Generate list of all the ending nodes in a directed graph.

    Parameters
    ----------
    G : nx.DiGraph
        acyclic directed graph.

    Returns
    -------
    list
        List of nodes with no nodes connecting from.

    """
    end_obj = []
    for n in G.nodes:
        if G.out_degree(n) == 0:
            end_obj.append(n)
    return end_obj


def subgraph_ntype(G, ntype):
    def fedge(n1, n2):
        return G.get_edge_data(n1, n2)["ntype"] == ntype

    return nx.subgraph_view(G, filter_edge=fedge)


def subgraph_overlap_thresh(G, thresh):
    def fedge(n1, n2):
        return G.get_edge_data(n1, n2)["max_overlap"] >= thresh

    return nx.subgraph_view(G, filter_edge=fedge)


def pred(G: nx.DiGraph, o, d: Union[dict, None] = None) -> dict:
    """
    Generate predecessors to node in DiGraph.
    I have found that nx.dfs_predecessors does not work properly on
    directed graphs (I think it generates the predecessors of all successors).

    Parameters
    ----------
    G : nx.DiGraph
        Acyclic directed graph.
    o : Node key in G.
        Required end node.
    d : Union[dict, None], optional
        Used recursively. Do not supply on call.

    Returns
    -------
    dict
        Key: Node in G.
        Values: list of nodes connecting to Key.

    """
    if d is None:
        d = {}
    if G.in_degree(o):
        pl = list(G.predecessors(o))
        if pl:
            d[o] = pl
            for p in pl:
                d = pred(G, p, d=d)

    return d


def related_objects(
    G: nx.DiGraph,
    obj,
    ref_times_sel: Union[list, None] = None,
    overlap_thresh: Union[float, None] = None,
    ntypes: Union[
        Tuple[
            int,
        ],
        None,
    ] = None,
) -> list:
    """
    Generate list of all objects with a path to or from obj.

    Parameters
    ----------
    G : nx.DiGraph
        acyclic directed graph.
    obj : node of the form (time:float, object_number:int)
        target node in G.
    ref_times_sel : Union[list, None], optional
        Only include objects these reference times. The default is None.
    overlap_thresh : Union[float, None], optional
        Only include objects with overlap >= object_thresh. The default is None.
    ntypes : Union[Tuple[int,], None], optional
        tuple of overlap types. The default is None which is interpreted as all.

    Returns
    -------
    list
        DESCRIPTION.

    """

    # p = nx.dfs_predecessors(G.to_undirected(), obj)
    if ntypes is None:
        ntypes = (1, 2)
    p = pred(G, obj)
    r = [obj]
    for dest, source in p.items():
        for sce in source:
            ntype = G.get_edge_data(sce, dest)["ntype"]
            if ntype not in ntypes:
                continue
            if overlap_thresh is None:
                r.append(sce)
                # print(sce)
            else:
                overlap = G.get_edge_data(sce, dest)["max_overlap"]
                if overlap is None or overlap >= overlap_thresh:
                    r.append(sce)
                    # print(sce)

    s = nx.dfs_successors(G, obj)
    # s = nx.descendants(G, obj)
    for source, dest in s.items():
        for d in dest:
            ntype = G.get_edge_data(source, d)["ntype"]
            if ntype not in ntypes:
                continue
            if overlap_thresh is None:
                r.append(d)
            else:
                overlap = G.get_edge_data(source, d)["max_overlap"]
                if overlap is None or overlap >= overlap_thresh:
                    r.append(d)
    r.sort()
    return r


def print_matching_objects_graph(G: nx.DiGraph):
    """
    Print all the starting nodes followed by nodes accessible from them.

    Parameters
    ----------
    G : nx.DiGraph
        acyclic directed graph.

    Returns
    -------
    None.

    """
    for n in G.nodes:
        if G.in_degree(n) == 0:
            o = f"{n}"
            for ns in G.successors(n):
                o += f" {ns}"
            print(o)
    return


def draw_object_graph(
    G: nx.DiGraph,
    nodelist: list = None,
    highlight_nodes: list = None,
    overlap_thresh: float = None,
    ntypes: Union[
        Tuple[
            int,
        ],
        None,
    ] = None,
    figsize: Union[Tuple[float, float], None] = None,
    save_file: Union[str, None] = None,
) -> plt.figure:
    """
    Draw connections between all or selected nodes.

    Parameters
    ----------
    G : nx.DiGraph
        Acyclic directed graph.
    nodelist : list, optional
        list of nodes in G. The default is None.
    highlight_nodes : list, optional
        list of nodes in nodelist to plot a different colour.
        The default is None.
    overlap_thresh : float, optional
        Only include objects with overlap >= object_thresh.
        The default is None.
    ntypes : Union[Tuple[int,],None], optional
        tuple of overlap types. The default is None which is interpreted as all.
    figsize : Union[Tuple[float, float], None], optional
        Figure size (inches) as per matplotlib. The default is None.
    save_file : Union[str, None], optional
        Name of file to save plot. The default is None.

    Returns
    -------
    Nothing

    """

    if highlight_nodes is None:
        highlight_nodes = []

    if nodelist is None:
        nodelist = list(G)

    if overlap_thresh is None:
        overlap_thresh = 0.0

    if ntypes is None:
        ntypes = (1, 2)

    # Generate layout for visualization
    pos = {}
    lab = {}
    times = []

    maxy = 0
    for n in G.nodes():
        (x, y) = n
        maxy = max(maxy, y)
        pos[n] = [x, y]
        if x not in times:
            times.append(x)

    times.sort()

    lab = {n: f"{n[1]}" for n in nodelist}

    if figsize is None:
        figsize = (0.6 * len(times), 0.2 * maxy)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plt.tight_layout()

    edgelist = [
        (u, v)
        for u, v in G.edges()
        if (u in nodelist and v in nodelist)
        if G.get_edge_data(u, v)["ntype"] in ntypes
        if "max_overlap" not in G.edges[u, v].keys()
        or G.get_edge_data(u, v)["max_overlap"] >= overlap_thresh
    ]

    edgewidth = [
        G.get_edge_data(u, v)["max_overlap"] * 5 + 1
        if "max_overlap" in G.edges[u, v].keys()
        else 1
        for u, v in edgelist
    ]

    edge_colors = [
        "m" if G.get_edge_data(u, v)["ntype"] == 1 else "b" for u, v in edgelist
    ]

    node_colors = ["green" if n in highlight_nodes else "darkblue" for n in nodelist]

    nx.draw_networkx_edges(
        G, pos, edgelist, ax=ax, alpha=0.3, width=edgewidth, edge_color=edge_colors
    )

    nx.draw_networkx_nodes(
        G, pos, nodelist=nodelist, ax=ax, alpha=0.9, node_color=node_colors
    )  # "#210070"

    nx.draw_networkx_labels(G, pos, lab, ax=ax, font_size=10, font_color="white")

    ax.set_xlabel("Reference time")
    ax.set_ylabel("Object Number")
    ax.set_xticks(times)

    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.set_title("Trajectory Family Graph")

    ax.set_ylim([-1, maxy + 1])
    ax.set_xlim([2 * times[0] - times[1], 2 * times[-1] - times[-2]])

    if save_file is not None:
        fig.savefig(save_file)

    plt.show()
    return
