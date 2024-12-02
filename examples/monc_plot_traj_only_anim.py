# -*- coding: utf-8 -*-
"""
monc_plot_traj_anim.py
Plot trajectory animations.
"""
from pathlib import Path

import xarray as xr
from load_data import load_data

from advtraj.plot.plot_trajectory_animation import plot_traj_animation

case = "cloud"

minim = "fixed_point_iterator"

interp_order = 5

expt = "std"

# root_path = "C:/Users/paclk/OneDrive - University of Reading/ug_project_data/Data/"
root_path = "F:/Data/ug_project_data/"


data_path = root_path + ""
odir = root_path + "trajectories/"

file_prefix = "diagnostics_3d_ts_"
ref_file = "diagnostics_ts_"

selector = "*"


files = list(Path(data_path).glob(f"{file_prefix}*.nc"))

ref_files = list(Path(data_path).glob(f"{ref_file}*.nc"))

if len(ref_files) == 0:
    ref_dataset = None
else:
    ref_dataset = xr.open_dataset(ref_files[0])

output_path = (
    odir + f"{file_prefix}trajectories_{case}_{interp_order}_{minim}_{expt}.nc"
)
output_path_data = odir + f"{file_prefix}data{case}_{interp_order}_{minim}_{expt}.nc"

if case == "w":

    ds_field = load_data(
        files=files, ref_dataset=ref_dataset, traj=False, fields_to_keep=["w"]
    )

    ds_subset = ds_field.isel(time=int(ds_field.time.count()) // 2).sel(
        z=slice(300, None)
    )

    # we'll use as starting points for the trajectories all points where
    # the vertical velocity is 80% of the maximum value
    w_max = ds_subset.w.max()

    mask_field = ds_field.w > 0.9 * w_max
    mask_field.name = "obj_mask"


elif case == "cloud":

    ds_field = load_data(
        files=files,
        ref_dataset=ref_dataset,
        traj=False,
        fields_to_keep=["q_cloud_liquid_mass"],
    )

    # we'll use as starting points for the trajectories all points where
    # the cloud liquid water mass is > 1E-5.
    thresh = 1e-5

    mask_field = ds_field.q_cloud_liquid_mass > thresh
    mask_field.name = "obj_mask"


ds_traj = xr.open_dataset(output_path)
print(ds_traj.ref_time)
# print(ds_traj)


anim = plot_traj_animation(
    ds_traj,
    # plot_mask=True,
    # class_no = traj_class_cloud,
    # class_no=traj_class_cloud_bl,
    # select=[9],
    # select=[9, 20, 23],
    # field_mask = mask_field,
    galilean=(-8, -1.5),
    plot_mask=False,
    # view_point=(30,30),
    anim_name="Traj_plot_cloud_class.gif",
    load_ds=True,
    figsize=(15, 12),
    legend=True,
    # with_boxes=True,
    # field_size = 4.0,
    # x_lim=[4400, 6400],
    # y_lim=[3000, 5000],
    title="Selected objects",
    # colors=colors,
)
