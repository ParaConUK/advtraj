# Notes on this branch.

##  Introduction

This branch contains the following, related, subpackages:
- `plot`
- `classify`
- `family`

The first of these provides some matplotlib-based (i.e. slow) functionality for 3D plotting of trajectories.
We shall use it to illustrate the other two.

## The plot subpackage

The module `plot.plot_trajectory_animation.py` contains two main functions.
The function `plot_traj_animation` is used to plot trajectories from one reference time.
At its most basic, it animates all of the objects in the input xarray (identified by `object_label`).
Here is an example from a MONC simulation of BOMEX, showing each object as a different colour:
![a MONC simulation of BOMEX](animations/Traj_plot_all.gif)
Optionally, a galilean transform can be applied to move the plot with the mean wind:
![move the plot with the mean wind](animations/Traj_plot_all_gal.gif)

Individual objects can be selected, and just a sub-domain plotted.
If the trajectory dataset contains the variable `obj_mask` then points with `obj_mask==True` can be distinguished
with different sized dots.
For example, here the mask was set up using `q_cloud_liquid_mass > 1E-5`.
Here is object 9:
![object 9](animations/Traj_plot_mask.gif)

Sometimes it is useful to compare the Lagrangian view, trajectories with origin at a given time, in cloud, say)
with the Eulerian view, grid points in cloud at different times.
This can be done by supplying a `field_mask` derived from the original gridded data.
For example, here is the same figure as above with the Eulerian cloud field as black dots.
![Eulerian field](animations/Traj_plot_field.gif)

## The classify subpackage

The classify/traj_classify module can be used to classify points along a trajectory using a combination of masks.
For example, the same mask as used above may identify points with or without cloud water (or, strictly, exceeding a non-zero threshold).
The function `classify.traj_classify.set_traj_class` provides the basic functionality.
It recognises that trajectories can go in and out of, say, cloud, multiple times.
Thus, this function starts at the trajectory reference time and moves both back in forward, identifying points that are 'in' or 'not in' the mask.
It also counts how many times the trajectory has been 'in' the mask.
So points at the reference time 'in' the mask are labelled '(In,0)' and those not in '(Pre,0)'.
Stepping backwards in time, the code looks for changes in status (i.e. 'in-mask' to 'not in-mask', and *vice versa*) and changes the label appropriately, so if a trajectory starts 'in-mask', going backwards in time it may change to `(Pre,0)`, `(In, -1)`, `(Pre, -1)`, `(In, -2)`, `(Pre, -2)` etc., so that 'in-mask' events before the one at the reference time are labelled with negative numbers.
The same process applies going forwards, so status changes to `(Post, 0)`, `(In, 1)`, `(Post, 1)` etc..
These are coded in integer data matching the trajectory data, and returned in a dictionary with item 'class'; the same dictionary provides a mapping of these integers to the tuples above in item 'key'.
Note that this system works most clearly where all the points at the reference time are 'in-mask'.
This is illustrated for the same object as above:
![Cloud classification](animations/Traj_plot_cloud_class.gif)

Masks can easily be combined using `classify.traj_classify.combine_traj_classes`; currently this only implements
a logical and between masks, so that the labels can simply be added as strings.
An example is given in examples cloud_classify.py in which the 'in-cloud' mask above is combined with 'in-BL', i.e. in the boundary-layer, defined as height less than cloud-base, evaluated as the minimum height of 'in-cloud' trajectory points.
This module also implements a simple function to re-label the combined mask, leading to a classification illustrated below, where, before entering cloud air is classified as pre-entrainment, either from the BL or not, and, likewise, after leaving cloud, air is Post-detrainment, and the possibility of multiple cloud encounters is retained.
![Cloud and BL classification](animations/Traj_plot_classes.gif)

The multiple cloud encounters can be put into context by including the Eulerian cloud field:
 ![Cloud and BL classification with Eulerian field](animations/Traj_plot_classes_field.gif).

 ## The family subpackage

 So far the trajectory code has dealt with forward and back trajectories started at a given reference time.
 Of course, and data time in the source data can be used as a reference time, and we define the set of trajectories from a contiuous range of reference times as a **family** of trajectories.
 If these start from selected objects, there is likely to be a different number of trajectories (and objects) at each reference time, which makes it difficult to combine the familt into one dataset.
 At present, we have adopted the simple solution of generating a separate NetCDF file for each reference time, using an numerical sequence number to the file name.

 The task of generating a family is met by `family.traj_family.traj_family`.
 This takes an xarray Dataset.
 This is as per the trajectory integration functions and can be generated using `xarray.open_mfdataset` to concatenate multiple files.
 See 'examples/load_data.py' for an example.

 Given a list of paths it is not obvious what data are available.
 A trajectory family can be analysed using `family.traj_family.analyse_traj_family`.
 The output from this can be summarised by passing it to `family.traj_family.summarise_traj_family`; an example output is:

    21060.0: ***********************************xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    21120.0: R***********************************xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    21180.0: *R***********************************xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    21240.0: **R***********************************xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    21300.0: ***R***********************************xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    21360.0: ****R***********************************xxxxxxxxxxxxxxxxxxxxxxxxxxxx
    21420.0: *****R***********************************xxxxxxxxxxxxxxxxxxxxxxxxxxx
    21480.0: ******R***********************************xxxxxxxxxxxxxxxxxxxxxxxxxx
    21540.0: *******R***********************************xxxxxxxxxxxxxxxxxxxxxxxxx
    21600.0: ********R***********************************xxxxxxxxxxxxxxxxxxxxxxxx
    21660.0: *********R***********************************xxxxxxxxxxxxxxxxxxxxxxx
    21720.0: **********R***********************************xxxxxxxxxxxxxxxxxxxxxx
    21780.0: ***********R***********************************xxxxxxxxxxxxxxxxxxxxx
    21840.0: ************R***********************************xxxxxxxxxxxxxxxxxxxx
    21900.0: *************R***********************************xxxxxxxxxxxxxxxxxxx
    21960.0: **************R***********************************xxxxxxxxxxxxxxxxxx
    22020.0: ***************R***********************************xxxxxxxxxxxxxxxxx
    22080.0: ****************R***********************************xxxxxxxxxxxxxxxx
    22140.0: *****************R***********************************xxxxxxxxxxxxxxx
    22200.0: ******************R***********************************xxxxxxxxxxxxxx
    22260.0: *******************R***********************************xxxxxxxxxxxxx
    22320.0: ********************R***********************************xxxxxxxxxxxx
    22380.0: *********************R***********************************xxxxxxxxxxx
    22440.0: **********************R***********************************xxxxxxxxxx
    22500.0: ***********************R***********************************xxxxxxxxx
    22560.0: ************************R***********************************xxxxxxxx
    22620.0: *************************R***********************************xxxxxxx
    22680.0: **************************R***********************************xxxxxx
    22740.0: ***************************R***********************************xxxxx
    22800.0: ****************************R***********************************xxxx
    22860.0: *****************************R***********************************xxx
    22920.0: ******************************R***********************************xx
    22980.0: *******************************R***********************************x
    23040.0: ********************************R***********************************
    23100.0: *********************************R**********************************
    23160.0: **********************************R*********************************
    23220.0: ***********************************R********************************
    23280.0: x***********************************R*******************************
    23340.0: xx***********************************R******************************
    23400.0: xxx***********************************R*****************************
    23460.0: xxxx***********************************R****************************
    23520.0: xxxxx***********************************R***************************
    23580.0: xxxxxx***********************************R**************************
    23640.0: xxxxxxx***********************************R*************************
    23700.0: xxxxxxxx***********************************R************************
    23760.0: xxxxxxxxx***********************************R***********************
    23820.0: xxxxxxxxxx***********************************R**********************
    23880.0: xxxxxxxxxxx***********************************R*********************
    23940.0: xxxxxxxxxxxx***********************************R********************
    24000.0: xxxxxxxxxxxxx***********************************R*******************
    24060.0: xxxxxxxxxxxxxx***********************************R******************
    24120.0: xxxxxxxxxxxxxxx***********************************R*****************
    24180.0: xxxxxxxxxxxxxxxx***********************************R****************
    24240.0: xxxxxxxxxxxxxxxxx***********************************R***************
    24300.0: xxxxxxxxxxxxxxxxxx***********************************R**************
    24360.0: xxxxxxxxxxxxxxxxxxx***********************************R*************
    24420.0: xxxxxxxxxxxxxxxxxxxx***********************************R************
    24480.0: xxxxxxxxxxxxxxxxxxxxx***********************************R***********
    24540.0: xxxxxxxxxxxxxxxxxxxxxx***********************************R**********
    24600.0: xxxxxxxxxxxxxxxxxxxxxxx***********************************R*********
    24660.0: xxxxxxxxxxxxxxxxxxxxxxxx***********************************R********
    24720.0: xxxxxxxxxxxxxxxxxxxxxxxxx***********************************R*******
    24780.0: xxxxxxxxxxxxxxxxxxxxxxxxxx***********************************R******
    24840.0: xxxxxxxxxxxxxxxxxxxxxxxxxxx***********************************R*****
    24900.0: xxxxxxxxxxxxxxxxxxxxxxxxxxxx***********************************R****
    24960.0: xxxxxxxxxxxxxxxxxxxxxxxxxxxxx***********************************R***
    25020.0: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx***********************************R**
    25080.0: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx***********************************R*
    25140.0: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx***********************************R
    25200.0: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx***********************************

Each *column* represents a file containing one member of the family with reference time
at 'R'.
The symbol '\*' denotes data present, while 'x' means data absent. The above was generated
using a fixed number of requested forward and backward points in each
trajectory, but the actual output was limited by data availability in the source data.

Once generated, the source varables can also be interpolated onto trajectories using `family.traj_family.data_to_traj_family`.

## Object matching

Some objects in different family members are likely to be related - e.g. part of the same cloud.
Functionality has been added to [cohobj](https://github.com/ParaConUK/cohobj) and `traj_family` to
find overlapping objects.

The function `family.traj_family.find_family_matching_objects` will find objects at different
reference times that overlap, in some way, the requested objects in a 'master' reference time.
A 'fast' method is provided that just uses bounding boxes for the 'in-mask' parts of an object.
This is illustrated in the following:
![Object bounding box](animations/Traj_plot_box.gif).
A slower method first screens using these bounding boxes (i.e. only considers objects with overlapping bounding boxes), then estimates the fractional overlap by mapping 'in-mask' points to nearest grid points and finding the common grid points.
The resulting dictionary can be printed using `family.traj_family.print_matching_objects`.
For example, for object 9 in master reference time 23160.0 above, the matching objects at reference time
22560.0 are printed below:

    Master Reference File Index: 34
    Master Reference Time: 23160.0
      Object 9
      Reference time: 22560.0
        Time: 21060.0 matching obj: 0: 0.00 3: 0.03
        Time: 21120.0 matching obj: 3: 0.00
        Time: 21180.0 matching obj: 0: 0.00 3: 0.02
        Time: 21240.0 matching obj: 3: 0.05
        Time: 21300.0 matching obj: 3: 0.05
        Time: 21360.0 matching obj: 0: 0.00 3: 0.06
        Time: 21420.0 matching obj: 3: 0.04
        Time: 21480.0 matching obj: 3: 0.04
        Time: 21540.0 matching obj: 3: 0.11
        Time: 21600.0 matching obj: 3: 0.13
        Time: 21660.0 matching obj: 0: 0.00 3: 0.08
        Time: 21720.0 matching obj: 3: 0.25
        Time: 21780.0 matching obj: 0: 0.00
        Time: 21840.0 matching obj: 0: 0.04 3: 0.17
        Time: 21900.0 matching obj: 0: 0.07 3: 0.16
        Time: 21960.0 matching obj: 0: 0.09 3: 0.09
        Time: 22020.0 matching obj: 0: 0.13 3: 0.10
        Time: 22080.0 matching obj: 0: 0.13 3: 0.11
        Time: 22140.0 matching obj: 0: 0.13 3: 0.09
        Time: 22200.0 matching obj: 0: 0.13 3: 0.08
        Time: 22260.0 matching obj: 0: 0.16 3: 0.07
        Time: 22320.0 matching obj: 0: 0.20 3: 0.07
        Time: 22380.0 matching obj: 0: 0.23 3: 0.06
        Time: 22440.0 matching obj: 0: 0.20 3: 0.06
        Time: 22500.0 matching obj: 0: 0.20 3: 0.05
        Time: 22560.0 matching obj: 0: 0.19 3: 0.04
        Time: 22620.0 matching obj: 0: 0.22 3: 0.03
        Time: 22680.0 matching obj: 0: 0.20 3: 0.02
        Time: 22740.0 matching obj: 0: 0.22 3: 0.02
        Time: 22800.0 matching obj: 0: 0.22 3: 0.03
        Time: 22860.0 matching obj: 0: 0.22 3: 0.02
        Time: 22920.0 matching obj: 0: 0.18 3: 0.02
        Time: 22980.0 matching obj: 0: 0.19 3: 0.04
        Time: 23040.0 matching obj: 0: 0.17 3: 0.04
        Time: 23100.0 matching obj: 0: 0.17 3: 0.04
        Time: 23160.0 matching obj: 0: 0.16 3: 0.04
        Time: 23220.0 matching obj: 0: 0.14 3: 0.04
        Time: 23280.0 matching obj: 0: 0.15 3: 0.02
        Time: 23340.0 matching obj: 0: 0.15 3: 0.03
        Time: 23400.0 matching obj: 0: 0.13
        Time: 23460.0 matching obj: 0: 0.15
        Time: 23520.0 matching obj: 0: 0.22
        Time: 23580.0 matching obj: 0: 0.22 3: 0.04
        Time: 24660.0 matching obj: 3: 0.00 6: 0.00
            All times matching obj: 0: 0.23 3: 0.25 6: 0.00

Note that some objects have zero overlap - this is because the 'fast' bounding-box overlap only considers the horizontal dimensions, as it is often of interest if one object lies beneath another.

It is possible to find all of the objects in all reference times that overlap objects in the master reference time.
This is illustrated for the same object 9 as above using `plot.plot_trajectory_animation.plot_family_animation`.
![All objects overlapping](animations/Family_plot_nomask_all.gif).

This is quite unwieldy.
If we just select two reference times, for example, we obtain
![Two reference time objects overlapping](animations/Family_plot_nomask.gif).
In this case, one reference time chosen is just 3 min (180 s) earlier than the master reference time, and the object overlapping (blue) is clearly much the same as the master (black); choosing 10 min earlier results in two overlapping objects (0 (red) and 3 (green)), and one (green) clearly represents an earlier 'cell' at the same location.
This is a little clearer if we include the 'in-cloud' mask:
![Two reference time objects overlapping with mask](animations/Family_plot.gif).
The bounding boxes for the 10 minute time gap is shown here:
![Bounding boxes overlapping](animations/Family_plot_boxes.gif).
And, finally, we include the Eulerian cloud field:
![Two reference time objects overlapping with field](animations/Family_plot_field.gif).

## Example code
Use of these packages clearly depends upon the source data.
However, to illustrate their use, we have included example scripts in the examples directory.
These can be summarised as follows:
- **monc_test_traj_compute.py**: Script for producing trajectories from MONC LES model output.
- **monc_test_data_to_traj.py**: Script to interpolate gridded data to trajectories.
- **monc_test_traj_classify.py**: Script to test trajectory classification.
- **monc_plot_traj_anim.py**: Plot trajectory animations.
- **monc_test_traj_family_compute.py**: Script to compute trajectory family from MONC data.
- **monc_test_data_to_traj_family.py**: Script to interpolate gridded data to trajectory family.
- **monc_test_traj_family_matching_objects.py**: Script to compute matching objects and plot family animations.