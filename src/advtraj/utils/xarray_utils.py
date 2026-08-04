# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 12:45:41 2026

@author: xm904103
"""


def get_dimorder(da, req_order="xyz"):
    dimlist = []
    dimorder = []
    for dim in req_order:
        for i, d in enumerate(da.dims):
            if dim in d:
                dimlist.append(d)
                dimorder.append(i)
                break
    return dimlist, dimorder


def da_force_dim_order(da, req_order="xyz"):

    dimlist, dimorder = get_dimorder(da, req_order=req_order)

    if len(dimlist) != len(da.dims):
        raise ValueError("Cannot re-order field with dimensions {da.dims}.")
    else:
        da = da.transpose(*dimlist)

    return da


def ds_force_dim_order(ds, req_order="xyz"):
    for da in ds.variables:
        ds[da] = da_force_dim_order(ds[da], req_order=req_order)

    return ds
