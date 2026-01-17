import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from examples.seismic import Model, AcquisitionGeometry

from devito import *
from examples.seismic import AcquisitionGeometry
from examples.seismic import Receiver


def get_model(vp, shape, spacing, origin, nbl=70, save=False, title='model'):

    model = Model(
        vp=vp,
        origin=origin,
        spacing=spacing,
        shape=shape,
        space_order=4,
        nbl=nbl,
        bcs='damp'
    )

    return model


def get_geometry(model,
                    src_x,
                    src_z,
                    rec_x,
                    rec_z,
                    t0: float,
                    tn: float,
                    nof: float,
                    src_type: str,
                    f0: float = 20.0,
                    dt: float = None):

    nsrc, nrec = len(src_x), len(rec_x)

    # Buat objek geometry per shot
    geometries = []

    src_coordinates = np.empty((nsrc, 2))
    src_coordinates[:, 0] = src_x
    src_coordinates[:, 1] = src_z

    for i in range(nsrc):

        rec_coordinates = np.empty((nrec, 2))
        rec_coordinates[:, 0] = rec_x
        rec_coordinates[:, 1] = rec_z 
        rec_coordinates[:, 0] = src_coordinates[i, 0] - rec_x - nof
        
        geometry = AcquisitionGeometry(
            model,
            rec_positions=rec_coordinates[:, :],
            src_positions=src_coordinates[i, :],
            t0=t0, tn=tn, f0=f0,
            src_type=src_type
        )
        geometries.append(geometry)

    return geometries

def grad_utils(model_true, geom):

    # Define grad function
    grad = Function(name='grad', grid=model_true.grid)

    # Define placeholders
    residual = Receiver(name='residual', grid=model_true.grid,
                        time_range=geom.time_axis, coordinates=geom.rec_positions)
    d_obs = Receiver(name='d_obs', grid=model_true.grid,
                        time_range=geom.time_axis, coordinates=geom.rec_positions)
    d_syn = Receiver(name='d_syn', grid=model_true.grid,
                        time_range=geom.time_axis, coordinates=geom.rec_positions)
    
    return grad, residual, d_obs, d_syn