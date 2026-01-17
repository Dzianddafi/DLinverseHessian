import numpy as np
import tqdm
from scipy.signal import butter, sosfilt
from devito import *
from examples.seismic import Model, AcquisitionGeometry
from examples.seismic.acoustic import AcousticWaveSolver
from scipy.ndimage import gaussian_filter
from distributed import wait

from devito_ta.model_geom import grad_utils
from devito_ta.prepro import Filter, filter_data, CustomSource

def fm_single_shot(model, geom, save=False, dt=4.0):
    solver = AcousticWaveSolver(model, geom, space_order=4)
    d_obs, u0, _ = solver.forward(vp=model.vp, save=save, src=geom.src) #[0:2]
    return d_obs.resample(dt), u0

def fm_multi_shots(model, geometry, n_workers, client, save=False, dt=4.0):
    futures = []
    shot_data = []

    i = 0
    while i < len(geometry):
        
        # Submit up to n_workers jobs at a time
        batch = 0
        while batch < n_workers and i < len(geometry):
            geometry_i = AcquisitionGeometry(
                model,
                rec_positions=geometry[i].rec_positions,
                src_positions=geometry[i].src_positions,
                t0=geometry[i].t0, tn=geometry[i].tn, f0=geometry[i].f0,
                src_type=geometry[i].src_type
            )
            futures.append(client.submit(fm_single_shot, model, geometry_i, save=save, dt=dt))
            i += 1
            batch += 1

        # Wait for the current batch
        wait(futures)

        # Gather results
        results = client.gather(futures)
        shot_data.extend(results)

        # Clear futures list for next batch
        futures = []

    return shot_data

def compute_residual(residual, d_obs, d_syn, geom):
    # Compute residual
    residual.data[:] = d_syn.data[:] - d_obs.resample(geom.dt).data[:][0:d_syn.data.shape[0], :]
    #residual.data[:] = d_syn.data[:] - d_obs.data[:][0:d_syn.data.shape[0], :]

    return residual

def grad_x_shot(model_true, model_init, geom, d_obs):
    
    # Get grad function and receivers
    grad_i, residual_i, _, _ = grad_utils(model_true, geom)

    # Initiate wave solver
    solver = AcousticWaveSolver(model_true, geom, space_order=4)
    d_syn, u_syn = solver.forward(vp=model_init.vp, save=True)[0:2]

    # Resample d_syn and residual_i
    residual_i = residual_i.resample(geom.dt)
    d_syn = d_syn.resample(geom.dt)

    # Get residual score
    residual = compute_residual(residual_i, d_obs, d_syn, geom)

    # Calculate objective value and gradient
    obj = 0.5 * norm(residual)**2
    solver.gradient(rec=residual, u=u_syn, vp=model_init.vp, grad=grad_i)

    # Remove nbl
    grad_i_crop = np.array(grad_i.data[:])[model_init.nbl:-model_init.nbl, model_init.nbl:-model_init.nbl]

    return obj, grad_i_crop

def normalize(data, vp):

    data /= (vp**3)
    
    return data
    
def grad_multi_shots(model_true, model_init, geoms, n_workers, client, d_true):
    nbl = model_init.nbl
    futures = []

    # Initial objective value and gradient
    obj_total = 0.0
    grad_total = np.zeros(model_init.vp.data[nbl:-nbl, nbl:-nbl].shape, dtype=np.float64)

    s = 0
    while s < len(geoms):
        # Submit up to n_workers jobs at a time
        batch = 0
        
        while batch < n_workers and s < len(geoms):
            geom_s = AcquisitionGeometry(
                model_true,
                rec_positions=geoms[s].rec_positions,
                src_positions=geoms[s].src_positions,
                t0=geoms[s].t0, tn=geoms[s].tn, f0=geoms[s].f0,
                src_type=geoms[s].src_type
            )
            
            futures.append(client.submit(grad_x_shot, 
                                         model_true, model_init, 
                                         geom_s, d_true[s]))
            
            
            
            s += 1
            batch += 1

        # Tunggu semua future batch selesai
        wait(futures)

        # Ambil hasil
        results = client.gather(futures)

        # Jumlahkan hasil objective dan grad
        for res in results:
            obj_total += res[0]
            grad_total += res[1]

        # Reset futures untuk batch berikutnya
        futures = []

    # Cut water layer
    #grad_total[:, :50] = 0
    grad_total = -grad_total     # deavtivate if using scipy optimizer

    # Normalize and blur gradient
    #grad_total = normalize(grad_total, model_init.vp.data[nbl:-nbl, nbl:-nbl])
    
    return obj_total, grad_total

# Born approximation
def born_x_shot(model_true, model_background, model_scat, geom):
    
    # Get grad function and receivers
    born_i = grad_utils(model_background, geom)[0]
    nbl = model_background.nbl
    
    # Demigrated operator
    solver = AcousticWaveSolver(model_true, geom, space_order=4)    
    _, u_fm = solver.forward(vp=model_background.vp, save=True)[:2]
    
    u = TimeFunction(name='u', grid=model_background.grid, time_order=2, space_order=4)
    
    d_born, u, U = solver.jacobian(dmin=model_scat, u=u, vp=model_background.vp)[:3]
    d_born = d_born.resample(geom.dt)
    
    # Migrated operator
    solver.jacobian_adjoint(rec=d_born, u=u_fm, vp=model_background.vp, grad=born_i)
    
    # Remove nbl
    born_i_crop = np.array(born_i.data[:])[nbl:-nbl, nbl:-nbl]
    
    return born_i_crop

def born_multi_shots(model_true, model_background, model_scat, geoms, n_workers, client):
    nbl = model_background.nbl
    futures = []

    # Initial born, forward wavefield, ad linear wavefield
    born_total = np.zeros(model_true.vp.data[nbl:-nbl, nbl:-nbl].shape, dtype=np.float64)
    
    s = 0
    while s < len(geoms):
        # Submit up to n_workers jobs at a time
        batch = 0
        
        while batch < n_workers and s < len(geoms):
            geom_s = AcquisitionGeometry(
                model_true,
                rec_positions=geoms[s].rec_positions,
                src_positions=geoms[s].src_positions,
                t0=geoms[s].t0, tn=geoms[s].tn, f0=geoms[s].f0,
                src_type=geoms[s].src_type
            )
            
            futures.append(client.submit(born_x_shot, model_true,
                                         model_background, 
                                         model_scat, geom_s))
            
            s += 1
            batch += 1

        # Tunggu semua future batch selesai
        wait(futures)

        # Ambil hasil
        results = client.gather(futures)

        # Jumlahkan hasil objective dan grad
        for res in results:
            born_total += res

        # Reset futures untuk batch berikutnya
        futures = []

    model_scat = model_scat.data[nbl:-nbl, nbl:-nbl]
    model_scat = normalize(model_scat, model_background.vp.data[nbl:-nbl, nbl:-nbl])
    
    
    return model_scat, born_total