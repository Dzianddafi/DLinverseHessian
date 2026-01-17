import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_shotrecord_cust(rec1, title1, 
                         rec2, title2,
                         rec3, title3,
                         model, t0, tn, 
                         save=False, colorbar=True):
    """
    Plot a shot record (receiver values over time).

    Parameters
    ----------
    rec :
        Receiver data with shape (time, points).
    model : Model
        object that holds the velocity model.
    t0 : int
        Start of time dimension to plot.
    tn : int
        End of time dimension to plot.
    """
    
    extent = [model.origin[0], rec1.shape[1], tn/1000, t0]
    #extent2 = [0, 17, 5, 0]
    title = [title1, title2, title3]
    data = [rec1, rec2, rec3]

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 8))

    for ax, title, data in zip(ax, title, data):
        cax = ax.imshow(
        data,
        cmap='seismic',
        vmin=-2, vmax=2,
        aspect='auto',
        extent=extent 
    )
        
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xlabel('Trace Number')
        ax.invert_xaxis()
        ax.set_ylabel('Time (s)')

    fig.subplots_adjust(right=0.88, wspace=0.3)

    # Create aligned colorbar on the right
    if colorbar:
        cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        fig.colorbar(cax, cax=cbar_ax)
    
    if save:
        plt.savefig('pics/shotrecord.png', dpi=300, bbox_inches='tight')

    plt.show()

def plot_velocity_cust(model, source=None, receiver=None, colorbar=True, cmap="jet", save=False, name='velocity', title='model'):
    """
    Plot a two-dimensional velocity field from a seismic `Model`
    object. Optionally also includes point markers for sources and receivers.

    Parameters
    ----------
    model : Model
        Object that holds the velocity model.
    source : array_like or float
        Coordinates of the source point.
    receiver : array_like or float
        Coordinates of the receiver points.
    colorbar : bool
        Option to plot the colorbar.
    """
    domain_size = 1.e-3 * np.array(model.domain_size)
    extent = [model.origin[0], model.origin[0] + domain_size[0],
              model.origin[1] + domain_size[1], model.origin[1]]

    slices = tuple(slice(model.nbl, -model.nbl) for _ in range(2))
    if getattr(model, 'vp', None) is not None:
        field = model.vp.data[slices]
    else:
        field = model.lam.data[slices]
    plot = plt.imshow(np.transpose(field), animated=True, cmap=cmap,
                      vmin=1.5, vmax=4.7,
                      extent=extent)
    plt.title(f'{title}', fontsize=14, pad=10)
    plt.xlabel('X position (km)')
    plt.ylabel('Depth (km)')

    # Plot source points, if provided
    if receiver is not None:
        plt.scatter(1e-3*receiver[:, 0], 1e-3*receiver[:, 1],
                    s=25, c='w', marker='D')

    # Plot receiver points, if provided
    if source is not None:
        plt.scatter(1e-3*source[:, 0], 1e-3*source[:, 1],
                    s=25, c='red', marker='o')

    # Ensure axis limits
    plt.xlim(model.origin[0], model.origin[0] + domain_size[0])
    plt.ylim(model.origin[1] + domain_size[1], model.origin[1])

    # Create aligned colorbar on the right
    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(plot, cax=cax)
        cbar.set_label('Velocity (km/s)')

    if save:
        plt.savefig(f'pics/{name}.png', dpi=300, bbox_inches='tight')

    plt.show()

def plot_acquisition_geometry(geometries, title: str):
    """
    Plot a survey design for a list of AcquisitionGeometry objects.
    
    Parameters:
    - geometries: list of AcquisitionGeometry objects
    - title: plot title
    """
    nshots = len(geometries)

    plt.figure(figsize=(20, 15))
    plt.title(title)

    for i, geom in enumerate(geometries):
        rec = geom.rec_positions[:, 0]
        src = geom.src_positions[0, 0]

        plt.scatter(rec, np.full_like(rec, i+1), marker='o', c='r', label='Receiver [200]' if i == 0 else "")
        plt.scatter(src, i+1, s=500, marker='*', c='b', label='Source [1]' if i == 0 else "")

    plt.xlabel('X Position (m)')
    plt.ylabel('Shot Number')
    plt.ylim(0, nshots + 10)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_freq(wavpad, wavelet, old_wave, geom, freq_lim, title_right=None, title_left=None):
    wav = wavelet

    f = np.fft.rfftfreq(wavpad, geom.dt)
    fwave = np.fft.rfft(wav[:wavpad], wavpad)
    old_fwave = np.fft.rfft(old_wave[:wavpad], wavpad)
 
    
    # Extract wavelet and pad it to allow filtering
    #wav = wav[:2*np.argmax(wav)+1]
    #wav = np.pad(wav, (wavpad, geom.nt - wavpad - wav.size))
    

    
    # Show source function

    extent = [0, freq_lim*1000,np.abs(fwave).min(),np.abs(fwave).max()]
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    ax[0].plot(wav[:], 'r')
    ax[0].set_title(title_left)
    ax[0].set_xlabel('Time (ms)')
    ax[0].set_ylabel('Amplitude')
    
    # Show freq distribution
    ax[1].plot(f, np.abs(fwave), 'r', label='Filtered Freq')
    ax[1].set_xlim(0, freq_lim)
    ax[1].set_title(title_right)
    ax[1].set_xlabel('Frequency (KHz)')
    ax[1].set_ylabel('Amplitude')
    
    # Show old freq distribution
    ax[0].plot(old_wave[:], 'k')
    ax[1].plot(f, np.abs(old_fwave), 'k', label='Old Freq')

    plt.legend()
    plt.show()