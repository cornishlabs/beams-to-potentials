# %%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from scipy.optimize import curve_fit
from copy import deepcopy
import scipy.constants

import beamsToPotentialsLib as bpl

epsilon_0 = scipy.constants.epsilon_0
c = scipy.constants.c
a0 = scipy.constants.physical_constants['Bohr radius'][0]
k_B = scipy.constants.Boltzmann
h = scipy.constants.h
u = scipy.constants.u

tweezer_params_base = {
    (1066, "Tweezer"):[
        {
            'wx0':1.05,
            'wy0':1.16,
            'wz0':1.19,
            'power_mW':4.4,
            'theta':0,
            'x0':0,
            'y0':0,
            'z0':0,
            'phase':0,
        }
    ],
    (817, "SAL"): [
        {
            'wx0':100,
            'wy0':40,
            'wz0':60,
            'power_mW':200,
            'theta': np.pi/2 + 7.5*(2*np.pi/360),
            'x0':0,
            'y0':0,
            'z0':0,
            'phase':0,
        },
        {
            'wx0':100,
            'wy0':40,
            'wz0':60,
            'power_mW':200,
            'theta': np.pi/2 - 7.5*(2*np.pi/360),
            'x0':0,
            'y0':0,
            'z0':0,
            'phase':1.9,
        }
    ]
}

# %%

def trap_changes(xs_values, ch_func, plot=False):

    atoms = ["Rb", "Cs"]
    colours = ["Blue","Red"]

    trap_freqs = []
    minimum_positions = []
    pot_mins_mhz = []
    
    for ch_var in xs_values:
        tweezer_params = deepcopy(tweezer_params_base)
        ch_func(tweezer_params,ch_var)
        
        atoms_bare_trap_freqs = []
        atoms_bare_minimum_positions = []
        atoms_pot_min_MHz = []

        if plot:
            fig,axs = plt.subplots(2,3,height_ratios=(4,1),sharey='row',sharex='none')
        for ai, (atom,col) in enumerate(zip(atoms,colours)):
            mass = bpl.species[atom]['m']
            grad_dec_min = bpl.my_find_potential_minimum((0,0,0),tweezer_params,atom)
            this_expand_around = grad_dec_min
            pot_min_value_MHz = bpl.total_potential(grad_dec_min,tweezer_params,atom)

            bare_trap_freqs_khz = []
            bare_length_scales = []
            fit_min = []

            x = np.linspace(-4,4,200)
            y = np.linspace(-2,2,42)
            z = np.linspace(-15,15,400)

            for ci in range(3):
                
                # Set axis
                axis_var = [x,y,z][ci]
                xyzaxis = ["x","y","z"][ci]
                axis_slice = tuple((this_expand_around[d] if d!=ci else axis_var for d in range(3)))
                xs = axis_var
                
                # Work out potential along axis
                tp = bpl.total_potential(axis_slice, tweezer_params, atom)
                ys = tp

                if plot:
                    ax_pair = (axs.T)[ci]
                    ax, ax_low = ax_pair
                    line_main = ax.plot(xs,tp,c=col,label=f'{atom}')

                # Fit minimum
                fit_threshold_um = 0.5
                fit_y_ratio = 0.8
                trap_freq, quad_popt = bpl.my_get_trap_frequency(xs,ys, mass,
                                                                x_fit_threshold_um=fit_threshold_um,y_fit_threshold_ratio=fit_y_ratio,
                                                                x_bounds=(grad_dec_min[ci]-0.3,grad_dec_min[ci]+0.3))
                # print(f"{atom} {xyzaxis} Trap Frequency {trap_freq*1e3}  (kHz)")
                
                # Name other statistics
                centre_position_um = quad_popt[0]
                centre_intensity_min = quad_popt[2]
                sig_times_3 = 3*np.sqrt((scipy.constants.hbar)/(mass * u * 2 * np.pi * trap_freq * 1e-6))
                
                # Quadratic Fitted minimum
                xs_fit = np.linspace(centre_position_um-fit_threshold_um,centre_position_um+fit_threshold_um,200)
                ys_fit = bpl.quad(xs_fit, *quad_popt)
                min_y_fit = np.min(ys_fit)

                # Add to summary stats for this axis
                fit_min.append(centre_position_um)
                bare_length_scales.append(sig_times_3)
                bare_trap_freqs_khz.append(trap_freq*1e3)
                
                # Work out wf/prob distribution
                wf_space,dxwfs = np.linspace(centre_position_um-1.3,centre_position_um+1.3,200,retstep=True)
                wf = (mass*u*2*np.pi*trap_freq*1e6/(np.pi*scipy.constants.hbar))**(0.25) * np.exp(-(mass * u * 2 * np.pi * trap_freq * 1e6 * ((wf_space-centre_position_um)*1e-6)**2)/(2*scipy.constants.hbar)) #+ centre_intensity_min
                wf_prob_per_m = ((wf)**2)
                wf_prob_per_um = wf_prob_per_m*1e-6
                # print(np.sum(wf_prob_per_um*dxwfs)) #should be 1
                
                # Plot
                if plot:
                    # ax.axvline(rb_min[ci] if atom == 'Rb' else cs_min[ci], ls='--', alpha=0.4, color=col)
                    ax.axvline(centre_position_um, lw=1, linestyle='--', color=col)
                    ax.axvline(grad_dec_min[ci], lw=1, linestyle='-', color=col)
                    # line_fit = ax.plot(xs_fit, ys_fit, lw=1, linestyle='-', c='red')
                    ax_low.text(centre_position_um,6-(3 if atom=='Rb' else 0), f"TF {trap_freq*1e3:.1f}(kHz)",size='x-small',color=col,ha='right' if atom=='Cs' else 'left')
                    ax_low.plot(wf_space,wf_prob_per_um, c=col,alpha=0.4)
                    ax_low.fill_between(wf_space,wf_prob_per_um,y2=0,color=col, alpha=0.2)

            # Add to sumary state for Atom
            atoms_bare_trap_freqs.append(bare_trap_freqs_khz)
            atoms_bare_minimum_positions.append(fit_min)
            atoms_pot_min_MHz.append(pot_min_value_MHz)

            if plot:
                for ax in axs[1,:]:
                    ax.set_xlim(-1.2,1.2)
                axs[0,0].set_xlim(-4,4)
                axs[1,0].set_xlim(-4,4)
                axs[0,1].set_xlim(-2,2)
                axs[0,2].set_xlim(-10,10)
                axs[1,0].set_ylabel(r'$|\psi|$ (um$^{-1}$)')
                axs[0,0].legend()

                for ax_pair, axxname in zip(axs.T,['x','y','z']):
                    ax_pair[-1].set_xlabel(f'{axxname} (um)')

                axs[0,0].set_ylabel('Trap depth (MHz)')

        if plot:
            plt.show()
    ########

        minimum_positions.append(atoms_bare_minimum_positions)
        trap_freqs.append(atoms_bare_trap_freqs)
        pot_mins_mhz.append(atoms_pot_min_MHz)

    return (np.array(pot_mins_mhz),np.array(minimum_positions), np.array(trap_freqs))


# %% Get lattice-free trap freqs.
def ch_func_power(tweezer_params,power):
    tweezer_params[(817,"SAL")][0]['power_mW'] = power
    tweezer_params[(817,"SAL")][1]['power_mW'] = power
no_power = [0]
_,_, tfs = trap_changes(no_power, ch_func_power, plot=True)

bare_trap_freqs = [tfs[0,0],tfs[0,1]]

# %% 2D plot
atom='Cs'
fig, ax = plt.subplots(constrained_layout=True)

tweezer_params = deepcopy(tweezer_params_base)

x = np.linspace(-5,5,200)
z = np.linspace(-20,20,200)

x_mesh,y_mesh = np.meshgrid(x,z,indexing='ij')

pot_mesh = bpl.total_potential((x_mesh,0,y_mesh),tweezer_params,atom)
norm = mpl.colors.Normalize(vmin=-7, vmax=+3)

cfa = ax.contourf(x_mesh,y_mesh,pot_mesh,levels=1000, cmap='afmhot_r', norm=norm)
cb = fig.colorbar(cfa, ax=ax)
cb.set_label('Potential (MHz)')

ax.set_xlabel('$x_{\mathrm{Twe}}$ (um)')
ax.set_ylabel('$z_{\mathrm{Twe}}$ (um)')

# ellipse = mpl.patches.Ellipse(xy=(fit_mins[0][0],fit_mins[0][2]),
#                               width=bare_length_scales_rbcs[0], height=bare_length_scales_rbcs[2], 
#                         edgecolor='r', fc='None', lw=2)
# ax.add_patch(ellipse)

plt.show()


# %% ch power
def ch_func_power(tweezer_params,power):
    tweezer_params[(817,"SAL")][0]['power_mW'] = power
    tweezer_params[(817,"SAL")][1]['power_mW'] = power

pows = np.linspace(0,500,100)
pm_MHz, mps, tfs = trap_changes(pows,ch_func_power,plot=False)

fig,axs = plt.subplots(3,1,constrained_layout=True,sharex=True)
for ai, atom in enumerate(["Rb","Cs"]):
    col = ['blue','red'][ai]
    axs[0].plot(pows,-pm_MHz[:,ai],label=f"{atom}",color=col,ls='-')
    for axi in range(3):
        ls = ['-','--',':'][axi]
        axs[1].plot(pows,mps[:,ai,axi],label=f"{atom}, {['x','y','z'][axi]}",color=col,ls=ls)
        axs[2].plot(pows,tfs[:,ai,axi],label=f"{atom}, {['x','y','z'][axi]}",color=col,ls=ls)

axs[0].set_ylabel('Trap Depth (MHz)')
axs[1].set_ylabel('Positions (um)')
axs[2].set_ylabel('Trap Frequency (KHz)')
axs[1].legend(bbox_to_anchor=(0.85, 1.16),
          ncol=6, fancybox=True, shadow=False,fontsize='x-small')
axs[2].set_xlabel("SAL power per beam (mW)")
plt.show()

# %% ch phase
def ch_func_phase(tweezer_params,phase):
    tweezer_params[(817,"SAL")][1]['phase'] = phase

phase = np.linspace(-np.pi-0.01,np.pi+0.01,100)
pm_MHz, mps, tfs = trap_changes(phase,ch_func_phase,plot=False)

fig,axs = plt.subplots(3,1,constrained_layout=True,sharex=True)
for ai, atom in enumerate(["Rb","Cs"]):
    col = ['blue','red'][ai]
    axs[0].plot(phase,-pm_MHz[:,ai],label=f"{atom}",color=col,ls='-')
    for axi in range(3):
        ls = ['-','--',':'][axi]
        axs[1].plot(phase,mps[:,ai,axi],label=f"{atom}, {['x','y','z'][axi]}",color=col,ls=ls)
        axs[2].plot(phase,tfs[:,ai,axi],label=f"{atom}, {['x','y','z'][axi]}",color=col,ls=ls)

axs[0].set_ylabel('Trap Depth (MHz)')
axs[1].set_ylabel('Position (um)')
axs[2].set_ylabel('Trap Frequency (KHz)')
axs[1].legend(bbox_to_anchor=(0.85, 1.16),
          ncol=6, fancybox=True, shadow=False,fontsize='x-small')
axs[2].set_xlabel("Lattice Phase (rad)")
plt.show()