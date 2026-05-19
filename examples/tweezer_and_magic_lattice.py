# %%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
# from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.optimize import curve_fit, fmin, fmin_tnc
from copy import deepcopy
import scipy.constants

from scipy.interpolate import interp1d
from math import floor, log10

import beams_to_potentials as bpl

epsilon_0 = scipy.constants.epsilon_0
c = scipy.constants.c
a0 = scipy.constants.physical_constants['Bohr radius'][0]
k_B = scipy.constants.Boltzmann
h = scipy.constants.h
u = scipy.constants.u

p_imp=7
tweezer_params_base_rbcs = {
    (1145,"beams"): [
        {
            'wx0':100,
            'wy0':40,
            'wz0':60,
            'power_mW':25*p_imp,
            'theta': np.pi/2,
            'x0':0,
            'y0':0,
            'z0':0,
            'phase':0,
        },
        {
            'wx0':100,
            'wy0':40,
            'wz0':60,
            'power_mW':25*p_imp,
            'theta': np.pi/2 + np.pi,
            'x0':0,
            'y0':0,
            'z0':0,
            'phase':0,
        },
        {
            'wx0':1.7,
            'wy0':1.7,
            'wz0':1.7,
            'power_mW':1,
            'theta':0,
            'x0':0.143,
            'y0':0,
            'z0':6,
            'phase':-1.9,
        },
        {
            'wx0':1.7,
            'wy0':1.7,
            'wz0':1.7,
            'power_mW':1,
            'theta':0,
            'x0':-0.143,
            'y0':0,
            'z0':-6,
            'phase':0,#np.pi,
        },
    ]
}


# %% Get lattice-free trap freqs.
fig,axs = plt.subplots(2,3,height_ratios=(4,1),sharey='row',sharex='none')

tweezer_params_rbcs = deepcopy(tweezer_params_base_rbcs)

tweezer_params_rbcs[(1145,"beams")][0]['power_mW']=0
tweezer_params_rbcs[(1145,"beams")][1]['power_mW']=0

# start_positions = [(-2,0,0),(2,0,0)]

rbcs_min = bpl.my_find_potential_minimum((-0.143,0,-5.5),tweezer_params_rbcs,"RbCs")
bare_trap_freqs_khz_rbcs = []
bare_length_scales_rbcs = []

fit_it=True
fit_mins = [[0,0,0]] # Will change
atom = "RbCs"
mass = bpl.species[atom]['m']
col = 'red'
colour = 'blue'
tweezer_params = tweezer_params_rbcs
this_expand_around = rbcs_min#[-0.2,0,0]#rbcs_min

x = np.linspace(-4,4,200) #np.concatenate(([0],np.linspace(-0.01,-2,20),np.linspace(0.01,2,20)))
y = np.linspace(-2,2,42) #np.concatenate(([0],np.linspace(-0.01,-2,20),np.linspace(0.01,2,20)))
z = np.linspace(-15,15,400)

for ci,ax_pair in enumerate(axs.T):
    ax, ax_low = ax_pair
    
    axis_var = [x,y,z][ci]
    axis_slice = tuple((this_expand_around[d] if d!=ci else axis_var for d in range(3)))
    
    xs = axis_var
    
    tp = bpl.total_potential(axis_slice, tweezer_params, atom)

    xyzaxis = ["x","y","z"][ci]
    ys = tp

    line_main = ax.plot(xs,tp,c=col,label=f'{atom}')
    # ax.axvline(rb_min[ci] if atom == 'Rb' else cs_min[ci], ls='--', alpha=0.4, color=col)

    fit_threshold_um = 0.5
    fit_y_ratio = 0.8

    trap_freq, quad_popt = bpl.my_get_trap_frequency(xs,ys, mass,
                                                    x_fit_threshold_um=fit_threshold_um,y_fit_threshold_ratio=fit_y_ratio,
                                                    x_bounds=(rbcs_min[ci]-0.3,rbcs_min[ci]+0.3))
    centre_position_um = quad_popt[0]
    centre_intensity_min = quad_popt[2]
    xs_fit = np.linspace(centre_position_um-fit_threshold_um,centre_position_um+fit_threshold_um,200)
    ys_fit = bpl.quad(xs_fit, *quad_popt)
    min_y_fit = np.min(ys_fit)
    # line_fit = ax.plot(xs_fit, ys_fit, lw=1, linestyle='-', c='red')
    fit_mins[0][ci] = centre_position_um

    wf_space,dxwfs = np.linspace(centre_position_um-1.3,centre_position_um+1.3,200,retstep=True)
    wf = (mass*u*2*np.pi*trap_freq*1e6/(np.pi*scipy.constants.hbar))**(0.25) * np.exp(-(mass * u * 2 * np.pi * trap_freq * 1e6 * ((wf_space-centre_position_um)*1e-6)**2)/(2*scipy.constants.hbar)) #+ centre_intensity_min
    ax.axvline(centre_position_um, lw=1, linestyle='--', color=col)
    ax.axvline(rbcs_min[ci], lw=1, linestyle='-', color=col)
    # ax_wf2.axvline(centre_position_um*1e3, lw=1, linestyle='--', color=col)
    sig_times_3 = 3*np.sqrt((scipy.constants.hbar)/(mass * u * 2 * np.pi * trap_freq * 1e-6))
    bare_length_scales_rbcs.append(sig_times_3)
    wf_prob_per_m = ((wf)**2)
    wf_prob_per_um = wf_prob_per_m*1e-6
    print(np.sum(wf_prob_per_um*dxwfs))
    print(f"{atom} {xyzaxis} Trap Frequency {trap_freq*1e3}  (kHz)")
    bare_trap_freqs_khz_rbcs.append(trap_freq*1e3)
    ax_low.text(centre_position_um,10-(3 if atom=='Rb' else 0), f"TF {trap_freq*1e3:.1f}(kHz)",size='x-small',color=col,ha='right' if atom=='Cs' else 'left')
    ax_low.plot(wf_space,wf_prob_per_um, c=col,alpha=0.4)
    ax_low.fill_between(wf_space,wf_prob_per_um,y2=0,color=col, alpha=0.2)

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

axs[0,0].set_ylabel('Trap depth (MHz?)')

plt.show()


# %% 2D plot
atom='RbCs'
fig, ax = plt.subplots(constrained_layout=True)

tweezer_params_rbcs = deepcopy(tweezer_params_base_rbcs)

# tweezer_params_rbcs[(1145,"beams")][0]['power_mW']=0
# tweezer_params_rbcs[(1145,"beams")][1]['power_mW']=0
# tweezer_params_rbcs[(1145,"beams")][1]['phase']=-1.4
# tweezer_params_rbcs[(1145,"beams")][2]['phase']=-1.9

x = np.linspace(-5,5,200)
z = np.linspace(-20,20,200)

x_mesh,y_mesh = np.meshgrid(x,z,indexing='ij')

pot_mesh = bpl.total_potential((x_mesh,0,y_mesh),tweezer_params_rbcs,atom)
norm = mpl.colors.Normalize(vmin=-2.5, vmax=0)

cfa = ax.contourf(x_mesh,y_mesh,pot_mesh,levels=1000, cmap='afmhot_r', norm=norm)
cb = fig.colorbar(cfa, ax=ax)
cb.set_label('Potential (MHz)')

ax.set_xlabel(r'$x_{\mathrm{Twe}}$ (um)')
ax.set_ylabel(r'$z_{\mathrm{Twe}}$ (um)')

ellipse = mpl.patches.Ellipse(xy=(fit_mins[0][0],fit_mins[0][2]),
                              width=bare_length_scales_rbcs[0], height=bare_length_scales_rbcs[2], 
                        edgecolor='r', fc='None', lw=2)
ax.add_patch(ellipse)

plt.show()

# %% Move two tweezers together

tweezer_params_rbcs = deepcopy(tweezer_params_base_rbcs)
# tweezer_params_rbcs[(1145,"beams")][0]['power_mW']=0
# tweezer_params_rbcs[(1145,"beams")][1]['power_mW']=0

start_positions = [[0,0,-6],[0,0,6]]
n_copies = len(start_positions)

positions = []
positions.append(start_positions)
lss = []

spacing_from_zeros = np.linspace(6,1,100)

for sfzi, spacing_from_zero in enumerate(spacing_from_zeros):
    tweezer_params_rbcs[(1145,"beams")][2]['z0']=+spacing_from_zero
    tweezer_params_rbcs[(1145,"beams")][3]['z0']=-spacing_from_zero
    # tweezer_params_rbcs[(1145,"beams")][1]['phase']=-spacing_from_zero
    previous_position = positions[-1]
    positions.append([])
    lss.append([])

    ########
    fig_2d,ax_2d=plt.subplots()
    x_2d = np.linspace(-5,5,200)
    z_2d = np.linspace(-20,20,200)

    x_mesh,y_mesh = np.meshgrid(x,z,indexing='ij')

    pot_mesh = bpl.total_potential((x_mesh,0,y_mesh),tweezer_params_rbcs,atom)
    norm = mpl.colors.Normalize(vmin=-3.8, vmax=0)

    cfa = ax_2d.contourf(x_mesh,y_mesh,pot_mesh,levels=1000, cmap='afmhot_r', norm=norm)
    cb = fig_2d.colorbar(cfa, ax=ax_2d)
    # cb.set_clim(vmin=-3.8, vmax=0)
    cb.set_label('Potential (MHz)')

    ax_2d.set_xlabel(r'$x_{\mathrm{Twe}}$ (um)')
    ax_2d.set_ylabel(r'$z_{\mathrm{Twe}}$ (um)')
    ########

    fig,axs = plt.subplots(2,3,height_ratios=(4,1),sharey='row',sharex='none')
    for si in range(n_copies):
        rbcs_min = bpl.my_find_potential_minimum(previous_position[si],tweezer_params_rbcs,"RbCs")
        bare_trap_freqs_khz_rbcs = []
        bare_length_scales_rbcs = []

        fit_it=True
        fit_mins = [[0,0,0]] # Will change
        atom = "RbCs"
        mass = bpl.species[atom]['m']
        col = 'red'
        tweezer_params = tweezer_params_rbcs
        this_expand_around = rbcs_min

        x = np.linspace(-4,4,200) #np.concatenate(([0],np.linspace(-0.01,-2,20),np.linspace(0.01,2,20)))
        y = np.linspace(-4,4,42) #np.concatenate(([0],np.linspace(-0.01,-2,20),np.linspace(0.01,2,20)))
        z = np.linspace(-10,10,400)

        for ci,ax_pair in enumerate(axs.T):
            ax, ax_low = ax_pair
            
            axis_var = [x,y,z][ci]
            axis_slice = tuple((this_expand_around[d] if d!=ci else axis_var for d in range(3)))
            
            xs = axis_var
            
            tp = bpl.total_potential(axis_slice, tweezer_params, atom)

            xyzaxis = ["x","y","z"][ci]
            ys = tp

            # ax.axvline(rb_min[ci] if atom == 'Rb' else cs_min[ci], ls='--', alpha=0.4, color=col)

            fit_threshold_um = 0.5
            fit_y_ratio = 0.8

            trap_freq, quad_popt = bpl.my_get_trap_frequency(xs,ys, mass,
                                                            x_fit_threshold_um=fit_threshold_um,y_fit_threshold_ratio=fit_y_ratio,
                                                            x_bounds=(rbcs_min[ci]-0.3,rbcs_min[ci]+0.3))
            centre_position_um = quad_popt[0]
            centre_intensity_min = quad_popt[2]
            xs_fit = np.linspace(centre_position_um-fit_threshold_um,centre_position_um+fit_threshold_um,200)
            ys_fit = bpl.quad(xs_fit, *quad_popt)
            min_y_fit = np.min(ys_fit)
            # line_fit = ax.plot(xs_fit, ys_fit, lw=1, linestyle='-', c='red')
            fit_mins[0][ci] = centre_position_um


            wf_space,dxwfs = np.linspace(centre_position_um-1.3,centre_position_um+1.3,200,retstep=True)
            wf = (mass*u*2*np.pi*trap_freq*1e6/(np.pi*scipy.constants.hbar))**(0.25) * np.exp(-(mass * u * 2 * np.pi * trap_freq * 1e6 * ((wf_space-centre_position_um)*1e-6)**2)/(2*scipy.constants.hbar)) #+ centre_intensity_min
            # ax_wf2.axvline(centre_position_um*1e3, lw=1, linestyle='--', color=col)
            sig_times_3 = 3*np.sqrt((scipy.constants.hbar)/(mass * u * 2 * np.pi * trap_freq * 1e-6))
            bare_length_scales_rbcs.append(sig_times_3)
            wf_prob_per_m = ((wf)**2)
            wf_prob_per_um = wf_prob_per_m*1e-6
            print(np.sum(wf_prob_per_um*dxwfs))
            print(f"{atom} {xyzaxis} Trap Frequency {trap_freq*1e3}  (kHz)")
            bare_trap_freqs_khz_rbcs.append(trap_freq*1e3)
            
            if si ==0:
                ax.axvline(centre_position_um, lw=1, linestyle='--', color=col)
                line_main = ax.plot(xs,tp,c=col,label=f'{atom}')
                ax_low.text(centre_position_um,10-(3 if atom=='Rb' else 0), f"TF {trap_freq*1e3:.1f}(kHz)",size='x-small',color=col,ha='right' if atom=='Cs' else 'left')
                ax_low.plot(wf_space,wf_prob_per_um, c=col,alpha=0.4)
                ax_low.fill_between(wf_space,wf_prob_per_um,y2=0,color=col, alpha=0.2)
        
        axs[1,0].set_ylabel(r'$|\psi|$ (um$^{-1}$)')


        positions[-1].append(rbcs_min)
        lss[-1].append(bare_length_scales_rbcs)
        axs[0,0].legend()

        for ax_pair, axxname in zip(axs.T,['x','y','z']):
            ax_pair[-1].set_xlabel(f'{axxname} (um)')

        axs[0,0].set_ylabel('Trap depth (MHz?)')

    
    ellipse = mpl.patches.Ellipse(xy=(positions[-1][0][0],positions[-1][0][2]),
                        width=lss[-1][0][0], height=lss[-1][0][2], 
                    edgecolor='None', fc='green', lw=0.2)
    ax_2d.add_patch(ellipse)

    ellipse_2 = mpl.patches.Ellipse(xy=(positions[-1][1][0],positions[-1][1][2]),
                        width=lss[-1][1][0], height=lss[-1][1][2], 
                    edgecolor='None', fc='green', lw=0.2)
    ax_2d.add_patch(ellipse_2)
    fig_2d.savefig(f'{sfzi}.png')
    plt.show()



# %%
fig,ax = plt.subplots()
positions = np.array(positions)
ax.plot(spacing_from_zeros,positions[:,0,2][1:],label='x0')#x-+
ax.plot(spacing_from_zeros,positions[:,1,2][1:],label='x1')#x-+
ax.set_xlabel('phase change (rad)')
ax.set_ylabel('z-particle position (um)')
plt.show()

# %%
