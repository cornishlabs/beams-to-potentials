import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.optimize import curve_fit, fmin, fmin_tnc
from copy import deepcopy
import scipy.constants

from scipy.interpolate import interp1d

def sqrt(x,a):
    return a*x**(1/2)

from math import floor, log10

def get_valerr_string(val,err):
    prec = floor(log10(err))
    err = round(err/10**prec)*10**prec
    val = round(val/10**prec)*10**prec
    if prec > 0:
        valerr = '{:.0f}({:.0f})'.format(val,err)
    else:
        valerr = '{:.{prec}f}({:.0f})'.format(val,err*10**-prec,prec=-prec)
    return valerr
    
#%% 817nm
rsc_calibration = 138.5/8**(1/2)
parametric_calibration = 86.3/3.7**(1/2)
average_calibration = (rsc_calibration+parametric_calibration)/2

def get_817_trap_freq_err(powers):
    trap_freqs = []
    trap_freqs_err = []
    for power in powers:
        trap_freqs.append(sqrt(power,average_calibration))
        # trap_freqs_err.append(max([sqrt(power,rsc_calibration)-sqrt(power,average_calibration),
        #                       sqrt(power,average_calibration)-sqrt(power,parametric_calibration)]))
        trap_freqs_err.append((sqrt(power,rsc_calibration)-sqrt(power,parametric_calibration))/2)
    return np.array(trap_freqs), np.array(trap_freqs_err)

#%% 1065nm
calibration_1065 = 118/5.42**(1/2)
calibration_1065_upper = calibration_1065*rsc_calibration/average_calibration
calibration_1065_lower = calibration_1065*parametric_calibration/average_calibration
parametric_calibration = 86.3/3.7**(1/2)
average_calibration = (rsc_calibration+parametric_calibration)/2

def get_1065_trap_freq_err(powers):
    trap_freqs = []
    trap_freqs_err = []
    for power in powers:
        trap_freqs.append(sqrt(power,calibration_1065))
        # trap_freqs_err.append(max([sqrt(power,calibration_1065_upper)-sqrt(power,calibration_1065),
        #                       sqrt(power,calibration_1065)-sqrt(power,calibration_1065_lower)]))
        trap_freqs_err.append((sqrt(power,calibration_1065_upper)-sqrt(power,calibration_1065_lower))/2)
    return np.array(trap_freqs), np.array(trap_freqs_err)

#%%
epsilon_0 = 8.854E-12
c = 2.99E+08
a0 = 5.29E-11
k_B = 1.38E-23
h = 6.62E-34
u = 1.66053906660e-27

pol_rb_1065 = 687
pol_rb_817 = 4690
pol_cs_1065 = 1168
pol_cs_817 = -3253
pol_rbcs_00_1065 = 1825 
pol_rbcs_33_1065 = 1825 - (1/3) * 1981 
pol_rbcs_00_817 = 443
pol_rbcs_33_817 = 443 - (1/3) * (-2816)

masses = {'Rb':87, 'Cs':133, 'RbCs00':87+133,'RbCs33':87+133}

tweezers_base = {'817':{'wx0':0.825,
                        'wy0':0.925,
                        'wz0':0.935,
                        'power_mW':2.5, # powers will be overwritten when the power for the correct trap freq is found
                        'pol_Rb':4690,
                        'pol_Cs':-3253,
                        'pol_RbCs00':443,
                        'pol_RbCs33':443 - (1/3) * (-2816),
                        'x0':0,
                        'y0':0,
                        'z0':0},
                  '1065':{'wx0':1.05,
                          'wy0':1.16,
                          'wz0':1.19,
                          'power_mW':11.4, # powers will be overwritten when the power for the correct trap freq is found
                          'pol_Rb':687,
                          'pol_Cs':1168,
                          'pol_RbCs00': 1825,
                          'pol_RbCs33': 1825 - (1/3) * 1981 ,
                          'x0':0,
                          'y0':0,
                          'z0':0}}

scattering_length_nm = 645*a0*1e9 #RbCs background scattering length in nm

color_817 = '#AA2A4A'
color_1065 = '#DDA81C'
color_cs = '#EE2519'
color_rb = '#2571E1'

def freq_min_jerk(start_freq_MHz=100,end_freq_MHz=101,start_phase=0,_time=None,_T=None):
    d = (end_freq_MHz-start_freq_MHz)
    if _T == None:
        _T = _time[-1] - _time[0]
    return d*(10*(_time/_T)**3 - 15*(_time/_T)**4 + 6*(_time/_T)**5) + start_freq_MHz

def freq_sweep(start_freq_MHz,end_freq_MHz,hybridicity,
               start_phase,_time):
    d = (end_freq_MHz-start_freq_MHz)
    T = _time[-1] - _time[0]
    deltat = T*(1-hybridicity)/2
    deltaf = d/(2+15/4*hybridicity/(1-hybridicity))
    
    time_cutoff = round((1-hybridicity)/2*len(_time))
    
    time1 = _time[:time_cutoff]
    time2 = _time[time_cutoff:len(_time)-time_cutoff]
    time3 = _time[len(_time)-time_cutoff:]
    
    try:
        freq1 = list(freq_min_jerk(start_freq_MHz,start_freq_MHz+2*deltaf,0,time1,2*deltat))
    except IndexError:
        freq1 = []
        
    try:
        freq2 = list(15/4*deltaf/2/deltat*(time2-time2[0])+(start_freq_MHz+deltaf))
    except IndexError:
        freq2 = []
    
    try:
        freq3 = list(freq_min_jerk(end_freq_MHz-2*deltaf,end_freq_MHz,0,time3-_time[-1]+2*deltat,2*deltat))
    except IndexError:
        freq3 = []
        
    return np.asarray(freq1+freq2+freq3)

def amp_approx_exp(start_amp=1,end_amp=0,index=20,_time=None):
    """Approximate exponential ramp that starts and ends at the specified
    value. If the index is set to a negative value, the slowly 
    rising part will be at the start instead.
    
    """
    if index > 0:
        return np.flip((index**(_time/_time[-1])-1)/(index-1)*(start_amp-end_amp) + end_amp)
    else:
        index = abs(index)
        return ((index**(_time/_time[-1])-1)/(index-1)*(end_amp-start_amp) + start_amp)

def gaussian(xrange,x0,polarisability):
    gauss = np.exp(-(xrange-x0)**2)
    return gauss/np.max(gauss)*(-polarisability)/1000

def waist(z,w0,wavelength):
    zR = np.pi*w0**2/wavelength
    return w0*np.sqrt(1+(z/zR)**2)

def gaussian_3d(x,y,z,x0,y0,z0,wx0,wy0,wz0,I0,wavelength):
    wx = waist(z-z0,wx0,wavelength)
    wy = waist(z-z0,wy0,wavelength)
    wz = waist(z-z0,wz0,wavelength)
    return I0*np.exp(-2*(x-x0)**2/wx**2)*np.exp(-2*(y-y0)**2/wy**2)*(wz0/wz)**2

def trap_potential(x,y,z,x0,y0,z0,wx0,wy0,wz0,polarisability,power_mW,wavelength):
    intensity = gaussian_3d(x,y,z,x0,y0,z0,wx0,wy0,wz0,1,wavelength)
    center_intensity_mW_um2 = 2*power_mW/(np.pi*wx0*wy0)
    center_intensity_W_m2 = center_intensity_mW_um2/1e3*(1e6)**2
    if True:
        print(f"{wavelength*1e3:.1f}nm: {power_mW:.2f}mW => Centre Intensity {center_intensity_mW_um2*1e2:.2f} kW/cm^2 ")
    polarisability_si = polarisability*4*np.pi*epsilon_0*a0**3
    trap_depth_J = center_intensity_W_m2*polarisability_si/(2*c*epsilon_0)
    trap_depth_MHz = trap_depth_J/h/1e6
    return -trap_depth_MHz*intensity

def quad(x,x0,a,y0):
    return a/2*(x-x0)**2 + y0

def get_trap_frequency(xs,potential,mass,x_fit_threshold_um=0.05,plot=False):
    """Fits a quadratic to the minimum of a potential to find a trap frequency."""
    mass_si = mass*u
    
    df = pd.DataFrame()
    df['x'] = xs
    df['y'] = potential
    
    idxmin = df['y'].idxmin()
    y0 = df['y'][idxmin]
    x0 = df['x'][idxmin]

    df_turn = df[np.abs(df['x'] - x0) < x_fit_threshold_um]
    
    fix_quad = lambda x,a : quad(x,x0,a,y0)
    
    popt,pcov = curve_fit(fix_quad,df_turn['x'],df_turn['y'])#,p0=[0])
    
    trap_frequency_MHz = np.sqrt(popt[0]*1e6*h*(1e6)**2/mass_si)/1e6/2/np.pi # k is in MHz/um**2
    
    if plot:
        plt.plot(xs,potential,c='blue',linestyle='-')
        ylim = plt.gca().get_ylim()
        plt.plot(xs, fix_quad(xs,popt[0]),c='red',linestyle='--')
        plt.ylabel('potential (MHz)')
        plt.xlabel('distance (um)')
        plt.ylim(ylim)
        plt.show()
    
    return trap_frequency_MHz, popt[0], x0

def get_817_power_mW(desired_trap_freq):
    """Uses an inputted value of the 817 trap frequency (kHz) and gets the 
    equivalent power in mW for this code to use in order to have the correct 
    initial trap frequency. The waists specified in the tweezer_base dict 
    are used during the trajectory calculation, but this gets the right power."""
    
    x0s = np.asarray([-2*1.854])
    xs = np.linspace(-5,1,1001)
    atom = 'Rb'
    
    trap_frequencies_x_MHz_Rb = []
    
    for x0 in x0s:
        tweezers = deepcopy(tweezers_base)
        tweezers['817']['x0'] = x0
        
        # Rb first
        atom = 'Rb'
        # get x trap frequency
        potential = np.zeros_like(xs)
        for wavelength_nm, params in deepcopy(tweezers).items():
            wavelength = float(wavelength_nm)/1e3
            
            potential += trap_potential(xs,0,0,
                                        params['x0'],params['y0'],params['z0'],
                                        params['wx0'],params['wy0'],params['wz0'],
                                        params['pol_'+atom],params['power_mW'],
                                        wavelength)
    
        trap_frequency_x_MHz, k_x, xmin = get_trap_frequency(xs,potential,masses[atom],plot=False)
        trap_frequencies_x_MHz_Rb.append(trap_frequency_x_MHz)
        
    current_trap_freq = trap_frequencies_x_MHz_Rb[0]*1e3
    required_power = tweezers['817']['power_mW']*(desired_trap_freq/current_trap_freq)**2
    return required_power

def get_1065_power_mW(desired_trap_freq):
    """Uses an inputted value of the 1065 trap frequency (kHz) and gets the 
    equivalent power in mW for this code to use in order to have the correct 
    initial trap frequency. The waists specified in the tweezer_base dict 
    are used during the trajectory calculation, but this gets the right power."""
    
    x0s = np.asarray([-2*1.854])
    xs = np.linspace(-5,1,1001)
    atom = 'Cs'

    trap_frequencies_x_MHz_Cs = []
    
    for x0 in x0s:
        tweezers = deepcopy(tweezers_base)
        tweezers['817']['x0'] = x0

        # get x trap frequency
        potential = np.zeros_like(xs)
        for wavelength_nm, params in deepcopy(tweezers).items():
            wavelength = float(wavelength_nm)/1e3
            
            potential += trap_potential(xs,0,0,
                                        params['x0'],params['y0'],params['z0'],
                                        params['wx0'],params['wy0'],params['wz0'],
                                        params['pol_'+atom],params['power_mW'],
                                        wavelength)
    
        trap_frequency_x_MHz, k_x, xmin = get_trap_frequency(xs,potential,masses[atom],plot=False)
        trap_frequencies_x_MHz_Cs.append(trap_frequency_x_MHz)
        
    current_trap_freq = trap_frequencies_x_MHz_Cs[0]*1e3
    required_power = tweezers['1065']['power_mW']*(desired_trap_freq/current_trap_freq)**2
    return required_power
    

def calc_trajectory(initial_trap_freq_817, initial_trap_freq_1065,
                    final_817_x_um=0,x_1065_um=0,len_ts=1000,calc_frac=None,
                    y_817_um=0,z_817_um=0,**kwargs):
    """Gets the trap frequencies along the merging direction by fitting the 
    potential minima."""
    # ts = np.concatenate([np.linspace(0,0.5,20),np.linspace(0.55,2.45,20),np.linspace(2.5,2.7,50)])
    final_817_pos=final_817_x_um #0.200
    ts = np.concatenate([np.linspace(0,2.7,len_ts)])
    x0s = freq_sweep(-2*1.854+final_817_pos,final_817_pos,0.95,0,ts)
    
    if calc_frac is not None:
        num_points = int(len(ts)*calc_frac)
        ts = ts[-num_points:]
        x0s = x0s[-num_points:]
    
    xs = np.linspace(-5,1,100001)
    
    trap_frequencies_x_MHz_Rb = []
    trap_frequencies_x_MHz_Cs = []
    
    xmin_Rb = []
    ymin_Rb = []
    zmin_Rb = []
    
    xmin_Cs = []
    ymin_Cs = []
    zmin_Cs = []
    
    power_817_mW = get_817_power_mW(initial_trap_freq_817)[0]
    power_1065_mW = get_1065_power_mW(initial_trap_freq_1065)[0]
    
    print('817 power = {} mW'.format(power_817_mW))
    print('1065 power = {} mW'.format(power_1065_mW))
    
    for num, x0 in enumerate(x0s):
        print(num,'/',len(x0s))
        tweezers = deepcopy(tweezers_base)
        tweezers['817']['x0'] = x0
        tweezers['817']['y0'] = y_817_um
        tweezers['817']['z0'] = z_817_um
        tweezers['817']['power_mW'] = power_817_mW
        tweezers['1065']['power_mW'] = power_1065_mW
        tweezers['1065']['x0'] = x_1065_um
        
        # Rb first
        atom = 'Rb'
        
        x_Rb, y_Rb, z_Rb = find_potential_minimum([x0,y_817_um,0],tweezers,atom)
        
        coord = [xs,y_Rb,z_Rb] # calculate the potential along the merging axis but for Rb's position
        potential = total_potential(coord,tweezers,atom)
    
        trap_frequency_x_MHz, k_x, xmin = get_trap_frequency(xs,potential,masses[atom],plot=False)
        trap_frequencies_x_MHz_Rb.append(trap_frequency_x_MHz)
        
        xmin_Rb.append(x_Rb)
        ymin_Rb.append(y_Rb)
        zmin_Rb.append(z_Rb)
        
        print(xmin,x_Rb,y_Rb,z_Rb)
        
        # now Cs
        atom = 'Cs'
        
        x_Cs, y_Cs, z_Cs = find_potential_minimum([0,0,0],tweezers,atom)
        
        coord = [xs,y_Cs,z_Cs] # calculate the potential along the merging axis but for Cs's position
        potential = total_potential(coord,tweezers,atom)
        
        trap_frequency_x_MHz, k_x, xmin = get_trap_frequency(xs,potential,masses[atom],plot=False)
        trap_frequencies_x_MHz_Cs.append(trap_frequency_x_MHz)
        xmin_Cs.append(x_Cs)
        ymin_Cs.append(y_Cs)
        zmin_Cs.append(z_Cs)
    
    trap_frequencies_x_MHz_Rb = np.array(trap_frequencies_x_MHz_Rb)
    trap_frequencies_x_MHz_Cs = np.array(trap_frequencies_x_MHz_Cs)
    
    xmin_Cs = np.array(xmin_Cs)
    ymin_Cs = np.array(ymin_Cs)
    zmin_Cs = np.array(zmin_Cs)
    
    xmin_Rb = np.array(xmin_Rb)
    ymin_Rb = np.array(ymin_Rb)
    zmin_Rb = np.array(zmin_Rb)
    
    trap_frequencies_x_MHz_both = np.sqrt((masses['Cs']*trap_frequencies_x_MHz_Rb**2+masses['Rb']*trap_frequencies_x_MHz_Cs**2)/(masses['Cs']+masses['Rb']))
    
    df = pd.DataFrame()
    df['time (ms)'] = ts
    df['817 position (um)'] = x0s
    df['trap frequency Rb (kHz)'] = trap_frequencies_x_MHz_Rb*1e3
    df['trap frequency Cs (kHz)'] = trap_frequencies_x_MHz_Cs*1e3
    df['Rb position (um)'] = xmin_Rb
    df['Cs position (um)'] = xmin_Cs
    df['Rb position (other radial) (um)'] = ymin_Rb
    df['Cs position (other radial) (um)'] = ymin_Cs
    df['Rb position (axial) (um)'] = zmin_Rb
    df['Cs position (axial) (um)'] = zmin_Cs
    
    df['deltaz (um)'] = xmin_Rb-xmin_Cs
    df['ddeltaz/dt (um/ms)'] = list(np.diff(xmin_Rb-xmin_Cs)/np.diff(ts))+[np.nan]
    
    df['trap frequency av (kHz)'] = np.sqrt((df['trap frequency Rb (kHz)']**2+df['trap frequency Cs (kHz)']**2)/2)
    df['trap frequency rel (kHz)'] = np.sqrt((masses['Rb']*df['trap frequency Cs (kHz)']**2+
                                              masses['Cs']*df['trap frequency Rb (kHz)']**2)/
                                             (masses['Rb']+masses['Cs']))

    df['beta (nm)'] = calc_confinement_length(df['trap frequency rel (kHz)'])

    df['deltaz_X (um)'] = df['beta (nm)']*np.sqrt(3+df['beta (nm)']**2/scattering_length_nm**2)*(-1e-3)
    
    return df

def calc_ramp(initial_trap_freq_817, initial_trap_freq_1065,
              coords_817=[0,0,0],len_ts=1000,calc_frac=None,**kwargs):
    """Gets the trap frequencies along the merging direction by fitting the 
    potential minima. Redefine the tweezers_base dict to specify the positions."""
    # ts = np.concatenate([np.linspace(0,0.5,20),np.linspace(0.55,2.45,20),np.linspace(2.5,2.7,50)])
    # final_817_pos=final_817_x_um #0.200
    ts = np.linspace(0,10,100)
    amp_817s = (amp_approx_exp(1,0,20,ts))**2
    
    if calc_frac is not None:
        num_points = int(len(ts)*calc_frac)
        ts = ts[-num_points:]
        amp_817s = amp_817s[-num_points:]
    
    xs = np.linspace(-5,1,100001)
    
    trap_frequencies_x_MHz_Rb = []
    trap_frequencies_x_MHz_Cs = []
    
    xmin_Rb = []
    ymin_Rb = []
    zmin_Rb = []
    
    xmin_Cs = []
    ymin_Cs = []
    zmin_Cs = []
    
    power_817_mW = get_817_power_mW(initial_trap_freq_817)[0]
    power_1065_mW = get_1065_power_mW(initial_trap_freq_1065)[0]
    
    print('817 power = {} mW'.format(power_817_mW))
    print('1065 power = {} mW'.format(power_1065_mW))
    
    for num, amp_817 in enumerate(amp_817s):
        print(num,'/',len(amp_817s))
        tweezers = deepcopy(tweezers_base)
        tweezers['817']['x0'] = coords_817[0]
        tweezers['817']['y0'] = coords_817[1]
        tweezers['817']['z0'] = coords_817[2]
        tweezers['817']['power_mW'] = power_817_mW*amp_817
        tweezers['1065']['power_mW'] = power_1065_mW
        
        # Rb first
        atom = 'Rb'
        
        x_Rb, y_Rb, z_Rb = find_potential_minimum(coords_817,tweezers,atom)
        
        coord = [xs,y_Rb,z_Rb] # calculate the potential along the merging axis but for Rb's position
        potential = total_potential(coord,tweezers,atom)
    
        trap_frequency_x_MHz, k_x, xmin = get_trap_frequency(xs,potential,masses[atom],plot=False)
        trap_frequencies_x_MHz_Rb.append(trap_frequency_x_MHz)
        
        xmin_Rb.append(x_Rb)
        ymin_Rb.append(y_Rb)
        zmin_Rb.append(z_Rb)
        
        print(xmin,x_Rb,y_Rb,z_Rb)
        
        # now Cs
        atom = 'Cs'
        
        x_Cs, y_Cs, z_Cs = find_potential_minimum([0,0,0],tweezers,atom)
        
        coord = [xs,y_Cs,z_Cs] # calculate the potential along the merging axis but for Cs's position
        potential = total_potential(coord,tweezers,atom)
        
        trap_frequency_x_MHz, k_x, xmin = get_trap_frequency(xs,potential,masses[atom],plot=False)
        trap_frequencies_x_MHz_Cs.append(trap_frequency_x_MHz)
        xmin_Cs.append(x_Cs)
        ymin_Cs.append(y_Cs)
        zmin_Cs.append(z_Cs)
    
    trap_frequencies_x_MHz_Rb = np.array(trap_frequencies_x_MHz_Rb)
    trap_frequencies_x_MHz_Cs = np.array(trap_frequencies_x_MHz_Cs)
    
    xmin_Cs = np.array(xmin_Cs)
    ymin_Cs = np.array(ymin_Cs)
    zmin_Cs = np.array(zmin_Cs)
    
    xmin_Rb = np.array(xmin_Rb)
    ymin_Rb = np.array(ymin_Rb)
    zmin_Rb = np.array(zmin_Rb)
    
    trap_frequencies_x_MHz_both = np.sqrt((masses['Cs']*trap_frequencies_x_MHz_Rb**2+masses['Rb']*trap_frequencies_x_MHz_Cs**2)/(masses['Cs']+masses['Rb']))
    
    df = pd.DataFrame()
    df['time (ms)'] = ts
    df['817_power (mW)'] = amp_817s*power_817_mW
    df['trap frequency Rb (kHz)'] = trap_frequencies_x_MHz_Rb*1e3
    df['trap frequency Cs (kHz)'] = trap_frequencies_x_MHz_Cs*1e3
    df['Rb position (um)'] = xmin_Rb
    df['Cs position (um)'] = xmin_Cs
    df['Rb position (other radial) (um)'] = ymin_Rb
    df['Cs position (other radial) (um)'] = ymin_Cs
    df['Rb position (axial) (um)'] = zmin_Rb
    df['Cs position (axial) (um)'] = zmin_Cs
    
    df['deltaz (um)'] = xmin_Rb-xmin_Cs
    df['ddeltaz/dt (um/ms)'] = list(np.diff(xmin_Rb-xmin_Cs)/np.diff(ts))+[np.nan]
    
    df['trap frequency av (kHz)'] = np.sqrt((df['trap frequency Rb (kHz)']**2+df['trap frequency Cs (kHz)']**2)/2)
    df['trap frequency rel (kHz)'] = np.sqrt((masses['Rb']*df['trap frequency Cs (kHz)']**2+
                                              masses['Cs']*df['trap frequency Rb (kHz)']**2)/
                                             (masses['Rb']+masses['Cs']))

    df['beta (nm)'] = calc_confinement_length(df['trap frequency rel (kHz)'])

    df['deltaz_X (um)'] = df['beta (nm)']*np.sqrt(3+df['beta (nm)']**2/scattering_length_nm**2)*(-1e-3)
    
    return df

def total_potential(coord,tweezers,atom):
    x = coord[0]
    y = coord[1]
    z = coord[2]
    potential = 0
    for wavelength_nm, params in deepcopy(tweezers).items():
        wavelength = float(wavelength_nm)/1e3
        
        potential += trap_potential(x,y,z,
                                    params['x0'],params['y0'],params['z0'],
                                    params['wx0'],params['wy0'],params['wz0'],
                                    params['pol_'+atom],params['power_mW'],
                                    wavelength)
    return potential

def find_potential_minimum(min_guess,tweezers,atom):
    xtol = 0.000001
    ftol = 0.000001
    min_x, min_y, min_z = fmin(total_potential,min_guess,args=(tweezers,atom),xtol=xtol,ftol=ftol,disp=False)
    return min_x, min_y, min_z

def my_find_potential_minimum(min_guess,tweezers,atom):
    # xtol = 0.000001
    # ftol = 0.000001
    minr, _,_ = fmin_tnc(total_potential,min_guess,args=(tweezers,atom),approx_grad=True, bounds=[(-5,5),(-5,5),(-5,5)],epsilon = 0.01,disp=False)
    return minr

#%%    
def calc_confinement_length(trap_freq_res_kHz):
    reduced_mass = 1/(1/masses['Cs']+1/masses['Rb'])
    u = 1.66053906660e-27
    h = 6.62E-34
    
    return np.sqrt(h/(trap_freq_res_kHz*1e3*reduced_mass*u))*1e9/(2*np.pi)
    
def calc_trajectory_from_817_voltage(power_817,power_1065_scale=0.45,calc_errors=False,**kwargs):
    if np.isnan(power_817):
        return [[np.nan,np.nan,np.nan,np.nan],[np.nan,np.nan],[np.nan,np.nan],[np.nan,np.nan]]
    power_1065 = power_817*power_1065_scale
    
    print('817 power = {:.2f} V'.format(power_817))
    print('1065 power = {:.2f} V'.format(power_1065))
    
    initial_trap_freq_817, initial_trap_freq_817_err = get_817_trap_freq_err([power_817])
    initial_trap_freq_1065, initial_trap_freq_1065_err = get_1065_trap_freq_err([power_1065])
    
    print(initial_trap_freq_817, initial_trap_freq_817_err)
    
    print('Rb 817 trap frequency = {} kHz'.format(get_valerr_string(*initial_trap_freq_817,*initial_trap_freq_817_err)))
    print('Cs 1065 trap frequency = {} kHz'.format(get_valerr_string(*initial_trap_freq_1065,*initial_trap_freq_1065_err)))
    
    trajectory_df = calc_trajectory(initial_trap_freq_817,initial_trap_freq_1065,**kwargs)
    if not calc_errors:
        return trajectory_df
    else:
        trajectory_df_upper_err = calc_trajectory(initial_trap_freq_817+initial_trap_freq_817_err,initial_trap_freq_1065+initial_trap_freq_1065_err,**kwargs)
        trajectory_df_lower_err = calc_trajectory(initial_trap_freq_817-initial_trap_freq_817_err,initial_trap_freq_1065-initial_trap_freq_1065_err,**kwargs)
        return trajectory_df,trajectory_df_upper_err,trajectory_df_lower_err

def calc_ramp_from_817_voltage(power_817,power_1065_scale=0.45,**kwargs):
    if np.isnan(power_817):
        return [[np.nan,np.nan,np.nan,np.nan],[np.nan,np.nan],[np.nan,np.nan],[np.nan,np.nan]]
    power_1065 = power_817*power_1065_scale
    
    print('817 power = {:.2f} V'.format(power_817))
    print('1065 power = {:.2f} V'.format(power_1065))
    
    initial_trap_freq_817, initial_trap_freq_817_err = get_817_trap_freq_err([power_817])
    initial_trap_freq_1065, initial_trap_freq_1065_err = get_1065_trap_freq_err([power_1065])
    
    print('Rb 817 trap frequency = {} kHz'.format(get_valerr_string(*initial_trap_freq_817,*initial_trap_freq_817_err)))
    print('Cs 1065 trap frequency = {} kHz'.format(get_valerr_string(*initial_trap_freq_1065,*initial_trap_freq_1065_err)))
    
    trajectory_df = calc_ramp(initial_trap_freq_817,initial_trap_freq_1065,**kwargs)
    
    return trajectory_df

def get_beta_crossing(power_817,**kwargs):
    df,df_upper,df_lower = calc_trajectory_from_817_voltage(power_817,calc_errors=True,**kwargs)
    
    betas = []
    for temp_df in [df,df_upper,df_lower]:
        time_to_beta = interp1d(temp_df['time (ms)'],temp_df['beta (nm)'])    
        deltaz_res_diff_to_t = interp1d(temp_df['deltaz (um)'] - temp_df['deltaz_X (um)'],temp_df['time (ms)'])
        time_crossing = deltaz_res_diff_to_t(0)
        beta_crossing = time_to_beta(time_crossing)
        betas.append(float(beta_crossing))
    return betas


def my_get_trap_frequency(xs,potential,mass,x_fit_threshold_um=1,y_fit_threshold_ratio=0.8, plot=False):
    """Fits a quadratic to the minimum of a potential to find a trap frequency."""
    mass_si = mass*u
    
    df = pd.DataFrame()
    df['x'] = xs
    df['y'] = potential
    
    idxmin = df['y'].idxmin()
    y0 = df['y'][idxmin]
    x0 = df['x'][idxmin]

    df_turn = df[(np.abs(df['x'] - x0) < x_fit_threshold_um)]
    df_turn = df_turn[(df_turn['y'] < y_fit_threshold_ratio * y0)]
    
    # fix_quad = lambda x,a : quad(x,x0,a,y0)
    # print(df_turn)
    popt,pcov = curve_fit(quad,df_turn['x'],df_turn['y'],p0=[x0,0.2,y0])#,p0=[0])
    
    trap_frequency_MHz = np.sqrt(popt[1]*1e6*h*(1e6)**2/mass_si)/1e6/2/np.pi # k is in MHz/um**2
    
    if plot:
        plt.plot(xs,potential,c='blue',linestyle='-')
        ylim = plt.gca().get_ylim()
        plt.plot(xs, quad(xs,*popt),c='red',linestyle='--')
        plt.ylabel('potential (MHz)')
        plt.xlabel('distance (um)')
        plt.ylim(ylim)
        plt.show()
    
    return trap_frequency_MHz, popt

#%% testing

from matplotlib.widgets import Button, Slider
# axamp = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
# amp_slider = Slider(
#     ax=axamp,
#     label="Amplitude",
#     valmin=0,
#     valmax=10,
#     valinit=init_amplitude,
#     orientation="vertical"
# )


# # The function to be called anytime a slider's value changes
# def update(val):
#     line.set_ydata(f(t, amp_slider.val, freq_slider.val))
#     fig.canvas.draw_idle()


# # register the update function with each slider
# freq_slider.on_changed(update)
# amp_slider.on_changed(update)

if __name__ == '__main__':
    fig, axs = plt.subplots(4,2,figsize=(12,10),constrained_layout=True)

    axp817 = fig.add_axes([0.1, 0.98, 0.8, 0.01])
    p817_slider = Slider(
        ax=axp817,
        label="p817",
        valmin=0,
        valmax=0.5,
        valinit=2.5 * 0.78  * 0.6177/ 8,
        orientation="horizontal"
    )
    
    axp1065 = fig.add_axes([0.1, 0.96, 0.8, 0.01])
    p1065_slider = Slider(
        ax=axp1065,
        label="p1065",
        valmin=0,
        valmax=10,
        valinit=30 * 0.43 / 9,
        orientation="horizontal"
    )

    axx1065 = fig.add_axes([0.1, 0.94, 0.8, 0.01])
    x1065_slider = Slider(
        ax=axx1065,
        label="x1065",
        valmin=0,
        valmax=2,
        valinit=0.75,
        orientation="horizontal"
    )

    axy1065 = fig.add_axes([0.1, 0.92, 0.8, 0.01])
    y1065_slider = Slider(
        ax=axy1065,
        label="y1065",
        valmin=0,
        valmax=2,
        valinit=0,
        orientation="horizontal"
    )

    axz1065 = fig.add_axes([0.1, 0.9, 0.8, 0.01])
    z1065_slider = Slider(
        ax=axz1065,
        label="z1065",
        valmin=0,
        valmax=10,
        valinit=0,
        orientation="horizontal"
    )

    def update_with_values(d):

        [ax.clear() for axr in axs for ax in axr]

        for txt in fig.texts:
            txt.set_visible(False)

        axs[0,0].set_axis_off()
        axs[0,1].set_axis_off()

        tweezer_params = {'817':{'wx0':0.885,
                            'wy0':0.985,
                            'wz0':0.995,
                            'power_mW':p817_slider.val,
                            'pol_Rb':4690,
                            'pol_Cs':-3253,
                            'pol_RbCs00':443,
                            'pol_RbCs33':443 - (1/3) * (-2816),
                            'x0':0,
                            'y0':0,
                            'z0':0},
                    '1065':{'wx0':1.05,
                            'wy0':1.16,
                            'wz0':1.19,
                            'power_mW':p1065_slider.val,
                            'pol_Rb':687,
                            'pol_Cs':1168,
                            'pol_RbCs00': 1825,
                            'pol_RbCs33': 1825 - (1/3) * 1981,
                            'x0':x1065_slider.val,
                            'y0':y1065_slider.val,
                            'z0':z1065_slider.val}}
        
        rb_min = my_find_potential_minimum((0,0,0),tweezer_params,"Rb")
        rbcs33_min = my_find_potential_minimum((0,0,0),tweezer_params,"RbCs33")
        real_both_min = [rb_min, rbcs33_min]
        print("Rb min", rb_min)
        print("Cs min", rbcs33_min)
        
        for expand_around,colour,fit_it in [(real_both_min,'blue',True)]: #([[0,0,0],[0,0,0]],'gray',False),
            fit_mins = [[0,0,0],[0,0,0]]
            for si in [0,1]:
                atom = ["Rb", "RbCs33"][si]
                this_expand_around = expand_around[si]
                
                x = np.linspace(-2,2,41) #np.concatenate(([0],np.linspace(-0.01,-2,20),np.linspace(0.01,2,20)))
                y = np.linspace(-2,2,42) #np.concatenate(([0],np.linspace(-0.01,-2,20),np.linspace(0.01,2,20)))
                z = np.linspace(-8,8,45)

                # x_mesh, y_mesh, z_mesh = np.meshgrid(x,y,z,indexing='ij')

                # tp_cs = total_potential((x_mesh,y_mesh,z_mesh), tweezer_params, "Cs")
                # tp_rb_z = total_potential((x_mesh,y_mesh,z_mesh),tweezer_params,"Rb")
                # tp_both = [tp_rb, tp_cs]

                for ci in [0,1,2]:
                    ax = axs[ci+1,si]
                    
                    axis_var = [x,y,z][ci]
                    axis_slice = tuple((this_expand_around[d] if d!=ci else axis_var for d in range(3)))
                    
                    tp = total_potential(axis_slice, tweezer_params, atom)
                    p_1065 = total_potential(axis_slice,{'1065':tweezer_params['1065']},atom)
                    p_817 = total_potential(axis_slice,{'817':tweezer_params['817']},atom)

                    xyzaxis = ["x","y","z"][ci]
                    # mesh = [x_mesh, y_mesh, z_mesh][ci]
                    # dimslice = tuple((((mesh.shape[d]-1)//2) if d!=ci else slice(None) for d in range(3)))

                    xs = axis_var
                    ys = tp

                    line_main = ax.plot(xs,ys,c=colour)
                    line_1065 = ax.plot(xs,p_1065,c='green',linestyle='--')
                    line_817 = ax.plot(xs,p_817,c='purple',linestyle='--')
                    
                    ax.set_xlabel(f"{xyzaxis} distance from 817 centre ($\mu$m)")
                    ax.set_ylabel(f"{atom} Trap Depth (MHz?)")
                    
                    if fit_it:
                        ax.axvline(this_expand_around[ci], lw=1, linestyle='--', c='k')
                        fit_threshold_um = 0.6
                        fit_y_ratio = 0.8
                        # try:
                        trap_freq, quad_popt = my_get_trap_frequency(xs,ys, masses[atom],x_fit_threshold_um=fit_threshold_um,y_fit_threshold_ratio=fit_y_ratio)
                        centre_position_um = quad_popt[0]
                        centre_intensity_min = quad_popt[2]
                        xs_fit = np.linspace(centre_position_um-fit_threshold_um,centre_position_um+fit_threshold_um,200)
                        ys_fit = quad(xs_fit, *quad_popt)
                        line_fit = ax.plot(xs_fit, ys_fit, lw=1, linestyle='-', c='red')
                        ax.axvline(centre_position_um, lw=1, linestyle='--', c='red')
                        fit_mins[si][ci] = centre_position_um

                        # beta = np.sqrt((masses[atom]*trap_freq/scipy.constants.hbar))

                        wf_space = np.linspace(-2,+2,100)
                        wf = (1/50)*(masses[atom]*u*2*np.pi*trap_freq/(np.pi*scipy.constants.hbar))**(0.25) * np.exp(-(masses[atom] * u * 2 * np.pi * trap_freq * 1e6 * ((wf_space-centre_position_um)*1e-6)**2)/(2*scipy.constants.hbar)) + centre_intensity_min
                        # except:
                        #     print("couldn't fit")
                        print(f"{atom} {xyzaxis} Trap Frequency {trap_freq*1e3}  (kHz)")
                        ax.plot(wf_space,wf, c='k',alpha=0.3)
                        ax.fill_between(wf_space,wf,y2=centre_intensity_min,color='k', alpha=0.15)
        grad_dec_dist_nm = np.sqrt(np.sum((np.asarray(rb_min)-np.asarray(rbcs33_min))**2))*1e3
        fit_dist_nm = np.sqrt(np.sum((np.asarray(fit_mins[0])-np.asarray(fit_mins[1]))**2))*1e3


        fig.text(0.5,0.8,f"GradDecDist: {grad_dec_dist_nm:.1f}nm , FitDist: {fit_dist_nm:.1f}nm",size=14,ha='center')
        fig.canvas.draw_idle()

    p817_slider.on_changed(update_with_values)
    p1065_slider.on_changed(update_with_values)
    x1065_slider.on_changed(update_with_values)
    y1065_slider.on_changed(update_with_values)
    z1065_slider.on_changed(update_with_values)

    update_with_values(0)
    
    plt.show()