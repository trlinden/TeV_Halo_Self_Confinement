import math
import sys


##Note: This code can be accelerated (somewhat significantly -- 10x-20x performance), by running on a CUDA enabled GPU with cupy enabled.
##If "import cupy" works on your machine, then this will transfer code execution to the GPU automatically, it's really that simple.
##The vesion of cupy must be relatively recent (I believe at least 9.2), as previous versions did not have some critical functions defined

try:
	import cupy as np
except:
	import numpy as np
	print("Running on CPU Only. GPU Acceleration not Enabled")
	pass

from scipy import integrate

##Notes:
####Units:
####Energy: GeV
####Simulation Distances: pc
####Simulation Time: s
####Particle Densities, as well as pulsar values: cm
####Time: s
####Magnetic Field: G
####ISRF Energy Density: eV/cm^3

####Note: We assume p_e = E_e for all electrons


##INPUT Parameters are:
##1 = Output Folder - this is a string to save the files into
##2 = Simulation dz, this is the stepsize of the simulation in pc
##3 = Simulation Momentum Bins. This is integer the number of logarithmically spaced bins between the minimum and maximum energies
##4 = Simulation timestep - This is the timestep in years. Note: The code lets you set anything, however for the code to converge
#######  The stepsize must be no more than ~ 0.05 yr / (z/1 pc)^2
##5 = Pulsar Efficiency = This is the fraction of the pulsar spindown power converted into e+e- pairs (1.0 = 100% efficiency)
##6 = Simulation Radius = This is the radial size of the flux tube, which just spreads out the particles in R^2, which is not
#######  Part of the 1D simulation. Thus, doubling the simulation radius can be identically offset by quadrupling the efficiency
##7 = Magnetic field strength (in Microgauss)
##8 = Neutral Gas Fraction in cm^-3. Setting this to 0 de-facto turns off ion-neutral damping, without affecting the rest of the code
##9 = Pulsar Momentum Spectrum. Note this is not the energy spectrum, and you must input the negative number.
####### e.g., "-3.5" will give you dN/dE = E^-1.5, inputing "3.5" or "-1.5" will give you a mess.
##10 = Simulation Starttime. If this is 0, the simulation starts at t=0, if it another number, the code tries to open the output
####### folder and open files called t_"starttime"_f_arr.npy and t_"starttime"_W_arr.npy, to continue a previously terminated
####### simulation from that time. If it cannot find those files, it will crash. If it runs, it will add new output files into the same
####### folder (potentially overwriting old versions of the output files)


##Constants
yr2sec = 31556926
speed_of_light = 2.997e10 ##speed of light (cm/s)
pc2cm = 3.08567758128e18 ##1 pc (cm)
kelvin2GeV = 8.617333262145e-14 ##1 Kelvin in eV
erg2GeV = 624.151 ##1 erg in GeV
mass_electron = 0.510998950e-3 ##mass of electron (GeV)
##Assigned Variables
speed_of_light_pc = speed_of_light / pc2cm

##Structure of bins
simulation_height = 500.0 ##Maximum height of the simulation (pc)
simulation_radius = float(sys.argv[6]) ##pc -- This is the radial size of the flux tube we are operating in
simulation_dz = float(sys.argv[2]) ##Simulation stepsize (pc)
simulation_sigma = 1.0 ##source size of the simulation (pc)
simulation_time = 1000000 * yr2sec ##simulation time (s)
simulation_timestep = float(sys.argv[4]) * yr2sec ##simulation_timestep (s)
simulation_momentum_logmin=0 ##Minimum Momentum in GV
simulation_momentum_logmax=6 ##Maximum Momentum in GV
simulation_momentum_bins = int(sys.argv[3]) ##Number of momentum bins per decade
simulation_background_turbulence = 'kolmogorov' ##Can be 'kolmogorov' (no caps) or 'kraichnan'

###Data Output Information
##We print out all relevant arrays at each time stamp given in this code
outfoldername=sys.argv[1] ##Folder we print out to
timestamps_start = int(100*yr2sec/simulation_timestep)
timestamps_mid=int(1000*yr2sec/simulation_timestep) ##Write an output every time the counter hits this, which is some number of years
timestamps_end = int(5000*yr2sec/simulation_timestep) 
timestamps_cut0 = 5000 * yr2sec / simulation_timestep
timestamps_cut1 = 100000 * yr2sec / simulation_timestep

##Background Model
magnetic_field_strength = float(sys.argv[7]) * 1.0e-6 ##in Gauss
magnetic_field_energy_density = (magnetic_field_strength*magnetic_field_strength/(8.0*np.pi))*erg2GeV*np.power(pc2cm, 3.0) ##energy density in GeV/pc^3 (which is dumb, i know)
magnetic_field_energy_density_eVcm3 = (magnetic_field_strength*magnetic_field_strength/(8.0*np.pi))*erg2GeV*1e9 ##energy density in eV/cm^3 , for the energy loss calculation

gas_density = 1.0 ##Neutral gas density in cm^-3
neutral_gas_fraction = float(sys.argv[8]) ##Neutral gas density in cm^-3
plasma_temperature = 1.0e4 ##Plasma Temperature Kelvin
UV_Energy_Density = 0.1 ##eV/cm^3
UV_Temperature = 20000.0 * kelvin2GeV ## GeV
OPT_Energy_Density = 0.6 ##eV/cm^3
OPT_Temperature = 5000.0 * kelvin2GeV ## GeV
IR_Energy_Density = 0.6 ##eV/cm^3
IR_Temperature = 20.0 * kelvin2GeV ## GeV
CMB_Energy_Density = 0.26 ##eV/cm^3
CMB_Temperature = 2.7 * kelvin2GeV ## GeV

initial_diffusion_constant = 3.466e28 / pc2cm / pc2cm ##Intitial diffusion constant at 1 GeV, in pc^2/s
initial_diffusion_spectrum = 0.33 ##Kolmogorov

###Some Derived Values
alfven_velocity = 2.18e11 * magnetic_field_strength / np.sqrt(gas_density) / pc2cm ##pc/s

##Pulsar Information
pulsar_efficiency = float(sys.argv[5]) ##Fraction of pulsar power that is converted into e+e- pairs  --TIM: 1.87/5.741 converts ot fit Payels code value
pulsar_source_size = 1.0
pulsar_spectrum=float(params[9]) ##pulsar spectrum in momentum (add 2 to get energy spectrum)
pulsar_cutoff = 1e5 ##pulsar spectral cutoff in GeV
pulsar_minimum_energy= 1.0 ##minimum energy in integral (GeV)
pulsar_maximum_energy = 1e7 ##maximum energy in integral (GeV)

def get_pulsar_initial_power():
	return (2.905e40 * pulsar_efficiency, 10000.0 * yr2sec) ##These are numbers to match Carmelo's current code, in GeV/s

def pulsar_normalization_integral():
    def __pulsar_normalization_integrand(x):
        return x**(3+pulsar_spectrum) * np.exp(-x/pulsar_cutoff) ##the energy integrand

    result = integrate.quad(__pulsar_normalization_integrand, pulsar_minimum_energy, pulsar_maximum_energy)
    return result[0] * mass_electron**(-1.0*pulsar_spectrum) ##These are the other normalizations in the Q0 term, from Payel notes an derived from Eq. 2

def __KN_Suppression(mometa, photon_energy): ##Called by energy_loss_rate
	term1 = (45.0 * mass_electron**2.0) / (64.0 * np.pi ** 2.0 * photon_energy**2.0)
	return term1 / (term1 + momenta ** 2.0 / mass_electron**2.0)

def get_energy_loss_rate(momenta): ##Energy Losses in GeV/s
	# Energy_loss_terms , taken from Hooper, Cholis, Linden, Fang equation (2.2)
	photon_energies = [CMB_Temperature, IR_Temperature, OPT_Temperature, UV_Temperature]
	CMB = __KN_Suppression(momenta, photon_energies[0])
	IR = __KN_Suppression(momenta, photon_energies[1])
	OPT = __KN_Suppression(momenta, photon_energies[2])
	UV = __KN_Suppression(momenta, photon_energies[3])

	loss_rate = 1.02e-16 * ((CMB_Energy_Density*CMB) + (IR_Energy_Density*IR) + (OPT_Energy_Density*OPT) + (UV_Energy_Density*UV) + magnetic_field_energy_density_eVcm3)
	momenta_loss_rate = - loss_rate * (momenta**2.0)   # Momentum loss rate : units = GeV/s
	return momenta_loss_rate

###Build the Grid that we are anayzing our functions on

##Build a 1D Momentum Grid
momentum_bins = np.power(10.0, np.linspace(simulation_momentum_logmin, simulation_momentum_logmax, simulation_momentum_bins+1))
momentum_bins_min = momentum_bins[0:-1]
momentum_bins_max = momentum_bins[1:]
#momenta = np.sqrt(momentum_bins_min*momentum_bins_max) ##logarithmic average of these values
momenta = momentum_bins
diff_momenta = momentum_bins_max - momentum_bins_min

##Build a 1D height Grid
spatial_bins = np.arange(-simulation_height, simulation_height + simulation_dz, simulation_dz)
spatial_bins_min = spatial_bins[0:-1]
spatial_bins_max = spatial_bins[1:]
spatial = spatial_bins ##In this version, we use a spatial setup that has a point 0
spatial_midpoint = int(np.floor(len(spatial_bins) / 2.0))
print(spatial_midpoint, spatial[spatial_midpoint])

spatial_bin_step = spatial_bins_max - spatial_bins_min ##In theory this code should also work for iterative grid sizes

##Build some 1D functions in momentum that don't change
larmor_radii = 333.6e4 * momenta / magnetic_field_strength / pc2cm ##pc
energy_loss_rates = get_energy_loss_rate(momenta)
alfven_vel = alfven_velocity * np.tanh(spatial/simulation_sigma) ##This becomes a 1D array that that reverts direction at the origin
bohm_diffusion = larmor_radii * speed_of_light_pc / 3.0  ##Minimum diffusion constant we can have


##Tile These into 2D arrays that fit our Model
larmor_radii = np.tile(larmor_radii, (len(spatial), 1))
energy_loss_rates = np.tile(energy_loss_rates, (len(spatial), 1))
alfven_vel = np.tile(alfven_vel, (len(momenta), 1)).T
bohm_diffusion = np.tile(bohm_diffusion, (len(spatial), 1))

##Build the numpy 2D arrays
##W_arr Units are: pc
##f Units are: 1 / GeV^3 / pc^3
W_arr = 4.0/(3.0 * np.pi) * speed_of_light_pc * larmor_radii**2.0 / (initial_diffusion_constant * momenta ** initial_diffusion_spectrum)
W_init = np.copy(W_arr) ##Make a copy
f_arr = np.zeros(W_arr.shape)

##Set the Gamma_NLD_inital_Value_Array
if(simulation_background_turbulence=='kolmogorov'):
        Gamma_NLD_init = 0.052 * np.abs(alfven_vel) / np.power(larmor_radii, 3.0/2.0) * np.power(W_arr, 1.0/2.0)
elif(simulation_background_turbulence=='kraichnan'):
        Gamma_NLD_init = 0.052 * np.abs(alfven_vel) / np.power(larmor_radii, 2.0) * W_arr
else:
	raise ValueError('Turbulence spectrum not defined, breaking')


print(np.shape(f_arr), len(momenta))

##Source_term
(pulsar_power_value, spindown_timescale) = get_pulsar_initial_power()
pulsar_normalization = pulsar_normalization_integral()

##New Definition for the source term based on Payel e-mail 07/07/21
source_term_energy = 1.0/(4.0*np.pi) * (pulsar_power_value * (momenta/mass_electron)**(pulsar_spectrum) * np.exp(-momenta/pulsar_cutoff)) / pulsar_normalization
source_term_spatial = (1.0/simulation_radius)**2.0 * np.exp(-(spatial*spatial)/(2.0 * pulsar_source_size **2.0)) / ( (2.0 * np.pi**3.0 )**(0.5) * pulsar_source_size)



source_term = np.outer(source_term_spatial, source_term_energy)

###Now the functional terms that do change in each iteration stepsize

def __centered_differential_gradient(in_array, in_axis):
	central_gradient = np.gradient(in_array, axis=0)
	central_gradient[spatial_midpoint, :] = (in_array[spatial_midpoint+1,:]  - in_array[spatial_midpoint,:]) ##This calculates a forward derivative for the central point
	return central_gradient

def __centered_differential(in_array, in_axis):
	return __centered_differential_gradient(in_array, in_axis)

def __pulsar_source(time):
    ##Units should be: 1.0 / GeV^3 / pc^3 / s
    return source_term * (1.0+time/spindown_timescale) ** -2.0

def __diffusion_coefficient(W_arr):
    ##Units are: pc^2/s from Equation 6 of 1807 paper
    in_diffusion_coefficient = 4.0/(3.0 * np.pi) * speed_of_light_pc * larmor_radii**2.0 / W_arr ##We include a minimum at the bohm diffusion coefficient
    in_diffusion_coefficient_bohm_minimum = np.maximum(in_diffusion_coefficient, bohm_diffusion)
    return in_diffusion_coefficient_bohm_minimum

def __alfven_term(f_arr, dfdz):
    ##Units are 1 / pc^3 / GeV^3
    return dfdz * alfven_vel

def __diffusion_term(f_arr, W_arr, dfdz, d2fdz2):
    ##Units are: 1 / pc^3 / GeV^3
    in_result = np.zeros(np.shape(f_arr))
    diff_constant = __diffusion_coefficient(W_arr)

    D_mid1 = ((diff_constant[2:,:] + diff_constant[1:-1,:]) / 2.0)
    D_mid2 = ((diff_constant[1:-1,:] + diff_constant[0:-2,:]) / 2.0)
    in_result[1:-1,:] = ( D_mid1 * (f_arr[2:,:] - f_arr[1:-1,:]) + D_mid2 * (f_arr[0:-2,:] - f_arr[1:-1,:]) ) / simulation_dz**2.0
    in_result[0,:] = 0.0
    in_result[-1,:] = 0.0
    in_result[spatial_midpoint,:] = ( (diff_constant[spatial_midpoint+1,:]-diff_constant[spatial_midpoint,:]) * (f_arr[spatial_midpoint+1,:] - f_arr[spatial_midpoint,:]) + diff_constant[spatial_midpoint,:] * d2fdz2[spatial_midpoint, :] ) / simulation_dz**2.0 ##This is a forward derivative for the central point specifically
    return in_result

def __loss_term(f_arr, W_arr):
    ##Inner Units are: GeV * GeV * GeV/s 1 / GeV ^3 / pc^3 = 1 / pc^3 / s
    ##Outer Units take this to become: 1 / GeV^2 / GeV -> 1 / GeV^3 / pc^3 / s
    inner_term = np.zeros(np.shape(f_arr))
    outer_term = np.zeros(np.shape(f_arr))

    inner_term[1:-1,:-1] = momenta[:-1] * momenta[:-1] * energy_loss_rates[1:-1,:-1] * f_arr[1:-1,:-1]
    inner_term[1:-1,-1] = 0.0

    outer_term[1:-1,:-1] = 1.0/momenta[:-1]/momenta[:-1] * (inner_term[1:-1,1:] - inner_term[1:-1,:-1]) / diff_momenta 
    outer_term[1:-1, -1] = outer_term[1:-1, -1] ##No energy losses on the upper boundary, which prevents negative values from propagating. There should be no particles here anyway
    return outer_term

def __alfven_derivative(f_arr, W_arr, alfven_vel):
    derivative_alfven_vel = __centered_differential(alfven_vel, 0) / simulation_dz
    derivative_f_momentum = np.gradient(f_arr, momenta, axis=1)
    return 1.0/3.0 * momenta * derivative_alfven_vel * derivative_f_momentum

def __gamma_CR(f_arr, W_arr, dfdz):
    ##Equation 29 from 1807.09263 -- note that larmor_radii = 1/k, and since everything is in pc, magnetic field energy density must be converted oddly
    in_CR_result =  np.abs(2.0*np.pi / 3.0 * alfven_vel * larmor_radii / W_arr / magnetic_field_energy_density * np.power(momenta, 4.0) * dfdz) ##why the FABS?????????
    return in_CR_result

def __gamma_NLD(f_arr, W_arr):
    gamma_NLD = 0.0
    gamma_IND = 0.0
    ##Equation 8 from 1807.09263 for Kolmogorov Turbulence
    if(simulation_background_turbulence=='kolmogorov'):
        gamma_NLD = 0.052 * np.abs(alfven_vel) / np.power(larmor_radii, 3.0/2.0) * np.power(W_arr, 1.0/2.0)
    elif(simulation_background_turbulence=='kraichnan'):
        gamma_NLD =  0.052 * np.abs(alfven_vel) / np.power(larmor_radii, 2.0) * W_arr
    else:
        raise ValueError('The NLD Turbulence Model is Not Defined.... Breaking....')
        exit()
    vin = 8.4e-9*(gas_density*neutral_gas_fraction) * (plasma_temperature/1.0e4) ** 0.4 ## in inverse seconds
    omega_k = (alfven_vel / larmor_radii) ##From equation 13 in 1710.10937 ##cm/s / cm == s^-1
    gamma_IND = vin/2.0 * (omega_k**2.0) / (omega_k**2.0 + (vin/neutral_gas_fraction)**2.0) ## s^-1 * s^-2 / s^-2 
    return (gamma_NLD + gamma_IND)

def coupled_ode(time, arguments, args):
    f_arr = arguments[0]
    W_arr = arguments[1]
    dt=args[0]
    counter=args[1]
    dfdz = __centered_differential(f_arr, 0) / simulation_dz
    appendval = np.zeros((2,len(momenta)))
    d2fdz2 = np.diff(np.diff(f_arr, axis=0, prepend=0), axis=0, append=0)##I don't use the centered diff here, because this explicitly matches Payel's method -- we divide by simulation_dz later, so don't need to do it here

    in_diffusion_term = __diffusion_term(f_arr, W_arr, dfdz, d2fdz2)
    in_alfven_term = __alfven_term(f_arr, dfdz)
    in_loss_term = __loss_term(f_arr, W_arr)
    
    in_pulsar_source_term = __pulsar_source(time)
    in_alfven_derivative = __alfven_derivative(f_arr, W_arr, alfven_vel)

    new_f = f_arr + dt * ( in_pulsar_source_term + in_diffusion_term - in_alfven_term - in_loss_term + in_alfven_derivative) ##Equation 5

    ##Set boundary conditions:
    new_f[0,:] = 0.0 ##Boundary conditions at +/- boundaries of diffusion region
    new_f[-1,:] = 0.0

    ###Compute the W terms
    in_gamma_CR = __gamma_CR(f_arr, W_arr, dfdz)
    in_gamma_NLD = __gamma_NLD(f_arr, W_arr)

    ##Equation 7 from Evoli et al.
    new_W = W_arr + dt * ( (in_gamma_CR - in_gamma_NLD) * W_arr + Gamma_NLD_init * W_init - alfven_vel * __centered_differential(W_arr, 0) / simulation_dz )

    new_W[0,:] = W_init[0,:] ##Reset the conditions at the boundary to ISM turbulence
    new_W[-1,:] = W_init[-1,:]

    diffusion_constant_printout = __diffusion_coefficient(new_W)*pc2cm*pc2cm

    if(counter % timestamps_start == 0 and counter < timestamps_cut0 or counter % timestamps_mid == 0 and counter < timestamps_cut1 or counter % timestamps_end == 0): ##Print all information from this timestamp
    	yrval=round(time/yr2sec)
    	np.save(outfoldername + '/t_' + str(yrval) + '_f_arr.npy', new_f)
    	np.save(outfoldername + '/t_' + str(yrval) + '_W_arr.npy', new_W)
    	np.save(outfoldername + '/t_' + str(yrval) + '_Gamma_CR.npy', in_gamma_CR)
    	np.save(outfoldername + '/t_' + str(yrval) + '_Gamma_NLD.npy', in_gamma_CR)
    	np.save(outfoldername + '/t_' + str(yrval) + '_f_Diffusion.npy', in_diffusion_term)
    	np.save(outfoldername + '/t_' + str(yrval) + '_f_Alfven.npy', in_alfven_term)
    	np.save(outfoldername + '/t_' + str(yrval) + '_f_Loss.npy', in_loss_term)
    	np.save(outfoldername + '/t_' + str(yrval) + '_f_Source.npy', in_pulsar_source_term)
    	np.save(outfoldername + '/t_' + str(yrval) + '_f_AlfvenDerivative.npy', in_alfven_derivative)
    	np.save(outfoldername + '/t_' + str(yrval) + '_DiffusionCoefficient.npy', diffusion_constant_printout)
    return [new_f, new_W]



####Main Function here:
starttime = int(sys.argv[9])
counter = int(starttime*yr2sec/timestamps_start)

tval=starttime*yr2sec
if(tval != 0):
	f_arr = np.load(outfoldername + '/t_' + str(starttime) + '_f_arr.npy')
	W_arr = np.load(outfoldername + '/t_' + str(starttime) + '_W_arr.npy')

while(tval < simulation_time):
    [f_arr, W_arr] = coupled_ode(tval, [f_arr, W_arr], [simulation_timestep, counter])
    check_diffusion_constant_at_10 = __diffusion_coefficient(W_arr)
    tval += simulation_timestep
    counter += 1
