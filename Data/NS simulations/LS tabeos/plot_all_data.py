import matplotlib
import matplotlib.pyplot as plt
import pylab # to show the plot
import pandas as pd

import yt
import numpy as np

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],'size':15})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)
from matplotlib import rc_context
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

#define the parameters to transform the code unit to cgs unit.
ggrav = 6.673e-8
clite = 2.99792458e10
clite_g = 2.99792458e10

rho_gf = 1.61930347e-18
press_gf = 1.80171810e-39
eps_gf = 1.11265006e-21
time_gf = 2.03001708e05
mass_gf = 5.02765209e-34
length_gf = 6.77140812e-06*1.0e5 # to km
energy_gf = 5.59424238e-55
lum_gf = 2.7556091e-60

mev_to_erg = 1.60217733e-6
erg_to_mev = 6.24150636e5
amu_cgs = 1.66053873e-24
massn_cgs = 1.674927211e-24
amu_mev = 931.49432e0
kb_erg = 1.380658e-16
kb_mev = 8.61738568e-11
temp_mev_to_kelvin = 1.1604447522806e10 * 1.e-9 ##
planck = 6.626176e-27
avo = 6.0221367e23
hbarc_mevcm = 1.97326966e-11

def gamma(field, data):
    return  np.sqrt( 1.0 + data['psi']**4*(data['W_vel1']**2) )
def veloc1(field, data):
    return data['W_vel1'] / data['gamma']

path='./'

# Load the dataset
ds0 = yt.load(path+'output0000.dat', geometry_override='spherical', unit_system='code')
ds0.add_field( ('amrvac','gamma'), function=gamma, sampling_type='cell')
ds0.add_field( ('amrvac','veloc1'), function=veloc1, sampling_type='cell')

ds = yt.load(path+'output0004.dat', geometry_override='spherical', unit_system='code')
ds.add_field( ('amrvac','gamma'), function=gamma, sampling_type='cell')
ds.add_field( ('amrvac','veloc1'), function=veloc1, sampling_type='cell')

# cutting the x-axis through the y=0,z=0 
plotdata0 = ds0.ortho_ray( 0, (0, 0) )
plotdata = ds.ortho_ray( 0, (0, 0) )

# Sort the ray values by 'x' so there are no discontinuities in the line plot
srt0 = np.argsort(plotdata0['r'])
srt = np.argsort(plotdata['r'])

x_min = min(plotdata0['r'][0], plotdata['r'][0])/length_gf
x_max = max(plotdata0['r'][-1], plotdata['r'][-1])/length_gf

#print (ds.field_list)
#print (ds.derived_field_list)

# plot the data
fig, axs = plt.subplots(4,2, sharex = 'col',figsize=(12,10))

# first col
axs[0,0].plot(np.array(plotdata0['r'][srt0])/length_gf, np.array(plotdata0['rho'][srt0])/ rho_gf, label='$t=%.2f \\rm{ms}$' %(float(ds0.current_time)*1.e3/time_gf) )
axs[0,0].plot(np.array(plotdata['r'][srt])/length_gf, np.array(plotdata['rho'][srt])/ rho_gf, label='$t=%.2f \\rm{ms}$' %(float(ds.current_time)*1.e3/time_gf) )
axs[0,0].set_ylabel('$\\rho$ $ \\rm{[ g / cm^3 ]}$',rotation=90)
axs[0,0].grid(True)
axs[0,0].legend(loc='best')
axs[0,0].set_yscale('log')

axs[1,0].plot(np.array(plotdata0['r'][srt0])/length_gf, np.array(plotdata0['veloc1'][srt0]))
axs[1,0].plot(np.array(plotdata['r'][srt])/length_gf, np.array(plotdata['veloc1'][srt]))
axs[1,0].set_ylabel('$ v^r / c $',rotation=90)
axs[1,0].grid(True)

axs[2,0].plot(np.array(plotdata0['r'][srt0])/length_gf, np.array(plotdata0['eps'][srt0])/eps_gf)
axs[2,0].plot(np.array(plotdata['r'][srt])/length_gf, np.array(plotdata['eps'][srt])/eps_gf)
axs[2,0].grid(True)
axs[2,0].set_ylabel('$\\epsilon$  $ [ \\rm{erg} / \\rm{g} ]$',rotation=90)
axs[2,0].set_yscale('log')

# temp, in K
#axs[3,0].plot(np.array(plotdata0['r'][srt0])/length_gf, np.array(np.exp(plotdata0['logtemp'][srt0])) * temp_mev_to_kelvin)
#axs[3,0].plot(np.array(plotdata['r'][srt])/length_gf, np.array(np.exp(plotdata['logtemp'][srt])) * temp_mev_to_kelvin)
#axs[3,0].set_ylabel('$T$ $ [ 10^9 \\rm{K} ]$',rotation=90)
#axs[3,0].grid(True)

# temp, in MeV
axs[3,0].plot(np.array(plotdata0['r'][srt0])/length_gf, np.array(np.exp(plotdata0['logtemp'][srt0])))
axs[3,0].plot(np.array(plotdata['r'][srt])/length_gf, np.array(np.exp(plotdata['logtemp'][srt])))
axs[3,0].set_ylabel('$T$ $ [ \\rm{MeV} ]$',rotation=90)
axs[3,0].grid(True)

axs[3,0].set_xlim(x_min, x_max)
axs[3,0].set_xscale('log')
axs[3,0].set_xlabel('$r$ [km]')

# second col
axs[0,1].plot(np.array(plotdata0['r'][srt0])/length_gf, np.array(plotdata0['alp'][srt0]))
axs[0,1].plot(np.array(plotdata['r'][srt])/length_gf, np.array(plotdata['alp'][srt]))
axs[0,1].set_ylabel('$ \\alpha$',rotation=90)
axs[0,1].grid(True)

axs[1,1].plot(np.array(plotdata0['r'][srt0])/length_gf, np.array(plotdata0['psi'][srt0]))
axs[1,1].plot(np.array(plotdata['r'][srt])/length_gf, np.array(plotdata['psi'][srt]))
axs[1,1].set_ylabel('$ \\psi$',rotation=90)
axs[1,1].grid(True)

axs[2,1].plot(np.array(plotdata0['r'][srt0])/length_gf, np.array(plotdata0['grid_level'][srt0]))
axs[2,1].plot(np.array(plotdata['r'][srt])/length_gf, np.array(plotdata['grid_level'][srt]))
axs[2,1].grid(True)
axs[2,1].set_ylabel('grid level',rotation=90)

axs[3,1].plot(np.array(plotdata0['r'][srt0])/length_gf, np.array(plotdata0['ye'][srt0]))
axs[3,1].plot(np.array(plotdata['r'][srt])/length_gf, np.array(plotdata['ye'][srt]))
axs[3,1].set_ylabel('$Y_e$',rotation=90)
axs[3,1].grid(True)

axs[3,1].set_xlim(x_min, x_max)
axs[3,1].set_xscale('log')
axs[3,1].set_xlabel('$r$ [km]')


# Save the line plot
plt.tight_layout()
#plt.subplots_adjust(hspace=0.05, wspace=0.1)
plt.subplots_adjust(hspace=0.05)
#plt.savefig('hydro.png',papertype='a0', dpi=200)
plt.savefig('hydro.png',papertype='a0')
plt.show()

