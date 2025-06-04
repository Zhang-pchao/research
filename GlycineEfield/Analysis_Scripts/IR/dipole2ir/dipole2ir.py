import numpy as np
from typing import Optional
import time, matplotlib.pyplot as plt, numpy as np
from spectra_flow.ir_flow.cal_ir import calculate_corr_vdipole,calculate_ir

#set bigger font sizes
SMALL_SIZE = 16
MEDIUM_SIZE = 16
BIG_SIZE = 20
plt.rc('font', size=SMALL_SIZE)        # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)   # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE-3)  # legend fontsize
plt.rc('figure', titlesize=BIG_SIZE)   # fontsize of the figure title

def get_ir_fig(total_dipole,win=10000,wid=240.,temp=300.,dt=0.0003):
    corr = calculate_corr_vdipole(total_dipole, dt_ps=dt, window=win)
    ir_g = calculate_ir(corr, width=wid, dt_ps=dt, temperature=temp,filter_type="gaussian")
    ir_l = calculate_ir(corr, width=wid, dt_ps=dt, temperature=temp,filter_type="lorenz")
    plt.plot(ir_g[:, 0], ir_g[:, 1] / 1000, label = r'calculated, gaussian filter', scalex = 1.5,scaley= 2.2)
    plt.plot(ir_l[:, 0], ir_l[:, 1] / 1000, label = r'calculated, lorenz filter', scalex = 1.5, scaley= 2.2)
    plt.xlim((0, 4000.))
    #plt.ylim((0, 15))
    plt.xlabel(r'Wavenumber($\rm cm^{-1}$)', fontdict = {'size': 12})
    plt.ylabel(r'$n(\omega)\alpha(\omega) (10^3 cm^{-1})$', fontdict = {'size': 12})
    plt.legend()
    plt.show()

def get_ir(total_dipole,win=10000,wid=240.,temp=300.,dt=0.0003):
    corr = calculate_corr_vdipole(total_dipole, dt_ps=dt, window=win)
    ir_g = calculate_ir(corr, width=wid, dt_ps=dt, temperature=temp,filter_type="gaussian")
    #ir_l = calculate_ir(corr, width=wid, dt_ps=dt, temperature=temp,filter_type="lorenz")
    return ir_g[:, 0], ir_g[:, 1] / 1000


#zwitterion
total_dipole_z = np.load("/home/pengchao/glycine/dpmd/glycine/046_dpmd/002_128w_gly_bulk/1-z-ir/3/IR/total_gly_dipole.npy")
x_z,y_z = get_ir(total_dipole_z[180000:],wid=200.)
#neutral
total_dipole_n = np.load("/home/pengchao/glycine/dpmd/glycine/046_dpmd/002_128w_gly_bulk/1-n-ir/3/IR/total_gly_dipole.npy")
x_n,y_n = get_ir(total_dipole_n[180000:],wid=200.)
#h2o
total_dipole_w = np.load("/home/pengchao/glycine/dpmd/glycine/045_dpmd/002_128w_bulk/1-ir/IR/total_dipole.npy")
x_w,y_w = get_ir(total_dipole_w[180000:],wid=200.)

fig = plt.figure(figsize=(8,4), dpi=150, facecolor='white')
ax = fig.add_subplot(111)

x_values = [3170, 2898, 1615, 1510, 1412, 1332, 1111, 1032, 893, 697, 608, 503]
for i, x in enumerate(x_values):
    if i == len(x_values) - 1:
        ax.axvline(x, color='grey', lw=1, label='Expt. (Rai et al, 2005)')
    else:
        ax.axvline(x, color='grey', lw=1)

x_values = [3020, 2860, 1725, 1410, 1295]
for i, x in enumerate(x_values):
    if i == len(x_values) - 1:
        ax.axvline(x, color='grey', lw=1, ls='--', label='Expt. (Maes et al, 2004)')
    else:
        ax.axvline(x, color='grey', lw=1, ls='--')

ax.plot(x_z,y_z, label = r'[Z] (in water)', )
ax.plot(x_n,y_n, label = r'[N] (in water)', )
ax.plot(x_w,y_w, label = r'Pure water',   ls='--')
ax.set_xlim((0, 4000.))
#ax.set_ylim((0, 15))
ax.set_xlabel(r'Wavenumber($\rm cm^{-1}$)')
ax.set_ylabel(r'$n(\omega)\alpha(\omega) (10^3 cm^{-1})$')
ax.legend()