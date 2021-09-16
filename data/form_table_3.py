from __future__ import division
import numpy as np
from numpy import genfromtxt
import csv
import os, sys
import h5py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


filename = 'Master_r_process_orbit_distGAIA_rvLITERATURE_30_cut.csv'


def get_column(N, Type, filename):
    return np.genfromtxt(filename, delimiter=',', dtype=Type, skip_header=1, usecols=N, comments=None)


def get_integral(Integral, order, showSign):
    Integral = Integral / (10**order)
    if showSign and Integral>0:
        Integral = '+'+format(Integral, '.2f')
    else:
        Integral = format(Integral, '.2f')
    return Integral

def get_else(Else):
    return format(Else, '.2f')


if __name__ == '__main__':
    Name = get_column(0, str, filename)[:]
    Energy = get_column(21, float, filename)[:]
    J_phi = get_column(13, float, filename)[:]
    Ecc = get_column(29, float, filename)[:]
    J_r = get_column(11, float, filename)[:]
    J_z = get_column(19, float, filename)[:]
    r_peri = get_column(23, float, filename)[:]
    r_apo = get_column(25, float, filename)[:]
    Z_max = get_column(27, float, filename)[:]

    savefile = open('table3.csv','wb')
    wr = csv.writer(savefile, quoting=csv.QUOTE_NONE)
    wr.writerow(['Name'] + ['Energy'] + ['Eccentricity'] + ['J_phi'] + ['J_r'] + ['J_z'] + ['r_peri'] + ['r_apo'] + ['Z_max'])
    wr.writerow([''] + [''] + [''] + [''] + [''] + [''] + [''] + [''] + [''])
    for i, n in enumerate(Name):
        wr.writerow([Name[i]] + [get_integral(Energy[i],5,False)] + [get_else(Ecc[i])] + [get_integral(J_phi[i],3,True)] + [get_integral(J_r[i],3,False)] + [get_integral(J_z[i],3,False)] + [get_else(r_peri[i])] + [get_else(r_apo[i])] + [get_else(Z_max[i])])
