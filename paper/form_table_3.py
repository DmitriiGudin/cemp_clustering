from __future__ import division
import numpy as np
from numpy import genfromtxt
import csv
import os, sys
import h5py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import clusters
import params

filename_1 = params.init_file
filename_2 = params.kin_file

distance_error_mark_points = [20, 30] # In percent.
distance_error_marks = ['*', '**']


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
    Name = get_column(0, str, filename_2)[:]
    Energy = get_column(21, float, filename_2)[:]
    J_phi = -get_column(13, float, filename_2)[:]
    Ecc = get_column(29, float, filename_2)[:]
    J_r = get_column(11, float, filename_2)[:]
    J_z = get_column(19, float, filename_2)[:]
    r_peri = get_column(23, float, filename_2)[:]
    r_apo = get_column(25, float, filename_2)[:]
    Z_max = get_column(27, float, filename_2)[:]
    dist = get_column(3, float, filename_1)[:]
    dist_err = get_column(4, float, filename_1)[:]

    Error_marks = []
    for d, de in zip(dist, dist_err):
        error_mark=''
        for mp, m in zip (distance_error_mark_points, distance_error_marks):
            if abs(de/d) > mp/100:
                error_mark = m
        Error_marks.append(error_mark)

    savefile = open('table3.csv','wb')
    wr = csv.writer(savefile, quoting=csv.QUOTE_NONE)
    wr.writerow(['Name'] + ['Energy'] + ['Eccentricity'] + ['J_r'] + ['J_phi'] + ['J_z'] + ['r_peri'] + ['r_apo'] + ['Z_max'])
    wr.writerow([''] + [''] + [''] + [''] + [''] + [''] + [''] + [''] + [''])
    for i, n in enumerate(Name):
        wr.writerow([Name[i] + Error_marks[i]] + [get_integral(Energy[i],5,False)] + [get_else(Ecc[i])] + [get_integral(J_r[i],3,False)] + [get_integral(J_phi[i],3,True)] + [get_integral(J_z[i],3,False)] + [get_else(r_peri[i])] + [get_else(r_apo[i])] + [get_else(Z_max[i])])
