from __future__ import division
import numpy as np
from numpy import genfromtxt
import csv
import os, sys
import h5py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from astropy.stats import biweight_scale, biweight_location
import clusters
import params


filename_1 = params.init_file
filename_2 = params.kin_file

CDTG_assoc = [([1,7,9,10,12,13,17,19,20,21,22], 'Gaia-Sausage', 0), ([26,30], 'Sequoia', 0), ([2,27], 'GL20:Thamnos 1', 0), ([3,5,8], 'Thick disk', 0), ([15], 'ZY20:DTG-3', 1), ([1], 'ZY20:DTG-7', 1), ([12], 'ZY20:DTG-16', 1), ([28], 'ZY20:DTG-46', 1), ([4], 'GL20:DTG-2', 3), ([15], 'GL20:DTG-3', 3), ([28], 'GL20:DTG-7', 2), ([13], 'GL20:DTG-30', 2), ([11,23], 'IR18:Group A', 2), ([13], 'IR18:Group B', 1), ([16], 'IR18:Group C', 2), ([10], 'IR18:Group D', 1), ([1], 'IR18:Group E', 2), ([22], 'IR18:Group F', 1), ([19], 'IR18:Group H', 1)]


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
    J_r = get_column(11, float, filename_2)[:]
    J_z = get_column(19, float, filename_2)[:]
    v_r = get_column(1, float, filename_2)[:]
    v_phi = -get_column(3, float, filename_2)[:]
    v_z = get_column(9, float, filename_2)[:]
    Clusters = clusters.final_clusters

    
    savefile = open('table6.csv','wb')
    wr = csv.writer(savefile, delimiter = ';', quoting=csv.QUOTE_NONE, quotechar='')
    wr.writerow(['Substructure (n_sub)'] + ['Groups'] + ['Component'] + ['(<v_r>, <v_phi>, <v_z>)'] + ['(<J_r>, <J_phi>, <J_z>)'] + ['<E>'])
    wr.writerow([''] + [''] + [''] + [''] + [''] + [''])
    for cdtg_assoc in CDTG_assoc:
        star_indeces = [c for clus in cdtg_assoc[0] for c in Clusters[clus-1]]
        
        CDTG_string = 'CDTG-'
        for c in cdtg_assoc[0]:
            CDTG_string = CDTG_string + str(c) + ','
        CDTG_string = CDTG_string[:-1]

        Name_string = cdtg_assoc[1]
        if cdtg_assoc[2]!=0:
            Name_string = Name_string + ' (' + str(cdtg_assoc[2]) + ')'

        if len(star_indeces)>3:
            v_string = '(' + get_else(biweight_location(v_r[star_indeces])) + ', '
            v_string = v_string + get_else(biweight_location(v_phi[star_indeces])) + ', '
            v_string = v_string  + get_else(biweight_location(v_z[star_indeces])) + ')'

            J_string = '(' + get_integral(biweight_location(J_r[star_indeces]), 3, False) + ', '
            J_string = J_string + get_integral(biweight_location(J_phi[star_indeces]), 3, True) + ', '
            J_string = J_string  + get_integral(biweight_location(J_z[star_indeces]), 3, False) + ')'

            E_string = get_integral(biweight_location(Energy[star_indeces]), 5, True)

            wr.writerow([Name_string] + [CDTG_string] + [''] + [v_string] + [J_string] + [E_string])

            v_string = '(' + get_else(biweight_scale(v_r[star_indeces])) + ', '
            v_string = v_string + get_else(biweight_scale(v_phi[star_indeces])) + ', '
            v_string = v_string  + get_else(biweight_scale(v_z[star_indeces])) + ')'

            J_string = '(' + get_integral(biweight_scale(J_r[star_indeces]), 3, False) + ', '
            J_string = J_string + get_integral(biweight_scale(J_phi[star_indeces]), 3, False) + ', '
            J_string = J_string  + get_integral(biweight_scale(J_z[star_indeces]), 3, False) + ')'

            E_string = get_integral(biweight_scale(Energy[star_indeces]), 5, True)
      
            wr.writerow([''] + [''] + [''] + [v_string] + [J_string] + [E_string])


        else:
            v_string = '(' + get_else(np.mean(v_r[star_indeces])) + ', '
            v_string = v_string + get_else(np.mean(v_phi[star_indeces])) + ', '
            v_string = v_string  + get_else(np.mean(v_z[star_indeces])) + ')'

            J_string = '(' + get_integral(np.mean(J_r[star_indeces]), 3, False) + ', '
            J_string = J_string + get_integral(np.mean(J_phi[star_indeces]), 3, True) + ', '
            J_string = J_string  + get_integral(np.mean(J_z[star_indeces]), 3, False) + ')'

            E_string = get_integral(np.mean(Energy[star_indeces]), 5, True)

            wr.writerow([Name_string] + [CDTG_string] + [''] + [v_string] + [J_string] + [E_string])

            v_string = '(' + get_else(np.std(v_r[star_indeces])) + ', '
            v_string = v_string + get_else(np.std(v_phi[star_indeces])) + ', '
            v_string = v_string  + get_else(np.std(v_z[star_indeces])) + ')'

            J_string = '(' + get_integral(np.std(J_r[star_indeces]), 3, False) + ', '
            J_string = J_string + get_integral(np.std(J_phi[star_indeces]), 3, False) + ', '
            J_string = J_string  + get_integral(np.std(J_z[star_indeces]), 3, False) + ')'

            E_string = get_integral(np.std(Energy[star_indeces]), 5, True)
      
            wr.writerow([''] + [''] + [''] + [v_string] + [J_string] + [E_string])
