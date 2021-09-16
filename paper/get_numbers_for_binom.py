from __future__ import division
import numpy as np
import h5py
import csv
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import params
from astropy.stats import biweight_scale


clusters_file = 'table4.csv'


def get_column(N, Type, filename):
    return np.genfromtxt(filename, delimiter=',', dtype=Type, skip_header=0, usecols=N, comments=None)


if __name__ == '__main__':
    First_Column = get_column(0, str, clusters_file)[:]
    clusters_list = []
    current_list = []
    for i, n in enumerate(First_Column):
        if not (n=='Name') and not (n=='Dispersion') and not (n=='Cumulative fraction values') and not (n=='') and not (n=='avg:') and not (n=='std:') and not (n[:7]=='Cluster'):
            current_list.append(i)
        elif len(current_list)>0:
            clusters_list.append(current_list)
            current_list = []
    clusters_list=clusters_list[:25]

    FeH = get_column(2, float, clusters_file)[:]
    CFe = get_column(4, float, clusters_file)[:]
    SrFe = get_column(5, float, clusters_file)[:]
    BaFe = get_column(6, float, clusters_file)[:]
    EuFe = get_column(7, float, clusters_file)[:]

    FeH_fractions, CFe_fractions, SrFe_fractions, BaFe_fractions, EuFe_fractions = [], [], [], [], []

    for Cluster in clusters_list:
        feh = FeH[Cluster]
        feh = feh[~np.isnan(feh)]
        filename = params.dispersion_distrib_file_mask[0] + 'FeH' + params.dispersion_distrib_file_mask[1]
    	f = h5py.File(filename, 'r')
        Dispersion_array = f["/dispersion"][len(Cluster)-params.dispersion_cluster_size[0]]
    	N_samples = len(Dispersion_array)
        if len(feh)==3:
            FeH_fractions.append(len(Dispersion_array[Dispersion_array<np.std(feh)])/len(Dispersion_array))
        elif len(feh)>3:
            FeH_fractions.append(len(Dispersion_array[Dispersion_array<biweight_scale(feh)])/len(Dispersion_array))

    for Cluster in clusters_list:
        cfe = CFe[Cluster]
        cfe = cfe[~np.isnan(cfe)]
        filename = params.dispersion_distrib_file_mask[0] + 'cCFe' + params.dispersion_distrib_file_mask[1]
    	f = h5py.File(filename, 'r')
        Dispersion_array = f["/dispersion"][len(Cluster)-params.dispersion_cluster_size[0]]
    	N_samples = len(Dispersion_array)
        if len(cfe)==3:
            CFe_fractions.append(len(Dispersion_array[Dispersion_array<np.std(cfe)])/len(Dispersion_array))
        elif len(cfe)>3:
            CFe_fractions.append(len(Dispersion_array[Dispersion_array<biweight_scale(cfe)])/len(Dispersion_array))

    for Cluster in clusters_list:
        srfe = SrFe[Cluster]
        srfe = srfe[~np.isnan(srfe)]
        filename = params.dispersion_distrib_file_mask[0] + 'SrFe' + params.dispersion_distrib_file_mask[1]
    	f = h5py.File(filename, 'r')
        Dispersion_array = f["/dispersion"][len(Cluster)-params.dispersion_cluster_size[0]]
    	N_samples = len(Dispersion_array)
        if len(srfe)==3:
            SrFe_fractions.append(len(Dispersion_array[Dispersion_array<np.std(srfe)])/len(Dispersion_array))
        elif len(srfe)>3:
            SrFe_fractions.append(len(Dispersion_array[Dispersion_array<biweight_scale(srfe)])/len(Dispersion_array))

    for Cluster in clusters_list:
        bafe = BaFe[Cluster]
        bafe = bafe[~np.isnan(bafe)]
        filename = params.dispersion_distrib_file_mask[0] + 'BaFe' + params.dispersion_distrib_file_mask[1]
    	f = h5py.File(filename, 'r')
        Dispersion_array = f["/dispersion"][len(Cluster)-params.dispersion_cluster_size[0]]
    	N_samples = len(Dispersion_array)
        if len(bafe)==3:
            BaFe_fractions.append(len(Dispersion_array[Dispersion_array<np.std(bafe)])/len(Dispersion_array))
        elif len(bafe)>3:
            BaFe_fractions.append(len(Dispersion_array[Dispersion_array<biweight_scale(bafe)])/len(Dispersion_array))

    for Cluster in clusters_list:
        eufe = EuFe[Cluster]
        eufe = eufe[~np.isnan(eufe)]
        filename = params.dispersion_distrib_file_mask[0] + 'EuFe' + params.dispersion_distrib_file_mask[1]
    	f = h5py.File(filename, 'r')
        Dispersion_array = f["/dispersion"][len(Cluster)-params.dispersion_cluster_size[0]]
    	N_samples = len(Dispersion_array)
        if len(eufe)==3:
            EuFe_fractions.append(len(Dispersion_array[Dispersion_array<np.std(eufe)])/len(Dispersion_array))
        elif len(eufe)>3:
            EuFe_fractions.append(len(Dispersion_array[Dispersion_array<biweight_scale(eufe)])/len(Dispersion_array))

    FeH_fractions, CFe_fractions, SrFe_fractions, BaFe_fractions, EuFe_fractions = np.array(FeH_fractions), np.array(CFe_fractions), np.array(SrFe_fractions), np.array(BaFe_fractions), np.array(EuFe_fractions)

    savefile = open('HDBSCAN_test.csv','wb')
    wr = csv.writer(savefile, quoting=csv.QUOTE_NONE)
    wr.writerow([''] + ['[Fe/H]'] + ['[C/Fe]c'] + ['[Sr/Fe]'] + ['[Ba/Fe]'] + ['[Eu/Fe]'])
    wr.writerow(['Total number'] + [len(FeH_fractions)] + [len(CFe_fractions)] + [len(SrFe_fractions)] + [len(BaFe_fractions)] + [len(EuFe_fractions)])
    wr.writerow(['Number < 0.5'] + [len(FeH_fractions[FeH_fractions<0.5])] + [len(CFe_fractions[CFe_fractions<0.5])] + [len(SrFe_fractions[SrFe_fractions<0.5])] + [len(BaFe_fractions[BaFe_fractions<0.5])] + [len(EuFe_fractions[EuFe_fractions<0.5])])
    wr.writerow(['Number < 0.33'] + [len(FeH_fractions[FeH_fractions<0.33])] + [len(CFe_fractions[CFe_fractions<0.33])] + [len(SrFe_fractions[SrFe_fractions<0.33])] + [len(BaFe_fractions[BaFe_fractions<0.33])] + [len(EuFe_fractions[EuFe_fractions<0.33])])
    wr.writerow(['Number < 0.25'] + [len(FeH_fractions[FeH_fractions<0.25])] + [len(CFe_fractions[CFe_fractions<0.25])] + [len(SrFe_fractions[SrFe_fractions<0.25])] + [len(BaFe_fractions[BaFe_fractions<0.25])] + [len(EuFe_fractions[EuFe_fractions<0.25])])
