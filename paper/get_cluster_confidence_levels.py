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

final_clusters = clusters.final_clusters
filename_1 = params.init_file
filename_2 = params.kin_file
confidences_filename = 'HDBSCAN_output_MetalPoor.csv'


def get_column(N, Type, filename):
    return np.genfromtxt(filename, delimiter=',', dtype=Type, skip_header=1, usecols=N, comments=None)


if __name__ == '__main__':
    conf_names = get_column(0, str, confidences_filename)[:]
    conf_probs = get_column(21, float, confidences_filename)[:]
    kin_names = get_column(0, str, filename_1)[:]

    confidences_list = []
    for cluster in final_clusters:
        confidences = []
        for c in cluster:
            name = kin_names[c]
            i = np.where(conf_names=='"'+name+'"')[0][0]
            confidences.append(conf_probs[i])
        confidences_list.append(np.mean(confidences))

    indeces = np.argsort(confidences_list)
    indeces = indeces[::-1]
    for i in indeces:
        print "CDTG-"+str(i+1)+": "+str(confidences_list[i])+"%"

    print final_clusters
    new_clusters = [final_clusters[i] for i in indeces]
    print new_clusters
