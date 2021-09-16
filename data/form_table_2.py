from __future__ import division
import numpy as np
from numpy import genfromtxt
import csv
import os, sys
import h5py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

filename = 'master_r_process_final.csv'


def get_column(N, Type, filename):
    return np.genfromtxt(filename, delimiter=',', dtype=Type, skip_header=1, usecols=N, comments=None)


def get_abundance(Abundance, flag=''):
    if np.isnan(Abundance):
        return '...'
    Abundance = format(Abundance, '.2f')
    if float(Abundance)>0:
        Abundance = '+'+Abundance
    Abundances = flag+Abundance
    return Abundance


if __name__ == '__main__':
    Name = get_column(0, str, filename)[:]
    Class = get_column(31, str, filename)[:]
    FeH = get_column(22, float, filename)[:]
    CFe = get_column(23, float, filename)[:]
    cCFe = get_column(24, float, filename)[:]
    SrFe = get_column(25, float, filename)[:]
    BaFe = get_column(26, float, filename)[:]
    EuFe = get_column(27, float, filename)[:]
    ulC = get_column(28, str, filename)[:]
    ulSr = get_column(29, str, filename)[:]
    ulBa = get_column(30, str, filename)[:]
    Temp = get_column(20, int, filename)[:]
    Ref = get_column(32, str, filename)[:]
    RPA = get_column(36, str, filename)[:]

    savefile = open('table2.csv','wb')
    wr = csv.writer(savefile, quoting=csv.QUOTE_NONE)
    wr.writerow(['Name'] + ['RPA Class'] + ['T_eff'] + ['[Fe/H]'] + ['[C/Fe]'] + ['c[C/Fe]'] + ['[Sr/Fe]'] + ['[Ba/Fe]'] + ['[Eu/Fe]']+ ['Reference'] + ['RPA'])
    wr.writerow([''] + [''] + ['(K)'] + [''] + [''] + [''] + [''] + [''] + [''] + [''] + [''])
    for i, n in enumerate(Name):
        if FeH[i]<=-1:
            wr.writerow([Name[i]] + [Class[i]] + [Temp[i]] + [get_abundance(FeH[i])] + [get_abundance(CFe[i], ulC[i])] + [get_abundance(cCFe[i], ulC[i])] + [get_abundance(SrFe[i], ulSr[i])] + [get_abundance(BaFe[i], ulBa[i])] + [get_abundance(EuFe[i])] + [Ref[i]] + [RPA[i]])
