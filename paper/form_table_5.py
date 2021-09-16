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


def get_column(N, Type, filename):
    return np.genfromtxt(filename, delimiter=',', dtype=Type, skip_header=1, usecols=N, comments=None)


def get_abundance(Abundance, Comment='', Abundance_symbol='dfhgjksdfhfjklndfgksdgnjksnlnlgndfs'):
    if np.isnan(Abundance):
        return '...'
    Abundance = format(Abundance, '.2f')
    if float(Abundance)>0:
        Abundance = '+'+Abundance
    return Abundance


def get_else(Else):
    return format(Else, '.2f')


if __name__ == '__main__':
    Class = get_column(17, str, filename_1)[:]
    FeH = get_column(8, float, filename_1)[:]
    CFe = get_column(9, float, filename_1)[:]
    cCFe = get_column(10, float, filename_1)[:]
    SrFe = get_column(11, float, filename_1)[:]
    BaFe = get_column(12, float, filename_1)[:]
    EuFe = get_column(13, float, filename_1)[:]
    EuH = EuFe[:] + FeH[:]
    Comments = get_column(14, str, filename_1)[:]

    Energy = get_column(21, float, filename_2)[:]
    J_r = get_column(11, float, filename_2)[:]
    J_phi = get_column(13, float, filename_2)[:]
    J_z = get_column(19, float, filename_2)[:]
    ecc = get_column(29, float, filename_2)[:]
    v_phi = get_column(3, float, filename_2)[:]

    
    savefile = open('table5.csv','wb')
    wr = csv.writer(savefile, quoting=csv.QUOTE_NONE)
    wr.writerow(['Name'] + ['N stars'] + ['<Energy (10^5)>'] + ['<J_r (10^3)>'] + ['<J_phi (10^3)>'] + ['<J_z (10^3)>'] + ['<ecc>'] + ['<v_phi>'] + ['[Fe/H]'] + ['c[C/Fe]'] + ['[Sr/Fe]'] + ['[Ba/Fe]'] + ['[Eu/Fe]'])
    for m, c in enumerate(final_clusters):
        CFe_nonan = []
        for cfe, C in zip(CFe[c], Comments[c]):
            if not np.isnan(cfe) and not 'C' in C:
                CFe_nonan.append(cfe)
        CFe_nonan = np.array(CFe_nonan)
        cCFe_nonan = []
        for ccfe, C in zip(cCFe[c], Comments[c]):
            if not np.isnan(ccfe) and not 'C' in C:
                cCFe_nonan.append(ccfe)
        cCFe_nonan = np.array(cCFe_nonan)
        SrFe_nonan = []
        for srfe, C in zip(SrFe[c], Comments[c]):
            if not np.isnan(srfe) and not 'Sr' in C:
                SrFe_nonan.append(srfe)
        SrFe_nonan = np.array(SrFe_nonan)
        BaFe_nonan = []
        for bafe, C in zip(BaFe[c], Comments[c]):
            if not np.isnan(bafe) and not 'Ba' in C:
                BaFe_nonan.append(bafe)
        BaFe_nonan = np.array(BaFe_nonan)
        FeH_nonan, EuFe_nonan, EuH_nonan = FeH[c][:], EuFe[c][:], EuH[c][:]
        FeH_nonan, EuFe_nonan, EuH_nonan = FeH_nonan[~np.isnan(FeH_nonan)], EuFe_nonan[~np.isnan(EuFe_nonan)], EuH_nonan[~np.isnan(EuH_nonan)]
        if len(FeH_nonan)==3:
            FeH_text = get_abundance(np.mean(FeH_nonan))+'+\-'+get_else(np.std(FeH_nonan))
        elif len(FeH_nonan)>3:
            FeH_text = get_abundance(biweight_location(FeH_nonan))+'+\-'+get_else(biweight_scale(FeH_nonan))
        else:
            FeH_text = '...'
        if len(CFe_nonan)==3:
            CFe_text = get_abundance(np.mean(CFe_nonan))+'+\-'+get_else(np.std(CFe_nonan))
        elif len(CFe_nonan)>3:
            CFe_text = get_abundance(biweight_location(CFe_nonan))+'+\-'+get_else(biweight_scale(CFe_nonan))
        else:
            CFe_text = '...'
        if len(cCFe_nonan)==3:
            cCFe_text = get_abundance(np.mean(cCFe_nonan))+'+\-'+get_else(np.std(cCFe_nonan))
        elif len(cCFe_nonan)>3:
            cCFe_text = get_abundance(biweight_location(cCFe_nonan))+'+\-'+get_else(biweight_scale(cCFe_nonan))
        else:
            cCFe_text = '...'
        if len(SrFe_nonan)==3:
            SrFe_text = get_abundance(np.mean(SrFe_nonan))+'+\-'+get_else(np.std(SrFe_nonan))
        elif len(SrFe_nonan)>3:
            SrFe_text = get_abundance(biweight_location(SrFe_nonan))+'+\-'+get_else(biweight_scale(SrFe_nonan))
        else:
            SrFe_text = '...'
        if len(BaFe_nonan)==3:
            BaFe_text = get_abundance(np.mean(BaFe_nonan))+'+\-'+get_else(np.std(BaFe_nonan))
        elif len(BaFe_nonan)>3:
            BaFe_text = get_abundance(biweight_location(BaFe_nonan))+'+\-'+get_else(biweight_scale(BaFe_nonan))
        else:
            BaFe_text = '...'
        if len(EuFe_nonan)==3:
            EuFe_text = get_abundance(np.mean(EuFe_nonan))+'+\-'+get_else(np.std(EuFe_nonan))
        elif len(EuFe_nonan)>3:
            EuFe_text = get_abundance(biweight_location(EuFe_nonan))+'+\-'+get_else(biweight_scale(EuFe_nonan))
        else:
            EuFe_text = '...'
        if len(EuH_nonan)==3:
            EuH_text = get_abundance(np.mean(EuH_nonan))+'+\-'+get_else(np.std(EuH_nonan))
        elif len(EuH_nonan)>3:
            EuH_text = get_abundance(biweight_location(EuH_nonan))+'+\-'+get_else(biweight_scale(EuH_nonan))
        else:
            EuH_text = '...'
        if len(c)==3:
            wr.writerow(['CDTG-'+str(m+1)] + [len(c)] + [get_else(np.mean(Energy[c])/1e5)] + [get_else(np.mean(J_r[c])/1e3)] + [get_else(np.mean(J_phi[c])/1e3)] + [get_else(np.mean(J_z[c])/1e3)] + [get_else(np.mean(ecc[c]))] + [get_else(np.mean(v_phi[c]))] + [FeH_text] + [cCFe_text] + [SrFe_text] + [BaFe_text] + [EuFe_text])
        else:
            wr.writerow(['CDTG-'+str(m+1)] + [len(c)] + [get_else(biweight_location(Energy[c])/1e5)] + [get_else(biweight_location(J_r[c])/1e3)] + [get_else(biweight_location(J_phi[c])/1e3)] + [get_else(biweight_location(J_z[c])/1e3)] + [get_else(biweight_location(ecc[c]))] + [get_else(biweight_location(v_phi[c]))] + [FeH_text] + [cCFe_text] + [SrFe_text] + [BaFe_text] + [EuFe_text])
        
