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

distance_error_mark_points = [20, 30] # In percent.
distance_error_marks = ['*', '**']


def get_column(N, Type, filename):
    return np.genfromtxt(filename, delimiter=',', dtype=Type, skip_header=1, usecols=N, comments=None)


def get_abundance(Abundance, Comment='', Abundance_symbol='dfhgjksdfhfjklndfgksdgnjksnlnlgndfs'):
    if np.isnan(Abundance):
        return '...'
    Abundance = format(Abundance, '.2f')
    if float(Abundance)>0:
        Abundance = '+'+Abundance
    if 'ul'+Abundance_symbol in Comment:
        Abundance = '<'+Abundance
    elif 'll'+Abundance_symbol in Comment:
        Abundance = '>'+Abundance
    return Abundance


def get_else(Else):
    return format(Else, '.2f')


if __name__ == '__main__':
    Name = get_column(0, str, filename_1)[:]
    Class = get_column(14, str, filename_1)[:]
    FeH = get_column(8, float, filename_1)[:]
    CFe = get_column(9, float, filename_1)[:]
    cCFe = get_column(9, float, filename_1)[:]
    SrFe = get_column(10, float, filename_1)[:]
    BaFe = get_column(11, float, filename_1)[:]
    EuFe = get_column(12, float, filename_1)[:]
    Comments = get_column(13, str, filename_1)[:]
    dist = get_column(3, float, filename_1)[:]
    dist_err = get_column(4, float, filename_1)[:]
    EuH = EuFe[:] + FeH[:]

    Error_marks = []
    for d, de in zip(dist, dist_err):
        error_mark=''
        for mp, m in zip (distance_error_mark_points, distance_error_marks):
            if abs(de/d) > mp/100:
                error_mark = m
        Error_marks.append(error_mark)

    filename = params.dispersion_distrib_file_mask[0] + 'FeH' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    cumuls_FeH = []
    for c in final_clusters:
        feh = FeH[c]
        feh = feh[~np.isnan(feh)]
        if len(feh)<=params.dispersion_cluster_size[1] and len(feh)>=params.dispersion_cluster_size[0]:
            Dispersion_array = f["/dispersion"][len(feh)-params.dispersion_cluster_size[0]]
            if len(feh) < params.biweight_estimator_min_cluster_size:
                cumuls_FeH.append(len(Dispersion_array[Dispersion_array<np.std(feh)])/len(Dispersion_array))
            else:
                cumuls_FeH.append(len(Dispersion_array[Dispersion_array<biweight_scale(feh)])/len(Dispersion_array))
        else:
            cumuls_FeH.append(np.nan)
    cumuls_FeH = np.array(cumuls_FeH)

    filename = params.dispersion_distrib_file_mask[0] + 'cCFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    cumuls_CFe = []
    for c in final_clusters:
        cfe = []
        for cf, C in zip(cCFe[c], Comments[c]):
            if not np.isnan(cf) and not 'C' in C:
                cfe.append(cf)
        if len(cfe)<=params.dispersion_cluster_size[1] and len(cfe)>=params.dispersion_cluster_size[0]:
            Dispersion_array = f["/dispersion"][len(cfe)-params.dispersion_cluster_size[0]]
            if len(cfe) < params.biweight_estimator_min_cluster_size:
                cumuls_CFe.append(len(Dispersion_array[Dispersion_array<np.std(cfe)])/len(Dispersion_array))
            else:
                cumuls_CFe.append(len(Dispersion_array[Dispersion_array<biweight_scale(cfe)])/len(Dispersion_array))
        else:
            cumuls_CFe.append(np.nan)
    cumuls_CFe = np.array(cumuls_CFe)

    filename = params.dispersion_distrib_file_mask[0] + 'SrFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    cumuls_SrFe = []
    for c in final_clusters:
        srfe = []
        for srf, C in zip(SrFe[c], Comments[c]):
            if not np.isnan(srf) and not 'Sr' in C:
                srfe.append(srf)
        if len(srfe)<=params.dispersion_cluster_size[1] and len(srfe)>=params.dispersion_cluster_size[0]:
            Dispersion_array = f["/dispersion"][len(srfe)-params.dispersion_cluster_size[0]]
            if len(srfe) < params.biweight_estimator_min_cluster_size:
                cumuls_SrFe.append(len(Dispersion_array[Dispersion_array<np.std(srfe)])/len(Dispersion_array))
            else:
                cumuls_SrFe.append(len(Dispersion_array[Dispersion_array<biweight_scale(srfe)])/len(Dispersion_array))
        else:
            cumuls_SrFe.append(np.nan)
    cumuls_SrFe = np.array(cumuls_SrFe)

    filename = params.dispersion_distrib_file_mask[0] + 'BaFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    cumuls_BaFe = []
    for c in final_clusters:
        bafe = []
        for baf, C in zip(BaFe[c], Comments[c]):
            if not np.isnan(baf) and not 'Ba' in C:
                bafe.append(baf)
        if len(bafe)<=params.dispersion_cluster_size[1] and len(bafe)>=params.dispersion_cluster_size[0]:
            Dispersion_array = f["/dispersion"][len(bafe)-params.dispersion_cluster_size[0]]
            if len(bafe) < params.biweight_estimator_min_cluster_size:
                cumuls_BaFe.append(len(Dispersion_array[Dispersion_array<np.std(bafe)])/len(Dispersion_array))
            else:
                cumuls_BaFe.append(len(Dispersion_array[Dispersion_array<biweight_scale(bafe)])/len(Dispersion_array))
        else:
            cumuls_BaFe.append(np.nan)
    cumuls_BaFe = np.array(cumuls_BaFe)

    filename = params.dispersion_distrib_file_mask[0] + 'EuFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    cumuls_EuFe = []
    for c in final_clusters:
        eufe = EuFe[c]
        eufe = eufe[~np.isnan(eufe)]
        if len(eufe)<=params.dispersion_cluster_size[1] and len(eufe)>=params.dispersion_cluster_size[0]:
            Dispersion_array = f["/dispersion"][len(eufe)-params.dispersion_cluster_size[0]]
            if len(eufe) < params.biweight_estimator_min_cluster_size:
                cumuls_EuFe.append(len(Dispersion_array[Dispersion_array<np.std(eufe)])/len(Dispersion_array))
            else:
                cumuls_EuFe.append(len(Dispersion_array[Dispersion_array<biweight_scale(eufe)])/len(Dispersion_array))
        else:
            cumuls_EuFe.append(np.nan)
    cumuls_EuFe = np.array(cumuls_EuFe)

    avg_FeH, avg_EuFe, avg_CFe, avg_SrFe, avg_BaFe = [], [], [], [], []
    for c in final_clusters:
        feh, eufe, cfe, srfe, bafe = FeH[c], EuFe[c], cCFe[c], SrFe[c], BaFe[c]
        feh, eufe, cfe, srfe, bafe = feh[~np.isnan(feh)], eufe[~np.isnan(eufe)], cfe[~np.isnan(cfe)], srfe[~np.isnan(srfe)], bafe[~np.isnan(bafe)]
        
        if len(feh)>2:
            avg_FeH.append(np.mean(feh))
        else:
            avg_FeH.append('...')
        
        if len(eufe)>2:
            avg_EuFe.append(np.mean(eufe))
        else:
            avg_EuFe.append('...')
        
        if len(cfe)>2:
            avg_CFe.append(np.mean(cfe))
        else:
            avg_CFe.append('...')
        
        if len(srfe)>2:
            avg_SrFe.append(np.mean(srfe))
        else:
            avg_SrFe.append('...')
        
        if len(bafe)>2:
            avg_BaFe.append(np.mean(bafe))
        else:
            avg_BaFe.append('...')


    cumuls_FeH_nonan = cumuls_FeH[~np.isnan(cumuls_FeH)]
    cumuls_CFe_nonan = cumuls_CFe[~np.isnan(cumuls_CFe)]
    cumuls_SrFe_nonan = cumuls_SrFe[~np.isnan(cumuls_SrFe)]
    cumuls_BaFe_nonan = cumuls_BaFe[~np.isnan(cumuls_BaFe)]
    cumuls_EuFe_nonan = cumuls_EuFe[~np.isnan(cumuls_EuFe)]

    
    savefile = open('table4.csv','wb')
    wr = csv.writer(savefile, quoting=csv.QUOTE_NONE)
    wr.writerow(['Name'] + ['Class'] + ['[Fe/H]'] + ['[C/Fe]'] + ['c[C/Fe]'] + ['[Sr/Fe]'] + ['[Ba/Fe]'] + ['[Eu/Fe]'] + ['[Eu/H]'])
    for m, c in enumerate(final_clusters):
        wr.writerow([''] + [''] + [''] + [''] + [''] + [''] + [''] + [''] + [''])
        for n, i in enumerate(c):
            wr.writerow([Name[i]+Error_marks[i]] + [Class[i]] + [get_abundance(FeH[i])] + [get_abundance(CFe[i],Comments[i],'C')] + [get_abundance(cCFe[i],Comments[i],'C')] + [get_abundance(SrFe[i],Comments[i],'Sr')] + [get_abundance(BaFe[i],Comments[i],'Ba')] + [get_abundance(EuFe[i])] + [get_abundance(EuH[i])])
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
        wr.writerow(['Dispersion'] + [''] + [FeH_text] + [CFe_text] + [cCFe_text] + [SrFe_text] + [BaFe_text] + [EuFe_text] + [EuH_text])
        wr.writerow(['Cumulative fraction values'] + [''] + [get_else(cumuls_FeH[m])] + [''] + [get_else(cumuls_CFe[m])] + [get_else(cumuls_SrFe[m])] + [get_else(cumuls_BaFe[m])] + [get_else(cumuls_EuFe[m])] + [''])

    cumuls_FeH = cumuls_FeH[~np.isnan(cumuls_FeH)]
    cumuls_CFe = cumuls_CFe[~np.isnan(cumuls_CFe)]
    cumuls_SrFe = cumuls_SrFe[~np.isnan(cumuls_SrFe)]
    cumuls_BaFe = cumuls_BaFe[~np.isnan(cumuls_BaFe)]
    cumuls_EuFe = cumuls_EuFe[~np.isnan(cumuls_EuFe)]

    print "[Fe/H] STATISTICS:"
    print "Number of fracs: ", len (cumuls_FeH)
    print "Number of fracs below 0.5: ", len(cumuls_FeH[cumuls_FeH<0.5])
    print "Number of fracs below 0.33: ", len(cumuls_FeH[cumuls_FeH<0.33])
    print "Number of fracs below 0.25: ", len(cumuls_FeH[cumuls_FeH<0.25])
    print "----"
    print r"[C/Fe]$_\mathrm{c}$ STATISTICS:"
    print "Number of fracs: ", len (cumuls_CFe)
    print "Number of fracs below 0.5: ", len(cumuls_CFe[cumuls_CFe<0.5])
    print "Number of fracs below 0.33: ", len(cumuls_CFe[cumuls_CFe<0.33])
    print "Number of fracs below 0.25: ", len(cumuls_CFe[cumuls_CFe<0.25])
    print "----"
    print "[Sr/Fe] STATISTICS:"
    print "Number of fracs: ", len (cumuls_SrFe)
    print "Number of fracs below 0.5: ", len(cumuls_SrFe[cumuls_SrFe<0.5])
    print "Number of fracs below 0.33: ", len(cumuls_SrFe[cumuls_SrFe<0.33])
    print "Number of fracs below 0.25: ", len(cumuls_SrFe[cumuls_SrFe<0.25])
    print "----"
    print "[Ba/Fe] STATISTICS:"
    print "Number of fracs: ", len (cumuls_BaFe)
    print "Number of fracs below 0.5: ", len(cumuls_BaFe[cumuls_BaFe<0.5])
    print "Number of fracs below 0.33: ", len(cumuls_BaFe[cumuls_BaFe<0.33])
    print "Number of fracs below 0.25: ", len(cumuls_BaFe[cumuls_BaFe<0.25])
    print "----"
    print "[Eu/Fe] STATISTICS:"
    print "Number of fracs: ", len (cumuls_EuFe)
    print "Number of fracs below 0.5: ", len(cumuls_EuFe[cumuls_EuFe<0.5])
    print "Number of fracs below 0.33: ", len(cumuls_EuFe[cumuls_EuFe<0.33])
    print "Number of fracs below 0.25: ", len(cumuls_EuFe[cumuls_EuFe<0.25])
