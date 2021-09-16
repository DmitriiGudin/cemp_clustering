from __future__ import division
import numpy as np
from numpy import genfromtxt
import csv
import os, sys
import h5py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

filename = 'master_r_process_final.csv'

GAIA_distance_error_mark_points = [20, 30] # In percent.
GAIA_distance_error_marks = ['*', '**']
STARHORSE_distance_error_mark_points = [20, 30] # In percent.
STARHORSE_distance_error_marks = ["*", "**"]

def get_column(N, Type, filename):
    return np.genfromtxt(filename, delimiter=',', dtype=Type, skip_header=1, usecols=N, comments=None)


def get_RA(RA):
    ra = RA[:6]
    ra_float = float(RA[6:])
    ra += format(ra_float, '05.2f')
    return ra


def get_DEC(DEC):
    if DEC[0] != '+' and DEC[0] != '-':
        DEC = '+'+DEC
    dec = DEC[:7]
    dec_float = float(DEC[7:])
    dec += format(dec_float, '04.1f')
    return dec

def get_RV(RV):
    if np.isnan(RV):
        return '...'
    RV = format(RV,'.1f')
    return RV
   

def get_else(Else):
    if np.isnan(Else):
        return '...'
    return format(Else, '.2f')

if __name__ == '__main__':
    Name = get_column(0, str, filename)[:]
    RA = get_column(2, str, filename)[:]
    DEC = get_column(3, str, filename)[:]
    GAIA_dist = get_column(6, float, filename)[:]
    GAIA_dist_err = get_column(7, float, filename)[:]
    StarHorse_dist = get_column(8, float, filename)[:]
    StarHorse_dist_err = get_column(9, float, filename)[:]
    FeH = get_column(22, float, filename)[:]
    # ---
    # Get RVs.
    RV = []
    Literature_RV = get_column(10, float, filename)[:]
    GAIA_RV = get_column(11, float, filename)[:]
    for l, g in zip(Literature_RV, GAIA_RV):
       if not np.isnan(g):
           RV.append(g)
       elif not np.isnan(l):
           RV.append(l)
       else:
           RV.append(np.nan)
    RV = np.array(RV)
    # ---
    pmRA = get_column(13, float, filename)[:]
    pmDEC = get_column(14, float, filename)[:]
    Vmag = get_column(16, float, filename)[:]
    RV_Reference = get_column(33, str, filename)[:]
   
    RV_Ref = []
    for r_l, r_g, r_l_Ref in zip(Literature_RV, GAIA_RV, RV_Reference):
        if not np.isnan(r_g):
            RV_Ref.append('GAIA DR2')
        elif not np.isnan(r_l):
            RV_Ref.append(r_l_Ref)
        else:
            RV_Ref.append('*')

    Error_marks_GAIA, Error_marks_StarHorse = [], []
    for gd, gde, sd, sde in zip(GAIA_dist, GAIA_dist_err, StarHorse_dist, StarHorse_dist_err):
        error_mark_GAIA=''
        for mp, m in zip (GAIA_distance_error_mark_points, GAIA_distance_error_marks):
            if not np.isnan(gde) and not np.isnan(gd) and abs(gde/gd) > mp/100:
                error_mark_GAIA = m
        error_mark_StarHorse=''
        for mp, m in zip (STARHORSE_distance_error_mark_points, STARHORSE_distance_error_marks):
            if not np.isnan(sde) and not np.isnan(sd) and abs(sde/sd) > mp/100:
                error_mark_StarHorse = m
        Error_marks_GAIA.append(error_mark_GAIA)
        Error_marks_StarHorse.append(error_mark_StarHorse)
 
    savefile = open('table1.csv','wb')
    wr = csv.writer(savefile, quoting=csv.QUOTE_NONE)
    wr.writerow(['Name'] + ['RA'] + ['DEC'] + ['Vmag'] + ['GAIA dist. (err)'] + ['StarHorse dist. (err)'] + ['Radial Velocity'] + ['pmRA'] + ['pmDEC'])
    wr.writerow([''] + ['(2000)'] + ['(2000)'] + [''] + ['(kpc)'] + ['(kpc)'] + ['km/s'] + ['mas/yr'] + ['mas/yr'])
    for i, n in enumerate(Name):
        if FeH[i]<=-1:
            if RV_Ref[i] != 'GAIA DR2':
                rv_ref = ' ' + RV_Ref[i]
            if RV_Ref[i] == '*' or RV_Ref[i] == '...':
                rv_ref = ' *'
            if np.isnan(RV[i]) or RV_Ref[i] == 'GAIA DR2':
                rv_ref = ''
            gaia_dist = get_else(GAIA_dist[i]) + ' (' + get_else(GAIA_dist_err[i])+ ')' + Error_marks_GAIA[i]
            starhorse_dist = get_else(StarHorse_dist[i]) + ' (' + get_else(StarHorse_dist_err[i]) + ')' + Error_marks_StarHorse[i]
            if get_else(GAIA_dist[i]) == '...':
                gaia_dist = '...'
            if get_else(StarHorse_dist[i]) == '...':
                starhorse_dist = '...'
            wr.writerow([Name[i]] + [get_RA(RA[i])] + [get_DEC(DEC[i])] + [get_else(Vmag[i])] + [gaia_dist] + [starhorse_dist] + [get_RV(RV[i]) + rv_ref] + [get_else(pmRA[i])] + [get_else(pmDEC[i])])
