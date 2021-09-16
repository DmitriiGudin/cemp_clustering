from __future__ import division
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
import matplotlib.lines as mlines
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from astropy.stats import biweight_scale, biweight_location
import params
import clusters
from sklearn.linear_model import LinearRegression


filename_1 = params.init_file
filename_2 = params.kin_file


CDTG_assoc = [([1,3,6,8,9,11,12,17,18,22], 'Sausage'), ([4], 'ZY20:DTG-19'), ([16], 'ZY20:DTG-29'), ([10], 'ZY20:DTG-36'), ([4,5], 'IR18:Group A'), ([1,7], 'IR18:Group C'), ([1], 'IR18:Group D, E'), ([3,6,11], 'IR18:Group F'), ([8,22], 'IR18:Group H')]


def get_column(N, Type, filename):
    return np.genfromtxt(filename, delimiter=',', dtype=Type, skip_header=1, usecols=N, comments=None)


def plot_EuFe_Assoc_paper(Clusters, FeH, EuFe, avg_FeH, std_FeH, avg_EuFe, std_EuFe, xlims, ylims, xstep=0.001, ystep=0.001):
    plt.clf()
    fig = plt.figure()

    x_array = np.arange(xlims[0], xlims[1]+xstep, xstep)
    y_array = np.arange(ylims[0], ylims[1]+ystep, ystep)
    x_array = x_array.reshape((-1,1))

    gs = GridSpec(4,16)

    ax_right = fig.add_subplot(gs[1:4,10:16])
    ax_right.set_xlim(xlims[0],xlims[1])
    ax_right.set_ylim(ylims[0],ylims[1])
    ax_right.set_xlabel('[Fe/H]')
    ax_right.set_ylabel('[Eu/Fe]')
    for i, (cdtg, af, ae, sf, se) in enumerate(zip(CDTG_assoc, avg_FeH, avg_EuFe, std_FeH, std_EuFe)):
        N_stars = 0
        for c in cdtg[0]:
            N_stars += len(Clusters[c-1])
        std_feh = sf/np.sqrt(N_stars)
        std_eufe = se/np.sqrt(N_stars)
        ax_right.errorbar(af, ae, xerr=std_feh, yerr=std_eufe, color=params.colors[i])

    ax_joint = fig.add_subplot(gs[1:4,0:8])
    ax_joint.set_xlim(xlims[0],xlims[1])
    ax_joint.set_ylim(ylims[0],ylims[1])
    ax_joint.set_xlabel('[Fe/H]')
    ax_joint.set_ylabel('[Eu/Fe]')
    ax_marg_x = fig.add_subplot(gs[0,0:8])
    ax_marg_y = fig.add_subplot(gs[1:4,8])
    ax_marg_x.set_xlim(xlims[0],xlims[1])
    ax_marg_y.set_ylim(ylims[0],ylims[1])

    #ax_joint.errorbar(avg_FeH, avg_EuFe, xerr=std_FeH, yerr=std_EuFe, color='black')
    for i, (af, ae, sf, se) in enumerate(zip(avg_FeH, avg_EuFe, std_FeH/2, std_EuFe/2)):
        ax_joint.add_patch(patches.Ellipse((af,ae), sf, se, color=params.colors[i], alpha=0.333))
    
    ax_marg_x.hist(FeH, bins=100, color='black',fill=False,linewidth=1,histtype='step')
    ax_marg_x.get_xaxis().set_ticklabels([])
    ax_marg_y.hist(EuFe, orientation="horizontal", bins=100, color='black',fill=False,linewidth=1,histtype='step')
    ax_marg_y.get_yaxis().set_ticklabels([])

    avg_FeH_temp = avg_FeH[:]
    avg_FeH_temp = avg_FeH_temp.reshape((-1,1))
    model = LinearRegression().fit(avg_FeH_temp,avg_EuFe)
    r_sq = model.score(avg_FeH_temp,avg_EuFe)
    print "Coefficient for the full fit:", r_sq 
    #ax_joint.plot(x_array, model.predict(x_array), color='black', linewidth=1, linestyle='-')

    indeces = np.where(avg_FeH>-2.6)[0]
    avg_FeH, avg_EuFe = avg_FeH[indeces], avg_EuFe[indeces]
    indeces = np.where(avg_EuFe<0.65)[0]
    avg_FeH, avg_EuFe = avg_FeH[indeces], avg_EuFe[indeces]
    avg_FeH = avg_FeH.reshape((-1,1))
    model = LinearRegression().fit(avg_FeH,avg_EuFe)
    r_sq = model.score(avg_FeH,avg_EuFe)
    print "Coefficient for the fit with no outliers:", r_sq 
    ax_joint.plot(x_array, model.predict(x_array), color='black', linewidth=1, linestyle='--')

    filename_eps = 'plots/EuFe_Assoc_paper.eps'
    filename_png = 'plots/EuFe_Assoc_paper.png'
    plt.subplots_adjust(wspace=0.2,hspace=0.5)
    plt.gcf().set_size_inches(12,6)
    plt.gcf().savefig(filename_eps, bbox_inches='tight', pad_inches=-0.002, dpi=100, format='eps')
    plt.gcf().savefig(filename_png, bbox_inches='tight', pad_inches=-0.002, dpi=100, format='png')


def plot_EuFe_Assoc_paper_v2(Clusters, FeH, EuFe, avg_FeH, std_FeH, avg_EuFe, std_EuFe, xlims, ylims, xstep=0.001, ystep=0.001):
    plt.clf()
    fig, axes = plt.subplots(nrows=1,ncols=2)

    x_array = np.arange(xlims[0], xlims[1]+xstep, xstep)
    y_array = np.arange(ylims[0], ylims[1]+ystep, ystep)
    x_array = x_array.reshape((-1,1))

    axes[0].set_xlim(xlims[0],xlims[1])
    axes[0].set_ylim(ylims[0],ylims[1])
    axes[0].set_xlabel('[Fe/H]')
    axes[0].set_ylabel('[Eu/Fe]')

    #ax_joint.errorbar(avg_FeH, avg_EuFe, xerr=std_FeH, yerr=std_EuFe, color='black')
    for i, (af, ae, sf, se) in enumerate(zip(avg_FeH, avg_EuFe, std_FeH/2, std_EuFe/2)):
        axes[0].add_patch(patches.Ellipse((af,ae), sf, se, color=params.colors[i], alpha=0.333))

    avg_FeH_original = avg_FeH[:]
    avg_EuFe_original = avg_EuFe[:]

    avg_FeH_temp = avg_FeH[:]
    avg_FeH_temp = avg_FeH_temp.reshape((-1,1))
    model = LinearRegression().fit(avg_FeH_temp,avg_EuFe)
    r_sq = model.score(avg_FeH_temp,avg_EuFe)
    print "Coefficient for the full fit:", r_sq 
    #ax_joint.plot(x_array, model.predict(x_array), color='black', linewidth=1, linestyle='-')

    indeces1 = np.where(avg_FeH>-2.6)[0]
    indeces2 = np.where(avg_EuFe<0.65)[0]
    indeces = []
    for i in range(len(avg_FeH)):
        if i in indeces1 and i in indeces2:
            indeces.append(i)
    indeces = np.array(indeces)
    for i in indeces2:
        if not i in indeces:
            print "Left outlier:", CDTG_assoc[i]
    for i in indeces1:
         if not i in indeces:
             print "Right outlier:", CDTG_assoc[i]

    avg_FeH, avg_EuFe = avg_FeH[indeces], avg_EuFe[indeces]
    avg_FeH = avg_FeH.reshape((-1,1))
    model = LinearRegression().fit(avg_FeH,avg_EuFe)
    r_sq = model.score(avg_FeH,avg_EuFe)
    print "Coefficient for the fit with no outliers:", r_sq 
    axes[0].plot(x_array, model.predict(x_array), color='black', linewidth=1, linestyle='--')

    axes[1].set_xlim(xlims[0]+0.2,xlims[1]-0.1)
    axes[1].set_ylim(ylims[0],ylims[1]-0.15)
    axes[1].set_xlabel('[Fe/H]')
    axes[1].set_ylabel('[Eu/Fe]')
    avg_FeH = avg_FeH_original[:]
    avg_EuFe = avg_EuFe_original[:]
    for i, (cdtg, af, ae, sf, se) in enumerate(zip(CDTG_assoc, avg_FeH, avg_EuFe, std_FeH, std_EuFe)):
        if i in indeces:
            N_stars = 0
            for c in cdtg[0]:
                N_stars += len(Clusters[c-1])
            std_feh = sf/np.sqrt(N_stars)
            std_eufe = se/np.sqrt(N_stars)
            axes[1].errorbar(af, ae, xerr=std_feh, yerr=std_eufe, color=params.colors[i])
        axes[1].plot(x_array, model.predict(x_array), color='black', linewidth=1, linestyle='--')

    filename_eps = 'plots/EuFe_Assoc_paper.eps'
    filename_png = 'plots/EuFe_Assoc_paper.png'
    filename_pdf = 'plots/EuFe_Assoc_paper.pdf'
    plt.subplots_adjust(wspace=0.3,hspace=0.5)
    plt.gcf().set_size_inches(9,4)
    plt.gcf().savefig(filename_eps, bbox_inches='tight', pad_inches=-0.002, dpi=100, format='eps')
    plt.gcf().savefig(filename_png, bbox_inches='tight', pad_inches=-0.002, dpi=100, format='png')
    plt.gcf().savefig(filename_pdf, bbox_inches='tight', pad_inches=-0.002, dpi=100, format='pdf')


def plot_EuFe_CDTGs_paper(Clusters, FeH, EuFe, xlims=(-5,0), ylims=(0,3), xstep=0.001, ystep=0.001):
    avg_FeH, avg_EuFe, std_FeH, std_EuFe = [], [], [], []
    for c in Clusters:
        if len(c)>3:
            avg_FeH.append(biweight_location(FeH[c]))
            std_FeH.append(biweight_scale(FeH[c]))
            avg_EuFe.append(biweight_location(EuFe[c]))
            std_EuFe.append(biweight_scale(EuFe[c]))
        else:
            avg_FeH.append(np.mean(FeH[c]))
            std_FeH.append(np.std(FeH[c]))
            avg_EuFe.append(np.mean(EuFe[c]))
            std_EuFe.append(np.std(EuFe[c]))

    avg_FeH, std_FeH, avg_EuFe, std_EuFe = np.array(avg_FeH), np.array(std_FeH), np.array(avg_EuFe), np.array(std_EuFe)

    plt.clf()
    fig, axes = plt.subplots(nrows=1,ncols=2)

    x_array = np.arange(xlims[0], xlims[1]+xstep, xstep)
    y_array = np.arange(ylims[0], ylims[1]+ystep, ystep)
    x_array = x_array.reshape((-1,1))

    axes[0].set_xlim(xlims[0],xlims[1])
    axes[0].set_ylim(ylims[0],ylims[1])
    axes[0].set_xlabel('[Fe/H]')
    axes[0].set_ylabel('[Eu/Fe]')

    #ax_joint.errorbar(avg_FeH, avg_EuFe, xerr=std_FeH, yerr=std_EuFe, color='black')
    for i, (af, sf, ae, se) in enumerate(zip(avg_FeH, std_FeH, avg_EuFe, std_EuFe)):
        axes[0].add_patch(patches.Ellipse((af, ae), sf/2, se/2, color=params.colors[i], alpha=0.333))

    avg_FeH_original = avg_FeH[:]
    avg_EuFe_original = avg_EuFe[:]

    avg_FeH_temp = avg_FeH[:]
    avg_FeH_temp = avg_FeH_temp.reshape((-1,1))
    model = LinearRegression().fit(avg_FeH_temp,avg_EuFe)
    r_sq = model.score(avg_FeH_temp,avg_EuFe)
    print "Coefficient for the full fit:", r_sq 
    #ax_joint.plot(x_array, model.predict(x_array), color='black', linewidth=1, linestyle='-')

    avg_FeH = avg_FeH.reshape((-1,1))
    model = LinearRegression().fit(avg_FeH,avg_EuFe)
    r_sq = model.score(avg_FeH,avg_EuFe)
    print "Coefficient for the fit with no outliers:", r_sq 
    axes[0].plot(x_array, model.predict(x_array), color='black', linewidth=1, linestyle='--')

    axes[1].set_xlim(xlims[0],xlims[1])
    axes[1].set_ylim(ylims[0],ylims[1])
    axes[1].set_xlabel('[Fe/H]')
    axes[1].set_ylabel('[Eu/Fe]')
    avg_FeH = avg_FeH_original[:]
    avg_EuFe = avg_EuFe_original[:]
    for i, (c, af, ae, sf, se) in enumerate(zip(Clusters, avg_FeH, avg_EuFe, std_FeH, std_EuFe)):
        N_stars = len(c)
        std_feh = sf/np.sqrt(N_stars)
        std_eufe = se/np.sqrt(N_stars)
        axes[1].errorbar(af, ae, xerr=std_feh, yerr=std_eufe, color=params.colors[i])
    axes[1].plot(x_array, model.predict(x_array), color='black', linewidth=1, linestyle='--')

    filename_eps = 'plots/EuFe_CDTGs_paper.eps'
    filename_png = 'plots/EuFe_CDTGs_paper.png'
    filename_pdf = 'plots/EuFe_CDTGs_paper.pdf'
    plt.subplots_adjust(wspace=0.3,hspace=0.5)
    plt.gcf().set_size_inches(9,4)
    plt.gcf().savefig(filename_eps, bbox_inches='tight', pad_inches=-0.002, dpi=100, format='eps')
    plt.gcf().savefig(filename_png, bbox_inches='tight', pad_inches=-0.002, dpi=100, format='png')
    plt.gcf().savefig(filename_pdf, bbox_inches='tight', pad_inches=-0.002, dpi=100, format='pdf')


if __name__ == '__main__':
    Clusters = clusters.final_clusters
    FeH = get_column(8, float, params.init_file)[:]
    EuFe = get_column(13, float, params.init_file)[:]
    avg_FeH, std_FeH, avg_EuFe, std_EuFe = [], [], [], []
    for assoc in CDTG_assoc:
        feh, eufe = [], []
        for c in assoc[0]:
            for i in Clusters[c-1]:
                feh.append(FeH[i])
                eufe.append(EuFe[i])
        if len(feh)>3:
            avg_FeH.append(biweight_location(feh))
            std_FeH.append(biweight_scale(feh))
        else:
            avg_FeH.append(np.mean(feh))
            std_FeH.append(np.std(feh))
        if len(eufe)>3:
            avg_EuFe.append(biweight_location(eufe))
            std_EuFe.append(biweight_scale(eufe))
        else:
            avg_EuFe.append(np.mean(eufe))
            std_EuFe.append(np.std(eufe))

    avg_FeH, std_FeH, avg_EuFe, std_EuFe = np.array(avg_FeH), np.array(std_FeH), np.array(avg_EuFe), np.array(std_EuFe)

    #plot_EuFe_Assoc_paper(Clusters, FeH, EuFe, avg_FeH, std_FeH, avg_EuFe, std_EuFe, xlims=(-2.8,-1.8), ylims=(0.4,0.9), xstep=0.001, ystep=0.001)
    #plot_EuFe_Assoc_paper_v2(Clusters, FeH, EuFe, avg_FeH, std_FeH, avg_EuFe, std_EuFe, xlims=(-2.8,-1.8), ylims=(0.4,0.9), xstep=0.001, ystep=0.001)
    plot_EuFe_CDTGs_paper(Clusters, FeH, EuFe, xlims=(-3,-1), ylims=(0.3,1.1), xstep=0.001, ystep=0.001)
