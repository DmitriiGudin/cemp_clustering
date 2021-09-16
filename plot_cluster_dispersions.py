from __future__ import division
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
import h5py
from sklearn.neighbors import KernelDensity
from astropy.stats import biweight_scale
import params
import clusters


sigma_1_frac = 0.68
sigma_2_frac = 0.95
sigma_3_frac = 0.997

cluster_IDs = ['c'+str(i) for i in range(50)]

bandwidth_refinement = 5


def get_column(N, Type, filename):
    return np.genfromtxt(filename, delimiter=',', dtype=Type, skip_header=1, usecols=N, comments=None)


def plot_dispersions_paper(Clusters, FeH, CFe, EuFe, SrFe, BaFe):
    plt.clf()
    #fig, axes = plt.subplots (nrows=3 , ncols=5)

    gs = gridspec.GridSpec(3, 10)
    ax00 = plt.subplot(gs[0, 0:1])
    ax01 = plt.subplot(gs[0, 1:3])
    ax02 = plt.subplot(gs[0, 3:5])
    ax03 = plt.subplot(gs[0, 5:7])
    ax04 = plt.subplot(gs[0, 7:9])
    ax05 = plt.subplot(gs[0, 9:10])
    ax10 = plt.subplot(gs[1, 0:2])
    ax11 = plt.subplot(gs[1, 2:4])
    ax12 = plt.subplot(gs[1, 4:6])
    ax13 = plt.subplot(gs[1, 6:8])
    ax14 = plt.subplot(gs[1, 8:10])
    ax20 = plt.subplot(gs[2, 0:2])
    ax21 = plt.subplot(gs[2, 2:4])
    ax22 = plt.subplot(gs[2, 4:6])
    ax23 = plt.subplot(gs[2, 6:8])
    ax24 = plt.subplot(gs[2, 8:10])
  
    indeces = [c[i] for c in Clusters for i in range(len(c))]
    CH = A_C-8.43

    #axes[0,0].axis('off')
    ax00.axis('off')
    ax05.axis('off')

    ax01.set_xlabel("[Fe/H]")
    ax01.set_ylabel("[C/Fe]")
    for i, c in enumerate(Clusters):
        ax01.scatter(FeH[c], CFe[c], c=params.colors[i], marker=params.markers[i], s=30)

    ax02.set_xlabel("[Fe/H]")
    ax02.set_ylabel("[Sr/Fe]")
    #ax02.set_ylim(-1,1)
    for i, c in enumerate(Clusters):
        ax02.scatter(FeH[c], SrFe[c], c=params.colors[i], marker=params.markers[i], s=30)

    ax03.set_xlabel("[Fe/H]")
    ax03.set_ylabel("[Ba/Fe]")
    for i, c in enumerate(Clusters):
        ax03.scatter(FeH[c], BaFe[c], c=params.colors[i], marker=params.markers[i], s=30)
   
    ax04.set_xlabel("[Fe/H]")
    ax04.set_ylabel("[Eu/Fe]")
    #ax04.set_ylim(-1,1)
    for i, c in enumerate(Clusters):
        ax04.scatter(FeH[c], EuFe[c], c=params.colors[i], marker=params.markers[i], s=30)

    filename = params.dispersion_distrib_file_mask[0] + 'FeH' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = f["/dispersion"][3-params.dispersion_cluster_size[0]]
    N_samples = len(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array])
    ax10.set_xlabel(r"$\sigma$ ([Fe/H])")
    ax10.set_ylabel("Cumulative fraction")
    ax10.plot(x_array, y_array, c='black', linewidth=1)
    ax10.text(0.75, 0.2, r"$N=3$", fontsize=15)
    for i, c in enumerate(Clusters):
        FeH_nonan = FeH[c]
        FeH_nonan = FeH_nonan[~np.isnan(FeH_nonan)]
        if len(FeH_nonan)==3:
            ax10.scatter(np.std(FeH_nonan), len(Dispersion_array[Dispersion_array<np.std(FeH_nonan)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100)

    filename = params.dispersion_distrib_file_mask[0] + 'CFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = f["/dispersion"][3-params.dispersion_cluster_size[0]]
    N_samples = len(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array])
    ax11.set_xlabel(r"$\sigma$ ([C/Fe])")
    ax11.set_ylabel("Cumulative fraction")
    ax11.plot(x_array, y_array, c='black', linewidth=1)
    ax11.text(1.25, 0.2, r"$N=3$", fontsize=15)
    for i, c in enumerate(Clusters):
        CFe_nonan = CFe[c]
        CFe_nonan = CFe_nonan[~np.isnan(CFe_nonan)]
        if len(CFe_nonan)==3:
            ax11.scatter(np.std(CFe_nonan), len(Dispersion_array[Dispersion_array<np.std(CFe_nonan)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100)

    filename = params.dispersion_distrib_file_mask[0] + 'SrFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = f["/dispersion"][3-params.dispersion_cluster_size[0]]
    N_samples = len(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array])
    ax12.set_xlabel(r"$\sigma$ ([Sr/Fe])")
    ax12.set_ylabel("Cumulative fraction")
    ax12.plot(x_array, y_array, c='black', linewidth=1)
    ax12.text(1.25, 0.2, r"$N=3$", fontsize=15)
    for i, c in enumerate(Clusters):
        SrFe_nonan = SrFe[c]
        SrFe_nonan = SrFe_nonan[~np.isnan(SrFe_nonan)]
        if len(SrFe_nonan)==3:
            ax12.scatter(np.std(SrFe_nonan), len(Dispersion_array[Dispersion_array<np.std(SrFe_nonan)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100)

    filename = params.dispersion_distrib_file_mask[0] + 'BaFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = f["/dispersion"][3-params.dispersion_cluster_size[0]]
    N_samples = len(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array])
    ax13.set_xlabel(r"$\sigma$ ([Ba/Fe])")
    ax13.set_ylabel("Cumulative fraction")
    ax13.plot(x_array, y_array, c='black', linewidth=1)
    ax13.text(1.15, 0.2, r"$N=3$", fontsize=15)
    for i, c in enumerate(Clusters):
        BaFe_nonan = BaFe[c]
        BaFe_nonan = BaFe_nonan[~np.isnan(BaFe_nonan)]
        if len(BaFe_nonan)==3:
            ax13.scatter(np.std(BaFe_nonan), len(Dispersion_array[Dispersion_array<np.std(BaFe_nonan)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100)

    filename = params.dispersion_distrib_file_mask[0] + 'EuFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = f["/dispersion"][3-params.dispersion_cluster_size[0]]
    N_samples = len(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array])
    ax14.set_xlabel(r"$\sigma$ ([Eu/Fe])")
    ax14.set_ylabel("Cumulative fraction")
    ax14.plot(x_array, y_array, c='black', linewidth=1)
    ax14.text(0.6, 0.2, r"$N=3$", fontsize=15)
    for i, c in enumerate(Clusters):
        EuFe_nonan = EuFe[c]
        EuFe_nonan = EuFe_nonan[~np.isnan(EuFe_nonan)]
        if len(EuFe_nonan)==3:
            ax14.scatter(np.std(EuFe_nonan), len(Dispersion_array[Dispersion_array<np.std(EuFe_nonan)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100)

        filename = params.dispersion_distrib_file_mask[0] + 'FeH' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = f["/dispersion"][4-params.dispersion_cluster_size[0]]
    N_samples = len(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array])
    ax20.set_xlabel(r"$\sigma$ ([Fe/H])")
    ax20.set_ylabel("Cumulative fraction")
    ax20.plot(x_array, y_array, c='black', linewidth=1)
    ax20.text(0.75, 0.2, r"$N=4$", fontsize=15)
    for i, c in enumerate(Clusters):
        FeH_nonan = FeH[c]
        FeH_nonan = FeH_nonan[~np.isnan(FeH_nonan)]
        if len(FeH_nonan)==4:
            ax20.scatter(np.std(FeH_nonan), len(Dispersion_array[Dispersion_array<np.std(FeH_nonan)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100)

    filename = params.dispersion_distrib_file_mask[0] + 'CFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = f["/dispersion"][4-params.dispersion_cluster_size[0]]
    N_samples = len(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array])
    ax21.set_xlabel(r"$\sigma$ ([C/Fe])")
    ax21.set_ylabel("Cumulative fraction")
    ax21.plot(x_array, y_array, c='black', linewidth=1)
    ax21.text(1.15, 0.2, r"$N=4$", fontsize=15)
    for i, c in enumerate(Clusters):
        CFe_nonan = CFe[c]
        CFe_nonan = CFe_nonan[~np.isnan(CFe_nonan)]
        if len(CFe_nonan)==4:
            ax21.scatter(np.std(CFe_nonan), len(Dispersion_array[Dispersion_array<np.std(CFe_nonan)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100)

    filename = params.dispersion_distrib_file_mask[0] + 'SrFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = f["/dispersion"][4-params.dispersion_cluster_size[0]]
    N_samples = len(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array])
    ax22.set_xlabel(r"$\sigma$ ([Sr/Fe])")
    ax22.set_ylabel("Cumulative fraction")
    ax22.plot(x_array, y_array, c='black', linewidth=1)
    ax22.text(1.15, 0.2, r"$N=4$", fontsize=15)
    for i, c in enumerate(Clusters):
        SrFe_nonan = SrFe[c]
        SrFe_nonan = SrFe_nonan[~np.isnan(SrFe_nonan)]
        if len(SrFe_nonan)==4:
            ax22.scatter(np.std(SrFe_nonan), len(Dispersion_array[Dispersion_array<np.std(SrFe_nonan)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100)

    filename = params.dispersion_distrib_file_mask[0] + 'BaFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = f["/dispersion"][4-params.dispersion_cluster_size[0]]
    N_samples = len(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array])
    ax23.set_xlabel(r"$\sigma$ ([Ba/Fe])")
    ax23.set_ylabel("Cumulative fraction")
    ax23.plot(x_array, y_array, c='black', linewidth=1)
    ax23.text(1, 0.2, r"$N=4$", fontsize=15)
    for i, c in enumerate(Clusters):
        BaFe_nonan = BaFe[c]
        BaFe_nonan = BaFe_nonan[~np.isnan(BaFe_nonan)]
        if len(BaFe_nonan)==4:
            ax23.scatter(np.std(BaFe_nonan), len(Dispersion_array[Dispersion_array<np.std(BaFe_nonan)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100)

    filename = params.dispersion_distrib_file_mask[0] + 'EuFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = f["/dispersion"][4-params.dispersion_cluster_size[0]]
    N_samples = len(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array])
    ax24.set_xlabel(r"$\sigma$ ([Eu/Fe])")
    ax24.set_ylabel("Cumulative fraction")
    ax24.plot(x_array, y_array, c='black', linewidth=1)
    ax24.text(0.6, 0.2, r"$N=4$", fontsize=15)
    for i, c in enumerate(Clusters):
        EuFe_nonan = EuFe[c]
        EuFe_nonan = EuFe_nonan[~np.isnan(EuFe_nonan)]
        if len(EuFe_nonan)==4:
            ax24.scatter(np.std(EuFe_nonan), len(Dispersion_array[Dispersion_array<np.std(EuFe_nonan)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100)

    plt.gcf().set_size_inches(15, 9)
    plt.tight_layout()
    plt.gcf().savefig("plots/dispersions_paper.eps", bbox_inches='tight',pad_inches=0.01, dpi=100, format='eps')
    plt.gcf().savefig("plots/dispersions_paper.png", bbox_inches='tight',pad_inches=0.01, dpi=100, format='png')
    plt.close()


def plot_dispersions_paper_v2(Clusters, FeH, cCFe, EuFe, SrFe, BaFe, Comments):
    plt.clf()

    gs = gridspec.GridSpec(5, 10)
    ax00 = plt.subplot(gs[0, 0:1])
    ax01 = plt.subplot(gs[0, 1:3])
    ax02 = plt.subplot(gs[0, 3:5])
    ax03 = plt.subplot(gs[0, 5:7])
    ax04 = plt.subplot(gs[0, 7:9])
    ax05 = plt.subplot(gs[0, 9:10])
    ax10 = plt.subplot(gs[1, 0:2])
    ax11 = plt.subplot(gs[1, 2:4])
    ax12 = plt.subplot(gs[1, 4:6])
    ax13 = plt.subplot(gs[1, 6:8])
    ax14 = plt.subplot(gs[1, 8:10])
    ax20 = plt.subplot(gs[2, 0:2])
    ax21 = plt.subplot(gs[2, 2:4])
    ax22 = plt.subplot(gs[2, 4:6])
    ax23 = plt.subplot(gs[2, 6:8])
    ax24 = plt.subplot(gs[2, 8:10])
    ax30 = plt.subplot(gs[3, 0:2])
    ax31 = plt.subplot(gs[3, 2:4])
    ax32 = plt.subplot(gs[3, 4:6])
    ax33 = plt.subplot(gs[3, 6:8])
    ax34 = plt.subplot(gs[3, 8:10])  

    indeces = [c[i] for c in Clusters for i in range(len(c))]
    CH = A_C-8.43

    #axes[0,0].axis('off')
    ax00.axis('off')
    ax05.axis('off')

    ax01.set_xlabel("[Fe/H]")
    ax01.set_ylabel(r"[C/Fe]$_\mathrm{c}$")
    for i, c in enumerate(Clusters):
        final_FeH, final_cCFe = [], []
        for feh, ccfe, C in zip(FeH[c], cCFe[c], Comments[c]):
            if not np.isnan(ccfe) and not 'C' in C:
                final_FeH.append(feh)
                final_cCFe.append(ccfe) 
        ax01.scatter(final_FeH, final_cCFe, c=params.colors[i], marker=params.markers[i], s=30)

    ax02.set_xlabel("[Fe/H]")
    ax02.set_ylabel("[Sr/Fe]")
    #ax02.set_ylim(-1,1)
    for i, c in enumerate(Clusters):
        final_FeH, final_SrFe = [], []
        for feh, srfe, C in zip(FeH[c], SrFe[c], Comments[c]):
            if not np.isnan(srfe) and not 'Sr' in C:
                final_FeH.append(feh)
                final_SrFe.append(srfe)
        ax02.scatter(FeH[c], SrFe[c], c=params.colors[i], marker=params.markers[i], s=30)

    ax03.set_xlabel("[Fe/H]")
    ax03.set_ylabel("[Ba/Fe]")
    for i, c in enumerate(Clusters):
        final_FeH, final_BaFe = [], []
        for feh, bafe, C in zip(FeH[c], BaFe[c], Comments[c]):
            if not np.isnan(bafe) and not 'Ba' in C:
                final_FeH.append(feh)
                final_BaFe.append(bafe) 
        ax03.scatter(final_FeH, final_BaFe, c=params.colors[i], marker=params.markers[i], s=30)
   
    ax04.set_xlabel("[Fe/H]")
    ax04.set_ylabel("[Eu/Fe]")
    #ax04.set_ylim(-1,1)
    for i, c in enumerate(Clusters):
        ax04.scatter(FeH[c], EuFe[c], c=params.colors[i], marker=params.markers[i], s=30)

    

    filename = params.dispersion_distrib_file_mask[0] + 'FeH' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = f["/dispersion"][3-params.dispersion_cluster_size[0]]
    N_samples = len(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array])
    ax10.set_xlabel(r"$\sigma$ ([Fe/H])")
    ax10.set_ylabel("Cumulative fraction")
    ax10.plot(x_array, y_array, c='black', linewidth=1, linestyle='--')
    ax10.plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    ax10.plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    ax10.text(0.75, 0.1, r"$N=3$", fontsize=15)
    for i, c in enumerate(Clusters):
        FeH_nonan = FeH[c]
        FeH_nonan = FeH_nonan[~np.isnan(FeH_nonan)]
        if len(FeH_nonan)==3:
            ax10.scatter(np.std(FeH_nonan), len(Dispersion_array[Dispersion_array<np.std(FeH_nonan)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100)

    filename = params.dispersion_distrib_file_mask[0] + 'cCFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = f["/dispersion"][3-params.dispersion_cluster_size[0]]
    N_samples = len(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array])
    ax11.set_xlabel(r"$\sigma$ ([C/Fe]$_\mathrm{c}$)")
    ax11.set_ylabel("Cumulative fraction")
    ax11.plot(x_array, y_array, c='black', linewidth=1, linestyle='--')
    ax11.plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    ax11.plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    ax11.text(1.25, 0.1, r"$N=3$", fontsize=15)
    for i, c in enumerate(Clusters):
        final_cCFe = []
        for ccfe, C in zip(cCFe[c], Comments[c]):
            if not np.isnan(ccfe) and not 'C' in C:
                final_cCFe.append(ccfe) 
        if len(final_cCFe)==3:
            ax11.scatter(np.std(final_cCFe), len(Dispersion_array[Dispersion_array<np.std(final_cCFe)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100)

    filename = params.dispersion_distrib_file_mask[0] + 'SrFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = f["/dispersion"][3-params.dispersion_cluster_size[0]]
    N_samples = len(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array])
    ax12.set_xlabel(r"$\sigma$ ([Sr/Fe])")
    ax12.set_ylabel("Cumulative fraction")
    ax12.plot(x_array, y_array, c='black', linewidth=1, linestyle='--')
    ax12.plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    ax12.plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    ax12.text(1.25, 0.1, r"$N=3$", fontsize=15)
    for i, c in enumerate(Clusters):
        final_SrFe = []
        for srfe, C in zip(SrFe[c], Comments[c]):
            if not np.isnan(srfe) and not 'C' in C:
                final_SrFe.append(srfe) 
        if len(final_SrFe)==3:
            ax12.scatter(np.std(final_SrFe), len(Dispersion_array[Dispersion_array<np.std(final_SrFe)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100)

    filename = params.dispersion_distrib_file_mask[0] + 'BaFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = f["/dispersion"][3-params.dispersion_cluster_size[0]]
    N_samples = len(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array])
    ax13.set_xlabel(r"$\sigma$ ([Ba/Fe])")
    ax13.set_ylabel("Cumulative fraction")
    ax13.plot(x_array, y_array, c='black', linewidth=1, linestyle='--')
    ax13.plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    ax13.plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    ax13.text(1.15, 0.1, r"$N=3$", fontsize=15)
    for i, c in enumerate(Clusters):
        final_BaFe = []
        for bafe, C in zip(BaFe[c], Comments[c]):
            if not np.isnan(bafe) and not 'C' in C:
                final_BaFe.append(bafe) 
        if len(final_BaFe)==3:
            ax13.scatter(np.std(final_BaFe), len(Dispersion_array[Dispersion_array<np.std(final_BaFe)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100)

    filename = params.dispersion_distrib_file_mask[0] + 'EuFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = f["/dispersion"][3-params.dispersion_cluster_size[0]]
    N_samples = len(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array])
    ax14.set_xlabel(r"$\sigma$ ([Eu/Fe])")
    ax14.set_ylabel("Cumulative fraction")
    ax14.plot(x_array, y_array, c='black', linewidth=1, linestyle='--')
    ax14.plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    ax14.plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    ax14.text(0.6, 0.1, r"$N=3$", fontsize=15)
    for i, c in enumerate(Clusters):
        EuFe_nonan = EuFe[c]
        EuFe_nonan = EuFe_nonan[~np.isnan(EuFe_nonan)]
        if len(EuFe_nonan)==3:
            ax14.scatter(np.std(EuFe_nonan), len(Dispersion_array[Dispersion_array<np.std(EuFe_nonan)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100)

    filename = params.dispersion_distrib_file_mask[0] + 'FeH' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = []
    for i in range (4, 6):
        Dispersion_array = Dispersion_array + list(f["/dispersion"][i-params.dispersion_cluster_size[0]])
    Dispersion_array = np.array(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = []
    for i in range (4, 6):
        Dispersion_array = f["/dispersion"][i-params.dispersion_cluster_size[0]]
        y_array.append(np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array]))
    y_array = np.transpose(y_array)
    min_y_array = np.array([min(y) for y in y_array])
    max_y_array = np.array([max(y) for y in y_array])
    ax20.set_xlabel(r"$\sigma$ ([Fe/H])")
    ax20.set_ylabel("Cumulative fraction")
    ax20.plot(x_array, min_y_array, c='black', linewidth=1, linestyle='--')
    ax20.plot(x_array, max_y_array, c='black', linewidth=1, linestyle='--')
    ax20.plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    ax20.plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    ax20.text(0.6, 0.1, r"$N=4-5$", fontsize=15)
    for i, c in enumerate(Clusters):
        FeH_nonan = FeH[c]
        FeH_nonan = FeH_nonan[~np.isnan(FeH_nonan)]
        if len(FeH_nonan)>=4 and len(FeH_nonan)<6:
            Dispersion_array = f["/dispersion"][len(FeH_nonan)-params.dispersion_cluster_size[0]]
            ax20.scatter(np.std(FeH_nonan), len(Dispersion_array[Dispersion_array<biweight_scale(FeH_nonan)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100)

    filename = params.dispersion_distrib_file_mask[0] + 'cCFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = []
    for i in range (4, 6):
        Dispersion_array = Dispersion_array + list(f["/dispersion"][i-params.dispersion_cluster_size[0]])
    Dispersion_array = np.array(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = []
    for i in range (4, 6):
        Dispersion_array = f["/dispersion"][i-params.dispersion_cluster_size[0]]
        y_array.append(np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array]))
    y_array = np.transpose(y_array)
    min_y_array = np.array([min(y) for y in y_array])
    max_y_array = np.array([max(y) for y in y_array])
    ax21.set_xlabel(r"$\sigma$ ([C/Fe]$_\mathrm{c}$)")
    ax21.set_ylabel("Cumulative fraction")
    ax21.plot(x_array, min_y_array, c='black', linewidth=1, linestyle='--')
    ax21.plot(x_array, max_y_array, c='black', linewidth=1, linestyle='--')
    ax21.plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    ax21.plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    ax21.text(0.6, 0.1, r"$N=4-5$", fontsize=15)
    for i, c in enumerate(Clusters):
        final_cCFe = []
        for ccfe, C in zip(cCFe[c], Comments[c]):
            if not np.isnan(ccfe) and not 'C' in C:
                final_cCFe.append(ccfe) 
        if len(final_cCFe)>=4 and len(final_cCFe)<6:
            Dispersion_array = f["/dispersion"][len(final_cCFe)-params.dispersion_cluster_size[0]]
            ax21.scatter(np.std(final_cCFe), len(Dispersion_array[Dispersion_array<biweight_scale(final_cCFe)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100)

    filename = params.dispersion_distrib_file_mask[0] + 'SrFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = []
    for i in range (4, 6):
        Dispersion_array = Dispersion_array + list(f["/dispersion"][i-params.dispersion_cluster_size[0]])
    Dispersion_array = np.array(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = []
    for i in range (4, 6):
        Dispersion_array = f["/dispersion"][i-params.dispersion_cluster_size[0]]
        y_array.append(np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array]))
    y_array = np.transpose(y_array)
    min_y_array = np.array([min(y) for y in y_array])
    max_y_array = np.array([max(y) for y in y_array])
    ax22.set_xlabel(r"$\sigma$ ([Sr/Fe])")
    ax22.set_ylabel("Cumulative fraction")
    ax22.plot(x_array, min_y_array, c='black', linewidth=1, linestyle='--')
    ax22.plot(x_array, max_y_array, c='black', linewidth=1, linestyle='--')
    ax22.plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    ax22.plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    ax22.text(0.6, 0.1, r"$N=4-5$", fontsize=15)
    for i, c in enumerate(Clusters):
        final_SrFe = []
        for srfe, C in zip(SrFe[c], Comments[c]):
            if not np.isnan(srfe) and not 'C' in C:
                final_SrFe.append(srfe) 
        if len(final_SrFe)>=4 and len(final_SrFe)<6:
            Dispersion_array = f["/dispersion"][len(final_SrFe)-params.dispersion_cluster_size[0]]
            ax22.scatter(np.std(final_SrFe), len(Dispersion_array[Dispersion_array<biweight_scale(final_SrFe)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100)

    filename = params.dispersion_distrib_file_mask[0] + 'BaFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = []
    for i in range (4, 6):
        Dispersion_array = Dispersion_array + list(f["/dispersion"][i-params.dispersion_cluster_size[0]])
    Dispersion_array = np.array(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = []
    for i in range (4, 6):
        Dispersion_array = f["/dispersion"][i-params.dispersion_cluster_size[0]]
        y_array.append(np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array]))
    y_array = np.transpose(y_array)
    min_y_array = np.array([min(y) for y in y_array])
    max_y_array = np.array([max(y) for y in y_array])
    ax23.set_xlabel(r"$\sigma$ ([Ba/Fe])")
    ax23.set_ylabel("Cumulative fraction")
    ax23.plot(x_array, min_y_array, c='black', linewidth=1, linestyle='--')
    ax23.plot(x_array, max_y_array, c='black', linewidth=1, linestyle='--')
    ax23.plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    ax23.plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    ax23.text(0.6, 0.1, r"$N=4-5$", fontsize=15)
    for i, c in enumerate(Clusters):
        final_BaFe = []
        for bafe, C in zip(BaFe[c], Comments[c]):
            if not np.isnan(bafe) and not 'C' in C:
                final_BaFe.append(bafe)
        if len(final_BaFe)>=4 and len(final_BaFe)<6:
            Dispersion_array = f["/dispersion"][len(final_BaFe)-params.dispersion_cluster_size[0]]
            ax23.scatter(np.std(final_BaFe), len(Dispersion_array[Dispersion_array<biweight_scale(final_BaFe)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100)

    filename = params.dispersion_distrib_file_mask[0] + 'EuFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = []
    for i in range (4, 6):
        Dispersion_array = Dispersion_array + list(f["/dispersion"][i-params.dispersion_cluster_size[0]])
    Dispersion_array = np.array(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = []
    for i in range (4, 6):
        Dispersion_array = f["/dispersion"][i-params.dispersion_cluster_size[0]]
        y_array.append(np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array]))
    y_array = np.transpose(y_array)
    min_y_array = np.array([min(y) for y in y_array])
    max_y_array = np.array([max(y) for y in y_array])
    ax24.set_xlabel(r"$\sigma$ ([Eu/Fe])")
    ax24.set_ylabel("Cumulative fraction")
    ax24.plot(x_array, min_y_array, c='black', linewidth=1, linestyle='--')
    ax24.plot(x_array, max_y_array, c='black', linewidth=1, linestyle='--')
    ax24.plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    ax24.plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    ax24.text(0.5, 0.1, r"$N=4-5$", fontsize=15)
    for i, c in enumerate(Clusters):
        EuFe_nonan = EuFe[c]
        EuFe_nonan = EuFe_nonan[~np.isnan(EuFe_nonan)]
        if len(EuFe_nonan)>=4 and len(EuFe_nonan)<6:
            Dispersion_array = f["/dispersion"][len(EuFe_nonan)-params.dispersion_cluster_size[0]]
            ax24.scatter(np.std(EuFe_nonan), len(Dispersion_array[Dispersion_array<biweight_scale(EuFe_nonan)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100)

    filename = params.dispersion_distrib_file_mask[0] + 'FeH' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = []
    for i in range (7, 16):
        Dispersion_array = Dispersion_array + list(f["/dispersion"][i-params.dispersion_cluster_size[0]])
    Dispersion_array = np.array(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = []
    for i in range (7, 16):
        Dispersion_array = f["/dispersion"][i-params.dispersion_cluster_size[0]]
        y_array.append(np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array]))
    y_array = np.transpose(y_array)
    min_y_array = np.array([min(y) for y in y_array])
    max_y_array = np.array([max(y) for y in y_array])
    ax30.set_xlabel(r"$\sigma$ ([Fe/H])")
    ax30.set_ylabel("Cumulative fraction")
    ax30.plot(x_array, min_y_array, c='black', linewidth=1, linestyle='--')
    ax30.plot(x_array, max_y_array, c='black', linewidth=1, linestyle='--')
    ax30.plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    ax30.plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    ax30.text(0.5, 0.1, r"$N=7-15$", fontsize=15)
    for i, c in enumerate(Clusters):
        FeH_nonan = FeH[c]
        FeH_nonan = FeH_nonan[~np.isnan(FeH_nonan)]
        if len(FeH_nonan)>=7 and len(FeH_nonan)<16:
            Dispersion_array = f["/dispersion"][len(FeH_nonan)-params.dispersion_cluster_size[0]]
            ax30.scatter(np.std(FeH_nonan), len(Dispersion_array[Dispersion_array<biweight_scale(FeH_nonan)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100)

    filename = params.dispersion_distrib_file_mask[0] + 'cCFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = []
    for i in range (7, 16):
        Dispersion_array = Dispersion_array + list(f["/dispersion"][i-params.dispersion_cluster_size[0]])
    Dispersion_array = np.array(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = []
    for i in range (7, 16):
        Dispersion_array = f["/dispersion"][i-params.dispersion_cluster_size[0]]
        y_array.append(np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array]))
    y_array = np.transpose(y_array)
    min_y_array = np.array([min(y) for y in y_array])
    max_y_array = np.array([max(y) for y in y_array])
    ax31.set_xlabel(r"$\sigma$ ([C/Fe]$_\mathrm{c}$)")
    ax31.set_ylabel("Cumulative fraction")
    ax31.plot(x_array, min_y_array, c='black', linewidth=1, linestyle='--')
    ax31.plot(x_array, max_y_array, c='black', linewidth=1, linestyle='--')
    ax31.plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    ax31.plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    ax31.text(0.6, 0.1, r"$N=7-15$", fontsize=15)
    for i, c in enumerate(Clusters):
        final_cCFe = []
        for ccfe, C in zip(cCFe[c], Comments[c]):
            if not np.isnan(ccfe) and not 'C' in C:
                final_cCFe.append(ccfe) 
        if len(final_cCFe)>=7 and len(final_cCFe)<16:
            Dispersion_array = f["/dispersion"][len(final_cCFe)-params.dispersion_cluster_size[0]]
            ax31.scatter(np.std(final_cCFe), len(Dispersion_array[Dispersion_array<biweight_scale(final_cCFe)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100)

    filename = params.dispersion_distrib_file_mask[0] + 'SrFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = []
    for i in range (7, 16):
        Dispersion_array = Dispersion_array + list(f["/dispersion"][i-params.dispersion_cluster_size[0]])
    Dispersion_array = np.array(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = []
    for i in range (7, 16):
        Dispersion_array = f["/dispersion"][i-params.dispersion_cluster_size[0]]
        y_array.append(np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array]))
    y_array = np.transpose(y_array)
    min_y_array = np.array([min(y) for y in y_array])
    max_y_array = np.array([max(y) for y in y_array])
    ax32.set_xlabel(r"$\sigma$ ([Sr/Fe])")
    ax32.set_ylabel("Cumulative fraction")
    ax32.plot(x_array, min_y_array, c='black', linewidth=1, linestyle='--')
    ax32.plot(x_array, max_y_array, c='black', linewidth=1, linestyle='--')
    ax32.plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    ax32.plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    ax32.text(0.6, 0.1, r"$N=7-15$", fontsize=15)
    for i, c in enumerate(Clusters):
        final_SrFe = []
        for srfe, C in zip(SrFe[c], Comments[c]):
            if not np.isnan(srfe) and not 'C' in C:
                final_SrFe.append(srfe)
        if len(final_SrFe)>=7 and len(final_SrFe)<16:
            Dispersion_array = f["/dispersion"][len(final_SrFe)-params.dispersion_cluster_size[0]]
            ax32.scatter(np.std(final_SrFe), len(Dispersion_array[Dispersion_array<biweight_scale(final_SrFe)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100)

    filename = params.dispersion_distrib_file_mask[0] + 'BaFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = []
    for i in range (7, 16):
        Dispersion_array = Dispersion_array + list(f["/dispersion"][i-params.dispersion_cluster_size[0]])
    Dispersion_array = np.array(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = []
    for i in range (7, 16):
        Dispersion_array = f["/dispersion"][i-params.dispersion_cluster_size[0]]
        y_array.append(np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array]))
    y_array = np.transpose(y_array)
    min_y_array = np.array([min(y) for y in y_array])
    max_y_array = np.array([max(y) for y in y_array])
    ax33.set_xlabel(r"$\sigma$ ([Ba/Fe])")
    ax33.set_ylabel("Cumulative fraction")
    ax33.plot(x_array, min_y_array, c='black', linewidth=1, linestyle='--')
    ax33.plot(x_array, max_y_array, c='black', linewidth=1, linestyle='--')
    ax33.plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    ax33.plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    ax33.text(0.6, 0.1, r"$N=7-15$", fontsize=15)
    for i, c in enumerate(Clusters):
        final_BaFe = []
        for bafe, C in zip(BaFe[c], Comments[c]):
            if not np.isnan(bafe) and not 'C' in C:
                final_BaFe.append(bafe)
        if len(final_BaFe)>=7 and len(final_BaFe)<16:
            Dispersion_array = f["/dispersion"][len(final_BaFe)-params.dispersion_cluster_size[0]]
            ax33.scatter(np.std(final_BaFe), len(Dispersion_array[Dispersion_array<biweight_scale(final_BaFe)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100)

    filename = params.dispersion_distrib_file_mask[0] + 'EuFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = []
    for i in range (7, 16):
        Dispersion_array = Dispersion_array + list(f["/dispersion"][i-params.dispersion_cluster_size[0]])
    Dispersion_array = np.array(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = []
    for i in range (7, 16):
        Dispersion_array = f["/dispersion"][i-params.dispersion_cluster_size[0]]
        y_array.append(np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array]))
    y_array = np.transpose(y_array)
    min_y_array = np.array([min(y) for y in y_array])
    max_y_array = np.array([max(y) for y in y_array])
    ax34.set_xlabel(r"$\sigma$ ([Eu/Fe])")
    ax34.set_ylabel("Cumulative fraction")
    ax34.plot(x_array, min_y_array, c='black', linewidth=1, linestyle='--')
    ax34.plot(x_array, max_y_array, c='black', linewidth=1, linestyle='--')
    ax34.plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    ax34.plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    ax34.text(0.4, 0.1, r"$N=7-15$", fontsize=15)
    for i, c in enumerate(Clusters):
        EuFe_nonan = EuFe[c]
        EuFe_nonan = EuFe_nonan[~np.isnan(EuFe_nonan)]
        if len(EuFe_nonan)>=7 and len(EuFe_nonan)<16:
            Dispersion_array = f["/dispersion"][len(EuFe_nonan)-params.dispersion_cluster_size[0]]
            ax34.scatter(np.std(EuFe_nonan), len(Dispersion_array[Dispersion_array<biweight_scale(EuFe_nonan)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100)


    plt.gcf().set_size_inches(15, 12)
    plt.tight_layout()
    plt.gcf().savefig("plots/dispersions_paper_v2.eps", bbox_inches='tight',pad_inches=0.01, dpi=100, format='eps')
    plt.gcf().savefig("plots/dispersions_paper_v2.png", bbox_inches='tight',pad_inches=0.01, dpi=100, format='png')
    plt.close()



def plot_dispersions_paper_abundance(Clusters, FeH, CFec, EuFe, SrFe, BaFe, Comments):
    plt.clf()
    fig, axes = plt.subplots(nrows=2, ncols=2)

    indeces = [c[i] for c in Clusters for i in range(len(c))]

    axes[0,0].set_xlabel("[Fe/H]")
    axes[0,0].set_ylabel(r"[C/Fe]$_\mathrm{c}$")
    axes[0,0].set_xlim(-3.6,-1)
    axes[0,0].plot((-3.6,-1),(0,0), linewidth=1, color='black')
    axes[0,0].plot((-3.6,-1),(0.7,0.7), linewidth=1, linestyle='--', color='black')
    for i, c in enumerate(Clusters):
        final_FeH, final_CFec = [], []
        for feh, cfec, C in zip(FeH[c], CFec[c], Comments[c]):
            if not np.isnan(cfec) and not 'C' in C:
                final_FeH.append(feh)
                final_CFec.append(cfec) 
            else:
                final_FeH.append(np.nan)
                final_CFec.append(np.nan)
        axes[0,0].scatter(final_FeH, final_CFec, c=params.colors[i], marker=params.markers[i], s=30, linewidth=params.marker_sizes[i])
    #axes[0,0].scatter([-1.25],[1.5], color='black', marker='.', s=25)
    axes[0,0].plot([-3.25-0.15,-3.25+0.15], [-0.4,-0.4], linewidth=1, color='black')
    axes[0,0].plot([-3.25,-3.25], [-0.4-0.1,-0.4+0.1], linewidth=1, color='black')

    axes[0,1].set_xlabel("[Fe/H]")
    axes[0,1].set_ylabel("[Sr/Fe]")
    axes[0,1].set_xlim(-3.6,-1)
    axes[0,1].plot((-3.6,-1),(0,0), linewidth=1, color='black')
    #ax02.set_ylim(-1,1)
    for i, c in enumerate(Clusters):
        final_FeH, final_SrFe = [], []
        for feh, srfe, C in zip(FeH[c], SrFe[c], Comments[c]):
            if not np.isnan(srfe) and not 'Sr' in C:
                final_FeH.append(feh)
                final_SrFe.append(srfe)
            else:
                final_FeH.append(np.nan)
                final_SrFe.append(np.nan)
        axes[0,1].scatter(FeH[c], SrFe[c], c=params.colors[i], marker=params.markers[i], s=30, linewidth=params.marker_sizes[i])
    #axes[0,1].scatter([-3.3],[-3.3], color='black', marker='.', s=25)
    axes[0,1].plot([-3.25-0.15,-3.25+0.15], [-2.75,-2.75], linewidth=1, color='black')
    axes[0,1].plot([-3.25,-3.25], [-2.75-0.1,-2.75+0.1], linewidth=1, color='black')

    axes[1,0].set_xlabel("[Fe/H]")
    axes[1,0].set_ylabel("[Ba/Fe]")
    axes[1,0].set_xlim(-3.6,-1)
    axes[1,0].plot((-3.6,-1),(0,0), linewidth=1, color='black')
    for i, c in enumerate(Clusters):
        final_FeH, final_BaFe = [], []
        for feh, bafe, C in zip(FeH[c], BaFe[c], Comments[c]):
            if not np.isnan(bafe) and not 'Ba' in C:
                final_FeH.append(feh)
                final_BaFe.append(bafe) 
            else:
                final_FeH.append(np.nan)
                final_BaFe.append(np.nan)
        axes[1,0].scatter(final_FeH, final_BaFe, c=params.colors[i], marker=params.markers[i], s=30, linewidth=params.marker_sizes[i])
    #axes[1,0].scatter([-3.3],[-0.7], color='black', marker='.', s=25)
    axes[1,0].plot([-3.25-0.15,-3.25+0.15], [-0.7,-0.7], linewidth=1, color='black')
    axes[1,0].plot([-3.25,-3.25], [-0.7-0.1,-0.7+0.1], linewidth=1, color='black')
   
    axes[1,1].set_xlabel("[Fe/H]")
    axes[1,1].set_ylabel("[Eu/Fe]")
    axes[1,1].set_xlim(-3.6,-1)
    axes[1,1].plot((-3.6,-1),(0,0), linewidth=1, color='black')
    axes[1,1].plot((-3.6,-1),(0.3,0.3), linewidth=1, linestyle='--', color='black')
    axes[1,1].plot((-3.6,-1),(0.7,0.7), linewidth=1, linestyle='--', color='black')
    #ax04.set_ylim(-1,1)
    for i, c in enumerate(Clusters):
        axes[1,1].scatter(FeH[c], EuFe[c], c=params.colors[i], marker=params.markers[i], s=30, linewidth=params.marker_sizes[i])
    #axes[1,1].scatter([-3.4],[1.5], color='black', marker='.', s=25)
    axes[1,1].plot([-3.25-0.15,-3.25+0.15], [0.125,0.125], linewidth=1, color='black')
    axes[1,1].plot([-3.25,-3.25], [0.125-0.1,0.125+0.1], linewidth=1, color='black')

    plt.gcf().set_size_inches(10, 7)
    plt.tight_layout()
    plt.gcf().savefig("plots/dispersions_paper_abundance.eps", bbox_inches='tight',pad_inches=0.01, dpi=100, format='eps')
    plt.gcf().savefig("plots/dispersions_paper_abundance.png", bbox_inches='tight',pad_inches=0.01, dpi=100, format='png')
    plt.close()


def plot_dispersions_paper_cumuls(Clusters, FeH, CFec, EuFe, SrFe, BaFe, Comments):
    plt.clf()
    fig, axes = plt.subplots(nrows=3, ncols=5)

    indeces = [c[i] for c in Clusters for i in range(len(c))]

    filename = params.dispersion_distrib_file_mask[0] + 'FeH' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = f["/dispersion"][3-params.dispersion_cluster_size[0]]
    N_samples = len(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array])
    axes[0,0].set_xlabel(r"$\sigma$ ([Fe/H])")
    axes[0,0].set_ylabel("Cumulative fraction")
    axes[0,0].plot(x_array, y_array, c='black', linewidth=1, linestyle='--')
    axes[0,0].plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[0,0].plot(x_array, [0.33 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[0,0].plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[0,0].text(0.75, 0.1, r"$N=3$", fontsize=15)
    for i, c in enumerate(Clusters):
        FeH_nonan = FeH[c]
        FeH_nonan = FeH_nonan[~np.isnan(FeH_nonan)]
        if len(FeH_nonan)==3:
            axes[0,0].scatter(np.std(FeH_nonan), len(Dispersion_array[Dispersion_array<np.std(FeH_nonan)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100)

    filename = params.dispersion_distrib_file_mask[0] + 'cCFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = f["/dispersion"][3-params.dispersion_cluster_size[0]]
    N_samples = len(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array])
    axes[0,1].set_xlabel(r"$\sigma$ ([C/Fe]$_\mathrm{c}$)")
    axes[0,1].set_ylabel("Cumulative fraction")
    axes[0,1].plot(x_array, y_array, c='black', linewidth=1, linestyle='--')
    axes[0,1].plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[0,1].plot(x_array, [0.33 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[0,1].plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[0,1].text(0.6, 0.1, r"$N=3$", fontsize=15)
    for i, c in enumerate(Clusters):
        final_CFec = []
        for cfec, C in zip(CFec[c], Comments[c]):
            if not np.isnan(cfec) and not 'C' in C:
                final_CFec.append(cfec) 
        if len(final_CFec)==3:
            axes[0,1].scatter(np.std(final_CFec), len(Dispersion_array[Dispersion_array<np.std(final_CFec)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100)

    filename = params.dispersion_distrib_file_mask[0] + 'SrFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = f["/dispersion"][3-params.dispersion_cluster_size[0]]
    N_samples = len(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array])
    axes[0,2].set_xlabel(r"$\sigma$ ([Sr/Fe])")
    axes[0,2].set_ylabel("Cumulative fraction")
    axes[0,2].plot(x_array, y_array, c='black', linewidth=1, linestyle='--')
    axes[0,2].plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[0,2].plot(x_array, [0.33 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[0,2].plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[0,2].text(1.25, 0.1, r"$N=3$", fontsize=15)
    for i, c in enumerate(Clusters):
        final_SrFe = []
        for srfe, C in zip(SrFe[c], Comments[c]):
            if not np.isnan(srfe) and not 'Sr' in C:
                final_SrFe.append(srfe) 
        if len(final_SrFe)==3:
            axes[0,2].scatter(np.std(final_SrFe), len(Dispersion_array[Dispersion_array<np.std(final_SrFe)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100)

    filename = params.dispersion_distrib_file_mask[0] + 'BaFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = f["/dispersion"][3-params.dispersion_cluster_size[0]]
    N_samples = len(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array])
    axes[0,3].set_xlabel(r"$\sigma$ ([Ba/Fe])")
    axes[0,3].set_ylabel("Cumulative fraction")
    axes[0,3].plot(x_array, y_array, c='black', linewidth=1, linestyle='--')
    axes[0,3].plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[0,3].plot(x_array, [0.33 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[0,3].plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[0,3].text(1.15, 0.1, r"$N=3$", fontsize=15)
    for i, c in enumerate(Clusters):
        final_BaFe = []
        for bafe, C in zip(BaFe[c], Comments[c]):
            if not np.isnan(bafe) and not 'Ba' in C:
                final_BaFe.append(bafe) 
        if len(final_BaFe)==3:
            axes[0,3].scatter(np.std(final_BaFe), len(Dispersion_array[Dispersion_array<np.std(final_BaFe)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100)

    filename = params.dispersion_distrib_file_mask[0] + 'EuFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = f["/dispersion"][3-params.dispersion_cluster_size[0]]
    N_samples = len(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array])
    axes[0,4].set_xlabel(r"$\sigma$ ([Eu/Fe])")
    axes[0,4].set_ylabel("Cumulative fraction")
    axes[0,4].plot(x_array, y_array, c='black', linewidth=1, linestyle='--')
    axes[0,4].plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[0,4].plot(x_array, [0.33 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[0,4].plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[0,4].text(0.6, 0.1, r"$N=3$", fontsize=15)
    for i, c in enumerate(Clusters):
        EuFe_nonan = EuFe[c]
        EuFe_nonan = EuFe_nonan[~np.isnan(EuFe_nonan)]
        if len(EuFe_nonan)==3:
            axes[0,4].scatter(np.std(EuFe_nonan), len(Dispersion_array[Dispersion_array<np.std(EuFe_nonan)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100)

    filename = params.dispersion_distrib_file_mask[0] + 'FeH' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = []
    for i in range (4, 6):
        Dispersion_array = Dispersion_array + list(f["/dispersion"][i-params.dispersion_cluster_size[0]])
    Dispersion_array = np.array(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = []
    for i in range (4, 6):
        Dispersion_array = f["/dispersion"][i-params.dispersion_cluster_size[0]]
        y_array.append(np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array]))
    y_array = np.transpose(y_array)
    min_y_array = np.array([min(y) for y in y_array])
    max_y_array = np.array([max(y) for y in y_array])
    axes[1,0].set_xlabel(r"$\sigma$ ([Fe/H])")
    axes[1,0].set_ylabel("Cumulative fraction")
    axes[1,0].plot(x_array, min_y_array, c='black', linewidth=1, linestyle='--')
    axes[1,0].plot(x_array, max_y_array, c='black', linewidth=1, linestyle='--')
    axes[1,0].plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[1,0].plot(x_array, [0.33 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[1,0].plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[1,0].text(0.6, 0.1, r"$N=4-5$", fontsize=15)
    for i, c in enumerate(Clusters):
        FeH_nonan = FeH[c]
        FeH_nonan = FeH_nonan[~np.isnan(FeH_nonan)]
        if len(FeH_nonan)>=4 and len(FeH_nonan)<6:
            Dispersion_array = f["/dispersion"][len(FeH_nonan)-params.dispersion_cluster_size[0]]
            axes[1,0].scatter(biweight_scale(FeH_nonan), len(Dispersion_array[Dispersion_array<biweight_scale(FeH_nonan)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100)

    filename = params.dispersion_distrib_file_mask[0] + 'cCFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = []
    for i in range (4, 6):
        Dispersion_array = Dispersion_array + list(f["/dispersion"][i-params.dispersion_cluster_size[0]])
    Dispersion_array = np.array(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = []
    for i in range (4, 6):
        Dispersion_array = f["/dispersion"][i-params.dispersion_cluster_size[0]]
        y_array.append(np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array]))
    y_array = np.transpose(y_array)
    min_y_array = np.array([min(y) for y in y_array])
    max_y_array = np.array([max(y) for y in y_array])
    axes[1,1].set_xlabel(r"$\sigma$ ([C/Fe]$_\mathrm{c}$)")
    axes[1,1].set_ylabel("Cumulative fraction")
    axes[1,1].plot(x_array, min_y_array, c='black', linewidth=1, linestyle='--')
    axes[1,1].plot(x_array, max_y_array, c='black', linewidth=1, linestyle='--')
    axes[1,1].plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[1,1].plot(x_array, [0.33 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[1,1].plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[1,1].text(0.6, 0.1, r"$N=4-5$", fontsize=15)
    for i, c in enumerate(Clusters):
        final_CFec = []
        for cfec, C in zip(CFec[c], Comments[c]):
            if not np.isnan(cfec) and not 'C' in C:
                final_CFec.append(cfec) 
        if len(final_CFec)>=4 and len(final_CFec)<6:
            Dispersion_array = f["/dispersion"][len(final_CFec)-params.dispersion_cluster_size[0]]
            axes[1,1].scatter(biweight_scale(final_CFec), len(Dispersion_array[Dispersion_array<biweight_scale(final_CFec)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100)

    filename = params.dispersion_distrib_file_mask[0] + 'SrFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = []
    for i in range (4, 6):
        Dispersion_array = Dispersion_array + list(f["/dispersion"][i-params.dispersion_cluster_size[0]])
    Dispersion_array = np.array(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = []
    for i in range (4, 6):
        Dispersion_array = f["/dispersion"][i-params.dispersion_cluster_size[0]]
        y_array.append(np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array]))
    y_array = np.transpose(y_array)
    min_y_array = np.array([min(y) for y in y_array])
    max_y_array = np.array([max(y) for y in y_array])
    axes[1,2].set_xlabel(r"$\sigma$ ([Sr/Fe])")
    axes[1,2].set_ylabel("Cumulative fraction")
    axes[1,2].plot(x_array, min_y_array, c='black', linewidth=1, linestyle='--')
    axes[1,2].plot(x_array, max_y_array, c='black', linewidth=1, linestyle='--')
    axes[1,2].plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[1,2].plot(x_array, [0.33 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[1,2].plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[1,2].text(0.6, 0.1, r"$N=4-5$", fontsize=15)
    for i, c in enumerate(Clusters):
        final_SrFe = []
        for srfe, C in zip(SrFe[c], Comments[c]):
            if not np.isnan(srfe) and not 'Sr' in C:
                final_SrFe.append(srfe) 
        if len(final_SrFe)>=4 and len(final_SrFe)<6:
            Dispersion_array = f["/dispersion"][len(final_SrFe)-params.dispersion_cluster_size[0]]
            axes[1,2].scatter(biweight_scale(final_SrFe), len(Dispersion_array[Dispersion_array<biweight_scale(final_SrFe)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100)

    filename = params.dispersion_distrib_file_mask[0] + 'BaFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = []
    for i in range (4, 6):
        Dispersion_array = Dispersion_array + list(f["/dispersion"][i-params.dispersion_cluster_size[0]])
    Dispersion_array = np.array(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = []
    for i in range (4, 6):
        Dispersion_array = f["/dispersion"][i-params.dispersion_cluster_size[0]]
        y_array.append(np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array]))
    y_array = np.transpose(y_array)
    min_y_array = np.array([min(y) for y in y_array])
    max_y_array = np.array([max(y) for y in y_array])
    axes[1,3].set_xlabel(r"$\sigma$ ([Ba/Fe])")
    axes[1,3].set_ylabel("Cumulative fraction")
    axes[1,3].plot(x_array, min_y_array, c='black', linewidth=1, linestyle='--')
    axes[1,3].plot(x_array, max_y_array, c='black', linewidth=1, linestyle='--')
    axes[1,3].plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[1,3].plot(x_array, [0.33 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[1,3].plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[1,3].text(0.6, 0.1, r"$N=4-5$", fontsize=15)
    for i, c in enumerate(Clusters):
        final_BaFe = []
        for bafe, C in zip(BaFe[c], Comments[c]):
            if not np.isnan(bafe) and not 'Ba' in C:
                final_BaFe.append(bafe)
        if len(final_BaFe)>=4 and len(final_BaFe)<6:
            Dispersion_array = f["/dispersion"][len(final_BaFe)-params.dispersion_cluster_size[0]]
            axes[1,3].scatter(biweight_scale(final_BaFe), len(Dispersion_array[Dispersion_array<biweight_scale(final_BaFe)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100)

    filename = params.dispersion_distrib_file_mask[0] + 'EuFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = []
    for i in range (4, 6):
        Dispersion_array = Dispersion_array + list(f["/dispersion"][i-params.dispersion_cluster_size[0]])
    Dispersion_array = np.array(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = []
    for i in range (4, 6):
        Dispersion_array = f["/dispersion"][i-params.dispersion_cluster_size[0]]
        y_array.append(np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array]))
    y_array = np.transpose(y_array)
    min_y_array = np.array([min(y) for y in y_array])
    max_y_array = np.array([max(y) for y in y_array])
    axes[1,4].set_xlabel(r"$\sigma$ ([Eu/Fe])")
    axes[1,4].set_ylabel("Cumulative fraction")
    axes[1,4].plot(x_array, min_y_array, c='black', linewidth=1, linestyle='--')
    axes[1,4].plot(x_array, max_y_array, c='black', linewidth=1, linestyle='--')
    axes[1,4].plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[1,4].plot(x_array, [0.33 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[1,4].plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[1,4].text(0.5, 0.1, r"$N=4-5$", fontsize=15)
    for i, c in enumerate(Clusters):
        EuFe_nonan = EuFe[c]
        EuFe_nonan = EuFe_nonan[~np.isnan(EuFe_nonan)]
        if len(EuFe_nonan)>=4 and len(EuFe_nonan)<6:
            Dispersion_array = f["/dispersion"][len(EuFe_nonan)-params.dispersion_cluster_size[0]]
            axes[1,4].scatter(biweight_scale(EuFe_nonan), len(Dispersion_array[Dispersion_array<biweight_scale(EuFe_nonan)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100)

    filename = params.dispersion_distrib_file_mask[0] + 'FeH' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = []
    for i in range (9, 46):
        Dispersion_array = Dispersion_array + list(f["/dispersion"][i-params.dispersion_cluster_size[0]])
    Dispersion_array = np.array(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = []
    for i in range (9, 46):
        Dispersion_array = f["/dispersion"][i-params.dispersion_cluster_size[0]]
        y_array.append(np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array]))
    y_array = np.transpose(y_array)
    min_y_array = np.array([min(y) for y in y_array])
    max_y_array = np.array([max(y) for y in y_array])
    axes[2,0].set_xlabel(r"$\sigma$ ([Fe/H])")
    axes[2,0].set_ylabel("Cumulative fraction")
    axes[2,0].plot(x_array, min_y_array, c='black', linewidth=1, linestyle='--')
    axes[2,0].plot(x_array, max_y_array, c='black', linewidth=1, linestyle='--')
    axes[2,0].plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[2,0].plot(x_array, [0.33 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[2,0].plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[2,0].text(0.45, 0.1, r"$N=9-45$", fontsize=15)
    for i, c in enumerate(Clusters):
        FeH_nonan = FeH[c]
        FeH_nonan = FeH_nonan[~np.isnan(FeH_nonan)]
        if len(FeH_nonan)>=9 and len(FeH_nonan)<46:
            Dispersion_array = f["/dispersion"][len(FeH_nonan)-params.dispersion_cluster_size[0]]
            axes[2,0].scatter(biweight_scale(FeH_nonan), len(Dispersion_array[Dispersion_array<biweight_scale(FeH_nonan)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100)

    filename = params.dispersion_distrib_file_mask[0] + 'cCFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = []
    for i in range (9, 46):
        Dispersion_array = Dispersion_array + list(f["/dispersion"][i-params.dispersion_cluster_size[0]])
    Dispersion_array = np.array(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = []
    for i in range (9, 46):
        Dispersion_array = f["/dispersion"][i-params.dispersion_cluster_size[0]]
        y_array.append(np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array]))
    y_array = np.transpose(y_array)
    min_y_array = np.array([min(y) for y in y_array])
    max_y_array = np.array([max(y) for y in y_array])
    axes[2,1].set_xlabel(r"$\sigma$ ([C/Fe]c)")
    axes[2,1].set_ylabel("Cumulative fraction")
    axes[2,1].plot(x_array, min_y_array, c='black', linewidth=1, linestyle='--')
    axes[2,1].plot(x_array, max_y_array, c='black', linewidth=1, linestyle='--')
    axes[2,1].plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[2,1].plot(x_array, [0.33 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[2,1].plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[2,1].text(0.45, 0.1, r"$N=9-45$", fontsize=15)
    for i, c in enumerate(Clusters):
        final_CFec = []
        for cfec, C in zip(CFec[c], Comments[c]):
            if not np.isnan(cfec) and not 'C' in C:
                final_CFec.append(cfec) 
        if len(final_CFec)>=9 and len(final_CFec)<46:
            Dispersion_array = f["/dispersion"][len(final_CFec)-params.dispersion_cluster_size[0]]
            axes[2,1].scatter(biweight_scale(final_CFec), len(Dispersion_array[Dispersion_array<biweight_scale(final_CFec)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100)

    filename = params.dispersion_distrib_file_mask[0] + 'SrFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = []
    for i in range (9, 46):
        Dispersion_array = Dispersion_array + list(f["/dispersion"][i-params.dispersion_cluster_size[0]])
    Dispersion_array = np.array(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = []
    for i in range (9, 46):
        Dispersion_array = f["/dispersion"][i-params.dispersion_cluster_size[0]]
        y_array.append(np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array]))
    y_array = np.transpose(y_array)
    min_y_array = np.array([min(y) for y in y_array])
    max_y_array = np.array([max(y) for y in y_array])
    axes[2,2].set_xlabel(r"$\sigma$ ([Sr/Fe])")
    axes[2,2].set_ylabel("Cumulative fraction")
    axes[2,2].plot(x_array, min_y_array, c='black', linewidth=1, linestyle='--')
    axes[2,2].plot(x_array, max_y_array, c='black', linewidth=1, linestyle='--')
    axes[2,2].plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[2,2].plot(x_array, [0.33 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[2,2].plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[2,2].text(0.45, 0.1, r"$N=9-45$", fontsize=15)
    for i, c in enumerate(Clusters):
        final_SrFe = []
        for srfe, C in zip(SrFe[c], Comments[c]):
            if not np.isnan(srfe) and not 'Sr' in C:
                final_SrFe.append(srfe)
        if len(final_SrFe)>=9 and len(final_SrFe)<46:
            Dispersion_array = f["/dispersion"][len(final_SrFe)-params.dispersion_cluster_size[0]]
            axes[2,2].scatter(biweight_scale(final_SrFe), len(Dispersion_array[Dispersion_array<biweight_scale(final_SrFe)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100)

    filename = params.dispersion_distrib_file_mask[0] + 'BaFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = []
    for i in range (9, 46):
        Dispersion_array = Dispersion_array + list(f["/dispersion"][i-params.dispersion_cluster_size[0]])
    Dispersion_array = np.array(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = []
    for i in range (9, 46):
        Dispersion_array = f["/dispersion"][i-params.dispersion_cluster_size[0]]
        y_array.append(np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array]))
    y_array = np.transpose(y_array)
    min_y_array = np.array([min(y) for y in y_array])
    max_y_array = np.array([max(y) for y in y_array])
    axes[2,3].set_xlabel(r"$\sigma$ ([Ba/Fe])")
    axes[2,3].set_ylabel("Cumulative fraction")
    axes[2,3].plot(x_array, min_y_array, c='black', linewidth=1, linestyle='--')
    axes[2,3].plot(x_array, max_y_array, c='black', linewidth=1, linestyle='--')
    axes[2,3].plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[2,3].plot(x_array, [0.33 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[2,3].plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[2,3].text(0.4, 0.1, r"$N=9-45$", fontsize=15)
    for i, c in enumerate(Clusters):
        final_BaFe = []
        for bafe, C in zip(BaFe[c], Comments[c]):
            if not np.isnan(bafe) and not 'Ba' in C:
                final_BaFe.append(bafe)
        if len(final_BaFe)>=9 and len(final_BaFe)<46:
            Dispersion_array = f["/dispersion"][len(final_BaFe)-params.dispersion_cluster_size[0]]
            axes[2,3].scatter(biweight_scale(final_BaFe), len(Dispersion_array[Dispersion_array<biweight_scale(final_BaFe)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100)

    filename = params.dispersion_distrib_file_mask[0] + 'EuFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = []
    for i in range (9, 46):
        Dispersion_array = Dispersion_array + list(f["/dispersion"][i-params.dispersion_cluster_size[0]])
    Dispersion_array = np.array(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = []
    for i in range (9, 46):
        Dispersion_array = f["/dispersion"][i-params.dispersion_cluster_size[0]]
        y_array.append(np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array]))
    y_array = np.transpose(y_array)
    min_y_array = np.array([min(y) for y in y_array])
    max_y_array = np.array([max(y) for y in y_array])
    axes[2,4].set_xlabel(r"$\sigma$ ([Eu/Fe])")
    axes[2,4].set_ylabel("Cumulative fraction")
    axes[2,4].plot(x_array, min_y_array, c='black', linewidth=1, linestyle='--')
    axes[2,4].plot(x_array, max_y_array, c='black', linewidth=1, linestyle='--')
    axes[2,4].plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[2,4].plot(x_array, [0.33 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[2,4].plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[2,4].text(0.35, 0.1, r"$N=9-45$", fontsize=15)
    for i, c in enumerate(Clusters):
        EuFe_nonan = EuFe[c]
        EuFe_nonan = EuFe_nonan[~np.isnan(EuFe_nonan)]
        if len(EuFe_nonan)>=9 and len(EuFe_nonan)<46:
            Dispersion_array = f["/dispersion"][len(EuFe_nonan)-params.dispersion_cluster_size[0]]
            axes[2,4].scatter(biweight_scale(EuFe_nonan), len(Dispersion_array[Dispersion_array<biweight_scale(EuFe_nonan)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100)


    plt.gcf().set_size_inches(15, 10)
    plt.tight_layout()
    plt.gcf().savefig("plots/dispersions_paper_cumuls.eps", bbox_inches='tight',pad_inches=0.01, dpi=100, format='eps')
    plt.gcf().savefig("plots/dispersions_paper_cumuls.png", bbox_inches='tight',pad_inches=0.01, dpi=100, format='png')
    plt.close()


def plot_appendix_cumul_paper(Clusters, FeH, Comments):
    plt.clf()

    N1_loc_x = 0.1
    N1_loc_y = 0.0

    N2_loc_x = 0.25
    N2_loc_y = 0.15

    N3_loc_x = 0.4
    N3_loc_y = 0.2

    N_loc_x = 1.0
    N_loc_y = 0.7
    
    indeces = [c[i] for c in Clusters for i in range(len(c))]

    filename = params.dispersion_distrib_file_mask[0] + 'FeH' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = f["/dispersion"][3-params.dispersion_cluster_size[0]]
    N_samples = len(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array])
    plt.xlabel(r"$\sigma$ ([X/Y])")
    plt.ylabel("Cumulative fraction")
    plt.xticks([]) 
    plt.plot(x_array, y_array, c='black', linewidth=1, linestyle='-')
    plt.plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    plt.plot(x_array, [0.33 for x in x_array], c='black', linewidth=1, linestyle='--')
    plt.plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')

    x_025, x_033, x_05 = 0, 0, 0
    passed_025, passed_033, passed_05 = False, False, False
    for x in x_array:
        if not passed_025: 
            if len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array)>=0.25:
                passed_025 = True
                x_025 = x
        if not passed_033: 
            if len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array)>=0.33:
                passed_033 = True
                x_033 = x
        if not passed_05: 
            if len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array)>=0.5:
                passed_05 = True
                x_05 = x
        if passed_025 and passed_033 and passed_05:
            break

    plt.plot((x_05,x_05), (0,len(Dispersion_array[Dispersion_array<x_05])/len(Dispersion_array)), c='red', linewidth=1, linestyle='--')
    plt.plot((x_033,x_033), (0,len(Dispersion_array[Dispersion_array<x_033])/len(Dispersion_array)), c='blue', linewidth=1, linestyle='--')
    plt.plot((x_025,x_025), (0,len(Dispersion_array[Dispersion_array<x_025])/len(Dispersion_array)), c='orange', linewidth=1, linestyle='--')
    plt.plot((1.2,1.2), (0,1), c='black', linewidth=1, linestyle='--')

    plt.text(N1_loc_x, N1_loc_y, r"$N_1$", color='orange', ha='left', va='bottom', fontsize=15)
    plt.text(N2_loc_x, N2_loc_y, r"$N_2$", color='blue', ha='left', va='bottom', fontsize=15)
    plt.text(N3_loc_x, N3_loc_y, r"$N_3$", color='red', ha='left', va='bottom', fontsize=15)
    plt.text(N_loc_x, N_loc_y, r"$N$", color='black', ha='left', va='bottom', fontsize=15)

    plt.gcf().set_size_inches(5, 5)
    plt.tight_layout()
    plt.gcf().savefig("plots/appendix_cumul_paper.eps", bbox_inches='tight',pad_inches=0.01, dpi=100, format='eps')
    plt.gcf().savefig("plots/appendix_cumul_paper.png", bbox_inches='tight',pad_inches=0.01, dpi=100, format='png')
    plt.close()


def plot_dispersions_paper_cumuls_flipped(Clusters, FeH, CFec, EuFe, SrFe, BaFe, Comments):
    plt.clf()
    fig, axes = plt.subplots(nrows=5, ncols=3)

    abundance_text_loc_x = 0.95
    numbers_text_loc_x = 0.95

    abundance_text_loc_y = 0.8
    numbers_text_loc_y = 0.1

    indeces = [c[i] for c in Clusters for i in range(len(c))]

    filename = params.dispersion_distrib_file_mask[0] + 'FeH' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = f["/dispersion"][3-params.dispersion_cluster_size[0]]
    N_samples = len(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array])
    axes[0,0].set_xlim(0, max(x_array))
    axes[0,0].set_ylim(0, 1)
    axes[0,0].set_xlabel(r"$\sigma$ ([Fe/H])")
    axes[0,0].set_ylabel("Cumulative fraction")
    axes[0,0].plot(x_array, y_array, c='black', linewidth=1, linestyle='--')
    axes[0,0].plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[0,0].plot(x_array, [0.33 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[0,0].plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[0,0].text(max(x_array)*numbers_text_loc_x, numbers_text_loc_y, r"$N=3$", ha='right', va='bottom', fontsize=15)
    axes[0,0].text(max(x_array)*abundance_text_loc_x, abundance_text_loc_y, "[Fe/H]", ha='right', va='bottom', fontsize=13)
    for i, c in enumerate(Clusters):
        FeH_nonan = FeH[c]
        FeH_nonan = FeH_nonan[~np.isnan(FeH_nonan)]
        if len(FeH_nonan)==3:
            axes[0,0].scatter(np.std(FeH_nonan), len(Dispersion_array[Dispersion_array<np.std(FeH_nonan)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100*params.marker_scales[i], linewidth=params.marker_linewidths[i])

    filename = params.dispersion_distrib_file_mask[0] + 'cCFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = f["/dispersion"][3-params.dispersion_cluster_size[0]]
    N_samples = len(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array])
    axes[1,0].set_xlim(0, max(x_array))
    axes[1,0].set_ylim(0, 1)
    axes[1,0].set_xlabel(r"$\sigma$ ([C/Fe]$_\mathrm{c}$)")
    axes[1,0].set_ylabel("Cumulative fraction")
    axes[1,0].plot(x_array, y_array, c='black', linewidth=1, linestyle='--')
    axes[1,0].plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[1,0].plot(x_array, [0.33 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[1,0].plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[1,0].text(max(x_array)*numbers_text_loc_x, numbers_text_loc_y, r"$N=3$", ha='right', va='bottom', fontsize=15)
    axes[1,0].text(max(x_array)*abundance_text_loc_x, abundance_text_loc_y, r"[C/Fe]$_\mathrm{c}$", ha='right', va='bottom', fontsize=13)
    for i, c in enumerate(Clusters):
        final_CFec = []
        for cfec, C in zip(CFec[c], Comments[c]):
            if not np.isnan(cfec) and not 'C' in C:
                final_CFec.append(cfec) 
        if len(final_CFec)==3:
            axes[1,0].scatter(np.std(final_CFec), len(Dispersion_array[Dispersion_array<np.std(final_CFec)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100*params.marker_scales[i], linewidth=params.marker_linewidths[i])

    filename = params.dispersion_distrib_file_mask[0] + 'SrFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = f["/dispersion"][3-params.dispersion_cluster_size[0]]
    N_samples = len(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array])
    axes[2,0].set_xlim(0, max(x_array))
    axes[2,0].set_ylim(0, 1)
    axes[2,0].set_xlabel(r"$\sigma$ ([Sr/Fe])")
    axes[2,0].set_ylabel("Cumulative fraction")
    axes[2,0].plot(x_array, y_array, c='black', linewidth=1, linestyle='--')
    axes[2,0].plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[2,0].plot(x_array, [0.33 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[2,0].plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[2,0].text(max(x_array)*numbers_text_loc_x, numbers_text_loc_y, r"$N=3$", ha='right', va='bottom', fontsize=15)
    axes[2,0].text(max(x_array)*abundance_text_loc_x, abundance_text_loc_y, "[Sr/Fe]", ha='right', va='bottom', fontsize=13)
    for i, c in enumerate(Clusters):
        final_SrFe = []
        for srfe, C in zip(SrFe[c], Comments[c]):
            if not np.isnan(srfe) and not 'Sr' in C:
                final_SrFe.append(srfe) 
        if len(final_SrFe)==3:
            axes[2,0].scatter(np.std(final_SrFe), len(Dispersion_array[Dispersion_array<np.std(final_SrFe)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100*params.marker_scales[i], linewidth=params.marker_linewidths[i])

    filename = params.dispersion_distrib_file_mask[0] + 'BaFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = f["/dispersion"][3-params.dispersion_cluster_size[0]]
    N_samples = len(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array])
    axes[3,0].set_xlim(0, max(x_array))
    axes[3,0].set_ylim(0, 1)
    axes[3,0].set_xlabel(r"$\sigma$ ([Ba/Fe])")
    axes[3,0].set_ylabel("Cumulative fraction")
    axes[3,0].plot(x_array, y_array, c='black', linewidth=1, linestyle='--')
    axes[3,0].plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[3,0].plot(x_array, [0.33 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[3,0].plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[3,0].text(max(x_array)*numbers_text_loc_x, numbers_text_loc_y, r"$N=3$", ha='right', va='bottom', fontsize=15)
    axes[3,0].text(max(x_array)*abundance_text_loc_x, abundance_text_loc_y, "[Ba/Fe]", ha='right', va='bottom', fontsize=13)
    for i, c in enumerate(Clusters):
        final_BaFe = []
        for bafe, C in zip(BaFe[c], Comments[c]):
            if not np.isnan(bafe) and not 'Ba' in C:
                final_BaFe.append(bafe) 
        if len(final_BaFe)==3:
            axes[3,0].scatter(np.std(final_BaFe), len(Dispersion_array[Dispersion_array<np.std(final_BaFe)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100*params.marker_scales[i], linewidth=params.marker_linewidths[i])

    filename = params.dispersion_distrib_file_mask[0] + 'EuFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = f["/dispersion"][3-params.dispersion_cluster_size[0]]
    N_samples = len(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array])
    axes[4,0].set_xlim(0, max(x_array))
    axes[4,0].set_ylim(0, 1)
    axes[4,0].set_xlabel(r"$\sigma$ ([Eu/Fe])")
    axes[4,0].set_ylabel("Cumulative fraction")
    axes[4,0].plot(x_array, y_array, c='black', linewidth=1, linestyle='--')
    axes[4,0].plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[4,0].plot(x_array, [0.33 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[4,0].plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[4,0].text(max(x_array)*numbers_text_loc_x, numbers_text_loc_y, r"$N=3$", ha='right', va='bottom', fontsize=15)
    axes[4,0].text(max(x_array)*abundance_text_loc_x, abundance_text_loc_y, "[Eu/Fe]", ha='right', va='bottom', fontsize=13)
    for i, c in enumerate(Clusters):
        EuFe_nonan = EuFe[c]
        EuFe_nonan = EuFe_nonan[~np.isnan(EuFe_nonan)]
        if len(EuFe_nonan)==3:
            axes[4,0].scatter(np.std(EuFe_nonan), len(Dispersion_array[Dispersion_array<np.std(EuFe_nonan)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100*params.marker_scales[i], linewidth=params.marker_linewidths[i])

    filename = params.dispersion_distrib_file_mask[0] + 'FeH' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = []
    for i in range (4, 6):
        Dispersion_array = Dispersion_array + list(f["/dispersion"][i-params.dispersion_cluster_size[0]])
    Dispersion_array = np.array(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = []
    for i in range (4, 6):
        Dispersion_array = f["/dispersion"][i-params.dispersion_cluster_size[0]]
        y_array.append(np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array]))
    y_array = np.transpose(y_array)
    min_y_array = np.array([min(y) for y in y_array])
    max_y_array = np.array([max(y) for y in y_array])
    axes[0,1].set_xlim(0, max(x_array))
    axes[0,1].set_ylim(0, 1)
    axes[0,1].set_xlabel(r"$\sigma$ ([Fe/H])")
    axes[0,1].set_ylabel("Cumulative fraction")
    axes[0,1].plot(x_array, min_y_array, c='black', linewidth=1, linestyle='--')
    axes[0,1].plot(x_array, max_y_array, c='black', linewidth=1, linestyle='--')
    axes[0,1].plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[0,1].plot(x_array, [0.33 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[0,1].plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[0,1].text(max(x_array)*numbers_text_loc_x, numbers_text_loc_y, r"$N=4-5$", ha='right', va='bottom', fontsize=15)
    axes[0,1].text(max(x_array)*abundance_text_loc_x, abundance_text_loc_y, "[Fe/H]", ha='right', va='bottom', fontsize=13)
    for i, c in enumerate(Clusters):
        FeH_nonan = FeH[c]
        FeH_nonan = FeH_nonan[~np.isnan(FeH_nonan)]
        if len(FeH_nonan)>=4 and len(FeH_nonan)<6:
            Dispersion_array = f["/dispersion"][len(FeH_nonan)-params.dispersion_cluster_size[0]]
            axes[0,1].scatter(biweight_scale(FeH_nonan), len(Dispersion_array[Dispersion_array<biweight_scale(FeH_nonan)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100*params.marker_scales[i], linewidth=params.marker_linewidths[i])

    filename = params.dispersion_distrib_file_mask[0] + 'cCFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = []
    for i in range (4, 6):
        Dispersion_array = Dispersion_array + list(f["/dispersion"][i-params.dispersion_cluster_size[0]])
    Dispersion_array = np.array(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = []
    for i in range (4, 6):
        Dispersion_array = f["/dispersion"][i-params.dispersion_cluster_size[0]]
        y_array.append(np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array]))
    y_array = np.transpose(y_array)
    min_y_array = np.array([min(y) for y in y_array])
    max_y_array = np.array([max(y) for y in y_array])
    axes[1,1].set_xlim(0, max(x_array))
    axes[1,1].set_ylim(0, 1)
    axes[1,1].set_xlabel(r"$\sigma$ ([C/Fe]$_\mathrm{c}$)")
    axes[1,1].set_ylabel("Cumulative fraction")
    axes[1,1].plot(x_array, min_y_array, c='black', linewidth=1, linestyle='--')
    axes[1,1].plot(x_array, max_y_array, c='black', linewidth=1, linestyle='--')
    axes[1,1].plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[1,1].plot(x_array, [0.33 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[1,1].plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[1,1].text(max(x_array)*numbers_text_loc_x, numbers_text_loc_y, r"$N=4-5$", ha='right', va='bottom', fontsize=15)
    axes[1,1].text(max(x_array)*abundance_text_loc_x, abundance_text_loc_y, r"[C/Fe]$_\mathrm{c}$", ha='right', va='bottom', fontsize=13)
    for i, c in enumerate(Clusters):
        final_CFec = []
        for cfec, C in zip(CFec[c], Comments[c]):
            if not np.isnan(cfec) and not 'C' in C:
                final_CFec.append(cfec) 
        if len(final_CFec)>=4 and len(final_CFec)<6:
            Dispersion_array = f["/dispersion"][len(final_CFec)-params.dispersion_cluster_size[0]]
            axes[1,1].scatter(biweight_scale(final_CFec), len(Dispersion_array[Dispersion_array<biweight_scale(final_CFec)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100*params.marker_scales[i], linewidth=params.marker_linewidths[i])

    filename = params.dispersion_distrib_file_mask[0] + 'SrFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = []
    for i in range (4, 6):
        Dispersion_array = Dispersion_array + list(f["/dispersion"][i-params.dispersion_cluster_size[0]])
    Dispersion_array = np.array(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = []
    for i in range (4, 6):
        Dispersion_array = f["/dispersion"][i-params.dispersion_cluster_size[0]]
        y_array.append(np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array]))
    y_array = np.transpose(y_array)
    min_y_array = np.array([min(y) for y in y_array])
    max_y_array = np.array([max(y) for y in y_array])
    axes[2,1].set_xlim(0, max(x_array))
    axes[2,1].set_ylim(0, 1)
    axes[2,1].set_xlabel(r"$\sigma$ ([Sr/Fe])")
    axes[2,1].set_ylabel("Cumulative fraction")
    axes[2,1].plot(x_array, min_y_array, c='black', linewidth=1, linestyle='--')
    axes[2,1].plot(x_array, max_y_array, c='black', linewidth=1, linestyle='--')
    axes[2,1].plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[2,1].plot(x_array, [0.33 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[2,1].plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[2,1].text(max(x_array)*numbers_text_loc_x, numbers_text_loc_y, r"$N=4-5$", ha='right', va='bottom', fontsize=15)
    axes[2,1].text(max(x_array)*abundance_text_loc_x, abundance_text_loc_y, "[Sr/Fe]", ha='right', va='bottom', fontsize=13)
    for i, c in enumerate(Clusters):
        final_SrFe = []
        for srfe, C in zip(SrFe[c], Comments[c]):
            if not np.isnan(srfe) and not 'Sr' in C:
                final_SrFe.append(srfe) 
        if len(final_SrFe)>=4 and len(final_SrFe)<6:
            Dispersion_array = f["/dispersion"][len(final_SrFe)-params.dispersion_cluster_size[0]]
            axes[2,1].scatter(biweight_scale(final_SrFe), len(Dispersion_array[Dispersion_array<biweight_scale(final_SrFe)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100*params.marker_scales[i], linewidth=params.marker_linewidths[i])

    filename = params.dispersion_distrib_file_mask[0] + 'BaFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = []
    for i in range (4, 6):
        Dispersion_array = Dispersion_array + list(f["/dispersion"][i-params.dispersion_cluster_size[0]])
    Dispersion_array = np.array(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = []
    for i in range (4, 6):
        Dispersion_array = f["/dispersion"][i-params.dispersion_cluster_size[0]]
        y_array.append(np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array]))
    y_array = np.transpose(y_array)
    min_y_array = np.array([min(y) for y in y_array])
    max_y_array = np.array([max(y) for y in y_array])
    axes[3,1].set_xlim(0, max(x_array))
    axes[3,1].set_ylim(0, 1)
    axes[3,1].set_xlabel(r"$\sigma$ ([Ba/Fe])")
    axes[3,1].set_ylabel("Cumulative fraction")
    axes[3,1].plot(x_array, min_y_array, c='black', linewidth=1, linestyle='--')
    axes[3,1].plot(x_array, max_y_array, c='black', linewidth=1, linestyle='--')
    axes[3,1].plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[3,1].plot(x_array, [0.33 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[3,1].plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[3,1].text(max(x_array)*numbers_text_loc_x, numbers_text_loc_y, r"$N=4-5$", ha='right', va='bottom', fontsize=15)
    axes[3,1].text(max(x_array)*abundance_text_loc_x, abundance_text_loc_y, "[Ba/Fe]", ha='right', va='bottom', fontsize=13)
    for i, c in enumerate(Clusters):
        final_BaFe = []
        for bafe, C in zip(BaFe[c], Comments[c]):
            if not np.isnan(bafe) and not 'Ba' in C:
                final_BaFe.append(bafe)
        if len(final_BaFe)>=4 and len(final_BaFe)<6:
            Dispersion_array = f["/dispersion"][len(final_BaFe)-params.dispersion_cluster_size[0]]
            axes[3,1].scatter(biweight_scale(final_BaFe), len(Dispersion_array[Dispersion_array<biweight_scale(final_BaFe)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100*params.marker_scales[i], linewidth=params.marker_linewidths[i])

    filename = params.dispersion_distrib_file_mask[0] + 'EuFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = []
    for i in range (4, 6):
        Dispersion_array = Dispersion_array + list(f["/dispersion"][i-params.dispersion_cluster_size[0]])
    Dispersion_array = np.array(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = []
    for i in range (4, 6):
        Dispersion_array = f["/dispersion"][i-params.dispersion_cluster_size[0]]
        y_array.append(np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array]))
    y_array = np.transpose(y_array)
    min_y_array = np.array([min(y) for y in y_array])
    max_y_array = np.array([max(y) for y in y_array])
    axes[4,1].set_xlim(0, max(x_array))
    axes[4,1].set_ylim(0, 1)
    axes[4,1].set_xlabel(r"$\sigma$ ([Eu/Fe])")
    axes[4,1].set_ylabel("Cumulative fraction")
    axes[4,1].plot(x_array, min_y_array, c='black', linewidth=1, linestyle='--')
    axes[4,1].plot(x_array, max_y_array, c='black', linewidth=1, linestyle='--')
    axes[4,1].plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[4,1].plot(x_array, [0.33 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[4,1].plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[4,1].text(max(x_array)*numbers_text_loc_x, numbers_text_loc_y, r"$N=4-5$", ha='right', va='bottom', fontsize=15)
    axes[4,1].text(max(x_array)*abundance_text_loc_x, abundance_text_loc_y, "[Eu/Fe]", ha='right', va='bottom', fontsize=13)
    for i, c in enumerate(Clusters):
        EuFe_nonan = EuFe[c]
        EuFe_nonan = EuFe_nonan[~np.isnan(EuFe_nonan)]
        if len(EuFe_nonan)>=4 and len(EuFe_nonan)<6:
            Dispersion_array = f["/dispersion"][len(EuFe_nonan)-params.dispersion_cluster_size[0]]
            axes[4,1].scatter(biweight_scale(EuFe_nonan), len(Dispersion_array[Dispersion_array<biweight_scale(EuFe_nonan)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100*params.marker_scales[i], linewidth=params.marker_linewidths[i])

    filename = params.dispersion_distrib_file_mask[0] + 'FeH' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = []
    for i in range (6, 13):
        Dispersion_array = Dispersion_array + list(f["/dispersion"][i-params.dispersion_cluster_size[0]])
    Dispersion_array = np.array(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = []
    for i in range (6, 13):
        Dispersion_array = f["/dispersion"][i-params.dispersion_cluster_size[0]]
        y_array.append(np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array]))
    y_array = np.transpose(y_array)
    min_y_array = np.array([min(y) for y in y_array])
    max_y_array = np.array([max(y) for y in y_array])
    axes[0,2].set_xlim(0, max(x_array))
    axes[0,2].set_ylim(0, 1)
    axes[0,2].set_xlabel(r"$\sigma$ ([Fe/H])")
    axes[0,2].set_ylabel("Cumulative fraction")
    axes[0,2].plot(x_array, min_y_array, c='black', linewidth=1, linestyle='--')
    axes[0,2].plot(x_array, max_y_array, c='black', linewidth=1, linestyle='--')
    axes[0,2].plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[0,2].plot(x_array, [0.33 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[0,2].plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[0,2].text(max(x_array)*numbers_text_loc_x, numbers_text_loc_y, r"$N=6-12$", ha='right', va='bottom', fontsize=15)
    axes[0,2].text(max(x_array)*abundance_text_loc_x, abundance_text_loc_y, "[Fe/H]", ha='right', va='bottom', fontsize=13)
    for i, c in enumerate(Clusters):
        FeH_nonan = FeH[c]
        FeH_nonan = FeH_nonan[~np.isnan(FeH_nonan)]
        if len(FeH_nonan)>=6 and len(FeH_nonan)<13:
            Dispersion_array = f["/dispersion"][len(FeH_nonan)-params.dispersion_cluster_size[0]]
            axes[0,2].scatter(biweight_scale(FeH_nonan), len(Dispersion_array[Dispersion_array<biweight_scale(FeH_nonan)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100*params.marker_scales[i], linewidth=params.marker_linewidths[i])

    filename = params.dispersion_distrib_file_mask[0] + 'cCFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = []
    for i in range (6, 13):
        Dispersion_array = Dispersion_array + list(f["/dispersion"][i-params.dispersion_cluster_size[0]])
    Dispersion_array = np.array(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = []
    for i in range (6, 13):
        Dispersion_array = f["/dispersion"][i-params.dispersion_cluster_size[0]]
        y_array.append(np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array]))
    y_array = np.transpose(y_array)
    min_y_array = np.array([min(y) for y in y_array])
    max_y_array = np.array([max(y) for y in y_array])
    axes[1,2].set_xlim(0, max(x_array))
    axes[1,2].set_ylim(0, 1)
    axes[1,2].set_xlabel(r"$\sigma$ ([C/Fe]c)")
    axes[1,2].set_ylabel("Cumulative fraction")
    axes[1,2].plot(x_array, min_y_array, c='black', linewidth=1, linestyle='--')
    axes[1,2].plot(x_array, max_y_array, c='black', linewidth=1, linestyle='--')
    axes[1,2].plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[1,2].plot(x_array, [0.33 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[1,2].plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[1,2].text(max(x_array)*numbers_text_loc_x, numbers_text_loc_y, r"$N=6-12$", ha='right', va='bottom', fontsize=15)
    axes[1,2].text(max(x_array)*abundance_text_loc_x, abundance_text_loc_y, r"[C/Fe]$_\mathrm{c}$", ha='right', va='bottom', fontsize=13)
    for i, c in enumerate(Clusters):
        final_CFec = []
        for cfec, C in zip(CFec[c], Comments[c]):
            if not np.isnan(cfec) and not 'C' in C:
                final_CFec.append(cfec) 
        if len(final_CFec)>=6 and len(final_CFec)<13:
            Dispersion_array = f["/dispersion"][len(final_CFec)-params.dispersion_cluster_size[0]]
            axes[1,2].scatter(biweight_scale(final_CFec), len(Dispersion_array[Dispersion_array<biweight_scale(final_CFec)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100*params.marker_scales[i], linewidth=params.marker_linewidths[i])

    filename = params.dispersion_distrib_file_mask[0] + 'SrFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = []
    for i in range (6, 13):
        Dispersion_array = Dispersion_array + list(f["/dispersion"][i-params.dispersion_cluster_size[0]])
    Dispersion_array = np.array(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = []
    for i in range (6, 13):
        Dispersion_array = f["/dispersion"][i-params.dispersion_cluster_size[0]]
        y_array.append(np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array]))
    y_array = np.transpose(y_array)
    min_y_array = np.array([min(y) for y in y_array])
    max_y_array = np.array([max(y) for y in y_array])
    axes[2,2].set_xlim(0, max(x_array))
    axes[2,2].set_ylim(0, 1)
    axes[2,2].set_xlabel(r"$\sigma$ ([Sr/Fe])")
    axes[2,2].set_ylabel("Cumulative fraction")
    axes[2,2].plot(x_array, min_y_array, c='black', linewidth=1, linestyle='--')
    axes[2,2].plot(x_array, max_y_array, c='black', linewidth=1, linestyle='--')
    axes[2,2].plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[2,2].plot(x_array, [0.33 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[2,2].plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[2,2].text(max(x_array)*numbers_text_loc_x, numbers_text_loc_y, r"$N=6-12$", ha='right', va='bottom', fontsize=15)
    axes[2,2].text(max(x_array)*abundance_text_loc_x, abundance_text_loc_y, r"[Sr/Fe]", ha='right', va='bottom', fontsize=13)
    for i, c in enumerate(Clusters):
        final_SrFe = []
        for srfe, C in zip(SrFe[c], Comments[c]):
            if not np.isnan(srfe) and not 'Sr' in C:
                final_SrFe.append(srfe)
        if len(final_SrFe)>=6 and len(final_SrFe)<13:
            Dispersion_array = f["/dispersion"][len(final_SrFe)-params.dispersion_cluster_size[0]]
            axes[2,2].scatter(biweight_scale(final_SrFe), len(Dispersion_array[Dispersion_array<biweight_scale(final_SrFe)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100*params.marker_scales[i], linewidth=params.marker_linewidths[i])

    filename = params.dispersion_distrib_file_mask[0] + 'BaFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = []
    for i in range (6, 13):
        Dispersion_array = Dispersion_array + list(f["/dispersion"][i-params.dispersion_cluster_size[0]])
    Dispersion_array = np.array(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = []
    for i in range (6, 13):
        Dispersion_array = f["/dispersion"][i-params.dispersion_cluster_size[0]]
        y_array.append(np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array]))
    y_array = np.transpose(y_array)
    min_y_array = np.array([min(y) for y in y_array])
    max_y_array = np.array([max(y) for y in y_array])
    axes[3,2].set_xlim(0, max(x_array))
    axes[3,2].set_ylim(0, 1)
    axes[3,2].set_xlabel(r"$\sigma$ ([Ba/Fe])")
    axes[3,2].set_ylabel("Cumulative fraction")
    axes[3,2].plot(x_array, min_y_array, c='black', linewidth=1, linestyle='--')
    axes[3,2].plot(x_array, max_y_array, c='black', linewidth=1, linestyle='--')
    axes[3,2].plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[3,2].plot(x_array, [0.33 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[3,2].plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[3,2].text(max(x_array)*numbers_text_loc_x, numbers_text_loc_y, r"$N=6-12$", ha='right', va='bottom', fontsize=15)
    axes[3,2].text(max(x_array)*abundance_text_loc_x, abundance_text_loc_y, "[Ba/Fe]", ha='right', va='bottom', fontsize=13)
    for i, c in enumerate(Clusters):
        final_BaFe = []
        for bafe, C in zip(BaFe[c], Comments[c]):
            if not np.isnan(bafe) and not 'Ba' in C:
                final_BaFe.append(bafe)
        if len(final_BaFe)>=6 and len(final_BaFe)<13:
            Dispersion_array = f["/dispersion"][len(final_BaFe)-params.dispersion_cluster_size[0]]
            axes[3,2].scatter(biweight_scale(final_BaFe), len(Dispersion_array[Dispersion_array<biweight_scale(final_BaFe)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100*params.marker_scales[i], linewidth=params.marker_linewidths[i])

    filename = params.dispersion_distrib_file_mask[0] + 'EuFe' + params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename, 'r')
    Dispersion_array = []
    for i in range (6, 13):
        Dispersion_array = Dispersion_array + list(f["/dispersion"][i-params.dispersion_cluster_size[0]])
    Dispersion_array = np.array(Dispersion_array)
    step = (max(Dispersion_array)-min(Dispersion_array))/10000
    x_array = np.arange(min(Dispersion_array),max(Dispersion_array+step),step)
    y_array = []
    for i in range (6, 13):
        Dispersion_array = f["/dispersion"][i-params.dispersion_cluster_size[0]]
        y_array.append(np.array([len(Dispersion_array[Dispersion_array<x])/len(Dispersion_array) for x in x_array]))
    y_array = np.transpose(y_array)
    min_y_array = np.array([min(y) for y in y_array])
    max_y_array = np.array([max(y) for y in y_array])
    axes[4,2].set_xlim(0, max(x_array))
    axes[4,2].set_ylim(0, 1)
    axes[4,2].set_xlabel(r"$\sigma$ ([Eu/Fe])")
    axes[4,2].set_ylabel("Cumulative fraction")
    axes[4,2].plot(x_array, min_y_array, c='black', linewidth=1, linestyle='--')
    axes[4,2].plot(x_array, max_y_array, c='black', linewidth=1, linestyle='--')
    axes[4,2].plot(x_array, [0.25 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[4,2].plot(x_array, [0.33 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[4,2].plot(x_array, [0.5 for x in x_array], c='black', linewidth=1, linestyle='--')
    axes[4,2].text(max(x_array)*numbers_text_loc_x, numbers_text_loc_y, r"$N=6-12$", ha='right', va='bottom', fontsize=15)
    axes[4,2].text(max(x_array)*abundance_text_loc_x, abundance_text_loc_y, "[Eu/Fe]", ha='right', va='bottom', fontsize=13)
    for i, c in enumerate(Clusters):
        EuFe_nonan = EuFe[c]
        EuFe_nonan = EuFe_nonan[~np.isnan(EuFe_nonan)]
        if len(EuFe_nonan)>=6 and len(EuFe_nonan)<13:
            Dispersion_array = f["/dispersion"][len(EuFe_nonan)-params.dispersion_cluster_size[0]]
            axes[4,2].scatter(biweight_scale(EuFe_nonan), len(Dispersion_array[Dispersion_array<biweight_scale(EuFe_nonan)])/len(Dispersion_array), c=params.colors[i], marker=params.markers[i], s=100*params.marker_scales[i], linewidth=params.marker_linewidths[i])


    plt.gcf().set_size_inches(12, 15)
    plt.tight_layout()
    plt.gcf().savefig("plots/dispersions_paper_cumuls_flipped.eps", bbox_inches='tight',pad_inches=0.01, dpi=100, format='eps')
    plt.gcf().savefig("plots/dispersions_paper_cumuls_flipped.png", bbox_inches='tight',pad_inches=0.01, dpi=100, format='png')
    plt.close()


def plot_dispersion_distrib(cluster_ID, cluster, Vars, var_names, var_labels):
    if len(Vars)>1:
        for i in range(len(Vars)):
            Vars[i] = (Vars[i] - np.mean(Vars[i])) / np.std(Vars[i])
    Vars = np.transpose(Vars)
    Vars = Vars[cluster]
    
    filename = params.dispersion_distrib_file_mask[0] + var_names[0]
    if len(var_names)>1:
        for v in var_names[1:]:
            filename += ('_' + v)
    filename += params.dispersion_distrib_file_mask[1]

    i = len(Vars)
    f = h5py.File(filename, 'r')
    a = f["/dispersion"][i-params.dispersion_cluster_size[0]]
    N_samples = len(a)
    
    x = np.linspace(min(a), max(a), N_samples)
    kde = KernelDensity(kernel='gaussian', bandwidth=(max(a)-min(a))/100/bandwidth_refinement).fit(a[:, None])
    a_kde = kde.score_samples(x[:, None])
    a_kde = np.exp(a_kde)

    plt.clf()
    plt.title("Dispersion distribution, " + str(i+params.dispersion_cluster_size[0]) + " stars", size=24)

    name = var_labels[0]
    if len(var_labels)>1:
        for v in var_labels[1:]:
            name += (', ' + v)
    name += ' standard deviation'
    plt.xlabel(name, size=24)

    plt.ylabel('Probability', size=24)
    plt.tick_params(labelsize=18)
    plt.fill_between(x, 0, a_kde, color='grey')

    sigma_line_height = N_samples/sum(a)
    a = sorted(a)
    plt.plot ([a[int((1-sigma_1_frac)*N_samples-1)], a[int((1-sigma_1_frac)*N_samples-1)]], [0, sigma_line_height], '--', color='black', linewidth=2)
    plt.plot ([a[int((1-sigma_2_frac)*N_samples-1)], a[int((1-sigma_2_frac)*N_samples-1)]], [0, sigma_line_height], '--', color='blue', linewidth=2)
    plt.plot ([a[int((1-sigma_3_frac)*N_samples-1)], a[int((1-sigma_3_frac)*N_samples-1)]], [0, sigma_line_height], '--', color='red', linewidth=2)

    plt.scatter (np.std(Vars), sigma_line_height/2, s=200, color=params.colors[cluster_ID], marker=params.markers[cluster_ID])

    plt.gcf().set_size_inches(25.6, 14.4)
        
    filename = params.cluster_dispersion_plot_file_mask[0] + var_names[0]
    if len(var_names)>1:
        for v in var_names[1:]:
            filename += ('_' + v)
    filename += ('_'+cluster_IDs[cluster_ID])
    filename += params.cluster_dispersion_plot_file_mask[1]
    plt.gcf().savefig(filename, dpi=100)
    plt.close()


def plot_dispersion_distrib_paper(cluster_ID, cluster, Vars, var_names, var_labels):
    if len(Vars)>1:
        for i in range(len(Vars)):
            Vars[i] = (Vars[i] - np.mean(Vars[i])) / np.std(Vars[i])
    Vars = np.transpose(Vars)
    Vars = Vars[cluster]
    
    filename = params.dispersion_distrib_file_mask[0] + var_names[0]
    if len(var_names)>1:
        for v in var_names[1:]:
            filename += ('_' + v)
    filename += params.dispersion_distrib_file_mask[1]

    i = len(Vars)
    f = h5py.File(filename, 'r')
    a = f["/dispersion"][i-params.dispersion_cluster_size[0]]
    N_samples = len(a)
    
    x = np.linspace(min(a), max(a), N_samples)
    kde = KernelDensity(kernel='gaussian', bandwidth=(max(a)-min(a))/100/bandwidth_refinement).fit(a[:, None])
    a_kde = kde.score_samples(x[:, None])
    a_kde = np.exp(a_kde)

    plt.clf()

    name = var_labels[0]
    if len(var_labels)>1:
        for v in var_labels[1:]:
            name += (', ' + v)
    name += ' standard deviation'
    plt.xlabel(name)

    plt.ylabel('Frequency')
    plt.fill_between(x, 0, a_kde, color='grey')

    sigma_line_height = N_samples/sum(a)
    a = sorted(a)
    plt.plot ([a[int((1-sigma_1_frac)*N_samples-1)], a[int((1-sigma_1_frac)*N_samples-1)]], [0, sigma_line_height], '--', color='black', linewidth=2)
    plt.plot ([a[int((1-sigma_2_frac)*N_samples-1)], a[int((1-sigma_2_frac)*N_samples-1)]], [0, sigma_line_height], '--', color='blue', linewidth=2)
    plt.plot ([a[int((1-sigma_3_frac)*N_samples-1)], a[int((1-sigma_3_frac)*N_samples-1)]], [0, sigma_line_height], '--', color='red', linewidth=2)

    plt.scatter (np.std(Vars), sigma_line_height/2, s=200, color=params.colors[cluster_ID], marker=params.markers[cluster_ID])

    plt.gcf().set_size_inches(5, 4)
        
    filename = params.cluster_dispersion_plot_file_mask[0] + var_names[0] + "_paper"
    if len(var_names)>1:
        for v in var_names[1:]:
            filename += ('_' + v)
    filename += ('_'+cluster_IDs[cluster_ID])
    filename += params.cluster_dispersion_plot_file_mask[1]
    plt.gcf().savefig(filename, bbox_inches='tight',pad_inches=0.01, dpi=100, format='eps')
    plt.close()


def plot_all_cluster_dispersions(Clusters, Vars, var_names, var_labels, markersize=200):
    plt.clf()
    plt.title("r-process clusters", size=24)
    plt.xlabel("Dispersion confidence levels for d"+var_labels[0], size=24)
    plt.ylabel("Clusters", size=24)
    plt.xlim(0, 1)
    plt.ylim(0, 1440) 
    plt.tick_params(labelsize=18)
    plt.gca().set_yticklabels([])

    N, Nv = len(Clusters), len(Vars)
    for i, v in enumerate(Vars):
        for j, c in enumerate(Clusters):
            # Calculate the position of the box on the plot.
            delta_x = 2540/Nv - 20
            delta_y = 1420/N - 20
            x1 = (i+1)*20 + i*delta_x
            x2 = (i+1)*20 + (i+1)*delta_x
            y1 = (j+1)*20 + j*delta_y
            y2 = (j+1)*20 + (j+1)*delta_y
            # Retrieve the dispersion array.
            filename = params.dispersion_distrib_file_mask[0] + var_names[i] + params.dispersion_distrib_file_mask[1]        
            f = h5py.File(filename, 'r')
            Dispersion_array = f["/dispersion"][len(c)-params.dispersion_cluster_size[0]]
            N_samples = len(Dispersion_array)
            # Calculate the variable dispersion in the cluster.
            var = np.transpose(v)
            var = var[c]
            var_std = np.std(var)
            # Find the variable location on the plot on the scale 0 to 1.
            point_x = x1 + (1/2 + (len(Dispersion_array[Dispersion_array<var_std])+1-len(Dispersion_array[Dispersion_array>var_std]))/2/N_samples)*delta_x
            # Plot everything.
            plt.plot([x1/2560,x2/2560], [y1,y1], color='black', linewidth=1)
            plt.plot([x1/2560,x2/2560], [y2,y2], color='black', linewidth=1)
            plt.plot([x1/2560,x1/2560], [y1,y2], color='black', linewidth=1) 
            plt.plot([x2/2560,x2/2560], [y1,y2], color='black', linewidth=1)
            plt.plot([(x2 - delta_x*sigma_1_frac)/2560, (x2 - delta_x*sigma_1_frac)/2560], [y1,y2], '--', color='black', linewidth=1)
            plt.plot([(x2 - delta_x*sigma_2_frac)/2560, (x2 - delta_x*sigma_2_frac)/2560], [y1,y2], '--', color='blue', linewidth=1)
            plt.plot([(x2 - delta_x*sigma_3_frac)/2560, (x2 - delta_x*sigma_3_frac)/2560], [y1,y2], '--', color='red', linewidth=1)
            plt.scatter(point_x/2560, (y1+y2)/2, color=params.colors[j], s=markersize, marker=params.markers[j])
    # Save the plot.
    filename = 'plots/all_cluster_dispersions_'+var_names[0]
    if len(var_names)>1:
        for v in var_names[1:]:
            filename += ('_' + v)
    filename += '.png'
    plt.gcf().set_size_inches(25.6, 14.4)
    plt.gcf().savefig(filename, dpi=100)
    plt.close()


def plot_all_cluster_dispersions_paper(Clusters, FeH, EuFe, CFe, r_apo, BaEu, SrBa):
    plt.clf()
    fig, axes = plt.subplots(nrows=2, ncols=3)

    v, w = FeH, EuFe
    for j, c in enumerate(Clusters):
        # Retrieve the dispersion array.
        filename = params.dispersion_distrib_file_mask[0] + 'FeH' + params.dispersion_distrib_file_mask[1]        
        f = h5py.File(filename, 'r')
        Dispersion_array = f["/dispersion"][len(c)-params.dispersion_cluster_size[0]]
        N_samples = len(Dispersion_array)
        # Calculate the variable dispersion in the cluster.
        var_std = np.std(v[c])
        var_mean = np.mean(w[c])
        x = var_mean
        # Find the variable location on the plot on the scale 0 to 1.
        y = 1/2 + (len(Dispersion_array[Dispersion_array<var_std])+1-len(Dispersion_array[Dispersion_array>var_std]))/2/N_samples
        # Plot everything.
        axes[0,0].set_ylabel("d[Fe/H] c.l.")
        axes[0,0].set_xticklabels([])
        axes[0,0].set_xlim(0.5,0.75)
        axes[0,0].set_ylim(0,1)
        axes[0,0].plot([-10, 10], [sigma_1_frac,sigma_1_frac], '--', color='black', linewidth=0.5)
        axes[0,0].plot([-10, 10], [sigma_2_frac,sigma_2_frac], '--', color='blue', linewidth=0.5)
        axes[0,0].plot([-10, 10], [sigma_3_frac,sigma_3_frac], '--', color='red', linewidth=0.5)
        axes[0,0].scatter(x, 1-y, color=params.colors[j], s=100, marker=params.markers[j])

    v, w = CFe, EuFe
    for j, c in enumerate(Clusters):
        # Retrieve the dispersion array.
        filename = params.dispersion_distrib_file_mask[0] + 'CFe' + params.dispersion_distrib_file_mask[1]        
        f = h5py.File(filename, 'r')
        Dispersion_array = f["/dispersion"][len(c)-params.dispersion_cluster_size[0]]
        N_samples = len(Dispersion_array)
        # Calculate the variable dispersion in the cluster.
        var_std = np.std(v[c])
        var_mean = np.mean(w[c])
        x = var_mean
        # Find the variable location on the plot on the scale 0 to 1.
        y = 1/2 + (len(Dispersion_array[Dispersion_array<var_std])+1-len(Dispersion_array[Dispersion_array>var_std]))/2/N_samples
        # Plot everything.
        axes[1,0].set_xlabel("<[Eu/Fe]>")
        axes[1,0].set_ylabel("d[C/Fe] c.l.")
        axes[1,0].set_xlim(0.5,0.75)
        axes[1,0].set_ylim(0,1)
        axes[1,0].plot([-10, 10], [sigma_1_frac,sigma_1_frac], '--', color='black', linewidth=0.5)
        axes[1,0].plot([-10, 10], [sigma_2_frac,sigma_2_frac], '--', color='blue', linewidth=0.5)
        axes[1,0].plot([-10, 10], [sigma_3_frac,sigma_3_frac], '--', color='red', linewidth=0.5)
        axes[1,0].scatter(x, 1-y, color=params.colors[j], s=100, marker=params.markers[j])

    v, w = EuFe, FeH
    for j, c in enumerate(Clusters):
        # Retrieve the dispersion array.
        filename = params.dispersion_distrib_file_mask[0] + 'EuFe' + params.dispersion_distrib_file_mask[1]        
        f = h5py.File(filename, 'r')
        Dispersion_array = f["/dispersion"][len(c)-params.dispersion_cluster_size[0]]
        N_samples = len(Dispersion_array)
        # Calculate the variable dispersion in the cluster.
        var_std = np.std(v[c])
        var_mean = np.mean(w[c])
        x = var_mean
        # Find the variable location on the plot on the scale 0 to 1.
        y = 1/2 + (len(Dispersion_array[Dispersion_array<var_std])+1-len(Dispersion_array[Dispersion_array>var_std]))/2/N_samples
        # Plot everything.
        axes[0,1].set_ylabel("d[Eu/Fe] c.l.")
        axes[0,1].set_xticklabels([])
        axes[0,1].set_xlim(-3,-2)
        axes[0,1].set_ylim(0,1)
        axes[0,1].plot([-10, 10], [sigma_1_frac,sigma_1_frac], '--', color='black', linewidth=0.5)
        axes[0,1].plot([-10, 10], [sigma_2_frac,sigma_2_frac], '--', color='blue', linewidth=0.5)
        axes[0,1].plot([-10, 10], [sigma_3_frac,sigma_3_frac], '--', color='red', linewidth=0.5)
        axes[0,1].scatter(x, 1-y, color=params.colors[j], s=100, marker=params.markers[j])

    v, w = r_apo, FeH
    for j, c in enumerate(Clusters):
        # Retrieve the dispersion array.
        filename = params.dispersion_distrib_file_mask[0] + 'r_apo' + params.dispersion_distrib_file_mask[1]        
        f = h5py.File(filename, 'r')
        Dispersion_array = f["/dispersion"][len(c)-params.dispersion_cluster_size[0]]
        N_samples = len(Dispersion_array)
        # Calculate the variable dispersion in the cluster.
        var_std = np.std(v[c])
        var_mean = np.mean(w[c])
        x = var_mean
        # Find the variable location on the plot on the scale 0 to 1.
        y = 1/2 + (len(Dispersion_array[Dispersion_array<var_std])+1-len(Dispersion_array[Dispersion_array>var_std]))/2/N_samples
        # Plot everything.
        axes[1,1].set_xlabel("<[Fe/H]>")
        axes[1,1].set_ylabel("d r_apo c.l.")
        axes[1,1].set_xlim(-3,-2)
        axes[1,1].set_ylim(0,1)
        axes[1,1].plot([-10, 10], [sigma_1_frac,sigma_1_frac], '--', color='black', linewidth=0.5)
        axes[1,1].plot([-10, 10], [sigma_2_frac,sigma_2_frac], '--', color='blue', linewidth=0.5)
        axes[1,1].plot([-10, 10], [sigma_3_frac,sigma_3_frac], '--', color='red', linewidth=0.5)
        axes[1,1].scatter(x, 1-y, color=params.colors[j], s=100, marker=params.markers[j])

    v, w = BaEu, FeH
    for j, c in enumerate(Clusters):
        # Retrieve the dispersion array.
        filename = params.dispersion_distrib_file_mask[0] + 'BaEu' + params.dispersion_distrib_file_mask[1]        
        f = h5py.File(filename, 'r')
        Dispersion_array = f["/dispersion"][len(c)-params.dispersion_cluster_size[0]]
        N_samples = len(Dispersion_array)
        # Calculate the variable dispersion in the cluster.
        var_std = np.std(v[c])
        var_mean = np.mean(w[c])
        x = var_mean
        # Find the variable location on the plot on the scale 0 to 1.
        y = 1/2 + (len(Dispersion_array[Dispersion_array<var_std])+1-len(Dispersion_array[Dispersion_array>var_std]))/2/N_samples
        # Plot everything.
        axes[0,2].set_ylabel("d[Ba/Eu] c.l.")
        axes[0,2].set_xticklabels([])
        axes[0,2].set_xlim(-3,-2)
        axes[0,2].set_ylim(0,1)
        axes[0,2].plot([-10, 10], [sigma_1_frac,sigma_1_frac], '--', color='black', linewidth=0.5)
        axes[0,2].plot([-10, 10], [sigma_2_frac,sigma_2_frac], '--', color='blue', linewidth=0.5)
        axes[0,2].plot([-10, 10], [sigma_3_frac,sigma_3_frac], '--', color='red', linewidth=0.5)
        axes[0,2].scatter(x, 1-y, color=params.colors[j], s=100, marker=params.markers[j])

    v, w = SrBa, FeH
    for j, c in enumerate(Clusters):
        # Retrieve the dispersion array.
        filename = params.dispersion_distrib_file_mask[0] + 'SrBa' + params.dispersion_distrib_file_mask[1]        
        f = h5py.File(filename, 'r')
        Dispersion_array = f["/dispersion"][len(c)-params.dispersion_cluster_size[0]]
        N_samples = len(Dispersion_array)
        # Calculate the variable dispersion in the cluster.
        var_std = np.std(v[c])
        var_mean = np.mean(w[c])
        x = var_mean
        # Find the variable location on the plot on the scale 0 to 1.
        y = 1/2 + (len(Dispersion_array[Dispersion_array<var_std])+1-len(Dispersion_array[Dispersion_array>var_std]))/2/N_samples
        # Plot everything.
        axes[1,2].set_xlabel("<[Fe/H]>")
        axes[1,2].set_ylabel("d[Sr/Ba] c.l.")
        axes[1,2].set_xlim(-3,-2)
        axes[1,2].set_ylim(0,1)
        axes[1,2].plot([-10, 10], [sigma_1_frac,sigma_1_frac], '--', color='black', linewidth=0.5)
        axes[1,2].plot([-10, 10], [sigma_2_frac,sigma_2_frac], '--', color='blue', linewidth=0.5)
        axes[1,2].plot([-10, 10], [sigma_3_frac,sigma_3_frac], '--', color='red', linewidth=0.5)
        axes[1,2].scatter(x, 1-y, color=params.colors[j], s=100, marker=params.markers[j])

    # Save the plot.
    filename_eps = 'plots/all_cluster_dispersions_paper.eps'
    filename_png = 'plots/all_cluster_dispersions_paper.png'
    plt.gcf().set_size_inches(10,6)
    plt.subplots_adjust(wspace=0.3)
    plt.gcf().savefig(filename_eps, bbox_inches='tight',pad_inches=0.01, dpi=100, format='eps')
    plt.gcf().savefig(filename_png, bbox_inches='tight',pad_inches=0.01, dpi=100, format='png')
    plt.close()



def plot_all_cluster_dispersions_2D(Clusters, Vars, var_names, var_labels, markersize=200, ylimits = [1,0]):
    plt.clf()
    plt.title("r-process clusters", size=24)
    plt.xlabel("<"+var_labels[0]+">", size=24)
    plt.ylabel("Dispersion confidence levels for d"+var_labels[1], size=24)
    C = []
    for c in Clusters:
        C.append(np.mean(list(Vars[0][c].flatten())))
    plt.xlim(min(C)-abs(max(C)-min(C))/20, max(C)+abs(max(C)-min(C))/20)
    plt.ylim(ylimits[0],ylimits[1])
    plt.tick_params(labelsize=18)
    plt.plot([min(Vars[0].flatten())-abs(max(Vars[0].flatten())-min(Vars[0].flatten()))/2, max(Vars[0].flatten())+abs(max(Vars[0].flatten())-min(Vars[0].flatten()))/2], [1-sigma_1_frac, 1-sigma_1_frac], '--', color='black', linewidth=1)
    plt.plot([min(Vars[0].flatten())-abs(max(Vars[0].flatten())-min(Vars[0].flatten()))/2, max(Vars[0].flatten())+abs(max(Vars[0].flatten())-min(Vars[0].flatten()))/2], [1-sigma_2_frac, 1-sigma_2_frac], '--', color='blue', linewidth=1)
    plt.plot([min(Vars[0].flatten())-abs(max(Vars[0].flatten())-min(Vars[0].flatten()))/2, max(Vars[0].flatten())+abs(max(Vars[0].flatten())-min(Vars[0].flatten()))/2], [1-sigma_3_frac, 1-sigma_3_frac], '--', color='red', linewidth=1)

    for j, c in enumerate(Clusters):
            # Retrieve the dispersion array.
            filename = params.dispersion_distrib_file_mask[0] + var_names[1] + params.dispersion_distrib_file_mask[1]        
            f = h5py.File(filename, 'r')
            Dispersion_array = f["/dispersion"][len(c)-params.dispersion_cluster_size[0]]
            N_samples = len(Dispersion_array)
            # Calculate the variable dispersion in the cluster.
            var = np.transpose(Vars[1])
            var = var[c]
            var_std = np.std(var)
            # Find the variable location on the plot on the scale 0 to 1.
            point_x = np.mean(Vars[0][c])
            point_y = 1/2+(len(Dispersion_array[Dispersion_array<var_std])+1-len(Dispersion_array[Dispersion_array>var_std]))/2/N_samples 
            plt.scatter(point_x, point_y, color=params.colors[j], s=markersize, marker=params.markers[j])

    # Save the plot
    filename = 'plots/all_cluster_dispersions_2D_'+var_names[0]
    if len(var_names)>1:
        for v in var_names[1:]:
            filename += ('_' + v)
    filename += '.png'
    plt.gcf().set_size_inches(25.6, 14.4)
    plt.gcf().savefig(filename, dpi=100)
    plt.close()

if __name__ == '__main__':
    init_file = params.init_file
    kin_file = params.kin_file
    FeH = get_column(8, float, init_file)[:]
    cCFe = get_column(10, float, init_file)[:]
    SrFe = get_column(11, float, init_file)[:]
    BaFe = get_column(12, float, init_file)[:]
    EuFe = get_column(13, float, init_file)[:]
    Comments = get_column(14, str, init_file)[:]
    SrBa = SrFe[:]-BaFe[:]
    BaEu = BaFe[:]-EuFe[:]
    A_Eu = (0.52 + FeH + EuFe)[:]
    A_C = (cCFe + FeH + 8.43)[:]
    Clusters = clusters.final_clusters
    r_apo = get_column(25, float, kin_file)[:] 

    #for i in range(0, len(Clusters)):
    #    plot_dispersion_distrib(i, Clusters[i], [FeH], ['FeH'], ['[Fe/H]'])
    #    plot_dispersion_distrib(i, Clusters[i], [A_C], ['AC'], ['A(C)'])
    #    plot_dispersion_distrib(i, Clusters[i], [FeH, A_C], ['FeH', 'AC'], ['[Fe/H]', 'A(C)'])
    #    plot_dispersion_distrib(i, Clusters[i], [CFe], ['CFe'], ['[C/Fe]'])
    #    plot_dispersion_distrib(i, Clusters[i], [vRg], ['vRg'], ['vRg'])
    #    plot_dispersion_distrib(i, Clusters[i], [vTg], ['vTg'], ['vTg'])
    #    plot_dispersion_distrib(i, Clusters[i], [r_apo], ['r_apo'], ['r_apo'])
    #    plot_dispersion_distrib_paper(i, Clusters[i], [FeH], ['FeH'], ['[Fe/H]'])
    #    plot_dispersion_distrib_paper(i, Clusters[i], [EuFe], ['EuFe'], ['[Eu/Fe]'])
    #plot_all_cluster_dispersions(Clusters, [FeH], ['FeH'], ['[Fe/H]'], markersize=600)
    #plot_all_cluster_dispersions(Clusters, [A_C], ['AC'], ['A(C)'], markersize=600)
    #plot_all_cluster_dispersions(Clusters, [r_apo], ['r_apo'], ['r_apo'], markersize=600)
    #plot_all_cluster_dispersions(Clusters, [CFe], ['CFe'], ['[C/Fe]'], markersize=600)
    #plot_all_cluster_dispersions(Clusters, [SrBa], ['SrBa'], ['[Sr/Ba]'], markersize=600)
    #plot_all_cluster_dispersions(Clusters, [BaEu], ['BaEu'], ['[Ba/Eu]'], markersize=600)
    #plot_all_cluster_dispersions(Clusters, [EuFe], ['EuFe'], ['[Eu/Fe]'], markersize=600)
    #plot_all_cluster_dispersions_2D(Clusters, [EuFe,FeH], ['EuFe','FeH'], ['[Eu/Fe]','[Fe/H]'], markersize=600, ylimits=[1,0])
    #plot_all_cluster_dispersions_2D(Clusters, [FeH,EuFe], ['FeH','EuFe'], ['[Fe/H]','[Eu/Fe]'], markersize=600, ylimits=[0.06,0])
    #plot_all_cluster_dispersions_2D(Clusters, [EuFe,CFe], ['EuFe','CFe'], ['[Eu/Fe]','[C/Fe]'], markersize=600, ylimits=[0.10,0])
    #plot_all_cluster_dispersions_paper(Clusters, FeH, EuFe, cCFe, r_apo, BaEu, SrBa)
    #plot_dispersions_paper(Clusters, FeH, cCFe, EuFe, SrFe, BaFe)
    #plot_dispersions_paper_v2(Clusters, FeH, cCFe, EuFe, SrFe, BaFe, Comments)
    #plot_dispersions_paper_abundance(Clusters, FeH, cCFe, EuFe, SrFe, BaFe, Comments)
    #plot_dispersions_paper_cumuls(Clusters, FeH, cCFe, EuFe, SrFe, BaFe, Comments)
    plot_dispersions_paper_cumuls_flipped(Clusters, FeH, cCFe, EuFe, SrFe, BaFe, Comments)
    #plot_appendix_cumul_paper(Clusters, FeH, Comments)
