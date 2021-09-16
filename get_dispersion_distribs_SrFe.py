from __future__ import division
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import h5py
from sklearn.neighbors import KernelDensity
from astropy.stats import biweight_scale
import params
import clusters


sigma_1_frac = 0.68
sigma_2_frac = 0.95
sigma_3_frac = 0.997


min_cluster_size = params.dispersion_cluster_size[0]
max_cluster_size = params.dispersion_cluster_size[1]
N_samples = 1000000 # A multiple of 1000, preferably >=10000. Otherwise, the sigmas may not be calculated properly.
bandwidth_refinement = 5


def get_column(N, Type, filename):
    return np.genfromtxt(filename, delimiter=',', dtype=Type, skip_header=1, usecols=N, comments=None)


def plot_stuff(X, Y, Y_kde, rv):
    plt.clf()
    plt.title("Gaussian distribution - KDE fit", size=24)
    plt.xlabel('X', size=24)
    plt.ylabel('G(X)', size=24)
    plt.tick_params(labelsize=18)
    plt.hist(Y, color='black', linewidth=2, density=True, histtype='step', bins=40)
    plt.plot(X, Y_kde, color='red', linewidth=2)
    plt.plot(X, rv.pdf(X), color='blue', linewidth=2)
    plt.gcf().set_size_inches(25.6, 14.4)
    plt.gcf().savefig('test.png', dpi=100)
    plt.close()


def plot_dispersion_distrib(dispersion_arrays, var_names, var_labels):
    for i, a in enumerate(dispersion_arrays):
        x = np.linspace(min(a), max(a), N_samples)
        kde = KernelDensity(kernel='gaussian', bandwidth=(max(a)-min(a))/100/bandwidth_refinement).fit(a[:, None])
        a_kde = kde.score_samples(x[:, None])
        a_kde = np.exp(a_kde)

        plt.clf()
        plt.title("Dispersion distribution, " + str(i+min_cluster_size) + " stars", size=24)

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

        plt.gcf().set_size_inches(25.6, 14.4)
        
        filename = params.dispersion_distrib_plot_file_mask[0] + var_names[0]
        if len(var_names)>1:
            for v in var_names[1:]:
                filename += ('_' + v)
        filename += ('_'+str(i+min_cluster_size))
        filename += params.dispersion_distrib_plot_file_mask[1]
        plt.gcf().savefig(filename, dpi=100)
        plt.close()


def get_dispersion_distrib(var, var_name, var_label):
    f = h5py.File(params.dispersion_distrib_file_mask[0]+var_name+params.dispersion_distrib_file_mask[1],'w')
    f.create_dataset("/dispersion", (max_cluster_size-min_cluster_size+1, N_samples), dtype='f')
    for i in range(min_cluster_size, max_cluster_size+1):
        dispersion_array = np.zeros((N_samples,))
        for j in range(N_samples):
            random.shuffle(var)
            v = var[0:i]
            if len(v) < params.biweight_estimator_min_cluster_size:
                dispersion_array[j] = np.std(v)
            else:
                dispersion_array[j] = biweight_scale(v)
            print var_name+": "+str(j+1)+" / "+str(N_samples)+" ("+str(i)+" clusters)"
        f["/dispersion"][i-min_cluster_size] = dispersion_array[:]

    #plot_dispersion_distrib (f["/dispersion"][:], [var_name], [var_label])
    f.close()


if __name__ == '__main__':
    init_file = params.init_file
    kin_file = params.kin_file
    FeH = get_column(8, float, init_file)[:]
    CFe = get_column(9, float, init_file)[:]
    cCFe = get_column(10, float, init_file)[:]  
    SrFe = get_column(11, float, init_file)[:]
    BaFe = get_column(12, float, init_file)[:]
    EuFe = get_column(13, float, init_file)[:]
    Comments = get_column(14, str, init_file)[:]

    final_FeH = FeH[:]
    final_FeH = final_FeH[~np.isnan(final_FeH)]

    final_CFe = []
    for c, C in zip(CFe, Comments):
        if not 'C' in C:
            final_CFe.append(c)
    final_CFe = np.array(final_CFe)
    final_CFe = final_CFe[~np.isnan(final_CFe)]

    final_cCFe = []
    for c, C in zip(cCFe, Comments):
        if not 'C' in C:
            final_cCFe.append(c)
    final_cCFe = np.array(final_cCFe)
    final_cCFe = final_cCFe[~np.isnan(final_cCFe)]

    final_SrFe = []
    for s, C in zip(SrFe, Comments):
        if not 'Sr' in C:
            final_SrFe.append(s)
    final_SrFe = np.array(final_SrFe)
    final_SrFe = final_SrFe[~np.isnan(final_SrFe)]

    final_BaFe = []
    for b, C in zip(BaFe, Comments):
        if not 'Ba' in C:
            final_BaFe.append(b)
    final_BaFe = np.array(final_BaFe)
    final_BaFe = final_BaFe[~np.isnan(final_BaFe)]
    
    final_EuFe = EuFe[:]
    final_EuFe = final_EuFe[~np.isnan(final_EuFe)]
 
    #get_dispersion_distrib(final_FeH, 'FeH', '[Fe/H]')
    #get_dispersion_distrib(final_CFe, 'CFe', '[C/Fe]')
    #get_dispersion_distrib(final_cCFe, 'cCFe', 'c[C/Fe]')
    get_dispersion_distrib(final_SrFe, 'SrFe', '[Sr/Fe]')
    #get_dispersion_distrib(final_BaFe, 'BaFe', '[Ba/Fe]')
    #get_dispersion_distrib(final_EuFe, 'EuFe', '[Eu/Fe]')

    
