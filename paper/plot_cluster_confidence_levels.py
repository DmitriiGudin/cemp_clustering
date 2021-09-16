from __future__ import division
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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


def plot_cluster_confidence_levels(dispersions, confidences_list):
    plt.clf()
    fig, axes = plt.subplots(nrows=5, ncols=1)
    xlabels = [r"CDF value of $\sigma$([Fe/H])",r"CDF value of $\sigma$([C/Fe]$_\mathrm{c}$)",r"CDF value of $\sigma$([Sr/Fe])",r"CDF value of $\sigma$([Ba/Fe])",r"CDF value of $\sigma$([Eu/Fe])"]
   
    for i, xl in enumerate(xlabels):
        axes[i].set_xlabel(xl)
        axes[i].set_ylabel(r"Confidence level, $\%$")
        for j in range(len(confidences_list)):
            if not (np.isnan(dispersions[i][j])):
                axes[i].scatter(dispersions[i][j], confidences_list[j], c=params.colors[j], marker=params.markers[j], s=30)
        indeces_1 = np.where(~np.isnan(dispersions[i]))
        indeces_2 = np.where(confidences_list>= 0.5)
        indeces = np.intersect1d(indeces_1, indeces_2)
        reg = LinearRegression().fit(dispersions[i][indeces].reshape((-1, 1)), confidences_list[indeces])
        x_array = np.arange(0,1+0.001,0.001)
        y_array = np.array([reg.intercept_ + reg.coef_*x for x in x_array])
	mod = sm.OLS(confidences_list[indeces],dispersions[i][indeces].reshape((-1, 1)))
	fii = mod.fit()
        r2 = fii.rsquared
	adjusted_r2 = 1-(1-r2)*(len(confidences_list)-1)/(len(confidences_list)-2)
	p_value = fii.pvalues[0]
        axes[i].plot(x_array,y_array,c='black',linewidth=1,linestyle='--')
	axes[i].text(0.2,17.5, r'Adjusted $R^2$: '+str(round(adjusted_r2,2)), ha='left', va='bottom')
	axes[i].text(0.2,5, r'$p$-value: '+str(int(round(p_value*10000000,0)))+r'$\times 10^{-7}$', ha='left', va='bottom')
        
    plt.gcf().set_size_inches(3.5, 12)
    plt.tight_layout()
    plt.gcf().savefig("plots/cluster_confidence_levels.eps", bbox_inches='tight',pad_inches=0.01, dpi=100, format='eps')
    plt.gcf().savefig("plots/cluster_confidence_levels.png", bbox_inches='tight',pad_inches=0.01, dpi=100, format='png')
    plt.close()


if __name__ == '__main__':
    conf_names = get_column(0, str, confidences_filename)[:]
    conf_probs = get_column(21, float, confidences_filename)[:]
    kin_names = get_column(0, str, filename_1)[:]

    FeH = get_column(8, float, filename_1)[:]
    cCFe = get_column(10, float, filename_1)[:]
    SrFe = get_column(11, float, filename_1)[:]
    BaFe = get_column(12, float, filename_1)[:]
    EuFe = get_column(13, float, filename_1)[:]
    Comments = get_column(14, str, filename_1)[:]

    confidences_list = []
    for cluster in final_clusters:
        confidences = []
        for c in cluster:
            name = kin_names[c]
            i = np.where(conf_names=='"'+name+'"')[0][0]
            confidences.append(conf_probs[i])
        confidences_list.append(np.mean(confidences))
    confidences_list = np.array(confidences_list)

    dispersions = [[] for i in range(5)]
    
    for i, (abundance, abundance_label, limit_label) in enumerate(zip([FeH, cCFe, SrFe, BaFe, EuFe], ['FeH', 'cCFe', 'SrFe', 'BaFe', 'EuFe'], ['Fe', 'C', 'Sr', 'Ba', 'Eu'])):
        filename = params.dispersion_distrib_file_mask[0] + abundance_label + params.dispersion_distrib_file_mask[1]
        f = h5py.File(filename, 'r')
        for c in final_clusters:
            abundance_list = []
            for abd, C in zip(abundance[c], Comments[c]):
                if not np.isnan(abd) and not limit_label in C:
                    abundance_list.append(abd)
            abundance_list = np.array(abundance_list)
            Dispersion_array = f["/dispersion"][len(abundance_list)-params.dispersion_cluster_size[0]]
            if len(abundance_list)<params.biweight_estimator_min_cluster_size and len(abundance_list)>=params.dispersion_cluster_size[0]:
                dispersions[i].append(len(Dispersion_array[Dispersion_array<np.std(abundance_list)])/len(Dispersion_array))
            elif len(abundance_list)>=params.biweight_estimator_min_cluster_size:
                dispersions[i].append(len(Dispersion_array[Dispersion_array<biweight_scale(abundance_list)])/len(Dispersion_array))
            else:
                dispersions[i].append(np.nan)
    dispersions = np.array(dispersions)
     
    plot_cluster_confidence_levels(dispersions, confidences_list)
        
