from __future__ import division
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
from matplotlib import ticker
from matplotlib import patches
from matplotlib.colors import LogNorm
import itertools
import os
import astropy
import astropy.units as u
import astropy.coordinates as coord
from sklearn.cluster import KMeans, MeanShift, AffinityPropagation, AgglomerativeClustering
from sklearn import datasets
from sklearn.manifold import TSNE
import params
import clusters


def get_column(N, Type, filename):
    return np.genfromtxt(filename, delimiter=',', dtype=Type, skip_header=1, usecols=N, comments=None)
  

def plot_data(name, var_name_1, var_name_2, var1, var2):
    plt.clf()
    plt.title("r-process sample", size=24)
    plt.xlabel(var_name_1, size=24)
    plt.ylabel(var_name_2, size=24)
    plt.tick_params(labelsize=18)
    plt.scatter(var1, var2, c='black', s=50)
    plt.gcf().set_size_inches(25.6, 14.4)
    plt.gcf().savefig("plots/"+name, dpi=100)
    plt.close()


def plot_concatenated_clusters (name, Clusters, colors, markers, var_name_1, var_name_2, var1, var2, show_all, Xlim=[0,0], Ylim=[0,0], markersize=(10,200)):
    indeces = [c[i] for c in Clusters for i in range(len(c))]
    non_indeces = [i for i in range(len(var1)) if not (i in indeces)]

    plt.clf()
    plt.title("r-process clusters", size=24)
    plt.xlabel(var_name_1, size=24)
    plt.ylabel(var_name_2, size=24)
    if Xlim[0] < Xlim[1]:
        plt.xlim(Xlim[0], Xlim[1])
    if Ylim[0] < Ylim[1]:
        plt.ylim(Ylim[0], Ylim[1])
    plt.tick_params(labelsize=18)
    for c, color, marker in zip(Clusters, colors, markers):
        plt.scatter(var1[c], var2[c], c=color, marker=marker, s=markersize[1])
    if show_all==True:
        plt.scatter(var1[non_indeces], var2[non_indeces], c='grey', marker='o', s=markersize[0])
    plt.gcf().set_size_inches(25.6, 14.4)
    plt.gcf().savefig("plots/"+name, dpi=100)
    plt.close()


def plot_concatenated_clusters_paper (kmeans_clusters, meanshift_clusters, afftypropagation_clusters, aggloclustering_euclidian_clusters, colors, markers, Energy, J_phi, varname, xlim):
    
    plt.clf()
    fig, axes = plt.subplots(nrows=4, ncols=1)

    indeces = [c[i] for c in kmeans_clusters for i in range(len(c))]
    non_indeces = [i for i in range(len(Energy)) if not (i in indeces)]
    axes[0].set_xlim(xlim[0],xlim[1])
    axes[0].set_ylim(-2.1,-1.2)
    axes[0].set_ylabel(r"Energy   (km$^2$ s$^{-2}\times 10^5$)")
    axes[0].set_xticklabels([])
    for c, color, marker in zip(kmeans_clusters, colors, markers):
        axes[0].scatter(J_phi[c]/1e3, Energy[c]/1e5, c=color, marker=marker, s=30)
    axes[0].scatter(J_phi[non_indeces]/1e3, Energy[non_indeces]/1e5, c='grey', marker='o', s=7)

    indeces = [c[i] for c in meanshift_clusters for i in range(len(c))]
    non_indeces = [i for i in range(len(Energy)) if not (i in indeces)]
    axes[1].set_xlim(xlim[0],xlim[1])
    axes[1].set_ylim(-2.1,-1.2)
    axes[1].set_ylabel(r"Energy   (km$^2$ s$^{-2}\times 10^5$)")
    axes[1].set_xticklabels([])
    for c, color, marker in zip(meanshift_clusters, colors, markers):
        axes[1].scatter(J_phi[c]/1e3, Energy[c]/1e5, c=color, marker=marker, s=30)
    axes[1].scatter(J_phi[non_indeces]/1e3, Energy[non_indeces]/1e5, c='grey', marker='o', s=7)

    indeces = [c[i] for c in afftypropagation_clusters for i in range(len(c))]
    non_indeces = [i for i in range(len(Energy)) if not (i in indeces)]
    axes[2].set_xlim(xlim[0],xlim[1])
    axes[2].set_ylim(-2.1,-1.2)
    axes[2].set_ylabel(r"Energy   (km$^2$ s$^{-2}\times 10^5$)")
    axes[2].set_xticklabels([])
    for c, color, marker in zip(afftypropagation_clusters, colors, markers):
        axes[2].scatter(J_phi[c]/1e3, Energy[c]/1e5, c=color, marker=marker, s=30)
    axes[2].scatter(J_phi[non_indeces]/1e3, Energy[non_indeces]/1e5, c='grey', marker='o', s=7)

    indeces = [c[i] for c in aggloclustering_euclidian_clusters for i in range(len(c))]
    non_indeces = [i for i in range(len(Energy)) if not (i in indeces)]
    axes[3].set_xlim(xlim[0],xlim[1])
    axes[3].set_ylim(-2.1,-1.2)
    axes[3].set_xlabel(r"J$_\phi$   (kpc km s$^{-1}\times 10^3$)")
    axes[3].set_ylabel(r"Energy   (km$^2$ s$^{-2}\times 10^5$)")
    for c, color, marker in zip(aggloclustering_euclidian_clusters, colors, markers):
        axes[3].scatter(J_phi[c]/1e3, Energy[c]/1e5, c=color, marker=marker, s=30)
    axes[3].scatter(J_phi[non_indeces]/1e3, Energy[non_indeces]/1e5, c='grey', marker='o', s=7)

    plt.gcf().set_size_inches(5, 12)
    fig.tight_layout()
    plt.gcf().savefig("plots/concatenated_clusters_paper_"+varname+".eps", bbox_inches='tight',pad_inches=0.01, dpi=100, format='eps')
    plt.gcf().savefig("plots/concatenated_clusters_paper_"+varname+".png", bbox_inches='tight',pad_inches=0.01, dpi=100, format='png')
    plt.close()


def plot_toomre_paper (v_phi, v_z, v_r):
    plt.clf()

    plt.xlabel(r"$V_\phi$ (km/s)")
    plt.ylabel(r"$V_\perp$ (km/s)")
    plt.ylim(0,500)
    plt.xlim(-600,600)
    s = plt.scatter(v_phi, np.abs(v_z), c=np.abs(v_r), s=5, cmap='gist_rainbow')
    
    MW_phi = np.arange (0, np.pi+0.0001, 0.0001)
    MW_x_arr = 100*np.cos(MW_phi) + 220 
    MW_y_arr = np.abs(100*np.sin(MW_phi))
    plt.plot(MW_x_arr, MW_y_arr, c='black', linewidth=1, linestyle='--')

    #cbar_ax = fig.add_axes([0.85,0.15,0.05,0.7])
    cbar=plt.colorbar()
    cbar.set_label(r"$|V_r|$ (km/s)")
    #plt.colorbar()

    plt.gcf().set_size_inches(10, 4)
    plt.gcf().savefig("plots/toomre_paper.eps", bbox_inches='tight',pad_inches=0.01, dpi=100, format='eps')
    plt.gcf().savefig("plots/toomre_paper.png", bbox_inches='tight',pad_inches=0.01, dpi=100, format='png')

def plot_toomre_paper_clusters (Clusters, v_phi, v_z):
    plt.clf()

    plt.xlabel(r"$V_\phi$ (km/s)")
    plt.ylabel(r"$V_\perp$ (km/s)")
    plt.ylim(0,200)
    plt.xlim(-300,300)
    
    MW_phi = np.arange (0, np.pi+0.0001, 0.0001)
    MW_x_arr = 100*np.cos(MW_phi) + 220 
    MW_y_arr = np.abs(100*np.sin(MW_phi))
    plt.plot(MW_x_arr, MW_y_arr, c='black', linewidth=1, linestyle='--')

    indeces = [c[i] for c in Clusters for i in range(len(c))]
    non_indeces = [i for i in range(len(v_phi)) if not (i in indeces)]
    
    s = plt.scatter(v_phi[non_indeces], np.abs(v_z)[non_indeces], c='grey', s=5)
    for c, color, marker in zip(Clusters, colors, markers):
        plt.scatter(v_phi[c], np.abs(v_z)[c], c=color, marker=marker, s=75)

    plt.gcf().set_size_inches(25, 10)
    plt.gcf().savefig("plots/toomre_paper_clusters.eps", bbox_inches='tight',pad_inches=0.01, dpi=100, format='eps')
    plt.gcf().savefig("plots/toomre_paper_clusters.png", bbox_inches='tight',pad_inches=0.01, dpi=100, format='png')

def plot_toomre_paper_v2 (v_phi, v_phi_err, v_z, v_z_err):
    plt.clf()

    plt.xlabel(r"$V_\phi$ (km s$^{-1}$)")
    plt.ylabel(r"$V_\perp$ (km s$^{-1}$)")
    plt.ylim(0,500)
    plt.xlim(-600,600)
    s = plt.scatter(v_phi, np.abs(v_z), c='black', s=5)
    #plt.scatter([-400], [400], marker='.', s=15, c='blue')
    plt.plot([500-5*np.mean(v_phi_err),500+5*np.mean(v_phi_err)], [400,400], linewidth=1, color='black')
    plt.plot([500,500], [400-5*np.mean(v_z_err),400+5*np.mean(v_z_err)], linewidth=1, color='black')
    
    MW_phi = np.arange (0, np.pi+0.0001, 0.0001)
    MW_x_arr = 100*np.cos(MW_phi) + 220 
    MW_y_arr = np.abs(100*np.sin(MW_phi))
    plt.plot(MW_x_arr, MW_y_arr, c='black', linewidth=1, linestyle='--')

    plt.plot((0,0), (0,500), c='black', linewidth=1, linestyle='-.')

    plt.gcf().set_size_inches(9, 4)
    plt.gcf().savefig("plots/toomre_paper_v2.eps", bbox_inches='tight',pad_inches=0.01, dpi=100, format='eps')
    plt.gcf().savefig("plots/toomre_paper_v2.png", bbox_inches='tight',pad_inches=0.01, dpi=100, format='png')


def plot_toomre_EuFe_paper (v_phi, v_z, EuFe):
    plt.clf()

    plt.xlabel(r"$V_\phi$ (km/s)")
    plt.ylabel(r"$V_\perp$ (km/s)")
    plt.ylim(0,500)
    plt.xlim(-600,600)
    s = plt.scatter(v_phi, np.abs(v_z), c=EuFe, s=5, cmap='gist_rainbow')
    
    MW_phi = np.arange (0, np.pi+0.0001, 0.0001)
    MW_x_arr = 100*np.cos(MW_phi) + 220 
    MW_y_arr = np.abs(100*np.sin(MW_phi))
    plt.plot(MW_x_arr, MW_y_arr, c='black', linewidth=1, linestyle='--')

    #cbar_ax = fig.add_axes([0.85,0.15,0.05,0.7])
    cbar=plt.colorbar()
    cbar.set_label("[Eu/Fe]")
    #plt.colorbar()

    plt.gcf().set_size_inches(10, 4)
    plt.gcf().savefig("plots/toomre_EuFe_paper.eps", bbox_inches='tight',pad_inches=0.01, dpi=100, format='eps')
    plt.gcf().savefig("plots/toomre_EuFe_paper.png", bbox_inches='tight',pad_inches=0.01, dpi=100, format='png')


def plot_kinematics_paper (Clusters, colors, markers, FeH, Energy, J_phi, J_r, J_z, Ecc, r_peri, r_apo, Z_max):
 
    Energy = Energy/100000
    J_phi = J_phi/1000
    J_r = J_r/1000
    J_z = J_z/1000

    plt.clf()
    fig, axes = plt.subplots(nrows=2, ncols=4)

    indeces = [c[i] for c in Clusters for i in range(len(c))]
    non_indeces = [i for i in range(len(Energy)) if not (i in indeces)]
    axes[0,0].set_ylabel(r"Energy   (km$^2$ s$^{-2}\times 10^5$)")
    axes[0,0].set_xlabel("[Fe/H]")
    #axes[0,0].scatter(FeH[non_indeces], Energy[non_indeces], c='grey', marker='o', s=5)
    for c, color, marker in zip(Clusters, colors, markers):
        axes[0,0].scatter(FeH[c], Energy[c], c=color, marker=marker, s=20)

    indeces = [c[i] for c in Clusters for i in range(len(c))]
    non_indeces = [i for i in range(len(J_phi)) if not (i in indeces)]
    axes[0,1].set_ylabel(r"J$_\phi$   (kpc km s$^{-1}\times 10^3$)")
    axes[0,1].set_xlabel("[Fe/H]")
    #axes[0,1].scatter(FeH[non_indeces], J_phi[non_indeces], c='grey', marker='o', s=5)
    for c, color, marker in zip(Clusters, colors, markers):
        axes[0,1].scatter(FeH[c], J_phi[c], c=color, marker=marker, s=20)

    indeces = [c[i] for c in Clusters for i in range(len(c))]
    non_indeces = [i for i in range(len(J_r)) if not (i in indeces)]
    axes[0,2].set_ylabel(r"J$_r$   (kpc km s$^{-1}\times 10^3$)")
    axes[0,2].set_xlabel("[Fe/H]")
    #axes[0,2].scatter(FeH[non_indeces], J_r[non_indeces], c='grey', marker='o', s=5)
    for c, color, marker in zip(Clusters, colors, markers):
        axes[0,2].scatter(FeH[c], J_r[c], c=color, marker=marker, s=20)

    indeces = [c[i] for c in Clusters for i in range(len(c))]
    non_indeces = [i for i in range(len(J_z)) if not (i in indeces)]
    axes[0,3].set_ylabel(r"J$_z$   (kpc km s$^{-1}\times 10^3$)")
    axes[0,3].set_xlabel("[Fe/H]")
    #axes[0,3].scatter(FeH[non_indeces], J_z[non_indeces], c='grey', marker='o', s=5)
    for c, color, marker in zip(Clusters, colors, markers):
        axes[0,3].scatter(FeH[c], J_z[c], c=color, marker=marker, s=20)

    indeces = [c[i] for c in Clusters for i in range(len(c))]
    non_indeces = [i for i in range(len(ecc)) if not (i in indeces)]
    axes[1,0].set_ylabel(r"Eccentricity ($e$)")
    axes[1,0].set_xlabel("[Fe/H]")
    #axes[1,0].scatter(FeH[non_indeces], ecc[non_indeces], c='grey', marker='o', s=5)
    for c, color, marker in zip(Clusters, colors, markers):
        axes[1,0].scatter(FeH[c], ecc[c], c=color, marker=marker, s=20)

    indeces = [c[i] for c in Clusters for i in range(len(c))]
    non_indeces = [i for i in range(len(r_peri)) if not (i in indeces)]
    axes[1,1].set_ylabel(r"r$_{peri}$   (kpc)")
    axes[1,1].set_xlabel("[Fe/H]")
    #axes[1,1].scatter(FeH[non_indeces], r_peri[non_indeces], c='grey', marker='o', s=5)
    for c, color, marker in zip(Clusters, colors, markers):
        axes[1,1].scatter(FeH[c], r_peri[c], c=color, marker=marker, s=20)

    indeces = [c[i] for c in Clusters for i in range(len(c))]
    non_indeces = [i for i in range(len(r_apo)) if not (i in indeces)]
    axes[1,2].set_ylabel(r"r$_{apo}$   (kpc)")
    axes[1,2].set_xlabel("[Fe/H]")
    #axes[1,2].scatter(FeH[non_indeces], r_apo[non_indeces], c='grey', marker='o', s=5)
    for c, color, marker in zip(Clusters, colors, markers):
        axes[1,2].scatter(FeH[c], r_apo[c], c=color, marker=marker, s=20)

    indeces = [c[i] for c in Clusters for i in range(len(c))]
    non_indeces = [i for i in range(len(Z_max)) if not (i in indeces)]
    axes[1,3].set_ylabel(r"Z$_{\max}$   (kpc)")
    axes[1,3].set_xlabel("[Fe/H]")
    #axes[1,3].scatter(FeH[non_indeces], Z_max[non_indeces], c='grey', marker='o', s=5)
    for c, color, marker in zip(Clusters, colors, markers):
        axes[1,3].scatter(FeH[c], Z_max[c], c=color, marker=marker, s=20)

    plt.gcf().set_size_inches(12, 6)
    fig.tight_layout()
    plt.gcf().savefig("plots/kinematics_paper.eps", bbox_inches='tight',pad_inches=0.01, dpi=100, format='eps')
    plt.gcf().savefig("plots/kinematics_paper.png", bbox_inches='tight',pad_inches=0.01, dpi=100, format='png')
    plt.close()


def plot_kinematics_paper_v2 (colors, markers, FeH, Energy, Energy_err, J_phi, J_phi_err, J_r, J_r_err, J_z, J_z_err, ecc, ecc_err, r_peri, r_peri_err, r_apo, r_apo_err, Z_max, Z_max_err):
 
    Energy = Energy/100000
    Energy_err = Energy_err/100000
    J_phi = J_phi/1000
    J_phi_err = J_phi_err/1000
    J_r = J_r/1000
    J_r_err = J_r_err/1000
    J_z = J_z/1000
    J_z_err = J_z_err/1000

    plt.clf()
    fig, axes = plt.subplots(nrows=2, ncols=4)

    axes[0,0].set_ylabel(r"Energy   (km$^2$ s$^{-2}\times 10^5$)")
    axes[0,0].set_xlabel("[Fe/H]")
    axes[0,0].scatter(FeH, Energy, s=5, color='black')
    axes[0,0].plot([-1.2-0.15,-1.2+0.15], [-0.25,-0.25], linewidth=1, color='black')
    axes[0,0].plot([-1.2,-1.2], [-0.25-5*np.mean(Energy_err),-0.25+5*np.mean(Energy_err)], linewidth=1, color='black')

    axes[0,1].set_ylabel(r"J$_r$   (kpc km s$^{-1}\times 10^3$)")
    axes[0,1].set_xlabel("[Fe/H]")
    axes[0,1].set_ylim(0,9) 
    axes[0,1].scatter(FeH, J_r, s=5, color='black')
    axes[0,1].plot([-1.2-0.15,-1.2+0.15], [8,8], linewidth=1, color='black')
    axes[0,1].plot([-1.2,-1.2], [8-5*np.mean(J_r_err),8+5*np.mean(J_r_err)], linewidth=1, color='black')

    axes[0,2].set_ylabel(r"J$_\phi$   (kpc km s$^{-1}\times 10^3$)")
    axes[0,2].set_xlabel("[Fe/H]")
    axes[0,2].scatter(FeH, J_phi, s=5, color='black')
    xlim = axes[0,2].get_xlim()
    axes[0,2].plot((xlim[0],xlim[1]), (0,0), color='black', linewidth=1, linestyle='--')
    axes[0,2].set_xlim(xlim[0],xlim[1])
    axes[0,2].plot([-1.2-0.15,-1.2+0.15], [3,3], linewidth=1, color='black')
    axes[0,2].plot([-1.2,-1.2], [3-5*np.mean(J_phi_err),3+5*np.mean(J_phi_err)], linewidth=1, color='black')

    axes[0,3].set_ylabel(r"J$_z$   (kpc km s$^{-1}\times 10^3$)")
    axes[0,3].set_xlabel("[Fe/H]")
    axes[0,3].scatter(FeH, J_z, s=5, color='black')
    axes[0,3].plot([-1.2-0.15,-1.2+0.15], [3.6,3.6], linewidth=1, color='black')
    axes[0,3].plot([-1.2,-1.2], [3.6-5*np.mean(J_z_err),3.6+5*np.mean(J_z_err)], linewidth=1, color='black')

    axes[1,0].set_ylabel(r"Eccentricity ($e$)")
    axes[1,0].set_xlabel("[Fe/H]")
    axes[1,0].scatter(FeH, ecc, s=5, color='black')
    axes[1,0].plot([-0.8-0.15,-0.8+0.15], [0.9,0.9], linewidth=1, color='black')
    axes[1,0].plot([-0.8,-0.8], [0.9-5*np.mean(ecc_err),0.9+5*np.mean(ecc_err)], linewidth=1, color='black')

    axes[1,1].set_ylabel(r"r$_{peri}$   (kpc)")
    axes[1,1].set_xlabel("[Fe/H]")
    axes[1,1].scatter(FeH, r_peri, s=5, color='black')
    axes[1,1].plot([-1.2-0.15,-1.2+0.15], [16,16], linewidth=1, color='black')
    axes[1,1].plot([-1.2,-1.2], [16-5*np.mean(r_peri_err),16+5*np.mean(r_peri_err)], linewidth=1, color='black')

    axes[1,2].set_ylabel(r"r$_{apo}$   (kpc)")
    axes[1,2].set_xlabel("[Fe/H]")
    axes[1,2].set_ylim(0,150)
    axes[1,2].scatter(FeH, r_apo, s=5, color='black')
    axes[1,2].plot([-1.2-0.15,-1.2+0.15], [130,130], linewidth=1, color='black')
    axes[1,2].plot([-1.2,-1.2], [130-5*np.mean(r_apo_err),130+5*np.mean(r_apo_err)], linewidth=1, color='black')

    axes[1,3].set_ylabel(r"Z$_{\max}$   (kpc)")
    axes[1,3].set_xlabel("[Fe/H]")
    axes[1,3].scatter(FeH, Z_max, s=5, color='black')
    axes[1,3].plot([-1.2-0.15,-1.2+0.15], [145,145], linewidth=1, color='black')
    axes[1,3].plot([-1.2,-1.2], [145-5*np.mean(Z_max_err),145+5*np.mean(Z_max_err)], linewidth=1, color='black')

    plt.gcf().set_size_inches(12, 6)
    fig.tight_layout()
    plt.gcf().savefig("plots/kinematics_paper_v2.eps", bbox_inches='tight',pad_inches=0.01, dpi=100, format='eps')
    plt.gcf().savefig("plots/kinematics_paper_v2.png", bbox_inches='tight',pad_inches=0.01, dpi=100, format='png')
    plt.close()


def plot_mean_metallicities_RV (Clusters, colors, FeH, RV):
    indeces = [c[i] for c in Clusters for i in range(len(c))]
    non_indeces = [i for i in range(len(FeH)) if not (i in indeces)]

    plt.clf()
    plt.title("r-process clusters", size=24)
    plt.xlabel("<[Fe/H]>", size=24)
    plt.ylabel("dRV (km/s)", size=24)
    plt.tick_params(labelsize=18)
    for i, c in enumerate(Clusters):
        FeH_avg = np.mean(FeH[c])
        RV_std = np.std(RV[c])
        plt.scatter (FeH_avg, RV_std, c=colors[i], marker=params.markers[i], s=400)
    plt.gcf().set_size_inches(25.6, 14.4)
    plt.gcf().savefig("plots/Metallicities_RV.png", dpi=100)
    plt.close()    


def plot_orbital_velocities (Clusters, vRg, vTg):
    indeces = [c[i] for c in Clusters for i in range(len(c))]
    non_indeces = [i for i in range(len(vRg)) if not (i in indeces)]

    plt.clf()
    plt.title("r-process clusters", size=24)
    plt.xlabel("vRg", size=24)
    plt.ylabel("vTg", size=24)
    plt.tick_params(labelsize=18)
    for i, c in enumerate(Clusters):
        plt.scatter(vRg[c], vTg[c], c=params.colors[i], marker=params.markers[i], s=200)
    plt.scatter(vRg[non_indeces], vTg[non_indeces], c='grey', marker='o', s=10)
    plt.gcf().set_size_inches(25.6, 14.4)
    plt.gcf().savefig("plots/Orbital_velocities_clusters.png", dpi=100)
    plt.close() 


def plot_vTg_rapo (Clusters, vTg, r_apo):
    indeces = [c[i] for c in Clusters for i in range(len(c))]
    non_indeces = [i for i in range(len(r_apo)) if not (i in indeces)]

    plt.clf()
    plt.title("r-process clusters", size=24)
    plt.xlabel("r_apo", size=24)
    plt.ylabel("vTg", size=24)
    plt.tick_params(labelsize=18)
    for i, c in enumerate(Clusters):
        plt.scatter(r_apo[c], vTg[c], c=params.colors[i], marker=params.markers[i], s=200)
    plt.gcf().set_size_inches(25.6, 14.4)
    plt.gcf().savefig("plots/vTg_rapo_clusters.png", dpi=100)
    plt.close() 


def plot_Eu_FeH_clusters_paper (Clusters, EuFe, A_Eu, FeH):
    indeces = [c[i] for c in Clusters for i in range(len(c))]
    non_indeces = [i for i in range(len(EuFe)) if not (i in indeces)]

    plt.clf()
    fig, axes = plt.subplots(nrows=2, ncols=1)

    xticks = ticker.MaxNLocator(5)
    yticks = ticker.FixedLocator([-2.5, -2, -1.5, -1, -0.5, 0, 0.5])

    axes[0].set_ylabel("A(Eu)")
    #axes[0].set_xlim(-3.5,-1.3)
    #axes[0].set_ylim(-2.5,0.5)
    axes[0].xaxis.set_major_locator(xticks)
    axes[0].yaxis.set_major_locator(yticks)
    axes[0].set_xticklabels([])
    axes[0].scatter(FeH[non_indeces], A_Eu[non_indeces], c='grey', marker='o', s=7)
    for i, c in enumerate(Clusters):
        axes[0].scatter(FeH[c], A_Eu[c], c=params.colors[i], marker=params.markers[i], s=30)
    
    xticks = ticker.MaxNLocator(5)
    yticks = ticker.MaxNLocator(5)

    axes[1].set_xlabel("[Fe/H]")
    axes[1].set_ylabel("[Eu/Fe]")
    #axes[1].set_xlim(-3.5,-1.3)
    #axes[1].set_ylim(0.2,2)
    axes[1].xaxis.set_major_locator(xticks)
    axes[1].yaxis.set_major_locator(yticks)
    axes[1].scatter(FeH[non_indeces], EuFe[non_indeces], c='grey', marker='o', s=7)
    for i, c in enumerate(Clusters):
        axes[1].scatter(FeH[c], EuFe[c], c=params.colors[i], marker=params.markers[i], s=30)

    plt.gcf().set_size_inches(5, 8)
    fig.tight_layout()
    plt.gcf().savefig("plots/Eu_FeH_clusters_paper.eps", bbox_inches='tight',pad_inches=0.01, dpi=100, format='eps')
    plt.gcf().savefig("plots/Eu_FeH_clusters_paper.png", bbox_inches='tight',pad_inches=0.01, dpi=100, format='png')
    plt.close()


def plot_Eu_FeH_paper (EuFe, A_Eu, FeH, T):
    plt.clf()
    fig, axes = plt.subplots(nrows=2, ncols=1)

    xticks = ticker.MaxNLocator(5)
    #yticks = ticker.FixedLocator([-2.5, -2, -1.5, -1, -0.5, 0, 0.5])

    axes[0].set_xlabel("[Fe/H]")
    axes[0].set_ylabel("[Eu/H]")
    #axes[0].set_xlim(-3.5,-1.0)
    #axes[0].set_ylim(-3.5,0.5)
    #axes[0].yaxis.set_ticks([-3.5,-2.5,-1.5,-0.5,0.5])
    #axes[0].xaxis.set_major_locator(xticks)
    #axes[0].yaxis.set_major_locator(yticks)
    #axes[0].set_yticklabels([-3.5,-2.5,-1.5,-0.5,0.5])
    axes[0].scatter(FeH, A_Eu-0.52, c=T, s=5, cmap='gist_rainbow_r')
    axes[0].plot([-4.0,-1.0],[-3.3,-0.3],linewidth=1, linestyle='--', color='black')
    axes[0].plot([-4.0,-1.0],[-3.7,-0.7], linewidth=1, color='black')    
    #axes[0].scatter([-1.5], [-3], marker='.', s=25)
    axes[0].plot([-1.25-0.15, -1.25+0.15], [-1.5, -1.5], linewidth=1, color='black')
    axes[0].plot([-1.25, -1.25], [-1.5+0.15,-1.5-0.15], linewidth=1, color='black')

    xticks = ticker.MaxNLocator(5)
    yticks = ticker.MaxNLocator(5)

    axes[1].set_xlabel("[Fe/H]")
    axes[1].set_ylabel("[Eu/Fe]")
    #axes[1].set_xlim(-3.5,-1.0)
    #axes[1].set_ylim(0,2.5)
    #axes[1].xaxis.set_major_locator(xticks)
    #axes[1].yaxis.set_major_locator(yticks)
    s_1 = axes[1].scatter(FeH, EuFe, c=T, s=5, cmap='gist_rainbow_r')
    axes[1].plot([-4.0,-1.0],[0.3,0.3],linewidth=1,color='black')
    axes[1].plot([-4.0,-1.0],[0.7,0.7],linewidth=1,linestyle='--',color='black')
    #axes[1].scatter([-1.5], [2.0], marker='.', s=25)
    axes[1].plot([-1.25-0.15, -1.25+0.15], [2.0, 2.0], linewidth=1, color='black')
    axes[1].plot([-1.25, -1.25], [2.0-0.1, 2.0+0.1], linewidth=1, color='black')

    fig.subplots_adjust(right=0.8, hspace=0.3)
    cbar_ax = fig.add_axes([0.85,0.15,0.05,0.7])
    cbar=fig.colorbar(s_1, cax=cbar_ax)
    cbar.set_label('Temperature (K)')

    plt.gcf().set_size_inches(5, 6)
    plt.gcf().savefig("plots/Eu_FeH_paper.eps", bbox_inches='tight',pad_inches=0.01, dpi=100, format='eps')
    plt.gcf().savefig("plots/Eu_FeH_paper.png", bbox_inches='tight',pad_inches=0.01, dpi=100, format='png')
    plt.close()


def plot_distance_paper (dist):
    plt.clf()
    plt.xlabel("Distance (kpc)")
    plt.ylabel("Frequency")
    plt.xlim(0,13)
    dist = dist[dist<=13]
    plt.hist(dist, bins=50, color='black',fill=False,linewidth=1,histtype='step')
    plt.gcf().set_size_inches(5,3)
    plt.gcf().savefig("plots/dist_paper.png", bbox_inches='tight', pad_inches=0.01, dpi=100, format='png')
    plt.gcf().savefig("plots/dist_paper.eps", bbox_inches='tight', pad_inches=0.01, dpi=100, format='eps')
    plt.close()


def plot_distances_paper (r_peri, r_apo, Z_max):
    plt.clf()
    fig, axes = plt.subplots(nrows=3, ncols=1)

    r_apo = r_apo[r_apo<=50]
    Z_max = Z_max[Z_max<=50]

    axes[0].set_xlabel(r"$r_\mathrm{peri}$ (kpc)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_xlim(0,20)
    axes[0].hist(r_peri, bins=50, color='black',fill=False,linewidth=1,histtype='step')

    axes[1].set_xlabel(r"$r_\mathrm{apo}$ (kpc)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_xlim(0,50)
    axes[1].hist(r_apo, bins=50, color='black',fill=False,linewidth=1,histtype='step')

    axes[2].set_xlabel(r"$Z_\mathrm{max}$ (kpc)")
    axes[2].set_ylabel("Frequency")
    axes[2].set_xlim(0,50)
    axes[2].hist(Z_max, bins=50, color='black',fill=False,linewidth=1,histtype='step')

    plt.gcf().set_size_inches(5,10)
    plt.gcf().savefig("plots/distances_paper.png", bbox_inches='tight', pad_inches=0.01, dpi=100, format='png')
    plt.gcf().savefig("plots/distances_paper.eps", bbox_inches='tight', pad_inches=0.01, dpi=100, format='eps')


def plot_motion_integrals_paper (Clusters, Energy, ecc, J_phi, Energy_limits, ecc_limits, J_phi_limits):
    indeces = [c[i] for c in Clusters for i in range(len(c))]
    non_indeces = [i for i in range(len(Energy)) if not (i in indeces)]

    plt.clf()
    fig, axes = plt.subplots(nrows=1, ncols=3)

    axes[0].set_xlabel(r"J$_\phi$   (kpc km s$^{-1}\times 10^3$)")
    axes[0].set_ylabel(r"Energy   (km$^2$ s$^{-2}\times 10^5$)")
    axes[0].set_xlim(J_phi_limits)
    axes[0].set_ylim(Energy_limits)
    axes[0].scatter(J_phi[non_indeces]/1e3, Energy[non_indeces]/1e5, c='grey', marker='o', s=3)
    for i, c in enumerate(Clusters):
        axes[0].scatter(J_phi[c]/1e3, Energy[c]/1e5, c=params.colors[i], marker=params.markers[i], s=30)

    axes[1].set_xlabel(r"Eccentricity ($e$)")
    axes[1].set_ylabel(r"Energy   (km$^2$ s$^{-2}\times 10^5$)")
    axes[1].set_xlim(ecc_limits)
    axes[1].set_ylim(Energy_limits)
    axes[1].scatter(ecc[non_indeces], Energy[non_indeces]/1e5, c='grey', marker='o', s=3)
    for i, c in enumerate(Clusters):
        axes[1].scatter(ecc[c], Energy[c]/1e5, c=params.colors[i], marker=params.markers[i], s=30)

    axes[2].set_xlabel(r"J$_\phi$   (kpc km s$^{-1}\times 10^3$)")
    axes[2].set_ylabel(r"Eccentricity ($e$)")
    axes[2].set_xlim(J_phi_limits)
    axes[2].set_ylim(ecc_limits)
    axes[2].scatter(J_phi[non_indeces]/1e3, ecc[non_indeces], c='grey', marker='o', s=3)
    for i, c in enumerate(Clusters):
        axes[2].scatter(J_phi[c]/1e3, ecc[c], c=params.colors[i], marker=params.markers[i], s=30)

    plt.gcf().set_size_inches(15, 4)
    plt.subplots_adjust(wspace=0.3,hspace=0.3)
    #fig.tight_layout()
    plt.gcf().savefig("plots/motion_integrals_paper.eps", bbox_inches='tight',pad_inches=0.01, dpi=100, format='eps')
    plt.gcf().savefig("plots/motion_integrals_paper.png", bbox_inches='tight',pad_inches=0.01, dpi=100, format='png')
    plt.close()

def plot_motion_integrals_Ian_paper (Clusters, Energy, J_r, J_z, J_phi, Energy_limits, J_r_limits, J_z_limits, J_phi_limits) :
    indeces = [c[i] for c in Clusters for i in range(len(c))]
    non_indeces = [i for i in range(len(Energy)) if not (i in indeces)]

    plt.clf()
    fig, axes = plt.subplots(nrows=3, ncols=3)

    axes[1,2].axis('off')
    axes[2,1].axis('off')
    axes[2,2].axis('off')

    #axes[0,0].set_xlabel(r"Energy   (km$^2$ s$^{-2}\times 10^5$)")
    axes[0,0].xaxis.set_ticklabels([])
    axes[0,0].set_ylabel(r"J$_r$   (kpc km s$^{-1}\times 10^3$)")
    axes[0,0].set_xlim(Energy_limits)
    axes[0,0].set_ylim(J_r_limits)
    axes[0,0].scatter(Energy[non_indeces]/1e5, J_r[non_indeces]/1e3, c='grey', marker='o', s=3)
    for i, c in enumerate(Clusters):
        axes[0,0].scatter(Energy[c]/1e5, J_r[c]/1e3, c=params.colors[i], marker=params.markers[i], s=30)

    #axes[1,0].set_xlabel(r"Energy   (km$^2$ s$^{-2}\times 10^5$)")
    axes[1,0].xaxis.set_ticklabels([])
    axes[1,0].set_ylabel(r"J$_z$   (kpc km s$^{-1}\times 10^3$)")
    axes[1,0].set_xlim(Energy_limits)
    axes[1,0].set_ylim(J_z_limits)
    axes[1,0].scatter(Energy[non_indeces]/1e5, J_z[non_indeces]/1e3, c='grey', marker='o', s=3)
    for i, c in enumerate(Clusters):
        axes[1,0].scatter(Energy[c]/1e5, J_z[c]/1e3, c=params.colors[i], marker=params.markers[i], s=30)

    axes[2,0].set_xlabel(r"Energy   (km$^2$ s$^{-2}\times 10^5$)")
    axes[2,0].set_ylabel(r"J$_\phi$   (kpc km s$^{-1}\times 10^3$)")
    axes[2,0].set_xlim(Energy_limits)
    axes[2,0].set_ylim(J_phi_limits)
    axes[2,0].scatter(Energy[non_indeces]/1e5, J_phi[non_indeces]/1e3, c='grey', marker='o', s=3)
    for i, c in enumerate(Clusters):
        axes[2,0].scatter(Energy[c]/1e5, J_phi[c]/1e3, c=params.colors[i], marker=params.markers[i], s=30)

    #axes[0,1].set_xlabel(r"J$_\phi$   (kpc km s$^{-1}\times 10^3$)")
    #axes[0,1].set_ylabel(r"J$_r$   (kpc km s$^{-1}\times 10^3$)")
    axes[0,1].xaxis.set_ticklabels([])
    axes[0,1].yaxis.set_ticklabels([])
    axes[0,1].set_xlim(J_phi_limits)
    axes[0,1].set_ylim(J_r_limits)
    axes[0,1].scatter(J_phi[non_indeces]/1e3, J_z[non_indeces]/1e3, c='grey', marker='o', s=3)
    for i, c in enumerate(Clusters):
        axes[0,1].scatter(J_phi[c]/1e3, J_r[c]/1e3, c=params.colors[i], marker=params.markers[i], s=30)

    axes[1,1].set_xlabel(r"J$_\phi$   (kpc km s$^{-1}\times 10^3$)")
    #axes[1,1].set_ylabel(r"J$_z$   (kpc km s$^{-1}\times 10^3$)")
    axes[1,1].yaxis.set_ticklabels([])
    axes[1,1].set_xlim(J_phi_limits)
    axes[1,1].set_ylim(J_z_limits)
    axes[1,1].scatter(J_phi[non_indeces]/1e3, J_z[non_indeces]/1e3, c='grey', marker='o', s=3)
    for i, c in enumerate(Clusters):
        axes[1,1].scatter(J_phi[c]/1e3, J_z[c]/1e3, c=params.colors[i], marker=params.markers[i], s=30)

    axes[0,2].set_xlabel(r"J$_z$   (kpc km s$^{-1}\times 10^3$)")
    #axes[0,2].set_ylabel(r"J$_r$   (kpc km s$^{-1}\times 10^3$)")
    axes[0,2].yaxis.set_ticklabels([])
    axes[0,2].set_xlim(J_z_limits)
    axes[0,2].set_ylim(J_r_limits) 
    axes[0,2].scatter(J_z[non_indeces]/1e3, J_r[non_indeces]/1e3, c='grey', marker='o', s=3)
    for i, c in enumerate(Clusters):
        axes[0,2].scatter(J_z[c]/1e3, J_r[c]/1e3, c=params.colors[i], marker=params.markers[i], s=30)

    plt.gcf().set_size_inches(15, 10)
    plt.subplots_adjust(wspace=0.1,hspace=0.1)
    #fig.tight_layout()
    plt.gcf().savefig("plots/motion_integrals_Ian_paper.eps", bbox_inches='tight',pad_inches=0.01, dpi=100, format='eps')
    plt.gcf().savefig("plots/motion_integrals_Ian_paper.png", bbox_inches='tight',pad_inches=0.01, dpi=100, format='png')
    plt.close()


def plot_motion_integrals_Ian_paper_v2 (Clusters, Energy, J_r, J_z, J_phi, Energy_limits, J_r_limits, J_z_limits, J_phi_limits) :
    indeces = [c[i] for c in Clusters for i in range(len(c))]
    non_indeces = [i for i in range(len(Energy)) if not (i in indeces)]

    plt.clf()
    fig, axes = plt.subplots(nrows=3, ncols=1)
 
    axes[0].set_xlabel(r"J$_r$   (kpc km s$^{-1}\times 10^3$)")
    axes[0].set_ylabel(r"Energy   (km$^2$ s$^{-2}\times 10^5$)")
    axes[0].set_xlim(J_r_limits)
    axes[0].set_ylim(Energy_limits)
    axes[0].scatter(J_r[non_indeces]/1e3, Energy[non_indeces]/1e5, c='grey', marker='o', s=3)
    for i, c in enumerate(Clusters):
        axes[0].scatter(J_r[c]/1e3, Energy[c]/1e5, c=params.colors[i], marker=params.markers[i], s=30)

    axes[1].set_xlabel(r"J$_z$   (kpc km s$^{-1}\times 10^3$)")
    axes[1].set_ylabel(r"Energy   (km$^2$ s$^{-2}\times 10^5$)")
    axes[1].set_xlim(J_z_limits)
    axes[1].set_ylim(Energy_limits)
    axes[1].scatter(J_z[non_indeces]/1e3, Energy[non_indeces]/1e5, c='grey', marker='o', s=3)
    for i, c in enumerate(Clusters):
        axes[1].scatter(J_z[c]/1e3, Energy[c]/1e5, c=params.colors[i], marker=params.markers[i], s=30)

    axes[2].set_xlabel(r"J$_\phi$   (kpc km s$^{-1}\times 10^3$)")
    axes[2].set_ylabel(r"Energy   (km$^2$ s$^{-2}\times 10^5$)")
    axes[2].set_xlim(J_phi_limits)
    axes[2].set_ylim(Energy_limits)
    axes[2].scatter(J_phi[non_indeces]/1e3, Energy[non_indeces]/1e5, c='grey', marker='o', s=3)
    for i, c in enumerate(Clusters):
        axes[2].scatter(J_phi[c]/1e3, Energy[c]/1e5, c=params.colors[i], marker=params.markers[i], s=30)

    plt.gcf().set_size_inches(5, 9)
    plt.subplots_adjust(hspace=0.3)
    plt.gcf().savefig("plots/motion_integrals_Ian_paper_v2.eps", bbox_inches='tight',pad_inches=0.01, dpi=100, format='eps')
    plt.gcf().savefig("plots/motion_integrals_Ian_paper_v2.png", bbox_inches='tight',pad_inches=0.01, dpi=100, format='png')
    plt.close()


def plot_E_J_phi_paper (Clusters, J_phi, J_phi_err, E, Energy_err, J_phi_lims=(-3,3), E_lims=(-3,0)):
    indeces = [c[i] for c in Clusters for i in range(len(c))]
    non_indeces = [i for i in range(len(E)) if not (i in indeces)]

    lengths_x = [1,10,4]
    lengths_y = [1,10,1]
    
    num0x = 0
    num1x = lengths_x[0]
    num2x = lengths_x[0]+lengths_x[1]
    num3x = lengths_x[0]+lengths_x[1]+2*lengths_x[2]
    num4x = lengths_x[0]+2*lengths_x[1]+2*lengths_x[2]
    num5x = 2*lengths_x[0]+2*lengths_x[1]+2*lengths_x[2]

    num0y = 0
    num1y = lengths_y[0]
    num2y = lengths_y[0]+lengths_y[1]
    num3y = lengths_y[0]+lengths_y[1]+2*lengths_y[2]
    num4y = lengths_y[0]+2*lengths_y[1]+2*lengths_y[2]
    num5y = 2*lengths_y[0]+2*lengths_y[1]+2*lengths_y[2]

    plt.clf()
    gs = gridspec.GridSpec(2*lengths_y[0]+2*lengths_y[1]+2*lengths_y[2], 2*lengths_x[0]+2*lengths_x[1]+2*lengths_x[2])
    ax00 = plt.subplot(gs[num1y:num2y, num1x:num2x])
    ax01 = plt.subplot(gs[num1y:num2y, num3x:num4x])
    ax10 = plt.subplot(gs[num3y:num4y, num1x:num2x])
    ax11 = plt.subplot(gs[num3y:num4y, num3x:num4x])
    ax_legend = plt.subplot(gs[num2y:num3y, num2x:num3x])
    ax_else_up = plt.subplot(gs[num1y:num2y, num2x:num3x])
    ax_else_down = plt.subplot(gs[num3y:num4y, num2x:num3x])
    ax_else_left = plt.subplot(gs[num2y:num3y, num1x:num2x])
    ax_else_right = plt.subplot(gs[num2y:num3y, num3x:num4x])
    ax_else_upup = plt.subplot(gs[num0y:num1y, num0x:num4x])
    ax_else_downdown = plt.subplot(gs[num4y:num5y, num1x:num5x])
    ax_else_leftleft = plt.subplot(gs[num1y:num5y, num0x:num1x])
    ax_else_rightright = plt.subplot(gs[num0y:num4y, num4x:num5x])
    
    ax_else_up.axis('off')
    ax_else_down.axis('off')
    ax_else_left.axis('off')
    ax_else_right.axis('off')
    ax_else_upup.axis('off')
    ax_else_downdown.axis('off')
    ax_else_leftleft.axis('off')
    ax_else_rightright.axis('off')
    #ax_legend.get_xaxis().set_visible(False)
    #ax_legend.get_yaxis().set_visible(False)
    ax_legend.axis('off') 

    ax00.set_xlim (J_phi_lims[0]-0.025,J_phi_lims[1]+0.025)
    ax00.set_ylim (E_lims[0]-0.025,E_lims[1]+0.025)
    ax00.set_xlabel (r'$J_\phi$ ($10^3$ kpc km s$^{-1}$)')
    ax00.set_ylabel (r'Energy ($10^5$ km$^2$ s$^{-2}$)')
    ax00.plot((J_phi_lims[0],J_phi_lims[0]), (E_lims[0],E_lims[1]), color="k", alpha=0.3)
    ax00.plot((J_phi_lims[0],J_phi_lims[1]), (E_lims[1],E_lims[1]), color="k", alpha=0.3)
    ax00.plot((J_phi_lims[1],J_phi_lims[1]), (E_lims[1],E_lims[0]), color="k", alpha=0.3)
    ax00.plot((J_phi_lims[1],J_phi_lims[0]), (E_lims[0],E_lims[0]), color="k", alpha=0.3)
    ax00.spines['bottom'].set_color('white')
    ax00.spines['top'].set_color('white') 
    ax00.spines['right'].set_color('white')
    ax00.spines['left'].set_color('white')
    #ax00.get_xaxis().set_visible(False)
    #ax00.get_yaxis().set_visible(False)
    ax00.grid(False)

    ax00.plot([-1.5-5*np.mean(J_phi_err)/1e3,-1.5+5*np.mean(J_phi_err)/1e3], [-2.1,-2.1], linewidth=1, color='black')
    ax00.plot([-1.5,-1.5], [-2.1-5*np.mean(Energy_err)/1e5,-2.1+5*np.mean(Energy_err)/1e5], linewidth=1, color='black')

    ax00.scatter(J_phi[non_indeces]/1e3, E[non_indeces]/1e5, color="grey", s=5)
    for i, c in enumerate(Clusters[0:8]):
        ax00.scatter(J_phi[c]/1e3, E[c]/1e5, c=params.colors[i], marker=params.markers[i], s=150*params.marker_scales[i], linewidth=params.marker_linewidths[i])

    legend_parts = []
    for i in range (0,8):
        legend_parts.append(mlines.Line2D([], [], color=params.colors[i], marker=params.markers[i], markeredgewidth=params.marker_linewidths[i], linewidth=params.marker_linewidths[i], linestyle='None', markersize=10*params.marker_scales[i], label='C'+str(i+1)))
    first_legend = ax_legend.legend(handles=legend_parts, loc='center', bbox_to_anchor=(0.4,0.4),fontsize=12)


    ax01.set_xlim (J_phi_lims[0]-0.025,J_phi_lims[1]+0.025)
    ax01.set_ylim (E_lims[0]-0.025,E_lims[1]+0.025)
    ax01.set_xlabel (r'$J_\phi$ ($10^3$ kpc km s$^{-1}$)')
    ax01.set_ylabel (r'Energy ($10^5$ km$^2$ s$^{-2}$)')
    ax01.plot((J_phi_lims[0],J_phi_lims[0]), (E_lims[0],E_lims[1]), color="k", alpha=0.3)
    ax01.plot((J_phi_lims[0],J_phi_lims[1]), (E_lims[1],E_lims[1]), color="k", alpha=0.3)
    ax01.plot((J_phi_lims[1],J_phi_lims[1]), (E_lims[1],E_lims[0]), color="k", alpha=0.3)
    ax01.plot((J_phi_lims[1],J_phi_lims[0]), (E_lims[0],E_lims[0]), color="k", alpha=0.3)
    ax01.spines['bottom'].set_color('white')
    ax01.spines['top'].set_color('white') 
    ax01.spines['right'].set_color('white')
    ax01.spines['left'].set_color('white')
    #ax01.get_xaxis().set_visible(False)
    #ax01.get_yaxis().set_visible(False)
    ax01.grid(False)

    ax01.plot([-1.5-5*np.mean(J_phi_err)/1e3,-1.5+5*np.mean(J_phi_err)/1e3], [-2.1,-2.1], linewidth=1, color='black')
    ax01.plot([-1.5,-1.5], [-2.1-5*np.mean(Energy_err)/1e5,-2.1+5*np.mean(Energy_err)/1e5], linewidth=1, color='black')

    ax01.scatter(J_phi[non_indeces]/1e3, E[non_indeces]/1e5, color="grey", s=5)
    for i, c in enumerate(Clusters[15:23]):
        ax01.scatter(J_phi[c]/1e3, E[c]/1e5, c=params.colors[i+15], marker=params.markers[i+15], s=150*params.marker_scales[i+15], linewidth=params.marker_linewidths[i+15])


    ax10.set_xlim (J_phi_lims[0]-0.025,J_phi_lims[1]+0.025)
    ax10.set_ylim (E_lims[0]-0.025,E_lims[1]+0.025)
    ax10.set_xlabel (r'$J_\phi$ ($10^3$ kpc km s$^{-1}$)')
    ax10.set_ylabel (r'Energy ($10^5$ km$^2$ s$^{-2}$)')
    ax10.plot((J_phi_lims[0],J_phi_lims[0]), (E_lims[0],E_lims[1]), color="k", alpha=0.3)
    ax10.plot((J_phi_lims[0],J_phi_lims[1]), (E_lims[1],E_lims[1]), color="k", alpha=0.3)
    ax10.plot((J_phi_lims[1],J_phi_lims[1]), (E_lims[1],E_lims[0]), color="k", alpha=0.3)
    ax10.plot((J_phi_lims[1],J_phi_lims[0]), (E_lims[0],E_lims[0]), color="k", alpha=0.3)
    ax10.spines['bottom'].set_color('white')
    ax10.spines['top'].set_color('white') 
    ax10.spines['right'].set_color('white')
    ax10.spines['left'].set_color('white')
    #ax10.get_xaxis().set_visible(False)
    #ax10.get_yaxis().set_visible(False)
    ax10.grid(False)

    ax10.plot([-1.5-5*np.mean(J_phi_err)/1e3,-1.5+5*np.mean(J_phi_err)/1e3], [-2.1,-2.1], linewidth=1, color='black')
    ax10.plot([-1.5,-1.5], [-2.1-5*np.mean(Energy_err)/1e5,-2.1+5*np.mean(Energy_err)/1e5], linewidth=1, color='black')

    ax10.scatter(J_phi[non_indeces]/1e3, E[non_indeces]/1e5, color="grey", s=5)
    for i, c in enumerate(Clusters[8:15]):
        ax10.scatter(J_phi[c]/1e3, E[c]/1e5, c=params.colors[i+8], marker=params.markers[i+8], s=150*params.marker_scales[i+8], linewidth=params.marker_linewidths[i+8])


    ax11.set_xlim (J_phi_lims[0]-0.025,J_phi_lims[1]+0.025)
    ax11.set_ylim (E_lims[0]-0.025,E_lims[1]+0.025)
    ax11.set_xlabel (r'$J_\phi$ ($10^3$ kpc km s$^{-1}$)')
    ax11.set_ylabel (r'Energy ($10^5$ km$^2$ s$^{-2}$)')
    ax11.plot((J_phi_lims[0],J_phi_lims[0]), (E_lims[0],E_lims[1]), color="k", alpha=0.3)
    ax11.plot((J_phi_lims[0],J_phi_lims[1]), (E_lims[1],E_lims[1]), color="k", alpha=0.3)
    ax11.plot((J_phi_lims[1],J_phi_lims[1]), (E_lims[1],E_lims[0]), color="k", alpha=0.3)
    ax11.plot((J_phi_lims[1],J_phi_lims[0]), (E_lims[0],E_lims[0]), color="k", alpha=0.3)
    ax11.spines['bottom'].set_color('white')
    ax11.spines['top'].set_color('white') 
    ax11.spines['right'].set_color('white')
    ax11.spines['left'].set_color('white')
    #ax11.get_xaxis().set_visible(False)
    #ax11.get_yaxis().set_visible(False)
    ax11.grid(False)

    ax11.plot([-1.5-5*np.mean(J_phi_err)/1e3,-1.5+5*np.mean(J_phi_err)/1e3], [-2.1,-2.1], linewidth=1, color='black')
    ax11.plot([-1.5,-1.5], [-2.1-5*np.mean(Energy_err)/1e5,-2.1+5*np.mean(Energy_err)/1e5], linewidth=1, color='black')

    ax11.scatter(J_phi[non_indeces]/1e3, E[non_indeces]/1e5, color="grey", s=5)
    for i, c in enumerate(Clusters[23:30]):
        ax11.scatter(J_phi[c]/1e3, E[c]/1e5, c=params.colors[i+23], marker=params.markers[i+23], s=150*params.marker_scales[i+23], linewidth=params.marker_linewidths[i+23])


    plt.gcf().set_size_inches(15, 12)
    plt.subplots_adjust(hspace = 0.001, wspace = 0.001)

    legend_parts = []
    for i in range (0,8):
        legend_parts.append(mlines.Line2D([], [], color=params.colors[i], marker=params.markers[i], markeredgewidth=params.marker_linewidths[i], linewidth=params.marker_linewidths[i], linestyle='None', markersize=10*params.marker_scales[i], label=str(i+1)))
    #legend_parts.append(mlines.Line2D([], [], color=params.colors[i], marker=params.markers[i], linestyle='None', markersize=0, label=''))
    legend_parts.append(mlines.Line2D([], [], color=params.colors[i], marker=params.markers[i], markeredgewidth=params.marker_linewidths[i], linewidth=params.marker_linewidths[i], linestyle='None', markersize=0, label=''))
    for i in range (8,15):
        legend_parts.append(mlines.Line2D([], [], color=params.colors[i], marker=params.markers[i], markeredgewidth=params.marker_linewidths[i], linewidth=params.marker_linewidths[i], linestyle='None', markersize=10*params.marker_scales[i], label=str(i+1)))
    for i in range (15,23):
        legend_parts.append(mlines.Line2D([], [], color=params.colors[i], marker=params.markers[i], markeredgewidth=params.marker_linewidths[i], linewidth=params.marker_linewidths[i], linestyle='None', markersize=10*params.marker_scales[i], label=str(i+1)))
    #legend_parts.append(mlines.Line2D([], [], color=params.colors[i], marker=params.markers[i], linestyle='None', markersize=0, label=''))
    legend_parts.append(mlines.Line2D([], [], color=params.colors[i], marker=params.markers[i], linestyle='None', markersize=0, label=''))
    for i in range (23,30):
        legend_parts.append(mlines.Line2D([], [], color=params.colors[i], marker=params.markers[i], markeredgewidth=params.marker_linewidths[i], linewidth=params.marker_linewidths[i], linestyle='None', markersize=10*params.marker_scales[i], label=str(i+1)))
    first_legend = ax_legend.legend(handles=legend_parts, bbox_to_anchor=(0.725,2.5), ncol=2, fontsize=12)

    plt.gcf().savefig("plots/E_J_phi_paper.eps", bbox_inches='tight',pad_inches=0.01, dpi=100, format='eps')
    plt.gcf().savefig("plots/E_J_phi_paper.png", bbox_inches='tight',pad_inches=0.01, dpi=100, format='png')
    plt.close()


def plot_rhombus_paper (Clusters, J_x, J_y):
    indeces = [c[i] for c in Clusters for i in range(len(c))]
    non_indeces = [i for i in range(len(Energy)) if not (i in indeces)]

    plt.clf()
    fig, axes = plt.subplots(nrows=2, ncols=2)

 
    axes[0,0].set_xlim (-1,1)
    axes[0,0].set_ylim (-1,1)
    axes[0,0].plot((0,1), (-1,0.), color="k", alpha=0.3)
    axes[0,0].plot((1,0), (0,1), color="k", alpha=0.3)
    axes[0,0].plot((0,-1), (1,0), color="k", alpha=0.3)
    axes[0,0].plot((-1,0), (0,-1), color="k", alpha=0.3)
    axes[0,0].spines['bottom'].set_color('white')
    axes[0,0].spines['top'].set_color('white') 
    axes[0,0].spines['right'].set_color('white')
    axes[0,0].spines['left'].set_color('white')
    axes[0,0].text(1.05,0.12,"prograde",rotation=270,fontsize=15)
    axes[0,0].text(-1.2,0.2,"retrograde",rotation=90,fontsize=15)
    axes[0,0].text(-0.2,-1.15,r"radial$\/(J_{\phi}=0)$",fontsize=15)
    axes[0,0].text(-0.25,1.1,r"polar$\/(J_{\phi}=0)$",fontsize=15)
    axes[0,0].text(0.3,-0.4,r"in plane$\/(J_z=0)$",rotation=45,fontsize=15)
    axes[0,0].text(-0.85,-0.4,r"in plane$\/(J_z=0)$",rotation=-45,fontsize=15)
    axes[0,0].text(-0.8,0.85,r"circular$\/(J_r=0)$",rotation=45,fontsize=15)
    axes[0,0].text(0.25,0.8,r"circular$\/(J_r=0)$",rotation=-45,fontsize=15)
    axes[0,0].get_xaxis().set_visible(False)
    axes[0,0].get_yaxis().set_visible(False)
    axes[0,0].grid(False)

    axes[0,0].scatter(J_x[non_indeces], J_y[non_indeces], color="grey", s=5)
    for i, c in enumerate(Clusters[0:7]):
        axes[0,0].scatter(J_x[c], J_y[c], c=params.colors[i], marker=params.markers[i], s=150*params.marker_scales[i], linewidth=params.marker_linewidths[i])

    legend_parts = []
    for i in range (0,8):
        legend_parts.append(mlines.Line2D([], [], color=params.colors[i], marker=params.markers[i], markeredgewidth=params.marker_linewidths[i], linewidth=params.marker_linewidths[i], linestyle='None', markersize=10*params.marker_scales[i], label='C'+str(i+1)))
    first_legend = plt.legend(handles=legend_parts, loc='center', bbox_to_anchor=(0.4,0.4),fontsize=12)


    axes[0,1].set_xlim (-1,1)
    axes[0,1].set_ylim (-1,1)
    axes[0,1].plot((0,1), (-1,0.), color="k", alpha=0.3)
    axes[0,1].plot((1,0), (0,1), color="k", alpha=0.3)
    axes[0,1].plot((0,-1), (1,0), color="k", alpha=0.3)
    axes[0,1].plot((-1,0), (0,-1), color="k", alpha=0.3)
    axes[0,1].spines['bottom'].set_color('white')
    axes[0,1].spines['top'].set_color('white') 
    axes[0,1].spines['right'].set_color('white')
    axes[0,1].spines['left'].set_color('white')
    axes[0,1].text(1.05,0.12,"prograde",rotation=270,fontsize=15)
    axes[0,1].text(-1.2,0.2,"retrograde",rotation=90,fontsize=15)
    axes[0,1].text(-0.2,-1.15,r"radial$\/(J_{\phi}=0)$",fontsize=15)
    axes[0,1].text(-0.25,1.1,r"polar$\/(J_{\phi}=0)$",fontsize=15)
    axes[0,1].text(0.3,-0.4,r"in plane$\/(J_z=0)$",rotation=45,fontsize=15)
    axes[0,1].text(-0.85,-0.4,r"in plane$\/(J_z=0)$",rotation=-45,fontsize=15)
    axes[0,1].text(-0.8,0.85,r"circular$\/(J_r=0)$",rotation=45,fontsize=15)
    axes[0,1].text(0.25,0.8,r"circular$\/(J_r=0)$",rotation=-45,fontsize=15)
    axes[0,1].get_xaxis().set_visible(False)
    axes[0,1].get_yaxis().set_visible(False)
    axes[0,1].grid(False)

    axes[0,1].scatter(J_x[non_indeces], J_y[non_indeces], color="grey", s=5)
    for i, c in enumerate(Clusters[15:23]):
        axes[0,1].scatter(J_x[c], J_y[c], c=params.colors[i+15], marker=params.markers[i+15], s=150*params.marker_scales[i+15], linewidth=params.marker_linewidths[i+15])


    axes[1,0].set_xlim (-1,1)
    axes[1,0].set_ylim (-1,1)
    axes[1,0].plot((0,1), (-1,0.), color="k", alpha=0.3)
    axes[1,0].plot((1,0), (0,1), color="k", alpha=0.3)
    axes[1,0].plot((0,-1), (1,0), color="k", alpha=0.3)
    axes[1,0].plot((-1,0), (0,-1), color="k", alpha=0.3)
    axes[1,0].spines['bottom'].set_color('white')
    axes[1,0].spines['top'].set_color('white') 
    axes[1,0].spines['right'].set_color('white')
    axes[1,0].spines['left'].set_color('white')
    axes[1,0].text(1.05,0.12,"prograde",rotation=270,fontsize=15)
    axes[1,0].text(-1.2,0.2,"retrograde",rotation=90,fontsize=15)
    axes[1,0].text(-0.2,-1.15,r"radial$\/(J_{\phi}=0)$",fontsize=15)
    axes[1,0].text(-0.25,1.1,r"polar$\/(J_{\phi}=0)$",fontsize=15)
    axes[1,0].text(0.3,-0.4,r"in plane$\/(J_z=0)$",rotation=45,fontsize=15)
    axes[1,0].text(-0.85,-0.4,r"in plane$\/(J_z=0)$",rotation=-45,fontsize=15)
    axes[1,0].text(-0.8,0.85,r"circular$\/(J_r=0)$",rotation=45,fontsize=15)
    axes[1,0].text(0.25,0.8,r"circular$\/(J_r=0)$",rotation=-45,fontsize=15)
    axes[1,0].get_xaxis().set_visible(False)
    axes[1,0].get_yaxis().set_visible(False)
    axes[1,0].grid(False)

    axes[1,0].scatter(J_x[non_indeces], J_y[non_indeces], color="grey", s=5)
    for i, c in enumerate(Clusters[8:15]):
        axes[1,0].scatter(J_x[c], J_y[c], c=params.colors[i+8], marker=params.markers[i+8], s=150*params.marker_scales[i+8], linewidth=params.marker_linewidths[i+8])


    axes[1,1].set_xlim (-1,1)
    axes[1,1].set_ylim (-1,1)
    axes[1,1].plot((0,1), (-1,0.), color="k", alpha=0.3)
    axes[1,1].plot((1,0), (0,1), color="k", alpha=0.3)
    axes[1,1].plot((0,-1), (1,0), color="k", alpha=0.3)
    axes[1,1].plot((-1,0), (0,-1), color="k", alpha=0.3)
    axes[1,1].spines['bottom'].set_color('white')
    axes[1,1].spines['top'].set_color('white') 
    axes[1,1].spines['right'].set_color('white')
    axes[1,1].spines['left'].set_color('white')
    axes[1,1].text(1.05,0.12,"prograde",rotation=270,fontsize=15)
    axes[1,1].text(-1.2,0.2,"retrograde",rotation=90,fontsize=15)
    axes[1,1].text(-0.2,-1.15,r"radial$\/(J_{\phi}=0)$",fontsize=15)
    axes[1,1].text(-0.25,1.1,r"polar$\/(J_{\phi}=0)$",fontsize=15)
    axes[1,1].text(0.3,-0.4,r"in plane$\/(J_z=0)$",rotation=45,fontsize=15)
    axes[1,1].text(-0.85,-0.4,r"in plane$\/(J_z=0)$",rotation=-45,fontsize=15)
    axes[1,1].text(-0.8,0.85,r"circular$\/(J_r=0)$",rotation=45,fontsize=15)
    axes[1,1].text(0.25,0.8,r"circular$\/(J_r=0)$",rotation=-45,fontsize=15)
    axes[1,1].get_xaxis().set_visible(False)
    axes[1,1].get_yaxis().set_visible(False)
    axes[1,1].grid(False)

    axes[1,1].scatter(J_x[non_indeces], J_y[non_indeces], color="grey", s=5)
    for i, c in enumerate(Clusters[23:30]):
        axes[1,1].scatter(J_x[c], J_y[c], c=params.colors[i+23], marker=params.markers[i+23], s=150*params.marker_scales[i+23], linewidth=params.marker_linewidths[i+23])


    plt.gcf().set_size_inches(12, 12)
    fig.tight_layout()
    plt.subplots_adjust(hspace = 0.3, wspace = 0.2)

    legend_parts = []
    for i in range (0,8):
        legend_parts.append(mlines.Line2D([], [], color=params.colors[i], marker=params.markers[i], markeredgewidth=params.marker_linewidths[i], linewidth=params.marker_linewidths[i], linestyle='None', markersize=10*params.marker_scales[i], label=str(i+1)))
    #legend_parts.append(mlines.Line2D([], [], color=params.colors[i], marker=params.markers[i], linestyle='None', markersize=0, label=''))
    legend_parts.append(mlines.Line2D([], [], color=params.colors[i], marker=params.markers[i], linestyle='None', markersize=0, label=''))
    for i in range (8,15):
        legend_parts.append(mlines.Line2D([], [], color=params.colors[i], marker=params.markers[i], markeredgewidth=params.marker_linewidths[i], linewidth=params.marker_linewidths[i], linestyle='None', markersize=10*params.marker_scales[i], label=str(i+1)))
    for i in range (15,23):
        legend_parts.append(mlines.Line2D([], [], color=params.colors[i], marker=params.markers[i], markeredgewidth=params.marker_linewidths[i], linewidth=params.marker_linewidths[i], linestyle='None', markersize=10*params.marker_scales[i], label=str(i+1)))
    #legend_parts.append(mlines.Line2D([], [], color=params.colors[i], marker=params.markers[i], linestyle='None', markersize=0, label=''))
    legend_parts.append(mlines.Line2D([], [], color=params.colors[i], marker=params.markers[i], linestyle='None', markersize=0, label=''))
    for i in range (23,30):
        legend_parts.append(mlines.Line2D([], [], color=params.colors[i], marker=params.markers[i], markeredgewidth=params.marker_linewidths[i], linewidth=params.marker_linewidths[i], linestyle='None', markersize=10*params.marker_scales[i], label=str(i+1)))
    first_legend = plt.legend(handles=legend_parts, bbox_to_anchor=(0.085,1.5), ncol=2, fontsize=12)

    plt.text(-1.2,1.23,"CDTG", ha='center', va='center', fontsize=14, color='black')

    plt.gcf().savefig("plots/rhombus_paper.eps", bbox_inches='tight',pad_inches=0.01, dpi=100, format='eps')
    plt.gcf().savefig("plots/rhombus_paper.png", bbox_inches='tight',pad_inches=0.01, dpi=100, format='png')
    plt.close()


def plot_rhombus_E_J_phi_paper (Clusters, J_x, J_y, J_phi, E, J_phi_lims=(-5,5), E_lims=(-5,0)):
    indeces = [c[i] for c in Clusters for i in range(len(c))]
    non_indeces = [i for i in range(len(Energy)) if not (i in indeces)]

    plt.clf()
    fig, axes = plt.subplots(nrows=1, ncols=2)

    axes[0].set_xlim (-1,1)
    axes[0].set_ylim (-1,1)
    axes[0].plot((0,1), (-1,0.), color="k", alpha=0.3)
    axes[0].plot((1,0), (0,1), color="k", alpha=0.3)
    axes[0].plot((0,-1), (1,0), color="k", alpha=0.3)
    axes[0].plot((-1,0), (0,-1), color="k", alpha=0.3)
    axes[0].spines['bottom'].set_color('white')
    axes[0].spines['top'].set_color('white') 
    axes[0].spines['right'].set_color('white')
    axes[0].spines['left'].set_color('white')
    axes[0].text(1.05,0.12,"prograde",rotation=270,fontsize=12)
    axes[0].text(-1.2,0.2,"retrograde",rotation=90,fontsize=12)
    axes[0].text(-0.2,-1.15,r"radial$\/(J_{\phi}=0)$",fontsize=12)
    axes[0].text(-0.25,1.1,r"polar$\/(J_{\phi}=0)$",fontsize=12)
    axes[0].text(0.3,-0.4,r"in plane$\/(J_z=0)$",rotation=45,fontsize=12)
    axes[0].text(-0.85,-0.4,r"in plane$\/(J_z=0)$",rotation=-45,fontsize=12)
    axes[0].text(-0.8,0.85,r"circular$\/(J_r=0)$",rotation=45,fontsize=12)
    axes[0].text(0.25,0.8,r"circular$\/(J_r=0)$",rotation=-45,fontsize=12)
    axes[0].get_xaxis().set_visible(False)
    axes[0].get_yaxis().set_visible(False)
    axes[0].grid(False)

    axes[0].scatter(J_x[non_indeces], J_y[non_indeces], color="grey", s=5)
    for i, c in enumerate(Clusters):
        axes[0].scatter(J_x[c], J_y[c], c=params.colors[i], marker=params.markers[i], s=150)

    axes[1].set_xlim (J_phi_lims[0]-0.025,J_phi_lims[1]+0.025)
    axes[1].set_ylim (E_lims[0]-0.025,E_lims[1]+0.025)
    axes[1].set_xlabel (r'$J_\phi$ ($10^3$ kpc km s$^{-1}$)',fontsize=12)
    axes[1].set_ylabel (r'Energy ($10^5$ km$^2$ s$^{-2}$)',fontsize=12)
    
    axes[1].scatter(J_phi[non_indeces]/1e3, E[non_indeces]/1e5, color="grey", s=5)
    for i, c in enumerate(Clusters):
        axes[1].scatter(J_phi[c]/1e3, E[c]/1e5, c=params.colors[i], marker=params.markers[i], s=100)

    legend_parts = []
    for i in range(len(Clusters)):
        legend_parts.append(mlines.Line2D([], [], color=params.colors[i], marker=params.markers[i], linestyle='None', markersize=10, label='C'+str(i+1)))
    legend = plt.legend(handles=legend_parts, loc='center', bbox_to_anchor=(-0.45,0.5),ncol=1,fontsize=12)

    plt.gcf().set_size_inches(12, 5)
    fig.tight_layout()
    plt.subplots_adjust(hspace = 0, wspace = 0.8)

    plt.gcf().savefig("plots/rhombus_E_J_phi_paper.eps", bbox_inches='tight',pad_inches=0.01, dpi=100, format='eps')
    plt.gcf().savefig("plots/rhombus_E_J_phi_paper.png", bbox_inches='tight',pad_inches=0.01, dpi=100, format='png')
    plt.close()


def plot_cluster_abundances_paper (Clusters, FeH, CFe, SrBa, BaEu, EuFe, Comments):
    indeces = [c[i] for c in Clusters for i in range(len(c))]
    non_indeces = [i for i in range(len(FeH)) if not (i in indeces)]

    plt.clf()
    fig, axes = plt.subplots(nrows=2, ncols=2)

    axes[0,0].set_ylabel("[Eu/Fe]")
    #axes[0,0].scatter(FeH[non_indeces], EuFe[non_indeces], c='grey', marker='o', s=7)
    for i, c in enumerate(Clusters):
        axes[0,0].scatter(FeH[c], EuFe[c], c=params.colors[i], marker=params.markers[i], s=100)

    axes[0,1].set_ylabel("[Ba/Eu]")
    #axes[0,1].scatter(FeH[non_indeces], BaEu[non_indeces], c='grey', marker='o', s=7)
    for i, c in enumerate(Clusters):
        final_FeH, final_BaEu = [], []
        for feh, baeu, C in zip(FeH[c], BaEu[c], Comments[c]):
            if not np.isnan(baeu) and not 'Ba' in C:
                final_FeH.append(feh)
                final_BaEu.append(baeu) 
        axes[0,1].scatter(final_FeH, final_BaEu, c=params.colors[i], marker=params.markers[i], s=100)

    axes[1,0].set_ylabel("[C/Fe]")
    axes[1,0].set_xlabel("[Fe/H]")
    #axes[1,0].scatter(FeH[non_indeces], CFe[non_indeces], c='grey', marker='o', s=7)
    for i, c in enumerate(Clusters):
        final_FeH, final_CFe = [], []
        for feh, cfe, C in zip(FeH[c], CFe[c], Comments[c]):
            if not np.isnan(cfe) and not 'C' in C:
                final_FeH.append(feh)
                final_CFe.append(cfe) 
        axes[1,0].scatter(final_FeH, final_CFe, c=params.colors[i], marker=params.markers[i], s=100)

    axes[1,1].set_ylabel("[Sr/Ba]")
    axes[1,1].set_xlabel("[Fe/H]")
    #axes[1,1].scatter(FeH[non_indeces], SrBa[non_indeces], c='grey', marker='o', s=7)
    for i, c in enumerate(Clusters):
        final_FeH, final_SrBa = [], []
        for feh, srba, C in zip(FeH[c], SrBa[c], Comments[c]):
            if not np.isnan(srba) and not 'Sr' in C:
                final_FeH.append(feh)
                final_SrBa.append(srba) 
        axes[1,1].scatter(final_FeH, final_SrBa, c=params.colors[i], marker=params.markers[i], s=100)

    plt.gcf().set_size_inches(6, 6)
    fig.tight_layout()
    plt.gcf().savefig("plots/cluster_abundances_paper.eps", bbox_inches='tight',pad_inches=0.01, dpi=100, format='eps')
    plt.gcf().savefig("plots/cluster_abundances_paper.png", bbox_inches='tight',pad_inches=0.01, dpi=100, format='png')
    plt.close()


def plot_coordinates (Clusters, l, b):
    indeces = [c[i] for c in Clusters for i in range(len(c))]
    non_indeces = [i for i in range(len(l)) if not (i in indeces)]

    plt.clf()
    plt.title("r-process clusters", size=24)
    plt.xlabel("l", size=24)
    plt.ylabel("b", size=24)
    plt.tick_params(labelsize=18)
    for i, c in enumerate(Clusters):
        plt.scatter(l[c], b[c], c=params.colors[i], marker=params.markers[i], s=200)
    plt.scatter(l[non_indeces], b[non_indeces], c='grey', marker='o', s=10)
    plt.gcf().set_size_inches(25.6, 14.4)
    plt.gcf().savefig("plots/coordinates_clusters.png", dpi=100)
    plt.close() 


def plot_distances_RV (Clusters, dist, RV):
    indeces = [c[i] for c in Clusters for i in range(len(c))]
    non_indeces = [i for i in range(len(RV)) if not (i in indeces)]

    plt.clf()
    plt.title("r-process clusters", size=24)
    plt.xlabel("dist", size=24)
    plt.ylabel("RV", size=24)
    plt.tick_params(labelsize=18)
    for i, c in enumerate(Clusters):
        plt.scatter(dist[c], RV[c], c=params.colors[i], marker=params.markers[i], s=200)
    plt.scatter(dist[non_indeces], RV[non_indeces], c='grey', marker='o', s=10)
    plt.gcf().set_size_inches(25.6, 14.4)
    plt.gcf().savefig("plots/dist_RV_clusters.png", dpi=100)
    plt.close() 


def plot_t_SNE (Clusters, Params, name, markersize = [10,250]):
    indeces = [c[i] for c in Clusters for i in range(len(c))]
    for i in range(len(Params)):
        Params[i] = (Params[i] - np.mean(Params[i]))/np.std(Params[i])
    non_indeces = [i for i in range(len(Params[0])) if not (i in indeces)]

    Params = np.transpose(Params)
    TSNE_coord_clusters = np.transpose(TSNE(n_components=2).fit_transform(Params))

    plt.clf()
    plt.title("r-process clusters", size=24)
    plt.xlabel ("t-SNE plot", size=24)
    plt.ylabel("", size=24)
    plt.tick_params(labelsize=0)
    for i, c in enumerate(Clusters):
        plt.scatter(TSNE_coord_clusters[0][c], TSNE_coord_clusters[1][c], c=params.colors[i], marker=params.markers[i], s=markersize[1])
    plt.scatter(TSNE_coord_clusters[0][non_indeces], TSNE_coord_clusters[1][non_indeces], c='grey', marker='o', s=markersize[0])
    plt.gcf().set_size_inches(25.6, 14.4)
    plt.gcf().savefig(name, dpi=100)
    plt.close() 


def plot_combined_classes (Clusters, Combined_Class):
    ratios = []
    for c in Clusters:
        comb = Combined_Class[c]
        ratios.append(len(np.where(comb=='r2')[0])/len(c))

    plt.clf()
    plt.title("r-process clusters", size=24)
    plt.xlabel ("Relative r2 content", size=24)
    plt.ylabel("Count", size=24)
    plt.tick_params(labelsize=18)
    plt.hist(ratios, color='black', linewidth=2, histtype='step', bins=20)
    plt.gcf().set_size_inches(25.6, 14.4)
    plt.gcf().savefig("plots/combined_classes.png", dpi=100)
    plt.close() 


def plot_Zhen_tests (v_phi, v_y, J_phi):
    plt.clf()
    fig, axes = plt.subplots(nrows=1, ncols=2)

    axes[0].set_xlabel(r"$v_\phi$")
    axes[0].set_ylabel(r"$J_\phi$")
    axes[0].scatter(v_phi, J_phi, c='black', s=5)
    axes[0].set_xlim(min(v_phi),max(v_phi))
    axes[0].set_ylim(min(J_phi),max(J_phi))

    axes[1].set_xlabel(r"$v_y$")
    axes[1].set_ylabel(r"$J_\phi$")
    axes[1].scatter(v_y, J_phi, c='black', s=5)
    axes[1].set_xlim(min(v_y),max(v_y))
    axes[1].set_ylim(min(v_y),max(v_y))

    plt.gcf().set_size_inches(11, 5)
    plt.gcf().savefig("plots/Zhen_tests.eps", bbox_inches='tight',pad_inches=0.01, dpi=100, format='eps')
    plt.gcf().savefig("plots/Zhen_tests.png", bbox_inches='tight',pad_inches=0.01, dpi=100, format='png')
    plt.close()


if __name__ == '__main__':
    init_file = params.init_file
    kin_file = params.kin_file
    Clusters = clusters.final_clusters
    #kmeans_clusters = clusters.kmeans_clusters
    #meanshift_clusters = clusters.meanshift_clusters
    #afftypropagation_clusters = clusters.afftypropagation_clusters
    #aggloclustering_euclidian_clusters = clusters.aggloclustering_euclidian_clusters

    Name = get_column(0, float, init_file)[:]
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
    RV = get_column(5, float, init_file)[:]
    Energy = get_column(21, float, kin_file)[:]
    Energy_err = get_column(22, float, kin_file)[:]
    J_r = get_column(11, float, params.kin_file)[:]
    J_r_err = get_column(12, float, params.kin_file)[:]
    J_phi = -get_column(13, float, params.kin_file)[:]
    J_phi_err = get_column(14, float, params.kin_file)[:]
    J_z = get_column(19, float, params.kin_file)[:]
    J_z_err = get_column(20, float, params.kin_file)[:]

    J_tot = np.fabs(J_phi)[:] + J_z[:] + J_r[:]
    J_x = J_phi[:] / J_tot[:]
    J_y = (J_z[:] - J_r[:]) / J_tot[:]

    r_apo = get_column(25, float, kin_file)[:]
    r_apo_err = get_column(26, float, kin_file)[:]
    dist = get_column(3, float, init_file)[:]
    Combined_Class = get_column(17, str, init_file)[:]
    T = get_column(15, float, init_file)[:]
    r_peri = get_column(23, float, kin_file)[:]
    r_peri_err = get_column(24, float, kin_file)[:]
    Z_max = get_column(27, float, kin_file)[:]
    Z_max_err = get_column(28, float, kin_file)[:]
    ecc = get_column(29, float, kin_file)[:]
    ecc_err = get_column(30, float, kin_file)[:]
    v_r = get_column(1, float, kin_file)[:]
    v_phi = -get_column(3, float, kin_file)[:]
    v_phi_err = -get_column(4, float, kin_file)[:]
    v_z = get_column(9, float, kin_file)[:]
    v_z_err = get_column(10, float, kin_file)[:]
    v_y = get_column(7, float, kin_file)[:]

    #plot_data("data_F_C.png", '[Fe/H]', '[C/Fe]', FeH, CFe)
    #plot_data("data_E_L_z.png", 'Energy', 'L_z', Energy, L_z)
    #plot_data("data_E_L_p.png", 'Energy', 'L_p', Energy, L_p)
    #plot_data("data_E_I_3.png", 'Energy', 'I_3', Energy, I_3)
    #plot_data("data_L_z_L_p.png", 'L_z', 'L_p', L_z, L_p)
    #plot_data("data_L_z_I_3.png", 'L_z', 'I_3', L_z, I_3)
    #plot_data("data_L_p_I_3.png", 'L_p', 'I_3', L_p, I_3)

    colors = params.colors
    markers = params.markers
    #plot_concatenated_clusters("clusters_F_C.png", Clusters, colors, markers, '[Fe/H]', '[C/Fe]', FeH, CFe, True)
    #plot_concatenated_clusters("clusters_F_AC.png", Clusters, colors, markers, '[Fe/H]', 'A(C)', FeH, A_C, True)
    #plot_concatenated_clusters("clusters_E_I_3.png", Clusters, colors, markers, 'Energy', 'I_3', Energy, I_3, False)
    #plot_concatenated_clusters("clusters_rapo_CFe.png", Clusters, colors, markers, 'r_apo', '[C/Fe]', r_apo, CFe, True)
    #plot_concatenated_clusters("clusters_rapo_EuFe.png", Clusters, colors, markers, 'r_apo', '[Eu/Fe]', r_apo, EuFe, True)
    #plot_concatenated_clusters("clusters_rapo_SrBa.png", Clusters, colors, markers, 'r_apo', '[Sr/Ba]', r_apo, SrBa, True)
    #plot_concatenated_clusters("clusters_rapo_BaEu.png", Clusters, colors, markers, 'r_apo', '[Ba/Eu]', r_apo, BaEu, True)

    #plot_concatenated_clusters("clusters_1_EuFe_FeH.png", kmeans_clusters, colors, markers, '[Fe/H]', '[Eu/Fe]', FeH, EuFe, True, Xlim=[-4,-1.2], Ylim=[0,2], markersize=[50, 300])
    #plot_concatenated_clusters("clusters_2_EuFe_FeH.png", meanshift_clusters, colors, markers, '[Fe/H]', '[Eu/Fe]', FeH, EuFe, True, Xlim=[-4,-1.2], Ylim=[0,2], markersize=[50, 300])
    #plot_concatenated_clusters("clusters_3_EuFe_FeH.png", afftypropagation_clusters, colors, markers, '[Fe/H]', '[Eu/Fe]', FeH, EuFe, True, Xlim=[-4,-1.2], Ylim=[0,2], markersize=[50, 300])
    #plot_concatenated_clusters("clusters_4_EuFe_FeH.png", aggloclustering_euclidian_clusters, colors, markers, '[Fe/H]', '[Eu/Fe]', FeH, EuFe, True, Xlim=[-4,-1.2], Ylim=[0,2], markersize=[50, 300])
    #plot_concatenated_clusters("clusters_all_EuFe_FeH.png", Clusters, colors, markers, '[Fe/H]', '[Eu/Fe]', FeH, EuFe, True, Xlim=[-4,-1.2], Ylim=[0,2], markersize=[50, 300])

    #plot_mean_metallicities_RV (Clusters, colors, FeH, RV)
    #plot_orbital_velocities (Clusters, vRg, vTg)
    #plot_vTg_rapo (Clusters, vTg, r_apo)
    #plot_coordinates (Clusters, l, b)
    #plot_distances_RV (Clusters, dist, RV)

    #plot_t_SNE (Clusters, [Energy, L_z, L_p, I_3], "plots/t_SNE.png", markersize=[50, 300])
    #plot_t_SNE (clusters.kmeans_clusters, [Energy, L_z, L_p, I_3], "plots/t_SNE_Kmeans.png")
    #plot_t_SNE (clusters.meanshift_clusters, [Energy, L_z, L_p, I_3], "plots/t_SNE_meanshift.png")
    #plot_t_SNE (clusters.afftypropagation_clusters, [Energy, L_z, L_p, I_3], "plots/t_SNE_afftypropagation.png")
    #plot_t_SNE (clusters.aggloclustering_euclidian_clusters, [Energy, L_z, L_p, I_3], "plots/t_SNE_aggloclustering_euclidian.png")
    #plot_t_SNE (clusters.aggloclustering_manhattan_clusters, [Energy, L_z, L_p, I_3], "plots/t_SNE_aggloclustering_manhattan.png")

    #plot_combined_classes (Clusters, Combined_Class)
    #plot_cluster_abundances_paper (Clusters, FeH, cCFe, SrBa, BaEu, EuFe, Comments)
    #plot_Eu_FeH_clusters_paper (Clusters, EuFe, A_Eu, FeH)
    #plot_concatenated_clusters_paper (kmeans_clusters, meanshift_clusters, afftypropagation_clusters, aggloclustering_euclidian_clusters, colors, markers, Energy, J_phi, "Jphi", [-2.5,2])
    #plot_concatenated_clusters_paper (kmeans_clusters, meanshift_clusters, afftypropagation_clusters, aggloclustering_euclidian_clusters, colors, markers, Energy, ecc, "ecc", [0,1.6])

    #plot_motion_integrals_paper (Clusters, Energy, ecc, J_phi, Energy_limits=(-2,-1.2), ecc_limits=(0,1), J_phi_limits=(-1,1.5))
    #plot_motion_integrals_Ian_paper (Clusters, Energy, J_r, J_z, J_phi, Energy_limits=(-2,-1.2), J_r_limits=(0,2), J_z_limits=(0,0.8), J_phi_limits=(-1,1.5)) 
    #plot_motion_integrals_Ian_paper_v2 (Clusters, Energy, J_r, J_z, J_phi, Energy_limits=(-2,-1.2), J_r_limits=(0,2), J_z_limits=(0,0.8), J_phi_limits=(-1,1.5)) 
    #plot_kinematics_paper (Clusters, colors, markers, FeH, Energy, J_phi, J_r, J_z, ecc, r_peri, r_apo, Z_max)
    #plot_kinematics_paper_v2 (colors, markers, FeH, Energy, Energy_err, J_phi, J_phi_err, J_r, J_r_err, J_z, J_z_err, ecc, ecc_err, r_peri, r_peri_err, r_apo, r_apo_err, Z_max, Z_max_err)
    #plot_distance_paper(dist)
    #plot_Eu_FeH_paper (EuFe, A_Eu, FeH, T)
    plot_rhombus_paper (Clusters, J_x, J_y)
    #plot_toomre_paper (v_phi, v_z, v_r)
    #plot_toomre_paper_v2 (v_phi, v_phi_err, v_z, v_z_err)
    #plot_toomre_EuFe_paper (v_phi, v_z, EuFe)
    #plot_toomre_paper_clusters (Clusters, v_phi, v_z)
    plot_E_J_phi_paper (Clusters, J_phi, J_phi_err, Energy, Energy_err, J_phi_lims=(-2,2), E_lims=(-2.3,-0.8))
    #plot_Zhen_tests (v_phi, v_y, J_phi)
    #plot_rhombus_E_J_phi_paper (Clusters, J_x, J_y, J_phi, Energy, J_phi_lims=(-2,2), E_lims=(-2.3,-0.8))
