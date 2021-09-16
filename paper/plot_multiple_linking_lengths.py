from __future__ import division
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import hdbscan
from astropy.stats import biweight_scale, biweight_location
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import params
import clusters
from get_binom_statistics import get_combined_cumul, get_tot_prob, get_prob


minimum_cluster_size = 3
minimum_samples = 3
#cluster_selection_epsilon = 0.25


cluster_selection_epsilon_arr = np.arange(0,0.34+0.01,0.01)
table_file = 'table4.csv'
report_frequency = 1
values = [0.5, 0.33, 0.25]

def get_column(N, Type, filename):
    return np.genfromtxt(filename, delimiter=',', dtype=Type, skip_header=1, usecols=N, comments=None)

def get_column_rows(N, rows, Type, filename):
    return np.genfromtxt(filename, delimiter=',', dtype=Type, skip_header=1, usecols=N, comments=None, max_rows=rows)


def clustering (e, jphi, ecc, minimum_samples, cluster_selection_epsilon):
    data = np.array([[a,b,c] for a, b, c in zip(e, jphi, ecc)])
    print data
    clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True, gen_min_span_tree=False, min_cluster_size=minimum_cluster_size, min_samples=minimum_samples, cluster_selection_epsilon=cluster_selection_epsilon, cluster_selection_method='leaf', prediction_data=True)
    clusterer.fit(data)
    groups = []
    N_groups = max(clusterer.labels_)
    for i in range(N_groups):
        groups.append(list(np.where(clusterer.labels_==i)[0]))
    return groups


def form_clusters_file (e, jphi, ecc, minimum_samples, cluster_selection_epsilon):
    clusters = clustering(e, jphi, ecc, minimum_samples, cluster_selection_epsilon)
    cut_clusters = [c for c in clusters if len(c)>2]
    if os.path.isfile(params.clusters_file):
        os.remove(params.clusters_file)
    with open(params.clusters_file,'w') as clusters_file:
        clusters_file.write("from __future__ import division\n")
        clusters_file.write("import numpy as np\n")
        clusters_file.write("\n")
        clusters_file.write("\n")
        clusters_file.write("all_clusters = " + str(list(clusters)) + "\n")
        clusters_file.write("\n")
        clusters_file.write("final_clusters = " + str(list(cut_clusters)) + "\n")


def plot_stuff (FeH_frac_05, FeH_frac_033, FeH_frac_025, CFe_frac_05, CFe_frac_033, CFe_frac_025, SrFe_frac_05, SrFe_frac_033, SrFe_frac_025, BaFe_frac_05, BaFe_frac_033, BaFe_frac_025, EuFe_frac_05, EuFe_frac_033, EuFe_frac_025, number, c=['red','blue','yellow','black'], linewidth=1):
    plt.clf()
    fig, axes = plt.subplots(nrows=6,ncols=1)

    axes[0].set_ylabel("Number of clusters")
    axes[0].set_xlim(min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr))
    axes[0].set_ylim(0,max(number))
    axes[0].plot(cluster_selection_epsilon_arr,number,c=c[3],linewidth=linewidth)

    axes[1].set_ylabel("CV for [Fe/H]")
    axes[1].set_xlim(min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr))
    axes[1].set_ylim(0,1)
    axes[1].plot([min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr)],[0.5,0.5],'--',color=c[0],linewidth=0.5)
    axes[1].plot([min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr)],[0.3333333,0.3333333],'--',color=c[1],linewidth=0.5)
    axes[1].plot([min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr)],[0.25,0.25],'--',color=c[2],linewidth=0.5)
    axes[1].plot(cluster_selection_epsilon_arr,FeH_frac_05,c=c[0],linewidth=linewidth)
    axes[1].plot(cluster_selection_epsilon_arr,FeH_frac_033,c=c[1],linewidth=linewidth)
    axes[1].plot(cluster_selection_epsilon_arr,FeH_frac_025,c=c[2],linewidth=linewidth)

    axes[2].set_ylabel("CV for [C/Fe]c")
    axes[2].set_xlim(min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr))
    axes[2].set_ylim(0,1)
    axes[2].plot([min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr)],[0.5,0.5],'--',color=c[0],linewidth=0.5)
    axes[2].plot([min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr)],[0.3333333,0.3333333],'--',color=c[1],linewidth=0.5)
    axes[2].plot([min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr)],[0.25,0.25],'--',color=c[2],linewidth=0.5)
    axes[2].plot(cluster_selection_epsilon_arr,CFe_frac_05,c=c[0],linewidth=linewidth)
    axes[2].plot(cluster_selection_epsilon_arr,CFe_frac_033,c=c[1],linewidth=linewidth)
    axes[2].plot(cluster_selection_epsilon_arr,CFe_frac_025,c=c[2],linewidth=linewidth)

    axes[3].set_ylabel("CV for [Sr/Fe]")
    axes[3].set_xlim(min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr))
    axes[3].set_ylim(0,1)
    axes[3].plot([min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr)],[0.5,0.5],'--',color=c[0],linewidth=0.5)
    axes[3].plot([min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr)],[0.3333333,0.3333333],'--',color=c[1],linewidth=0.5)
    axes[3].plot([min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr)],[0.25,0.25],'--',color=c[2],linewidth=0.5)
    axes[3].plot(cluster_selection_epsilon_arr,SrFe_frac_05,c=c[0],linewidth=linewidth)
    axes[3].plot(cluster_selection_epsilon_arr,SrFe_frac_033,c=c[1],linewidth=linewidth)
    axes[3].plot(cluster_selection_epsilon_arr,SrFe_frac_025,c=c[2],linewidth=linewidth)

    axes[4].set_ylabel("CV for [Ba/Fe]")
    axes[4].set_xlim(min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr))
    axes[4].set_ylim(0,1)
    axes[4].plot([min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr)],[0.5,0.5],'--',color=c[0],linewidth=0.5)
    axes[4].plot([min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr)],[0.3333333,0.3333333],'--',color=c[1],linewidth=0.5)
    axes[4].plot([min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr)],[0.25,0.25],'--',color=c[2],linewidth=0.5)
    axes[4].plot(cluster_selection_epsilon_arr,BaFe_frac_05,c=c[0],linewidth=linewidth)
    axes[4].plot(cluster_selection_epsilon_arr,BaFe_frac_033,c=c[1],linewidth=linewidth)
    axes[4].plot(cluster_selection_epsilon_arr,BaFe_frac_025,c=c[2],linewidth=linewidth)

    axes[5].set_xlabel("Cluster selection epsilon")
    axes[5].set_ylabel("CV for [Eu/Fe]")
    axes[5].set_xlim(min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr))
    axes[5].set_ylim(0,1)
    axes[5].plot([min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr)],[0.5,0.5],'--',color=c[0],linewidth=0.5)
    axes[5].plot([min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr)],[0.3333333,0.3333333],'--',color=c[1],linewidth=0.5)
    axes[5].plot([min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr)],[0.25,0.25],'--',color=c[2],linewidth=0.5)
    axes[5].plot(cluster_selection_epsilon_arr,EuFe_frac_05,c=c[0],linewidth=linewidth)
    axes[5].plot(cluster_selection_epsilon_arr,EuFe_frac_033,c=c[1],linewidth=linewidth)
    axes[5].plot(cluster_selection_epsilon_arr,EuFe_frac_025,c=c[2],linewidth=linewidth)

    filename_eps = 'plots/multiple_linking_lengths.eps'
    filename_png = 'plots/multiple_linking_lengths.png'
    plt.gcf().set_size_inches(4,18)
    plt.subplots_adjust(wspace=0.3)
    plt.gcf().savefig(filename_eps, bbox_inches='tight', pad_inches=-0.002, dpi=100, format='eps')
    plt.gcf().savefig(filename_png, bbox_inches='tight', pad_inches=-0.002, dpi=100, format='png')
    plt.close()


def plot_linking_length_N (number, avg_N, c='black', linewidth=1):
    plt.clf()
    fig, axes = plt.subplots(nrows=1,ncols=2)

    axes[0].set_xlabel("Cluster selection epsilon")
    axes[0].set_ylabel("Number of CDTGs")
    axes[0].set_xlim(min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr))
    axes[0].set_ylim(0,max(number))
    axes[0].plot(cluster_selection_epsilon_arr,number,c=c,linewidth=linewidth)

    axes[1].set_xlabel("Cluster selection epsilon")
    axes[1].set_ylabel("Average Member Stars per CDTG")
    axes[1].set_xlim(min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr))
    axes[1].set_ylim(3,max(avg_N))
    axes[1].plot(cluster_selection_epsilon_arr,avg_N,c=c,linewidth=linewidth)

    filename_eps = 'plots/multiple_linking_lengths_N.eps'
    filename_png = 'plots/multiple_linking_lengths_N.png'
    plt.subplots_adjust(wspace=0.3)
    plt.gcf().set_size_inches(10,3)
    plt.gcf().savefig(filename_eps, bbox_inches='tight', pad_inches=-0.002, dpi=100, format='eps')
    plt.gcf().savefig(filename_png, bbox_inches='tight', pad_inches=-0.002, dpi=100, format='png')


def plot_linking_length_probs (FeH_num_05, FeH_num_033, FeH_num_025, FeH_num, CFe_num_05, CFe_num_033, CFe_num_025, CFe_num, SrFe_num_05, SrFe_num_033, SrFe_num_025, SrFe_num, BaFe_num_05, BaFe_num_033, BaFe_num_025, BaFe_num, EuFe_num_05, EuFe_num_033, EuFe_num_025, EuFe_num, colors=['red','blue','orange', '#228B22', '#FF1493', 'black'], linewidth=1):
    plt.clf()

    fig, axes = plt.subplots(nrows=2,ncols=1)

    FeH_prob, CFe_prob, SrFe_prob, BaFe_prob, EuFe_prob, tot_prob = [], [], [], [], [], []
    print "Calculating probabilities..."
    for i, (f_05, f_033, f_025, f_n, c_05, c_033, c_025, c_n, s_05, s_033, s_025, s_n, b_05, b_033, b_025, b_n, e_05, e_033, e_025, e_n) in enumerate(zip(FeH_num_05, FeH_num_033, FeH_num_025, FeH_num, CFe_num_05, CFe_num_033, CFe_num_025, CFe_num, SrFe_num_05, SrFe_num_033, SrFe_num_025, SrFe_num, BaFe_num_05, BaFe_num_033, BaFe_num_025, BaFe_num, EuFe_num_05, EuFe_num_033, EuFe_num_025, EuFe_num)):
        if (i+1) % report_frequency == 0:
            print i+1, "/", len(cluster_selection_epsilon_arr)
        FeH_prob.append(get_combined_cumul([f_05, f_033, f_025], f_n, values))
        CFe_prob.append(get_combined_cumul([c_05, c_033, c_025], c_n, values))
        SrFe_prob.append(get_combined_cumul([s_05, s_033, s_025], s_n, values))
        BaFe_prob.append(get_combined_cumul([b_05, b_033, b_025], b_n, values))
        EuFe_prob.append(get_combined_cumul([e_05, e_033, e_025], e_n, values))
    print "Calculating total probabilities..."
    for i, (f, c, s, b, e) in enumerate(zip(FeH_prob, CFe_prob, SrFe_prob, BaFe_prob, EuFe_prob)):
        if (i+1) % report_frequency == 0:
            print i+1, "/", len(cluster_selection_epsilon_arr)
        tot_prob.append(get_combined_cumul(np.array([5, 4, 3, 2, 1]), 5, np.array([f, c, s, b, e])))

    axes[0].set_xlabel("Cluster selection epsilon")
    axes[0].set_ylabel(r"GEAD probabilities ($\%$)")
    axes[0].set_xlim(min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr))
    axes[0].set_ylim(0,100)
    axes[0].plot(cluster_selection_epsilon_arr, np.array(FeH_prob)*100, colors[0], linewidth=linewidth)
    axes[0].plot(cluster_selection_epsilon_arr, np.array(CFe_prob)*100, colors[1], linewidth=linewidth)
    axes[0].plot(cluster_selection_epsilon_arr, np.array(SrFe_prob)*100, colors[2], linewidth=linewidth)
    axes[0].plot(cluster_selection_epsilon_arr, np.array(BaFe_prob)*100, colors[3], linewidth=linewidth)
    axes[0].plot(cluster_selection_epsilon_arr, np.array(EuFe_prob)*100, colors[4], linewidth=linewidth)
 
    legend_parts = []
    legend_parts.append(mlines.Line2D([], [], color=colors[0], markersize=10, label='[Fe/H]'))
    legend_parts.append(mlines.Line2D([], [], color=colors[1], markersize=10, label=r'[C/Fe]$_\mathrm{c}$'))
    legend_parts.append(mlines.Line2D([], [], color=colors[2], markersize=10, label='[Sr/Fe]'))
    legend_parts.append(mlines.Line2D([], [], color=colors[3], markersize=10, label='[Ba/Fe]'))
    legend_parts.append(mlines.Line2D([], [], color=colors[4], markersize=10, label='[Eu/Fe]'))
    legend = axes[0].legend(handles=legend_parts, bbox_to_anchor=(0.10,0.7), loc=3, ncol=2, fontsize=10)

    axes[1].set_xlabel("Cluster selection epsilon")
    axes[1].set_ylabel(r"OEAD probability ($\log_{10}$)")
    axes[1].set_xlim(min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr))
    axes[1].set_ylim(int(min(np.log10(np.array(tot_prob))))-1,0)
    axes[1].plot(cluster_selection_epsilon_arr, np.log10(np.array(tot_prob)), colors[5], linewidth=linewidth)

    filename_eps = 'plots/multiple_linking_lengths_probs.eps'
    filename_png = 'plots/multiple_linking_lengths_probs.png'
    plt.gcf().set_size_inches(5,10)
    plt.subplots_adjust(hspace=0.2)
    plt.gcf().savefig(filename_eps, bbox_inches='tight', pad_inches=-0.002, dpi=100, format='eps')
    plt.gcf().savefig(filename_png, bbox_inches='tight', pad_inches=-0.002, dpi=100, format='png')
    plt.close()


def plot_linking_length_split (FeH_num_05, FeH_num_033, FeH_num_025, FeH_num, CFe_num_05, CFe_num_033, CFe_num_025, CFe_num, SrFe_num_05, SrFe_num_033, SrFe_num_025, SrFe_num, BaFe_num_05, BaFe_num_033, BaFe_num_025, BaFe_num, EuFe_num_05, EuFe_num_033, EuFe_num_025, EuFe_num, colors=['red','blue','black'], linewidth=1):
    plt.clf()

    FeH_prob, CFe_prob, SrFe_prob, BaFe_prob, EuFe_prob, tot_prob_env, tot_prob_prog, tot_prob = [], [], [], [], [], [], [], []
    print "Calculating probabilities..."
    for i, (f_05, f_033, f_025, f_n, c_05, c_033, c_025, c_n, s_05, s_033, s_025, s_n, b_05, b_033, b_025, b_n, e_05, e_033, e_025, e_n) in enumerate(zip(FeH_num_05, FeH_num_033, FeH_num_025, FeH_num, CFe_num_05, CFe_num_033, CFe_num_025, CFe_num, SrFe_num_05, SrFe_num_033, SrFe_num_025, SrFe_num, BaFe_num_05, BaFe_num_033, BaFe_num_025, BaFe_num, EuFe_num_05, EuFe_num_033, EuFe_num_025, EuFe_num)):
        if (i+1) % report_frequency == 0:
            print i+1, "/", len(cluster_selection_epsilon_arr)
        FeH_prob.append(get_combined_cumul([f_05, f_033, f_025], f_n, values))
        CFe_prob.append(get_combined_cumul([c_05, c_033, c_025], c_n, values))
        SrFe_prob.append(get_combined_cumul([s_05, s_033, s_025], s_n, values))
        BaFe_prob.append(get_combined_cumul([b_05, b_033, b_025], b_n, values))
        EuFe_prob.append(get_combined_cumul([e_05, e_033, e_025], e_n, values))
    print "Calculating environmental total probabilities..."
    for i, (f, c) in enumerate(zip(FeH_prob, CFe_prob)):
        if (i+1) % report_frequency == 0:
            print i+1, "/", len(cluster_selection_epsilon_arr)
        tot_prob_env.append(get_combined_cumul(np.array([2, 1]), 2, np.array([f, c])))
    print "Calculating progenitors total probabilities..."
    for i, (s, b, e) in enumerate(zip(SrFe_prob, BaFe_prob, EuFe_prob)):
        if (i+1) % report_frequency == 0:
            print i+1, "/", len(cluster_selection_epsilon_arr)
        tot_prob_prog.append(get_combined_cumul(np.array([3, 2, 1]), 3, np.array([s, b, e])))
    print "Calculating total probabilities..."
    for i, (f, c, s, b, e) in enumerate(zip(FeH_prob, CFe_prob, SrFe_prob, BaFe_prob, EuFe_prob)):
        if (i+1) % report_frequency == 0:
            print i+1, "/", len(cluster_selection_epsilon_arr)
        tot_prob.append(get_combined_cumul(np.array([5, 4, 3, 2, 1]), 5, np.array([f, c, s, b, e])))

    plt.xlabel("Cluster selection epsilon")
    plt.ylabel(r"OEAD probabilities ($\%$)")
    plt.xlim(min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr))
    #plt.ylim(0,100)
    plt.plot(cluster_selection_epsilon_arr, np.log10(np.array(tot_prob_env)), colors[0], linewidth=linewidth)
    plt.plot(cluster_selection_epsilon_arr, np.log10(np.array(tot_prob_prog)), colors[1], linewidth=linewidth)
    plt.plot(cluster_selection_epsilon_arr, np.log10(np.array(tot_prob)), colors[2], linewidth=linewidth)
 
    legend_parts = []
    legend_parts.append(mlines.Line2D([], [], color=colors[0], markersize=10, label='Environmental OEAD probability'))
    legend_parts.append(mlines.Line2D([], [], color=colors[1], markersize=10, label='Progenitor OEAD probability'))
    legend_parts.append(mlines.Line2D([], [], color=colors[2], markersize=10, label='Overall OEAD probability'))
    legend = plt.legend(handles=legend_parts, bbox_to_anchor=(0.10,0.7), loc=3, ncol=2, fontsize=10)

    filename_eps = 'plots/multiple_linking_lengths_split.eps'
    filename_png = 'plots/multiple_linking_lengths_split.png'
    plt.gcf().set_size_inches(10,6)
    plt.subplots_adjust(hspace=0.2)
    plt.gcf().savefig(filename_eps, bbox_inches='tight', pad_inches=-0.002, dpi=100, format='eps')
    plt.gcf().savefig(filename_png, bbox_inches='tight', pad_inches=-0.002, dpi=100, format='png')
    plt.close()


def plot_linking_length_fracs (FeH_frac_05, FeH_frac_033, FeH_frac_025, CFe_frac_05, CFe_frac_033, CFe_frac_025, SrFe_frac_05, SrFe_frac_033, SrFe_frac_025, BaFe_frac_05, BaFe_frac_033, BaFe_frac_025, EuFe_frac_05, EuFe_frac_033, EuFe_frac_025, c=['red','blue','yellow'], linewidth=1):
    plt.clf()

    text_loc_x = 0.25
    text_loc_y = 0.8

    gs = gridspec.GridSpec(2, 6)
    ax00 = plt.subplot(gs[0, 0:1])
    ax01 = plt.subplot(gs[0, 1:3])
    ax02 = plt.subplot(gs[0, 3:5])
    ax03 = plt.subplot(gs[0,5:6])
    ax10 = plt.subplot(gs[1, 0:2])
    ax11 = plt.subplot(gs[1, 2:4])
    ax12 = plt.subplot(gs[1, 4:6]) 

    ax00.axis('off')
    ax03.axis('off') 

    ax01.set_xlabel("Cluster selection epsilon")
    ax01.set_ylabel("CDTG CDF Fractions ([Fe/H])")
    ax01.set_xlim(min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr))
    ax01.set_ylim(0,1)
    ax01.plot([min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr)],[0.5,0.5],'--',color=c[0],linewidth=0.5)
    ax01.plot([min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr)],[0.3333333,0.3333333],'--',color=c[1],linewidth=0.5)
    ax01.plot([min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr)],[0.25,0.25],'--',color=c[2],linewidth=0.5)
    ax01.plot(cluster_selection_epsilon_arr,FeH_frac_05,c=c[0],linewidth=linewidth)
    ax01.plot(cluster_selection_epsilon_arr,FeH_frac_033,c=c[1],linewidth=linewidth)
    ax01.plot(cluster_selection_epsilon_arr,FeH_frac_025,c=c[2],linewidth=linewidth)
    ax01.text(text_loc_x, text_loc_y, "[Fe/H]", ha='right', va='bottom', fontsize=13)

    ax02.set_xlabel("Cluster selection epsilon")
    ax02.set_ylabel(r"CDTG CDF Fractions ([C/Fe]$_\mathrm{c}$)")
    ax02.set_xlim(min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr))
    ax02.set_ylim(0,1)
    ax02.plot([min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr)],[0.5,0.5],'--',color=c[0],linewidth=0.5)
    ax02.plot([min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr)],[0.3333333,0.3333333],'--',color=c[1],linewidth=0.5)
    ax02.plot([min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr)],[0.25,0.25],'--',color=c[2],linewidth=0.5)
    ax02.plot(cluster_selection_epsilon_arr,CFe_frac_05,c=c[0],linewidth=linewidth)
    ax02.plot(cluster_selection_epsilon_arr,CFe_frac_033,c=c[1],linewidth=linewidth)
    ax02.plot(cluster_selection_epsilon_arr,CFe_frac_025,c=c[2],linewidth=linewidth)
    ax02.text(text_loc_x, text_loc_y, r"[C/Fe]$_\mathrm{c}$", ha='right', va='bottom', fontsize=13)

    ax10.set_xlabel("Cluster selection epsilon")
    ax10.set_ylabel("CDTG CDF Fractions ([Sr/Fe])")
    ax10.set_xlim(min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr))
    ax10.set_ylim(0,1)
    ax10.plot([min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr)],[0.5,0.5],'--',color=c[0],linewidth=0.5)
    ax10.plot([min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr)],[0.3333333,0.3333333],'--',color=c[1],linewidth=0.5)
    ax10.plot([min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr)],[0.25,0.25],'--',color=c[2],linewidth=0.5)
    ax10.plot(cluster_selection_epsilon_arr,SrFe_frac_05,c=c[0],linewidth=linewidth)
    ax10.plot(cluster_selection_epsilon_arr,SrFe_frac_033,c=c[1],linewidth=linewidth)
    ax10.plot(cluster_selection_epsilon_arr,SrFe_frac_025,c=c[2],linewidth=linewidth)
    ax10.text(text_loc_x, text_loc_y, "[Sr/Fe]", ha='right', va='bottom', fontsize=13)

    ax11.set_xlabel("Cluster selection epsilon")
    ax11.set_ylabel("CDTG CDF Fractions ([Ba/Fe])")
    ax11.set_xlim(min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr))
    ax11.set_ylim(0,1)
    ax11.plot([min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr)],[0.5,0.5],'--',color=c[0],linewidth=0.5)
    ax11.plot([min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr)],[0.3333333,0.3333333],'--',color=c[1],linewidth=0.5)
    ax11.plot([min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr)],[0.25,0.25],'--',color=c[2],linewidth=0.5)
    ax11.plot(cluster_selection_epsilon_arr,BaFe_frac_05,c=c[0],linewidth=linewidth)
    ax11.plot(cluster_selection_epsilon_arr,BaFe_frac_033,c=c[1],linewidth=linewidth)
    ax11.plot(cluster_selection_epsilon_arr,BaFe_frac_025,c=c[2],linewidth=linewidth)
    ax11.text(text_loc_x, text_loc_y, "[Ba/Fe]", ha='right', va='bottom', fontsize=13)

    ax12.set_xlabel("Cluster selection epsilon")
    ax12.set_ylabel("CDTG CDF Fractions ([Eu/Fe])")
    ax12.set_xlim(min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr))
    ax12.set_ylim(0,1)
    ax12.plot([min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr)],[0.5,0.5],'--',color=c[0],linewidth=0.5)
    ax12.plot([min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr)],[0.3333333,0.3333333],'--',color=c[1],linewidth=0.5)
    ax12.plot([min(cluster_selection_epsilon_arr),max(cluster_selection_epsilon_arr)],[0.25,0.25],'--',color=c[2],linewidth=0.5)
    ax12.plot(cluster_selection_epsilon_arr,EuFe_frac_05,c=c[0],linewidth=linewidth)
    ax12.plot(cluster_selection_epsilon_arr,EuFe_frac_033,c=c[1],linewidth=linewidth)
    ax12.plot(cluster_selection_epsilon_arr,EuFe_frac_025,c=c[2],linewidth=linewidth)
    ax12.text(text_loc_x, text_loc_y, "[Eu/Fe]", ha='right', va='bottom', fontsize=13)

    filename_eps = 'plots/multiple_linking_lengths_fracs.eps'
    filename_png = 'plots/multiple_linking_lengths_fracs.png'
    plt.gcf().set_size_inches(15,10)
    plt.subplots_adjust(wspace=0.7,hspace=0.3)
    plt.gcf().savefig(filename_eps, bbox_inches='tight', pad_inches=-0.002, dpi=100, format='eps')
    plt.gcf().savefig(filename_png, bbox_inches='tight', pad_inches=-0.002, dpi=100, format='png')
    plt.close()


if __name__ == '__main__':
    Name = get_column(0, str, params.init_file)[:]
   
    E = get_column(21, float, params.kin_file)[:]
    J_phi = get_column(13, float, params.kin_file)[:]
    Ecc = get_column(29, float, params.kin_file)[:]

    e = (E - biweight_location(E))/biweight_scale(E)
    jphi = (J_phi - biweight_location(J_phi))/biweight_scale(J_phi)
    ecc = (Ecc - biweight_location(Ecc))/biweight_scale(Ecc)

    FeH_frac_05, FeH_frac_033, FeH_frac_025, CFe_frac_05, CFe_frac_033, CFe_frac_025, SrFe_frac_05, SrFe_frac_033, SrFe_frac_025, BaFe_frac_05, BaFe_frac_033, BaFe_frac_025, EuFe_frac_05, EuFe_frac_033, EuFe_frac_025, number, avg_N = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    FeH_num_05, FeH_num_033, FeH_num_025, FeH_num, CFe_num_05, CFe_num_033, CFe_num_025, CFe_num, SrFe_num_05, SrFe_num_033, SrFe_num_025, SrFe_num, BaFe_num_05, BaFe_num_033, BaFe_num_025, BaFe_num, EuFe_num_05, EuFe_num_033, EuFe_num_025, EuFe_num = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    for i, c_s_e in enumerate(cluster_selection_epsilon_arr):

        form_clusters_file (e, jphi, ecc, minimum_samples, float(c_s_e))
        os.system('python form_table_4.py')
        fracs_05 = get_column_rows(1, 5, float, table_file)[:]
        fracs_033 = get_column_rows(2, 5, float, table_file)[:]
        fracs_025 = get_column_rows(3, 5, float, table_file)[:]
        nums_05 = get_column_rows(4, 5, float, table_file)[:]
        nums_033 = get_column_rows(5, 5, float, table_file)[:]
        nums_025 = get_column_rows(6, 5, float, table_file)[:]
        nums = get_column_rows(7, 5, int, table_file)[:]
        CFe_frac_05.append(fracs_05[0])
        SrFe_frac_05.append(fracs_05[1])
        BaFe_frac_05.append(fracs_05[2])
        EuFe_frac_05.append(fracs_05[3])
        FeH_frac_05.append(fracs_05[4])
        CFe_frac_033.append(fracs_033[0])
        SrFe_frac_033.append(fracs_033[1])
        BaFe_frac_033.append(fracs_033[2])
        EuFe_frac_033.append(fracs_033[3])
        FeH_frac_033.append(fracs_033[4])
        CFe_frac_025.append(fracs_025[0])
        SrFe_frac_025.append(fracs_025[1])
        BaFe_frac_025.append(fracs_025[2])
        EuFe_frac_025.append(fracs_025[3])
        FeH_frac_025.append(fracs_025[4])
        CFe_num_05.append(nums_05[0])
        SrFe_num_05.append(nums_05[1])
        BaFe_num_05.append(nums_05[2])
        EuFe_num_05.append(nums_05[3])
        FeH_num_05.append(nums_05[4])
        CFe_num_033.append(nums_033[0])
        SrFe_num_033.append(nums_033[1])
        BaFe_num_033.append(nums_033[2])
        EuFe_num_033.append(nums_033[3])
        FeH_num_033.append(nums_033[4])
        CFe_num_025.append(nums_025[0])
        SrFe_num_025.append(nums_025[1])
        BaFe_num_025.append(nums_025[2])
        EuFe_num_025.append(nums_025[3])
        FeH_num_025.append(nums_025[4])
        CFe_num.append(nums[0])
        SrFe_num.append(nums[1])
        BaFe_num.append(nums[2])
        EuFe_num.append(nums[3])
        FeH_num.append(nums[4])
        number.append(int(get_column_rows(8, 1, int, table_file)))
        reload(clusters)
        Clusters = clusters.final_clusters
        avg_N.append(np.mean([len(c) for c in Clusters]))
        if (i+1) % report_frequency == 0:
            print i+1, "/", len(cluster_selection_epsilon_arr)

    FeH_frac_05, FeH_frac_033, FeH_frac_025, CFe_frac_05, CFe_frac_033, CFe_frac_025, SrFe_frac_05, SrFe_frac_033, SrFe_frac_025, BaFe_frac_05, BaFe_frac_033, BaFe_frac_025, EuFe_frac_05, EuFe_frac_033, EuFe_frac_025, number = np.array(FeH_frac_05), np.array(FeH_frac_033), np.array(FeH_frac_025), np.array(CFe_frac_05), np.array(CFe_frac_033), np.array(CFe_frac_025), np.array(SrFe_frac_05), np.array(SrFe_frac_033), np.array(SrFe_frac_025), np.array(BaFe_frac_05), np.array(BaFe_frac_033), np.array(BaFe_frac_025), np.array(EuFe_frac_05), np.array(EuFe_frac_033), np.array(EuFe_frac_025), np.array(number)

    plot_linking_length_N (number, avg_N, c='black', linewidth=1) 
    plot_linking_length_fracs (FeH_frac_05, FeH_frac_033, FeH_frac_025, CFe_frac_05, CFe_frac_033, CFe_frac_025, SrFe_frac_05, SrFe_frac_033, SrFe_frac_025, BaFe_frac_05, BaFe_frac_033, BaFe_frac_025, EuFe_frac_05, EuFe_frac_033, EuFe_frac_025, c=['red','blue','orange'], linewidth=1)
    plot_linking_length_probs (FeH_num_05, FeH_num_033, FeH_num_025, FeH_num, CFe_num_05, CFe_num_033, CFe_num_025, CFe_num, SrFe_num_05, SrFe_num_033, SrFe_num_025, SrFe_num, BaFe_num_05, BaFe_num_033, BaFe_num_025, BaFe_num, EuFe_num_05, EuFe_num_033, EuFe_num_025, EuFe_num, colors=['red','blue','orange', '#228B22', '#FF1493', 'black'], linewidth=1)
    #plot_linking_length_split (FeH_num_05, FeH_num_033, FeH_num_025, FeH_num, CFe_num_05, CFe_num_033, CFe_num_025, CFe_num, SrFe_num_05, SrFe_num_033, SrFe_num_025, SrFe_num, BaFe_num_05, BaFe_num_033, BaFe_num_025, BaFe_num, EuFe_num_05, EuFe_num_033, EuFe_num_025, EuFe_num, colors=['red','blue', 'black'], linewidth=1)

