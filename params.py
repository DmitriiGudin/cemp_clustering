from __future__ import division
import numpy as np


init_file = '/home/may/Work/CEMP_StarGrouping/Master_runs/Final_version_StarHorse_GAIA_HDBSCAN/data/base_GAIA_dist_GAIA_RV_Group_I_30.csv'
kin_file = '/home/may/Work/CEMP_StarGrouping/Master_runs/Final_version_StarHorse_GAIA_HDBSCAN/data/orbit_GAIA_dist_GAIA_RV_Group_I_30.csv'
master_file = '/home/may/Work/CEMP_StarGrouping/Master_runs/Final_version_StarHorse_GAIA_HDBSCAN/data/Master_CEMP_data.csv'
clusters_file = '/home/may/Work/CEMP_StarGrouping/Master_runs/Final_version_StarHorse_GAIA_HDBSCAN/clusters.py'

dispersion_distrib_file_mask = ['/home/may/Work/CEMP_StarGrouping/Master_runs/Final_version_StarHorse_GAIA_HDBSCAN/data/dispersion_distribs/dispersion_distrib_','.hdf5']
dispersion_distrib_plot_file_mask = ['/home/may/Work/CEMP_StarGrouping/Master_runs/Final_version_StarHorse_GAIA_HDBSCAN/data/dispersion_distribs/plots/dispersion_distrib_','.eps']
cluster_dispersion_plot_file_mask = ['/home/may/Work/CEMP_StarGrouping/Master_runs/Final_version_StarHorse_GAIA_HDBSCAN/plots/cluster_dispersions/cluster_dispersion_','.png']
dispersion_cluster_size = (3,30)
biweight_estimator_min_cluster_size = 4

colors_markers = [\
('red', '#FF0000', '+', 2, 1),\
('blue', '#0000FF', '1', 2, 1.1),\
('seagreen', '#2E8B57', '2', 2, 1.1),\
('black', '#000000', 'x', 2, 1),\
('purple', '#800080', '*', 1, 1.1),\
('orange', '#FFA500', '.', 1, 1.25),\
('teal', '#008080', '^', 1, 1),\
('darkred', '#8B0000', 'o', 1, 1),\
('darkblue', '#00008B', 'd', 1, 1),\
('darkgreen', '#006400', '3', 2, 1.1),\
('saddlebrown', '#8B4513', '*', 1, 1),\
('darkviolet', '#9400D3', '+', 2, 1),\
('darkorange', '#FF8C00', '1', 2, 1.1),\
('darkmagenta', '#8B008B', '2', 2, 1.1),\
('chocolate', '#D2691E', 'x', 2, 1),\
('royalblue', '#4169E1', '4', 2, 1.1),\
('forestgreen', '#228B22', '.', 1, 1.25),\
('deeppink', '#FF1493', 'd', 1, 1),\
('indigo', '#4B0082', 'x', 2, 1),\
('mediumvioletred', '#C71585', 'o', 1, 1),\
('fuchsia', '#FF00FF', '3', 2, 1.1),\
('orangered', '#FF4500', '4', 2, 1.1),\
('navy', '#000080', '.', 1, 1.25),\
('dodgerblue', '#1E90FF', '*', 1, 1),\
('red', '#FF0000', '*', 1, 1),\
('blue', '#0000FF', 'd', 1, 1),\
('darkmagenta', '#8B008B', 'd', 1, 1),\
('chocolate', '#D2691E', '1', 2, 1.1),\
('purple', '#800080', '+', 2, 1),\
('orange', '#FFA500', '*', 1, 1)\
]
# (Color name, color code, marker type, marker linewidth, marker scaling factor)

colors = [a[1] for a in colors_markers]
markers = [a[2] for a in colors_markers]
marker_linewidths = [a[3] for a in colors_markers]
marker_scales = [a[4] for a in colors_markers]
