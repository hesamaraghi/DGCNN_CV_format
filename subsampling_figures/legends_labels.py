datasets_name_and_num_classes = {
    "NCARS": {"name": "N-Cars", "num_classes": 2},
    "NASL": {"name": "N-ASL", "num_classes": 24},
    "NCALTECH101": {"name": "N-Caltech101", "num_classes": 101},
    "DVSGESTURE_TONIC": {"name": "DVS-Gesture", "num_classes": 11},
    "FAN1VS3": {"name": "Fan1vs3", "num_classes": 2}
}

legend_labels = {
                    'spatial_zero_offsets':     'Spatial (zero offsets)', 
                    'spatial_dataset_wide_random_offsets': 'Spatial (dataset wide random offsets)',
                    'spatial_8x10_nonzero_offsets': 'Spatial (8x10 non-zero offsets)',
                    'temporal_zero_offset':    'Temporal (zero offset)', 
                    'temporal_zero_offset_fixed_interval': 'Temporal (fixed interval)',
                    'temporal_dataset_wide_random_offset': 'Temporal (dataset wide random offsets)',
                    'random':      'Random (not fixed subsampling)', 
                    'random_fixed': 'Random (fixed subsampling)',
                    'spatiotemporal': 'Spatiotemporal',
                    'spatiotemporal_fixed': 'Spatiotemporal (fixed subsampling)',
                    'spatiotemporal_fixed_30_7x7': 'Spatiotemporal 30 7x7',
                    'spatial_random_offsets': 'Spatial (fixed random offsets)',
                    'temporal_random_offsets': 'Temporal (fixed random offsets)',
                    'spatiotemporal_fixed_normalized_mean': 'Spatiotemporal (mean normalized)', 
                    'baseline_event_count': 'Baseline (event count)',
                    'tos2dharris_fixed': 'TOS 2D Harris',
}

legend_labels_main_short = {
                    'spatial_dataset_wide_random_offsets': 'Spatial',
                    'temporal_dataset_wide_random_offset': 'Temporal',
                    'random_fixed': 'Random',
                    'spatiotemporal_fixed_30_7x7': 'Causal Density-based',
                    'spatiotemporal_fixed_normalized_mean': 'Density-based (mean normalized)',
                    'baseline_event_count': 'Event Count',
                    'tos2dharris_fixed': 'Corner-based w/ 2D Harris',
                    # 'tos2dharris_fixed': 'Harris Corner Detector', 
}
legend_labels_main_short = legend_labels | legend_labels_main_short

#   "olive", 
#      "lime", "maroon", "navy", "teal",
#     "gold", "coral", "indigo", "turquoise"

color_dict = {
                'spatial_zero_offsets': 'gray',
                'spatial_dataset_wide_random_offsets': 'red',
                'spatial_8x10_nonzero_offsets': 'olive',
                'temporal_zero_offset': 'cyan',
                'temporal_zero_offset_fixed_interval': 'black',
                'temporal_dataset_wide_random_offset': 'green',
                'random': 'pink',
                'random_fixed': 'blue',
                'spatiotemporal_fixed_30_7x7': 'orange',
                'spatiotemporal_fixed': 'cyan',
                'spatial_random_offsets': 'magenta',
                'temporal_random_offsets': 'yellow',
                'spatiotemporal_fixed_normalized_mean': 'violet',
                'baseline_event_count': 'purple',
                'tos2dharris_fixed': 'brown',
}
marker_dict = {
                'spatial_zero_offsets': '_',
                'spatial_dataset_wide_random_offsets': 'o',
                'spatial_8x10_nonzero_offsets': 's',
                'temporal_zero_offset': 'x',
                'temporal_zero_offset_fixed_interval': '*',
                'temporal_dataset_wide_random_offset': '^',
                'random': '^',
                'random_fixed': 'D',
                'spatiotemporal_fixed_30_7x7': 's',
                'spatiotemporal_fixed': 'x',
                'spatial_random_offsets': '+',
                'temporal_random_offsets': 'p',
                'spatiotemporal_fixed_normalized_mean': 'H',
                'baseline_event_count': '<',
                'tos2dharris_fixed': 'h',
}
dataset_name_dict = {
    "NASL":    "N-ASL"    ,   
    "FAN1VS3":   "Fan1vs3"  ,   
    "NCALTECH101": "N-Caltech101"   ,
    "DVSGESTURE_TONIC": "DVS-Gesture" ,
    "NCARS":          "N-Cars"
}

colormap_dict = {
                'spatial_zero_offsets': 'Greys',
                'spatial_dataset_wide_random_offsets': 'Reds',
                'spatial_8x10_nonzero_offsets': 'YlGn',
                'temporal_zero_offset': 'GnBu',
                'temporal_zero_offset_fixed_interval': 'YlGnBu',
                'temporal_dataset_wide_random_offset': 'Greens',
                'random': 'PuBuGn',
                'random_fixed': 'Blues',
                'spatiotemporal_fixed_30_7x7': 'Oranges',
                'spatiotemporal_fixed': 'BuGn',
                'spatial_random_offsets': 'RdPu',
                'temporal_random_offsets': 'YlOrBr',
                'spatiotemporal_fixed_normalized_mean': 'PuRd',
                'baseline_event_count': 'Purples',
                'tos2dharris_fixed': 'YlOrRd',
}

colormap_list = [
                    'Blues', 
                    'Greens', 
                    'Oranges', 
                    'Reds', 
                    'Purples', 
                    'Greys', 
                    'YlOrBr', 
                    'YlOrRd', 
                    'OrRd', 
                    'PuRd', 
                    'RdPu', 
                    'BuPu', 
                    'GnBu', 
                    'PuBu', 
                    'YlGnBu', 
                    'PuBuGn', 
                    'BuGn', 
                    'YlGn'
                 ]

marker_list = ['o', 's', 'D', 'x', '^', 'v', '<', '>', 'p', 'P', '*', 'X', 'd', 'H', 'h', '+', '|', '_']