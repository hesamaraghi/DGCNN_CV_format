import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps

from importlib import reload

import legends_labels
legends_labels = reload(legends_labels)
from legends_labels import *

def plot_all(result_list, 
             save_to = None, 
             mean_line = True, 
             scatter_dots = False, 
             scatter_range = (0.4, 0.8),
             colormap_dict_ = colormap_dict,
             color_dict_ = color_dict,
             marker_dict_ = marker_dict,
             x_axis = 'num_events',
             label_dict = legend_labels_main_short,
             figsize = (10, 6),
             ):
    assert x_axis in ['num_events', 'bandwidth'], "x_axis must be either 'num_events' or 'bandwidth'"
    assert isinstance(scatter_range, tuple) and len(scatter_range) == 2, "scatter_range must be a tuple of length 2"
    assert mean_line or scatter_dots, "Nothing to plot! At least one of mean_line or scatter_dots must be True"
    if not isinstance(result_list, list):
        result_list = [result_list]
    
    unique_tuples = [r.df.sort_values(by=x_axis)[r._zipped_parameters_label].unique() for r in result_list]

    # Generate a color for each unique tuple
    # line_colors_list = [colormaps[colormap_list[i]](0.5) for i in range(len(result_list))]  

    color_dict_ = {**color_dict, **color_dict_}
    plt.subplots(figsize=figsize)
    for result in result_list:
        unique_tuples = result.df.sort_values(by=x_axis)[result._zipped_parameters_label].unique()
        colors_list = colormaps[colormap_dict_[result.get_name()]](np.linspace(scatter_range[0], scatter_range[1], len(unique_tuples)))
        tuple_to_color_list = {t: colors_list[i] for i, t in enumerate(unique_tuples)}
        if scatter_dots:
            plt.scatter(result.df[x_axis], result.df['test_acc'], 
                        c=[tuple_to_color_list[pr] for pr in result.df[result._zipped_parameters_label]],
                        marker='o', 
                        s=8, 
                        label=label_dict[result.get_name()],
                        )
        if mean_line:
            plt.errorbar(result.df_zipped[x_axis]['nanmean'], result.df_zipped['test_acc']['nanmean'], 
                                # xerr=result.df_zipped[x_axis]["nanstd"],
                                yerr=result.df_zipped['test_acc']["nanstd"], 
                                # label=result.label,
                                label=label_dict[result.get_name()],
                                # ecolor=line_colors_list[i],
                                ecolor=color_dict_[result.get_name()],
                                # c=line_colors_list[i],
                                c=color_dict_[result.get_name()],
                                capsize=5, 
                                # marker='o', 
                                marker=marker_dict_[result.get_name()],
                                linestyle=':', 
                                markersize=8, 
                                markeredgewidth=1,
                                linewidth=3, 
                                elinewidth=2, 
                                alpha=.85)
    x_label ='#events per video' if x_axis == 'num_events' else 'Bandwidth (Ev/s)'
    plt.xlabel(x_label, fontsize=18)  # Increase label font size
    plt.ylabel('Test Accuracy', fontsize=18)  # Increase label font size
    plt.title('Test Accuracy (with std error bars) vs ' + 
              ("#events" if x_axis == 'num_events' else 'Bandwidth'), fontsize=20)  # Increase title font size
    plt.xscale('log')  # Log scale if sparsity values span several orders of magnitude

    plt.xticks(fontsize=16)  # Increase tick font size
    plt.yticks(fontsize=16)  # Increase tick font size
    plt.grid(True)
    plt.legend(
        fontsize=16,
        # bbox_to_anchor=(1.04, 0), 
        loc="lower right", 
        # borderaxespad=0,
        )  # Increase legend font size , 
    if save_to:
        plt.savefig(save_to, bbox_inches='tight')
    plt.show()
