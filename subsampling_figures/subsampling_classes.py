from utils.results_utils import BaseSubsamplingType
from legends_labels import *

'''
- Each curve should has its own class.
- Parameters are those that the accuracies would be averaged over them.
- Each parameter set has **ONLY ONE** test acc.
- (?!) Each dataset has its own class instance.
'''

allowed_seeds = [
      "42",
      "420",
      '4200',
      "42000",
      "420000",
      "4200000",
      "12",
      "120",
      "1200",
      "12000",
      "120000",
      "1200000",
      "24",
      "240",
      "2400",
      "24000",
      "240000",
      "2400000",
]

class BaseSpatial(BaseSubsamplingType):
    
    def __init__(self, dataset_name = None, label = None, color = None, marker = None):
        super().__init__(dataset_name = dataset_name, label = label, color = color, marker = marker)

    def get_name(self):
        raise NotImplementedError
        
    def set_parameter_loader(self):
        return {
            "seed": lambda config: config["seed"],
            "h_r": lambda config: config["transform"]["train"]["spatial_subsampling"]["h_r"],
            "v_r": lambda config: config["transform"]["train"]["spatial_subsampling"]["v_r"],
            "h_r_offset": lambda config: config["transform"]["train"]["spatial_subsampling"]["h_r_offset"],
            "v_r_offset": lambda config: config["transform"]["train"]["spatial_subsampling"]["v_r_offset"],
        }
    
    def zipped_parameters(self):
        raise NotImplementedError
     
    def filter_run(self, run):
        raise NotImplementedError
    

class Spatial(BaseSubsamplingType):
    
    def __init__(self, dataset_name = None, label = None, color = None, marker = None):
        super().__init__(dataset_name = dataset_name, label = label, color = color, marker = marker)

    def get_name(self):
        return "spatial_dataset_wide_random_offsets"
        
    def set_parameter_loader(self):
        return {
            "seed": lambda config: config["seed"],
            "h_r": lambda config: config["transform"]["train"]["spatial_subsampling"]["h_r"],
            "v_r": lambda config: config["transform"]["train"]["spatial_subsampling"]["v_r"],
            "h_r_offset": lambda config: config["transform"]["train"]["spatial_subsampling"]["h_r_offset"],
            "v_r_offset": lambda config: config["transform"]["train"]["spatial_subsampling"]["v_r_offset"],
        }
    
    def zipped_parameters(self):
        return ["h_r", "v_r"]
     
    def filter_run(self, run):
        if 'transform' in run.config and 'train' in run.config['transform']:
            cfg_dict = run.config["transform"]["train"]
            if 'spatial_subsampling' in cfg_dict and cfg_dict["spatial_subsampling"]["transform"]:
                if 'dataset_wide_random_offsets' in cfg_dict["spatial_subsampling"] and cfg_dict["spatial_subsampling"]["dataset_wide_random_offsets"]:
                    return True
                # if "dataset_wide_random_offset" in run.project:
                #     return True
        return False
                # else:
                #     key_name = 'spatial_zero_offsets'
    
    
class SpatialRandom(Spatial):

    def get_name(self):
        return "spatial_random_offsets"
        
    def set_parameter_loader(self):
        return {
            "seed": lambda config: config["seed"],
            "h_r": lambda config: config["transform"]["train"]["spatial_subsampling_random"]["h_r"],
            "v_r": lambda config: config["transform"]["train"]["spatial_subsampling_random"]["v_r"],
        }
     
    def filter_run(self, run):
        if 'transform' in run.config and 'train' in run.config['transform']:
            cfg_dict = run.config["transform"]["train"]
            if 'spatial_subsampling_random' in cfg_dict and cfg_dict["spatial_subsampling_random"]["transform"]:
                return True
        return False
                # else:
                #     key_name = 'spatial_zero_offsets'

class SpatialZeroOffset(BaseSpatial):

    def get_name(self):
        return "spatial_zero_offsets"
        
    def set_parameter_loader(self):
        return {
            "seed": lambda config: config["seed"],
            "h_r": lambda config: config["transform"]["train"]["spatial_subsampling"]["h_r"],
            "v_r": lambda config: config["transform"]["train"]["spatial_subsampling"]["v_r"],
            # "h_r_offset": lambda config: config["transform"]["train"]["spatial_subsampling"]["h_r_offset"],
            # "v_r_offset": lambda config: config["transform"]["train"]["spatial_subsampling"]["v_r_offset"],
        }
    
    def zipped_parameters(self):
        return ["h_r", "v_r"]
     
    def filter_run(self, run):
        if 'seed' not in run.config or str(run.config['seed']) not in allowed_seeds:
            return False
        if 'transform' in run.config and 'train' in run.config['transform']:
            cfg_dict = run.config["transform"]["train"]
            if 'spatial_subsampling' in cfg_dict and cfg_dict["spatial_subsampling"]["transform"]:
                if ("dataset_wide_random_offset" not in cfg_dict["spatial_subsampling"]) or (not cfg_dict["spatial_subsampling"]["dataset_wide_random_offset"]):
                    if 'h_r_offset' in cfg_dict["spatial_subsampling"] and 'v_r_offset' in cfg_dict["spatial_subsampling"]:
                        if cfg_dict["spatial_subsampling"]["h_r_offset"] == 0 and cfg_dict["spatial_subsampling"]["v_r_offset"] == 0:
                            return True
                    else:
                        print("OLD VERSION: h_r_offset or v_r_offset not found in config")
                        return True
        return False
                # else:
                #     key_name = 'spatial_zero_offsets'

class Temporal(BaseSubsamplingType):
    
    def __init__(self, dataset_name = None, label = None, color = None, marker = None):
        super().__init__(dataset_name = dataset_name, label = label, color = color, marker = marker)

    def get_name(self):
        return "temporal_dataset_wide_random_offset"
    
    def set_parameter_loader(self):
        return {
            "seed": lambda config: config["seed"],
            "t_r": lambda config: config["transform"]["train"]["temporal_subsampling"]["subsampling_ratio"],
            "window_size": lambda config: config["transform"]["train"]["temporal_subsampling"]["window_size"],
            "time_offset_coefficient": lambda config: config["transform"]["train"]["temporal_subsampling"]["time_offset_coefficient"],
            # "fixed_interval": lambda config: config["transform"]["train"]["temporal_subsampling"]["fixed_interval"],
        }
    
    def zipped_parameters(self):
        return ["t_r"]
     
    def filter_run(self, run):
        if 'transform' in run.config and 'train' in run.config['transform']:
            cfg_dict = run.config["transform"]["train"]
            if 'temporal_subsampling' in cfg_dict and cfg_dict["temporal_subsampling"]["transform"]:
                if "dataset_wide_random_offset" in run.project:
                    if "fixed_interval" in cfg_dict["temporal_subsampling"] and cfg_dict["temporal_subsampling"]["fixed_interval"]:
                        return False
                    return True
        return False
    
class TemporalRandom(BaseSubsamplingType):
    
    def __init__(self, dataset_name = None, label = None, color = None, marker = None):
        super().__init__(dataset_name = dataset_name, label = label, color = color, marker = marker)

    def get_name(self):
        return "temporal_random_offsets"
    
    def set_parameter_loader(self):
        return {
            "seed": lambda config: config["seed"],
            "t_r": lambda config: config["transform"]["train"]["temporal_subsampling_random"]["subsampling_ratio"],
            "window_size": lambda config: config["transform"]["train"]["temporal_subsampling_random"]["window_size"],
            # "time_offset_coefficient": lambda config: config["transform"]["train"]["temporal_subsampling"]["time_offset_coefficient"],
            # "fixed_interval": lambda config: config["transform"]["train"]["temporal_subsampling"]["fixed_interval"],
        }
    
    def zipped_parameters(self):
        return ["t_r"]
     
    def filter_run(self, run):
        if 'transform' in run.config and 'train' in run.config['transform']:
            cfg_dict = run.config["transform"]["train"]
            if 'temporal_subsampling_random' in cfg_dict and cfg_dict["temporal_subsampling_random"]["transform"]:
                if "fixed_interval" in cfg_dict["temporal_subsampling_random"] and cfg_dict["temporal_subsampling_random"]["fixed_interval"]:
                    return False
                return True
        return False
    
class TemporalZeroOffset(BaseSubsamplingType):
    
    def __init__(self, dataset_name = None, label = None, color = None, marker = None):
        super().__init__(dataset_name = dataset_name, label = label, color = color, marker = marker)

    def get_name(self):
        return "temporal_zero_offset"
    
    def set_parameter_loader(self):
        return {
            "seed": lambda config: config["seed"],
            "t_r": lambda config: config["transform"]["train"]["temporal_subsampling"]["subsampling_ratio"],
            "window_size": lambda config: config["transform"]["train"]["temporal_subsampling"]["window_size"],
            # "time_offset_coefficient": lambda config: config["transform"]["train"]["temporal_subsampling"]["time_offset_coefficient"],
            # "fixed_interval": lambda config: config["transform"]["train"]["temporal_subsampling"]["fixed_interval"],
        }
    
    def zipped_parameters(self):
        return ["t_r"]
     
    def filter_run(self, run):
        if 'transform' in run.config and 'train' in run.config['transform']:
            cfg_dict = run.config["transform"]["train"]
            if 'temporal_subsampling' in cfg_dict and cfg_dict["temporal_subsampling"]["transform"]:
                if "dataset_wide_random_offset" not in run.project:
                    if "fixed_interval" in cfg_dict["temporal_subsampling"] and cfg_dict["temporal_subsampling"]["fixed_interval"]:
                        return False
                    if 'time_offset_coefficient' in cfg_dict["temporal_subsampling"]:
                        if cfg_dict["temporal_subsampling"]["time_offset_coefficient"] == 0:
                            return True
                    else:
                        print("OLD VERSION: time_offset_coefficient not found in config")
                        return True
        return False        
    
class Random(BaseSubsamplingType):
    
    def __init__(self, dataset_name = None, label = None, color = None, marker = None):
        super().__init__(dataset_name = dataset_name, label = label, color = color, marker = marker)

    def get_name(self):
        return "random_fixed"
        
    def set_parameter_loader(self):
        return {
            "seed": lambda config: config["seed"],
            "p": lambda config: config["transform"]["train"]["random_ratio_subsampling"],
            "fixed_subsampling": lambda config: self.get_fixed_vs_not(config),
        }
    
    def get_fixed_vs_not(self, config):
        if 'fixed_sampling' in config['transform']['train'] and config['transform']['train']['fixed_sampling']['transform']:
            if "seed_str" in  config['transform']['train']['fixed_sampling']:
                return config['transform']['train']['fixed_sampling']["seed_str"]
        return None
    
    def zipped_parameters(self):
        return ["p"]
     
    def filter_run(self, run):
        if 'transform' in run.config and 'train' in run.config['transform']:
            cfg_dict = run.config["transform"]["train"]
            if 'random_ratio_subsampling' in cfg_dict and cfg_dict["random_ratio_subsampling"] is not None:
                if 'fixed_sampling' in cfg_dict and cfg_dict['fixed_sampling']['transform']:
                    return True
        return False
    
class Spatiotemporal(BaseSubsamplingType):
    
    def __init__(self, dataset_name = None, label = None, color = None, marker = None):
        super().__init__(dataset_name = dataset_name, label = label, color = color, marker = marker)

    def get_name(self):
        return "spatiotemporal_fixed"
        
    def set_parameter_loader(self):
        return {
            "seed": lambda config: config["seed"],
            "tau": lambda config: config["transform"]["train"]["spatiotemporal_filtering_subsampling"]["tau"],
            "filter_size": lambda config: config["transform"]["train"]["spatiotemporal_filtering_subsampling"]["filter_size"],
            "sampling_threshold": lambda config: config["transform"]["train"]["spatiotemporal_filtering_subsampling"]["sampling_threshold"],
            "normalization_length": lambda config: config["transform"]["train"]["spatiotemporal_filtering_subsampling"]["normalization_length"],
            "fixed_subsampling": lambda config: self.get_fixed_vs_not(config),
        }
    
    def get_fixed_vs_not(self, config):
        if 'fixed_sampling' in config['transform']['train'] and config['transform']['train']['fixed_sampling']['transform']:
            if "seed_str" in  config['transform']['train']['fixed_sampling']:
                return config['transform']['train']['fixed_sampling']["seed_str"]
        return None
    
    def zipped_parameters(self):
        return ["sampling_threshold"]
     
    def filter_run(self, run):
        if 'transform' in run.config and 'train' in run.config['transform']:
            cfg_dict = run.config["transform"]["train"]
            if 'spatiotemporal_filtering_subsampling' in cfg_dict and cfg_dict["spatiotemporal_filtering_subsampling"]["transform"]:     
                if 'fixed_sampling' in cfg_dict and cfg_dict['fixed_sampling']['transform']:
                    return True
        return False
    
class TOS2DHarris(BaseSubsamplingType):
    
    def __init__(self, dataset_name = None, label = None, color = None, marker = None):
        super().__init__(dataset_name = dataset_name, label = label, color = color, marker = marker)

    def get_name(self):
        return "tos2dharris_fixed"
        
    def set_parameter_loader(self):
        return {
            "seed": lambda config: config["seed"],
            "TOS_T": lambda config: config["transform"]["train"]["tos_2DHarris_subsampling"]["TOS_T"],
            "filter_size": lambda config: config["transform"]["train"]["tos_2DHarris_subsampling"]["filter_size"],
            "Harris_block_size": lambda config: config["transform"]["train"]["tos_2DHarris_subsampling"]["Harris_block_size"],
            "Harris_ksize": lambda config: config["transform"]["train"]["tos_2DHarris_subsampling"]["Harris_ksize"],
            "Harris_k": lambda config: config["transform"]["train"]["tos_2DHarris_subsampling"]["Harris_k"],
            "sampling_threshold": lambda config: config["transform"]["train"]["tos_2DHarris_subsampling"]["sampling_threshold"],
            "fixed_subsampling": lambda config: self.get_fixed_vs_not(config),
        }
    
    def get_fixed_vs_not(self, config):
        if 'fixed_sampling' in config['transform']['train'] and config['transform']['train']['fixed_sampling']['transform']:
            if "seed_str" in  config['transform']['train']['fixed_sampling']:
                return config['transform']['train']['fixed_sampling']["seed_str"]
        return None
    
    def zipped_parameters(self):
        return ["sampling_threshold"]
     
    def filter_run(self, run):
        if 'transform' in run.config and 'train' in run.config['transform']:
            cfg_dict = run.config["transform"]["train"]
            if 'tos_2DHarris_subsampling' in cfg_dict and cfg_dict["tos_2DHarris_subsampling"]["transform"]:     
                if 'fixed_sampling' in cfg_dict and cfg_dict['fixed_sampling']['transform']:
                    return True
        return False
    
class BaseLineEventCount(BaseSubsamplingType):
    
    def __init__(self, dataset_name = None, label = None, color = None, marker = None):
        super().__init__(dataset_name = dataset_name, label = label, color = color, marker = marker)

    def get_name(self):
        return "baseline_event_count"
        
    def set_parameter_loader(self):
        return {
            "seed": lambda config: config["seed"],
            "h_r": lambda config: config["pre_transform"]["train"]["baseline_event_count"]["h_r"],
            "v_r": lambda config: config["pre_transform"]["train"]["baseline_event_count"]["v_r"],
            "threshold": lambda config: config["pre_transform"]["train"]["baseline_event_count"]["threshold"],
        }
    
    def zipped_parameters(self):
        return ["h_r", "v_r", "threshold"]
     
    def filter_run(self, run):
        if 'pre_transform' in run.config and 'train' in run.config['pre_transform']:
            cfg_dict = run.config["pre_transform"]["train"]
            if 'baseline_event_count' in cfg_dict and cfg_dict["baseline_event_count"]["transform"]:
                return True
        return False
    
class Spatiotemporal10(Spatiotemporal):
    def get_name(self):
        return "spatiotemporal_fixed_10_7x7"
    
    def get_color(self):
        return colormap_list[0]
    
    def get_marker(self):
        return marker_list[0]
    
    def filter_run(self, run):
        if super().filter_run(run):
            if run.config["transform"]["train"]["spatiotemporal_filtering_subsampling"]["tau"] == 10 and run.config["transform"]["train"]["spatiotemporal_filtering_subsampling"]["filter_size"] == 7:
                return True
        return False
    
class Spatiotemporal100(Spatiotemporal):
    def get_name(self):
        return "spatiotemporal_fixed_100_7x7"
    
    def get_color(self):
        return colormap_list[1]
    
    def get_marker(self):
        return marker_list[1]
    
    def filter_run(self, run):
        if super().filter_run(run):
            if run.config["transform"]["train"]["spatiotemporal_filtering_subsampling"]["tau"] == 100 and run.config["transform"]["train"]["spatiotemporal_filtering_subsampling"]["filter_size"] == 7:
                return True
        return False
    
class Spatiotemporal30(Spatiotemporal):
    def get_name(self):
        return "spatiotemporal_fixed_30_7x7"
    
    def get_color(self):
        return colormap_list[2]
    
    def get_marker(self):
        return marker_list[2]
    
    def filter_run(self, run):
        if super().filter_run(run):
            if run.config["transform"]["train"]["spatiotemporal_filtering_subsampling"]["tau"] == 30 and \
                run.config["transform"]["train"]["spatiotemporal_filtering_subsampling"]["filter_size"] == 7:
                if "mean_normalized" in run.config["transform"]["train"]["spatiotemporal_filtering_subsampling"] and \
                    run.config["transform"]["train"]["spatiotemporal_filtering_subsampling"]["mean_normalized"] == True:  
                        return False
                return True
        return False   
    
class Spatiotemporal100_5x5(Spatiotemporal):
    def get_name(self):
        return "spatiotemporal_fixed_100_5x5"
    
    def get_color(self):
        return colormap_list[3]
    
    def get_marker(self):
        return marker_list[3]
    
    def filter_run(self, run):
        if super().filter_run(run):
            if run.config["transform"]["train"]["spatiotemporal_filtering_subsampling"]["tau"] == 100 and run.config["transform"]["train"]["spatiotemporal_filtering_subsampling"]["filter_size"] == 5:
                return True
        return False
    
class Spatiotemporal100_9x9(Spatiotemporal):
    def get_name(self):
        return "spatiotemporal_fixed_100_9x9"
    
    def get_color(self):
        return colormap_list[4]
    
    def get_marker(self):
        return marker_list[4]
    
    def filter_run(self, run):
        if super().filter_run(run):
            if run.config["transform"]["train"]["spatiotemporal_filtering_subsampling"]["tau"] == 100 and run.config["transform"]["train"]["spatiotemporal_filtering_subsampling"]["filter_size"] == 9:
                return True
        return False
    
class Spatiotemporal100_11x11(Spatiotemporal):
    def get_name(self):
        return "spatiotemporal_fixed_100_11x11"
    
    def get_color(self):
        return colormap_list[5]
    
    def get_marker(self):
        return marker_list[5]
    
    def filter_run(self, run):
        if super().filter_run(run):
            if run.config["transform"]["train"]["spatiotemporal_filtering_subsampling"]["tau"] == 100 and run.config["transform"]["train"]["spatiotemporal_filtering_subsampling"]["filter_size"] == 11:
                return True
        return False

class Spatiotemporal_normalized_mean(Spatiotemporal):
    def get_name(self):
        return "spatiotemporal_fixed_normalized_mean"
    
    def get_color(self):
        return colormap_list[6]
    
    def get_marker(self):
        return marker_list[6]
    
    def filter_run(self, run):
        if super().filter_run(run):
            if      run.config["transform"]["train"]["spatiotemporal_filtering_subsampling"]["tau"] == 30 and \
                    run.config["transform"]["train"]["spatiotemporal_filtering_subsampling"]["filter_size"] == 7 and \
                    "mean_normalized" in run.config["transform"]["train"]["spatiotemporal_filtering_subsampling"] and \
                    run.config["transform"]["train"]["spatiotemporal_filtering_subsampling"]["mean_normalized"] == True:       
                return True
        return False
    
class Spatial8x10NonZeroOffset(BaseSpatial):

    def get_name(self):
        return "spatial_8x10_nonzero_offsets"
        
    def set_parameter_loader(self):
        return {
            "seed": lambda config: config["seed"],
            "h_r": lambda config: config["transform"]["train"]["spatial_subsampling"]["h_r"],
            "v_r": lambda config: config["transform"]["train"]["spatial_subsampling"]["v_r"],
            "h_r_offset": lambda config: config["transform"]["train"]["spatial_subsampling"]["h_r_offset"],
            "v_r_offset": lambda config: config["transform"]["train"]["spatial_subsampling"]["v_r_offset"],
        }
    
    def zipped_parameters(self):
        return ["h_r_offset", "v_r_offset"]
     
    def filter_run(self, run):
        if 'seed' not in run.config or str(run.config['seed']) not in allowed_seeds:
            return False
        if 'transform' in run.config and 'train' in run.config['transform']:
            cfg_dict = run.config["transform"]["train"]
            if 'spatial_subsampling' in cfg_dict and cfg_dict["spatial_subsampling"]["transform"]:
                if 'h_r_offset' in cfg_dict["spatial_subsampling"] and 'v_r_offset' in cfg_dict["spatial_subsampling"]:
                    if cfg_dict["spatial_subsampling"]["h_r"] == 8 and cfg_dict["spatial_subsampling"]["v_r"] == 10:
                        if cfg_dict["spatial_subsampling"]["h_r_offset"] != 0 or cfg_dict["spatial_subsampling"]["v_r_offset"] != 0:
                            return True
                else:
                    return False
        return False
