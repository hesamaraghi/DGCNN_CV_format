import os
from omegaconf import OmegaConf, listconfig
import random
import wandb
from utils.config_utils import show_cfg
import yaml
import numpy as np

def main(config = None, seed = None, tags = None):

    cfg = OmegaConf.load(config.cfg_path)
    cfg = OmegaConf.merge(cfg,config)
    cfg_bare = OmegaConf.load("config_bare.yaml")
    cfg = OmegaConf.merge(cfg_bare,cfg)
    
    cfg.seed = seed
    
    run = wandb.run if wandb.run else wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, reinit=True, tags=[f"seed_{seed}", *tags])
    
    cfg.wandb.id = run.id
    cfg.wandb.experiment_name = run.name
    cfg.wandb.entity = run.entity
    cfg.wandb.project = run.project
    
    default_root_dir = os.path.join(os.getcwd(),'pl_default_dir', run.project, run.name)
    if not os.path.exists(default_root_dir):
        os.makedirs(default_root_dir)
    cfg.train.default_root_dir = default_root_dir
    final_config_path = os.path.join(default_root_dir,"config.yml")
    cfg.cfg_path = final_config_path
    OmegaConf.save(config=cfg, f=final_config_path)

    wandb.config["resume_config"] = OmegaConf.to_container(cfg, resolve=True)
    sbatch_command = f"python train.py cfg_path={final_config_path}"
    print(sbatch_command)
    with open(os.path.join(os.getcwd(),"pl_default_dir", run.project, "sweep_sbatch_commands.txt"), "a") as f:
        f.write(sbatch_command + "\n")
    wandb.finish()

if __name__ == '__main__':
    
    cmd_cfg = OmegaConf.from_cli()

    if hasattr(cmd_cfg, "sweep_config"):
        sweep_config = OmegaConf.load(cmd_cfg.sweep_config)
    else:
        raise ValueError("sweep_config not found in args")
                              
    if hasattr(sweep_config, "sweep_parameters"):  
        cfg_parameters = sweep_config.sweep_parameters
    else:
        raise ValueError("sweep_parameters not found in config file")

    tags = []
    if hasattr(cmd_cfg, "tags"):
        if not isinstance(cmd_cfg.tags, listconfig.ListConfig):
            if isinstance(cmd_cfg.tags, str):
                tags = [cmd_cfg.tags]
            else:
                raise ValueError("tags should be a list or string")
        else:
            tags = cmd_cfg.tags

    cfg = OmegaConf.create()
    cfg.wandb = OmegaConf.create()
    cfg.wandb.entity = sweep_config.entity
    cfg.wandb.project = sweep_config.project
    
    if hasattr(sweep_config, "cfg_path"):  
        cfg.cfg_path = sweep_config.cfg_path
    else:
        raise ValueError("cfg_path not found in config file")

    # Extract parameters
    seeds = cfg_parameters['seed']

    # Create all parameter combinations for spatial subsampling


    if hasattr(cfg_parameters['transform']['train'], 'spatial_subsampling') and cfg_parameters['transform']['train']['spatial_subsampling']['transform']:
          
        spatial_subsampling_hr = cfg_parameters['transform']['train']['spatial_subsampling']['h_r']
        spatial_subsampling_vr = cfg_parameters['transform']['train']['spatial_subsampling']['v_r']
        spatial_subsampling_combinations = list(zip(spatial_subsampling_hr, spatial_subsampling_vr))
        
        for hr_vr in spatial_subsampling_combinations:
            for seed in seeds:
                if "dataset_wide_random_offsets" in cfg_parameters['transform']['train']['spatial_subsampling'] and cfg_parameters['transform']['train']['spatial_subsampling']['dataset_wide_random_offsets']:
                    rng = np.random.default_rng(seed)
                    spatial_subsampling_hr_offset = [rng.integers(hr_vr[0], dtype=int)]
                    spatial_subsampling_vr_offset = [rng.integers(hr_vr[1], dtype=int)]
                else:
                    spatial_subsampling_hr_offset = [0]
                    spatial_subsampling_vr_offset = [0]
                    if ('h_r_offset' in cfg_parameters['transform']['train']['spatial_subsampling']) and cfg_parameters['transform']['train']['spatial_subsampling']['h_r_offset']:
                        spatial_subsampling_hr_offset = cfg_parameters['transform']['train']['spatial_subsampling']['h_r_offset']
                    if ('v_r_offset' in cfg_parameters['transform']['train']['spatial_subsampling']) and cfg_parameters['transform']['train']['spatial_subsampling']['v_r_offset']:
                        spatial_subsampling_vr_offset = cfg_parameters['transform']['train']['spatial_subsampling']['v_r_offset']

                for hr_offset in spatial_subsampling_hr_offset:
                    for vr_offset in spatial_subsampling_vr_offset:
                        cfg.transform = OmegaConf.create()
                        cfg.transform.train = OmegaConf.create()
                        cfg.transform.train.transform = True
                        cfg.transform.train.spatial_subsampling = OmegaConf.create()
                        cfg.transform.train.spatial_subsampling.transform = cfg_parameters.transform.train.spatial_subsampling.transform
                        cfg.transform.train.spatial_subsampling.dataset_wide_random_offsets = cfg_parameters.transform.train.spatial_subsampling.dataset_wide_random_offsets
                        print(cfg.transform.train.spatial_subsampling)
                        print(hr_offset, vr_offset)
                        cfg.transform.train.spatial_subsampling.h_r_offset = hr_offset
                        cfg.transform.train.spatial_subsampling.v_r_offset = vr_offset
                        cfg.transform.train.spatial_subsampling.h_r = hr_vr[0]
                        cfg.transform.train.spatial_subsampling.v_r = hr_vr[1]
                        print(cfg.transform.train.spatial_subsampling)
                    
                        main(cfg, seed, tags = ["spatial_subsampling", f"hr_{hr_vr[0]}_vr_{hr_vr[1]}"] + tags)

    if hasattr(cfg_parameters['transform']['train'], 'spatial_subsampling_random') and cfg_parameters['transform']['train']['spatial_subsampling_random']['transform']:
        spatial_subsampling_hr = cfg_parameters['transform']['train']['spatial_subsampling_random']['h_r']
        spatial_subsampling_vr = cfg_parameters['transform']['train']['spatial_subsampling_random']['v_r']
       
        spatial_subsampling_combinations = list(zip(spatial_subsampling_hr, spatial_subsampling_vr))
        cfg.transform = OmegaConf.create()
        cfg.transform.train = OmegaConf.create()
        cfg.transform.train.transform = True
        cfg.transform.train.spatial_subsampling_random = cfg_parameters.transform.train.spatial_subsampling_random
        for hr_vr in spatial_subsampling_combinations:
            cfg.transform.train.spatial_subsampling_random.h_r = hr_vr[0]
            cfg.transform.train.spatial_subsampling_random.v_r = hr_vr[1]
            for seed in seeds:
                main(cfg, seed, tags = ["spatial_subsampling_random", f"hr_{hr_vr[0]}_vr_{hr_vr[1]}"] + tags)

                
    if hasattr(cfg_parameters['transform']['train'], 'temporal_subsampling') and cfg_parameters['transform']['train']['temporal_subsampling']['transform']:
        temporal_subsampling_ratio = cfg_parameters['transform']['train']['temporal_subsampling']['subsampling_ratio']
        
        for ratio in temporal_subsampling_ratio:
            for seed in seeds:
                if "dataset_wide_random_time_offset" in cfg_parameters['transform']['train']['temporal_subsampling'] and cfg_parameters['transform']['train']['temporal_subsampling']['dataset_wide_random_time_offset']:
                    rng = np.random.default_rng(seed)
                    temporal_subsampling_time_offset_coefficient = [rng.random()]
                else:
                    temporal_subsampling_time_offset_coefficient = [0]
                    if ('time_offset_coefficient' in cfg_parameters['transform']['train']['temporal_subsampling']) and cfg_parameters['transform']['train']['temporal_subsampling']['time_offset_coefficient']:
                        temporal_subsampling_time_offset_coefficient = cfg_parameters['transform']['train']['temporal_subsampling']['time_offset_coefficient']
                for time_offset_coefficient in temporal_subsampling_time_offset_coefficient:
                    cfg.transform = OmegaConf.create()
                    cfg.transform.train = OmegaConf.create()
                    cfg.transform.train.transform = True
                    cfg.transform.train.temporal_subsampling = OmegaConf.create()
                    cfg.transform.train.temporal_subsampling.transform = cfg_parameters.transform.train.temporal_subsampling.transform
                    cfg.transform.train.temporal_subsampling.dataset_wide_random_time_offset = cfg_parameters.transform.train.temporal_subsampling.dataset_wide_random_time_offset
                    cfg.transform.train.temporal_subsampling.window_size = cfg_parameters.transform.train.temporal_subsampling.window_size
                    print(cfg.transform.train.temporal_subsampling)
                    print(time_offset_coefficient)
                    cfg.transform.train.temporal_subsampling.time_offset_coefficient = time_offset_coefficient
                    cfg.transform.train.temporal_subsampling.subsampling_ratio = ratio
                    print(cfg.transform.train.temporal_subsampling)
                  
                    main(cfg, seed, tags=["temporal_subsampling", f"temporal_subsampling_ratio_{ratio}"] + tags)

    if hasattr(cfg_parameters['transform']['train'], 'temporal_subsampling_random') and cfg_parameters['transform']['train']['temporal_subsampling_random']['transform']:
        temporal_subsampling_ratio = cfg_parameters['transform']['train']['temporal_subsampling_random']['subsampling_ratio']
       
        cfg.transform = OmegaConf.create()
        cfg.transform.train = OmegaConf.create()
        cfg.transform.train.transform = True
        cfg.transform.train.temporal_subsampling_random = OmegaConf.create()
        cfg.transform.train.temporal_subsampling_random.transform = cfg_parameters.transform.train.temporal_subsampling_random.transform
        cfg.transform.train.temporal_subsampling_random.window_size = cfg_parameters.transform.train.temporal_subsampling_random.window_size
        cfg.transform.train.temporal_subsampling_random.fixed_interval = cfg_parameters.transform.train.temporal_subsampling_random.fixed_interval
        for ratio in temporal_subsampling_ratio:
            cfg.transform.train.temporal_subsampling_random.subsampling_ratio = ratio
            for seed in seeds:
                main(cfg, seed, tags = ["temporal_subsampling_random", f"temporal_subsampling_ratio_{ratio}"] + tags)

    if hasattr(cfg_parameters['transform']['train'], 'spatiotemporal_filtering_subsampling') and cfg_parameters['transform']['train']['spatiotemporal_filtering_subsampling']['transform']:
        sampling_threshold = cfg_parameters['transform']['train']['spatiotemporal_filtering_subsampling']['sampling_threshold']
        cfg.transform = OmegaConf.create()
        cfg.transform.train = OmegaConf.create()
        cfg.transform.train.transform = True
        cfg.transform.train.spatiotemporal_filtering_subsampling = cfg_parameters.transform.train.spatiotemporal_filtering_subsampling
        for threshold in sampling_threshold:
            cfg.transform.train.spatiotemporal_filtering_subsampling.sampling_threshold = threshold
            for seed in seeds:
                main(cfg, seed, tags=["spatiotemporal_filtering_subsampling", f"sampling_threshold_{threshold}"] + tags)

    if hasattr(cfg_parameters['transform']['train'], 'tos_2DHarris_subsampling') and cfg_parameters['transform']['train']['tos_2DHarris_subsampling']['transform']:
        sampling_threshold = cfg_parameters['transform']['train']['tos_2DHarris_subsampling']['sampling_threshold']
        cfg.transform = OmegaConf.create()
        cfg.transform.train = OmegaConf.create()
        cfg.transform.train.transform = True
        cfg.transform.train.tos_2DHarris_subsampling = cfg_parameters.transform.train.tos_2DHarris_subsampling
        for threshold in sampling_threshold:
            cfg.transform.train.tos_2DHarris_subsampling.sampling_threshold = threshold
            for seed in seeds:
                main(cfg, seed, tags=["tos_2DHarris_subsampling", f"sampling_threshold_{threshold}"] + tags)

    if hasattr(cfg_parameters['transform']['train'], 'random_ratio_subsampling'):
        random_ratio_subsampling = cfg_parameters['transform']['train']['random_ratio_subsampling']
        cfg.transform = OmegaConf.create()
        cfg.transform.train = OmegaConf.create()
        cfg.transform.train.transform = True
        for ratio in random_ratio_subsampling:
            cfg.transform.train.random_ratio_subsampling = ratio
            for seed in seeds:
                main(cfg, seed, tags=["random_subsampling", f"random_ratio_subsampling_{ratio}"] + tags)
                
                
    if hasattr(cfg_parameters['pre_transform']['train'], 'baseline_event_count') and cfg_parameters['pre_transform']['train']['baseline_event_count']['transform']:
          
        baseline_event_count_hr = cfg_parameters['pre_transform']['train']['baseline_event_count']['h_r']
        baseline_event_count_vr = cfg_parameters['pre_transform']['train']['baseline_event_count']['v_r']
        baseline_event_count_threshold = cfg_parameters['pre_transform']['train']['baseline_event_count']['threshold']
        baseline_event_count_combinations = list(zip(baseline_event_count_hr, 
                                                     baseline_event_count_vr,
                                                     baseline_event_count_threshold))
        
        for hr_vr_thrsh in baseline_event_count_combinations:
            for seed in seeds:
                cfg.pre_transform = OmegaConf.create()
                cfg.pre_transform.train = OmegaConf.create()
                cfg.pre_transform.train.transform = True
                cfg.pre_transform.train.baseline_event_count = OmegaConf.create()
                cfg.pre_transform.train.baseline_event_count.transform = cfg_parameters.pre_transform.train.baseline_event_count.transform
                cfg.pre_transform.train.baseline_event_count.h_r = hr_vr_thrsh[0]
                cfg.pre_transform.train.baseline_event_count.v_r = hr_vr_thrsh[1]
                cfg.pre_transform.train.baseline_event_count.threshold = hr_vr_thrsh[2]
                print(cfg.pre_transform.train.baseline_event_count)
                main(cfg, seed, tags = ["baseline_event_count", f"hr_{hr_vr_thrsh[0]}_vr_{hr_vr_thrsh[1]}", f"thrsh_{hr_vr_thrsh[2]}"] + tags)