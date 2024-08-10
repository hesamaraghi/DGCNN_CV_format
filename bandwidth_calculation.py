# Package imports, from conda or pip
import os
os.environ['WANDB__SERVICE_WAIT'] = "1000"
os.environ["WANDB_INIT_TIMEOUT"] = "300"
import wandb
import glob 
import torch
from omegaconf import OmegaConf
import numpy as np
import time

# Imports of own files
from graph_data_module import GraphDataModule


def main(cfg_path: str = None):
    # Load defaults and overwrite by command-line arguments

    cmd_cfg = OmegaConf.from_cli()
    
    if hasattr(cmd_cfg, 'cfg_path') and cmd_cfg.cfg_path is not None:
        cfg_path = cmd_cfg.cfg_path
        print(f"cfg_path is loaded from command line: cfg_path={cfg_path}")
    else:
        if not cfg_path:
            raise ValueError("cfg_path is not provided")
            
    cfg = OmegaConf.load(cfg_path)
    cfg.cfg_path = cfg_path
    
    cfg = OmegaConf.merge(cfg, cmd_cfg)
    cfg_bare = OmegaConf.load("config_bare.yaml")
    cfg = OmegaConf.merge(cfg_bare,cfg)
    print(OmegaConf.to_yaml(cfg), flush=True)

    # Create datasets using factory pattern
    
    cfg.dataset.in_memory = False
    start = time.time()
    gdm = GraphDataModule(cfg)
    print(f"Time to create GraphDataModule: {time.time()-start}", flush=True)
    cfg.dataset.num_classes = gdm.num_classes

    # Set cache dir to W&B logging directory
    os.environ["WANDB_CACHE_DIR"] = os.path.join(cfg.wandb.dir, 'cache')

    
    # torch.set_num_threads(1)
    print(f"Number of threads: {torch.get_num_threads()}")
    if cfg.dataset.num_workers > 0:
        torch.set_num_threads(cfg.dataset.num_workers)
        print(f"Number of threads is set to: {torch.get_num_threads()}")

    if hasattr(cfg.wandb, 'offline') and cfg.wandb.offline:
        wandb_mode = 'offline'
    else:
        wandb_mode = 'online'
        
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_object(cfg),
        dir=cfg.wandb.dir,
        mode=wandb_mode,
        name=cfg.wandb.experiment_name + '_bandwidth')
    
    dataloader_split = {
        'train':gdm.train_dataloader(),
        'val':gdm.val_dataloader()[0],
        'test':gdm.test_dataloader()[0]
    }
    num_event_dict = {}
    duration_dict = {}
    bandwidth_dict = {}
    for split in dataloader_split.keys(): 
        num_event_list = []
        duration_list = []
        bandwidth_list = [] 
        for batch in dataloader_split[split]:
            for batch_number in range(len(batch.y)):
                data_pos = batch.pos[batch.batch == batch_number,:]
                num_event = data_pos.shape[0]
                if num_event < 15:
                    continue
                duration = (data_pos[:,2].max() - data_pos[:,2].min()) / 1e3
                bandwidth = num_event/duration * 1e3
                num_event_list.append(num_event)
                duration_list.append(duration)
                bandwidth_list.append(bandwidth)
                wandb.log({
                    split + '/num_events': num_event,
                    split + '/duration': duration,
                    split + '/bandwidth': bandwidth
                    })
        num_event_dict[split] = num_event_list
        duration_dict[split] = duration_list
        bandwidth_dict[split] = bandwidth_list    
        wandb.log({
            # split + '/num_events_list': num_event_list,
            split + '/num_events_mean': np.mean(num_event_list), 
            split + '/num_events_std': np.std(num_event_list), 
            split + '/num_events_max': np.max(num_event_list), 
            split + '/num_events_min': np.min(num_event_list),
            # split + '/duration_list': duration_list,
            split + '/duration_mean': np.mean(duration_list), 
            split + '/duration_std': np.std(duration_list), 
            split + '/duration_max': np.max(duration_list), 
            split + '/duration_min': np.min(duration_list),
            # split + '/bandwidth_list': bandwidth_list,
            split + '/bandwidth_mean': np.mean(bandwidth_list), 
            split + '/bandwidth_std': np.std(bandwidth_list),
            split + '/bandwidth_max': np.max(bandwidth_list), 
            split + '/bandwidth_min': np.min(bandwidth_list)
        })

    all_num_events = np.concatenate([num_event_dict[split] for split in num_event_dict.keys()])
    all_duration = np.concatenate([duration_dict[split] for split in duration_dict.keys()])
    all_bandwidth = np.concatenate([bandwidth_dict[split] for split in bandwidth_dict.keys()])
    
    wandb.log({
        'all/num_events_mean': np.mean(all_num_events),
        'all/num_events_std': np.std(all_num_events),
        'all/num_events_max': np.max(all_num_events),
        'all/num_events_min': np.min(all_num_events),
        'all/duration_mean': np.mean(all_duration),
        'all/duration_std': np.std(all_duration),
        'all/duration_max': np.max(all_duration),
        'all/duration_min': np.min(all_duration),
        'all/bandwidth_mean': np.mean(all_bandwidth),
        'all/bandwidth_std': np.std(all_bandwidth),
        'all/bandwidth_max': np.max(all_bandwidth),
        'all/bandwidth_min': np.min(all_bandwidth)
    })
    
    wandb.finish()
        
if __name__ == '__main__':
    main()