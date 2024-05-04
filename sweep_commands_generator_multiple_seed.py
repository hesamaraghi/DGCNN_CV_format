import os
from omegaconf import OmegaConf
import random
import wandb
from utils.config_utils import show_cfg

def extract_parameter_values(config_list):
    sub_config = OmegaConf.create()
    for key, value in config_list.items():
        if "parameters" in value: 
                sub_config[key] = extract_parameter_values(value["parameters"])
        elif "values" in value:
                sub_config[key] = random.choice(value['values'])
        else:
            sub_config[key] = value
    return sub_config

def main(config = None, seed = None, num_events = None):

    cfg = OmegaConf.load(config.cfg_path)
    cfg = OmegaConf.merge(cfg,config)
    cfg_bare = OmegaConf.load("config_bare.yaml")
    cfg = OmegaConf.merge(cfg_bare,cfg)
    
    cfg.seed = seed
    cfg.transform.train.num_events_per_sample = num_events
    
    run = wandb.run if wandb.run else wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, reinit=True, tags=[f"seed_{seed}", f"num_events_{num_events}"])
    
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
    agent_count = getattr(cmd_cfg, "count", 1)
  
    if hasattr(cmd_cfg, "sweep_config"):
        sweep_config = OmegaConf.load(cmd_cfg.sweep_config)
    else:
        raise ValueError("sweep_config not found in args")
    if hasattr(cmd_cfg, "seed_list"):
        seed_list = cmd_cfg.seed_list
    else:
        raise ValueError("seed_list not found in args")
    if hasattr(cmd_cfg, "num_events_list"):
        num_events_list = cmd_cfg.num_events_list
    else:
        raise ValueError("num_events_list not found in args")

    sweep_id = sweep_config.name
    
    # sweep_config
    for _ in range(agent_count):
        cfg = extract_parameter_values(sweep_config.parameters)
        cfg.wandb = OmegaConf.create()
        cfg.wandb.entity = sweep_config.entity
        cfg.wandb.project = sweep_config.project
        for seed in seed_list:
            for num_events in num_events_list:
                main(cfg, seed, num_events)
                    
    
