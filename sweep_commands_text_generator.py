import os
from omegaconf import OmegaConf
import wandb

def main():

    run = wandb.run if wandb.run else wandb.init()
    cfg = OmegaConf.load(run.config.cfg_path)
    cfg = OmegaConf.merge(cfg,OmegaConf.structured(vars(run.config)["_items"]))
    cfg_bare = OmegaConf.load("config_bare.yaml")
    cfg = OmegaConf.merge(cfg_bare,cfg)
    
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

if __name__ == '__main__':
    
    cmd_cfg = OmegaConf.from_cli()
    agent_count = getattr(cmd_cfg, "agent_count", 1)
    if hasattr(cmd_cfg, "sweep_id"):
        sweep_id = cmd_cfg.sweep_id
    else:
        raise ValueError("sweep_id not found in args")
    
    os.environ["WANDB_CACHE_DIR"] = os.path.join('.', 'cache')
    entity, project, _ = sweep_id.split("/")
    
    run_id = wandb.wandb.util.generate_id(9)
    wandb.agent(sweep_id, function=main, count=agent_count)
