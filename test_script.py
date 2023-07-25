# %%
from glob import glob
import os.path as osp
from collections import OrderedDict

import wandb
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F

import model_factory
import dataset_factory


# %%
run = wandb.init()

# %%
exprement_number = 'ewrilqhd'
batch_number = 'best'
path_files = glob(osp.join('wandb','*-'+exprement_number,'files','config.yaml'))
cfg = OmegaConf.load(path_files[0])
cfg.dataset = cfg.dataset.value
cfg.model = cfg.model.value
cfg.optimize = cfg.optimize.value
cfg.seed = cfg.seed.value
cfg.train = cfg.train.value
cfg.transform = cfg.transform.value
cfg.wandb = cfg.wandb.value
path_files = glob(osp.join(cfg.wandb.project,exprement_number,'checkpoints','*'))
pl_ckpt_file = path_files[0]
print(pl_ckpt_file)

# %%
artifact = run.use_artifact(osp.join(cfg.wandb.entity,cfg.wandb.project,f'model-{exprement_number}:{batch_number}'), type='model')
artifact_dir = artifact.download()
wandb_ckpt_file = osp.join(artifact_dir,'model.ckpt')
print(wandb_ckpt_file)

# %%
checkpoint = torch.load(wandb_ckpt_file)
ordered_list = [(a[6:],b) for a , b in checkpoint['state_dict'].items()]
stated_dict_wandb = OrderedDict(ordered_list)

checkpoint = torch.load(pl_ckpt_file)
ordered_list = [(a[6:],b) for a , b in checkpoint['state_dict'].items()]
stated_dict_pl = OrderedDict(ordered_list)

is_same = all([torch.allclose(v,stated_dict_pl[c]) for c,v in stated_dict_wandb.items()])
print(f'Are pl and wandb the same? {is_same}')


# %%
loaders = dataset_factory.factory(cfg)
train_dataset_loader, val_dataset_loader, test_dataset_loader = loaders


# %%

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg.model.name = "DGCNN"
model_2knn = model_factory.factory(cfg).to(device)
cfg.model.name = "DGCNN2"
model_1knn = model_factory.factory(cfg).to(device)

# %%
def test(loader, model):
    model.eval()
    all_pred = []
    all_true = []
    correct = 0
    total_loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
            pred = out.max(dim=1)[1]
            all_pred.append(pred)
            all_true.append(data.y)
            loss = F.nll_loss(out, data.y)
            correct += pred.eq(data.y).sum().item()
            total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset), correct / len(loader.dataset), all_pred, all_true

# %%
model_2knn.load_state_dict(stated_dict_pl)
model_1knn.load_state_dict(stated_dict_pl)

if not is_same:
    model_2knn.state_dict(stated_dict_wandb)
    model_1knn.state_dict(stated_dict_wandb)

# %%
for _ in range(5):
    perf = test(test_dataset_loader, model_2knn)
    print(perf[0], perf[1])
