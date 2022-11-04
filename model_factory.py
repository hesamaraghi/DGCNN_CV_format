import models


def factory(cfg):
    if cfg.model.name == 'DGCNN':
        return models.NetTwoKNN(cfg)
    if cfg.model.name == 'DGCNN2':
        return models.NetOnlyFirstKNN(cfg)    
    else:
        raise NotImplementedError(f"Model {cfg.model.name}")
