import torch_geometric.transforms as T
from datatransforms.event_transforms import *
from omegaconf import OmegaConf

def factory(cfg):
      cfg_dict = OmegaConf.to_object(cfg)
      if 'transform' in cfg_dict and cfg.transform:
            transform_list = []
            if 'filter_nodes' in cfg_dict and cfg.filter_nodes is not None:
                  transform_list.append(FilterNodes(cfg))
            if 'thresh_quantile' in cfg_dict and cfg.thresh_quantile is not None:
                  transform_list.append(RemoveOutliers(cfg))
            if 'shift_and_flip' in cfg_dict and cfg.shift_and_flip.transform:
                  transform_list.append(ShiftAndFlip(cfg.shift_and_flip))
            if 'temporal_scale' in cfg_dict and cfg.temporal_scale is not None:
                  transform_list.append(TemporalScaling(cfg))
            if 'temporal_quantization' in cfg_dict and cfg.temporal_quantization is not None:
                  transform_list.append(TemporalQuantization(cfg))
            if 'spatial_centering' in cfg_dict and cfg.spatial_centering:
                  transform_list.append(SpatialCentering())
            if 'spatial_subsampling' in cfg_dict and cfg.spatial_subsampling.transform:
                  transform_list.append(SpatialSubsampling((cfg.spatial_subsampling.h_r, cfg.spatial_subsampling.v_r)))
            if 'temporal_subsampling' in cfg_dict and cfg.temporal_subsampling.transform:
                  transform_list.append(TemporalSubsampling(cfg.temporal_subsampling.subsampling_ratio, cfg.temporal_subsampling.window_size))
            if 'random_ratio_subsampling' in cfg_dict and cfg.random_ratio_subsampling is not None:
                  transform_list.append(DropEventRandomly(cfg.random_ratio_subsampling))
            if 'num_events_per_sample' in cfg_dict and cfg.num_events_per_sample is not None:
                  if "fixed_sampling" in cfg_dict and cfg.fixed_sampling.transform is True:
                        if "seed_str" in cfg_dict["fixed_sampling"] and cfg.fixed_sampling.seed_str is not None:
                              transform_list.append(FixedSubsampling(cfg, replace = False, allow_duplicates = False))
                  else:
                        transform_list.append(T.FixedPoints(cfg.num_events_per_sample, replace = False, allow_duplicates = True))
            elif 'random_num_events_per_sample' in cfg_dict and cfg.random_num_events_per_sample.transform:
                  transform_list.append(VaryingSamplingPoints((   cfg.random_num_events_per_sample.min_num_events, 
                                                                  cfg.random_num_events_per_sample.max_num_events), 
                                                                  replace = False, 
                                                                  allow_duplicates = False,
                                                                  inverse_sampling = cfg.random_num_events_per_sample.inverse_sampling)) # TODO: allow duplicates is set to False for now
            if 'random_flip' in cfg_dict and cfg.random_flip is not None:
                  transform_list.append(T.RandomFlip(axis = 0, p = cfg.random_flip))
            if 'scale_limits' in cfg_dict and cfg.scale_limits is not None:
                  transform_list.append(SpatialScaling(cfg))
            if 'degree_limit' in cfg_dict and cfg.degree_limit is not None:
                  transform_list.append(T.RandomRotate(cfg.degree_limit, axis = 2))
            if 'radius_graph' in cfg_dict and cfg.radius_graph.transform:
                  transform_list.append(T.RadiusGraph(r = cfg.radius_graph.r,max_num_neighbors = cfg.radius_graph.max_num_neighbors))
            if 'radius_graph' in cfg_dict and cfg.radius_graph.add_edge_attr.transform:
                  transform_list.append(AddEdgeAttr(cfg.radius_graph.add_edge_attr))
            return T.Compose(transform_list)
      else:
            return None
      