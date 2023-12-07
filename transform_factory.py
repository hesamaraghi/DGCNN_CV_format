import torch_geometric.transforms as T
from datatransforms.event_transforms import *

def factory(cfg):
      
      if cfg.transform:
            transform_list = []
            if cfg.filter_nodes is not None:
                  transform_list.append(FilterNodes(cfg))
            if cfg.thresh_quantile is not None:
                  transform_list.append(RemoveOutliers(cfg))
            if cfg.shift_and_flip.transform:
                  transform_list.append(ShiftAndFlip(cfg.shift_and_flip))
            if cfg.temporal_scale is not None:
                  transform_list.append(TemporalScaling(cfg))
            if cfg.spatial_centering:
                  transform_list.append(SpatialCentering())
            if cfg.num_events_per_sample is not None:
                  transform_list.append(T.FixedPoints(cfg.num_events_per_sample, replace = False, allow_duplicates = True))
            elif cfg.random_num_events_per_sample.transform:
                  transform_list.append(VaryingSamplingPoints((   cfg.random_num_events_per_sample.min_num_events, 
                                                                  cfg.random_num_events_per_sample.max_num_events), 
                                                                  replace = False, allow_duplicates = False)) # TODO: allow duplicates is set to False for now
            if cfg.random_flip is not None:
                  transform_list.append(T.RandomFlip(axis = 0, p = cfg.random_flip))
            if cfg.scale_limits is not None:
                  transform_list.append(SpatialScaling(cfg))
            if cfg.degree_limit is not None:
                  transform_list.append(T.RandomRotate(cfg.degree_limit, axis = 2))
            if cfg.radius_graph.transform:
                  transform_list.append(T.RadiusGraph(r = cfg.radius_graph.r,max_num_neighbors = cfg.radius_graph.max_num_neighbors))
            if cfg.radius_graph.add_edge_attr.transform:
                  transform_list.append(AddEdgeAttr(cfg.radius_graph.add_edge_attr))
            return T.Compose(transform_list)
      else:
            return None
      
