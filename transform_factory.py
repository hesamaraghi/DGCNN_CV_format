import torch_geometric.transforms as T
from datatransforms.event_transforms import TemporalScaling, RemoveOutliers, SpatialCentering, SpatialScaling, AddEdgeAttr

def factory(cfg):
      
      if cfg.transform is not None:
            transform_list = []
            if cfg.temporal_scale is not None:
                  transform_list.append(TemporalScaling(cfg))
            if cfg.spatial_centering:
                  transform_list.append(SpatialCentering())
            if cfg.thresh_quantile is not None:
                  transform_list.append(RemoveOutliers(cfg))
            if cfg.num_events_per_sample is not None:
                  transform_list.append(T.FixedPoints(cfg.num_events_per_sample, replace = False, allow_duplicates = True))
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
      
