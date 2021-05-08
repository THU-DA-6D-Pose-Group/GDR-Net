# https://github.com/NVlabs/latentfusion/blob/master/latentfusion/three/utils.py
import torch
from torch.nn import functional as F


def farthest_points(
    data,
    n_clusters: int,
    dist_func=F.pairwise_distance,
    return_center_indexes=True,
    return_distances=False,
    verbose=False,
    init_center=True,
):
    """Performs farthest point sampling on data points.

    Args:
      data (torch.tensor): data points.
      n_clusters (int): number of clusters.
      dist_func (Callable): distance function that is used to compare two data points.
      return_center_indexes (bool): if True, returns the indexes of the center of clusters.
      return_distances (bool): if True, return distances of each point from centers.
    Returns clusters, [centers, distances]:
      clusters (torch.tensor): the cluster index for each element in data.
      centers (torch.tensor): the integer index of each center.
      distances (torch.tensor): closest distances of each point to any of the cluster centers.
    """
    if n_clusters >= data.shape[0]:
        if return_center_indexes:
            return (torch.arange(data.shape[0], dtype=torch.long), torch.arange(data.shape[0], dtype=torch.long))

        return torch.arange(data.shape[0], dtype=torch.long)

    clusters = torch.full((data.shape[0],), fill_value=-1, dtype=torch.long)
    centers = torch.zeros(n_clusters, dtype=torch.long)

    if init_center:
        broadcasted_data = torch.mean(data, 0, keepdim=True).expand(data.shape[0], -1)
        distances = dist_func(broadcasted_data, data)
    else:
        distances = torch.full((data.shape[0],), fill_value=1e7, dtype=torch.float32)

    for i in range(n_clusters):
        center_idx = torch.argmax(distances)
        centers[i] = center_idx

        broadcasted_data = data[center_idx].unsqueeze(0).expand(data.shape[0], -1)
        new_distances = dist_func(broadcasted_data, data)
        distances = torch.min(distances, new_distances)
        clusters[distances == new_distances] = i
        if verbose:
            print("farthest points max distance : {}".format(torch.max(distances)))

    if return_center_indexes:
        if return_distances:
            return clusters, centers, distances
        return clusters, centers

    return clusters


def get_fps_and_center_torch(points, num_fps: int, init_center=True, dist_func=F.pairwise_distance):
    center = torch.mean(points, 0, keepdim=True)
    _, fps_inds = farthest_points(
        points, n_clusters=num_fps, dist_func=dist_func, return_center_indexes=True, init_center=init_center
    )
    fps_pts = points[fps_inds]
    return torch.cat([fps_pts, center], dim=0)
