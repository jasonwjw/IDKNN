"""
Author: Haoxi Ran
Date: 05/10/2022
"""

import torch

#try:
#    from modules.pointops.functions.pointops import furthestsampling, gathering, ballquery, knnquery, \
#        grouping, interpolation, nearestneighbor
#except:
#    raise Exception('Failed to load pointops')


def square_distance(src, dst):
    """
    Calculate Squared distance between each two points.

    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx, cuda=False, is_group=False):
    if cuda:
        if is_group:
            points = grouping(points.transpose(1, 2).contiguous(), idx)
            return points.permute(0, 2, 3, 1).contiguous()
        else:
            points = gathering(points.transpose(1, 2).contiguous(), idx)
            return points.permute(0, 2, 1).contiguous()
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint, cuda=False):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]

    FLOPs:
        S * (3 + 3 + 2)
    """
    if cuda:
        if not xyz.is_contiguous():
            xyz = xyz.contiguous()
        return furthestsampling(xyz, npoint)
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz, debug=False, cuda=False):
    if cuda:
        if not xyz.is_contiguous():
            xyz = xyz.contiguous()
        if not new_xyz.is_contiguous():
            new_xyz = new_xyz.contiguous()
        return ballquery(radius, nsample, xyz, new_xyz)
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    if debug:
        num_miss = torch.sum(mask)
        num_over = torch.sum(torch.clamp(torch.sum(sqrdists < radius ** 2, dim=2) - nsample, min=0))
        return num_miss, num_over
    return group_idx


def query_knn_point(k, xyz, new_xyz, cuda=False):
    if cuda:
        if not xyz.is_contiguous():
            xyz = xyz.contiguous()
        if not new_xyz.is_contiguous():
            new_xyz = new_xyz.contiguous()
        return knnquery(k, xyz, new_xyz)
    dist = square_distance(new_xyz, xyz)
    group_idx = dist.sort(descending=False, dim=-1)[1][:, :, :k]
    return group_idx

def query_knn_point_wjw(d,k, xyz, new_xyz, cuda=False):
    if cuda:
        if not xyz.is_contiguous():
            xyz = xyz.contiguous()
        if not new_xyz.is_contiguous():
            new_xyz = new_xyz.contiguous()
        return knnquery(k, xyz, new_xyz)

    sqrdists = square_distance(new_xyz, xyz)  # b,s,n
    w = k // d
    feature_distance, idxall = sqrdists.topk(k=k * d, dim=-1,largest=False,sorted=True)  # b,n,k*d
    if d == 1:  # if w > 1, need unsqueeze(-1) to keep dim
        retidx = idxall[:, :, :k]  # b,n,k
        retdis = feature_distance[:, :, :k]
    elif d == 2:
        retidx = torch.cat((idxall[:, :, :w], idxall[:, :, w + 1:w + 2 * (k - w) + 1:2]), dim=-1)  # b, n,k
        retdis = torch.cat((feature_distance[:, :, :w], feature_distance[:, :, w + 1:w + 2 * (k - w) + 1:2]),
                           dim=-1)  # b, n,k
    elif d == 3:
        retidx = torch.cat(
            (idxall[:, :, :w], idxall[:, :, w + 1:w * 3:2], idxall[:, :, w * 3 + 2:(k - 2 * w) * 3 + 3 * w + 1:3]),
            dim=-1)
        retdis = torch.cat(
            (feature_distance[:, :, :w], feature_distance[:, :, w + 1:w * 3:2],
             feature_distance[:, :, w * 3 + 2:(k - 2 * w) * 3 + 3 * w + 1:3]),
            dim=-1)
    elif d == 4:
        retidx = torch.cat((idxall[:, :, :w], idxall[:, :, w + 1:w * 3:2], idxall[:, :, 3 * w + 2: 6 * w:3],
                            idxall[:, :, 6 * w + 3:(k - 3 * w) * 4 + 6 * w + 1:4]), dim=-1)
        retdis = torch.cat((feature_distance[:, :, :w], feature_distance[:, :, w + 1:w * 3:2],
                            feature_distance[:, :, 3 * w + 2: 6 * w:3],
                            feature_distance[:, :, 6 * w + 3:(k - 3 * w) * 4 + 6 * w + 1:4]), dim=-1)
    elif d == 5:
        retidx = torch.cat((idxall[:, :, :w], idxall[:, :, w + 1:w * 3:2], idxall[:, :, 3 * w + 2: 6 * w:3],
                            idxall[:, :, 6 * w + 3: w * 10:4], idxall[:, :, 10 * w + 4:5 * (k - 4 * w) + 10 * w + 1:5]),
                           dim=-1)
        retdis = torch.cat((feature_distance[:, :, :w], feature_distance[:, :, w + 1:w * 3:2],
                            feature_distance[:, :, 3 * w + 2: 6 * w:3],
                            feature_distance[:, :, 6 * w + 3: w * 10:4],
                            feature_distance[:, :, 10 * w + 4:5 * (k - 4 * w) + 10 * w + 1:5]),
                           dim=-1)
    return retidx





def sample(nsample, feature, cuda=False):
    feature = feature.permute(0, 2, 1)
    xyz = feature[:, :, :3]

    fps_idx = farthest_point_sample(xyz, nsample, cuda=cuda)  # [B, npoint, C]
    torch.cuda.empty_cache()
    feature = index_points(feature, fps_idx, cuda=cuda, is_group=False)
    torch.cuda.empty_cache()
    feature = feature.permute(0, 2, 1)

    return feature
