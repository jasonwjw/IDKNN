#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM

Modified by 
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@Time: 2020/3/9 9:32 PM
"""

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from net_cls import Transformer3dAttention,idx_pt
from pointnet_util import sampling_by_RSF_distance, index_points, farthest_point_sample, sampling_by_RF_distance, \
    sampling_by_SF_distance, sampling_by_RS_distance
from torch import einsum


class LinearGeneral(nn.Module):
    def __init__(self, in_dim=(768,), feat_dim=(12, 64)):
        super(LinearGeneral, self).__init__()

        self.weight = nn.Parameter(torch.randn(*in_dim, *feat_dim))
        self.bias = nn.Parameter(torch.zeros(*feat_dim))

    def forward(self, x, dims):
        a = torch.tensordot(x, self.weight, dims=dims) + self.bias
        return a


class SelfAttention(nn.Module):
    def __init__(self, in_dim, heads=8, dropout_rate=0.1):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.head_dim = in_dim // heads
        self.scale = self.head_dim ** 0.5

        self.query = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.key = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.value = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.out = LinearGeneral((self.heads, self.head_dim), (in_dim,))

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        # x: b,n,c
        b, n, _ = x.shape

        q = self.query(x, dims=([2], [0]))
        k = self.key(x, dims=([2], [0]))
        v = self.value(x, dims=([2], [0]))

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = out.permute(0, 2, 1, 3)

        out = self.out(out, dims=([2, 3], [0, 1]))

        return out


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # 选择距离最小的，直接先取负数，再取topK

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx
def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

#3个方位的极坐标的表达方式的叠加
def  LocalFeatureRepresentaion_polar(xyz,knn_points,return_dis=True):
    # xyz:b,n,3
    # knn_points:b,n,k,3
    b,n,k,_ = knn_points.shape
    knn_points_norm = knn_points - xyz.unsqueeze(-2) # b,n,k,3 去心之后,相对位置
    local_dis = torch.sqrt(torch.sum(knn_points_norm **2 ,dim=-1)) # b,n,k
    local_x = knn_points_norm[:,:,:,0] # b,n,k
    local_y = knn_points_norm[:,:,:,1]# b,n,k
    local_z = knn_points_norm[:,:,:,2] # b,n,k
    local_xy = torch.sqrt(local_x ** 2 + local_y ** 2)  # b,n, k
    local_xz = torch.sqrt(local_x ** 2 + local_z ** 2) # b,n.k
    local_yz = torch.sqrt(local_y ** 2 + local_z ** 2) # b,n,k

    # center_mass = torch.mean(knn_points_norm,dim=-2)  # b,n,3
    # z_fi_center = torch.atan2(center_mass[:,:,1], center_mass[:,:,0]) # b,n
    # z_ceta_center = torch.atan2(center_mass[:,:,2], torch.sqrt(center_mass[:,:,0] ** 2 + center_mass[:,:,1] ** 2))
    #
    # y_fi_center = torch.atan2(center_mass[:,:,0],center_mass[:,:,2])
    # y_ceta_center = torch.atan2(center_mass[:,:,1],torch.sqrt(center_mass[:,:,0] ** 2 + center_mass[:,:,2] ** 2))
    #
    # x_fi_center = torch.atan2(center_mass[:,:,2],center_mass[:,:,1])
    # x_ceta_center = torch.atan2(center_mass[:,:,0], torch.sqrt(center_mass[:,:,1] ** 2 + center_mass[:,:,2] ** 2))

    # z- invariant features
    z_fi = torch.atan2(local_y,local_x) # b,n,k
    z_ceta = torch.atan2(local_z,local_xy) # b,n,k
    # z_fi = z_fi - z_fi_center.unsqueeze(-1)
    # z_ceta = z_ceta - z_ceta_center.unsqueeze(-1)

    # y-invariant
    y_fi = torch.atan2(local_z,local_x) # b,n,k
    y_ceta = torch.atan2(local_y,local_xz) # b,n,k
    # y_fi = y_fi - y_fi_center.unsqueeze(-1)
    # y_ceta = y_ceta - y_ceta_center.unsqueeze(-1)

    #x-invariant
    x_fi = torch.atan2(local_z,local_y) #b,n,k
    x_ceta = torch.atan2(local_x,local_yz) # b, n, k
    # x_fi = x_fi - x_fi_center.unsqueeze(-1)
    # x_ceta = x_ceta - x_ceta_center.unsqueeze(-1)
    #source pos
    xyzlift = xyz.unsqueeze(-2).repeat(1,1,k,1)  # ,b,n,k,3
    #taerget pos   knn_points  absolute pos
    if return_dis:
        local_feature = torch.cat((local_dis.unsqueeze(-1),z_fi.unsqueeze(-1),z_ceta.unsqueeze(-1),y_fi.unsqueeze(-1)
                               ,y_ceta.unsqueeze(-1),x_fi.unsqueeze(-1),x_ceta.unsqueeze(-1),knn_points_norm,xyzlift),dim = -1)   # b,n,k,10+3
    else:
        local_feature = torch.cat(( z_fi.unsqueeze(-1), z_ceta.unsqueeze(-1), y_fi.unsqueeze(-1)  ,  y_ceta.unsqueeze(-1),x_fi.unsqueeze(-1),  x_ceta.unsqueeze(-1)
                                   ,  knn_points_norm,
                                   xyzlift), dim=-1)  # b,n,k,12


    return local_feature,local_dis.unsqueeze(-1),knn_points_norm # b,n,k,1

#used in paper only contain source ,realative ,distance
def LocalFeatureRepresentaion_1(xyz,knn_points):
#xyz:b,n,3
#knn_points:b,n,k,3
    b,n,k,_ = knn_points.shape
    knn_points_norm = knn_points - xyz.unsqueeze(-2) # b,n,k,3
    xyz = xyz.view(b,n,-1,3).repeat(1,1,k,1) # b,n,k,3
    local_dis = torch.sqrt(torch.sum(knn_points_norm ** 2,dim=-1)).unsqueeze(-1) # b,n,k,1
    # local = torch.cat((xyz,knn_points,knn_points_norm,local_dis),dim=-1)  # b,n,k,10/
    local = torch.cat((xyz,knn_points_norm,local_dis),dim=-1) # b,n,k,7
    return local

def LocalFeatureRepresentaion_cylinder(xyz,knn_points,nsample,dila3 = False):
    """

    :param xyz: b,n,3
    :param knn_points: b,n,k,3
    :param nsample: k
    :param radius: r
    :param ball: use knn or ball query
    :return:
    """

    # knn_points = index_points(xyz,knn_index) # b,n,k,3
    knn_points_norm = knn_points - xyz.unsqueeze(-2) # b,n,k,3 去心之后,相对位置
    local_dis = torch.sqrt(torch.sum(knn_points_norm **2 ,dim=-1)) # b,n,k

    # center_mass = torch.mean(knn_points_norm,dim = -2)# b,n,3
    # z_ceta_center = torch.atan2(center_mass[:,:,1],center_mass[:,:,0]) # b,n
    # y_ceta_center = torch.atan2(center_mass[:,:,0],center_mass[:,:,2])
    # x_ceta_center = torch.atan2(center_mass[:,:,2],center_mass[:,:,1])

    #z_invarient
    z_z = knn_points_norm[:,:,:,2] # b,n,k
    z_r = torch.sqrt(knn_points_norm[:,:,:,0] ** 2 + knn_points_norm[:,:,:,1] ** 2) # b,n,k
    z_ceta = torch.atan2(knn_points_norm[:,:,:,1],knn_points_norm[:,:,:,0])
    # z_ceta = z_ceta - z_ceta_center.unsqueeze(-1) # b,n,k

    # y-invariant
    y_y = knn_points_norm[:,:,:,1] # b,n,k
    y_r = torch.sqrt(knn_points_norm[:,:,:,0] ** 2 + knn_points_norm[:,:,:,2] ** 2)
    y_ceta = torch.atan2(knn_points_norm[:,:,:,0],knn_points_norm[:,:,:,2])
    # y_ceta = y_ceta  - y_ceta_center.unsqueeze(-1) # b,n,k

    # x_invariant
    x_x = knn_points_norm[:,:,:,0] # b,n,k
    x_r = torch.sqrt(knn_points_norm[:,:,:,1] ** 2 + knn_points_norm[:,:,:,2] ** 2)
    x_ceta = torch.atan2(knn_points_norm[:,:,:,2],knn_points_norm[:,:,:,1])
    # x_ceta = x_ceta - x_ceta_center.unsqueeze(-1)  # b,n,k
    if dila3:
        xyz_lift = xyz.unsqueeze(-2).repeat(1,1,3*nsample,1)  # b,n,k,3
    else:
        xyz_lift = xyz.unsqueeze(-2).repeat(1, 1, nsample, 1)  # b,n,k,3

    local_features = torch.cat((z_z.unsqueeze(-1),y_y.unsqueeze(-1),x_x.unsqueeze(-1),z_r.unsqueeze(-1),z_ceta.unsqueeze(-1),y_r.unsqueeze(-1),y_ceta.unsqueeze(-1)
                                ,x_r.unsqueeze(-1),x_ceta.unsqueeze(-1),local_dis.unsqueeze(-1),xyz_lift),dim = -1)  # b,n,k,10+3

    return local_features


def my_get_graph_feature(conv_op,x,xyz,d,k,dim9 = False,idx = None):
    """

    :param x: point features: b,c,n
    :param xyz: b,3,n
    :param d:  dilation
    :param k:  k = 20
    :param dim9: false
    :return: b,2c,n
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            if len(d) == 3:
                idx = knn_with_dilation_multiple(x, k=k, d=d)  # (batch_size, num_points, k)
            else:
                idx = knn_with_dilation(x, k=k, d=d[0])
        else:
            if len(d) == 3:
                idx = knn_with_dilation_multiple(x[:, 6:], k=k, d=d)  # (batch_size, num_points, k)
            else:
                idx = knn_with_dilation(x[:, 6:], k=k, d=d[0])
    idx_dila = idx
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)  # flatten
    _, num_dims, _ = x.size()
    x = x.transpose(2,1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    # local feature------------------------
    feature = x.view(batch_size * num_points, -1)[idx, :]
    if len(d) == 3:
        feature = feature.view(batch_size, num_points, 3*k, num_dims)  # b,n,3k,c
        x_repeat = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, 3*k, 1)  # b,n,3k,c
    else:
        feature = feature.view(batch_size, num_points, k, num_dims)  # b,n,k,c
        x_repeat = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # b,n,k,c
    corr = F.relu(torch.matmul(x_repeat,feature.permute(0,1,3,2)))[:,:,0,:].unsqueeze(-2)   #b,n,k,c * b,n,c,k--- b,n,k,k ---b,n,1,k
    corr = corr / corr.sum(dim=-1,keepdim=True)  # normalize
    corr_feats = torch.matmul(corr,feature).squeeze() # b,n,1,k * ,b,n,k,c ---- b,n,1,c----b,n,c
    corr_feats = corr_feats.permute(0,2,1)  #b,n,c- >b,c,n



    # local shape--------------------------
    xyz_k = idx_pt(xyz.permute(0,2,1),idx_dila)  #  b,n,3-- b,n,k----- b,n,k,3
    if len(d) == 3:
        local_cylinder = LocalFeatureRepresentaion_cylinder(xyz.permute(0,2,1),xyz_k,k,dila3=True)  # b,n,k,13
    else:
        local_cylinder = LocalFeatureRepresentaion_cylinder(xyz.permute(0, 2, 1), xyz_k, k, dila3=False)
    local_shape = conv_op(local_cylinder.permute(0,3,1,2)).permute(0,2,1,3).max(dim=-1)[0]  # b,c,n,k---  b,n,c,k  ---b,n,c
    local_shape = local_shape.permute(0,2,1) # b,n,c-- > b,c,n

    # feature = torch.cat((corr_feats,local_shape),dim=-1).permute(0,2,1)  #  b,n,2c -----b, 2c,n
    return corr_feats,local_shape

def my_get_graph_feature_after(conv_op,cylinder_fea,x,xyz,d,k,dim9 = False,idx = None):
    """

    :param x: point features: b,c,n
    :param cylinder_fea: b,c,n
    :param xyz: b,3,n
    :param d:  dilation
    :param k:  k = 20
    :param dim9: false
    :return: b,2c,n
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            if len(d) == 3:
                idx = knn_with_dilation_multiple(x, k=k, d=d)  # (batch_size, num_points, k)
            else:
                idx = knn_with_dilation(x, k=k, d=d[0])
        else:
            if len(d) == 3:
                idx = knn_with_dilation_multiple(x[:, 6:], k=k, d=d)  # (batch_size, num_points, k)
            else:
                idx = knn_with_dilation(x[:, 6:], k=k, d=d[0])
    idx_dila = idx
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)  # flatten
    _, num_dims, _ = x.size()
    x = x.transpose(2,1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    # local feature------------------------
    feature = x.view(batch_size * num_points, -1)[idx, :]
    if len(d) == 3:
        feature = feature.view(batch_size, num_points, 3*k, num_dims)  # b,n,3k,c
        x_repeat = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, 3*k, 1)  # b,n,3k,c
    else:
        feature = feature.view(batch_size, num_points, k, num_dims)  # b,n,k,c
        x_repeat = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # b,n,k,c
    corr = F.relu(torch.matmul(x_repeat,feature.permute(0,1,3,2)))[:,:,0,:].unsqueeze(-2)   #b,n,k,c * b,n,c,k--- b,n,k,k ---b,n,1,k
    corr = corr / corr.sum(dim=-1,keepdim=True)  # normalize
    corr_feats = torch.matmul(corr,feature).squeeze() # b,n,1,k * ,b,n,k,c ---- b,n,1,c----b,n,c
    corr_feats = corr_feats.permute(0,2,1)  #b,n,c- >b,c,n



    # local shape--------------------------
    local_cylinder = idx_pt(cylinder_fea.permute(0,2,1),idx_dila)  #  b,n,c,- b,n,k----- b,n,k,c

    local_shape = conv_op(local_cylinder.permute(0,3,1,2)).permute(0,2,1,3).max(dim=-1)[0]  # b,c,n,k---  b,n,c,k  ---b,n,c
    local_shape = local_shape.permute(0,2,1) # b,n,c-- > b,c,n

    # feature = torch.cat((corr_feats,local_shape),dim=-1).permute(0,2,1)  #  b,n,2c -----b, 2c,n
    return corr_feats,local_shape


def my_get_graph_feature_nocylinder(x,xyz,d,k,dim9 = False,idx = None):
    """

    :param x: point features: b,c,n
    :param cylinder_fea: b,c,n
    :param xyz: b,3,n
    :param d:  dilation
    :param k:  k = 20
    :param dim9: false
    :return: b,2c,n
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            if len(d) > 1:
                idx = knn_with_dilation_multiple(x, k=k, d=d)  # (batch_size, num_points, k)
            else:
                idx = knn_with_dilation(x, k=k, d=d[0])
        else:
            if len(d) > 1:
                idx = knn_with_dilation_multiple(x[:, 6:], k=k, d=d)  # (batch_size, num_points, k)
            else:
                idx = knn_with_dilation(x[:, 6:], k=k, d=d[0])
    idx_dila = idx
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)  # flatten
    _, num_dims, _ = x.size()
    x = x.transpose(2,1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    # local feature------------------------
    feature = x.view(batch_size * num_points, -1)[idx, :]
    if len(d) > 1:
        feature = feature.view(batch_size, num_points, len(d)*k, num_dims)  # b,n,3k,c
        x_repeat = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, 3*k, 1)  # b,n,3k,c
    else:
        feature = feature.view(batch_size, num_points, k, num_dims)  # b,n,k,c
        x_repeat = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # b,n,k,c
    #normalize projection
    corr = (torch.matmul(x_repeat,feature.permute(0,1,3,2)) / torch.sqrt(torch.sum(x_repeat ** 2, dim=-1,keepdim=True)))[:,:,0,:].unsqueeze(-2) #b,n,k,c * b,n,c,k--- b,n,k,k ---b,n,1,k
   # corr = F.relu(torch.matmul(x_repeat,feature.permute(0,1,3,2)))[:,:,0,:].unsqueeze(-2)   #b,n,k,c * b,n,c,k--- b,n,k,k ---b,n,1,k
   #  corr = corr / corr.sum(dim=-1,keepdim=True)  #l1 normalize
    # corr = torch.sigmoid(corr) - 0.5 #sigmoid normalize
    # corr = (torch.exp(corr) - 1) / (torch.sum(torch.exp(corr)-1 ,dim=-1,keepdim=True) + 1e-8 )#softmax normalize
    corr = F.softmax(corr,dim=-1)
    # print(corr.shape)
    corr_feats = torch.matmul(corr,feature).squeeze() # b,n,1,k * ,b,n,k,c ---- b,n,1,c----b,n,c
    corr_feats = corr_feats.permute(0,2,1)  #b,n,c- >b,c,n


    return corr_feats



class knn_with_dilation_transformer(nn.Module):
    def __init__(self, inchannel):
        super(knn_with_dilation_transformer, self).__init__()
        self.trans = PointTransformerBlock_scaler(dim=inchannel)
        self.conv1 = nn.Sequential(nn.Conv1d(inchannel, inchannel, 1, bias=False), nn.BatchNorm1d(inchannel),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(inchannel, 1, 1, bias=False), nn.BatchNorm1d(1))

    def forward(self, x, k, dmax=4):
        """

        :param x:b,c,n
        :param k: nsample
        :param d:  dilaiton
        :return:  b,n, k index
        """
        agg_feature = self.trans(x)  # b,c,n
        adaptative_thre = self.conv2(self.conv1(agg_feature)).squeeze()  # b,c,n --- b,1,n --- b,n
        adaptative_thre = torch.sigmoid(adaptative_thre)  # b,n
        metric = 4 * adaptative_thre + 0.5
        value1 = torch.where((metric >= 0.5) & (metric < 1.5), torch.full_like(metric, 1), torch.full_like(metric, 0))
        value2 = torch.where((metric >= 1.5) & (metric < 2.5), torch.full_like(metric, 2), torch.full_like(metric, 0))
        value3 = torch.where((metric >= 2.5) & (metric < 3.5), torch.full_like(metric, 3), torch.full_like(metric, 0))
        value4 = torch.where((metric >= 3.5) & (metric <= 4.5), torch.full_like(metric, 4), torch.full_like(metric, 0))
        value = value1 + value2 + value3 + value4  # b,n 每个点对应的dilation
        # assert 0 in value
        inner = -2 * torch.matmul(x.transpose(2, 1), x)  # b,n,n
        xx = torch.sum(x ** 2, dim=1, keepdim=True)  # b, 1, n
        pair_distance = -xx - inner - xx.transpose(2, 1)  # b, n,n  ascending
        idxall = pair_distance.topk(k=k * dmax, dim=-1)[1]  # b, n, k
        retidx = idx_with_dilation_veryslow(idxall, value, k)
        return retidx


def knn_with_dilation(x, k, d):
    """

    :param x:  b,c,n 可以是点的位置，也可以是点的特征
    :param k:  k nearest  points
    :param d:  dilation
    :return:   b, n, k :index
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # b,n,n
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # b, 1, n
    pair_distance = -xx - inner - xx.transpose(2, 1)  # b, n,n  ascending
    w = k // d
    assert w > 1
    feature_distance = pair_distance.topk(k=k * d, dim=-1)[0]  # b,n,k*d
    idxall = pair_distance.topk(k=k * d, dim=-1)[1]  # b, n, k*d
    if d == 1:  # if w > 1, need unsqueeze(-1) to keep dim
        retidx = idxall[:, :, :k]  # b,n,k
        retdis = feature_distance[:, :, :k]
    elif d == 2:
        retidx = torch.cat((idxall[:, :, :w], idxall[:, :, w + 1:w + 2 * (k - w) + 1:2]), dim=-1)  # b, n,k
        retdis = torch.cat((feature_distance[:, :, :w], feature_distance[:, :, w + 1:w + 2 * (k - w) + 1:2]), dim=-1)  # b, n,k
    elif d == 3:
        retidx = torch.cat(
            (idxall[:, :, :w], idxall[:, :, w + 1:w * 3:2], idxall[:, :, w * 3 + 2:(k - 2 * w) * 3 + 3 * w + 1:3]),
            dim=-1)
        retdis = torch.cat(
            (feature_distance[:, :, :w], feature_distance[:, :, w + 1:w * 3:2], feature_distance[:, :, w * 3 + 2:(k - 2 * w) * 3 + 3 * w + 1:3]),
            dim=-1)
    elif d == 4:
        retidx = torch.cat((idxall[:, :, :w], idxall[:, :, w + 1:w * 3:2], idxall[:, :, 3 * w + 2: 6 * w:3],
                            idxall[:, :, 6 * w + 3:(k - 3 * w) * 4 + 6 * w + 1:4]), dim=-1)
        retdis = torch.cat((feature_distance[:, :, :w], feature_distance[:, :, w + 1:w * 3:2], feature_distance[:, :, 3 * w + 2: 6 * w:3],
                            feature_distance[:, :, 6 * w + 3:(k - 3 * w) * 4 + 6 * w + 1:4]), dim=-1)
    elif d == 5:
        retidx = torch.cat((idxall[:, :, :w], idxall[:, :, w + 1:w * 3:2], idxall[:, :, 3 * w + 2: 6 * w:3],
                            idxall[:, :, 6 * w + 3: w * 10:4], idxall[:, :, 10 * w + 4:5 * (k - 4 * w) + 10 * w + 1:5]),
                           dim=-1)
        retdis = torch.cat((feature_distance[:, :, :w], feature_distance[:, :, w + 1:w * 3:2], feature_distance[:, :, 3 * w + 2: 6 * w:3],
                            feature_distance[:, :, 6 * w + 3: w * 10:4], feature_distance[:, :, 10 * w + 4:5 * (k - 4 * w) + 10 * w + 1:5]),
                           dim=-1)
    return retidx,retdis

def knn_with_dilation_src_dst(x_src,x_dst, k, d):
    """

    :param x:  b,c,n 可以是点的位置，也可以是点的特征
    :param k:  k nearest  points
    :param d:  dilation
    :return:   b, n, k :index
    """
    pair_distance = square_distance(x_src.permute(0,2,1),x_dst.permute(0,2,1)) * -1 # b,m,n    m < n
    w = k // d
    assert w > 1
    feature_distance = pair_distance.topk(k=k * d, dim=-1)[0]  # b,m,k*d
    idxall = pair_distance.topk(k=k * d, dim=-1)[1]  # b, m, k*d
    if d == 1:  # if w > 1, need unsqueeze(-1) to keep dim
        retidx = idxall[:, :, :k]  # b,n,k
        retdis = feature_distance[:, :, :k]
    elif d == 2:
        retidx = torch.cat((idxall[:, :, :w], idxall[:, :, w + 1:w + 2 * (k - w) + 1:2]), dim=-1)  # b, n,k
        retdis = torch.cat((feature_distance[:, :, :w], feature_distance[:, :, w + 1:w + 2 * (k - w) + 1:2]), dim=-1)  # b, n,k
    elif d == 3:
        retidx = torch.cat(
            (idxall[:, :, :w], idxall[:, :, w + 1:w * 3:2], idxall[:, :, w * 3 + 2:(k - 2 * w) * 3 + 3 * w + 1:3]),
            dim=-1)
        retdis = torch.cat(
            (feature_distance[:, :, :w], feature_distance[:, :, w + 1:w * 3:2], feature_distance[:, :, w * 3 + 2:(k - 2 * w) * 3 + 3 * w + 1:3]),
            dim=-1)
    elif d == 4:
        retidx = torch.cat((idxall[:, :, :w], idxall[:, :, w + 1:w * 3:2], idxall[:, :, 3 * w + 2: 6 * w:3],
                            idxall[:, :, 6 * w + 3:(k - 3 * w) * 4 + 6 * w + 1:4]), dim=-1)
        retdis = torch.cat((feature_distance[:, :, :w], feature_distance[:, :, w + 1:w * 3:2], feature_distance[:, :, 3 * w + 2: 6 * w:3],
                            feature_distance[:, :, 6 * w + 3:(k - 3 * w) * 4 + 6 * w + 1:4]), dim=-1)
    elif d == 5:
        retidx = torch.cat((idxall[:, :, :w], idxall[:, :, w + 1:w * 3:2], idxall[:, :, 3 * w + 2: 6 * w:3],
                            idxall[:, :, 6 * w + 3: w * 10:4], idxall[:, :, 10 * w + 4:5 * (k - 4 * w) + 10 * w + 1:5]),
                           dim=-1)
        retdis = torch.cat((feature_distance[:, :, :w], feature_distance[:, :, w + 1:w * 3:2], feature_distance[:, :, 3 * w + 2: 6 * w:3],
                            feature_distance[:, :, 6 * w + 3: w * 10:4], feature_distance[:, :, 10 * w + 4:5 * (k - 4 * w) + 10 * w + 1:5]),
                           dim=-1)
    return retidx,retdis


def knn_with_dilation_close(x, k, d):
    """

    :param x:  b,c,n 可以是点的位置，也可以是点的特征
    :param k:  k nearest  points
    :param d:  dilation
    :return:   b, n, k :index
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # b,n,n
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # b, 1, n
    pair_distance = -xx - inner - xx.transpose(2, 1)  # b, n,n  ascending
    w = k // d
    assert w > 1
    feature_distance = pair_distance.topk(k=k * d, dim=-1)[0]  # b,n,k*d
    idxall = pair_distance.topk(k=k * d, dim=-1)[1]  # b, n, k*d
    if d == 1:  # if w > 1, need unsqueeze(-1) to keep dim
        retidx = idxall[:, :, :k]  # b,n,k
    elif d == 2:
        retidx = torch.cat((idxall[:, :, :w], idxall[:, :, w:w + 2 * (k - w):2]), dim=-1)  # b, n,k
    elif d == 3:
        retidx = torch.cat((idxall[:, :, :w], idxall[:, :, w:w * 3:2], idxall[:, :, w * 3:(k - 2 * w) * 3 + 3 * w:3]),
                           dim=-1)
    elif d == 4:
        retidx = torch.cat((idxall[:, :, :w], idxall[:, :, w:w * 3:2], idxall[:, :, 3 * w: 6 * w:3],
                            idxall[:, :, 6 * w:(k - 3 * w) * 4 + 6 * w:4]), dim=-1)
    elif d == 5:
        retidx = torch.cat((idxall[:, :, :w], idxall[:, :, w:w * 3:2], idxall[:, :, 3 * w: 6 * w:3],
                            idxall[:, :, 6 * w: w * 10:4], idxall[:, :, 10 * w:5 * (k - 4 * w) + 10 * w:5]), dim=-1)
    return retidx


def idx_with_dilation_veryslow(idxall, bn, k):
    """

    :param idxall: B,N,M   index
    :param bn: B,N
    :param k:
    :return: B,N,K
    """
    device = idxall.device
    ret_N = torch.tensor([]).to(device)

    _, _, M = idxall.shape
    B, N = bn.shape
    for b in range(B):
        ret_B = torch.tensor([]).to(device)
        for n in range(N):
            d = bn[b, n]
            d = int(d)
            w = int(k // d)
            assert w > 1
            if d == 1:  # if w > 1, need unsqueeze(-1) to keep dim
                retidx = idxall[b, n][:k]  # k
            elif d == 2:
                retidx = torch.cat((idxall[b, n][:w], idxall[b, n][w + 1:w + 2 * (k - w) + 1:2]), dim=-1)  # b, n,k
            elif d == 3:
                retidx = torch.cat(
                    (idxall[b, n][:w], idxall[b, n][w + 1:w * 3:2],
                     idxall[b, n][w * 3 + 2:(k - 2 * w) * 3 + 3 * w + 1:3]),
                    dim=-1)
            elif d == 4:
                retidx = torch.cat((idxall[b, n][:w], idxall[b, n][w + 1:w * 3:2], idxall[b, n][3 * w + 2: 6 * w:3],
                                    idxall[b, n][6 * w + 3:(k - 3 * w) * 4 + 6 * w + 1:4]), dim=-1)
            elif d == 5:
                retidx = torch.cat((idxall[b, n][:w], idxall[b, n][w + 1:w * 3:2], idxall[b, n][3 * w + 2: 6 * w:3],
                                    idxall[b, n][6 * w + 3: w * 10:4],
                                    idxall[b, n][10 * w + 4:5 * (k - 4 * w) + 10 * w + 1:5]),
                                   dim=-1)
            retidx = retidx.unsqueeze(0)  # (1,k)
            ret_B = torch.cat((ret_B, retidx), dim=0)  # (n,k)
        ret_B = ret_B.unsqueeze(0)  # (1,n,k)
        ret_N = torch.cat((ret_N, ret_B), dim=0)

    return ret_N


def idx_with_dilation(idxall, k, d, w):
    assert w > 1
    if d == 1:  # if w > 1, need unsqueeze(-1) to keep dim
        retidx = idxall[:, :, :k]  # b,n,k
    elif d == 2:
        retidx = torch.cat((idxall[:, :, :w], idxall[:, :, w + 1:w + 2 * (k - w) + 1:2]), dim=-1)  # b, n,k
    elif d == 3:
        retidx = torch.cat(
            (idxall[:, :, :w], idxall[:, :, w + 1:w * 3:2], idxall[:, :, w * 3 + 2:(k - 2 * w) * 3 + 3 * w + 1:3]),
            dim=-1)
    elif d == 4:
        retidx = torch.cat((idxall[:, :, :w], idxall[:, :, w + 1:w * 3:2], idxall[:, :, 3 * w + 2: 6 * w:3],
                            idxall[:, :, 6 * w + 3:(k - 3 * w) * 4 + 6 * w + 1:4]), dim=-1)
    elif d == 5:
        retidx = torch.cat((idxall[:, :, :w], idxall[:, :, w + 1:w * 3:2], idxall[:, :, 3 * w + 2: 6 * w:3],
                            idxall[:, :, 6 * w + 3: w * 10:4], idxall[:, :, 10 * w + 4:5 * (k - 4 * w) + 10 * w + 1:5]),
                           dim=-1)
    return retidx


def knn_with_dilation_multiple(x, k, d=[1, 2, 4]):
    """
    改为向上取整，多取点nearest
    :param x:  b,c,n 可以是点的位置，也可以是点的特征
    :param k:  k nearest  points
    :param d:  dilation list []
    :return:   b, n, k :index
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # b,n,n
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # b, 1, n
    pair_distance = -xx - inner - xx.transpose(2, 1)  # b, n,n  ascending
    w0 = k // d[0]  # + 1 if k % d[0] is not 0 else k // d[0]
    w1 = k // d[1]  # + 1 if k % d[1] is not 0 else k // d[1]
    w2 = k // d[2]  # + 1 if k % d[2] is not 0 else k // d[2]
    # assert w > 1
    # feature_distance = pair_distance.topk(k=k*d,dim=-1)[0] # b,n,k*d
    idxall = pair_distance.topk(k=k * d[2], dim=-1)[1]  # b, n, k*d
    retidx0 = idx_with_dilation(idxall, k, d[0], w0)
    retidx1 = idx_with_dilation(idxall, k, d[1], w1)
    retidx2 = idx_with_dilation(idxall, k, d[2], w2)  # b,n,k
    retidx = torch.cat((retidx0, retidx1, retidx2), dim=-1)  # b,n,3k

    return retidx

def knn_with_dilation_multiple_dlength2(x, k, d=[1, 2]):
    """
    改为向上取整，多取点nearest
    :param x:  b,c,n 可以是点的位置，也可以是点的特征
    :param k:  k nearest  points
    :param d:  dilation list []
    :return:   b, n, k :index
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # b,n,n
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # b, 1, n
    pair_distance = -xx - inner - xx.transpose(2, 1)  # b, n,n  ascending
    w0 = k // d[0]  # + 1 if k % d[0] is not 0 else k // d[0]
    w1 = k // d[1]  # + 1 if k % d[1] is not 0 else k // d[1]

    # feature_distance = pair_distance.topk(k=k*d,dim=-1)[0] # b,n,k*d
    idxall = pair_distance.topk(k=k * d[1], dim=-1)[1]  # b, n, k*d
    retidx0 = idx_with_dilation(idxall, k, d[0], w0)
    retidx1 = idx_with_dilation(idxall, k, d[1], w1)

    retidx = torch.cat((retidx0, retidx1), dim=-1)  # b,n,3k

    return retidx

def cat_dim(points):
    """

    :param points: b, g,k,c
    :return: b,g*k,c
    """
    device = points.device
    b, g, k = points.size(0),points.size(1),points.size(2)
    out = torch.tensor([]).to(device)
    for i in range(g):
        out = torch.cat((out, points[:, i]), dim=1)
    # out = out.to(device)
    return out

def cat_dim_overlap(points):
    """

    :param points: b, g,k,c
    :return: b,g*k,c
    """
    device = points.device
    b, g, k, c = points.shape
    d = k // 2
    out = torch.tensor([]).to(device)
    for i in range(g):
        if i == g -1:
            out = torch.cat((out, points[:, i,:,:]), dim=1)
        else:
            out = torch.cat((out, points[:, i,:d,:]), dim=1)
    # out = out.to(device)
    return out

def cat_dim_overlap_index(index):
    """

    :param index: b, g,k
    :return: b,n        g*k > n
     """
    device = index.device
    b, g, k = index.shape
    d = k // 2
    out = torch.tensor([]).to(device)
    for i in range(g):
        if i == g -1:
            out = torch.cat((out, index[:, i,:]), dim=1)
        else:
            out = torch.cat((out, index[:, i,:d]), dim=1)
    # out = out.to(device)
    return out


def split_byxyz_net(xyz, points, mode='z', split_size=16):
    '''
    author:  ----wjw---------
    :param points:  b,n,c  ,按照z周进行划分所有的点,split_size :划分为16块局域
    :return:[[b,k1,c],[b,k2,c],[]... ]
    '''
    assert mode in ['x', 'y', 'z']
    b, n, c = points.shape
    # xyz = points[:,:,:3]  #b,n,3
    if mode == 'z':
        sort_xyz = xyz[:, :, 2]  # b,n
    elif mode == 'x':
        sort_xyz = xyz[:, :, 0]
    else:
        sort_xyz = xyz[:, :, 1]
    device = xyz.device
    # print(sort_xyz.device)
    ret_list_points, ret_list_index = torch.Tensor().to(device), torch.Tensor().to(device)
    sort_value, sort_index = torch.sort(sort_xyz, dim=-1, descending=True)  # b,n
    split_numbers = int(n // split_size)  # n/4
    for i in range(0, split_size):
        temp_index = sort_index[:, i * split_numbers:(i + 1) * split_numbers]  # b, n/1
        tmp_points = index_points(points, temp_index)
        temp_index = temp_index.unsqueeze(1)
        tmp_points = tmp_points.unsqueeze(1)
        ret_list_points = torch.cat((ret_list_points, tmp_points), dim=1)
        ret_list_index = torch.cat((ret_list_index, temp_index), dim=1)

    return ret_list_points, ret_list_index  # b,j,k


def split_byxyz_net_halfoverlap(xyz, points, mode='z', split_size=16):
    '''
    author:  ----wjw---------
    :param points:  b,n,c  ,按照z周进行划分所有的点,split_size :每个区域16个点，与上面的不一样
    :return:[[b,k1,c],[b,k2,c],[]... ]
    '''
    assert mode in ['x', 'y', 'z']
    b, n, c = points.shape
    # xyz = points[:,:,:3]  #b,n,3
    if mode == 'z':
        sort_xyz = xyz[:, :, 2]  # b,n
    elif mode == 'x':
        sort_xyz = xyz[:, :, 0]
    else:
        sort_xyz = xyz[:, :, 1]
    device = xyz.device
    overlap = split_size // 2
    # print(sort_xyz.device)
    ret_list_points, ret_list_index = torch.Tensor().to(device), torch.Tensor().to(device)
    sort_value, sort_index = torch.sort(sort_xyz, dim=-1, descending=True)  # b,n
    split_numbers = int((n - split_size) // overlap + 1 ) # n/4
    for i in range(0, split_numbers):
        temp_index = sort_index[:, i * overlap : i * overlap + split_size]  # b, n/1
        tmp_points = index_points(points, temp_index)
        temp_index = temp_index.unsqueeze(1)
        tmp_points = tmp_points.unsqueeze(1)
        ret_list_points = torch.cat((ret_list_points, tmp_points), dim=1)
        ret_list_index = torch.cat((ret_list_index, temp_index), dim=1)

    return ret_list_points, ret_list_index  # b,j,k


# scaler attention
class PointTransformerBlock_scaler(nn.Module):
    def __init__(self, dim, normalize=False):
        super(PointTransformerBlock_scaler, self).__init__()
        # self.pre_block = nn.Linear(dim,dim)
        self.normalize = normalize
        self.to_q = nn.Conv1d(dim, dim // 4, 1, bias=False)
        self.to_k = nn.Conv1d(dim, dim // 4, 1, bias=False)
        self.to_v = nn.Conv1d(dim, dim, 1, bias=False)

        self.attn_mlp = nn.Sequential(
            nn.Conv1d(dim, dim, 1, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, xyz):
        """

        :param
        :param xyz: point postions b,c,n
          absolute position encoding
        :return: attentive points features b,g,c
        """

        q = self.to_q(xyz).permute(0, 2, 1)  # b,c/4,n ---> b, n, c/4
        k = self.to_k(xyz)  # b,c/4,n
        v = self.to_v(xyz)  # b,c,n

        energy = torch.bmm(q, k)  # b,n,n
        attn = F.softmax(energy, dim=-1)  # b,n,n(nor)
        # attn = attn / np.sqrt(v.size(-1))  # b,g,k,k(norrmalize)
        # if self.normalize:
        #     attn = attn / (1e-8 + attn.sum(dim=-2,keepdim=True))

        agg_feature = torch.bmm(v, attn)  # b,c,n
        agg_feature = self.attn_mlp(agg_feature)  # b,c,n
        out = agg_feature + xyz
        # agg_feature = torch.max(agg_feature,dim-2)[0]
        # agg_feature = torch.sum(agg_feature,dim=-2)  # b,g,dim

        return out


# vector attention
class PointTransformerBlock(nn.Module):
    def __init__(self,indim,dim):
        super(PointTransformerBlock, self).__init__()

        self.to_q = nn.Linear(indim,dim,bias=False)
        self.to_k = nn.Linear(indim,dim,bias=False)
        self.to_v = nn.Linear(indim,dim,bias=False)


        self.attn_mlp = nn.Sequential(
            nn.Linear(dim,dim),
            nn.ReLU(True),
            nn.Linear(dim,dim)
        )
        self.final_linear = nn.Linear(dim,dim)

    def forward(self,points):
        """

        :param points: points features, b,n,k,c

        :return: attentive points features b,n,c
        """
        q = self.to_q(points)   # b,n,k,dim
        k = self.to_k(points)  # b,n,k,dim
        v = self.to_v(points)  # b,n,k,dim

        # pos_enc = self.pos_mlp(xyz)  # b,n,k,dim   absolute postion encoding
        attn = self.attn_mlp(q- k ) #b,n,k,dim
        attn = F.softmax(attn / np.sqrt(k.size(-1)),dim=-2)  # b,n,k(norrmalize),dim

        agg_feature = einsum('b i j k, b i j k -> b i k',attn , v ) # b, n ,dim
        # agg_pos = einsum('b i j k, b i j k -> b i k',attn , )
        # agg = self.final_linear(agg) + x_pre

        return agg_feature

class Layerwise3dAttention(nn.Module):
    def __init__(self, npoints, inchannel, split_size=[16, 16, 32], local_agg='gk'):
        super(Layerwise3dAttention, self).__init__()
        self.npoints = npoints
        self.splitx, self.splity, self.splitz = split_size[0], split_size[1], split_size[2]
        self.mlp_x = nn.Sequential(
            nn.Conv1d(self.splitx, self.splitx, 1, bias=False),
            nn.BatchNorm1d(self.splitx),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.splitx, 1, 1, bias=False),
            nn.Softmax(-1)
            # nn.Sigmoid()
        )
        self.mlp_y = nn.Sequential(
            nn.Conv1d(self.splity, self.splity, 1, bias=False),
            nn.BatchNorm1d(self.splity),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.splity, 1, 1, bias=False),
            nn.Softmax(-1)
            # nn.Sigmoid()
        )
        self.mlp_z = nn.Sequential(
            nn.Conv1d(self.splitz, self.splitz, 1, bias=False),
            nn.BatchNorm1d(self.splitz),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.splitz, 1, 1, bias=False),
            nn.Softmax(-1)
            # nn.Sigmoid()
        )
        self.local_agg = local_agg
        # parameter free:mean,max,sum --- independent way: conv2d ---local way: k1,gk,transformer
        assert self.local_agg in ['mean', 'max', 'sum', 'conv2d', 'k1', 'gk', 'transformer']
        if self.local_agg == 'conv2d':
            self.group_x = nn.Sequential(nn.Conv2d(inchannel, inchannel, 1, bias=False),
                                         nn.BatchNorm2d(inchannel),
                                         nn.LeakyReLU(negative_slope=0.2),
                                         nn.Conv2d(inchannel, inchannel, 1, bias=False),
                                         nn.BatchNorm2d(inchannel),
                                         nn.LeakyReLU(negative_slope=0.2)
                                         )
            self.group_y = nn.Sequential(nn.Conv2d(inchannel, inchannel, 1, bias=False),
                                         nn.BatchNorm2d(inchannel),
                                         nn.LeakyReLU(negative_slope=0.2),
                                         nn.Conv2d(inchannel, inchannel, 1, bias=False),
                                         nn.BatchNorm2d(inchannel),
                                         nn.LeakyReLU(negative_slope=0.2)
                                         )
            self.group_z = nn.Sequential(nn.Conv2d(inchannel, inchannel, 1, bias=False),
                                         nn.BatchNorm2d(inchannel),
                                         nn.LeakyReLU(negative_slope=0.2),
                                         nn.Conv2d(inchannel, inchannel, 1, bias=False),
                                         nn.BatchNorm2d(inchannel),
                                         nn.LeakyReLU(negative_slope=0.2)
                                         )
        elif self.local_agg == 'k1':
            self.group_x = nn.Parameter(torch.ones((self.npoints // self.splitx, 1)), requires_grad=True)
            self.group_y = nn.Parameter(torch.ones((self.npoints // self.splity, 1)), requires_grad=True)
            self.group_z = nn.Parameter(torch.ones((self.npoints // self.splitz, 1)), requires_grad=True)
            self.sigmoid = nn.Sigmoid()
            self.softmax = nn.Softmax(dim=0)
        elif self.local_agg == 'gk':
            self.group_x = nn.Parameter(torch.ones((self.splitx, self.npoints // self.splitx)), requires_grad=True)
            self.group_y = nn.Parameter(torch.ones((self.splity, self.npoints // self.splity)), requires_grad=True)
            self.group_z = nn.Parameter(torch.ones((self.splitz, self.npoints // self.splitz)), requires_grad=True)
            self.sigmoid = nn.Sigmoid()
            self.softmax = nn.Softmax(dim=-1)
        elif self.local_agg == 'transformer':
            self.group_x = PointTransformerBlock_scaler(dim=inchannel, normalize=False)
            self.group_y = PointTransformerBlock_scaler(dim=inchannel, normalize=False)
            self.group_z = PointTransformerBlock_scaler(dim=inchannel, normalize=False)

    def forward(self, xyz, points):
        """

        :param xyz: b,n,3
        :param points: b,n,c
        :return: 3d attention feats
        """
        # b,g,k   ----k = n / g(group)
        point_x, index_x = split_byxyz_net(xyz, points, mode='x', split_size=self.splitx)
        point_y, index_y = split_byxyz_net(xyz, points, mode='y', split_size=self.splity)
        point_z, index_z = split_byxyz_net(xyz, points, mode='z', split_size=self.splitz)

        x = index_x.long()  # b, g,k
        y = index_y.long()  # b,g,k
        z = index_z.long()
        points_x_norm = point_x - point_x.mean(dim=-2,keepdim=True) # b,g,k,c
        points_y_norm = point_y - point_y.mean(dim=-2,keepdim=True)
        points_z_norm = point_z - point_z.mean(dim=-2,keepdim=True)

        # x_xyz = index_points(xyz, x)
        # y_xyz = index_points(xyz, y)
        # z_xyz = index_points(xyz, z)

        # group-wise spatial attention    and   group-wise channel attention
        if self.local_agg == 'conv2d':
            points_x_local = self.group_x(points_x_norm.permute(0, 3, 1, 2))  # b,c,g,k
            points_y_local = self.group_y(points_y_norm.permute(0, 3, 1, 2))  # b,c,g,k
            points_z_local = self.group_z(points_z_norm.permute(0, 3, 1, 2))  # b,c,g,k
            points_x_local = torch.mean(points_x_local, dim=-1).permute(0, 2, 1)  # b,c,g --- b,g,c
            points_y_local = torch.mean(points_y_local, dim=-1).permute(0, 2, 1)
            points_z_local = torch.mean(points_z_local, dim=-1).permute(0, 2, 1)
        elif self.local_agg == 'k1':
            # b,g,k,c --> b,c,g,k X k,1 ---> b,c,g,1 ---> b,c,g ----> b,g,c
            points_x_local = torch.matmul(points_x_norm.permute(0, 3, 1, 2), self.sigmoid(self.group_x)).squeeze().permute(0,
                                                                                                                      2,
                                                                                                                      1)
            points_y_local = torch.matmul(points_y_norm.permute(0, 3, 1, 2), self.sigmoid(self.group_y)).squeeze().permute(0,
                                                                                                                      2,
                                                                                                                      1)
            points_z_local = torch.matmul(points_z_norm.permute(0, 3, 1, 2), self.sigmoid(self.group_z)).squeeze().permute(0,
                                                                                                                      2,
                                                                                                                      1)
        elif self.local_agg == 'gk':
            points_x_local = points_x_norm * self.sigmoid(self.group_x).unsqueeze(0).unsqueeze(
                -1)  # b,g,k,c- - 1,g, k,1---b,g,k,c
            points_y_local = points_y_norm * self.sigmoid(self.group_y).unsqueeze(0).unsqueeze(-1)
            points_z_local = points_z_norm * self.sigmoid(self.group_z).unsqueeze(0).unsqueeze(-1)
            points_x_local = torch.mean(points_x_local, dim=-2)  # b,g,c
            points_y_local = torch.mean(points_y_local, dim=-2)
            points_z_local = torch.mean(points_z_local, dim=-2)
        elif self.local_agg == 'mean':
            # points_x_local = torch.mean(points_x, dim=-2)  # b,g,c
            # points_y_local = torch.mean(points_y, dim=-2)
            # points_z_local = torch.mean(points_z, dim=-2)
            points_x_local = torch.max(points_x_norm, dim=-2)[0]  # b,g,c
            points_y_local = torch.max(points_y_norm, dim=-2)[0]
            points_z_local = torch.max(points_z_norm, dim=-2)[0]
        # test two condition: 1. layereise attention  2. channel-wise attention
        x_attention = self.mlp_x(points_x_local)  # b, 1, c
        y_attention = self.mlp_y(points_y_local)  # b ,1 ,c
        z_attention = self.mlp_z(points_z_local)  # b, 1, c

        agg_x = point_x * x_attention.unsqueeze(-2)  # b,g,k,c ----回复到原始的b,n,c 的shape
        formal_points = cat_dim(agg_x)  # b,n,c
        points_y = index_points(formal_points, y)  # b,g,k,c
        agg_y = points_y * y_attention.unsqueeze(-2)
        formal_points = cat_dim(agg_y)
        points_z = index_points(formal_points, z)
        agg_z = points_z * z_attention.unsqueeze(-2)
        formal_points = cat_dim(agg_z)

        # all_att = x_attention * y_attention * z_attention
        # formal_points = points * all_att

        # -------------------add residual------------------
        # residual = self.mlp_residual(points)
        formal_points = formal_points + points  # residual

        return formal_points


class Layerwise3dAttention2(nn.Module):
    def __init__(self, npoints, inchannel, split_size=[16, 16, 32], local_agg='gk'):
        super(Layerwise3dAttention2, self).__init__()
        self.npoints = npoints
        self.splitx, self.splity, self.splitz = split_size[0], split_size[1], split_size[2]
        self.mlp_x = nn.Sequential(
            nn.Conv1d(inchannel, inchannel//4 , 1, bias=False),
            nn.BatchNorm1d(inchannel //4),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(inchannel // 4, inchannel , 1, bias=False),
            nn.BatchNorm1d(inchannel),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.mlp_y = nn.Sequential(
            nn.Conv1d(inchannel, inchannel // 4, 1, bias=False),
            nn.BatchNorm1d(inchannel // 4),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(inchannel // 4, inchannel, 1, bias=False),
            nn.BatchNorm1d(inchannel),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.mlp_z = nn.Sequential(
            nn.Conv1d(inchannel, inchannel // 4, 1, bias=False),
            nn.BatchNorm1d(inchannel // 4),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(inchannel // 4, inchannel, 1, bias=False),
            nn.BatchNorm1d(inchannel),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.norm = nn.Softmax(1)
        self.local_agg = local_agg
        # parameter free:mean,max,sum --- independent way: conv2d ---local way: k1,gk,transformer
        assert self.local_agg in ['mean', 'max', 'sum', 'conv2d', 'k1', 'gk', 'transformer']
        if self.local_agg == 'conv2d':
            self.group_x = nn.Sequential(nn.Conv2d(inchannel, inchannel, 1, bias=False),
                                         nn.BatchNorm2d(inchannel),
                                         nn.LeakyReLU(negative_slope=0.2),
                                         # nn.Conv2d(inchannel, inchannel, 1, bias=False),
                                         # nn.BatchNorm2d(inchannel),
                                         # nn.LeakyReLU(negative_slope=0.2)
                                         )
            self.group_y = nn.Sequential(nn.Conv2d(inchannel, inchannel, 1, bias=False),
                                         nn.BatchNorm2d(inchannel),
                                         nn.LeakyReLU(negative_slope=0.2),
                                         # nn.Conv2d(inchannel, inchannel, 1, bias=False),
                                         # nn.BatchNorm2d(inchannel),
                                         # nn.LeakyReLU(negative_slope=0.2)
                                         )
            self.group_z = nn.Sequential(nn.Conv2d(inchannel, inchannel, 1, bias=False),
                                         nn.BatchNorm2d(inchannel),
                                         nn.LeakyReLU(negative_slope=0.2),
                                         # nn.Conv2d(inchannel, inchannel, 1, bias=False),
                                         # nn.BatchNorm2d(inchannel),
                                         # nn.LeakyReLU(negative_slope=0.2)
                                         )
        elif self.local_agg == 'k1':
            self.group_x = nn.Parameter(torch.ones((self.npoints // self.splitx, 1)), requires_grad=True)
            self.group_y = nn.Parameter(torch.ones((self.npoints // self.splity, 1)), requires_grad=True)
            self.group_z = nn.Parameter(torch.ones((self.npoints // self.splitz, 1)), requires_grad=True)
            self.sigmoid = nn.Sigmoid()
            self.softmax = nn.Softmax(dim=0)
        elif self.local_agg == 'gk':
            self.group_x = nn.Parameter(torch.ones((self.splitx, self.npoints // self.splitx)), requires_grad=True)
            self.group_y = nn.Parameter(torch.ones((self.splity, self.npoints // self.splity)), requires_grad=True)
            self.group_z = nn.Parameter(torch.ones((self.splitz, self.npoints // self.splitz)), requires_grad=True)
            self.sigmoid = nn.Sigmoid()
            self.softmax = nn.Softmax(dim=-1)
        elif self.local_agg == 'transformer':
            self.group_x = PointTransformerBlock_scaler(dim=inchannel, normalize=False)
            self.group_y = PointTransformerBlock_scaler(dim=inchannel, normalize=False)
            self.group_z = PointTransformerBlock_scaler(dim=inchannel, normalize=False)

    def forward(self, xyz, points):
        """

        :param xyz: b,n,3
        :param points: b,n,c
        :return: 3d attention feats
        """
        # b,g,k   ----k = n / g(group)
        point_x, index_x = split_byxyz_net(xyz, points, mode='x', split_size=self.splitx)
        point_y, index_y = split_byxyz_net(xyz, points, mode='y', split_size=self.splity)
        point_z, index_z = split_byxyz_net(xyz, points, mode='z', split_size=self.splitz)

        x = index_x.long()  # b, g,k
        y = index_y.long()  # b,g,k
        z = index_z.long()
        points_x = point_x  # b,g,k,c
        points_y = point_y
        points_z = point_z

        # points_x_norm = point_x - point_x.mean(dim=-2, keepdim=True)  # b,g,k,c
        # points_y_norm = point_y - point_y.mean(dim=-2, keepdim=True)
        # points_z_norm = point_z - point_z.mean(dim=-2, keepdim=True)
        # x_xyz = index_points(xyz, x)
        # y_xyz = index_points(xyz, y)
        # z_xyz = index_points(xyz, z)

        # group-wise spatial attention    and   group-wise channel attention
        if self.local_agg == 'conv2d':
            points_x_local = self.group_x(points_x.permute(0, 3, 1, 2))  # b,c,g,k
            points_y_local = self.group_y(points_y.permute(0, 3, 1, 2))  # b,c,g,k
            points_z_local = self.group_z(points_z.permute(0, 3, 1, 2))  # b,c,g,k
            points_x_local = torch.mean(points_x_local, dim=-1).permute(0, 2, 1)  # b,c,g --- b,g,c
            points_y_local = torch.mean(points_y_local, dim=-1).permute(0, 2, 1)
            points_z_local = torch.mean(points_z_local, dim=-1).permute(0, 2, 1)
        elif self.local_agg == 'k1':
            # b,g,k,c --> b,c,g,k X k,1 ---> b,c,g,1 ---> b,c,g ----> b,g,c
            points_x_local = torch.matmul(points_x.permute(0, 3, 1, 2), self.sigmoid(self.group_x)).squeeze().permute(0,
                                                                                                                      2,
                                                                                                                      1)
            points_y_local = torch.matmul(points_y.permute(0, 3, 1, 2), self.sigmoid(self.group_y)).squeeze().permute(0,
                                                                                                                      2,
                                                                                                                      1)
            points_z_local = torch.matmul(points_z.permute(0, 3, 1, 2), self.sigmoid(self.group_z)).squeeze().permute(0,
                                                                                                                      2,
                                                                                                                      1)
        elif self.local_agg == 'gk':
            points_x_local = points_x * self.sigmoid(self.group_x).unsqueeze(0).unsqueeze(
                -1)  # b,g,k,c- - 1,g, k,1---b,g,k,c
            points_y_local = points_y * self.sigmoid(self.group_y).unsqueeze(0).unsqueeze(-1)
            points_z_local = points_z * self.sigmoid(self.group_z).unsqueeze(0).unsqueeze(-1)
            points_x_local = torch.mean(points_x_local, dim=-2)  # b,g,c
            points_y_local = torch.mean(points_y_local, dim=-2)
            points_z_local = torch.mean(points_z_local, dim=-2)
        elif self.local_agg == 'mean':
            points_x_local = torch.mean(points_x, dim=-2)  # b,g,c
            points_y_local = torch.mean(points_y, dim=-2)
            points_z_local = torch.mean(points_z, dim=-2)

        # test two condition: 1. layereise attention  2. channel-wise attention
        x_attention = self.mlp_x(points_x_local.permute(0,2,1))  # b,g,c-- b,c,g---> b,c,g
        x_attention = torch.max(x_attention,dim=-1,keepdim=True)[0]     #b,c,g---b,c,1
        y_attention = self.mlp_y(points_y_local.permute(0,2,1))
        y_attention = torch.max(y_attention, dim=-1, keepdim=True)[0]
        z_attention = self.mlp_z(points_z_local.permute(0,2,1))
        z_attention = torch.max(z_attention, dim=-1, keepdim=True)[0]

        x_attention = self.norm(x_attention).permute(0,2,1)  # b,c ,1----> b, 1, c
        y_attention = self.norm(y_attention).permute(0, 2, 1)
        z_attention = self.norm(z_attention).permute(0, 2, 1)

        agg_x = points_x * x_attention.unsqueeze(-2)  # b,g,k,c ----回复到原始的b,n,c 的shape
        formal_points = cat_dim(agg_x)  # b,n,c
        points_y = index_points(formal_points, y)  # b,g,k,c
        agg_y = points_y * y_attention.unsqueeze(-2)
        formal_points = cat_dim(agg_y)
        points_z = index_points(formal_points, z)
        agg_z = points_z * z_attention.unsqueeze(-2)
        formal_points = cat_dim(agg_z)

        # all_att = x_attention * y_attention * z_attention
        # formal_points = points * all_att

        # -------------------add residual------------------
        # residual = self.mlp_residual(points)
        formal_points = formal_points + points  # residual

        return formal_points


class Layerwise3dAttention3(nn.Module):
    def __init__(self, npoints, inchannel,outchannel, split_size=[16, 16, 32], local_agg='gk'):
        super(Layerwise3dAttention3, self).__init__()
        self.npoints = npoints
        self.splitx, self.splity, self.splitz = split_size[0], split_size[1], split_size[2]
        self.mlp_x = nn.Sequential(
            nn.Conv1d(inchannel, outchannel // 4, kernel_size=1, bias=False),
            nn.BatchNorm1d(outchannel //4),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(outchannel // 4, outchannel , 1, bias=False),
            nn.BatchNorm1d(outchannel),
            # nn.LeakyReLU(negative_slope=0.2)
        )
        self.mlp_y = nn.Sequential(
            nn.Conv1d(inchannel, outchannel//4 , kernel_size=1, bias=False),
            nn.BatchNorm1d(outchannel//4),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(outchannel//4, outchannel, 1, bias=False),
            nn.BatchNorm1d(outchannel),
        )
        self.mlp_z = nn.Sequential(
            nn.Conv1d(inchannel, outchannel //4, kernel_size=1, bias=False),
            nn.BatchNorm1d(outchannel//4),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(outchannel//4, outchannel, 1, bias=False),
            nn.BatchNorm1d(outchannel),
        )
        self.norm = nn.Softmax(dim=1)
        self.local_agg = local_agg
        self.residual = False
        if inchannel != outchannel:
            self.residual = True
            self.mlp_residual = nn.Sequential(nn.Conv1d(inchannel,outchannel,1,bias=False),
                                              nn.BatchNorm1d(outchannel),
                                              nn.LeakyReLU(negative_slope=0.2))


    def forward(self, xyz, points):
        """

        :param xyz: b,n,3
        :param points: b,n,c
        :return: 3d attention feats
        """
        # b,g,k   ----k = n / g(group)
        point_x, index_x = split_byxyz_net(xyz, points, mode='x', split_size=self.splitx)
        point_y, index_y = split_byxyz_net(xyz, points, mode='y', split_size=self.splity)
        point_z, index_z = split_byxyz_net(xyz, points, mode='z', split_size=self.splitz)

        x = index_x.long()  # b, g,k
        y = index_y.long()  # b,g,k
        z = index_z.long()
        points_x_norm = point_x - point_x.mean(dim=-2,keepdim=True) # b,g,k,c  test this to model the local shape -######################
        points_y_norm = point_y - point_y.mean(dim=-2,keepdim=True)
        points_z_norm = point_z - point_z.mean(dim=-2,keepdim=True)


        # points_x_local = torch.mean(points_x_norm, dim=-2)  # b,g,2c
        # points_y_local = torch.mean(points_y_norm, dim=-2)
        # points_z_local = torch.mean(points_z_norm, dim=-2)

        points_x_local = torch.max(points_x_norm, dim=-2)[0]  # b,g,c
        points_y_local = torch.max(points_y_norm, dim=-2)[0]
        points_z_local = torch.max(points_z_norm, dim=-2)[0]

        # test two condition: 1. layereise attention  2. channel-wise attention
        x_attention = self.mlp_x(points_x_local.permute(0,2,1))  # b,g,c-- b,c,g---> b,c,g
        x_attention = torch.max(x_attention,dim=-1,keepdim=True)[0]     #b,c,g---b,c,1
        y_attention = self.mlp_y(points_y_local.permute(0,2,1))
        y_attention = torch.max(y_attention, dim=-1, keepdim=True)[0]
        z_attention = self.mlp_z(points_z_local.permute(0,2,1))
        z_attention = torch.max(z_attention, dim=-1, keepdim=True)[0]

        x_attention = self.norm(x_attention).permute(0,2,1)  # b,c ,1----> b, 1, c
        y_attention = self.norm(y_attention).permute(0, 2, 1)
        z_attention = self.norm(z_attention).permute(0, 2, 1)

        # agg_x = point_x * x_attention.unsqueeze(-2)  # b,g,k,c ----回复到原始的b,n,c 的shape
        # formal_points = cat_dim(agg_x)  # b,n,c
        # points_y = index_points(formal_points, y)  # b,g,k,c
        # agg_y = points_y * y_attention.unsqueeze(-2)
        # formal_points = cat_dim(agg_y)
        # points_z = index_points(formal_points, z)
        # agg_z = points_z * z_attention.unsqueeze(-2)
        # formal_points = cat_dim(agg_z)

        agg_z = point_z * z_attention.unsqueeze(-2)  # b,g,k,c ----回复到原始的b,n,c 的shape
        formal_points = cat_dim(agg_z)  # b,n,c
        points_y = index_points(formal_points, y)  # b,g,k,c
        agg_y = points_y * y_attention.unsqueeze(-2)
        formal_points = cat_dim(agg_y)
        points_x = index_points(formal_points, x)
        agg_x = points_x * x_attention.unsqueeze(-2)
        formal_points = cat_dim(agg_x)


        # all_att = x_attention * y_attention * z_attention
        # formal_points = points * all_att

        # -------------------add residual------------------
        if self.residual:
            residual = self.mlp_residual(points)
            print('use mpl residual')
        else:
            residual = points
        formal_points = formal_points + residual  # residual

        return formal_points


class Layerwiseagg(nn.Module):
    def __init__(self,npoints,inchannel,split_size = [8,8,8],local_agg = 'mean'):
        super(Layerwiseagg, self).__init__()
        self.npoints = npoints
        self.splitx,self.splity,self.splitz = split_size[0],split_size[1],split_size[2]
        self.mlp_x_kwise = nn.Sequential(nn.Conv2d(inchannel,inchannel,kernel_size=[1,3],stride=1,padding=[0,1]),
                                         nn.BatchNorm2d(inchannel),
                                         nn.LeakyReLU(0.2))
        self.mlp_x_gwise = nn.Sequential(nn.Conv2d(inchannel,inchannel,kernel_size=[3,1],stride=1,padding=[1,0]),
                                         nn.BatchNorm2d(inchannel),
                                         nn.LeakyReLU(0.2))

        self.mlp_y_kwise = nn.Sequential(nn.Conv2d(inchannel, inchannel, kernel_size=[1, 3], stride=1, padding=[0, 1]),
                                         nn.BatchNorm2d(inchannel),
                                         nn.LeakyReLU(0.2))
        self.mlp_y_gwise = nn.Sequential(nn.Conv2d(inchannel, inchannel, kernel_size=[3, 1], stride=1, padding=[1, 0]),
                                         nn.BatchNorm2d(inchannel),
                                         nn.LeakyReLU(0.2))

        self.mlp_z_kwise = nn.Sequential(nn.Conv2d(inchannel, inchannel, kernel_size=[1, 3], stride=1, padding=[0, 1]),
                                         nn.BatchNorm2d(inchannel),
                                         nn.LeakyReLU(0.2))
        self.mlp_z_gwise = nn.Sequential(nn.Conv2d(inchannel, inchannel, kernel_size=[3, 1], stride=1, padding=[1, 0]),
                                         nn.BatchNorm2d(inchannel),
                                         nn.LeakyReLU(0.2))

    def forward(self,xyz,points):
        """
        :param xyz: b,n,3
        :param points: b,n,c
        :return: 3d attention feats
        """
        # b,g,k   ----k = n / g(group) --- b,g,k,c
        point_x, index_x = split_byxyz_net(xyz, points, mode='x', split_size=self.splitx)
        point_y, index_y = split_byxyz_net(xyz, points, mode='y', split_size=self.splity)
        point_z, index_z = split_byxyz_net(xyz, points, mode='z', split_size=self.splitz)

        x = index_x.long()  # b, g,k
        y = index_y.long()  # b,g,k
        z = index_z.long()

        xgroup = self.mlp_x_gwise(self.mlp_x_kwise(point_x.permute(0,3,1,2)))  # b,g,k,c--b,c,g,k---b,c,g,k
        xformal = cat_dim(xgroup.permute(0,2,3,1))  # b,n,c

        ygroup = self.mlp_y_gwise(self.mlp_y_kwise(point_y.permute(0, 3, 1, 2)))  # b,g,k,c--b,c,g,k---b,c,g,k
        yformal = cat_dim(ygroup.permute(0, 2, 3, 1))  # b,n,c

        zgroup = self.mlp_z_gwise(self.mlp_z_kwise(point_z.permute(0, 3, 1, 2)))  # b,g,k,c--b,c,g,k---b,c,g,k
        zformal = cat_dim(zgroup.permute(0, 2, 3, 1))  # b,n,c

        out = (xformal + yformal + zformal) / 3.0 + points
        return out

class AdaptiveConv(nn.Module):
    def __init__(self, in_channels, out_channels, feat_channels):
        super(AdaptiveConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feat_channels = feat_channels

        self.conv0 = nn.Conv2d(feat_channels, out_channels, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(out_channels, out_channels*in_channels, kernel_size=1, bias=False)
        self.bn0 = nn.BatchNorm2d(out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, y):
        # x: (bs, 3, num_points, k), y: (bs, feat_channels, num_points, k)
        batch_size, n_dims, num_points, k = x.size()

        y = self.conv0(y) # (bs, out, num_points, k)
        y = self.leaky_relu(self.bn0(y))
        y = self.conv1(y) # (bs, in*out, num_points, k)
        y = y.permute(0, 2, 3, 1).view(batch_size, num_points, k, self.out_channels, self.in_channels) # (bs, num_points, k, out, in)

        x = x.permute(0, 2, 3, 1).unsqueeze(4) # (bs, num_points, k, in_channels, 1)
        x = torch.matmul(y, x).squeeze(4) # (bs, num_points, k, out_channels)
        x = x.permute(0, 3, 1, 2).contiguous() # (bs, out_channels, num_points, k)

        x = self.bn1(x)
        x = self.leaky_relu(x)

        return x

class LayerSelfAttention_group(nn.Module):
    def __init__(self,inchannel,outchannel,linearnorm=True):
        super(LayerSelfAttention_group, self).__init__()
        self.conv = nn.Conv1d(inchannel,outchannel,kernel_size=1,bias=False)
        self.linearnorm = linearnorm
        if not linearnorm:
            self.norm = nn.Softmax(dim=1)

    def forward(self,x):
        """
        x:b,c,g
        """
        f_bcg = self.conv(x)  # b,c,g
        f_nor = nn.functional.normalize(f_bcg,p=2,dim=1)  # normalize in the c channel, we want to model the relationship between layers
        corr = torch.matmul(f_nor.permute(0,2,1),f_nor)  # b,g,g
        if not self.linearnorm:
            corr = self.norm(corr)
        else:
            corr /= corr.sum(dim=1,keepdim=True)
        out = torch.matmul(f_nor,corr) + x # b,c,g * b,g,g = b,c,g
        return out

class LayerSelfAttention_channel(nn.Module):
    def __init__(self,inchannel,outchannel,linearnorm=True):
        super(LayerSelfAttention_channel, self).__init__()
        self.conv = nn.Conv1d(inchannel,outchannel,kernel_size=1,stride=1,padding=0,bias=False)
        self.linearnorm = linearnorm
        if not linearnorm:
            self.norm = nn.Softmax(dim=-1)

    def forward(self,x):
        """
        x:b,c,g
        """
        f_bcg = self.conv(x)  # b,c,g
        f_nor = nn.functional.normalize(f_bcg,p=2,dim=-1)  # normalize in the g channel, we want to model the relationship between channels  -----b,c,g
        corr = torch.matmul(f_nor,f_nor.permute(0,2,1))  # b,c,c
        if not self.linearnorm:
            corr = self.norm(corr)
        else:
            corr /= (corr.sum(dim=-1,keepdim=True)+1e-8)
        out = torch.matmul(corr,f_nor) + x # b,c,c * b,c,g = b,c,g
        return out


def localrelativeagg(feature,relaxyz,mode='x'):
    """
    feature:b,c,g,k
    relaxyz :b,3,g,k
    mode:split by x
    return :b,c,g,k
    """
    if mode == 'x':
        y = relaxyz[:,1,:,:].unsqueeze(1)  # b,1,g,k
        z = relaxyz[:,2,:,:].unsqueeze(1)
        r_y = y / (y.max(dim=-1,keepdim=True)[0] - y.min(dim=-1,keepdim=True)[0]+1e-8)  #b,1,g,1
        r_z = z / (z.max(dim=-1,keepdim=True)[0] - z.min(dim=-1,keepdim=True)[0]+1e-8)
        fy = r_y * feature # b,c,g,k
        fz = r_z * feature
        agg = torch.cat((fy,fz),dim=1) # b,2c,g,k
    elif mode == 'y':
        x = relaxyz[:, 0, :, :].unsqueeze(1)  # b,1,g,k
        z = relaxyz[:, 2, :, :].unsqueeze(1)
        r_x = x / (x.max(dim=-1, keepdim=True)[0] - x.min(dim=-1, keepdim=True)[0]+1e-8)  # b,1,g,1
        r_z = z / (z.max(dim=-1, keepdim=True)[0] - z.min(dim=-1, keepdim=True)[0]+1e-8)
        fx = r_x * feature  # b,c,g,k
        fz = r_z * feature
        agg = torch.cat((fx, fz), dim=1)  # b,2c,g,k
    elif mode == 'z':
        y = relaxyz[:, 1, :, :].unsqueeze(1)  # b,1,g,k
        x = relaxyz[:, 0, :, :].unsqueeze(1)
        r_y = y / (y.max(dim=-1, keepdim=True)[0] - y.min(dim=-1, keepdim=True)[0]+1e-8)  # b,1,g,1
        r_x = x / (x.max(dim=-1, keepdim=True)[0] - x.min(dim=-1, keepdim=True)[0]+1e-8)
        fy = r_y * feature  # b,c,g,k
        fx = r_x * feature
        agg = torch.cat((fy, fx), dim=1)  # b,2c,g,k
    elif mode == 'xyz':
        z = relaxyz[:,2,:,:].unsqueeze(1)
        y = relaxyz[:, 1, :, :].unsqueeze(1)  # b,1,g,k
        x = relaxyz[:, 0, :, :].unsqueeze(1)
        r_z = z / (z.max(dim=-1, keepdim=True)[0] - z.min(dim=-1, keepdim=True)[0]+1e-8)
        r_y = y / (y.max(dim=-1, keepdim=True)[0] - y.min(dim=-1, keepdim=True)[0] + 1e-8)  # b,1,g,1
        r_x = x / (x.max(dim=-1, keepdim=True)[0] - x.min(dim=-1, keepdim=True)[0] + 1e-8)
        fy = r_y * feature  # b,c,g,k
        fx = r_x * feature
        fz = r_z * feature
        agg = torch.cat((fz,fy, fx), dim=1)  # b,2c,g,k

    return agg


class Layerwise3dAttention_car(nn.Module):
    def __init__(self, npoints, inchannel, split_size=[16, 16, 32], local_agg='gk'):
        super(Layerwise3dAttention_car, self).__init__()
        self.npoints = npoints
        self.splitx, self.splity, self.splitz = split_size[0], split_size[1], split_size[2]
        self.kernel_x, self.kernel_y,self.kernel_z  = npoints // self.splitx, npoints // self.splity, npoints//self.splitz
        self.norm = nn.Softmax(-1)
        self.local_agg = local_agg
        # parameter free:mean,max,sum --- independent way: conv2d ---local way: k1,gk,transformer
        assert self.local_agg in ['mean', 'max', 'sum', 'conv2d', 'k1', 'gk', 'transformer']


        self.group_x_1 = nn.Sequential(nn.Conv2d( inchannel *2,inchannel,kernel_size=(1,1),padding = (0,0),bias=False),
                                     nn.BatchNorm2d(inchannel), nn.LeakyReLU(0.2))

        self.inter_layer_x = nn.Sequential(nn.Conv1d(inchannel,inchannel,1,stride=1,padding=0,bias=False),)
                                           # nn.BatchNorm1d(inchannel))

        self.group_y_1 = nn.Sequential(nn.Conv2d(inchannel *2, inchannel, kernel_size=(1, 1),padding = (0,0),bias=False),
                                       nn.BatchNorm2d(inchannel), nn.LeakyReLU(0.2))

        self.inter_layer_y = nn.Sequential(nn.Conv1d(inchannel, inchannel, 1,stride=1,padding= 0, bias=False),)
                                           # nn.BatchNorm1d(inchannel))

        self.group_z_1 = nn.Sequential(nn.Conv2d(inchannel *2, inchannel, kernel_size=(1, 1),padding = (0,0),bias=False),
                                       nn.BatchNorm2d(inchannel), nn.LeakyReLU(0.2))

        self.inter_layer_z = nn.Sequential(nn.Conv1d(inchannel, inchannel, 1,stride=1,padding=0, bias=False),)
                                           # nn.BatchNorm1d(inchannel))


        self.pos_x = nn.Sequential(nn.Conv2d(3,inchannel,kernel_size=1,bias=False),
                                     nn.BatchNorm2d(inchannel), nn.LeakyReLU(0.2))
        self.pos_y = nn.Sequential(nn.Conv2d(3,inchannel,kernel_size=1,bias=False),
                                     nn.BatchNorm2d(inchannel), nn.LeakyReLU(0.2))
        self.pos_z = nn.Sequential(nn.Conv2d(3,inchannel,kernel_size=1,bias=False),
                                     nn.BatchNorm2d(inchannel), nn.LeakyReLU(0.2))

    def forward(self, xyz, points):
        """

        :param xyz: b,n,3
        :param points: b,n,c
        :return: 3d attention feats
        """
        # b,g,k   ----k = n / g(group)
        point_x, index_x = split_byxyz_net_halfoverlap(xyz, points, mode='x', split_size=self.splitx)  #points feature 变成有序的
        point_y, index_y = split_byxyz_net_halfoverlap(xyz, points, mode='y', split_size=self.splity)
        point_z, index_z = split_byxyz_net_halfoverlap(xyz, points, mode='z', split_size=self.splitz)

        x = index_x.long()  # b, g,k
        y = index_y.long()  # b,g,k
        z = index_z.long()
        points_x = point_x  # b,g,k,c
        points_y = point_y
        points_z = point_z


#        get the group salient features
        points_x_norm = point_x - point_x.mean(dim=-2, keepdim=True)  # b,g,k,c
        points_y_norm = point_y - point_y.mean(dim=-2, keepdim=True)
        points_z_norm = point_z - point_z.mean(dim=-2, keepdim=True)

        x_xyz = index_points(xyz, x) #b,g,k,3
        y_xyz = index_points(xyz, y)
        z_xyz = index_points(xyz, z)
        #
        # #add postion encoding
        x_xyz_norm = x_xyz - x_xyz.mean(dim=-2,keepdim=True)  # b,g,k,3
        y_xyz_norm = y_xyz - y_xyz.mean(dim=-2,keepdim=True)
        z_xyz_norm = z_xyz - z_xyz.mean(dim=-2,keepdim=True)

        points_x_pos = self.pos_x(x_xyz_norm.permute(0,3,1,2)) # b,c,g,k
        points_y_pos = self.pos_y(y_xyz_norm.permute(0, 3, 1, 2))  # b,c,g,k
        points_z_pos = self.pos_z(z_xyz_norm.permute(0, 3, 1, 2))  # b,c,g,k

        local_x_points = torch.cat((points_x_pos, points_x_norm.permute(0,3,1,2)), dim=1)  #b,2c,g,k
        local_y_points = torch.cat((points_y_pos, points_y_norm.permute(0,3,1,2)), dim=1)  # b,2c,g,k
        local_z_points = torch.cat((points_z_pos, points_z_norm.permute(0,3,1,2)), dim=1)  # b,2c,g,k

#fuse the point feature and the position encoding  to get the group-wise feature   #组内信息
        inter_x_1 = self.group_x_1(local_x_points) # b,c,g,k
        inter_y_1 = self.group_y_1(local_y_points)
        inter_z_1 = self.group_z_1(local_z_points)

        inter_x_1 = torch.max(inter_x_1,dim=-1)[0]  # b,c,g
        inter_y_1 = torch.max(inter_y_1, dim=-1)[0]  # b,c,g
        inter_z_1 = torch.max(inter_z_1, dim=-1)[0]  # b,c,g

        # inter_x_1 = torch.mean(inter_x_1, dim=-1)  # b,c,g
        # inter_y_1 = torch.mean(inter_y_1, dim=-1)  # b,c,g
        # inter_z_1 = torch.mean(inter_z_1, dim=-1)  # b,c,g


# inter group relation mapping  组间信息
        intra_x2 = self.inter_layer_x(inter_x_1)   # b,c,g
        intra_x =  intra_x2

        intra_y2 = self.inter_layer_y(inter_y_1)
        intra_y = intra_y2

        intra_z2 = self.inter_layer_z(inter_z_1)
        intra_z = intra_z2


        x_attention = self.norm(intra_x).unsqueeze(-1).permute(0,2,3,1)  # b,c ,g-k---> --b,g,k,c
        y_attention = self.norm(intra_y).unsqueeze(-1).permute(0,2,3,1)
        z_attention = self.norm(intra_z).unsqueeze(-1).permute(0,2,3,1)

#        points_x 包含overlap  ，x_attention则是按照这个有overlap的特征计算出来的
        agg_x = points_x * x_attention  # b,g,k,c ----回复到原始的b,n,c 的shape
        formal_points = cat_dim_overlap(agg_x)  # b,n,c  这里是按照x轴的顺序的，已经有了index，如何返回
        formal_index = cat_dim_overlap_index(x)  # b,n  #这里是按照x轴顺序的index，则按照0-max排序就好了
        sort_v,sort_in = torch.sort(formal_index)  # 从小到大排序
        formal_points = index_points(formal_points,sort_in)


        points_y = index_points(formal_points, y)  # b,g,k,c
        agg_y = points_y * y_attention
        formal_points = cat_dim_overlap(agg_y)
        formal_index = cat_dim_overlap_index(y)  # b,n  #这里是按照y轴顺序的index，则按照0-max排序就好了
        sort_v, sort_in = torch.sort(formal_index)  # 从小到大排序
        formal_points = index_points(formal_points, sort_in)


        points_z = index_points(formal_points, z)
        agg_z = points_z * z_attention
        formal_points = cat_dim_overlap(agg_z)
        formal_index = cat_dim_overlap_index(z)  # b,n  #这里是按照z轴顺序的index，则按照0-max排序就好了
        sort_v, sort_in = torch.sort(formal_index)  # 从小到大排序
        formal_points = index_points(formal_points, sort_in)  # 这样就回复到原始的点的顺序

        # all_att = x_attention * y_attention * z_attention
        # formal_points = points * all_att

        # -------------------add residual------------------
        # residual = self.mlp_residual(points)
        formal_points = formal_points + points  # residual

        return formal_points

class Layerwise3dAttention_car_notoverlap(nn.Module):
    def __init__(self, npoints, inchannel, split_size=[16, 16, 32], local_agg='gk'):
        super(Layerwise3dAttention_car_notoverlap, self).__init__()
        self.npoints = npoints
        self.splitx, self.splity, self.splitz = split_size[0], split_size[1], split_size[2]
        self.kernel_x, self.kernel_y,self.kernel_z  = npoints // self.splitx, npoints // self.splity, npoints//self.splitz
        self.norm = nn.Softmax(1)
        self.local_agg = local_agg
        # parameter free:mean,max,sum --- independent way: conv2d ---local way: k1,gk,transformer
        assert self.local_agg in ['mean', 'max', 'sum', 'conv2d', 'k1', 'gk', 'transformer']


        self.group_x_1 = nn.Sequential(nn.Conv2d( inchannel *2,inchannel,kernel_size=(1,1),padding = (0,0),bias=False),
                                     nn.BatchNorm2d(inchannel), nn.LeakyReLU(0.2))

        self.inter_layer_x = nn.Sequential(nn.Conv1d(inchannel,inchannel,3,stride=1,padding=1,bias=False),)
                                           # nn.BatchNorm1d(inchannel))

        self.group_y_1 = nn.Sequential(nn.Conv2d(inchannel *2, inchannel, kernel_size=(1, 1),padding = (0,0),bias=False),
                                       nn.BatchNorm2d(inchannel), nn.LeakyReLU(0.2))

        self.inter_layer_y = nn.Sequential(nn.Conv1d(inchannel, inchannel, 3,stride=1,padding= 1, bias=False),)
                                           # nn.BatchNorm1d(inchannel))

        self.group_z_1 = nn.Sequential(nn.Conv2d(inchannel *2, inchannel, kernel_size=(1, 1),padding = (0,0),bias=False),
                                       nn.BatchNorm2d(inchannel), nn.LeakyReLU(0.2))

        self.inter_layer_z = nn.Sequential(nn.Conv1d(inchannel, inchannel, 3,stride=1,padding=1, bias=False),)
                                           # nn.BatchNorm1d(inchannel))


        self.pos_x = nn.Sequential(nn.Conv2d(3,inchannel,kernel_size=1,bias=False),
                                     nn.BatchNorm2d(inchannel), nn.LeakyReLU(0.2))
        self.pos_y = nn.Sequential(nn.Conv2d(3,inchannel,kernel_size=1,bias=False),
                                     nn.BatchNorm2d(inchannel), nn.LeakyReLU(0.2))
        self.pos_z = nn.Sequential(nn.Conv2d(3,inchannel,kernel_size=1,bias=False),
                                     nn.BatchNorm2d(inchannel), nn.LeakyReLU(0.2))

    def forward(self, xyz, points):
        """

        :param xyz: b,n,3
        :param points: b,n,c
        :return: 3d attention feats
        """
        # b,g,k   ----k = n / g(group)
        point_x, index_x = split_byxyz_net(xyz, points, mode='x', split_size=self.splitx)  #points feature 变成有序的
        point_y, index_y = split_byxyz_net(xyz, points, mode='y', split_size=self.splity)
        point_z, index_z = split_byxyz_net(xyz, points, mode='z', split_size=self.splitz)

        x = index_x.long()  # b, g,k
        y = index_y.long()  # b,g,k
        z = index_z.long()
        points_x = point_x  # b,g,k,c
        points_y = point_y
        points_z = point_z


#        get the group salient features
        points_x_norm = point_x - point_x.mean(dim=-2, keepdim=True)  # b,g,k,c
        points_y_norm = point_y - point_y.mean(dim=-2, keepdim=True)
        points_z_norm = point_z - point_z.mean(dim=-2, keepdim=True)

        x_xyz = index_points(xyz, x) #b,g,k,3
        y_xyz = index_points(xyz, y)
        z_xyz = index_points(xyz, z)
        #
        # #add postion encoding
        x_xyz_norm = x_xyz - x_xyz.mean(dim=-2,keepdim=True)  # b,g,k,3
        y_xyz_norm = y_xyz - y_xyz.mean(dim=-2,keepdim=True)
        z_xyz_norm = z_xyz - z_xyz.mean(dim=-2,keepdim=True)

        points_x_pos = self.pos_x(x_xyz_norm.permute(0,3,1,2)) # b,c,g,k
        points_y_pos = self.pos_y(y_xyz_norm.permute(0, 3, 1, 2))  # b,c,g,k
        points_z_pos = self.pos_z(z_xyz_norm.permute(0, 3, 1, 2))  # b,c,g,k

        local_x_points = torch.cat((points_x_pos, points_x_norm.permute(0,3,1,2)), dim=1)  #b,2c,g,k
        local_y_points = torch.cat((points_y_pos, points_y_norm.permute(0,3,1,2)), dim=1)  # b,2c,g,k
        local_z_points = torch.cat((points_z_pos, points_z_norm.permute(0,3,1,2)), dim=1)  # b,2c,g,k

#fuse the point feature and the position encoding  to get the group-wise feature   #组内信息
        inter_x_1 = self.group_x_1(local_x_points) # b,c,g,k
        inter_y_1 = self.group_y_1(local_y_points)
        inter_z_1 = self.group_z_1(local_z_points)

        inter_x_1 = torch.max(inter_x_1,dim=-1)[0]  # b,c,g
        inter_y_1 = torch.max(inter_y_1, dim=-1)[0]  # b,c,g
        inter_z_1 = torch.max(inter_z_1, dim=-1)[0]  # b,c,g


# inter group relation mapping  组间信息
        intra_x2 = self.inter_layer_x(inter_x_1)   # b,c,g
        intra_x =  intra_x2

        intra_y2 = self.inter_layer_y(inter_y_1)
        intra_y = intra_y2

        intra_z2 = self.inter_layer_z(inter_z_1)
        intra_z = intra_z2


        x_attention = self.norm(intra_x).unsqueeze(-1).permute(0,2,3,1)  # b,c ,g-k---> --b,g,k,c
        y_attention = self.norm(intra_y).unsqueeze(-1).permute(0,2,3,1)
        z_attention = self.norm(intra_z).unsqueeze(-1).permute(0,2,3,1)

        # points_x 包含overlap  ，x_attention则是按照这个有overlap的特征计算出来的
        agg_x = points_x * x_attention  # b,g,k,c ----回复到原始的b,n,c 的shape
        formal_points = cat_dim(agg_x)  # b,n,c  这里是按照x轴的顺序的，已经有了index，如何返回
        formal_index = cat_dim(x)  # b,n  #这里是按照x轴顺序的index，则按照0-max排序就好了
        sort_v,sort_in = torch.sort(formal_index)  # 从小到大排序
        formal_points = index_points(formal_points,sort_in)


        points_y = index_points(formal_points, y)  # b,g,k,c
        agg_y = points_y * y_attention
        formal_points = cat_dim(agg_y)
        formal_index = cat_dim(y)  # b,n  #这里是按照y轴顺序的index，则按照0-max排序就好了
        sort_v, sort_in = torch.sort(formal_index)  # 从小到大排序
        formal_points = index_points(formal_points, sort_in)


        points_z = index_points(formal_points, z)
        agg_z = points_z * z_attention
        formal_points = cat_dim(agg_z)
        formal_index = cat_dim(z)  # b,n  #这里是按照z轴顺序的index，则按照0-max排序就好了
        sort_v, sort_in = torch.sort(formal_index)  # 从小到大排序
        formal_points = index_points(formal_points, sort_in)  # 这样就回复到原始的点的顺序

        # all_att = x_attention * y_attention * z_attention
        # formal_points = points * all_att

        # -------------------add residual------------------
        # residual = self.mlp_residual(points)
        formal_points = formal_points + points  # residual

        return formal_points

class Layerwise3dAttention_k3(nn.Module):
    def __init__(self, npoints, inchannel, split_size=[16, 16, 32], local_agg='gk'):
        super(Layerwise3dAttention_k3, self).__init__()
        self.npoints = npoints
        self.splitx, self.splity, self.splitz = split_size[0], split_size[1], split_size[2]
        self.mlp_x = nn.Sequential(
            nn.Conv1d(inchannel, inchannel // 4, kernel_size=3,stride=1,padding=1, bias=False),
            nn.BatchNorm1d(inchannel // 4),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(inchannel//4, inchannel, 1, bias=False),
            nn.BatchNorm1d(inchannel),
            nn.LeakyReLU(negative_slope=0.2),
            nn.AdaptiveAvgPool1d(1),
            # nn.Softmax(1)
            nn.Sigmoid()
        )
        self.mlp_y = nn.Sequential(
            nn.Conv1d(inchannel, inchannel//4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(inchannel//4),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(inchannel//4, inchannel, 1, bias=False),
            nn.BatchNorm1d(inchannel),
            nn.LeakyReLU(negative_slope=0.2),
            nn.AdaptiveAvgPool1d(1),
            # nn.Softmax(1)
            nn.Sigmoid()
        )
        self.mlp_z = nn.Sequential(
            nn.Conv1d(inchannel, inchannel//4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(inchannel//4),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(inchannel//4, inchannel, 1, bias=False),
            nn.BatchNorm1d(inchannel),
            nn.LeakyReLU(negative_slope=0.2),
            nn.AdaptiveAvgPool1d(1),
            # nn.Softmax(1)
            nn.Sigmoid()
        )
        self.local_agg = local_agg
        # parameter free:mean,max,sum --- independent way: conv2d ---local way: k1,gk,transformer
        assert self.local_agg in ['mean', 'max', 'sum', 'conv2d', 'k1', 'gk', 'transformer']
        if self.local_agg == 'conv2d':
            self.group_x = nn.Sequential(nn.Conv2d(inchannel, inchannel, 1, bias=False),
                                         nn.BatchNorm2d(inchannel),
                                         nn.LeakyReLU(negative_slope=0.2),
                                         nn.Conv2d(inchannel, inchannel, 1, bias=False),
                                         nn.BatchNorm2d(inchannel),
                                         nn.LeakyReLU(negative_slope=0.2)
                                         )
            self.group_y = nn.Sequential(nn.Conv2d(inchannel, inchannel, 1, bias=False),
                                         nn.BatchNorm2d(inchannel),
                                         nn.LeakyReLU(negative_slope=0.2),
                                         nn.Conv2d(inchannel, inchannel, 1, bias=False),
                                         nn.BatchNorm2d(inchannel),
                                         nn.LeakyReLU(negative_slope=0.2)
                                         )
            self.group_z = nn.Sequential(nn.Conv2d(inchannel, inchannel, 1, bias=False),
                                         nn.BatchNorm2d(inchannel),
                                         nn.LeakyReLU(negative_slope=0.2),
                                         nn.Conv2d(inchannel, inchannel, 1, bias=False),
                                         nn.BatchNorm2d(inchannel),
                                         nn.LeakyReLU(negative_slope=0.2)
                                         )
        elif self.local_agg == 'k1':
            self.group_x = nn.Parameter(torch.ones((self.npoints // self.splitx, 1)), requires_grad=True)
            self.group_y = nn.Parameter(torch.ones((self.npoints // self.splity, 1)), requires_grad=True)
            self.group_z = nn.Parameter(torch.ones((self.npoints // self.splitz, 1)), requires_grad=True)
            self.sigmoid = nn.Sigmoid()
            self.softmax = nn.Softmax(dim=0)
        elif self.local_agg == 'gk':
            self.group_x = nn.Parameter(torch.ones((self.splitx, self.npoints // self.splitx)), requires_grad=True)
            self.group_y = nn.Parameter(torch.ones((self.splity, self.npoints // self.splity)), requires_grad=True)
            self.group_z = nn.Parameter(torch.ones((self.splitz, self.npoints // self.splitz)), requires_grad=True)
            self.sigmoid = nn.Sigmoid()
            self.softmax = nn.Softmax(dim=-1)
        elif self.local_agg == 'transformer':
            self.group_x = PointTransformerBlock_scaler(dim=inchannel, normalize=False)
            self.group_y = PointTransformerBlock_scaler(dim=inchannel, normalize=False)
            self.group_z = PointTransformerBlock_scaler(dim=inchannel, normalize=False)

    def forward(self, xyz, points):
        """

        :param xyz: b,n,3
        :param points: b,n,c
        :return: 3d attention feats
        """
        # b,g,k   ----k = n / g(group)
        point_x, index_x = split_byxyz_net(xyz, points, mode='x', split_size=self.splitx)
        point_y, index_y = split_byxyz_net(xyz, points, mode='y', split_size=self.splity)
        point_z, index_z = split_byxyz_net(xyz, points, mode='z', split_size=self.splitz)

        x = index_x.long()  # b, g,k
        y = index_y.long()  # b,g,k
        z = index_z.long()
        points_x = point_x  # b,g,k,c
        points_y = point_y
        points_z = point_z

        x_xyz = index_points(xyz, x)
        y_xyz = index_points(xyz, y)
        z_xyz = index_points(xyz, z)

        # group-wise spatial attention    and   group-wise channel attention
        if self.local_agg == 'conv2d':
            points_x_local = self.group_x(points_x.permute(0, 3, 1, 2))  # b,c,g,k
            points_y_local = self.group_y(points_y.permute(0, 3, 1, 2))  # b,c,g,k
            points_z_local = self.group_z(points_z.permute(0, 3, 1, 2))  # b,c,g,k
            points_x_local = torch.mean(points_x_local, dim=-1).permute(0, 2, 1)  # b,c,g --- b,g,c
            points_y_local = torch.mean(points_y_local, dim=-1).permute(0, 2, 1)
            points_z_local = torch.mean(points_z_local, dim=-1).permute(0, 2, 1)
        elif self.local_agg == 'k1':
            # b,g,k,c --> b,c,g,k X k,1 ---> b,c,g,1 ---> b,c,g ----> b,g,c
            points_x_local = torch.matmul(points_x.permute(0, 3, 1, 2), self.sigmoid(self.group_x)).squeeze().permute(0,
                                                                                                                      2,
                                                                                                                      1)
            points_y_local = torch.matmul(points_y.permute(0, 3, 1, 2), self.sigmoid(self.group_y)).squeeze().permute(0,
                                                                                                                      2,
                                                                                                                      1)
            points_z_local = torch.matmul(points_z.permute(0, 3, 1, 2), self.sigmoid(self.group_z)).squeeze().permute(0,
                                                                                                                      2,
                                                                                                                      1)
        elif self.local_agg == 'gk':
            points_x_local = points_x * self.sigmoid(self.group_x).unsqueeze(0).unsqueeze(
                -1)  # b,g,k,c- - 1,g, k,1---b,g,k,c
            points_y_local = points_y * self.sigmoid(self.group_y).unsqueeze(0).unsqueeze(-1)
            points_z_local = points_z * self.sigmoid(self.group_z).unsqueeze(0).unsqueeze(-1)
            points_x_local = torch.mean(points_x_local, dim=-2)  # b,g,c
            points_y_local = torch.mean(points_y_local, dim=-2)
            points_z_local = torch.mean(points_z_local, dim=-2)
        elif self.local_agg == 'mean':
            points_x_local = torch.mean(points_x, dim=-2)  # b,g,c
            points_y_local = torch.mean(points_y, dim=-2)
            points_z_local = torch.mean(points_z, dim=-2)

        # test two condition: 1. layereise attention  2. channel-wise attention
        x_attention = self.mlp_x(points_x_local.permute(0,2,1)).permute(0,2,1)  # b, 1, c
        y_attention = self.mlp_y(points_y_local.permute(0,2,1)).permute(0,2,1)  # b ,1 ,c
        z_attention = self.mlp_z(points_z_local.permute(0,2,1)).permute(0,2,1)  # b, 1, c

        agg_x = points_x * x_attention.unsqueeze(-2)  # b,g,k,c ----回复到原始的b,n,c 的shape
        formal_points = cat_dim(agg_x)  # b,n,c
        points_y = index_points(formal_points, y)  # b,g,k,c
        agg_y = points_y * y_attention.unsqueeze(-2)
        formal_points = cat_dim(agg_y)
        points_z = index_points(formal_points, z)
        agg_z = points_z * z_attention.unsqueeze(-2)
        formal_points = cat_dim(agg_z)

        # all_att = x_attention * y_attention * z_attention
        # formal_points = points * all_att

        # -------------------add residual------------------
        # residual = self.mlp_residual(points)
        formal_points = formal_points + points  # residual

        return formal_points

class Layerwise3dAttention_SE(nn.Module):
    def __init__(self, inchannel, split_size=[16, 16, 32], mean=False):
        super(Layerwise3dAttention_SE, self).__init__()
        self.splitx, self.splity, self.splitz = split_size[0], split_size[1], split_size[2]
        self.mlp_x = nn.Sequential(
            nn.Conv1d(inchannel,inchannel,kernel_size=self.splitx,bias=False),
            nn.BatchNorm1d(inchannel),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(inchannel, inchannel // 4, 1, bias=False),
            nn.BatchNorm1d(inchannel // 4),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(inchannel // 4, inchannel, 1, bias=False),
            nn.Sigmoid()
        )
        self.mlp_y = nn.Sequential(
            nn.Conv1d(inchannel, inchannel, kernel_size=self.splity, bias=False),
            nn.BatchNorm1d(inchannel),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(inchannel, inchannel // 4, 1, bias=False),
            nn.BatchNorm1d(inchannel // 4),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(inchannel // 4, inchannel, 1, bias=False),
            nn.Sigmoid()
        )
        self.mlp_z = nn.Sequential(
            nn.Conv1d(inchannel, inchannel, kernel_size=self.splitz, bias=False),
            nn.BatchNorm1d(inchannel),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(inchannel, inchannel // 4, 1, bias=False),
            nn.BatchNorm1d(inchannel // 4),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(inchannel // 4, inchannel, 1, bias=False),
            nn.Sigmoid()
        )
        self.mean = mean
        if not self.mean:
            self.transformer_x = PointTransformerBlock_scaler(dim=inchannel, normalize=False)
            self.transformer_y = PointTransformerBlock_scaler(dim=inchannel, normalize=False)
            self.transformer_z = PointTransformerBlock_scaler(dim=inchannel, normalize=False)
        # self.mlp_residual = nn.Sequential(
        #     nn.Linear(inchannel,inchannel,bias=False),
        #     # nn.BatchNorm1d(inchannel),
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, xyz, points):
        """

        :param xyz: b,n,3
        :param points: b,n,c
        :return: 3d attention feats
        """
        # b,g,k   ----k = n / g(group)
        point_x, index_x = split_byxyz_net(xyz, points, mode='x', split_size=self.splitx)
        point_y, index_y = split_byxyz_net(xyz, points, mode='y', split_size=self.splity)
        point_z, index_z = split_byxyz_net(xyz, points, mode='z', split_size=self.splitz)

        x = index_x.long()  # b, g,k
        y = index_y.long()  # b,g,k
        z = index_z.long()
        points_x = point_x  # b,g,k,c
        points_y = point_y
        points_z = point_z

        x_xyz = index_points(xyz, x)
        y_xyz = index_points(xyz, y)
        z_xyz = index_points(xyz, z)

        # group-wise spatial attention    and   group-wise channel attention
        if self.mean == True:
            points_x_local = torch.mean(points_x, dim=-2)  # b,g,c
            points_y_local = torch.mean(points_y, dim=-2)  # b,g,c
            points_z_local = torch.mean(points_z, dim=-2)  # b,g,c
        else:

            points_x_local = self.transformer_x(points_x, x_xyz)  # b,g,c
            points_y_local = self.transformer_y(points_y, y_xyz)
            points_z_local = self.transformer_z(points_z, z_xyz)

        # points_x_local = torch.mean(points_x_local, dim=1, keepdim=True)  # b,g,c --> b , 1, c
        # points_x_local = points_x_local.permute(0, 2, 1)  # b, c , 1
        #
        # points_y_local = torch.mean(points_y_local, dim=1, keepdim=True)  # b,g,c --> b , 1, c
        # points_y_local = points_y_local.permute(0, 2, 1)  # b, c , 1
        #
        # points_z_local = torch.mean(points_z_local, dim=1, keepdim=True)  # b,g,c --> b , 1, c
        # points_z_local = points_z_local.permute(0, 2, 1)  # b, c , 1

        # test two condition: 1. layereise attention  2. channel-wise attention
        x_attention = self.mlp_x(points_x_local.permute(0, 2, 1) )  # b, c, 1
        y_attention = self.mlp_y(points_y_local.permute(0, 2, 1) )
        z_attention = self.mlp_z(points_z_local.permute(0, 2, 1) )

        agg_x = points_x * x_attention.permute(0, 2, 1).unsqueeze(-2)  # b,g,k,c ----回复到原始的b,n,c 的shape
        formal_points = cat_dim(agg_x)  # b,n,c
        points_y = index_points(formal_points, y)  # b,g,k,c
        agg_y = points_y * y_attention.permute(0, 2, 1).unsqueeze(-2)
        formal_points = cat_dim(agg_y)
        points_z = index_points(formal_points, z)
        agg_z = points_z * z_attention.permute(0, 2, 1).unsqueeze(-2)
        formal_points = cat_dim(agg_z)

        # all_att = x_attention * y_attention * z_attention
        # formal_points = points * all_att

        # -------------------add residual------------------
        # residual = self.mlp_residual(points)
        formal_points = formal_points + points  # residual

        return formal_points


class Layerwise3dAttention_channel_spatial(nn.Module):
    def __init__(self, inchannel, split_size=[16, 16, 32], mean=True):
        super(Layerwise3dAttention_channel_spatial, self).__init__()
        self.splitx, self.splity, self.splitz = split_size[0], split_size[1], split_size[2]
        # channel-wise
        self.mlp_x_channel = nn.Sequential(
            nn.Conv1d(self.splitx, self.splitx, 1, bias=False),
            nn.BatchNorm1d(self.splitx),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.splitx, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.mlp_y_channel = nn.Sequential(
            nn.Conv1d(self.splity, self.splity, 1, bias=False),
            nn.BatchNorm1d(self.splity),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.splity, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.mlp_z_channnel = nn.Sequential(
            nn.Conv1d(self.splitz, self.splitz, 1, bias=False),
            nn.BatchNorm1d(self.splitz),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.splitz, 1, 1, bias=False),
            nn.Sigmoid()
        )
        # spatial-wise
        self.mlp_x_spatial = nn.Sequential(
            nn.Conv1d(inchannel, inchannel, 1, bias=False),
            nn.BatchNorm1d(inchannel),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(inchannel, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.mlp_y_spatial = nn.Sequential(
            nn.Conv1d(inchannel, inchannel, 1, bias=False),
            nn.BatchNorm1d(inchannel),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(inchannel, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.mlp_z_spatial = nn.Sequential(
            nn.Conv1d(inchannel, inchannel, 1, bias=False),
            nn.BatchNorm1d(inchannel),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(inchannel, 1, 1, bias=False),
            nn.Sigmoid()
        )

        self.mean = mean
        if not self.mean:
            self.transformer_x = PointTransformerBlock_scaler(dim=inchannel, normalize=False)
            self.transformer_y = PointTransformerBlock_scaler(dim=inchannel, normalize=False)
            self.transformer_z = PointTransformerBlock_scaler(dim=inchannel, normalize=False)
        # self.mlp_residual = nn.Sequential(
        #     nn.Linear(inchannel,inchannel,bias=False),
        #     # nn.BatchNorm1d(inchannel),
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, xyz, points):
        """

        :param xyz: b,n,3
        :param points: b,n,c
        :return: 3d attention feats
        """
        # b,g,k   ----k = n / g(group)
        point_x, index_x = split_byxyz_net(xyz, points, mode='x', split_size=self.splitx)
        point_y, index_y = split_byxyz_net(xyz, points, mode='y', split_size=self.splity)
        point_z, index_z = split_byxyz_net(xyz, points, mode='z', split_size=self.splitz)

        x = index_x.long()  # b, g,k
        y = index_y.long()  # b,g,k
        z = index_z.long()
        points_x = point_x  # b,g,k,c
        points_y = point_y
        points_z = point_z

        x_xyz = index_points(xyz, x)
        y_xyz = index_points(xyz, y)
        z_xyz = index_points(xyz, z)

        # group-wise spatial attention    and   group-wise channel attention
        if self.mean == True:
            points_x_local = torch.mean(points_x, dim=-2)  # b,g,c
            points_y_local = torch.mean(points_y, dim=-2)  # b,g,c
            points_z_local = torch.mean(points_z, dim=-2)  # b,g,c
        else:

            points_x_local = self.transformer_x(points_x, x_xyz)
            points_y_local = self.transformer_y(points_y, y_xyz)
            points_z_local = self.transformer_z(points_z, z_xyz)

        # test two condition: 1. channel-wise attention  2. spatial attention
        x_attention_channel = self.mlp_x_channel(points_x_local)  # b,g,c---> b, 1, c
        y_attention_channel = self.mlp_y_channel(points_y_local)  # b ,1 ,c
        z_attention_channel = self.mlp_z_channnel(points_z_local)  # b, 1, c

        agg_x = points_x * x_attention_channel.unsqueeze(-2)  # b,g,k,c ----回复到原始的b,n,c 的shape
        formal_points = cat_dim(agg_x)  # b,n,c
        points_y = index_points(formal_points, y)  # b,g,k,c
        agg_y = points_y * y_attention_channel.unsqueeze(-2)
        formal_points = cat_dim(agg_y)
        points_z = index_points(formal_points, z)
        agg_z = points_z * z_attention_channel.unsqueeze(-2)
        formal_points = cat_dim(agg_z)

        # spatial attention
        x_attention_spatial = self.mlp_x_spatial(points_x_local.permute(0, 2, 1))  # b,g,c-->b,c,g-->b,1,g
        y_attention_spatial = self.mlp_y_spatial(points_y_local.permute(0, 2, 1))  # b ,1 ,g
        z_attention_spatial = self.mlp_z_spatial(points_z_local.permute(0, 2, 1))  # b, 1, g

        x_attention_spatial = x_attention_spatial.permute(0, 2, 1)  # b,1,g  -->b,g,1
        y_attention_spatial = y_attention_spatial.permute(0, 2, 1)  # b ,1 ,g
        z_attention_spatial = z_attention_spatial.permute(0, 2, 1)  # b, 1, g

        agg_x_s = points_x * x_attention_spatial.unsqueeze(-1)  # b,g,k,c ----回复到原始的b,n,c 的shape
        formal_points_s = cat_dim(agg_x_s)  # b,n,c
        points_y = index_points(formal_points_s, y)  # b,g,k,c
        agg_y_s = points_y * y_attention_spatial.unsqueeze(-1)
        formal_points_s = cat_dim(agg_y_s)
        points_z = index_points(formal_points_s, z)
        agg_z_s = points_z * z_attention_spatial.unsqueeze(-1)
        formal_points_s = cat_dim(agg_z_s)

        # -------------------add residual------------------
        # residual = self.mlp_residual(points)
        formal_points = formal_points + points + formal_points_s

        return formal_points


def get_graph_feature(x, k=20, idx=None, dim9=False):
    # x : B,C,N
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)  # flatten

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  # (batch_size, 2*num_dims, num_points, k)


def get_graph_feature_withdilation(x, d, k=20, idx=None, dim9=False):
    # x : B,C,N
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx,dis = knn_with_dilation(x, k=k, d=d)  # (batch_size, num_points, k)
        else:
            idx,dis = knn_with_dilation(x[:, 6:], k=k, d=d)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)  # flatten

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  # (batch_size, 2*num_dims, num_points, k)

def get_graph_feature_withdilation_and_localshape(conv,withshape,x,xyz, d, k=20, idx_xyz=None, dim9=False):
    # x : B,C,N  xyz:b,3,n
    batch_size = x.size(0)
    num_dims = x.size(1)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    tmp = x.permute(0,2,1) # b,n,c
    if dim9 == False:
        if len(d) == 1:
            idx,dis = knn_with_dilation(x, k=k, d=d[0])  # (batch_size, num_points, k)
            if idx_xyz is None:
                idxxyz = knn_with_dilation(xyz, k=k, d=d[0])  # (batch_size, num_points, k)
        else:
            idx = knn_with_dilation_multiple(x, k=k, d=d)  # (batch_size, num_points, k)
            if idx_xyz is None:
                idxxyz = knn_with_dilation_multiple(xyz, k=k, d=d)  # (batch_size, num_points, k)
    else:
        if len(d) == 1:
            idx ,dis= knn_with_dilation(x, k=k, d=d[0])  # (batch_size, num_points, k)
            if idx_xyz is None:
                idxxyz = knn_with_dilation(xyz, k=k, d=d[0])  # (batch_size, num_points, k)
        else:
            idx = knn_with_dilation_multiple(x, k=k, d=d)  # (batch_size, num_points, k)
            if idx_xyz is None:
                idxxyz = knn_with_dilation_multiple(xyz, k=k, d=d)  # (batch_size, num_points, k)
    device = x.device
    if idx_xyz is not None:
        idxxyz = idx_xyz

    feature = idx_pt(x.permute(0,2,1),idx)
    # feature = feature.view(batch_size, num_points, len(d) * k, num_dims)
    xyz_k = idx_pt(xyz.permute(0,2,1),idx) # b,n,k,c
    xyz_k_r = xyz_k.permute(0,3,1,2) - xyz.unsqueeze(-1)  # b,3,n,k
    x = x.permute(0,2,1).view(batch_size, num_points, 1, num_dims).repeat(1, 1, len(d) * k, 1) # b,n,k,c
    feature_r = feature - x   # b,n,k,c
    cos = localfeaturecosineagg(feature,tmp).squeeze(-1)  # b,n,k
    # cos = 1 - cos
    learn_cos = conv(cos).unsqueeze(-1)  # b,n,k,1
    feature_r = feature_r * learn_cos + feature_r
    # feature_r = torch.cat((feature_r,feature_r * learn_cos),dim=-1)  # b,n,k,2c
    # feature_r= feature - feature * torch.nn.functional.normalize(x,p=2,dim=-1) * torch.nn.functional.normalize(x,p=2,dim=-1)
    # feature_r_r = conv(feature_r.permute(0,3,1,2)).permute(0,2,3,1) # b,c,n,k -> b,n,k,c
    # weight = torch.sigmoid(feature_r_r)     #     sigmoid channel attention
    # feature_r = feature_r * weight
    # feature_r = feature - x * feature * torch.nn.functional.normalize(feature,p=2,dim=-1)   # N - (N ` C) ` N / |N|   b,n,k,c
    # feature_r = x - x * feature * torch.nn.functional.normalize(x,p=2,dim=-1) # c - n`c `c/|C|
    # center_r = x - x * feature * torch.nn.functional.normalize(feature,p=2,dim=-1) # b,n,k,c
    dila = True if len(d) ==3 else False
    # local_shape = conv_geo(xyz_k_r)  # b,c,n,k
    # feature_with_localshape = feature_r.permute(0, 3, 1, 2) +  local_shape

    local_shape,local_dis,knn_xyz_norm = LocalFeatureRepresentaion_polar(xyz.permute(0,2,1),xyz_k,return_dis=False) # b,n,k,13
    # feature_reweight = localcossinagg(feature_r.permute(0,3,1,2),knn_xyz_norm,local_dis)
    # feature_reweight = localdistanceagg(feature_r.permute(0,3,1,2), local_dis) # b,c,n,k
    if withshape:
        feature_with_localshape = torch.cat((feature_r.permute(0,3,1,2),local_shape.permute(0,3,1,2)),dim=1)  # b,c+12,n,k
    else:
        feature_with_localshape = feature_r.permute(0,3,1,2)
    # feature_with_localshape = feature_with_localshape  * learn_cos.permute(0,3,1,2) + feature_with_localshape
    # cos = localfeaturecosineagg(feature,tmp)  # b,n,k,1
    # feature_with_localshape = feature_with_localshape * cos.permute(0,3,1,2)


    return  feature_with_localshape# (batch_size, 3*num_dims, num_points, k)

def get_graph_feature_withdilation_and_localshape_src_dst(conv,withshape,downindex,x_src,x_dst,xyz_src,xyz_dst, d, k=20, idx_xyz=None, dim9=False):
    # x_src : B,C,m  x_dst:b,c,n   xyz_src:b,3,m   xyz_dst:b,3,n
    batch_size = x_src.size(0)
    num_dims = x_src.size(1)
    num_points = x_src.size(2)
    tmp = x_dst.permute(0,2,1) # b,n,c
    if dim9 == False:
        if len(d) == 1:
            idx,dis = knn_with_dilation_src_dst(x_dst,x_dst, k=k, d=d[0])  # b,n,k
            if idx_xyz is None:
                idxxyz = knn_with_dilation_src_dst(x_src,x_dst, k=k, d=d[0])  #
        else:
            idx = knn_with_dilation_multiple(x_src, k=k, d=d)  #
            if idx_xyz is None:
                idxxyz = knn_with_dilation_multiple(x_src, k=k, d=d)  #
    else:
        if len(d) == 1:
            idx ,dis= knn_with_dilation_src_dst(x_dst,x_dst, k=k, d=d[0])  #
            if idx_xyz is None:
                idxxyz = knn_with_dilation_src_dst(x_src,x_dst, k=k, d=d[0])  #
        else:
            idx = knn_with_dilation_multiple(x_src, k=k, d=d)  #
            if idx_xyz is None:
                idxxyz = knn_with_dilation_multiple(x_src, k=k, d=d)  # (
    device = x_src.device
    if idx_xyz is not None:
        idxxyz = idx_xyz

    feature = idx_pt(x_dst.permute(0,2,1),idx) # b,n,k,c  根据dis在dst中找neighbors
    feature = index_points(feature,downindex)  # b,m,k,c   根据downindex找到需要的中心点
    # tmp = index_points(tmp,downindex) # b,m,c   这个就是x_src
    # a = torch.sum(tmp - x_src.permute(0,2,1))
    tmp = x_src.permute(0,2,1)  # b,m,c
    # feature = feature.view(batch_size, num_points, len(d) * k, num_dims)
    xyz_k = idx_pt(xyz_dst.permute(0,2,1),idx) # b,n,k,3
    xyz_k = index_points(xyz_k,downindex)  # b,m,k,3
    xyz_k_r = xyz_k.permute(0,3,1,2) - xyz_src.unsqueeze(-1)  # b,3,m,k
    x_src_repeat = x_src.permute(0,2,1).view(batch_size, num_points, 1, num_dims).repeat(1, 1, len(d) * k, 1) # b,m,k,c
    feature_r = feature - x_src_repeat   # b,m,k,c
    cos  = localfeaturecosineagg(feature,tmp).squeeze(-1)# b,m,k,1---bmk
    # cos = localrelativefeaturecosineagg(feature_r) # b,m,k,k
    learn_cos = conv(cos).unsqueeze(-1)
    feature_r = feature_r * learn_cos + feature_r
    # b,m,k,k = cos.shape
    # learn_cos = conv(cos.view(b,m,-1).contiguous()).view(b,m,k,k).contiguous()  # b,m,k*k---- b,m,k,k
    # learn_cos = F.softmax(learn_cos,dim=-1)
    # feature_r = torch.matmul(learn_cos , feature_r) + feature_r
    # feature_r = feature - x * feature * torch.nn.functional.normalize(feature,p=2,dim=-1)   # N - (N ` C) ` N / |N|   b,n,k,c
    # feature_r = x - x * feature * torch.nn.functional.normalize(x,p=2,dim=-1) # c - n`c `c/|C|
    # center_r = x - x * feature * torch.nn.functional.normalize(feature,p=2,dim=-1) # b,n,k,c
    dila = True if len(d) ==3 else False
    # local_shape = conv_geo(xyz_k_r)  # b,c,n,k
    # feature_with_localshape = feature_r.permute(0, 3, 1, 2) +  local_shape

    local_shape,local_dis,knn_xyz_norm = LocalFeatureRepresentaion_polar(xyz_src.permute(0,2,1),xyz_k,return_dis=False) # b,n,k,13
    # feature_reweight = localcossinagg(feature_r.permute(0,3,1,2),knn_xyz_norm,local_dis)
    # feature_reweight = localdistanceagg(feature_r.permute(0,3,1,2), local_dis) # b,c,n,k
    if withshape:
        feature_with_localshape = torch.cat((feature_r.permute(0,3,1,2),local_shape.permute(0,3,1,2)),dim=1)  # b,c+12,m,k
    else:
        feature_with_localshape = feature_r.permute(0,3,1,2)
    # feature_with_localshape = feature_with_localshape + feature_with_localshape * learn_cos.permute(0,3,1,2)
    # cos = localfeaturecosineagg(feature,tmp)  # b,n,k,1
    # feature_with_localshape = feature_with_localshape * cos.permute(0,3,1,2)


    return  feature_with_localshape# (batch_size, 3*num_dims, num_points, k)


def localcossinagg(local_feature,local_knn_norm,local_dis):
    """
    local_feature:b,c,n,k
    local_knn_norm:b,n,k,3
    local_dis:b,n,k,1
    """
    local_x = local_knn_norm[:,:,:,0].unsqueeze(-1)  # b,n,k,1
    cosx_sq = (local_x / (local_dis + 1e-8)) ** 2 #b,n,k,1

    local_y = local_knn_norm[:, :, :, 1].unsqueeze(-1)  # b,n,k,1
    cosy_sq = (local_y / (local_dis + 1e-8)) ** 2  # b,n,k,1

    local_z = local_knn_norm[:, :, :, 2].unsqueeze(-1)  # b,n,k,1
    cosz_sq = (local_z / (local_dis + 1e-8)) ** 2  # b,n,k,1

    agg_x = local_feature * cosx_sq.permute(0,3,1,2)
    agg_y = local_feature * cosy_sq.permute(0,3,1,2)
    agg_z = local_feature * cosz_sq.permute(0,3,1,2)
    out = torch.cat((agg_x,agg_y,agg_z),dim=1)  # b,3c, n,k

    return out


def localfeaturecosineagg(local_feature,center_feature):
    """
    local_feature:b,n,k,c
    center_feature:b,n,c
    """
    local_norm = torch.nn.functional.normalize(local_feature,p=2,dim=-1)  # b,n,k,c
    cemter_norm = torch.nn.functional.normalize(center_feature,p=2,dim=-1) # b,n,c
    cos = torch.matmul(local_norm, cemter_norm.unsqueeze(-1))  #b,n,k,1
    return cos

def localfeaturecosineagg_reverse(local_feature,center_feature):
    """
    local_feature:b,n,k,c
    center_feature:b,n,c
    """
    local_norm = torch.nn.functional.normalize(local_feature,p=2,dim=-1)  # b,n,k,c
    cemter_norm = torch.nn.functional.normalize(center_feature,p=2,dim=-1) # b,n,c
    cos = torch.matmul(local_norm, cemter_norm.unsqueeze(-1))  #b,n,k,1
    return cos

def localrelativefeaturecosineagg(local_feature):
    """
    local_feature:b,n,k,c
    center_feature:b,n,c
    """
    local_norm = torch.nn.functional.normalize(local_feature,p=2,dim=-1)  # b,n,k,c

    cos = torch.matmul(local_norm, local_norm.permute(0,1,3,2))  #b,n,k,k
    return cos


def localdistanceagg(local_feature,local_dis):
    """
    local_feature:b,c,n,k
    local_dis:b,n,k,1
    """

    local_weight = 1 - local_dis / (local_dis.sum(dim=-2,keepdim=True) + 1e-8)  # b,n,k,1
    out = local_weight.permute(0,3,1,2) * local_feature
    return out


def get_graph_feature_withdilation_and_localshape_noconv(x,xyz, d, k=20, idx_xyz=None, dim9=False):
    # x : B,C,N  xyz:b,3,n
    batch_size = x.size(0)
    num_dims = x.size(1)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    tmp = x.permute(0,2,1) # b,n,c
    if dim9 == False:
        if len(d) == 1:
            idx,dis = knn_with_dilation(x, k=k, d=d[0])  # (batch_size, num_points, k)
            if idx_xyz is None:
                idxxyz = knn_with_dilation(xyz, k=k, d=d[0])  # (batch_size, num_points, k)
        else:
            idx = knn_with_dilation_multiple(x, k=k, d=d)  # (batch_size, num_points, k)
            if idx_xyz is None:
                idxxyz = knn_with_dilation_multiple(xyz, k=k, d=d)  # (batch_size, num_points, k)
    else:
        if len(d) == 1:
            idx ,dis= knn_with_dilation(x, k=k, d=d[0])  # (batch_size, num_points, k)
            if idx_xyz is None:
                idxxyz = knn_with_dilation(xyz, k=k, d=d[0])  # (batch_size, num_points, k)
        else:
            idx = knn_with_dilation_multiple(x, k=k, d=d)  # (batch_size, num_points, k)
            if idx_xyz is None:
                idxxyz = knn_with_dilation_multiple(xyz, k=k, d=d)  # (batch_size, num_points, k)
    device = x.device
    if idx_xyz is not None:
        idxxyz = idx_xyz

    feature = idx_pt(x.permute(0,2,1),idx)  # b,n,k,c
    # feature = feature.view(batch_size, num_points, len(d) * k, num_dims)
    xyz_k = idx_pt(xyz.permute(0,2,1),idx) # b,n,k,c
    x = x.permute(0,2,1).unsqueeze(-1) # b,n,c,1
    # feature_r = feature - x  # b,n,k,c
    dila = True if len(d) ==3 else False
    # feature = conv_sem(feature_r.permute(0,3,1,2))  # b,c,n,k
    # local_shape = LocalFeatureRepresentaion_cylinder(xyz.permute(0,2,1),xyz_k,nsample=20,dila3 = dila)
    # local_shape_lift = conv_geo(local_shape.permute(0,3,1,2)).permute(0,2,1,3)#b,c,n,k--b,n,c,k
    fea_nor = torch.nn.functional.normalize(feature,p=2,dim=-1)# b,n,k,c
    x_nor = torch.nn.functional.normalize(x,p=2,dim=-2)  # b,n,c,1
    # local_shape_lift_nor = torch.nn.functional.normalize(local_shape_lift,p=2,dim=-2)# b,n,c,k
    corr = F.relu(torch.matmul(fea_nor,x_nor)) # b,n,k,1

    feature_corr = torch.matmul((feature-x.permute(0,1,3,2)).permute(0,1,3,2),corr).repeat(1,1,1,feature.size(2))  # b,n,c,k * b,n,k,1 ---b,n,c,1--b,n,c,k --b,n,k,c


    local_shape = LocalFeatureRepresentaion_cylinder(xyz.permute(0,2,1),xyz_k,nsample=20,dila3 = dila)  # b,n,k,13
    feature_with_localshape = torch.cat(( feature_corr.permute(0,1,3,2),local_shape), dim=3)#.permute(0, 3, 1, 2).contiguous()

      # b,n,k,k * b,n,k,c = b,n,k,c

    return feature_with_localshape.permute(0,3,1,2) # (batch_size, 3*num_dims, num_points, k)

def get_graph_feature_withdilation_and_localshape_noconv_featurerelation(x,xyz, d, k=20, idx_xyz=None, dim9=False):
    # x : B,C,N  xyz:b,3,n
    batch_size = x.size(0)
    num_dims = x.size(1)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    tmp = x.permute(0,2,1) # b,n,c
    if dim9 == False:
        if len(d) == 1:
            idx,dis = knn_with_dilation(x, k=k, d=d[0])  # (batch_size, num_points, k)
            if idx_xyz is None:
                idxxyz = knn_with_dilation(xyz, k=k, d=d[0])  # (batch_size, num_points, k)
        else:
            idx = knn_with_dilation_multiple(x, k=k, d=d)  # (batch_size, num_points, k)
            if idx_xyz is None:
                idxxyz = knn_with_dilation_multiple(xyz, k=k, d=d)  # (batch_size, num_points, k)
    else:
        if len(d) == 1:
            idx ,dis= knn_with_dilation(x, k=k, d=d[0])  # (batch_size, num_points, k)
            if idx_xyz is None:
                idxxyz = knn_with_dilation(xyz, k=k, d=d[0])  # (batch_size, num_points, k)
        else:
            idx = knn_with_dilation_multiple(x, k=k, d=d)  # (batch_size, num_points, k)
            if idx_xyz is None:
                idxxyz = knn_with_dilation_multiple(xyz, k=k, d=d)  # (batch_size, num_points, k)
    device = x.device
    if idx_xyz is not None:
        idxxyz = idx_xyz

    feature = idx_pt(x.permute(0,2,1),idx) # b,n,k,c
    # feature = feature.view(batch_size, num_points, len(d) * k, num_dims)
    xyz_k = idx_pt(xyz.permute(0,2,1),idx) # b,n,k,c
    x = x.permute(0,2,1).unsqueeze(-1) # b,n,c,1
    feature_r = feature - x.permute(0,1,3,2)  # b,n,k,c
    dila = True if len(d) ==3 else False
    # feature = conv_sem(feature_r.permute(0,3,1,2))  # b,c,n,k
    # local_shape = LocalFeatureRepresentaion_cylinder(xyz.permute(0,2,1),xyz_k,nsample=20,dila3 = dila)
    # local_shape_lift = conv_geo(local_shape.permute(0,3,1,2)).permute(0,2,1,3)#b,c,n,k--b,n,c,k
    fea_r_nor = torch.nn.functional.normalize(feature_r,p=2,dim=-1)# b,n,k,c
    x_nor = torch.nn.functional.normalize(x,p=2,dim=-2)  # b,n,c,1
    # local_shape_lift_nor = torch.nn.functional.normalize(local_shape_lift,p=2,dim=-2)# b,n,c,k
    corr = F.relu(torch.matmul(fea_r_nor,x_nor)) # b,n,k,1
    corr = 1 - corr

    feature_corr = torch.matmul(feature.permute(0,1,3,2),corr).repeat(1,1,1,feature.size(2))  # b,n,c,k * b,n,k,1 ---b,n,c,1--b,n,c,k --b,n,k,c


    local_shape = LocalFeatureRepresentaion_cylinder(xyz.permute(0,2,1),xyz_k,nsample=20,dila3 = dila)  # b,n,k,13
    feature_with_localshape = torch.cat(( feature_corr.permute(0,1,3,2),local_shape), dim=3)#.permute(0, 3, 1, 2).contiguous()

      # b,n,k,k * b,n,k,c = b,n,k,c

    return feature_with_localshape.permute(0,3,1,2) # (batch_size, 3*num_dims, num_points, k)

def get_graph_feature_withdilation_catlocalshape_attentionpooling(x,xyz, d, k=20, idx_xyz=None, dim9=False):
    # x : B,C,N  xyz:b,3,n
    batch_size = x.size(0)
    num_dims = x.size(1)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    tmp = x.permute(0,2,1) # b,n,c
    if dim9 == False:
        if len(d) == 1:
            idx ,dis= knn_with_dilation(x, k=k, d=d[0])  # (batch_size, num_points, k)
            # if idx_xyz is None:
            #     idxxyz = knn_with_dilation(xyz, k=k, d=d[0])  # (batch_size, num_points, k)
        else:
            idx = knn_with_dilation_multiple(x, k=k, d=d)  # (batch_size, num_points, k)
            # if idx_xyz is None:
            #     idxxyz = knn_with_dilation_multiple(xyz, k=k, d=d)  # (batch_size, num_points, k)
    else:
        if len(d) == 1:
            idx ,dis= knn_with_dilation(x, k=k, d=d[0])  # (batch_size, num_points, k)
            # if idx_xyz is None:
            #     idxxyz = knn_with_dilation(xyz, k=k, d=d[0])  # (batch_size, num_points, k)
        else:
            idx = knn_with_dilation_multiple(x, k=k, d=d)  # (batch_size, num_points, k)
            # if idx_xyz is None:
            #     idxxyz = knn_with_dilation_multiple(xyz, k=k, d=d)  # (batch_size, num_points, k)
    device = x.device
    if idx_xyz is not None:
        idxxyz = idx_xyz

    feature_k = idx_pt(x.permute(0,2,1),idx)#  b,n,k,c
    xyz_k = idx_pt(xyz.permute(0,2,1),idx) # b,n,k,c
    x = x.permute(0,2,1).view(batch_size, num_points, 1, num_dims).repeat(1, 1, len(d) * k, 1)
    feature_r = feature_k - x  # b,n,k,c   local feature relation
    dila = True if len(d) ==3 else False
    local_shape = LocalFeatureRepresentaion_cylinder(xyz.permute(0,2,1),xyz_k,nsample=20,dila3 = dila)
    # local_shape = LocalFeatureRepresentaion_polar(xyz.permute(0,2,1),xyz_k)
    # local_shape = LocalFeatureRepresentaion_1(xyz.permute(0,2,1),xyz_k)
    feature = torch.cat(( feature_r,local_shape), dim=3).permute(0, 3, 1, 2).contiguous()
    if len(d) == 1:
        return feature ,xyz_k,dis # (batch_size, 3*num_dims, num_points, k)
    else:
        return feature ,xyz_k


def get_graph_feature_inner_boundary_neighbour(x,xyz, d, k=20, idx_xyz=None, dim9=False):
    # x : B,C,N  xyz:b,3,n
    batch_size = x.size(0)
    num_dims = x.size(1)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    tmp = x.permute(0,2,1) # b,n,c
    if dim9 == False:
        if len(d) == 1:
            idx ,dis= knn_with_dilation(x, k=k, d=d[0])  # (batch_size, num_points, k)
            # if idx_xyz is None:
            #     idxxyz = knn_with_dilation(xyz, k=k, d=d[0])  # (batch_size, num_points, k)
        else:
            idx = knn_with_dilation_multiple(x, k=k, d=d)  # (batch_size, num_points, k)
            # if idx_xyz is None:
            #     idxxyz = knn_with_dilation_multiple(xyz, k=k, d=d)  # (batch_size, num_points, k)
    else:
        if len(d) == 1:
            idx ,dis= knn_with_dilation(x, k=k, d=d[0])  # (batch_size, num_points, k)
            # if idx_xyz is None:
            #     idxxyz = knn_with_dilation(xyz, k=k, d=d[0])  # (batch_size, num_points, k)
        else:
            idx = knn_with_dilation_multiple(x, k=k, d=d)  # (batch_size, num_points, k)
            # if idx_xyz is None:
            #     idxxyz = knn_with_dilation_multiple(xyz, k=k, d=d)  # (batch_size, num_points, k)
    device = x.device
    if idx_xyz is not None:
        idxxyz = idx_xyz

    feature_k = idx_pt(x.permute(0,2,1),idx)#  b,n,k,c
    feature_k_nor = torch.nn.functional.normalize(feature_k,p=2,dim=-1)
    xyz_k = idx_pt(xyz.permute(0,2,1),idx) # b,n,k,c
    x = x.permute(0,2,1).view(batch_size, num_points, 1, num_dims) #center feature - b,n,1,c
    x_nor = torch.nn.functional.normalize(x,p=2,dim=-1)  #normalized center feature
    # feature_r = feature_k - x  # b,n,k,c   local feature relation
    dila = True if len(d) ==3 else False
    local_relation = F.relu(torch.matmul(feature_k_nor,x_nor.permute(0,1,3,2))).squeeze(-1)  # b,n,k,1--b,n,k
    relation = local_relation == 0  # if 0 exits, boundary point
    inner_index = relation.sum(-1) == 0  #b,n
    bound_index = relation.sum(-1) != 0  # b,n
    inner_feature_k = feature_k * inner_index.unsqueeze(-1).unsqueeze(-1)  # b,n,k,c
    bound_feature_k = feature_k * bound_index.unsqueeze(-1).unsqueeze(-1)
    inner_center = x.squeeze(-2) * inner_index.unsqueeze(-1)  # b,n,c
    bound_center = x.squeeze(-2) * bound_index.unsqueeze(-1)
    # local_shape = LocalFeatureRepresentaion_cylinder(xyz.permute(0,2,1),xyz_k,nsample=20,dila3 = dila)
    local_shape = LocalFeatureRepresentaion_polar(xyz.permute(0,2,1),xyz_k)  # b,n,k,13
    # local_shape = LocalFeatureRepresentaion_1(xyz.permute(0,2,1),xyz_k)
    # feature = torch.cat(( feature_r,local_shape), dim=3).permute(0, 3, 1, 2).contiguous()


    return inner_feature_k,bound_feature_k,inner_center,bound_center,local_shape,inner_index,bound_index,local_relation


def get_graph_feature_inner_boundary_neighbour_localrealation(x,xyz, d, k=20, idx_xyz=None, dim9=False):
    # x : B,C,N  xyz:b,3,n
    batch_size = x.size(0)
    num_dims = x.size(1)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    tmp = x.permute(0,2,1) # b,n,c
    if dim9 == False:
        if len(d) == 1:
            idx ,dis= knn_with_dilation(x, k=k, d=d[0])  # (batch_size, num_points, k)
            # if idx_xyz is None:
            #     idxxyz = knn_with_dilation(xyz, k=k, d=d[0])  # (batch_size, num_points, k)
        else:
            idx ,dis= knn_with_dilation_multiple(x, k=k, d=d)  # (batch_size, num_points, k)
            # if idx_xyz is None:
            #     idxxyz = knn_with_dilation_multiple(xyz, k=k, d=d)  # (batch_size, num_points, k)
    else:
        if len(d) == 1:
            idx ,dis= knn_with_dilation(x, k=k, d=d[0])  # (batch_size, num_points, k)
            # if idx_xyz is None:
            #     idxxyz = knn_with_dilation(xyz, k=k, d=d[0])  # (batch_size, num_points, k)
        else:
            idx ,dis= knn_with_dilation_multiple(x, k=k, d=d)  # (batch_size, num_points, k)
            # if idx_xyz is None:
            #     idxxyz = knn_with_dilation_multiple(xyz, k=k, d=d)  # (batch_size, num_points, k)
    device = x.device
    if idx_xyz is not None:
        idxxyz = idx_xyz

    feature_k = idx_pt(x.permute(0,2,1),idx)#  b,n,k,c
    # feature_k_nor = torch.nn.functional.normalize(feature_k,p=2,dim=-1)
    xyz_k = idx_pt(xyz.permute(0,2,1),idx) # b,n,k,c
    x = x.permute(0,2,1).view(batch_size, num_points, 1, num_dims) #center feature - b,n,1,c
    # x_nor = torch.nn.functional.normalize(x,p=2,dim=-1)  #normalized center feature
    feature_r = feature_k - x  # b,n,k,c   local feature relation
    feature_r_nor = torch.nn.functional.normalize(feature_r,p=2,dim=-1) # b,n,k,c
    dila = True if len(d) ==3 else False
    local_relation = F.relu(torch.matmul(feature_r_nor,feature_r_nor.permute(0,1,3,2)).view(batch_size,num_points,-1)) # b,n,k,c--b,n,c,k---b,n,k,k---b,n,k*k
    relation = local_relation == 0  # if 0 exits, boundary point
    inner_index = relation.sum(-1) == 0  #b,n
    bound_index = relation.sum(-1) != 0  # b,n
    inner_feature_k = feature_k * inner_index.unsqueeze(-1).unsqueeze(-1)  # b,n,k,c
    bound_feature_k = feature_k * bound_index.unsqueeze(-1).unsqueeze(-1)
    inner_center = x.squeeze(-2) * inner_index.unsqueeze(-1)  # b,n,c
    bound_center = x.squeeze(-2) * bound_index.unsqueeze(-1)
    # local_shape = LocalFeatureRepresentaion_cylinder(xyz.permute(0,2,1),xyz_k,nsample=20,dila3 = dila)
    local_shape = LocalFeatureRepresentaion_polar(xyz.permute(0,2,1),xyz_k)  # b,n,k,13
    # local_shape = LocalFeatureRepresentaion_1(xyz.permute(0,2,1),xyz_k)
    # feature = torch.cat(( feature_r,local_shape), dim=3).permute(0, 3, 1, 2).contiguous()


    return inner_feature_k,bound_feature_k,inner_center,bound_center,local_shape,inner_index,bound_index,local_relation

def get_inner_feature(inner_feature_k,inner_center,inner_index,local_shape,conv1,conv2):
    """
    inner_index: b,n
    inner_feature_k:b,n,k,c
    inner_center:b,n,c
    local_shape:b,n,k,13
    conv:local conv and max
    """
    inner_feature_with_localshape = torch.cat((inner_feature_k,local_shape * inner_index.unsqueeze(-1).unsqueeze(-1)),dim=-1) # b,n,k,c+13
    # inner_feature_with_localshape = inner_feature_k
    out = conv1(inner_feature_with_localshape.permute(0,3,1,2)).max(dim=-1)[0] # b,c,n,k-->b,c,n
    out = out + conv2(inner_center.permute(0,2,1))  # b,c,n + b,c,n--- b,c,n
    return out

def get_bound_feature(bound_feature_k,bound_center,bound_index,local_shape,conv1,conv2,local_edge_realtion):
    """
    b_f_K:b,n,k,c
    b_c:b,n,c
    b_i:b,n
    l_s:b,n,k,13
    conv1:
    conv2:
    """
    bound_feature_with_localshape = torch.cat((bound_feature_k - bound_center.unsqueeze(-2),local_shape * bound_index.unsqueeze(-1).unsqueeze(-1)),dim=-1) # b,n,k,+13
    reverse_edge_weight = 1 - local_edge_realtion #1 - b,n,k = b,n,k
    # relative_feature = bound_feature_k - bound_center.unsqueeze(-2)  #  b,n, k,c
    reweight_local_feature = reverse_edge_weight.unsqueeze(-1) * bound_feature_with_localshape   # b,n,k,c+13
    local_feature = reweight_local_feature.sum(dim=-2)  # b,n,c+13
    out = conv1(local_feature.permute(0,2,1))
    # out = conv1(bound_feature_with_localshape.permute(0,3,1,2)).max(dim=-1)[0] # b,c,n,k ---- b,c,n
    out = out + conv2(bound_center.permute(0,2,1))  # b,c,n + b,c,n
    return out

def get_bound_feature_kreweighting(bound_feature_k,bound_center,bound_index,local_shape,conv1,conv2,local_edge_realtion):
    """
    b_f_K:b,n,k,c
    b_c:b,n,c
    b_i:b,n
    l_s:b,n,k,13
    conv1:
    conv2:
    """
    bound_feature_with_localshape = torch.cat((bound_feature_k - bound_center.unsqueeze(-2),local_shape * bound_index.unsqueeze(-1).unsqueeze(-1)),dim=-1) # b,n,k,+13
    # bound_feature_with_localshape = bound_feature_k - bound_center.unsqueeze(-2)
    reverse_edge_weight = 1 - local_edge_realtion #1 - b,n,k = b,n,k
    # relative_feature = bound_feature_k - bound_center.unsqueeze(-2)  #  b,n, k,c
    reweight_local_feature = reverse_edge_weight.unsqueeze(-1) * bound_feature_with_localshape   # b,n,k,c+13
    out = conv1(reweight_local_feature.permute(0,3,1,2)).max(dim=-1)[0]
    # out = conv1(bound_feature_with_localshape.permute(0,3,1,2)).max(dim=-1)[0] # b,c,n,k ---- b,c,n
    out = out + conv2(bound_center.permute(0,2,1))  # b,c,n + b,c,n
    return out



def get_graph_feature_withdilation_catlocalshape_attentionpooling_firstlayer(x,xyz, d, k=20, idx_xyz=None, dim9=False):
    # x : B,C,N  xyz:b,3,n
    batch_size = x.size(0)
    num_dims = x.size(1)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    tmp = x.permute(0,2,1) # b,n,c
    if dim9 == False:
        if len(d) == 1:
            idx ,dis= knn_with_dilation(xyz, k=k, d=d[0])  # (batch_size, num_points, k)
            # if idx_xyz is None:
            #     idxxyz = knn_with_dilation(xyz, k=k, d=d[0])  # (batch_size, num_points, k)
        else:
            idx = knn_with_dilation_multiple(xyz, k=k, d=d)  # (batch_size, num_points, k)
            # if idx_xyz is None:
            #     idxxyz = knn_with_dilation_multiple(xyz, k=k, d=d)  # (batch_size, num_points, k)
    else:
        # xyz = x[:,6:,:]  # dim9 True use the last 3 dimension
        if len(d) == 1:
            idx ,dis= knn_with_dilation(xyz, k=k, d=d[0])  # (batch_size, num_points, k)
            # if idx_xyz is None:
            #     idxxyz = knn_with_dilation(xyz, k=k, d=d[0])  # (batch_size, num_points, k)
        else:
            idx = knn_with_dilation_multiple(xyz, k=k, d=d)  # (batch_size, num_points, k)
            # if idx_xyz is None:
            #     idxxyz = knn_with_dilation_multiple(xyz, k=k, d=d)  # (batch_size, num_points, k)
    if idx_xyz is not None:
        idxxyz = idx_xyz

    feature_k = idx_pt(x.permute(0,2,1),idx)#  b,n,k,c

    xyz_k = idx_pt(xyz.permute(0,2,1),idx) # b,n,k,3
    x = x.permute(0,2,1).view(batch_size, num_points, 1, num_dims).repeat(1, 1, len(d) * k, 1)
    if dim9:
        feature_r = feature_k - x  # b,n,k,c   local feature relation
        feature_r = feature_r[:,:,:,:6]  # like dgcnn
        # feature_r = feature_r[:, :, :, 3:]  # like pointnet++

    dila = True if len(d) ==3 else False
    # local_shape = LocalFeatureRepresentaion_cylinder(xyz.permute(0,2,1),xyz_k,nsample=20,dila3 = dila)  # b,n,k,13
    local_shape ,local_dis,knn_xyz_norm = LocalFeatureRepresentaion_polar(xyz.permute(0,2,1),xyz_k)

    local_shape = local_shape.permute(0,3,1,2)
    if dim9:
        local_shape = torch.cat((local_shape,feature_r.permute(0,3,1,2)),dim=1) # b,13+6,n,k

    if len(d) == 1:
        return local_shape ,xyz_k,dis # (batch_size, 3*num_dims, num_points, k)
    else:
        return local_shape,xyz_k

def get_graph_feature_withdilation_catlocalshape_attentionpooling_firstlayer_src_dst(downindex,x_src,x_dst,xyz_src,xyz_dst, d, k=20, idx_xyz=None, dim9=False):
    #   xyz_src:b,3,m    xyz_dst: b,3,n    m 1024 n 4096   x_src:b,c,m   x_dst:b,c,n
    batch_size = x_src.size(0)
    num_dims = x_src.size(1)
    num_points = x_src.size(2)
    # x_dst = x_dst.view(batch_size, -1, num_points)
    tmp = x_dst.permute(0,2,1) # b,n,c
    if dim9 == False:
        if len(d) == 1:
            idx ,dis= knn_with_dilation_src_dst(xyz_dst,xyz_dst, k=k, d=d[0])  # (batch_size, n, k)  在n个点中找knn,充分利用所有的点的信息
        else:
            idx = knn_with_dilation_multiple(xyz_src, k=k, d=d)
    else:
        # xyz = x[:,6:,:]  # dim9 True use the last 3 dimension
        if len(d) == 1:
            idx ,dis= knn_with_dilation_src_dst(xyz_dst,xyz_dst, k=k, d=d[0])  # (batch_size, n, k)
        else:
            idx = knn_with_dilation_multiple(xyz_src, k=k, d=d)
    if idx_xyz is not None:
        idxxyz = idx_xyz

    feature_k = idx_pt(x_dst.permute(0,2,1),idx)#  b,n,k,c
    feature_k = index_points(feature_k,downindex) # b,m,k,c

    xyz_k = idx_pt(xyz_dst.permute(0,2,1),idx) # b,n,k,3
    xyz_k = index_points(xyz_k,downindex) # b,m,k,3
    x_src_repeat = x_src.permute(0,2,1).view(batch_size, num_points, 1, num_dims).repeat(1, 1, len(d) * k, 1)  # b,m,k,c
    if dim9:
        feature_r = feature_k - x_src_repeat  # b,n,k,c   local feature relation
        feature_r = feature_r[:,:,:,:6]
    dila = True if len(d) ==3 else False
    # local_shape = LocalFeatureRepresentaion_cylinder(xyz.permute(0,2,1),xyz_k,nsample=20,dila3 = dila)  # b,n,k,13
    local_shape ,local_dis,knn_xyz_norm = LocalFeatureRepresentaion_polar(xyz_src.permute(0,2,1),xyz_k)

    local_shape = local_shape.permute(0,3,1,2)
    if dim9:
        local_shape = torch.cat((local_shape,feature_r.permute(0,3,1,2)),dim=1) # b,13+6,n,k

    if len(d) == 1:
        return local_shape ,xyz_k,dis # (batch_size, 3*num_dims, num_points, k)
    else:
        return local_shape,xyz_k



def get_graph_feature_withdilation_and_xyz_liftchannel_feature(x,xyz, d, k=20, idx=None, dim9=False):
    # x : B,C,N
    #xyz:b,3,n
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    tmp = x.permute(0,2,1) # b,n,c
    if idx is None:
        if dim9 == False:
            if len(d) == 1:
                idx = knn_with_dilation(x, k=k, d=d[0])  # (batch_size, num_points, k)
                # idxxyz = knn_with_dilation(xyz, k=k, d=d[0])  # (batch_size, num_points, k)
            else:
                idx = knn_with_dilation_multiple(x, k=k, d=d)  # (batch_size, num_points, k)
                # idxxyz = knn_with_dilation_multiple(xyz, k=k, d=d)  # (batch_size, num_points, k)
        else:
            if len(d) == 1:
                idx = knn_with_dilation(x, k=k, d=d[0])  # (batch_size, num_points, k)
                # idxxyz = knn_with_dilation(xyz, k=k, d=d[0])  # (batch_size, num_points, k)
            else:
                idx = knn_with_dilation_multiple(x[:, 6:], k=k, d=d)
                # idxxyz = knn_with_dilation_multiple(xyz[:, 6:], k=k, d=d)
    tmpidx = idx
    device = x.device
    length = len(d)
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)  # flatten

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, length * k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, length * k, 1)
    xyz_k = idx_pt(xyz.permute(0,2,1),tmpidx) # b,n,k,3
    xyz_k = xyz_k - xyz.permute(0,2,1).unsqueeze(-2)  # b,n,k,3
    xyz_k = nn.functional.normalize(xyz_k,p=2,dim=-1)# b,n,k,3
    corr_323 = torch.matmul(xyz_k,xyz_k.permute(0,1,3,2)) # ,b,n,k,k
    corr_fea = torch.matmul(corr_323,feature-x) # b,n,k,c

    feature = torch.cat((corr_fea,x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  # (batch_size, 2*num_dims, num_points, k)



def get_graph_feature_withdilation_last_catdim(x, d, k=20, idx=None, dim9=False):
    # x : B,C,N
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn_with_dilation_multiple(x, k=k, d=d)  # (batch_size, num_points, 3k)
        else:
            idx = knn_with_dilation_multiple(x[:, 6:], k=k, d=d)

    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)  # flatten

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, 3 * k, num_dims)  # b, b ,3k ,c
    feature_chunk = torch.chunk(feature, 3, dim=-2)  # tuple (b,n,k,c)
    feature_cat = torch.cat((feature_chunk[0], feature_chunk[1], feature_chunk[2]), dim=-1)  # b,n,k,3c

    x_c = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # b,n,k,c
    x_3c = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 3)  # b,n,k,c


    feature = torch.cat((feature_cat - x_3c, x_c), dim=3).permute(0, 3, 1, 2)

    return feature  # (batch_size, 4*num_dims, num_points, k)


def get_graph_feature_withdilation_last_catk(x, d, k=20, idx=None, dim9=False):
    # x : B,C,N
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn_with_dilation_multiple(x, k=k, d=d)  # (batch_size, num_points, 3k)
        else:
            idx = knn_with_dilation_multiple(x[:, 6:], k=k, d=d)

    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)  # flatten

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, len(d) * k, num_dims)  # b, b ,3k ,c


    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, len(d)*k, 1)  # b,n,3*k,c


    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  # (batch_size, 2*num_dims, num_points, 3*k)


def get_graph_feature_withdilation_last_catk_dlength2(x, d, k=20, idx=None, dim9=False):
    # x : B,C,N
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn_with_dilation_multiple_dlength2(x, k=k, d=d)  # (batch_size, num_points, 3k)
        else:
            idx = knn_with_dilation_multiple_dlength2(x[:, 6:], k=k, d=d)

    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)  # flatten

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, len(d) * k, num_dims)  # b, b ,3k ,c


    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, len(d)*k, 1)  # b,n,3*k,c


    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  # (batch_size, 2*num_dims, num_points, 3*k)

# transformer in 3k dimension  ---- use 3 trans to get three features and concat or sum
def get_graph_feature_withdilation_last_withknn_transformer(linear_op,trans_op,x, d, k=20, idx=None, dim9=False):
    # x : B,C,N
    tmp = x
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn_with_dilation_multiple(x, k=k, d=d)  # (batch_size, num_points, 3k)
        else:
            idx = knn_with_dilation_multiple(x[:, 6:], k=k, d=d)

    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)  # flatten

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, 3 * k, num_dims)  # b, n ,3k ,c

    feature = trans_op(feature).permute(0,2,1)  # b,n,c----> b,c,n
#---------------------------------------------------------------  two mode   sum and concat
    # feature = torch.cat((feature , tmp),dim=1)
    feature = feature + linear_op(tmp)
    return feature  # (batch_size,  c, n)


def get_graph_feature_withdilation_singal_transformer(linear_op,trans_op,x, d, k=20, idx=None, dim9=False):
    # x : B,C,N
    tmp = x
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn_with_dilation(x, k=k, d=d)  # (batch_size, num_points, k)
        else:
            idx = knn_with_dilation(x[:, 6:], k=k, d=d)

    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)  # flatten

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points,  k, num_dims)  # b, n ,k ,c

    feature = trans_op(feature).permute(0,2,1)  # b,n,c----> b,c,n
#---------------------------------------------------------------  two mode   sum and concat
    # feature = torch.cat((feature , tmp),dim=1)
    feature = feature + linear_op(tmp)
    return feature  # (batch_size,  c, n)


def get_graph_feature_withdilation_and_global_faeture(x, d, k=20, idx=None, dim9=False):
    # x : B,C,N
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    tmp = x# b,c,n
    if idx is None:
        if dim9 == False:
            if len(d) == 1:
                idx = knn_with_dilation(x,k,d=d[0])
            elif len(d) == 2:
                idx = knn_with_dilation_multiple_dlength2(x,k=k, d=d)
            else:
                idx = knn_with_dilation_multiple(x, k=k, d=d)  # (batch_size, num_points, 3k)
        else:
            if len(d) == 1:
                idx = knn_with_dilation(x[:, 6:],k,d=d[0])
            elif len(d) == 2:
                idx = knn_with_dilation_multiple_dlength2(x,k=k, d=d)
            else:
                idx = knn_with_dilation_multiple(x[:, 6:], k=k, d=d)

    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)  # flatten

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, len(d) * k, num_dims)  # b, n ,3k ,c


    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, len(d)*k, 1)  # b,n,3*k,c

#    avg = torch.mean(tmp,dim=-1).unsqueeze(-2).unsqueeze(-2).repeat(1,num_points,len(d)*k,1) # b,n,3*k,c

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  # (batch_size, 3*num_dims, num_points, 3*k)


class get_adaptative_graph_feature_withtransformer(nn.Module):
    def __init__(self, inchannel, k, dmax):
        super(get_adaptative_graph_feature_withtransformer, self).__init__()
        self.idx = knn_with_dilation_transformer(inchannel)
        self.k = k
        self.dmax = dmax

    def forward(self, x, dim9=False):
        # x : B,C,N
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)

        if dim9 == False:
            idx = self.idx(x, k=self.k, dmax=self.dmax)  # (batch_size, num_points, k)
        else:
            idx = self.idx(x[:, 6:], k=self.k, d=self.dmax)
        device = x.device

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

        idx = idx + idx_base

        idx = idx.view(-1)  # flatten
        idx = idx.long()
        _, num_dims, _ = x.size()

        x = x.transpose(2,
                        1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, self.k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, self.k, 1)

        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

        return feature  # (batch_size, 2*num_dims, num_points, k)


class SharedMLP(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(SharedMLP, self).__init__()
        self.conv1 = nn.Conv1d(inchannel, outchannel, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(outchannel)
        self.ac1 = nn.LeakyReLU(negative_slope=0.2)
        self.conv2 = nn.Conv1d(outchannel, outchannel, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(outchannel)
        self.ac2 = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.ac1(self.bn1(self.conv1(x)))
        x = self.ac2(self.bn2(self.conv2(x)))
        return x

class DGCNN_cls(nn.Module):
    def __init__(self, output_channels=40, normal_channel=False, type='L', use=False, use_down=False):
        super(DGCNN_cls, self).__init__()
        # self.args = args
        self.k = 20
        self.use = use
        self.d = [1, 2, 3, 4]
        if use:
            if type == 'L':
                # self.attn1 = Layerwise3dAttention_car(npoints=1024, inchannel=64, split_size=[32, 32, 32], local_agg='mean')
                self.attn2 = Layerwise3dAttention_car(npoints=1024, inchannel=64, split_size=[16, 16, 16], local_agg='mean')
                self.attn3 = Layerwise3dAttention_car(npoints=1024, inchannel=128, split_size=[8, 8, 8], local_agg='mean')
                # self.attn4 = Layerwise3dAttention_SE(npoints=1024, inchannel=256, split_size=[32, 32, 32], local_agg='mean')
                self.attn4 = Layerwise3dAttention_car(npoints=1024,inchannel=256,split_size=[4,4,4],local_agg='mean')
            else:
                self.attn1 = Transformer3dAttention(inchannel=64, split_size=[64, 64, 64])
                self.attn2 = Transformer3dAttention(inchannel=64, split_size=[64, 64, 64])
                self.attn3 = Transformer3dAttention(inchannel=128, split_size=[64, 64, 64])
                self.attn4 = Transformer3dAttention(inchannel=256, split_size=[64, 64, 64])

        self.use_down = use_down
        self.inchannel = 12 if normal_channel == True else 6

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)
        #         self.bn5 = nn.BatchNorm1d(512)
        #         self.knn_transformer_adaptaive = get_adaptative_graph_feature_withtransformer(inchannel=128,k=self.k,dmax=4)

        self.conv1 = nn.Sequential(nn.Conv2d(13, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv1_2 = nn.Sequential(nn.Conv1d(3 , 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2)

        )
        self.conv2 = nn.Sequential(nn.Conv2d(64  + 13, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2_2 = nn.Sequential(nn.Conv1d(64 , 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64  + 13, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3_2 = nn.Sequential(nn.Conv1d(64 , 128, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128  + 13, 256, kernel_size=1, bias=False),
                                   self.bn4,                                      
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4_2 = nn.Sequential(nn.Conv1d(128, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        # self.conv5 = nn.Sequential(nn.Conv1d(256, 512, kernel_size=1, bias=False),
        #                            self.bn5,
        #                            nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1024 * 2, 512, bias=False)
        #         self.linear1 = nn.Linear(512 * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        tmp = x
        xyz = x[:, :3, :]  # b,3,n
        #
        # x ,idxxyz = get_graph_feature_withdilation_andxyz_idx_fature(x,xyz, d=[1],
        #                                    k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x,x11 = get_graph_feature_withdilation_catlocalshape_attentionpooling_firstlayer(x,xyz, d=[self.d[0]],k=self.k
                                                )
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0] + self.conv1_2(tmp) # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        # if self.use:
        #     x1 = self.attn1(xyz.permute(0,2,1), x1.permute(0, 2, 1))
        #     x1 = x1.permute(0, 2, 1)
        if self.use_down:
            downindex1 = sampling_by_RSF_distance(radius=None, xyz=xyz, points=x1, knn=20, downrate=2,
                                                  select_method='dilationk')
            xyz1 = index_points(xyz, downindex1)
            x1 = index_points(x1.permute(0, 2, 1), downindex1)
            x1 = x1.permute(0, 2, 1)

        # x,_ = get_graph_feature_withdilation_andxyz_idx_fature(x1,xyz, d=[1],
        #                                    k=self.k,idx_xyz= idxxyz)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x ,x_k1= get_graph_feature_withdilation_catlocalshape_attentionpooling(x1 ,xyz,d=[self.d[1]],k=self.k
                                                )
        x = self.conv2(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0] + self.conv2_2(x1) # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        # x2 = torch.cat((  x.max(dim=-1, keepdim=False)[0],x1),dim=1)  # b,128,n
        if self.use:
            x2 = self.attn2(xyz.permute(0,2,1), x2.permute(0, 2, 1))
            x2 = x2.permute(0, 2, 1)
        if self.use_down:
            downindex2 = sampling_by_RSF_distance(radius=None, xyz=xyz1, points=x2, knn=20, downrate=2,
                                                  select_method='dilationk')
            xyz2 = index_points(xyz1, downindex2)
            x2 = index_points(x2.permute(0, 2, 1), downindex2)
            x2 = x2.permute(0, 2, 1)

        # x,_ = get_graph_feature_withdilation_andxyz_idx_fature(x2, xyz,d=[1],
        #                                    k=self.k,idx_xyz= idxxyz)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x,x_k2 = get_graph_feature_withdilation_catlocalshape_attentionpooling(x2,xyz, d=[self.d[2]],k=self.k
                                                )
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0] + self.conv3_2(x2) # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)
        # x3 = torch.cat(( x.max(dim=-1, keepdim=False)[0] , x2),dim=1)
        if self.use:
            x3 = self.attn3(xyz.permute(0,2,1), x3.permute(0, 2, 1))
            x3 = x3.permute(0, 2, 1)
        if self.use_down:
            downindex3 = sampling_by_RSF_distance(radius=None, xyz=xyz2, points=x3, knn=20, downrate=2,
                                                  select_method='dilationk')
            xyz3 = index_points(xyz2, downindex3)
            x3 = index_points(x3.permute(0, 2, 1), downindex3)
            x3 = x3.permute(0, 2, 1)

        x ,x_k3= get_graph_feature_withdilation_catlocalshape_attentionpooling(x3,xyz,d=[1,2,4],k=self.k
                                                ) # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)  # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0] + self.conv4_2(x3) # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)
        # x4 = torch.cat(( x.max(dim=-1, keepdim=False)[0] , x3), dim = 1)
        if self.use:
            x4 = self.attn4(xyz.permute(0,2,1), x4.permute(0, 2, 1))
            x4 = x4.permute(0, 2, 1)
        if self.use_down:
            downindex4 = sampling_by_RSF_distance(radius=None, xyz=xyz3, points=x4, knn=20, downrate=2,
                                                  select_method='dilationk')
            xyz4 = index_points(xyz3, downindex4)
            x4 = index_points(x4.permute(0, 2, 1), downindex4)
            x4 = x4.permute(0, 2, 1)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)

        x = self.conv5(x)  # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        # 这儿参数1就是最后输出shape的形状
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size,
                                              -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size,
                                              -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)  # (batch_size, emb_dims*2)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)  # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)  # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x = self.linear3(x)  # (batch_size, 256) -> (batch_size, output_channels)

        return x, x4


class DGCNN_cls_voxelandpoints(nn.Module):
    def __init__(self, output_channels=40, normal_channel=False, type='L', use=False, use_down=False):
        super(DGCNN_cls_voxelandpoints, self).__init__()
        # self.args = args
        self.k = 20
        self.use = use
        self.d = [1, 2, 3, 4]
        if use:
            if type == 'L':
                # self.attn1 = Layerwise3dAttention3(npoints=1024, inchannel=3,outchannel=64, split_size=[32, 32, 32], local_agg='mean')
                self.attn2 = Layerwise3dAttention_car(npoints=1024, inchannel=64, split_size=[32, 32, 32], local_agg='mean')
                self.attn3 = Layerwise3dAttention_car(npoints=1024, inchannel=128, split_size=[16, 16, 16], local_agg='mean')
                self.attn4 = Layerwise3dAttention_car(npoints=1024, inchannel=256, split_size=[8, 8, 8], local_agg='mean')
            else:
                self.attn1 = Transformer3dAttention(inchannel=64, split_size=[64, 64, 64])
                self.attn2 = Transformer3dAttention(inchannel=64, split_size=[64, 64, 64])
                self.attn3 = Transformer3dAttention(inchannel=128, split_size=[64, 64, 64])
                self.attn4 = Transformer3dAttention(inchannel=256, split_size=[64, 64, 64])

        self.use_down = use_down
        self.inchannel = 12 if normal_channel == True else 6

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)


        self.conv1 = nn.Sequential(nn.Conv2d(self.inchannel, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2*2 , 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2 *2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(960, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024 * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)  #x : b,c,n
        xyz = x[:, :3, :]  # b,3,n

        # if self.use:
        #     x1_voxel = self.attn1(xyz.permute(0,2,1), x.permute(0, 2, 1))  # b,n,c
        #     x1_voxel = x1_voxel.permute(0, 2, 1)  # b,c,n

        x1_p = get_graph_feature_withdilation(x, d=self.d[0],
                                           k=self.k)
        x1_p = self.conv1(x1_p)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1_p = x1_p.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        # x1 = torch.cat((x1_voxel,x1_p),dim=1) # b , 128,n
        x1 = x1_p   # b,64,n

        if self.use:
            x2_voxel = self.attn2(xyz.permute(0,2,1), x1.permute(0, 2, 1))  # 64
            x2_voxel = x2_voxel.permute(0, 2, 1)

        x2_p = get_graph_feature_withdilation(x1, d=self.d[1],
                                           k=self.k)
        x2_p = self.conv2(x2_p)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2_p = x2_p.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x2 = torch.cat((x2_voxel, x2_p), dim=1)  # b , 128,n
        # x2 = x2_p   # b,64,n
        if self.use:
            x3_voxel = self.attn3(xyz.permute(0,2,1), x2.permute(0, 2, 1))
            x3_voxel = x3_voxel.permute(0, 2, 1)   # b, 128,n

        x3_p = get_graph_feature_withdilation(x2, d=self.d[2],
                                           k=self.k)
        x3_p = self.conv3(x3_p)  # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3_p = x3_p.max(dim=-1, keepdim=False)[0]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)
        x3 = torch.cat((x3_voxel, x3_p), dim=1)  # b , 256,n
        # x3 = x3_p
        if self.use:
            x4_voxel = self.attn4(xyz.permute(0, 2, 1), x3.permute(0, 2, 1))
            x4_voxel = x4_voxel.permute(0, 2, 1)

        x4_p = get_graph_feature_withdilation(x3, d=4,
                                           k=self.k)  # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x4_p = self.conv4(x4_p)  # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4_p = x4_p.max(dim=-1, keepdim=False)[0]  # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)
        x4 = torch.cat((x4_voxel, x4_p), dim=1)  # b , 512,n
        # x4 = x4_p + x4_voxel

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 128+128+256+512, num_points)
        x = self.conv5(x)  # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        # 这儿参数1就是最后输出shape的形状
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size,
                                              -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size,
                                              -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)  # (batch_size, emb_dims*2)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)  # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)  # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x = self.linear3(x)  # (batch_size, 256) -> (batch_size, output_channels)

        return x, x4


def get_nearsest_neighbors(target,source):
    """
    target:b,s1,c
    source:b,s2,c
    return b,s1,1
    """
    inner = torch.bmm(target,source.permute(0,2,1))  # b,s1,s2
    tar_2 = torch.sum(target ** 2,dim=-1) # b,s1
    sou_2 = torch.sum(source ** 2,dim=-1) # b,s2
    d = tar_2.unsqueeze(-1) + sou_2.unsqueeze(1) - 2 * inner
    index = torch.topk(d,k=1,dim=-1,largest=False)[1].squeeze(-1)
    return index #b,s1

def get_nearsest_3neighbors_anddis(target,source):
    """
    target:b,s1,c
    source:b,s2,c
    return b,s1,k
    """
    inner = torch.bmm(target,source.permute(0,2,1))  # b,s1,s2
    tar_2 = torch.sum(target ** 2,dim=-1) # b,s1
    sou_2 = torch.sum(source ** 2,dim=-1) # b,s2
    d = tar_2.unsqueeze(-1) + sou_2.unsqueeze(1) - 2 * inner
    dis,index = torch.topk(d,k=3,dim=-1,largest=False)

    return dis,index #b,s1

class DGCNN_cls_cylinder(nn.Module):
    def __init__(self, output_channels=40, normal_channel=False, type='L', use=True, use_down=False):
        super(DGCNN_cls_cylinder, self).__init__()
        # self.args = args
        self.k = 20
        self.use = use
        # if use:
        #     if type == 'L':
        #         # self.attn1 = Layerwise3dAttention(npoints=1024, inchannel=64, split_size=[32, 32, 32], local_agg='gk')
        #         self.attn2 = Layerwise3dAttention_car(npoints=1024, inchannel=64, split_size=[256,256,256], local_agg='mean')
        #         self.attn3 = Layerwise3dAttention_car(npoints=1024, inchannel=128, split_size=[256,256,256], local_agg='mean')
        #         self.attn4 = Layerwise3dAttention_car(npoints=1024, inchannel=256, split_size=[256,256,256], local_agg='mean')
        #     else:
        #         self.attn1 = Transformer3dAttention(inchannel=64, split_size=[64, 64, 64])
        #         self.attn2 = Transformer3dAttention(inchannel=64, split_size=[64, 64, 64])
        #         self.attn3 = Transformer3dAttention(inchannel=128, split_size=[64, 64, 64])
        #         self.attn4 = Transformer3dAttention(inchannel=256, split_size=[64, 64, 64])

        self.use_down = use_down
        self.inchannel = 6 if normal_channel == True else 3

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(13, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )
        self.conv1_2 = nn.Sequential(nn.Conv1d(self.inchannel, 64, kernel_size=1, bias=False),
                                     nn.BatchNorm1d(64),
                                     nn.LeakyReLU(negative_slope=0.2)
                                     )


        self.conv2_sem = nn.Sequential(nn.Linear(self.k,self.k),#nn.LeakyReLU(0.2),nn.Linear(self.k,self.k),
                                   nn.Softmax(-1))
        self.conv2 = nn.Sequential(nn.Conv2d(64  +12, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )
        self.conv2_2 = nn.Sequential(nn.Conv1d(64 , 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   )
        # self.conv2_fuse = nn.Sequential(nn.Conv1d(128, 128, kernel_size=1, bias=False),  # 128---64
        #                              nn.BatchNorm1d(128),                                # 128---64
        #                              nn.LeakyReLU(negative_slope=0.2),
        #                              )


        self.conv3_sem = nn.Sequential(nn.Linear(self.k,self.k),#nn.LeakyReLU(0.2),nn.Linear(self.k,self.k),
                                   nn.Softmax(-1))
        self.conv3 = nn.Sequential(nn.Conv2d(64  +12, 128, kernel_size=1, bias=False),        # 128 -------64
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )
        self.conv3_2 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=1, bias=False),      # 128 ----64
                                     nn.BatchNorm1d(128),
                                     nn.LeakyReLU(negative_slope=0.2),
                                     )

        # self.conv3_fuse = nn.Sequential(nn.Conv1d(128+128, 256, kernel_size=1, bias=False),    #256 ---128,64
        #                              nn.BatchNorm1d(256),
        #                              nn.LeakyReLU(negative_slope=0.2),
        #                              )

        self.conv4_sem = nn.Sequential(nn.Linear(self.k,self.k),#nn.LeakyReLU(0.2),nn.Linear(self.k,self.k),
                                   nn.Softmax(-1))
        self.conv4 = nn.Sequential(nn.Conv2d(128  +12, 256, kernel_size=1, bias=False),    # 256 ----128
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2)

                                   )
        self.conv4_2 = nn.Sequential(nn.Conv1d(128, 256, kernel_size=1, bias=False),          # 256 ,,128
                                     nn.BatchNorm1d(256),
                                     nn.LeakyReLU(negative_slope=0.2),
                                     )
        # self.conv4_fuse = nn.Sequential(nn.Conv1d(256 +256, 512, kernel_size=1, bias=False),
        #                              nn.BatchNorm1d(512),
        #                              nn.LeakyReLU(negative_slope=0.2),
        #                              )

        # self.conv5 = nn.Sequential(nn.Conv1d(512+256+128+64, 1024, kernel_size=1, bias=False),
        #                            self.bn5,
        #                            nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(64 + 256 + 128+64, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1024*2 , 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        xyz = x[:, :3, :]  # b,3,n
        xyz_bn3 = xyz.permute(0,2,1) # b,n,3
        tmp = x

        x1_localcylinder,x1_knn,dis1 = get_graph_feature_withdilation_catlocalshape_attentionpooling_firstlayer(x,xyz,d=[1], k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        # x1_localcylinder = self.conv1_k(x1_localcylinder.permute(0,3,1,2)).permute(0,2,3,1) # b,k,c,g----b,c,g,k
        x = self.conv1(x1_localcylinder)  # (batch_size, 10, num_points, k) -> (batch_size, 64, num_points, k)
        # x = localrelativeagg(x,x1_knn.permute(0,3,1,2) - xyz.unsqueeze(-1),mode='xyz')
        # x = self.conv1_3(x)
        x1 = self.conv1_2(tmp) - x.max(dim=-1, keepdim=False)[0]   # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        # x1_up = x1 #b,64,1024
        # if self.use:
        #     x1 = self.attn1(xyz, x1.permute(0, 2, 1))
        #     x1 = x1.permute(0, 2, 1)
        if self.use_down:
            # downindex1 = sampling_by_RSF_distance(radius=None, xyz=xyz, points=x1, knn=20, downrate=2,
            #                                       select_method='dilationk')
            downindex1 = farthest_point_sample(xyz_bn3,512)
            xyz1 = index_points(xyz_bn3, downindex1)  #b,n,3
            x1 = index_points(x1.permute(0, 2, 1), downindex1)
            x1 = x1.permute(0, 2, 1)  # b,c,n


        if self.use:
            x2_voxel = self.attn2(xyz_bn3, x1.permute(0, 2, 1))  # 64
            x2_voxel = x2_voxel.permute(0, 2, 1)
        # x1_local,x1_k ,dis2= get_graph_feature_withdilation_catlocalshape_attentionpooling(x1,xyz,d = [2] ,k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x1_local= get_graph_feature_withdilation_and_localshape(self.conv2_sem,True,x1,xyz,d = [2] ,k=self.k)  # ,x1_centerspecific
        # x1_local = x1_local * (dis2 / (dis2.sum(dim=-1, keepdim=True) + 1e-8)).unsqueeze(1)\]
        # x1_local = self.conv2_k(x1_local.permute(0,3,1,2)).permute(0,2,3,1)
        att1 = self.conv2(x1_local)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        # center1 = self.conv2_2(x1_centerspe)
        # x1_geo = self.conv2_geo(local_shape1)
        # att1 = localrelativeagg(att1,x1_k.permute(0,3,1,2)-xyz.unsqueeze(-1),mode='xyz')
        # att1 = self.conv2_3(att1)
        # x2_local_agg = (att1 + center1 + x1_geo).max(dim=-1,keepdim=False)[0]
        x2_local_agg = self.conv2_2(x1) - att1.max(dim=-1,keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)self.conv2_2(x1) +
        x2 = x2_local_agg
        # x2 = torch.cat((x2_local_agg,x2_voxel),dim=1)  #b,128,n
        # x2 = self.conv2_fuse(x2)

        # index_512_2_1024 = get_nearsest_neighbors(xyz_bn3,xyz1) #b,1024
        # x2_up = index_points(x2.permute(0,2,1),index_512_2_1024).permute(0,2,1)

        if self.use_down:
            # downindex1 = sampling_by_RSF_distance(radius=None, xyz=xyz, points=x1, knn=20, downrate=2,
            #                                       select_method='dilationk')
            downindex2 = farthest_point_sample(xyz1, 256)
            xyz2 = index_points(xyz1, downindex2)  # b,n,3
            x2 = index_points(x2.permute(0, 2, 1), downindex2)
            x2 = x2.permute(0, 2, 1)  # b,c,n



        if self.use:
            x3_voxel = self.attn3(xyz_bn3, x2.permute(0, 2, 1))  # 128
            x3_voxel = x3_voxel.permute(0, 2, 1)
        x2_local= get_graph_feature_withdilation_and_localshape(self.conv3_sem,True,x2,xyz,d=[3], k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        # x2_local = x2_local * (dis3 / (dis3.sum(dim=-1, keepdim=True) + 1e-8)).unsqueeze(1)   , x2_centerspecific
        # x2_local = self.conv3_k(x2_local.permute(0,3,1,2)).permute(0,2,3,1)
        att2 = self.conv3(x2_local)  # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        # center2 = self.conv3_2(x2_centerspe)
        # x2_geo = self.conv3_geo(local_shape2)
        # att2 = localrelativeagg(att2,x2_k.permute(0,3,1,2)-xyz.unsqueeze(-1),mode='xyz')
        # att2 = self.conv3_3(att2)
        # x3_local_agg = (att2 + center2 + x2_geo).max(dim=-1,keepdim=False)[0]
        x3_local_agg = self.conv3_2(x2) - att2.max(dim=-1,keepdim=False)[0]   # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)self.conv3_2(x2) +
        x3 = x3_local_agg
        # x3 = torch.cat((x3_local_agg,x3_voxel),dim=1)  # b, 256,n
        # x3 = self.conv3_fuse(x3)


        if self.use_down:
            # downindex1 = sampling_by_RSF_distance(radius=None, xyz=xyz, points=x1, knn=20, downrate=2,
            #                                       select_method='dilationk')
            downindex3 = farthest_point_sample(xyz2, 128)
            xyz3 = index_points(xyz2, downindex3)  # b,n,3
            x3 = index_points(x3.permute(0, 2, 1), downindex3)
            x3 = x3.permute(0, 2, 1)  # b,c,n

        if self.use:
            x4_voxel = self.attn4(xyz_bn3, x3.permute(0, 2, 1))  # 256
            x4_voxel = x4_voxel.permute(0, 2, 1)
        x3_local= get_graph_feature_withdilation_and_localshape(self.conv4_sem,True,x3, xyz, d=[4],k=self.k)  # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        att3 = self.conv4(x3_local)  # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4_local_agg = self.conv4_2(x3) + att3.max(dim=-1,keepdim=False)[0]    # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points) self.conv4_2(x3) +
        x4 = x4_local_agg
        # x4 = torch.cat((x4_local_agg,x4_voxel),dim=1)   # 512
        # x4 = self.conv4_fuse(x4)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # b,960,n

        x = self.conv5(x)  # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        # 这儿参数1就是最后输出shape的形状
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size,
                                              -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size,
                                              -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)  # (batch_size, emb_dims*2)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)  # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)  # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x = self.linear3(x)  # (batch_size, 256) -> (batch_size, output_channels)

        return x, x4


class DGCNN_cls_cylinder_splitneigh(nn.Module):
    def __init__(self, output_channels=40, normal_channel=False, type='L', use=True, use_down=False):
        super(DGCNN_cls_cylinder_splitneigh, self).__init__()
        # self.args = args
        self.k = 20
        self.use = use
        if use:
            if type == 'L':
                # self.attn1 = Layerwise3dAttention(npoints=1024, inchannel=64, split_size=[32, 32, 32], local_agg='gk')
                self.attn2 = Layerwise3dAttention_car(npoints=1024, inchannel=64, split_size=[32, 32, 32], local_agg='mean')
                self.attn3 = Layerwise3dAttention_car(npoints=1024, inchannel=128, split_size=[16, 16, 16], local_agg='mean')
                self.attn4 = Layerwise3dAttention_car(npoints=1024, inchannel=256, split_size=[8, 8, 8], local_agg='mean')
            else:
                self.attn1 = Transformer3dAttention(inchannel=64, split_size=[64, 64, 64])
                self.attn2 = Transformer3dAttention(inchannel=64, split_size=[64, 64, 64])
                self.attn3 = Transformer3dAttention(inchannel=128, split_size=[64, 64, 64])
                self.attn4 = Transformer3dAttention(inchannel=256, split_size=[64, 64, 64])

        self.use_down = use_down
        self.inchannel = 12 if normal_channel == True else 6

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(13, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )
        self.conv1_2 = nn.Sequential(nn.Conv1d(3, 64, kernel_size=1, bias=False),
                                     nn.BatchNorm1d(64),
                                     nn.LeakyReLU(negative_slope=0.2)
                                     )
        self.conv1_3 = nn.Sequential(nn.Conv2d(64 *3, 64, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(64),
                                     nn.LeakyReLU(negative_slope=0.2)
                                     )
        self.conv2_in = nn.Sequential(nn.Conv2d(64 +13, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )
        self.conv2_2_in = nn.Sequential(nn.Conv1d(64 , 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   )
        self.conv2_bo = nn.Sequential(nn.Conv2d(64 + 13, 64, kernel_size=1, bias=False),
                                      self.bn2,
                                      nn.LeakyReLU(negative_slope=0.2)
                                      )
        self.conv2_2_bo = nn.Sequential(nn.Conv1d(64, 64, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(64),
                                        nn.LeakyReLU(negative_slope=0.2),
                                        )

        self.conv3_in = nn.Sequential(nn.Conv2d(128 +13, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )
        self.conv3_2_in = nn.Sequential(nn.Conv1d(128, 128, kernel_size=1, bias=False),
                                     nn.BatchNorm1d(128),
                                     nn.LeakyReLU(negative_slope=0.2),
                                     )
        self.conv3_bo = nn.Sequential(nn.Conv2d(128 + 13, 128, kernel_size=1, bias=False),
                              self.bn3,
                              nn.LeakyReLU(negative_slope=0.2)
                              )
        self.conv3_2_bo = nn.Sequential(nn.Conv1d(128, 128, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(128),
                                        nn.LeakyReLU(negative_slope=0.2),
                                        )

        self.conv4_in = nn.Sequential(nn.Conv2d(256 +13, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2)

                                   )
        self.conv4_2_in = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                     nn.BatchNorm1d(256),
                                     nn.LeakyReLU(negative_slope=0.2),
                                     )
        self.conv4_bo = nn.Sequential(nn.Conv2d(256 + 13, 256, kernel_size=1, bias=False),
                                      self.bn4,
                                      nn.LeakyReLU(negative_slope=0.2)

                                      )
        self.conv4_2_bo = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(256),
                                        nn.LeakyReLU(negative_slope=0.2),
                                        )


        self.conv5 = nn.Sequential(nn.Conv1d(512+256+128+64, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1024 * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        xyz = x[:, :3, :]  # b,3,n
        tmp = x

        x1_localcylinder,x1_knn,dis = get_graph_feature_withdilation_catlocalshape_attentionpooling_firstlayer(x,xyz,d=[1], k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x1_localcylinder)  # (batch_size, 10, num_points, k) -> (batch_size, 64, num_points, k)
        x = localrelativeagg(x,x1_knn.permute(0,3,1,2) - xyz.unsqueeze(-1),mode='xyz')
        x = self.conv1_3(x)
        x1 = x.max(dim=-1, keepdim=False)[0] + self.conv1_2(tmp) # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        # if self.use:
        #     x1 = self.attn1(xyz, x1.permute(0, 2, 1))
        #     x1 = x1.permute(0, 2, 1)
        if self.use_down:
            downindex1 = sampling_by_RSF_distance(radius=None, xyz=xyz, points=x1, knn=20, downrate=2,
                                                  select_method='dilationk')
            xyz1 = index_points(xyz, downindex1)
            x1 = index_points(x1.permute(0, 2, 1), downindex1)
            x1 = x1.permute(0, 2, 1)


        if self.use:
            x2_voxel = self.attn2(xyz.permute(0,2,1), x1.permute(0, 2, 1))  # 64
            x2_voxel = x2_voxel.permute(0, 2, 1)
        x1_in_f_k,x1_b_f_k,x1_in_c,x1_b_c,localshape1,x1_in_index,x1_bo_index = get_graph_feature_inner_boundary_neighbour(x1,xyz,d = [2] ,k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x1_inner = get_inner_feature(x1_in_f_k,x1_in_c,x1_in_index,localshape1,conv1=self.conv2_in,conv2 = self.conv2_2_in) # b,c,n
        x1_bound = get_bound_feature(x1_b_f_k,x1_b_c,x1_bo_index,localshape1,conv1=self.conv2_bo,conv2=self.conv2_2_bo) # b,c,n
        x2_local_agg = x1_inner + x1_bound # b,c,n

        x2 = torch.cat((x2_local_agg,x2_voxel),dim=1)  #b,128,n




        if self.use:
            x3_voxel = self.attn3(xyz.permute(0,2,1), x2.permute(0, 2, 1))  # 128
            x3_voxel = x3_voxel.permute(0, 2, 1)
        x2_in_f_k, x2_b_f_k, x2_in_c, x2_b_c, localshape2, x2_in_index, x2_bo_index = get_graph_feature_inner_boundary_neighbour(
            x2, xyz, d=[3], k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x2_inner = get_inner_feature(x2_in_f_k, x2_in_c, x2_in_index, localshape2, conv1=self.conv3_in,
                                     conv2=self.conv3_2_in)  # b,c,n
        x2_bound = get_bound_feature(x2_b_f_k, x2_b_c, x2_bo_index, localshape2, conv1=self.conv3_bo,
                                     conv2=self.conv3_2_bo)  # b,c,n
        x3_local_agg = x2_inner + x2_bound  # b,c,n
        x3 = torch.cat((x3_local_agg,x3_voxel),dim=1)  # b, 256,n

        if self.use:
            x4_voxel = self.attn4(xyz.permute(0,2,1), x3.permute(0, 2, 1))  # 256
            x4_voxel = x4_voxel.permute(0, 2, 1)
        x3_in_f_k, x3_b_f_k, x3_in_c, x3_b_c, localshape3, x3_in_index, x3_bo_index = get_graph_feature_inner_boundary_neighbour(
            x3, xyz, d=[4], k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x3_inner = get_inner_feature(x3_in_f_k, x3_in_c, x3_in_index, localshape3, conv1=self.conv4_in,
                                     conv2=self.conv4_2_in)  # b,c,n
        x3_bound = get_bound_feature(x3_b_f_k, x3_b_c, x3_bo_index, localshape3, conv1=self.conv4_bo,
                                     conv2=self.conv4_2_bo)  # b,c,n
        x4_local_agg = x3_inner + x3_bound  # b,c,n
        x4 = torch.cat((x4_local_agg,x4_voxel),dim=1)   # 512

        # if self.use_down:
        #     downindex4 = sampling_by_RSF_distance(radius=None, xyz=xyz3, points=x4, knn=20, downrate=2,
        #                                           select_method='dilationk')
        #     xyz4 = index_points(xyz3, downindex4)
        #     x4 = index_points(x4.permute(0, 2, 1), downindex4)
        #     x4 = x4.permute(0, 2, 1)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # b,960,n

        x = self.conv5(x)  # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        # 这儿参数1就是最后输出shape的形状
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size,
                                              -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size,
                                              -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)  # (batch_size, emb_dims*2)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)  # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)  # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x = self.linear3(x)  # (batch_size, 256) -> (batch_size, output_channels)

        return x, x4

class Transform_Net(nn.Module):
    def __init__(self, inchannel):
        super(Transform_Net, self).__init__()
        # self.args = args
        self.k = 3

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(inchannel, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3 * 3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)  # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)  # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)  # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)  # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)  # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x


class DGCNN_partseg(nn.Module):
    def __init__(self, num_part, normal_channel=False, type='L', use=True, down=False):
        super(DGCNN_partseg, self).__init__()
        # self.args = args
        self.seg_num_all = 50
        self.k = 20
        self.dilation = [1, 2, 3, 4]
        self.inchannel = 12 if normal_channel == True else 6
        self.transform_net = Transform_Net(self.inchannel)
        self.down = down
        self.use = use
        if use:
            if type == 'L':
                # self.attn1 = Layerwise3dAttention3(npoints = 2048,inchannel=64, split_size=[64, 64, 64], local_agg = 'mean')
                self.attn2 = Layerwise3dAttention3(npoints=2048,inchannel=64,outchannel=64, split_size=[32, 32, 32], local_agg = 'mean')
                self.attn3 = Layerwise3dAttention3(npoints = 2048,inchannel=64 ,outchannel=64, split_size=[16, 16, 16], local_agg = 'mean')

            else:
                self.attn1 = Transformer3dAttention(inchannel=64, split_size=[64, 64, 64])
                self.attn2 = Transformer3dAttention(inchannel=64, split_size=[64, 64, 64])
                self.attn3 = Transformer3dAttention(inchannel=64, split_size=[64, 64, 64])

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(1024)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        # self.conv2_op = nn.Sequential(nn.Conv2d(3, 64, 1, bias=False), nn.BatchNorm2d(64),
        #                               nn.LeakyReLU(negative_slope=0.2))
        # self.conv3_op = nn.Sequential(nn.Conv2d(3, 64, 1, bias=False), nn.BatchNorm2d(64),
        #                               nn.LeakyReLU(negative_slope=0.2))

        self.conv1 = nn.Sequential(nn.Conv2d(9, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 3, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64 * 3, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, 1024, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(1280, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=0.5)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=0.5)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                    self.bn10,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, 50, kernel_size=1, bias=False)

    def forward(self, x, l):
        batch_size = x.size(0)
        num_points = x.size(2)
        xyz = x.permute(0, 2, 1)[:, :, :3]  # b,n,3
        xyz_b3n = xyz.permute(0,2,1)
        if self.inchannel == 6:
            x0 = get_graph_feature_withdilation(x, k=self.k, d=self.dilation[0])
                                                  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
            t = self.transform_net(x0)  # (batch_size, 3, 3)
            x = x.transpose(2, 1)  # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
            x = torch.bmm(x, t)  # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
            x = x.transpose(2, 1)  # (batch_size, num_points, 3) -> (batch_size, 3, num_points)
        else:
            x0 = get_graph_feature_withdilation(x, k=self.k, d=self.dilation[0])
            t = self.transform_net(x0)
            p1 = torch.bmm(x[:, 0:3, :].transpose(2, 1), t)
            p2 = torch.bmm(x[:, 3:6, :].transpose(2, 1), t)
            x = torch.cat((p1, p2), dim=2).transpose(2, 1).contiguous()

        x = get_graph_feature_withdilation_and_global_faeture(x, k=self.k,d=[1]
                                                )   # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        # if self.use:
        #     x1 = self.attn1(xyz, x1.permute(0, 2, 1))
        #     x1 = x1.permute(0, 2, 1)

        x = get_graph_feature_withdilation_and_global_faeture(x1,k=self.k, d=[1,2]
                                                )  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        if self.use:
            x2 = self.attn2(xyz, x2.permute(0, 2, 1))
            x2 = x2.permute(0, 2, 1)

        x = get_graph_feature_withdilation_and_global_faeture(x2, k=self.k,d=[1,2,4]
                                                ) # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        if self.use:
            x3 = self.attn3(xyz, x3.permute(0, 2, 1))
            x3 = x3.permute(0, 2, 1)

        x = torch.cat((x1, x2, x3), dim=1)  # (batch_size, 64*3, num_points)

        x = self.conv6(x)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        l = l.view(batch_size, -1, 1)  # (batch_size, num_categoties, 1)
        l = self.conv7(l)  # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)

        x = torch.cat((x, l), dim=1)  # (batch_size, 1088, 1)
        x = x.repeat(1, 1, num_points)  # (batch_size, 1088, num_points)

        x = torch.cat((x, x1, x2, x3), dim=1)  # (batch_size, 1088+64*3, num_points)

        x = self.conv8(x)  # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)  # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        x = self.dp2(x)
        x = self.conv10(x)  # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        x = self.conv11(x)  # (batch_size, 256, num_points) -> (batch_size, seg_num_all, num_points)

        return x
class DGCNN_partseg_voxelandpoints(nn.Module):
    def __init__(self, num_part, normal_channel=False, type='L', use=True, down=True):
        super(DGCNN_partseg_voxelandpoints, self).__init__()
        # self.args = args
        self.npoints = 2048
        self.seg_num_all = 50
        self.k = 20
        self.d = [1, 2, 3, 4]
        self.inchannel = 12 if normal_channel == True else 6
        # self.transform_net = Transform_Net(self.inchannel)
        self.down = down
        self.use = use
        if use:
            if type == 'L':
                # self.attn1 = Layerwise3dAttention3(npoints = 2048,inchannel=64, split_size=[64, 64, 64], local_agg = 'mean')
                self.attn2 = Layerwise3dAttention_car(npoints=2048,inchannel=64, split_size=[16, 16, 16], local_agg = 'mean')
                self.attn3 = Layerwise3dAttention_car(npoints = 2048,inchannel=128 , split_size=[8,8,8], local_agg = 'mean')
                self.attn4 = Layerwise3dAttention_car(npoints=2048, inchannel=256, split_size=[2, 2, 2],local_agg='mean')

            else:
                self.attn1 = Transformer3dAttention(inchannel=64, split_size=[64, 64, 64])
                self.attn2 = Transformer3dAttention(inchannel=64, split_size=[64, 64, 64])
                self.attn3 = Transformer3dAttention(inchannel=64, split_size=[64, 64, 64])

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(1024)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(512)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)


        self.conv1 = nn.Sequential(nn.Conv2d(13, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        #adddd-----------------
        self.conv1_2 = nn.Sequential(nn.Conv1d(3, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 + 12, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
# adddd---------------dddd--------
        self.conv4_2 = nn.Sequential(nn.Conv1d(64, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv5 = nn.Sequential(nn.Conv2d(64 + 12 , 64, kernel_size=1, bias=False),   #####  128 ---64
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5_sec = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),  #####  128 ---64
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
# addddddd-----------
        self.conv5_2 = nn.Sequential(nn.Conv1d(64 , 64, kernel_size=1, bias=False),   #####  128---64
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
#----------
        self.conv5_3 = nn.Sequential(nn.Conv2d(64 + 12, 64, kernel_size=1, bias=False),   ##256--128
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5_3_sec = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),  ##256--128
                                     nn.BatchNorm2d(64),
                                     nn.LeakyReLU(negative_slope=0.2))
        self.conv5_4 = nn.Sequential(nn.Conv1d(64, 64, kernel_size=1, bias=False),  # 256---128
                                     nn.BatchNorm1d(64),
                                     nn.LeakyReLU(negative_slope=0.2))


        self.conv6 = nn.Sequential(nn.Conv1d(64, 1024, kernel_size=1, bias=False),   ### 512   --256
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(1024+64+64+64+64+64, 512, kernel_size=1, bias=False),  #1024+64+64+128+256+512  -----  1024+64+64+128+256+64
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=0.5)
        self.conv9 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=0.5)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                    self.bn10,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, 50, kernel_size=1, bias=False)

    def forward(self, x, l):
        tmp = x
        batch_size = x.size(0)
        num_points = x.size(2)
        # if self.inchannel == 6:
        #     x0 = get_graph_feature_withdilation(x,d=1, k=self.k)
        #                                           # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        #     t = self.transform_net(x0)  # (batch_size, 3, 3)
        #     x = x.transpose(2, 1)  # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        #     x = torch.bmm(x, t)  # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        #     x_t = x.transpose(2, 1)  # b,3,2048
        # else:
        #     x0 = get_graph_feature_withdilation(x, d=1,k=self.k)
        #     t = self.transform_net(x0)
        #     p1 = torch.bmm(x[:, 0:3, :].transpose(2, 1), t)
        #     p2 = torch.bmm(x[:, 3:6, :].transpose(2, 1), t)
        #     x_t = torch.cat((p1, p2), dim=2).transpose(2, 1).contiguous()  # b, 6, 2048

        xyz_bn3 = x.permute(0, 2, 1)[:, :, :3]  # b,2048,3
        xyz = xyz_bn3.permute(0, 2, 1)  # b,3,2048


        x,x11,x12 = get_graph_feature_withdilation_catlocalshape_attentionpooling_firstlayer(x,xyz, d=[1],k=self.k )   # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0] + self.conv1_2(tmp)# b,64,2048
        if self.down:
            downindex1 = farthest_point_sample(xyz_bn3,self.npoints // 4) # b, 512
            xyz1_bn3 = index_points(xyz_bn3,downindex1)  # n,512,3
            xyz1 = xyz1_bn3.permute(0,2,1)
            x2 = index_points(x1.permute(0,2,1),downindex1)  # b,512,c
            x2 = x2.permute(0,2,1) # b, 64,512

        x2_p= get_graph_feature_withdilation_and_localshape_src_dst(True,downindex1,x2,x1,xyz1,xyz, d=[2] ,k=self.k )
        x2_p = self.conv3(x2_p)
        x2_p = self.conv4(x2_p)
        x2_p = x2_p.max(dim=-1, keepdim=False)[0] + self.conv4_2(x2) # b,64,512
        if self.use:
            x2_voxel = self.attn2(xyz1_bn3, x2.permute(0, 2, 1))
            x2_voxel = x2_voxel.permute(0, 2, 1)
        x2 = x2_p # b,64,512
        # x2 = torch.cat((x2_voxel,x2_p),dim=1)  # b,128,512


        if self.down:
            downindex2 = farthest_point_sample(xyz1_bn3,self.npoints // 4 //4) # b, 128
            xyz2_bn3 = index_points(xyz1_bn3,downindex2)  # n,128,3
            xyz2 = xyz2_bn3.permute(0,2,1)
            x3 = index_points(x2.permute(0,2,1),downindex2)  # b,128,c
            x3 = x3.permute(0,2,1) # b, 128,128  #---------------b,64,128 nouse
        x3_p = get_graph_feature_withdilation_and_localshape_src_dst(True,downindex2,x3,x2,xyz2,xyz1,d=[3],k=self.k)
        x3_p = self.conv5(x3_p)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3_p = self.conv5_sec(x3_p)
        x3_p = x3_p.max(dim=-1, keepdim=False)[0] + self.conv5_2(x3) # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        if self.use:
            x3_voxel = self.attn3(xyz2_bn3, x3.permute(0, 2, 1))
            x3_voxel = x3_voxel.permute(0, 2, 1)
        x3 = x3_p   # b,128,128
        # x3 = torch.cat((x3_voxel,x3_p),dim=1)  # b,256,128

        if self.down:
            downindex3 = farthest_point_sample(xyz2_bn3,self.npoints // 4 //4  //4) # b, 32
            xyz3_bn3 = index_points(xyz2_bn3,downindex3)  # n,32,3
            xyz3 = xyz3_bn3.permute(0,2,1)
            x4 = index_points(x3.permute(0,2,1),downindex3)  # b,32,c
            x4 = x4.permute(0,2,1) # b, 256,32    #### b,128,32
        x4_p = get_graph_feature_withdilation_and_localshape_src_dst(True,downindex3,x4,x3,xyz3,xyz2,d=[4],k=self.k)
        x4_p = self.conv5_3(x4_p)
        x4_p = self.conv5_3_sec(x4_p)
        x4_p = x4_p.max(dim=-1, keepdim=False)[0] + self.conv5_4(x4)
        if self.use:
            x4_voxel = self.attn4(xyz3_bn3, x4.permute(0, 2, 1))
            x4_voxel = x4_voxel.permute(0, 2, 1)
        x4 = x4_p
        # x4 = torch.cat((x4_voxel,x4_p),dim=1)  # b,512,32


        # x = torch.cat((x1, x2, x3), dim=1)  # (batch_size, 64*3, num_points)

        x = self.conv6(x4)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.mean(dim=-1, keepdim=True)  # (b,1024,128) ->  b,1024,1

        l = l.view(batch_size, -1, 1)  # (batch_size, num_categoties, 1)
        l = self.conv7(l)  # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)

        x = torch.cat((x, l), dim=1)  # (batch_size, 1088, 1)
        x = x.repeat(1, 1, num_points)  # (b, 1088, n)

        index32_2_2048 = get_nearsest_neighbors(xyz_bn3,xyz3_bn3)
        x4_up = index_points(x4.permute(0, 2, 1), index32_2_2048).permute(0, 2, 1)  # b,512,2048   #---b,256,2048

        index128_2_2048 = get_nearsest_neighbors(xyz_bn3, xyz2_bn3)
        x3_up = index_points(x3.permute(0, 2, 1), index128_2_2048).permute(0, 2, 1)  # b,256,2048   # b,128,2048

        index512_2_2048 = get_nearsest_neighbors(xyz_bn3, xyz1_bn3)
        x2_up = index_points(x2.permute(0, 2, 1), index512_2_2048).permute(0, 2, 1)  # b,128,2048    ---64

        x1_up = x1
        # index64_2_4096 = get_nearsest_neighbors(xyz_bn3, xyz3_bn3)
        # x3_up = index_points(x3.permute(0, 2, 1), index64_2_4096).permute(0, 2, 1)  # b,256,4096     ---128

        x = torch.cat((x, x1_up, x2_up, x3_up,x4_up), dim=1)  # (batch_size, 1088+64+128+256, num_points)

        x = self.conv8(x)  # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)  # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        x = self.dp2(x)
        x = self.conv10(x)  # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        x = self.conv11(x)  # (batch_size, 256, num_points) -> (batch_size, seg_num_all, num_points)

        return x

class DGCNN_partseg_voxelandpoints_decoder(nn.Module):
    def __init__(self, num_part, normal_channel=False, type='L', use=True, down=True):
        super(DGCNN_partseg_voxelandpoints_decoder, self).__init__()
        # self.args = args
        self.npoints = 2048
        self.seg_num_all = 50
        self.k = 20
        self.d = [1, 2, 3, 4]
        self.inchannel = 12 if normal_channel == True else 6
        # self.transform_net = Transform_Net(self.inchannel)
        self.down = down
        self.use = use
        if use:
            if type == 'L':
                # self.attn1 = Layerwise3dAttention3(npoints = 2048,inchannel=64, split_size=[64, 64, 64], local_agg = 'mean')
                self.attn2 = Layerwise3dAttention_car(npoints=2048,inchannel=64, split_size=[128,128,128], local_agg = 'mean')
                self.attn3 = Layerwise3dAttention_car(npoints = 2048,inchannel=128 , split_size=[32,32,32], local_agg = 'mean')
                self.attn4 = Layerwise3dAttention_car(npoints=2048, inchannel=256, split_size=[8,8,8],local_agg='mean')

            else:
                self.attn1 = Transformer3dAttention(inchannel=64, split_size=[64, 64, 64])
                self.attn2 = Transformer3dAttention(inchannel=64, split_size=[64, 64, 64])
                self.attn3 = Transformer3dAttention(inchannel=64, split_size=[64, 64, 64])

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm1d(1024)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(512)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv2d(13, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        #adddd-----------------
        self.conv1_2 = nn.Sequential(nn.Conv1d(3, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 + 12, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2_2modality = nn.Sequential(nn.Conv1d(128, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2_2localatten = nn.Sequential(nn.Linear(self.k,self.k),  # nn.LeakyReLU(0.2),nn.Linear(self.k,self.k),
                                   nn.Softmax(-1))


# adddd---------------dddd--------
        self.conv4_2 = nn.Sequential(nn.Conv1d(64, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv5 = nn.Sequential(nn.Conv2d(128 + 12 , 128, kernel_size=1, bias=False),   #####  128 ---64
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5_sec = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3_2localatten = nn.Sequential(nn.Linear(self.k,self.k), #nn.LeakyReLU(0.2),nn.Linear(self.k,self.k),
                                   nn.Softmax(-1))
        self.conv3_2modality = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                             nn.BatchNorm1d(256),
                                             nn.LeakyReLU(negative_slope=0.2))

# addddddd-----------
        self.conv5_2 = nn.Sequential(nn.Conv1d(128 , 128, kernel_size=1, bias=False),   #####  128---64
                                   nn.BatchNorm1d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
#----------
        self.conv5_3 = nn.Sequential(nn.Conv2d(256 + 12, 256, kernel_size=1, bias=False),   ##256--128
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5_3_sec = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.LeakyReLU(negative_slope=0.2))
        self.conv4_2localatten = nn.Sequential(nn.Linear(self.k,self.k), #nn.LeakyReLU(0.2),nn.Linear(self.k,self.k),
                                   nn.Softmax(-1))
        self.conv4_2modality = nn.Sequential(nn.Conv1d(512, 512, kernel_size=1, bias=False),
                                             nn.BatchNorm1d(512),
                                             nn.LeakyReLU(negative_slope=0.2))


        self.conv5_4 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),  # 256---128
                                     nn.BatchNorm1d(256),
                                     nn.LeakyReLU(negative_slope=0.2))
#-----------------------     #   512- ---- 256
        self.de4 = nn.Sequential(nn.Conv1d(1088+512,256,1,bias=False),nn.BatchNorm1d(256),nn.LeakyReLU(negative_slope=0.2),nn.Conv1d(256,256,1,bias=False),nn.BatchNorm1d(256),nn.LeakyReLU(negative_slope=0.2))
        if self.training:
            self.deout4 = nn.Conv1d(256,50,kernel_size=1,bias=False)
#                                     -----128 ----256
        self.de3 = nn.Sequential(nn.Conv1d(256+256,256,1,bias=False),nn.BatchNorm1d(256),nn.LeakyReLU(negative_slope=0.2),nn.Conv1d(256,256,1,bias=False),nn.BatchNorm1d(256),nn.LeakyReLU(negative_slope=0.2))
        if self.training:
            self.deout3 = nn.Conv1d(256,50,kernel_size=1,bias=False)
                                     # ----64 ---128
        self.de2 = nn.Sequential(nn.Conv1d(128+ 256, 256, 1, bias=False), nn.BatchNorm1d(256),
                                 nn.LeakyReLU(negative_slope=0.2), nn.Conv1d(256, 128, 1, bias=False),
                                 nn.BatchNorm1d(128), nn.LeakyReLU(negative_slope=0.2))
        if self.training:
            self.deout2 = nn.Conv1d(128, 50, kernel_size=1, bias=False)

        self.de1 = nn.Sequential(nn.Conv1d(64 + 128, 128, 1, bias=False), nn.BatchNorm1d(128),
                                 nn.LeakyReLU(negative_slope=0.2), nn.Conv1d(128, 128, 1, bias=False),
                                 nn.BatchNorm1d(128), nn.LeakyReLU(negative_slope=0.2))

        self.deout1 = nn.Conv1d(128, 50, kernel_size=1, bias=False)

        self.conv6 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),   ### 512   --256
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x, l,label=None):
        tmp = x
        batch_size = x.size(0)
        num_points = x.size(2)
        # if self.inchannel == 6:
        #     x0 = get_graph_feature_withdilation(x,d=1, k=self.k)
        #                                           # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        #     t = self.transform_net(x0)  # (batch_size, 3, 3)
        #     x = x.transpose(2, 1)  # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        #     x = torch.bmm(x, t)  # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        #     x_t = x.transpose(2, 1)  # b,3,2048
        # else:
        #     x0 = get_graph_feature_withdilation(x, d=1,k=self.k)
        #     t = self.transform_net(x0)
        #     p1 = torch.bmm(x[:, 0:3, :].transpose(2, 1), t)
        #     p2 = torch.bmm(x[:, 3:6, :].transpose(2, 1), t)
        #     x_t = torch.cat((p1, p2), dim=2).transpose(2, 1).contiguous()  # b, 6, 2048

        xyz_bn3 = x.permute(0, 2, 1)[:, :, :3]  # b,2048,3
        xyz = xyz_bn3.permute(0, 2, 1)  # b,3,2048


        x,x11,x12 = get_graph_feature_withdilation_catlocalshape_attentionpooling_firstlayer(x,xyz, d=[1],k=self.k )   # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0] + self.conv1_2(tmp)# b,64,2048
        if self.down:
            downindex1 = farthest_point_sample(xyz_bn3,self.npoints // 4) # b, 512
            xyz1_bn3 = index_points(xyz_bn3,downindex1)  # n,512,3
            xyz1 = xyz1_bn3.permute(0,2,1)
            x2 = index_points(x1.permute(0,2,1),downindex1)  # b,512,c
            x2 = x2.permute(0,2,1) # b, 64,512


        x2_p= get_graph_feature_withdilation_and_localshape_src_dst(self.conv2_2localatten,True,downindex1,x2,x1,xyz1,xyz, d=[2] ,k=self.k )
        x2_p = self.conv3(x2_p)
        x2_p = self.conv4(x2_p)
        x2_p = x2_p.max(dim=-1, keepdim=False)[0] + self.conv4_2(x2) # b,64,512
        if self.use:
            x2_voxel = self.attn2(xyz1_bn3, x2.permute(0, 2, 1))
            x2_voxel = x2_voxel.permute(0, 2, 1)
        # x2 = x2_p # b,64,512
        x2 = torch.cat((x2_voxel,x2_p),dim=1)  # b,128,512
        x2 = self.conv2_2modality(x2)


        if self.down:
            downindex2 = farthest_point_sample(xyz1_bn3,self.npoints // 4 //4) # b, 128
            xyz2_bn3 = index_points(xyz1_bn3,downindex2)  # n,128,3
            xyz2 = xyz2_bn3.permute(0,2,1)
            x3 = index_points(x2.permute(0,2,1),downindex2)  # b,128,c
            x3 = x3.permute(0,2,1) # b, 128,128  #---------------b,64,128 nouse
        x3_p = get_graph_feature_withdilation_and_localshape_src_dst(self.conv3_2localatten,True,downindex2,x3,x2,xyz2,xyz1,d=[3],k=self.k)
        x3_p = self.conv5(x3_p)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3_p = self.conv5_sec(x3_p)
        x3_p = x3_p.max(dim=-1, keepdim=False)[0] + self.conv5_2(x3) # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        if self.use:
            x3_voxel = self.attn3(xyz2_bn3, x3.permute(0, 2, 1))
            x3_voxel = x3_voxel.permute(0, 2, 1)
        # x3 = x3_p   # b,128,128
        x3 = torch.cat((x3_voxel,x3_p),dim=1)  # b,256,128
        x3 = self.conv3_2modality(x3)

        if self.down:
            downindex3 = farthest_point_sample(xyz2_bn3,self.npoints // 4 //4  //4) # b, 32
            xyz3_bn3 = index_points(xyz2_bn3,downindex3)  # n,32,3
            xyz3 = xyz3_bn3.permute(0,2,1)
            x4 = index_points(x3.permute(0,2,1),downindex3)  # b,32,c
            x4 = x4.permute(0,2,1) # b, 256,32    #### b,128,32
        x4_p = get_graph_feature_withdilation_and_localshape_src_dst(self.conv4_2localatten,True,downindex3,x4,x3,xyz3,xyz2,d=[4],k=self.k)
        x4_p = self.conv5_3(x4_p)
        x4_p = self.conv5_3_sec(x4_p)
        x4_p = x4_p.max(dim=-1, keepdim=False)[0] + self.conv5_4(x4)
        if self.use:
            x4_voxel = self.attn4(xyz3_bn3, x4.permute(0, 2, 1))
            x4_voxel = x4_voxel.permute(0, 2, 1)
        # x4 = x4_p
        x4 = torch.cat((x4_voxel,x4_p),dim=1)  # b,512,32
        x4 = self.conv4_2modality(x4)

        if label is not None:
            label2 = index_points(label,downindex1)
            label3 = index_points(label2,downindex2)
            label4 = index_points(label3,downindex3)

        x = self.conv6(x4)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x_mean = x.mean(dim=-1, keepdim=True)  # (b,1024,128) ->  b,1024,1
        # x_max = x.max(dim=-1,keepdim=True)[0] # b,1024,1

        l = l.view(batch_size, -1, 1)  # (batch_size, num_categoties, 1)
        l = self.conv7(l)  # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)
        # l = l.repeat(1,1,num_points // 4 // 4 // 4)

        x = torch.cat((x_mean, l), dim=1)  # (batch_size, 1088, 1)
        # global_fea = x #(#b,1088,1)  #_---------------add by wjw-----------------
        x = x.repeat(1, 1, num_points // 4 // 4 // 4)  # (b, 1088, n)

        de_4 = torch.cat((x,x4),dim=1) # b,1088+256 ,32
        de_4 = self.de4(de_4)  #b,256,32
        if label is not None:
            out4 = self.deout4(de_4)   # ,b,50,32

         # b,n,k  b,128,3
        dis32,index_32_2_128 = get_nearsest_3neighbors_anddis(xyz2_bn3,xyz3_bn3)
        # index32_2_2048 = get_nearsest_neighbors(xyz_bn3,xyz3_bn3)# b,512,2048   #---b,256,2048
        de4_up = index_points(de_4.permute(0, 2, 1), index_32_2_128)  # b,128,3,256  b,n,k,c
        dist_recip3 = 1.0 / (dis32 + 1e-8)
        norm3 = torch.sum(dist_recip3, dim=2, keepdim=True)  # 点越近给的权重就越大
        weight3 = dist_recip3 / norm3  # b,n,k
        de4_up = torch.sum(de4_up * weight3.unsqueeze(-1), dim=2).permute(0,2,1)  # b,n,c---b,c,n
        # global_fea3 = global_fea.repeat(1,1,num_points //4 //4)#-----------------------------------add by wjw
        de3 = torch.cat((de4_up,x3),dim=1)  # b,256+128,128
        de3 = self.de3(de3)  # b,256,128
        if label is not None:
            out3 = self.deout3(de3)  # b,50,128


        dis128,index128_2_512 = get_nearsest_3neighbors_anddis(xyz1_bn3, xyz2_bn3)      # b,256,2048   # b,128,2048
        de3_up = index_points(de3.permute(0, 2, 1), index128_2_512)  # b,128,3,256  b,n,k,c
        dist_recip2 = 1.0 / (dis128 + 1e-8)
        norm2 = torch.sum(dist_recip2, dim=2, keepdim=True)  # 点越近给的权重就越大
        weight2 = dist_recip2 / norm2  # b,n,k
        de3_up = torch.sum(de3_up * weight2.unsqueeze(-1), dim=2).permute(0, 2, 1)  # b,n,c---b,c,n
        # global_fea2 = global_fea.repeat(1, 1, num_points // 4 )  # -----------------------------------add by wjw
        de2 = torch.cat((de3_up,x2),dim=1)  #b,256+64,512
        de2 = self.de2(de2)  # b,128,512
        if label is not None:
            out2 = self.deout2(de2)  # b, 128,512


        dis512,index512_2_2048 = get_nearsest_3neighbors_anddis(xyz_bn3, xyz1_bn3)
        de2_up = index_points(de2.permute(0, 2, 1), index512_2_2048)  # b,128,3,256  b,n,k,c
        dist_recip1 = 1.0 / (dis512 + 1e-8)
        norm1 = torch.sum(dist_recip1, dim=2, keepdim=True)  # 点越近给的权重就越大
        weight1 = dist_recip1 / norm1  # b,n,k
        de2_up = torch.sum(de2_up * weight1.unsqueeze(-1), dim=2).permute(0, 2, 1)  # b,n,c---b,c,n
        # global_fea1 = global_fea.repeat(1, 1, num_points )  # -----------------------------------add by wjw
        de1 = torch.cat((de2_up,x1),dim=1)  # b,128+64,2048
        de1 = self.de1(de1) # b,128,2048
        out1 = self.deout1(de1) # b,50,2048

        if label is not None:
            return out1,out2,out3,out4,label2,label3,label4
        else:
            return out1

class DGCNN_partseg_voxelandpoints_splitneigh(nn.Module):
    def __init__(self, num_part, normal_channel=False, type='L', use=True, down=False):
        super(DGCNN_partseg_voxelandpoints_splitneigh, self).__init__()
        # self.args = args
        self.npoints = 2048
        self.seg_num_all = 50
        self.k = 20
        self.d = [1, 2, 3, 4]
        self.inchannel = 12 if normal_channel == True else 6
        self.transform_net = Transform_Net(self.inchannel)
        self.downsampling = down
        self.use = use
        if use:
            if type == 'L':
                # self.attn1 = Layerwise3dAttention3(npoints = 2048,inchannel=64, split_size=[64, 64, 64], local_agg = 'mean')
                self.attn2 = Layerwise3dAttention_car(npoints=2048,inchannel=64, split_size=[64, 64, 64], local_agg = 'mean')
                self.attn3 = Layerwise3dAttention_car(npoints = 2048,inchannel=128 , split_size=[32, 32, 32], local_agg = 'mean')

            else:
                self.attn1 = Transformer3dAttention(inchannel=64, split_size=[64, 64, 64])
                self.attn2 = Transformer3dAttention(inchannel=64, split_size=[64, 64, 64])
                self.attn3 = Transformer3dAttention(inchannel=64, split_size=[64, 64, 64])

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        # self.bn3 = nn.BatchNorm2d(64)
        # self.bn4 = nn.BatchNorm2d(64)
        # self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm1d(1024)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)


        self.conv1 = nn.Sequential(nn.Conv2d(13, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        #adddd-----------------
        self.conv1_2 = nn.Sequential(nn.Conv1d(3, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2_in = nn.Sequential(nn.Conv2d(64+13 , 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2_2_in = nn.Sequential(nn.Conv1d(64, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2_bo = nn.Sequential(nn.Conv2d(64 +13, 64, kernel_size=1, bias=False),
                                      nn.BatchNorm2d(64),
                                      nn.LeakyReLU(negative_slope=0.2))
        self.conv2_2_bo = nn.Sequential(nn.Conv1d(64, 64, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(64),
                                        nn.LeakyReLU(negative_slope=0.2))

        self.conv3_in = nn.Sequential(nn.Conv2d(128 +13, 128, kernel_size=1, bias=False),
                                      nn.BatchNorm2d(128),
                                      nn.LeakyReLU(negative_slope=0.2))
        self.conv3_2_in = nn.Sequential(nn.Conv1d(128, 128, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(128),
                                        nn.LeakyReLU(negative_slope=0.2))
        self.conv3_bo = nn.Sequential(nn.Conv2d(128 +13, 128, kernel_size=1, bias=False),
                                      nn.BatchNorm2d(128),
                                      nn.LeakyReLU(negative_slope=0.2))
        self.conv3_2_bo = nn.Sequential(nn.Conv1d(128, 128, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(128),
                                        nn.LeakyReLU(negative_slope=0.2))

        self.conv6 = nn.Sequential(nn.Conv1d(256+128+64, 1024, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(1024+64+64+128+256, 256, kernel_size=1, bias=False),  #1280
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=0.5)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=0.5)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                    self.bn10,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, 50, kernel_size=1, bias=False)

    def forward(self, x, l):
        tmp = x
        batch_size = x.size(0)
        num_points = x.size(2)
        xyz = x.permute(0, 2, 1)[:, :, :3]  # b,n,3
        xyz_b3n = xyz.permute(0,2,1)
        if self.inchannel == 6:
            x0 = get_graph_feature_withdilation(x,d=1, k=self.k)
                                                  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
            t = self.transform_net(x0)  # (batch_size, 3, 3)
            x = x.transpose(2, 1)  # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
            x = torch.bmm(x, t)  # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
            x = x.transpose(2, 1)  # (batch_size, num_points, 3) -> (batch_size, 3, num_points)
        else:
            x0 = get_graph_feature_withdilation(x, d=1,k=self.k)
            t = self.transform_net(x0)
            p1 = torch.bmm(x[:, 0:3, :].transpose(2, 1), t)
            p2 = torch.bmm(x[:, 3:6, :].transpose(2, 1), t)
            x = torch.cat((p1, p2), dim=2).transpose(2, 1).contiguous()

        x,x11,dis = get_graph_feature_withdilation_catlocalshape_attentionpooling_firstlayer(x,xyz_b3n, d=[1],k=self.k )   # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0] + self.conv1_2(tmp)# (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x1_in_f_k, x1_b_f_k, x1_in_c, x1_b_c, localshape1, x1_in_index, x1_bo_index,local_relation1 = get_graph_feature_inner_boundary_neighbour(
            x1, xyz_b3n, d=[2], k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x1_inner = get_inner_feature(x1_in_f_k, x1_in_c, x1_in_index, localshape1, conv1=self.conv2_in,
                                     conv2=self.conv2_2_in)  # b,c,n
        x1_bound = get_bound_feature_kreweighting(x1_b_f_k, x1_b_c, x1_bo_index, localshape1, conv1=self.conv2_bo,
                                     conv2=self.conv2_2_bo,local_edge_realtion=local_relation1)  # b,c,n
        x2_p = x1_inner + x1_bound  # b,c,n
        if self.use:
            x2_voxel = self.attn2(xyz, x1.permute(0, 2, 1))
            x2_voxel = x2_voxel.permute(0, 2, 1)
        x2 = torch.cat((x2_voxel,x2_p),dim=1)  # b,128,n
        # x2 = x2_p

        x2_in_f_k, x2_b_f_k, x2_in_c, x2_b_c, localshape2, x2_in_index, x2_bo_index,local_realation2 = get_graph_feature_inner_boundary_neighbour(
            x2, xyz_b3n, d=[3], k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x2_inner = get_inner_feature(x2_in_f_k, x2_in_c, x2_in_index, localshape2, conv1=self.conv3_in,
                                     conv2=self.conv3_2_in)  # b,c,n
        x2_bound = get_bound_feature_kreweighting(x2_b_f_k, x2_b_c, x2_bo_index, localshape2, conv1=self.conv3_bo,
                                     conv2=self.conv3_2_bo,local_edge_realtion=local_realation2)  # b,c,n
        x3_p = x2_inner + x2_bound  # b,c,n
        if self.use:
            x3_voxel = self.attn3(xyz, x2.permute(0, 2, 1))
            x3_voxel = x3_voxel.permute(0, 2, 1)
        x3 = torch.cat((x3_voxel,x3_p),dim=1)  # b,256,n

        x = torch.cat((x1, x2, x3), dim=1)  # (batch_size, 64 + 64 + 128 + 256 + 512, num_points)

        x = self.conv6(x)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        l = l.view(batch_size, -1, 1)  # (batch_size, num_categoties, 1)
        l = self.conv7(l)  # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)

        x = torch.cat((x, l), dim=1)  # (batch_size, 1088, 1)
        x = x.repeat(1, 1, num_points)  # (batch_size, 1088, num_points)

        x = torch.cat((x, x1, x2, x3), dim=1)  # (batch_size, 1088+64*3, num_points)

        x = self.conv8(x)  # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)  # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        x = self.dp2(x)
        x = self.conv10(x)  # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        x = self.conv11(x)  # (batch_size, 256, num_points) -> (batch_size, seg_num_all, num_points)

        return x

class DGCNN_partseg_voxelandpoints_splitneigh_down(nn.Module):
    def __init__(self, num_part, normal_channel=False, type='L', use=True, down=False):
        super(DGCNN_partseg_voxelandpoints_splitneigh_down, self).__init__()
        # self.args = args
        self.npoints = 2048
        self.seg_num_all = 50
        self.k = 20
        self.d = [1, 2, 3, 4]
        self.inchannel = 12 if normal_channel == True else 6
        self.transform_net = Transform_Net(self.inchannel)
        self.downsampling = down
        self.use = use
        if use:
            if type == 'L':
                # self.attn1 = Layerwise3dAttention3(npoints = 2048,inchannel=64, split_size=[64, 64, 64], local_agg = 'mean')
                self.attn2 = Layerwise3dAttention_car(npoints=2048,inchannel=64, split_size=[64, 64, 64], local_agg = 'mean')
                self.attn3 = Layerwise3dAttention_car(npoints = 2048,inchannel=128 , split_size=[32, 32, 32], local_agg = 'mean')

            else:
                self.attn1 = Transformer3dAttention(inchannel=64, split_size=[64, 64, 64])
                self.attn2 = Transformer3dAttention(inchannel=64, split_size=[64, 64, 64])
                self.attn3 = Transformer3dAttention(inchannel=64, split_size=[64, 64, 64])

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        # self.bn3 = nn.BatchNorm2d(64)
        # self.bn4 = nn.BatchNorm2d(64)
        # self.bn5 = nn.BatchNorm2d(128)
        # self.bn6 = nn.BatchNorm1d(1024)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(512)
        self.bn9 = nn.BatchNorm1d(256)
        # self.bn10 = nn.BatchNorm1d(128)


        self.conv1 = nn.Sequential(nn.Conv2d(13, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        # self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
        #                            self.bn2,
        #                            nn.LeakyReLU(negative_slope=0.2))
        #adddd-----------------
        self.conv1_2 = nn.Sequential(nn.Conv1d(3, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2_in = nn.Sequential(nn.Conv2d(64+13 , 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2_2_in = nn.Sequential(nn.Conv1d(64, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2_bo = nn.Sequential(nn.Conv2d(64 +13, 64, kernel_size=1, bias=False),
                                      nn.BatchNorm2d(64),
                                      nn.LeakyReLU(negative_slope=0.2))
        self.conv2_2_bo = nn.Sequential(nn.Conv1d(64, 64, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(64),
                                        nn.LeakyReLU(negative_slope=0.2))

        self.conv3_in = nn.Sequential(nn.Conv2d(64 +13, 128, kernel_size=1, bias=False),
                                      nn.BatchNorm2d(128),
                                      nn.LeakyReLU(negative_slope=0.2))
        self.conv3_2_in = nn.Sequential(nn.Conv1d(64, 128, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(128),
                                        nn.LeakyReLU(negative_slope=0.2))
        self.conv3_bo = nn.Sequential(nn.Conv2d(64 +13, 128, kernel_size=1, bias=False),
                                      nn.BatchNorm2d(128),
                                      nn.LeakyReLU(negative_slope=0.2))
        self.conv3_2_bo = nn.Sequential(nn.Conv1d(64, 128, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(128),
                                        nn.LeakyReLU(negative_slope=0.2))

        self.conv4_in = nn.Sequential(nn.Conv2d(128+13, 256, kernel_size=1, bias=False),
                                      nn.BatchNorm2d(256),
                                      nn.LeakyReLU(negative_slope=0.2))
        self.conv4_2_in = nn.Sequential(nn.Conv1d(128, 256, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(256),
                                        nn.LeakyReLU(negative_slope=0.2))
        self.conv4_bo = nn.Sequential(nn.Conv2d(128+13, 256, kernel_size=1, bias=False),
                                      nn.BatchNorm2d(256),
                                      nn.LeakyReLU(negative_slope=0.2))
        self.conv4_2_bo = nn.Sequential(nn.Conv1d(128, 256, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(256),
                                        nn.LeakyReLU(negative_slope=0.2))
        self.mlp_5 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # self.conv6 = nn.Sequential(nn.Conv1d(256+128+64, 1024, kernel_size=1, bias=False),
        #                            self.bn6,
        #                            nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(64+64+64+128+256+512, 512, kernel_size=1, bias=False),  #1280
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=0.5)
        self.conv9 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=0.5)
        # self.conv10 = nn.Sequential(nn.Conv1d(256, 50, kernel_size=1, bias=False),
        #                             self.bn10,
        #                             nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(256, 50, kernel_size=1, bias=False)

    def forward(self, x, l):
        tmp = x
        batch_size = x.size(0)
        num_points = x.size(2)
        xyz = x.permute(0, 2, 1)[:, :, :3]  # b,n,3
        xyz_b3n = xyz.permute(0,2,1)
        if self.inchannel == 6:
            x0 = get_graph_feature_withdilation(x,d=1, k=self.k)
                                                  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
            t = self.transform_net(x0)  # (batch_size, 3, 3)
            x = x.transpose(2, 1)  # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
            x = torch.bmm(x, t)  # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
            x = x.transpose(2, 1)  # (batch_size, num_points, 3) -> (batch_size, 3, num_points)
        else:
            x0 = get_graph_feature_withdilation(x, d=1,k=self.k)
            t = self.transform_net(x0)
            p1 = torch.bmm(x[:, 0:3, :].transpose(2, 1), t)
            p2 = torch.bmm(x[:, 3:6, :].transpose(2, 1), t)
            x = torch.cat((p1, p2), dim=2).transpose(2, 1).contiguous()

        x,x11,dis = get_graph_feature_withdilation_catlocalshape_attentionpooling_firstlayer(x,xyz_b3n, d=[1],k=self.k )   # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        # x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0] + self.conv1_2(tmp)# (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x1_in_f_k, x1_b_f_k, x1_in_c, x1_b_c, localshape1, x1_in_index, x1_bo_index,local_relation1 = get_graph_feature_inner_boundary_neighbour(
            x1, xyz_b3n, d=[1], k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x1_inner = get_inner_feature(x1_in_f_k, x1_in_c, x1_in_index, localshape1, conv1=self.conv2_in,
                                     conv2=self.conv2_2_in)  # b,c,n
        x1_bound = get_bound_feature_kreweighting(x1_b_f_k, x1_b_c, x1_bo_index, localshape1, conv1=self.conv2_bo,
                                     conv2=self.conv2_2_bo,local_edge_realtion=local_relation1)  # b,c,n
        x2_p = x1_inner + x1_bound  # b,c,n
        # if self.use:
        #     x2_voxel = self.attn2(xyz, x1.permute(0, 2, 1))
        #     x2_voxel = x2_voxel.permute(0, 2, 1)
        # x2 = torch.cat((x2_voxel,x2_p),dim=1)  # b,128,n
        x2 = x2_p

        if self.downsampling:
            downindex1 = farthest_point_sample(xyz,self.npoints // 4)  # 512
            xyz2 = index_points(xyz,downindex1)  # b,n/4 ,3
            x2 = index_points(x2.permute(0,2,1),downindex1)
            x2 = x2.permute(0,2,1) # b,c,n/4


        x2_in_f_k, x2_b_f_k, x2_in_c, x2_b_c, localshape2, x2_in_index, x2_bo_index,local_realation2 = get_graph_feature_inner_boundary_neighbour(
            x2, xyz2.permute(0,2,1), d=[2], k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x2_inner = get_inner_feature(x2_in_f_k, x2_in_c, x2_in_index, localshape2, conv1=self.conv3_in,
                                     conv2=self.conv3_2_in)  # b,c,n
        x2_bound = get_bound_feature_kreweighting(x2_b_f_k, x2_b_c, x2_bo_index, localshape2, conv1=self.conv3_bo,
                                     conv2=self.conv3_2_bo,local_edge_realtion=local_realation2)  # b,c,n
        x3_p = x2_inner + x2_bound  # b,c,n
        # if self.use:
        #     x3_voxel = self.attn3(xyz, x2.permute(0, 2, 1))
        #     x3_voxel = x3_voxel.permute(0, 2, 1)
        # x3 = torch.cat((x3_voxel,x3_p),dim=1)  # b,256,n
        x3 = x3_p

        if self.downsampling:
            downindex2 = farthest_point_sample(xyz2, self.npoints // 4 // 4)  # 128
            xyz3 = index_points(xyz2,downindex2)
            x3 = index_points(x3.permute(0,2,1),downindex2)
            x3 = x3.permute(0,2,1)

        x3_in_f_k, x3_b_f_k, x3_in_c, x3_b_c, localshape3, x3_in_index, x3_bo_index, local_realation3 = get_graph_feature_inner_boundary_neighbour(
            x3, xyz3.permute(0, 2, 1), d=[3],
            k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x3_inner = get_inner_feature(x3_in_f_k, x3_in_c, x3_in_index, localshape3, conv1=self.conv4_in,
                                     conv2=self.conv4_2_in)  # b,c,n
        x3_bound = get_bound_feature_kreweighting(x3_b_f_k, x3_b_c, x3_bo_index, localshape3, conv1=self.conv4_bo,
                                                  conv2=self.conv4_2_bo, local_edge_realtion=local_realation3)  # b,c,n
        x4_p = x3_inner + x3_bound  # b,c,n
        # if self.use:
        #     x3_voxel = self.attn3(xyz, x2.permute(0, 2, 1))
        #     x3_voxel = x3_voxel.permute(0, 2, 1)
        # x3 = torch.cat((x3_voxel,x3_p),dim=1)  # b,256,n
        x4 = x4_p

        if self.downsampling:
            downindex3 = farthest_point_sample(xyz3, self.npoints // 4 // 4 // 2)  # 64
            xyz4 = index_points(xyz3,downindex3)
            x4 = index_points(x4.permute(0,2,1),downindex3)
            x4 = x4.permute(0,2,1)

        x5 = self.mlp_5(x4) # b, 512, 64
        global_x5 = torch.mean(x5,dim=-1,keepdim=True)  # b,512,1
        x5_up = global_x5.repeat(1,1,2048)

        index512_2_2048 = get_nearsest_neighbors(xyz,xyz2)
        x2_up = index_points(x2.permute(0,2,1),index512_2_2048).permute(0,2,1) # b,64,2048

        index128_2_2048 = get_nearsest_neighbors(xyz,xyz3)
        x3_up = index_points(x3.permute(0,2,1),index128_2_2048).permute(0,2,1) # b,128,2048

        index64_2_2048 = get_nearsest_neighbors(xyz,xyz4)
        x4_up = index_points(x4.permute(0,2,1),index64_2_2048).permute(0,2,1)  # b,256,2048




        x = torch.cat((x1, x2_up, x3_up,x4_up,x5_up), dim=1)  # (batch_size, 64 + 64 + 128 + 256 + 512, num_points)

        # x = self.conv6(x)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        # x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        l = l.view(batch_size, -1, 1)  # (batch_size, num_categoties, 1)
        l = self.conv7(l).repeat(1,1,2048)  # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)

        x = torch.cat((x, l), dim=1)  # (batch_size, 1088, 1)
        # x = x.repeat(1, 1, num_points)  # (batch_size, 1088, num_points)

        # x = torch.cat((x, x1, x2, x3), dim=1)  # (batch_size, 1088+64*3, num_points)

        x = self.conv8(x)  # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)  # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        x = self.dp2(x)
        # x = self.conv10(x)  # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        x = self.conv11(x)  # (batch_size, 256, num_points) -> (batch_size, seg_num_all, num_points)

        return x


class DGCNN_semseg(nn.Module):
    def __init__(self, type='L', use=True,down=True):
        super(DGCNN_semseg, self).__init__()
        # self.args = args
        self.use = use
        self.down = down
        self.npoints = 4096
        self.k = 20
        if use:
            if type == 'L':
                self.attn1 = Layerwise3dAttention_car(npoints=4096,inchannel=64, split_size=[16,16,16], local_agg='mean')
                self.attn2 = Layerwise3dAttention_car(npoints=4096,inchannel=128, split_size=[4,4,4], local_agg='mean')
                self.attn3 = Layerwise3dAttention_car(npoints=4096,inchannel=256, split_size=[2,2,2], local_agg='mean')

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(128)
        # self.bn6 = nn.BatchNorm1d(1024)
        # self.bn7 = nn.BatchNorm1d(512)
        # self.bn8 = nn.BatchNorm1d(256)

        self.conv1 = nn.Sequential(nn.Conv2d(19, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv1_2 = nn.Sequential(
            nn.Conv1d(9,64,kernel_size=1,bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        #------------------
        self.conv3 = nn.Sequential(nn.Conv2d(64 + 12, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2_2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        #-------------
        self.conv5 = nn.Sequential(nn.Conv2d(128 + 12, 128, kernel_size=1, bias=False),   # 128   64
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5_1 = nn.Sequential(nn.Conv2d(128 , 128, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3_2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, bias=False),  # 128   64
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2)
        )
        #----------
        self.mlp_5 = nn.Sequential(nn.Conv1d(512 , 1024, kernel_size=1, bias=False),   # 512   256
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))
#------------------------
        self.conv10 = nn.Sequential(nn.Conv2d(256 + 12, 256, kernel_size=1, bias=False),  # 128 256
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.LeakyReLU(negative_slope=0.2))
        self.conv4_2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2)
        )
#960  ----  512
        self.conv6 = nn.Sequential(
            nn.Conv1d(960+1024, 512, kernel_size=1),   #  512 + 1204 ------960 + 1024
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=0.5),
            nn.Conv1d(512, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=0.5),
            nn.Conv1d(256, 13, kernel_size=1),
        )


    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)
        xyz_bn3 = x.permute(0, 2, 1)[:, :, 6:]  # b,n,3   use the normalized xyz like dgcnn
        xyz = x[:,6:,:]  # b,3,n
        tmp = x # b,9,n

        if self.down:
            downindex1 = farthest_point_sample(xyz_bn3, self.npoints // 4)  # 1024
            xyz1_bn3 = index_points(xyz_bn3, downindex1)  # b,1024 ,3
            xyz1 = xyz1_bn3.permute(0,2,1) # b,3, 1024
            x1 = index_points(x.permute(0, 2, 1), downindex1)  # b,1024,c
            x1 = x1.permute(0, 2, 1)  # b,c,1024
        x1_localpolar,x1_knn,dis1 = get_graph_feature_withdilation_catlocalshape_attentionpooling_firstlayer_src_dst(downindex1,x1,x,xyz1,xyz,d=[1],k=self.k,idx_xyz=None,dim9=True)
        # x = get_graph_feature(x, k=self.k, dim9=True)  # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x1_localpolar = self.conv1(x1_localpolar)  # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1_localpolar = self.conv2(x1_localpolar)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x1_localpolar.max(dim=-1, keepdim=False)[0] + self.conv1_2(x1) # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)



        if self.down:
            downindex2 = farthest_point_sample(xyz1_bn3, self.npoints // 4 //4)  # 256
            xyz2_bn3 = index_points(xyz1_bn3, downindex2)  # b,n/4/4 ,3
            xyz2 = xyz2_bn3.permute(0,2,1) # b, 3, 256
            x2 = index_points(x1.permute(0, 2, 1), downindex2)
            x2 = x2.permute(0, 2, 1)  # b,c,256
        if self.use:
            x2_voxel = self.attn1(xyz2_bn3, x2.permute(0, 2, 1))
            x2_voxel = x2_voxel.permute(0, 2, 1)
        x2_local = get_graph_feature_withdilation_and_localshape_src_dst(True,downindex2,x2,x1,xyz2,xyz1,d=[2],k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64 + 12, num_points, k)
        x2_local = self.conv3(x2_local)  # (batch_size, 64+12, num_points, k) -> (batch_size, 64, num_points, k)
        x2_local = self.conv4(x2_local)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2_local = x2_local.max(dim=-1, keepdim=False)[0] + self.conv2_2(x2)# (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        # x2 = x2_local  # b,64,256
        x2 = torch.cat((x2_local,x2_voxel),dim=1) # b,128,n


        if self.down:
            downindex3 = farthest_point_sample(xyz2_bn3, self.npoints // 4 //4 //4)  # 64
            xyz3_bn3 = index_points(xyz2_bn3, downindex3)  # b,n/4 ,3
            xyz3 = xyz3_bn3.permute(0,2,1)
            x3 = index_points(x2.permute(0, 2, 1), downindex3)
            x3 = x3.permute(0, 2, 1)  # b,c,64
        if self.use:
            x3_voxel = self.attn2(xyz3_bn3, x3.permute(0, 2, 1))
            x3_voxel = x3_voxel.permute(0, 2, 1)
        x3_local = get_graph_feature_withdilation_and_localshape_src_dst(True,downindex3,x3,x2,xyz3,xyz2,d=[3],k=self.k)  # b,128 + 12,n
        x3_local = self.conv5(x3_local)  # (batch_size, 128+12, num_points, k) -> (batch_size, 128, num_points, k)
        x3_local = self.conv5_1(x3_local)
        x3_local = x3_local.max(dim=-1, keepdim=False)[0] + self.conv3_2(x3) # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)
        # x3 = x3_local   # b, 128,64
        x3 = torch.cat((x3_local,x3_voxel),dim=1) # b,256,64


        if self.down:
            downindex4 = farthest_point_sample(xyz3_bn3, self.npoints // 4 //4 //4 //4)  # 16
            xyz4_bn3 = index_points(xyz3_bn3, downindex4)  # b,n/4 ,3
            xyz4 = xyz4_bn3.permute(0,2,1)  # b, 3, n
            x4 = index_points(x3.permute(0, 2, 1), downindex4)
            x4 = x4.permute(0, 2, 1)  # b,c,16
        if self.use:
            x4_voxel = self.attn3(xyz4_bn3, x4.permute(0, 2, 1))
            x4_voxel = x4_voxel.permute(0, 2, 1)
        x4_local = get_graph_feature_withdilation_and_localshape_src_dst(True, downindex4,x4,x3, xyz4,xyz3, d=[3], k=self.k)  # b,128 + 12,n
        x4_local = self.conv10(x4_local)  # (batch_size, 128+12, num_points, k) -> (batch_size, 128, num_points, k)
        x4_local = self.conv11(x4_local)
        x4_local = x4_local.max(dim=-1, keepdim=False)[0] + self.conv4_2(x4)  # b,256,16
        # x4 = x4_local  # b,256,16
        x4 = torch.cat((x4_local, x4_voxel), dim=1)  # b,512,16

        x5 = self.mlp_5(x4)  # b, 1024, 16
        global_x5 = torch.mean(x5, dim=-1, keepdim=True)  # b,1024,1
        x5_up = global_x5.repeat(1, 1, 4096)

        index1024_2_4096 = get_nearsest_neighbors(xyz_bn3, xyz1_bn3)
        x1_up = index_points(x1.permute(0, 2, 1), index1024_2_4096).permute(0, 2, 1)  # b,64,4096

        index256_2_4096 = get_nearsest_neighbors(xyz_bn3, xyz2_bn3)
        x2_up = index_points(x2.permute(0, 2, 1), index256_2_4096).permute(0, 2, 1)  # b,128,4096    ---64

        index64_2_4096 = get_nearsest_neighbors(xyz_bn3, xyz3_bn3)
        x3_up = index_points(x3.permute(0, 2, 1), index64_2_4096).permute(0, 2, 1)  # b,256,4096     ---128

        index16_2_4096 = get_nearsest_neighbors(xyz_bn3,xyz4_bn3)
        x4_up = index_points(x4.permute(0,2,1),index16_2_4096).permute(0,2,1) # b,512,4096        ---256

        x = torch.cat((x1_up, x2_up, x3_up,x4_up,x5_up), dim=1)  # (batch_size, 256+128+64+512+1024, num_points)

        x = self.conv6(x)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        # x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)
        #
        # x = x.repeat(1, 1, num_points)  # (batch_size, 1024, num_points)
        # x = torch.cat((x, x1, x2, x3), dim=1)  # (batch_size, 1024+64*3, num_points)
        #
        # x = self.conv7(x)  # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        # x = self.conv8(x)  # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        # x = self.dp1(x)
        # x = self.conv9(x)  # (batch_size, 256, num_points) -> (batch_size, 13, num_points)

        return x, x3

class DGCNN_semseg_l(nn.Module):
    def __init__(self, type='L', use=True,down=True):
        super(DGCNN_semseg_l, self).__init__()
        # self.args = args
        self.use = use
        self.down = down
        self.npoints = 4096
        self.k = 20
        if use:
            if type == 'L':
                self.attn1 = Layerwise3dAttention_car(npoints=4096,inchannel=64, split_size=[16,16,16], local_agg='mean')
                self.attn2 = Layerwise3dAttention_car(npoints=4096,inchannel=128, split_size=[4,4,4], local_agg='mean')
                self.attn3 = Layerwise3dAttention_car(npoints=4096,inchannel=256, split_size=[2,2,2], local_agg='mean')

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(128)
        # self.bn6 = nn.BatchNorm1d(1024)
        # self.bn7 = nn.BatchNorm1d(512)
        # self.bn8 = nn.BatchNorm1d(256)

        self.conv1 = nn.Sequential(nn.Conv2d(19, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv1_2 = nn.Sequential(
            nn.Conv1d(9,64,kernel_size=1,bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        #------------------
        self.conv3 = nn.Sequential(nn.Conv2d(64 + 12, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2_2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        #-------------
        self.conv5 = nn.Sequential(nn.Conv2d(64 + 12, 128, kernel_size=1, bias=False),   # 128   64
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5_1 = nn.Sequential(nn.Conv2d(128 , 128, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3_2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1, bias=False),  # 128   64
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2)
        )
        #----------
        self.mlp_5 = nn.Sequential(nn.Conv1d(256 , 1024, kernel_size=1, bias=False),   # 512   256
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))
#------------------------
        self.conv10 = nn.Sequential(nn.Conv2d(128 + 12, 256, kernel_size=1, bias=False),  # 128 256
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.LeakyReLU(negative_slope=0.2))
        self.conv4_2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2)
        )
#960  ----  512
        self.conv6 = nn.Sequential(
            nn.Conv1d(512+1024, 512, kernel_size=1),   #  512 + 1204 ------960 + 1024
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=0.5),
            nn.Conv1d(512, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=0.5),
            nn.Conv1d(256, 13, kernel_size=1),
        )


    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)
        xyz_bn3 = x.permute(0, 2, 1)[:, :, 6:]  # b,n,3   use the normalized xyz like dgcnn
        xyz = x[:,6:,:]  # b,3,n
        tmp = x # b,9,n


        x1_localpolar, x11, x12 = get_graph_feature_withdilation_catlocalshape_attentionpooling_firstlayer(x, xyz, d=[1], k=self.k,idx_xyz=None, dim9=True)
        # x1_localpolar,x1_knn,dis1 = get_graph_feature_withdilation_catlocalshape_attentionpooling_firstlayer_src_dst(downindex1,x1,x,xyz1,xyz,d=[1],k=self.k,idx_xyz=None,dim9=True)
        # x = get_graph_feature(x, k=self.k, dim9=True)  # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x1_localpolar = self.conv1(x1_localpolar)  # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1_localpolar = self.conv2(x1_localpolar)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x1_localpolar.max(dim=-1, keepdim=False)[0] + self.conv1_2(tmp) # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        if self.down:
            downindex1 = farthest_point_sample(xyz_bn3, self.npoints // 4)  # 1024
            xyz1_bn3 = index_points(xyz_bn3, downindex1)  # b,1024 ,3
            xyz1 = xyz1_bn3.permute(0,2,1) # b,3, 1024
            x1_d = index_points(x1.permute(0, 2, 1), downindex1)  # b,1024,c
            x1_d = x1_d.permute(0, 2, 1)  # b,64,1024



        if self.use:
            x2_voxel = self.attn1(xyz1_bn3, x1_d.permute(0, 2, 1))
            x2_voxel = x2_voxel.permute(0, 2, 1)
        x2_local = get_graph_feature_withdilation_and_localshape_src_dst(True,downindex1,x1_d,x1,xyz1,xyz,d=[2],k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64 + 12, num_points, k)
        x2_local = self.conv3(x2_local)  # (batch_size, 64+12, num_points, k) -> (batch_size, 64, num_points, k)
        x2_local = self.conv4(x2_local)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2_local = x2_local.max(dim=-1, keepdim=False)[0] + self.conv2_2(x1_d)# (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x2 = x2_local  # b,64,256
        # x2 = torch.cat((x2_local,x2_voxel),dim=1) # b,128,1024
        if self.down:
            downindex2 = farthest_point_sample(xyz1_bn3, self.npoints // 4 // 4)  # 256
            xyz2_bn3 = index_points(xyz1_bn3, downindex2)  # b,n/4/4 ,3
            xyz2 = xyz2_bn3.permute(0, 2, 1)  # b, 3, 256
            x2_d = index_points(x2.permute(0, 2, 1), downindex2)
            x2_d = x2_d.permute(0, 2, 1)  # b,128,256



        if self.use:
            x3_voxel = self.attn2(xyz2_bn3, x2_d.permute(0, 2, 1))
            x3_voxel = x3_voxel.permute(0, 2, 1)
        x3_local = get_graph_feature_withdilation_and_localshape_src_dst(True,downindex2,x2_d,x2,xyz2,xyz1,d=[3],k=self.k)  # b,128 + 12,n
        x3_local = self.conv5(x3_local)  # (batch_size, 128+12, num_points, k) -> (batch_size, 128, num_points, k)
        x3_local = self.conv5_1(x3_local)
        x3_local = x3_local.max(dim=-1, keepdim=False)[0] + self.conv3_2(x2_d) # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)
        x3 = x3_local   # b, 128,64
        # x3 = torch.cat((x3_local,x3_voxel),dim=1) # b,256,256
        if self.down:
            downindex3 = farthest_point_sample(xyz2_bn3, self.npoints // 4 // 4 // 4)  # 64
            xyz3_bn3 = index_points(xyz2_bn3, downindex3)  # b,n/4 ,3
            xyz3 = xyz3_bn3.permute(0, 2, 1)
            x3_d = index_points(x3.permute(0, 2, 1), downindex3)
            x3_d = x3_d.permute(0, 2, 1)  # b,256,64


        if self.use:
            x4_voxel = self.attn3(xyz3_bn3, x3_d.permute(0, 2, 1))
            x4_voxel = x4_voxel.permute(0, 2, 1)
        x4_local = get_graph_feature_withdilation_and_localshape_src_dst(True, downindex3,x3_d,x3, xyz3,xyz2, d=[4], k=self.k)  # b,128 + 12,n
        x4_local = self.conv10(x4_local)  # (batch_size, 128+12, num_points, k) -> (batch_size, 128, num_points, k)
        x4_local = self.conv11(x4_local)
        x4_local = x4_local.max(dim=-1, keepdim=False)[0] + self.conv4_2(x3_d)  # b,256,16
        x4 = x4_local  # b,256,16
        # x4 = torch.cat((x4_local, x4_voxel), dim=1)  # b,512,64

        # if self.down:
        #     downindex4 = farthest_point_sample(xyz3_bn3, self.npoints // 4 //4 //4 //4)  # 16
        #     xyz4_bn3 = index_points(xyz3_bn3, downindex4)  # b,n/4 ,3
        #     xyz4 = xyz4_bn3.permute(0,2,1)  # b, 3, n
        #     x4 = index_points(x3.permute(0, 2, 1), downindex4)
        #     x4 = x4.permute(0, 2, 1)  # b,c,16

        x5 = self.mlp_5(x4)  # b, 1024, 64
        global_x5 = torch.mean(x5, dim=-1, keepdim=True)  # b,1024,1
        x5_up = global_x5.repeat(1, 1, 4096)

        index1024_2_4096 = get_nearsest_neighbors(xyz_bn3, xyz1_bn3)
        x2_up = index_points(x2.permute(0, 2, 1), index1024_2_4096).permute(0, 2, 1)  # b,128,4096

        index256_2_4096 = get_nearsest_neighbors(xyz_bn3, xyz2_bn3)
        x3_up = index_points(x3.permute(0, 2, 1), index256_2_4096).permute(0, 2, 1)  # b,256,4096    ---64

        index64_2_4096 = get_nearsest_neighbors(xyz_bn3, xyz3_bn3)
        x4_up = index_points(x4.permute(0, 2, 1), index64_2_4096).permute(0, 2, 1)  # b,512,4096     ---128

        # index16_2_4096 = get_nearsest_neighbors(xyz_bn3,xyz4_bn3)
        # x4_up = index_points(x4.permute(0,2,1),index16_2_4096).permute(0,2,1) # b,512,4096        ---256
        x1_up = x1  # b,64,4096

        x = torch.cat((x1_up, x2_up, x3_up,x4_up,x5_up), dim=1)  # (batch_size, 256+128+64+512+1024, num_points)

        x = self.conv6(x)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        # x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)
        #
        # x = x.repeat(1, 1, num_points)  # (batch_size, 1024, num_points)
        # x = torch.cat((x, x1, x2, x3), dim=1)  # (batch_size, 1024+64*3, num_points)
        #
        # x = self.conv7(x)  # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        # x = self.conv8(x)  # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        # x = self.dp1(x)
        # x = self.conv9(x)  # (batch_size, 256, num_points) -> (batch_size, 13, num_points)

        return x, x3

class DGCNN_semseg_l_decoder(nn.Module):
    def __init__(self, type='L', use=True,down=True):
        super(DGCNN_semseg_l_decoder, self).__init__()
        # self.args = args
        self.use = use
        self.down = down
        self.npoints = 4096
        self.k = 20
        if use:
            if type == 'L':
                self.attn1 = Layerwise3dAttention_car(npoints=4096,inchannel=64, split_size=[256,256,256], local_agg='mean')
                self.attn2 = Layerwise3dAttention_car(npoints=4096,inchannel=128, split_size=[64,64,64], local_agg='mean')
                self.attn3 = Layerwise3dAttention_car(npoints=4096,inchannel=256, split_size=[16,16,16], local_agg='mean')

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(128)
        # self.bn6 = nn.BatchNorm1d(1024)
        # self.bn7 = nn.BatchNorm1d(512)
        # self.bn8 = nn.BatchNorm1d(256)

        self.conv1 = nn.Sequential(nn.Conv2d(19, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv1_2 = nn.Sequential(
            nn.Conv1d(9,64,kernel_size=1,bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        #------------------
        self.conv2_localatt = nn.Sequential(nn.Linear(self.k,self.k), # nn.LeakyReLU(0.2),nn.Linear(self.k,self.k),
                                   nn.Softmax(-1))
        self.conv3 = nn.Sequential(nn.Conv2d(64 + 12, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2_modality = nn.Sequential(
            nn.Conv1d(128,128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        #-------------
        self.conv3_localatt =nn.Sequential(nn.Linear(self.k,self.k),  # nn.LeakyReLU(0.2),nn.Linear(self.k,self.k),
                                   nn.Softmax(-1))
        self.conv5 = nn.Sequential(nn.Conv2d(128 + 12, 128, kernel_size=1, bias=False),   # 128   64
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5_1 = nn.Sequential(nn.Conv2d(128 , 128, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3_modality = nn.Sequential(
            nn.Conv1d(256,256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, bias=False),  # 128   64
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2)
        )
        #----------
        self.mlp_5 = nn.Sequential(nn.Conv1d(512 , 1024, kernel_size=1, bias=False),   # 512   256
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))

        # self.multi_hot_label = nn.Linear(1024,13)
#------------------------
        self.conv4_localatt = nn.Sequential(nn.Linear(self.k,self.k),  #nn.LeakyReLU(0.2),nn.Linear(self.k,self.k),
                                   nn.Softmax(-1))
        self.conv10 = nn.Sequential(nn.Conv2d(256 + 12, 256, kernel_size=1, bias=False),  # 128 256
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.LeakyReLU(negative_slope=0.2))
        self.conv4_modality = nn.Sequential(
            nn.Conv1d(512,512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv4_2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1, bias=False),   #256 ---128
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2)
        )
#256  ----  512

        self.de4 = nn.Sequential(
            nn.Conv1d(1024+512,256,1,bias=False),nn.BatchNorm1d(256),nn.LeakyReLU(negative_slope=0.2),nn.Conv1d(256,256,1,bias=False),nn.BatchNorm1d(256),nn.LeakyReLU(negative_slope=0.2)
        )
        if self.training:
            self.out4 = nn.Conv1d(256  ,13,1,bias=False)

        self.de3 = nn.Sequential(
            nn.Conv1d(256 + 256 , 256, 1, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(negative_slope=0.2),   # 128---256
            nn.Conv1d(256, 256, 1, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(negative_slope=0.2)
        )
        if self.training:
            self.out3 = nn.Conv1d(256  , 13, 1, bias=False)

        self.de2 = nn.Sequential(
            nn.Conv1d(256 + 128, 256, 1, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(negative_slope=0.2),   # 64---128
            nn.Conv1d(256, 128, 1, bias=False), nn.BatchNorm1d(128), nn.LeakyReLU(negative_slope=0.2)
        )
        if self.training:
            self.out2 = nn.Conv1d(128  , 13, 1, bias=False)

        self.de1 = nn.Sequential(
            nn.Conv1d(128 + 64, 128, 1, bias=False), nn.BatchNorm1d(128), nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(128, 128, 1, bias=False), nn.BatchNorm1d(128), nn.LeakyReLU(negative_slope=0.2)
        )

        self.out1 = nn.Conv1d(128  , 13, 1, bias=False)

    def forward(self, x,label=None):
        batch_size = x.size(0)
        num_points = x.size(2)
        # xyz_bn3 = x.permute(0, 2, 1)[:, :, :3]  # b,n,3   use the normalized xyz like pointnwt++
        xyz_bn3 = x.permute(0, 2, 1)[:, :, 6:]  # b,n,3   use the normalized xyz like dgcnn
        xyz = x[:,6:,:]  # b,3,n  like dgcnn
        # xyz = x[:,:3,:]  # like pointnet++
        tmp = x # b,9,n


        x1_localpolar, x11, x12 = get_graph_feature_withdilation_catlocalshape_attentionpooling_firstlayer(x, xyz, d=[1], k=self.k,idx_xyz=None, dim9=True)
        # x1_localpolar,x1_knn,dis1 = get_graph_feature_withdilation_catlocalshape_attentionpooling_firstlayer_src_dst(downindex1,x1,x,xyz1,xyz,d=[1],k=self.k,idx_xyz=None,dim9=True)
        # x = get_graph_feature(x, k=self.k, dim9=True)  # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x1_localpolar = self.conv1(x1_localpolar)  # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1_localpolar = self.conv2(x1_localpolar)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x1_localpolar.max(dim=-1, keepdim=False)[0] + self.conv1_2(tmp) # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        if self.down:
            downindex1 = farthest_point_sample(xyz_bn3, self.npoints // 4)  # 1024
            xyz1_bn3 = index_points(xyz_bn3, downindex1)  # b,1024 ,3
            xyz1 = xyz1_bn3.permute(0,2,1) # b,3, 1024
            x1_d = index_points(x1.permute(0, 2, 1), downindex1)  # b,1024,c
            x1_d = x1_d.permute(0, 2, 1)  # b,64,1024



        if self.use:
            x2_voxel = self.attn1(xyz1_bn3, x1_d.permute(0, 2, 1))
            x2_voxel = x2_voxel.permute(0, 2, 1)
        x2_local = get_graph_feature_withdilation_and_localshape_src_dst(self.conv2_localatt,True,downindex1,x1_d,x1,xyz1,xyz,d=[2],k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64 + 12, num_points, k)
        x2_local = self.conv3(x2_local)  # (batch_size, 64+12, num_points, k) -> (batch_size, 64, num_points, k)
        x2_local = self.conv4(x2_local)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2_local = x2_local.max(dim=-1, keepdim=False)[0] + self.conv2_2(x1_d)# (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        # x2 = x2_local  # b,64,256
        x2 = torch.cat((x2_local,x2_voxel),dim=1) # b,128,1024
        x2 = self.conv2_modality(x2)

        if self.down:
            downindex2 = farthest_point_sample(xyz1_bn3, self.npoints // 4 // 4)  # 256
            xyz2_bn3 = index_points(xyz1_bn3, downindex2)  # b,n/4/4 ,3
            xyz2 = xyz2_bn3.permute(0, 2, 1)  # b, 3, 256
            x2_d = index_points(x2.permute(0, 2, 1), downindex2)
            x2_d = x2_d.permute(0, 2, 1)  # b,128,256



        if self.use:
            x3_voxel = self.attn2(xyz2_bn3, x2_d.permute(0, 2, 1))
            x3_voxel = x3_voxel.permute(0, 2, 1)
        x3_local = get_graph_feature_withdilation_and_localshape_src_dst(self.conv3_localatt,True,downindex2,x2_d,x2,xyz2,xyz1,d=[3],k=self.k)  # b,128 + 12,n
        x3_local = self.conv5(x3_local)  # (batch_size, 128+12, num_points, k) -> (batch_size, 128, num_points, k)
        x3_local = self.conv5_1(x3_local)
        x3_local = x3_local.max(dim=-1, keepdim=False)[0] + self.conv3_2(x2_d) # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)
        # x3 = x3_local   # b, 128,64
        x3 = torch.cat((x3_local,x3_voxel),dim=1) # b,256,256
        x3 = self.conv3_modality(x3)

        if self.down:
            downindex3 = farthest_point_sample(xyz2_bn3, self.npoints // 4 // 4 // 4)  # 64
            xyz3_bn3 = index_points(xyz2_bn3, downindex3)  # b,n/4 ,3
            xyz3 = xyz3_bn3.permute(0, 2, 1)
            x3_d = index_points(x3.permute(0, 2, 1), downindex3)
            x3_d = x3_d.permute(0, 2, 1)  # b,256,64


        if self.use:
            x4_voxel = self.attn3(xyz3_bn3, x3_d.permute(0, 2, 1))
            x4_voxel = x4_voxel.permute(0, 2, 1)
        x4_local = get_graph_feature_withdilation_and_localshape_src_dst(self.conv4_localatt,True, downindex3,x3_d,x3, xyz3,xyz2, d=[4], k=self.k)  # b,128 + 12,n
        x4_local = self.conv10(x4_local)  # (batch_size, 128+12, num_points, k) -> (batch_size, 128, num_points, k)
        x4_local = self.conv11(x4_local)
        x4_local = x4_local.max(dim=-1, keepdim=False)[0] + self.conv4_2(x3_d)  # b,256,16
        # x4 = x4_local  # b,256,64
        x4 = torch.cat((x4_local, x4_voxel), dim=1)  # b,512,64
        x4 = self.conv4_modality(x4)

        if label is not None:
            label2 = index_points(label,downindex1)
            label3 = index_points(label2,downindex2)
            label4 = index_points(label3,downindex3)
        x5 = self.mlp_5(x4)  # b, 1024, 64
        global_x5 = torch.mean(x5, dim=-1, keepdim=True)  # b,1024,1
        # global_x5 = torch.max(x5,dim=-1,keepdim=True)[0]

        # global_multi_label = self.multi_hot_label(global_x5.squeeze(-1))  # b,13
        # global_multi_label = torch.sigmoid(global_multi_label)

        x5_up = global_x5.repeat(1, 1, self.npoints // 4 // 4 // 4)
        de4 = torch.cat((x5_up,x4),dim=1) # b,1024+64,64
        # de4 = torch.cat((de4, global_multi_label.unsqueeze(-1).repeat(1, 1, self.npoints // 4 // 4 // 4)), dim=1)  # increase the weight of current labels
        de4 = self.de4(de4)
        if label is not None:
            out4 = self.out4(de4)

        dis3,index3 = get_nearsest_3neighbors_anddis(xyz2_bn3,xyz3_bn3)
        de4up = index_points(de4.permute(0,2,1),index3)
        dist3 = 1 / (dis3 + 1e-8)
        norm3 = torch.sum(dist3,dim=2,keepdim=True)
        weight3 = dist3 / norm3
        de4up = torch.sum(de4up * weight3.unsqueeze(-1),dim=-2).permute(0,2,1)
        de3 = torch.cat((de4up,x3),dim=1)
        # de3 = torch.cat( (de3  ) ,dim=1)
        de3 = self.de3(de3)
        if label is not None:
            out3 = self.out3(de3)

        dis2, index2 = get_nearsest_3neighbors_anddis(xyz1_bn3, xyz2_bn3)
        de3up = index_points(de3.permute(0, 2, 1), index2)
        dist2 = 1 / (dis2 + 1e-8)
        norm2 = torch.sum(dist2, dim=2, keepdim=True)
        weight2 = dist2 / norm2
        de3up = torch.sum(de3up * weight2.unsqueeze(-1), dim=-2).permute(0, 2, 1)
        de2 = torch.cat((de3up, x2), dim=1)
        de2 = self.de2(de2)
        # de2 = torch.cat( (de2 , global_multi_label.unsqueeze(-1).repeat(1,1,64*4*4) ) ,dim=1)
        if label is not None:
            out2 = self.out2(de2)

        dis1, index1 = get_nearsest_3neighbors_anddis(xyz_bn3, xyz1_bn3)
        de2up = index_points(de2.permute(0, 2, 1), index1)
        dist1 = 1 / (dis1 + 1e-8)
        norm1 = torch.sum(dist1, dim=2, keepdim=True)
        weight1 = dist1 / norm1
        de2up = torch.sum(de2up * weight1.unsqueeze(-1), dim=-2).permute(0, 2, 1)
        de1 = torch.cat((de2up, x1 ), dim=1)
        de1 = self.de1(de1)
        # de1 = torch.cat( (de1 , global_multi_label.unsqueeze(-1).repeat(1,1,4096) ),dim=1 )
        out1 = self.out1(de1)

        if self.training:
            return out1,out2,out3,out4,label2,label3,label4
        else:
            return out1, x3

class get_part_loss(nn.Module):
    def __init__(self):
        super(get_part_loss, self).__init__()

    def forward(self, pre, target):
        loss = cal_loss(pre, target)
        return loss

class get_semseg_multihot_loss(nn.Module):
    def __init__(self):
        super(get_semseg_multihot_loss, self).__init__()

    def forward(self, pre,ml_pre, target,ml):
        loss1 = cal_loss(pre, target)
        loss2 = F.binary_cross_entropy(ml_pre,ml)
        loss = loss1 + loss2
        return loss



class get_part_decoder_loss(nn.Module):
    def __init__(self):
        super(get_part_decoder_loss,self).__init__()
    def forward(self,pre1,pre2,pre3,pre4,label1,label2,label3,label4):

        pre2 = pre2.permute(0,2,1).contiguous().view(-1,50)
        pre3 = pre3.permute(0,2,1).contiguous().view(-1,50)
        pre4 = pre4.permute(0, 2, 1).contiguous().view(-1, 50)

        label2 = label2.view(-1,1)[:,0]
        label3 = label3.view(-1, 1)[:, 0]
        label4 = label4.view(-1, 1)[:, 0]

        loss1 = cal_loss(pre1,label1)
        loss2 = cal_loss(pre2, label2)
        loss3 = cal_loss(pre3, label3)
        loss4 = cal_loss(pre4, label4)
        return loss1 +  loss2 +loss3 + loss4

class get_semseg_decoder_multihot_nultilabel_loss(nn.Module):
    def __init__(self):
        super(get_semseg_decoder_multihot_nultilabel_loss,self).__init__()
    def forward(self,pre1,pre2,pre3,pre4,label1,label2,label3,label4,ml_pre,ml):

        pre2 = pre2.permute(0,2,1).contiguous().view(-1,13)
        pre3 = pre3.permute(0,2,1).contiguous().view(-1,13)
        pre4 = pre4.permute(0, 2, 1).contiguous().view(-1, 13)

        label2 = label2.view(-1,1)[:,0]
        label3 = label3.view(-1, 1)[:, 0]
        label4 = label4.view(-1, 1)[:, 0]

        loss1 = cal_loss(pre1,label1)
        loss2 = cal_loss(pre2, label2)
        loss3 = cal_loss(pre3, label3)
        loss4 = cal_loss(pre4, label4)
      #  lossml = F.binary_cross_entropy(ml_pre, ml)
        return loss1 + loss2 + loss3 + loss4 #+ 0.5 * lossml

class get_seg_loss(nn.Module):
    def __init__(self):
        super(get_seg_loss, self).__init__()

    def forward(self, pre, target, feas, weight):
        loss = cal_loss_seg(pre, target, weight, smoothing=True)
        return loss


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, diff1, diff2, diff3, diff4):
        smooth_loss = cal_loss(pred, target)
        loss1 = torch.norm(diff1, dim=1)  # b,n
        loss1 = torch.mean(torch.mean(loss1, -1), -1)  # scalar

        loss2 = torch.norm(diff2, dim=1)  # b,n
        loss2 = torch.mean(torch.mean(loss2, -1), -1)  # scalar
        loss3 = torch.norm(diff3, dim=1)  # b,n
        loss3 = torch.mean(torch.mean(loss3, -1), -1)  # scalar
        loss4 = torch.norm(diff4, dim=1)  # b,n
        loss4 = torch.mean(torch.mean(loss4, -1), -1)  # scalar

        total_loss = smooth_loss + 0.1 * loss1 + 0.1 * loss2 + 0.1 * loss3 + 0.1 * loss4

        return total_loss


class get_loss2(nn.Module):
    def __init__(self):
        super(get_loss2, self).__init__()

    def forward(self, pred, target, diff1):
        total_loss = cal_loss(pred, target)
        return total_loss


def cal_loss(pred, gold, smoothing=True):
    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


def cal_loss_seg(pred, gold, weight, smoothing=True):
    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, weight=weight, reduction='mean')

    return loss


if __name__ == "__main__":
    import datetime
    import time
    print(datetime.datetime.now().time())
    points = torch.rand(1, 9 // 3, 4096 // 4).cuda()
    l = torch.rand(1, 13)
    # model = DGCNN_partseg_voxelandpoints_splitneigh(num_part=50,normal_channel=False,type='L',use=True,down=False)
    model = DGCNN_cls_cylinder(normal_channel=False, type='L', use=False, use_down=False).cuda()
    # model = DGCNN_partseg_voxelandpoints_decoder(50, normal_channel=False, type='L', use=True, down=True)
    # model = DGCNN_semseg_l_decoder(type='L', use=False, down=True)
    s = time.time()
    out = model(points)
    e = time.time()
    print(1/((e-s)/10))
    # from thoe-sp import profile
    #
    # # input1 = torch.randn(4, 3, 224, 224)
    # flops, params = profile(model, inputs=(points,))
    # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    # print('Params = ' + str(params / 1000 ** 2) + 'M')




