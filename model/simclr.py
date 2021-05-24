import sys
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision
import itertools
from IPython import embed

sys.path.append('../')
from backbone.select_backbone import select_backbone

from utils.utils import GatherLayer
from utils.soft_dtw_cuda import SoftDTW

class SimCLR_Naked(nn.Module):
    '''
    Basically, it's a MoCo for video input: https://arxiv.org/abs/1911.05722
    '''
    def __init__(self, network='s3d', dim=128, T=0.07, distributed=True, nonlinear=True):
        '''
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 2048)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        '''
        super(SimCLR_Naked, self).__init__()

        self.dim = dim
        self.distributed = distributed
        self.T = T

        # assert not distributed, 'distributed simclr is not supported yet'
        self.nonlinear = nonlinear

        # create the encoders (including non-linear projection head: 2 FC layers)
        backbone, self.param = select_backbone(network)
        feature_size = self.param['feature_size']
        self.encoder_q = nn.ModuleList([
            backbone,
            nn.AdaptiveAvgPool3d((1, 1, 1))])
        if nonlinear:
            self.encoder_q.extend([
                nn.Conv3d(feature_size, feature_size, kernel_size=1, bias=True),
                nn.ReLU(),
                nn.Conv3d(feature_size, dim, kernel_size=1, bias=True)
            ])

        self.criterion = nn.CrossEntropyLoss()

        # Notes: for handling sibling videos, e.g. for UCF101 dataset

    def calc_contrast_loss(self, features, n_views=2, prefix='clip_'):
        # input features is normed features
        assert len(features.size()) == 3, features.size()
        B, N, dim = features.size()
        assert N == n_views and dim == self.dim, features.size()
        # distributed gathering
        if self.distributed:
            features = torch.cat(GatherLayer.apply(features), dim=0)
            N = features.size(0)
            features = features.view(N, n_views, dim).permute(1,0,2).contiguous().view(n_views*N, dim)# (2N)xd
        # assert features.size(0) % 2 == 0
        # N = features.size(0)// n_views
        labels = torch.cat([torch.arange(N) for i in range(n_views)], dim=0)  # (2, N) -> (2*N,)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # (2B, 2B)
        labels = labels.cuda()

        similarity_matrix = torch.matmul(features, features.T)  # (2B,2B)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)  # (2B, 2B-1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)  # (2B, 2B-1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0]).long().cuda()

        logits = logits / self.T

        contrast_loss = self.criterion(logits, labels)

        ret = {
            f"{prefix}logits": logits,
            f"{prefix}labels": labels,
            f"{prefix}contrast_loss": contrast_loss
        }

        return ret

    def forward(self, block):
        '''
            modified from simCLR
            https://github.com/sthalles/SimCLR/blob/1848fc934ad844ae630e6c452300433fe99acfd9/simclr.py#L26
        '''
        B = block.size(0)

        (batch_size, n_views, *_) = block.shape  # [B,N,C,T,H,W]
        assert n_views == 2
        x = block.view(-1, *(block.size()[2:])) # (B*n, ...)
        features = x
        for i, mod in enumerate(self.encoder_q):
            features = mod(features)
            if i == 1:
                backbone_features = features

        features = F.normalize(features, dim=1).squeeze().view(B, n_views, self.dim)

        ret = self.calc_contrast_loss(features, n_views, 'clip_')

        return ret

    def get_features(self, block):
        B, C, T, H, W = block.size()
        _, feature_list = self.encoder_q[0](block, ret_frame_feature=True, multi_level=True)
        attn_list = [feature.mean(dim=1) for feature in feature_list]
        return attn_list


class SimCLR_TimeSeriesV4(nn.Module):
    '''
    Basically, it's a MoCo for video input: https://arxiv.org/abs/1911.05722
    '''

    def __init__(self, network='s3d', dim=128, T=0.07, distributed=True, nonlinear=True, n_series=2, series_dim=64,
                 series_T=0.07, aligned_T=0.07, mode="clip-sr-tc", args=None):
        '''
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 2048)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        '''
        super(SimCLR_TimeSeriesV4, self).__init__()

        self.cnt = 0
        self.args = args
        self.dim = dim
        self.distributed = distributed
        self.T = T
        # assert not distributed, 'distributed simclr is not supported yet'
        self.nonlinear = nonlinear
        self.n_series = n_series
        self.series_dim = series_dim
        self.series_T = series_T
        self.aligned_T = aligned_T
        self.mode = mode
        self.with_clip = 'clip' in mode
        self.with_sr = 'sr' in mode
        self.with_tc = 'tc' in mode

        # create the encoders (including non-linear projection head: 2 FC layers)
        backbone, self.param = select_backbone(network)
        feature_size = self.param['feature_size']
        self.encoder_q = nn.ModuleList([
            backbone,
            nn.AdaptiveAvgPool3d((1, 1, 1))])
        if nonlinear and self.with_clip:
            self.encoder_q.extend([
                nn.Conv3d(feature_size, feature_size, kernel_size=1, bias=True),
                nn.ReLU(),
                nn.Conv3d(feature_size, dim, kernel_size=1, bias=True)
            ])

        self.criterion = nn.CrossEntropyLoss()

        self.series_proj_head = nn.Sequential(
                    nn.Conv3d(feature_size, feature_size, kernel_size=1, bias=True),
                    nn.ReLU(),
                    nn.Conv3d(feature_size, series_dim*self.n_series, kernel_size=1, bias=True)
                )
        # Notes: for handling sibling videos, e.g. for UCF101 dataset

    def calc_clip_contrast_loss(self, features, n_views=2, prefix='clip_'):
        # input features is normed features
        assert len(features.size()) == 3, features.size()
        B, N, dim = features.size()
        assert N == n_views , features.size()
        # distributed gathering
        N = B
        if self.distributed:
            features = torch.cat(GatherLayer.apply(features), dim=0)
            N = features.size(0)
            features = features.view(N, n_views, dim).permute(1,0,2).contiguous().view(n_views*N, dim)# (2N)xd
        else:
            features = features.permute(1,0,2).contiguous().view(n_views*B, dim)
        # assert features.size(0) % 2 == 0
        # N = features.size(0)// n_views
        labels = torch.cat([torch.arange(N) for i in range(n_views)], dim=0)  # (2, N) -> (2*N,)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # (2B, 2B)
        labels = labels.cuda()

        similarity_matrix = torch.matmul(features, features.T)  # (2B,2B)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)  # (2B, 2B-1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)  # (2B, 2B-1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0]).long().cuda()

        logits = logits / self.T

        contrast_loss = self.criterion(logits, labels)

        ret = {
            f"{prefix}logits": logits,
            f"{prefix}labels": labels,
            f"{prefix}contrast_loss": contrast_loss
        }

        return ret

    def calc_ranking_loss(self, features, n_views=2, prefix='ranking_', weight=1.):
        '''
            corresponding shuffled features should be the same
            while also surpasing the second highest features by margin (hyperparam) = 0
        '''
        # input features is normed features
        assert len(features.size()) == 4, features.size()
        Bn, n_series, N, dim = features.size()
        assert n_series == self.n_series
        assert N == n_views and dim == self.series_dim, features.size()

        labels = torch.cat([torch.arange(n_series) for i in range(n_views)], dim=0)  # (2, n_series) -> (2*n_Series,)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # (2s, 2s)
        labels = labels.cuda()

        features = features.permute(0,2,1,3).contiguous().view(Bn, n_views*n_series, dim)

        similarity_matrix = torch.bmm(features, features.transpose(2,1).contiguous()) # (bn, 2s, 2s)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda().unsqueeze(0).expand_as(similarity_matrix)
        corr_mask_1 = torch.cat([torch.zeros(n_series, n_series), torch.eye(n_series)], dim=1)
        corr_mask_2 = torch.cat([torch.eye(n_series), torch.zeros(n_series, n_series)], dim=1)
        corr_mask = torch.cat([corr_mask_1, corr_mask_2]).cuda().bool().unsqueeze(0).expand_as(similarity_matrix)
        left_mask = ~(mask | corr_mask)

        highest_similarity = similarity_matrix[corr_mask].view(Bn, 2*n_series, 1)
        second_highest_similarity = similarity_matrix[left_mask].view(Bn, 2*n_series, 2*n_series-2)
        diff =  second_highest_similarity - highest_similarity
        margin_loss = weight * torch.log(1 + torch.exp((diff / self.args.shufflerank_theta).clip(max=5.0))).mean()

        margin_logits = torch.cat([highest_similarity, second_highest_similarity], dim=2).view(-1, 2*n_series-1)
        margin_labels = torch.zeros(margin_logits.size(0)).long().cuda()

        # margin_loss = weight * F.relu(second_highest_similarity - highest_similarity).mean()
        # if self.cnt % 100 == 0:
        #     correct_rate = ((second_highest_similarity - highest_similarity) < 0)
        #     print(f"<<<<<<< margin_acc ")

        # sim_loss = weight * (1- highest_similarity).mean()

        ret = {
            f"{prefix}margin_logits": margin_logits,
            f"{prefix}margin_labels": margin_labels,
            f"{prefix}margin_contrast_loss": margin_loss
        }

        return ret

    def calc_tc_contrast_loss(self, features, prefix="tc_"):
        B, n_views, n_series, dim = features.size()
        assert n_series == self.n_series and dim == self.series_dim
        rank = 0
        world_size = 1
        if self.distributed:
            features = torch.cat(GatherLayer.apply(features), dim=0)
            rank = dist.get_rank()
            world_size = dist.get_world_size()

        # rank = 0
        N = features.size(0)
        N_per_rank = N // world_size
        i_base = N_per_rank * rank
        row_features = features.view(N, n_views, n_series, dim)[i_base:i_base+N_per_rank].permute(1,0,2,3).contiguous().view(n_views*N_per_rank, n_series, dim)
        col_features = features.view(N, n_views, n_series, dim).permute(1, 0, 2, 3).contiguous().view(n_views * N, n_series, dim)

        series_similarity_matrix = torch.matmul(row_features.unsqueeze(1), col_features.unsqueeze(0).transpose(3,2)).contiguous() # (2n, 2N, n_series, n_series)

        col_labels = torch.cat([torch.arange(N) for i in range(n_views)], dim=0)  # (2, N) -> (2*N,)
        row_labels = torch.cat([torch.arange(i_base, i_base+N_per_rank) for i in range(n_views)], dim=0)
        labels = (row_labels.unsqueeze(1) == col_labels.unsqueeze(0)).float()  # (2n, 2N)
        labels = labels.cuda()

        similarity_matrix = series_similarity_matrix.mean(dim=(2,3))
        similarity_matrix = similarity_matrix.contiguous()

        # discard the main diagonal from both: labels and similarities matrix
        row_inst = torch.arange(i_base, i_base+N_per_rank).unsqueeze(0).expand(n_views, N_per_rank)
        row_padded_ind = torch.arange(n_views).unsqueeze(1)*N
        row_inst = (row_inst + row_padded_ind).view(-1).unsqueeze(1)
        col_inst = torch.arange(N*n_views).unsqueeze(0)
        mask = (row_inst == col_inst).cuda()

        labels = labels[~mask].view(labels.shape[0], -1)  # (2n, 2N-1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)  # (2n, 2N-1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0]).long().cuda()

        logits = logits / self.aligned_T

        contrast_loss = self.criterion(logits, labels)

        ret = {
            f"{prefix}logits": logits,
            f"{prefix}labels": labels,
            f"{prefix}contrast_loss": contrast_loss
        }

        return ret

    def forward(self, block):
        '''
            modified from simCLR
            https://github.com/sthalles/SimCLR/blob/1848fc934ad844ae630e6c452300433fe99acfd9/simclr.py#L26
        '''
        block = block.contiguous()
        B = block.size(0)
        assert block.size(1) == 3
        extra_block = block[:, 2]
        (batch_size, _, C, T, H, W) = block.shape  # [B,N,C,T,H,W]
        n_views = 2
        N_views = 3
        assert n_views == 2
        x = block.view(-1, *(block.size()[2:])) # (B*n, ...)
        features = x
        for i, mod in enumerate(self.encoder_q):
            features = mod(features)
            if i == 1:
                backbone_features = features

        features = F.normalize(features, dim=1).squeeze().view(B, N_views, self.dim)
        features = features[:, :2].contiguous()
        ret = dict()
        if self.with_clip:
            ret.update(self.calc_contrast_loss(features, n_views))

        # get series projections
        series_features = self.series_proj_head(backbone_features).view(B, N_views, self.n_series, self.series_dim)
        series_features = F.normalize(series_features, dim=3)
        contrast_series_features = series_features[:, :n_views].contiguous()
        series_features = series_features.view(B * N_views, self.n_series, self.series_dim)
        if self.with_tc:
            ret.update(self.calc_tc_contrast_loss(
                contrast_series_features.view(B, n_views, self.n_series, self.series_dim)))

        if self.with_sr:
            orig_series_features = series_features.view(B, N_views, self.n_series, self.series_dim)[:,[0, 2]].contiguous()
            #### get time-series features and contrast them
            # shuffle input clips
            x = extra_block.view(B, C, self.n_series, T // self.n_series, H, W)
            sample_indices = torch.tensor(
                np.array([np.random.permutation(self.n_series) for i in range(B)])
            ).long().cuda()
            sample_gather_indices = sample_indices.view(B, 1, self.n_series, 1, 1, 1).expand_as(x)
            shuffled_x = torch.gather(x, 2, sample_gather_indices).contiguous().view(B, C, T, H, W)  # (B*n, C, T, H, W)
            shuffled_features = shuffled_x
            for i, mod in enumerate(self.encoder_q):
                if i > 1: break
                shuffled_features = mod(shuffled_features)
            shuffled_series_features = self.series_proj_head(shuffled_features).view(B, self.n_series, self.series_dim)
            sample_scatter_indices = sample_indices.view(B, self.n_series, 1).expand_as(shuffled_series_features)
            calibrated_shuffled_series_features = torch.scatter(
                shuffled_series_features, 1, sample_scatter_indices, shuffled_series_features)\
                .contiguous().view(B, self.n_series, self.series_dim)
            calibrated_shuffled_series_features = F.normalize(calibrated_shuffled_series_features, dim=2)
            # separated weighting the loss
            orig_shuffled_series_features = torch.stack([orig_series_features[:, 0], calibrated_shuffled_series_features], dim=2).contiguous() # (B, n_series, 2, dim)
            aug_shuffled_series_features = torch.stack([orig_series_features[:, 1], calibrated_shuffled_series_features], dim=2).contiguous() # (B, n_series, 2, dim)
            ret.update(self.calc_ranking_loss(orig_shuffled_series_features, 2,  'aug_ranking_', weight=0.5))
            ret.update(self.calc_ranking_loss(aug_shuffled_series_features, 2,  'unaug_ranking_', weight=0.5))

        return ret