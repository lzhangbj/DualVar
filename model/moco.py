# MoCo-related code is modified from https://github.com/facebookresearch/moco
import sys
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../')
from backbone.select_backbone import select_backbone

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class MoCo_Naked(nn.Module):
    '''
    Basically, it's a MoCo for video input: https://arxiv.org/abs/1911.05722
    '''

    def __init__(self, network='s3d', dim=128, K=2048, m=0.999, T=0.07, distributed=True, nonlinear=True):
        '''
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 2048)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        '''
        super(MoCo_Naked, self).__init__()

        self.dim = dim
        self.K = K
        self.m = m
        self.T = T
        self.distributed = distributed
        self.nonlinear = nonlinear

        ##########################################
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

        backbone, _ = select_backbone(network)
        self.encoder_k = nn.ModuleList([
            backbone,
            nn.AdaptiveAvgPool3d((1, 1, 1))])
        if nonlinear:
            self.encoder_k.extend([
                nn.Conv3d(feature_size, feature_size, kernel_size=1, bias=True),
                nn.ReLU(),
                nn.Conv3d(feature_size, dim, kernel_size=1, bias=True)
            ])

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.criterion = nn.CrossEntropyLoss()
        # Notes: for handling sibling videos, e.g. for UCF101 dataset

    def _build_nonlinear_layer(self, input_size, output_size, ds_num=0):
        mod_list = [
            nn.Conv3d(input_size, input_size, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv3d(input_size, output_size, kernel_size=1, bias=True)
        ]

        mod_list = [
                       nn.Conv3d(input_size, input_size, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1), bias=True),
                       nn.BatchNorm3d(input_size),
                       nn.ReLU()
                   ] * ds_num + mod_list

        return nn.Sequential(
            *mod_list
        )

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        '''Momentum update of the key encoder'''
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        if self.distributed:
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        # get ptr
        ptr = int(self.queue_ptr) # global ptr
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T

        # update ptr
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        '''
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        '''
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        '''
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        '''
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, block):
        '''Output: logits, targets'''
        (B, N, *_) = block.shape  # [B,N,C,T,H,W]
        assert N == 2
        x1 = block[:, 0, :].contiguous()
        x2 = block[:, 1, :].contiguous()

        # compute query features
        unnorm_q = x1
        for i, mod in enumerate(self.encoder_q):
            if i == 0:
                unnorm_q = mod(unnorm_q)
                backbone_feat_q = F.adaptive_avg_pool3d(unnorm_q, (1,1,1))
            else:
                unnorm_q = mod(unnorm_q)
        q = nn.functional.normalize(unnorm_q, dim=1)
        q = q.view(B, self.dim)

        ###########################################
        # calculate normalized sampled frame embeds and possibly video lvl embeds
        in_train_mode = q.requires_grad

        # compute key features
        with torch.no_grad():  # no gradient to keys
            if in_train_mode: self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            if self.distributed:
                x2, idx_unshuffle = self._batch_shuffle_ddp(x2)

            unnorm_k = x2
            for i, mod in enumerate(self.encoder_k):
                if i == 0:
                    unnorm_k = mod(unnorm_k)
                    backbone_feat_k = F.adaptive_avg_pool3d(unnorm_k, (1,1,1))
                else:
                    unnorm_k = mod(unnorm_k)
            k = nn.functional.normalize(unnorm_k, dim=1)

            ###########################################
            # calculate normalized sampled frame embeds and possibly video lvl embeds
            # undo shuffle
            if self.distributed:
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        k = k.view(B, self.dim)

        # shared labels: positive key indicators
        labels = torch.zeros(B, dtype=torch.long).cuda()

        #### compute video-video contrastive loss
        vid_to_vid_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        vid_to_vid_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        vid_to_vid_logits = torch.cat([vid_to_vid_pos, vid_to_vid_neg], dim=1) / self.T # logits: B,(1+K)
        vid_to_vid_contrast_loss = self.criterion(vid_to_vid_logits, labels)
        ret = {
            'clip_logits': vid_to_vid_logits,
            'clip_labels': labels,
            'clip_contrast_loss': vid_to_vid_contrast_loss,
        }

        # dequeue and enqueue
        if in_train_mode: self._dequeue_and_enqueue(k)

        return ret


class MoCo_TimeSeriesV4(nn.Module):
    '''
    Basically, it's a MoCo for video input: https://arxiv.org/abs/1911.05722
    '''

    def __init__(self, network='s3d', dim=128, K=2048, m=0.999, T=0.07, distributed=True, nonlinear=True,
                 n_series=2, series_dim=64, series_T=0.07, aligned_T=0.07, mode="clip-sr-tc", args=None):
        '''
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 2048)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        '''
        super(MoCo_TimeSeriesV4, self).__init__()

        self.dim = dim
        self.K = K
        self.m = m
        self.T = T
        self.distributed = distributed
        self.nonlinear = nonlinear
        self.n_series = n_series
        self.series_dim = series_dim
        self.mode = mode
        self.series_T = series_T
        self.aligned_T = aligned_T
        self.with_clip = 'clip' in mode
        self.with_sr = 'sr' in mode
        self.with_tc = 'tc' in mode

        # assert mode in (
        # 'bi-discrete-dtw', 'bi-diff-dtw', 'bi-diff-dtw-cuda', 'non-aligned', 'bimin-diff-dtw', 'bimax-diff-dtw')

        ##########################################
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
        self.series_proj_head_q = nn.Sequential(
            nn.Conv3d(feature_size, feature_size, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv3d(feature_size, series_dim*n_series, kernel_size=1, bias=True)
        )

        backbone, _ = select_backbone(network)
        self.encoder_k = nn.ModuleList([
            backbone,
            nn.AdaptiveAvgPool3d((1, 1, 1))])
        if nonlinear:
            self.encoder_k.extend([
                nn.Conv3d(feature_size, feature_size, kernel_size=1, bias=True),
                nn.ReLU(),
                nn.Conv3d(feature_size, dim, kernel_size=1, bias=True)
            ])
        self.series_proj_head_k = nn.Sequential(
            nn.Conv3d(feature_size, feature_size, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv3d(feature_size, series_dim * n_series, kernel_size=1, bias=True)
        )

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        for param_q, param_k in zip(self.series_proj_head_q.parameters(), self.series_proj_head_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        # series queue
        self.register_buffer("series_queue", torch.randn(series_dim*n_series, K))
        self.series_queue = nn.functional.normalize(self.series_queue.view(n_series, series_dim, K), dim=1).view(n_series*series_dim, K)

        self.criterion = nn.CrossEntropyLoss()
        # Notes: for handling sibling videos, e.g. for UCF101 dataset

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        '''Momentum update of the key encoder'''
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.series_proj_head_q.parameters(), self.series_proj_head_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, series_keys):
        # gather keys before updating queue
        if self.distributed:
            keys = concat_all_gather(keys)
            series_keys = concat_all_gather(series_keys)

        batch_size = keys.shape[0]

        # get ptr
        ptr = int(self.queue_ptr) # global ptr
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.series_queue[:, ptr:ptr + batch_size] = series_keys.T

        # update ptr
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        '''
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        '''
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        '''
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        '''
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def calc_tc_contrast_loss(self, q, k, queue, prefix="tc_"):
        B, n_series, dim = q.size()
        assert n_series == self.n_series and dim == self.series_dim
        queue = queue.clone().detach()

        base = q.view(B, n_series, dim)
        pos  = k.view(B, n_series, dim)
        neg  = queue.T.contiguous().view(self.K, n_series, dim)

        vid_to_vid_pos = torch.matmul(base, pos.transpose(2,1).contiguous()).mean(dim=(1,2)).unsqueeze(1)
        vid_to_vid_neg = torch.matmul(base.unsqueeze(1), neg.transpose(2,1).contiguous()).mean(dim=(2,3))

        labels = torch.zeros(q.size(0), dtype=torch.long).cuda()
        vid_to_vid_logits = torch.cat([vid_to_vid_pos, vid_to_vid_neg], dim=1) / self.aligned_T # logits: B,(1+K)
        vid_to_vid_contrast_loss = self.criterion(vid_to_vid_logits, labels)
        ret = {
            f'{prefix}logits': vid_to_vid_logits,
            f'{prefix}labels': labels,
            f'{prefix}contrast_loss': vid_to_vid_contrast_loss,
        }
        return ret

    def calc_clip_contrast_loss(self, q, k, queue, prefix='clip_'):
        labels = torch.zeros(q.size(0), dtype=torch.long).cuda()

        vid_to_vid_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        vid_to_vid_neg = torch.einsum('nc,ck->nk', [q, queue.clone().detach()])
        vid_to_vid_logits = torch.cat([vid_to_vid_pos, vid_to_vid_neg], dim=1) / self.T # logits: B,(1+K)
        vid_to_vid_contrast_loss = self.criterion(vid_to_vid_logits, labels)
        ret = {
            f'{prefix}logits': vid_to_vid_logits,
            f'{prefix}labels': labels,
            f'{prefix}contrast_loss': vid_to_vid_contrast_loss,
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
        margin_loss = weight * torch.log(1 + torch.exp(diff / 0.05)).mean()

        margin_logits = torch.cat([highest_similarity, second_highest_similarity], dim=2).view(-1, 2*n_series-1)
        margin_labels = torch.zeros(margin_logits.size(0)).long().cuda()

        ret = {
            f"{prefix}margin_logits": margin_logits,
            f"{prefix}margin_labels": margin_labels,
            f"{prefix}margin_contrast_loss": margin_loss
        }

        return ret

    def forward(self, block):
        '''Output: logits, targets'''
        ret  = {}
        B, N, C, T, H, W = block.shape  # [B,N,C,T,H,W]
        assert N == 3
        x1 = block[:, 0, :].contiguous()
        x2 = block[:, 1, :].contiguous()
        aug_x1 = block[:, 2, :].contiguous()

        # compute query features
        unnorm_q = x1
        for i, mod in enumerate(self.encoder_q):
            unnorm_q = mod(unnorm_q)
            if i == 1:
                backbone_feat_q = unnorm_q

        q = nn.functional.normalize(unnorm_q, dim=1)
        q = q.view(B, self.dim)
        # get series projections of x1
        series_features = self.series_proj_head_q(backbone_feat_q).view(B, self.n_series, self.series_dim)
        series_features = F.normalize(series_features, dim=2)

        in_train_mode = q.requires_grad

        # compute key features
        with torch.no_grad():  # no gradient to keys
            if in_train_mode: self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            if self.distributed:
                x2, idx_unshuffle = self._batch_shuffle_ddp(x2)

            unnorm_k = x2
            for i, mod in enumerate(self.encoder_k):
                unnorm_k = mod(unnorm_k)
                if i == 1:
                    backbone_feat_k = unnorm_k
            k = nn.functional.normalize(unnorm_k, dim=1)
            series_features_k = self.series_proj_head_k(backbone_feat_k).view(B, self.n_series, self.series_dim)
            series_features_k = F.normalize(series_features_k, dim=2).view(B, self.n_series*self.series_dim)

            if self.distributed:
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)
                series_features_k = self._batch_unshuffle_ddp(series_features_k, idx_unshuffle)

        k = k.view(B, self.dim)

        #### compute video-video contrastive loss
        ret.update(self.calc_contrast_loss(q, k, self.queue, 'clip_'))

        ### compute aligned series contrastive loss
        series_features = series_features.view(B, self.n_series, self.series_dim)
        series_features_k = series_features_k.view(B, self.n_series, self.series_dim)
        if self.with_tc:
            ret.update(self.calc_tc_contrast_loss(series_features, series_features_k, self.series_queue, 'tc_'))

        # dequeue and enqueue
        if in_train_mode: self._dequeue_and_enqueue(k, series_features_k.view(B, self.n_series*self.series_dim))

        # series-series loss is only calculated on shuffled x1 and aug-x1
        # calc shuffled series_features for x1 and x2
        aug_x1 = aug_x1.view(B, C, self.n_series, T // self.n_series, H, W)
        sample_indices = torch.tensor(
            np.array([np.random.permutation(self.n_series) for i in range(B)])
        ).long().cuda()
        sample_gather_indices = sample_indices.view(B, 1, self.n_series, 1, 1, 1).expand_as(aug_x1)
        shuffled_aug_x1 = torch.gather(aug_x1, 2, sample_gather_indices).contiguous().view(B, C, T, H, W)  # (B*n, C, T, H, W)
        aug_x1 = aug_x1.view(B, C, T, H, W)

        dual_aug_x1 = torch.cat([aug_x1, shuffled_aug_x1], dim=0)
        dual_aug_x1_feat = dual_aug_x1
        for i, mod in enumerate(self.encoder_q):
            dual_aug_x1_feat = mod(dual_aug_x1_feat)
            if i == 1:
                break
        dual_aug_x1_series_feat = F.normalize(self.series_proj_head_q(dual_aug_x1_feat).view(B*2, self.n_series, self.series_dim), dim=2)

        aug_x1_series_features = dual_aug_x1_series_feat[:B]

        shuffled_aug_x1_series_feat = dual_aug_x1_series_feat[B:]
        sample_scatter_indices = sample_indices.view(B , self.n_series, 1).expand_as(shuffled_aug_x1_series_feat)
        calibrated_shuffled_aug_x1_series_features = torch.scatter(
            shuffled_aug_x1_series_feat, 1, sample_scatter_indices, shuffled_aug_x1_series_feat) \
            .contiguous().view(B, self.n_series, self.series_dim)

        orig_shuffled_series_features = torch.stack([series_features, calibrated_shuffled_aug_x1_series_features], dim=2)
        aug_shuffled_series_features = torch.stack([aug_x1_series_features, calibrated_shuffled_aug_x1_series_features], dim=2)

        ret.update(self.calc_ranking_loss(orig_shuffled_series_features, 2, 'unaug_ranking_', weight=0.5))
        ret.update(self.calc_ranking_loss(aug_shuffled_series_features, 2, 'aug_ranking_', weight=0.5))

        return ret


