import os
import sys
import glob
from io import BytesIO
import torch
from PIL import Image
import pandas as pd
from tqdm import tqdm
import random
import numpy as np
import math
import csv
import json
from pathlib import Path
import itertools

# naming convension:
# {}_2CLIP is for pretraining
# without 2CLIP is for action classification

__all__ = [
    'UCF101LMDB', 'UCF101_10CLIP', 'UCF101LMDB_2CLIP_Stage_Prototype',
    'HMDB51LMDB', 'HMDB51_10CLIP', 'HMDB51LMDB_2CLIP_Stage_Prototype',
    'K400LMDB', 'K400_10CLIP', 'K400LMDB_2CLIP_Stage_Prototype'
]

# rewrite for yourself:
lmdb_root = './data'


def read_file(path):
    with open(path, 'r') as f:
        content = f.readlines()
    content = [i.strip() for i in content]
    return content


def pil_from_raw_rgb(raw):
    return Image.open(BytesIO(raw)).convert('RGB')


def read_json(path):
    with open(path, 'r') as f:
        content = json.load(f)
    return content


def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)



#### UCF101 datasets
class UCF101LMDB_2CLIP(object):
    def __init__(self, root='%s/../process_data/data/ucf101' % os.path.dirname(os.path.abspath(__file__)),
                 db_path="data/UCF101/frame",
                 num_frames=16,
                 transform=None,
                 mode='val',
                 ds=1,
                 which_split=1,
                 return_path=False,
                 return_label=False,):

        self.num_readers = 1
        # self.reader = KVReader(db_path, self.num_readers)   
        self.root = root
        self.db_path = db_path
        self.transform = transform
        self.mode = mode
        self.num_frames = num_frames
        self.ds = ds
        self.which_split = which_split
        self.return_label = return_label
        self.return_path = return_path

        classes = read_file(os.path.join(root, 'ClassInd.txt'))
        if ',' in classes[0]: classes = [i.split(',')[-1].strip() for i in classes]
        print('Frame Dataset from "%s" has #class %d' % (root, len(classes)))

        self.num_class = len(classes)
        self.class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.idx_to_class = {i: classes[i] for i in range(len(classes))}

        print('Loading data from %s, split:%d' % (self.db_path, self.which_split))
        split_mode = mode
        if mode == 'test':
            video_info = pd.read_csv(os.path.join(root, '%s_split%02d.csv' % (split_mode, which_split)),
                                     header=None)  # first column
            video_info[2] = video_info[0].str.split('/').str.get(-3)  # class for each row, e.g. makeup
            video_info[3] = video_info[2] + '/' + video_info[0].str.split('/').str.get(
                -2)  # frame dir for each row, e.g. makeup/v_makeup_g01_c01
            assert len(pd.unique(video_info[2])) == self.num_class
        else:
            if mode == 'val': split_mode = 'train'
            video_info = pd.read_csv(os.path.join(root, '%s_split%02d.csv' % (split_mode, which_split)),
                                     header=None)  # first column
            video_info[2] = video_info[0].str.split('/').str.get(-3)  # class for each row, e.g. makeup
            video_info[3] = video_info[2] + '/' + video_info[0].str.split('/').str.get(
                -2)  # frame dir for each row, e.g. makeup/v_makeup_g01_c01
            val_split = video_info.sample(n=800, random_state=666)
            train_split = video_info.drop(val_split)
            video_info = train_split if mode == 'train' else val_split
            assert len(pd.unique(video_info[2])) == self.num_class

    def frame_sampler(self, total):
        if self.mode == 'test':  # half overlap - 1
            if total - self.num_frames * self.ds <= 0:  # pad left, only sample once
                sequence = np.arange(self.num_frames) * self.ds
                if random.randint(0, 1):  # pad left
                    seq_idx = np.zeros_like(sequence)
                    sequence = sequence[sequence < total]
                    seq_idx[-len(sequence)::] = sequence
                else:
                    seq_idx = np.ones_like(sequence) * (total - 1)
                    sequence = sequence[sequence < total]
                    seq_idx[:len(sequence)] = sequence
            else:
                available = total - self.num_frames * self.ds
                start = np.expand_dims(np.arange(0, available + 1, self.num_frames * self.ds // 2 - 1), 1)
                seq_idx = np.expand_dims(np.arange(self.num_frames) * self.ds, 0) + start  # [test_sample, num_frames]
                seq_idx = seq_idx.flatten()
        else:  # train or val
            if total - self.num_frames * self.ds <= 0:
                sequence = np.arange(self.num_frames) * self.ds + np.random.choice(range(self.ds), 1)
                if random.randint(0, 1): # pad left
                    seq_idx = np.zeros_like(sequence)
                    sequence = sequence[sequence < total]
                    seq_idx[-len(sequence)::] = sequence
                else: # pad right
                    seq_idx = np.ones_like(sequence)*(total-1)
                    sequence = sequence[sequence < total]
                    seq_idx[:len(sequence)] = sequence
            else:
                start = np.random.choice(range(total - self.num_frames * self.ds), 1)
                seq_idx = np.arange(self.num_frames) * self.ds + start
        return seq_idx

    def double_sampler(self, total):
        seq1 = self.frame_sampler(total)
        seq2 = self.frame_sampler(total)
        return np.concatenate([seq1, seq2])

    def __getitem__(self, index):
        vpath, vlen, vlabel, vname = self.video_subset.iloc[index]

        frame_index = self.double_sampler(vlen)
        keys = [f'{vname}/image_{i + 1:05d}.jpg' for i in frame_index]
        seq = [Image.open(os.path.join(self.db_path, key)) for key in keys]
        if self.transform is not None:
            seq = self.transform(seq)
        seq = torch.stack(seq, dim=1)

        ret = {'seq': seq}

        if self.return_label:
            vid = self.encode_action(vlabel)
            if self.return_path:
                ret['vpath'] = vpath
                ret['vid'] = vid
            else:
                ret['vid'] = vid
        return ret

    def __len__(self):
        return len(self.video_subset)

    def encode_action(self, action_name):
        return self.class_to_idx[action_name]

    def decode_action(self, action_code):
        return self.idx_to_class[action_code]


class UCF101LMDB(UCF101LMDB_2CLIP):
    def __init__(self, **kwargs):
        super(UCF101LMDB, self).__init__(**kwargs)

    def __getitem__(self, index):
        vpath, vlen, vlabel, vname = self.video_subset.iloc[index]

        frame_index = self.frame_sampler(vlen)

        keys = [f'{vname}/image_{i + 1:05d}.jpg' for i in frame_index]

        seq = [Image.open(os.path.join(self.db_path, key)) for key in keys]

        if self.transform is not None: seq = self.transform(seq)
        seq = torch.stack(seq, 1)

        ret = {'seq': seq,
               'vname': vname}
        if self.return_label:
            vid = self.encode_action(vlabel)
            if self.return_path:
                ret['vpath'] = vpath
                ret['vid'] = vid
            else:
                ret['vid'] = vid
        return ret


class UCF101LMDB_2CLIP_Protytype(UCF101LMDB_2CLIP):
    def __init__(self, **kwargs):
        super(UCF101LMDB_2CLIP_Protytype, self).__init__(**kwargs)

    def frame_sampler(self, total, center_lower=0, center_upper=0):
        if center_upper == 0:
            center_upper = total
        center_ind = np.random.randint(center_lower, center_upper)
        diff_seq = (np.arange(self.num_frames)-self.num_frames//2)*self.ds
        sequence = np.clip(diff_seq + center_ind, 0, total-1).astype(np.int32)
        return sequence

    def sample_prototype(self, total):
        return self.frame_sampler(total, 0, total)

    def __getitem__(self, index):
        vpath, vlen, vlabel, vname = self.video_subset.iloc[index]

        frame_index1 = self.sample_prototype(vlen)
        frame_index2 = self.sample_prototype(vlen)
        frame_index = np.concatenate((frame_index1, frame_index2))
        keys = [f'{vname}/image_{i + 1:05d}.jpg' for i in frame_index]
        try:
            seq = [Image.open(os.path.join(self.db_path, key)) for key in keys]
        except:
            raise Exception("Loading Error")
        if self.transform is not None:
            seq = self.transform(seq)
        seq = torch.stack(seq, dim=1)

        ret = {'seq': seq}

        if self.return_label:
            vid = self.encode_action(vlabel)
            if self.return_path:
                ret['vpath'] = vpath
                ret['vid'] = vid
            else:
                ret['vid'] = vid
        return ret


class UCF101LMDB_2CLIP_Stage_Prototype(UCF101LMDB_2CLIP_Protytype):
    def __init__(self, rand_flip=False, aug_series=True, **kwargs):
        super(UCF101LMDB_2CLIP_Stage_Prototype, self).__init__(**kwargs)
        self.aug_series = aug_series
        self.rand_flip = rand_flip

    def frame_sampler(self, total, center_lower=0, center_upper=0, repeat_prob=0.25, length=0):
        length = self.num_frames if length == 0 else length
        if center_upper == 0:
            center_upper = total
        center_ind = np.random.randint(center_lower, center_upper)
        diff_seq = (np.arange(length)-length//2)*self.ds
        if random.uniform(0., 1.) >= repeat_prob: # allow cross boundary
            center_lower = 0
        if random.uniform(0.,1.) >= repeat_prob: # allow cross boundary
            center_upper = total
        sequence = np.clip(diff_seq + center_ind, center_lower, center_upper-1).astype(np.int32)
        return sequence

    def sample_prototype(self, total):
        return self.frame_sampler(total, 0, total)

    def sample_prototype_split(self, total):
        return self.frame_sampler(total, 0, total, repeat_prob=1.0, length=self.num_frames) # do not allow cross boundary

    def __getitem__(self, index):
        vpath, vlen, vlabel, vname = self.video_subset.iloc[index]

        flip = 0
        if self.rand_flip: flip = random.randint(0,1)

        frame_index1 = self.sample_prototype(vlen)
        if flip:
            frame_index1 = frame_index1[::-1]
        frame_index2 = self.sample_prototype(vlen)
        if flip:
            frame_index2 = frame_index2[::-1]
        frame_index = np.concatenate((frame_index1, frame_index2))

        keys = [f'{vname}/image_{i + 1:05d}.jpg' for i in frame_index]
        try:
            seq = [Image.open(os.path.join(self.db_path, key)) for key in keys]
        except:
            raise Exception("Loading Error")

        if self.aug_series:
            seq = seq[:2*self.num_frames] + seq[:self.num_frames]

        if self.transform is not None:
            seq = self.transform(seq)
        seq = torch.stack(seq, dim=1)

        ret = {'seq': seq}

        if self.return_label:
            vid = self.encode_action(vlabel)
            if self.return_path:
                ret['vpath'] = vpath
                ret['vid'] = vid
                ret['vname'] = vname
            else:
                ret['vid'] = vid
        return ret


class UCF101_10CLIP(UCF101LMDB_2CLIP):
    def __init__(self, **kwargs):
        super(UCF101_10CLIP, self).__init__(**kwargs)

    def __getitem__(self, index):
        vpath, vlen, vlabel, vname = self.video_subset.iloc[index]

        frame_index = range(vlen)

        keys = [f'{vname}/image_{i + 1:05d}.jpg' for i in frame_index]

        seq = [Image.open(os.path.join(self.db_path, key)) for key in keys]

        new_seq_indices = []

        # assert vlen >= self.ds * (self.num_frames-1) + 1, vlen
        min_index = min(self.num_frames * self.ds // 2, vlen)
        max_index = max(min_index, vlen - self.num_frames * self.ds // 2)
        for clip_center in np.linspace(min_index, max_index, 10):
            clip_start = max(0, int(clip_center - self.num_frames * self.ds // 2))
            clip_indices = list(range(clip_start, clip_start + self.num_frames*self.ds, self.ds))
            clip_indices = [min(t, vlen-1) for t in clip_indices]
            new_seq_indices.extend(clip_indices)

        seq = [seq[i] for i in new_seq_indices]

        if self.transform is not None: seq = self.transform(seq)
        seq = torch.stack(seq, 1)

        ret = {'seq': seq,
               'vname': vname}

        if self.return_label:
            vid = self.encode_action(vlabel)
            if self.return_path:
                ret['vpath'] = vpath
                ret['vid'] = vid
            else:
                ret['vid'] = vid
        return ret


#### HMDB51 dataset
class HMDB51LMDB(UCF101LMDB):
    def __init__(self, root='%s/../process_data/data/hmdb51' % os.path.dirname(os.path.abspath(__file__)),
                 db_path="data/HMDB51/frame",
                 **kwargs):
        super(HMDB51LMDB, self).__init__(root=root, db_path=db_path, **kwargs)


class HMDB51_10CLIP(UCF101_10CLIP):
    def __init__(self, root='%s/../process_data/data/hmdb51' % os.path.dirname(os.path.abspath(__file__)),
                 db_path="data/HMDB51/frame",
                 **kwargs):
        super(HMDB51_10CLIP, self).__init__(root=root, db_path=db_path, **kwargs)


class HMDB51LMDB_2CLIP_Stage_Prototype(UCF101LMDB_2CLIP_Stage_Prototype):
    def __init__(self, root='%s/../process_data/data/hmdb51' % os.path.dirname(os.path.abspath(__file__)),
                 db_path="data/HMDB51/frame",
                 **kwargs):
        super(HMDB51LMDB_2CLIP_Stage_Prototype, self).__init__(root=root, db_path=db_path, **kwargs)


#### Kinetics400 dataset
class K400LMDB(UCF101LMDB):
    def __init__(self, root='%s/../process_data/data/k400' % os.path.dirname(os.path.abspath(__file__)),
                 db_path="data/K400/frame",
                 **kwargs):
        super(K400LMDB, self).__init__(root=root, db_path=db_path, **kwargs)


class K400_10CLIP(UCF101_10CLIP):
    def __init__(self, root='%s/../process_data/data/k400' % os.path.dirname(os.path.abspath(__file__)),
                 db_path="data/K400/frame",
                 **kwargs):
        super(K400_10CLIP, self).__init__(root=root, db_path=db_path, **kwargs)


class K400LMDB_2CLIP_Stage_Prototype(UCF101LMDB_2CLIP_Stage_Prototype):
    def __init__(self, root='%s/../process_data/data/k400' % os.path.dirname(os.path.abspath(__file__)),
                 db_path="data/K400/frame",
                 **kwargs):
        super(K400LMDB_2CLIP_Stage_Prototype, self).__init__(root=root, db_path=db_path, **kwargs)




if __name__ == '__main__':
    # dataset = UCF101LMDB(root='%s/../process_data/data/ucf101' % os.path.dirname(os.path.abspath(__file__)),
    #              db_path=os.path.join(lmdb_root, 'UCF101/frame'),
    #              transform=None, mode='val',
    #              num_frames=32, ds=1, which_split=1,
    #              window=False,
    #              return_path=False,
    #              return_label=False,
    #              return_source=False)

    pass


