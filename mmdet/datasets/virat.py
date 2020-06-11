import torch.utils.data as data


import os
from os import listdir
from os.path import isfile, join

from PIL import Image
import numpy as np
from numpy.random import randint
import operator
import torch
import cv2
from cv2 import imread


import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC

#from .builder import DATASETS
from .custom import CustomDataset
from .transforms import (ImageTransform, BboxTransform, MaskTransform,
                         Numpy2Tensor)
from .utils import to_tensor, random_scale

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])
    
    @property
    def height(self):
        return int(self._data[2])
    
    @property
    def width(self):
        return int(self._data[3])

#@DATASETS.register_module()
class VIRAT_dataset(CustomDataset):
    def __init__(self, ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 multiscale_mode='value',
                 size_divisor=None,
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0,
                 with_mask=True,
                 with_crowd=True,
                 with_label=True,
                 extra_aug=None,
                 resize_keep_ratio=True,
                 test_mode=False,
                 dense_sample=False,uniform_sample=True,random_sample=False,
                 strided_sample=False,num_segments = 8):
         
        self.train_path = img_prefix
        self.list_file = ann_file
        self.num_segments = num_segments

        self._parse_list()
        
        
        # with label is False for RPN
        self.with_label = with_label
        # in test mode or not
        self.test_mode = test_mode

        if proposal_file is not None:
            self.proposals = self.load_proposals(proposal_file)
        else:
            self.proposals = None
        
        # (long_edge, short_edge) or [(long1, short1), (long2, short2), ...]
        self.img_scales = img_scale if isinstance(img_scale,
                                                  list) else [img_scale]
        assert mmcv.is_list_of(self.img_scales, tuple)
        # normalization configs
        self.img_norm_cfg = img_norm_cfg

        # multi-scale mode (only applicable for multi-scale training)
        self.multiscale_mode = multiscale_mode
        assert multiscale_mode in ['value', 'range']
     
        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

        # max proposals per image
        self.num_max_proposals = num_max_proposals
        # flip ratio
        self.flip_ratio = flip_ratio
        assert flip_ratio >= 0 and flip_ratio <= 1
        # padding border to ensure the image size can be divided by
        # size_divisor (used for FPN)
        self.size_divisor = size_divisor
        
        self.dense_sample = dense_sample
        self.uniform_sample = uniform_sample
        self.strided_sample = strided_sample
        self.random_sample = random_sample
        
        self.img_transform = ImageTransform(
            size_divisor=self.size_divisor, **self.img_norm_cfg)
        self.bbox_transform = BboxTransform()
        self.numpy2tensor = Numpy2Tensor()

        # image rescale if keep ratio
        self.resize_keep_ratio = resize_keep_ratio

        #validation ann info
        self.val_frame_ind = []
        self.val_gt_path = []
        

    def _parse_list(self):
        frame_path = [x.strip().split(' ') for x in open(self.list_file)]  
        self.video_list = [VideoRecord(item) for item in frame_path]
        print('Sequence number/ video number:%d' % (len(self.video_list)))
    
    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            width =self.video_list[i].width
            height = self.video_list[i].height
            if width/ height > 1:
                   self.flag[i] = 1
    
    def _sample_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets)
        elif self.uniform_sample:  # normal sample
            average_duration = (record.num_frames) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                       size=self.num_segments)
            return offsets 
        elif self.random_sample:
            offsets = np.sort(randint(record.num_frames + 1, size=self.num_segments))
            return offsets 
        elif self.strided_sample:
            average_duration = (record.num_frames) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + average_duration//2
            return offsets
        else:
            offsets = np.zeros((self.num_segments,))    
            return offsets  
    CLASSES = ('activity_walking','activity_standing',
                    'activity_carrying','activity_gesturing','Closing',
                    'Opening','Interacts','Exiting','Entering','Talking',
                    'Transport_HeavyCarry','Unloading','Pull','Loading',
                    'Open_Trunk','Closing_Trunk','Riding','specialized_texting_phone',
                    'Person_Person_Interaction','specialized_talking_phone',
                    'activity_running','PickUp','specialized_using_tool',
                    'SetDown','activity_crouching','activity_sitting',
                    'Object_Transfer','Push','PickUp_Person_Vehicle',
                    'BG')
    def __len__(self):
        return (len(self.video_list))



    def prepare_train_img(self, idx):
        record = self.video_list[idx]
        segment_indices = self._sample_indices(record)
        segment_indices = np.sort(segment_indices)
        #img_info = self.img_infos[idx]

        #load frame path and ann_file for the sequence
        frame = ((record.path).split('/')[8]).split('_')[0]
        sequence_path = str(record.path).strip().split('/frames/')[0]
        npy_file = (os.path.join(str(sequence_path),'ground_truth.npy'))
        gt_data = np.load(npy_file)
        fin_data = []
        
        for i in segment_indices:
            p = int(i) + int(frame)
            
            # load image
            img = mmcv.imread(os.path.join(record.path,'{:06d}.jpg'.format(p)))
            #img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
            ori_shape = (img.shape[0], img.shape[1], 3)
            #ori_shape = (img_info['height'], img_info['width'], 3)
            # load proposals if necessary
            if self.proposals is not None:
                proposals = self.proposals[idx][:self.num_max_proposals]
                # TODO: Handle empty proposals properly. Currently images with
                # no proposals are just ignored, but they can be used for
                # training in concept.
                if len(proposals) == 0:
                    return None
                if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
                    raise AssertionError(
                     'proposals should have shapes (n, 4) or (n, 5), '
                     'but found {}'.format(proposals.shape))
                if proposals.shape[1] == 5:
                    scores = proposals[:, 4, None]
                    proposals = proposals[:, :4]
                else:
                    scores = None

            #ann = self.get_ann_info(idx)
            #gt_bboxes = ann['bboxes']
            #gt_labels = ann['labels']
            gt_bboxes = []
            gt_labels =[]

            for dat in range(len(gt_data)):
                if gt_data[dat][0] == p:
                    #append last column with BG labels
                    gt_bboxes.append(list(gt_data[dat][2:6]))
                    gt_labels.append(list(np.append(gt_data[dat][7:7+len(self.CLASSES)-1],gt_data[dat][6:7])))
            gt_bboxes = np.asarray(gt_bboxes,dtype=np.float32)
            gt_labels = np.asarray(gt_labels,dtype=np.int64)
            #gt_bboxes = [data[i][2:6] for i in range(len(data)) if data[i][0] == p]
            #gt_labels = [data[i][6:] for i in range(len(data)) if data[i][0] == p]
            # skip the image if there is no valid gt bbox
            if len(gt_bboxes) == 0:
               return None

            # extra augmentation
            '''if self.extra_aug is not None:
                img, gt_bboxes, gt_labels = self.extra_aug(img, gt_bboxes,
                                                       gt_labels)
            '''
            # apply transforms
            flip = True if np.random.rand() < self.flip_ratio else False
            # randomly sample a scale
            img_scale = random_scale(self.img_scales, self.multiscale_mode)
            '''if self.extra_aug is not None:
                if 'RandomResizeCrop' in [x.__class__.__name__ for x in self.extra_aug.transforms]:
                    img_scale = img.shape[:2]
            '''
            img, img_shape, pad_shape, scale_factor = self.img_transform(
                 img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
            img = img.copy()
            if self.proposals is not None:
                proposals = self.bbox_transform(proposals, img_shape, scale_factor,
                                            flip)
                proposals = np.hstack(
                [proposals, scores]) if scores is not None else proposals
            gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor,
                                        flip)
            
            img_meta = dict(
               ori_shape=ori_shape,
               img_shape=img_shape,
               pad_shape=pad_shape,
               scale_factor=scale_factor,
               flip=flip)

            data = dict(
                img=DC(to_tensor(img), stack=True),
                img_meta=DC(img_meta, cpu_only=True),
                gt_bboxes=DC(to_tensor(gt_bboxes)))
            if self.proposals is not None:
                data['proposals'] = DC(to_tensor(proposals))
            if self.with_label:
                data['gt_labels'] = DC(to_tensor(gt_labels))
            fin_data.append(data)
        return fin_data

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        record = self.video_list[idx]
        segment_indices = self._sample_indices(record)
        segment_indices = np.sort(segment_indices)
        
        frame = ((record.path).split('/')[8]).split('_')[0]
        sequence_path = str(record.path).strip().split('/frames/')[0]

        #create run time info for annotation
        npy_file = (os.path.join(str(sequence_path),'ground_truth.npy'))
        self.val_gt_path.append(npy_file)
        self.val_frame_ind.append(segment_indices)

        fin_data = []
        
        for i in segment_indices:
            p = int(i) + int(frame)
        
            #img_info = self.img_infos[idx]
            # load image
            img = mmcv.imread(os.path.join(record.path,'{:06d}.jpg'.format(p)))
            #img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
            if self.proposals is not None:
               proposal = self.proposals[idx][:self.num_max_proposals]
               if not (proposal.shape[1] == 4 or proposal.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposal.shape))
            else:
               proposal = None

            def prepare_single(img, scale, flip, proposal=None):
              _img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, scale, flip, keep_ratio=self.resize_keep_ratio)
              _img = to_tensor(_img)
              _img_meta = dict(
                ori_shape=(img.shape[0], img.shape[1], 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip)
              if proposal is not None:
                if proposal.shape[1] == 5:
                    score = proposal[:, 4, None]
                    proposal = proposal[:, :4]
                else:
                    score = None
                _proposal = self.bbox_transform(proposal, img_shape,
                                                scale_factor, flip)
                _proposal = np.hstack(
                    [_proposal, score]) if score is not None else _proposal
                _proposal = to_tensor(_proposal)
              else:
                _proposal = None
              return _img, _img_meta, _proposal

            imgs = []
            img_metas = []
            proposals = []
            for scale in self.img_scales:
                _img, _img_meta, _proposal = prepare_single(
                   img, scale, False, proposal)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
                proposals.append(_proposal)
                if self.flip_ratio > 0:
                    _img, _img_meta, _proposal = prepare_single(
                       img, scale, True, proposal)
                    imgs.append(_img)
                    img_metas.append(DC(_img_meta, cpu_only=True))
                    proposals.append(_proposal)
            data = dict(img=imgs, img_meta=img_metas)
            if self.proposals is not None:
                data['proposals'] = proposals
            fin_data. append(data)
        return fin_data

    def get_ann_info(self,idx):
        val_data = np.load(self.val_gt_path[idx])
        frame_idx_list = self.val_frame_ind[idx]
        gt_bboxes = []
        gt_labels = []
        for frame in frame_idx_list:
            p = int(val_data[0][0]) + int(frame)
            for dat in range(len(val_data)):
                if val_data[dat][0] == p:
                    gt_bboxes.append(list(val_data[dat][2:6]))
                    gt_labels.append(list(np.append(val_data[dat][7:7+len(self.CLASSES)-1],val_data[dat][6:7])))
        assert len(gt_bboxes) == len(gt_labels)
        gt_bboxes = np.asarray(gt_bboxes,dtype=np.float32)
        gt_labels = np.asarray(gt_labels,dtype=np.int64)
        return gt_bboxes,gt_labels
        

