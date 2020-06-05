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

activity2id_hard = {
    "BG": 0,  # background
    "activity_gesturing": 1,
    "Closing": 2,
    "Opening": 3,
    "Interacts": 4,
    "Exiting": 5,
    "Entering": 6,
    "Talking": 7,
    "Transport_HeavyCarry": 8,
    "Unloading": 9,
    "Pull": 10,
    "Loading": 11,
    "Open_Trunk": 12,
    "Closing_Trunk": 13,
    "Riding": 14,
    "specialized_texting_phone": 15,
    "Person_Person_Interaction": 16,
    "specialized_talking_phone": 17,
    "activity_running": 18,
    "PickUp": 19,
    "specialized_using_tool": 20,
    "SetDown": 21,
    "activity_crouching": 22,
    "activity_sitting": 23,
    "Object_Transfer": 24,
    "Push": 25,
    "PickUp_Person_Vehicle": 26
    }
    

activity2id = {
    "BG": 0,  # background
    "activity_walking": 1,
    "activity_standing": 2,
    "activity_carrying": 3,
    "activity_gesturing": 4,
    "Closing": 5,
    "Opening": 6,
    "Interacts": 7,
    "Exiting": 8,
    "Entering": 9,
    "Talking": 10,
    "Transport_HeavyCarry": 11,
    "Unloading": 12,
    "Pull": 13,
    "Loading": 14,
    "Open_Trunk": 15,
    "Closing_Trunk": 16,
    "Riding": 17,
    "specialized_texting_phone": 18,
    "Person_Person_Interaction": 19,
    "specialized_talking_phone": 20,
    "activity_running": 21,
    "PickUp": 22,
    "specialized_using_tool": 23,
    "SetDown": 24,
    "activity_crouching": 25,
    "activity_sitting": 26,
    "Object_Transfer": 27,
    "Push": 28,
    "PickUp_Person_Vehicle": 29,
    "vehicle_turning_right": 30,
    "vehicle_moving": 31,
    "vehicle_stopping" : 32,
    "vehicle_starting" :33,
    "vehicle_turning_left": 34,
    "vehicle_u_turn": 35,
    "specialized_miscellaneous": 36,
    "DropOff_Person_Vehicle" : 37,
    "Misc" : 38,
    "Drop" : 39}
CLASSES = ('BG','activity_walking','activity_standing',
    'activity_carrying','activity_gesturing','Closing',
    'Opening','Interacts','Exiting','Entering','Talking',
    'Transport_HeavyCarry','Unloading','Pull','Loading',
    'Open_Trunk','Closing_Trunk','Riding','specialized_texting_phone',
    'Person_Person_Interaction','specialized_talking_phone',
    'activity_running','PickUp','specialized_using_tool',
    'SetDown','activity_crouching','activity_sitting',
    'Object_Transfer','Push','PickUp_Person_Vehicle',
    'vehicle_turning_right','vehicle_moving','vehicle_stopping',
    'vehicle_starting','vehicle_turning_left','vehicle_u_turn',
    'specialized_miscellaneous','DropOff_Person_Vehicle','Misc',
    'Drop')

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
        # filter images with no annotation during training
        '''if not test_mode:
            valid_inds = self._filter_imgs()
            self.img_infos = [self.img_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]'''

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

    def get(self,index,record, indices):
      
      sequence_path = str(record.path).strip().split('/frames/')[0]
      label = list()
      bbox = list()
      images = list()
      img_path = list()
      gt = np.zeros((len(indices),self.cfg.MAX_NUM_GT_BOXES,(self.num_class + 4)),
                  dtype=np.float32)
      num_boxes = np.zeros((self.num_segments),dtype=np.float32)
      im_info = np.zeros((self.num_segments,3),dtype=np.float32)
      npy_file = (os.path.join(str(sequence_path),'ground_truth.npy'))
      data = np.load(npy_file)
      frame = data[0][0]
      j =0 
      for seg_ind in indices: #iterate through every image
                    count = 0
                    
                    bboxes = np.zeros((self.cfg.MAX_NUM_GT_BOXES,(self.num_class + 4)),dtype= float)
                    p = int(seg_ind) + int(frame)
                    image_path = os.path.join(record.path, '{:06d}.jpg'.format(p))
                    im = imread(image_path)
                    im = im[:,:,::-1]
                    im = im.astype(np.float32, copy=False)
                    height,width,_= im.shape #h=1080,w=1920
                    im_size_min= min(height,width)
                    im_size_max = max(height,width)
                    im_scale = float(self.new_size) / float(im_size_min)
                    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
                    im_info[j,:]=self.new_size,len(im[2]),im_scale
                    img_path.append(image_path)
                    for i in data:
                        if i[0] == p:
                            bbox_new =[]
                            bbox = (i[2:6])*im_scale
                            bbox_new[0:4] = bbox
                            #change to train only hard class
                            #bbox_new[4:] = i[6:7]
                            #bbox_new[5:] = i[10:10+self.num_class -1]
                            bbox_new[4:]=i[6:6+self.num_class]#change here to train only less hard class
                            bboxes[count,:]+=bbox_new
                            count+=1
                            
                    num_boxes[j,]+=count
                    gt[j,:,:] = bboxes
                    j = j+1     
                    images.append(im)
      
      max_shape = np.array([imz.shape for imz in images]).max(axis=0)
      num_images = len(images)
      blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
      for i in range(len(images)):
        im1 = images[i]
        blob[i,0:im1.shape[0], 0:im1.shape[1], :] = im1


      process_data = self.transform(blob)
      return process_data, gt, num_boxes , im_info ,img_path

                          
    '''def __getitem__(self, index):
        record = self.video_list[index]
        #self.yaml_file(index)
        segment_indices = self._sample_indices(record)
        segment_indices = np.sort(segment_indices)
        return self.get( index, record, segment_indices)'''
    
               
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
                    gt_bboxes.append(list(gt_data[dat][2:6]))
                    gt_labels.append(list(gt_data[dat][6:]))
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



#parse the yml file into the variables


