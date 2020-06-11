# model settings
model = dict(
    type='FasterRCNN',
    pretrained='hrnetv2_pretrained/hrnetv2_w18_imagenet_pretrained.pth',
    backbone=dict(
        type='HighResolutionNet',
        frozen_stages = 1,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4,),
                num_channels=(64,),
                fuse_method='SUM'),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(18, 36),
                fuse_method='SUM'),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(18, 36, 72),
                fuse_method='SUM'),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(18, 36, 72, 144),
                fuse_method='SUM'),
            shift=dict(
                num_segments = 8,
                shift_div = 8,
                shift_place = 'blockres',
                is_shift = 'True'))),
    neck=dict(
        type='HRFPN',
        in_channels=[18, 36, 72, 144],
        out_channels=256),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        use_sigmoid_cls=True),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='SharedFCBBoxHead',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=30,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        smoothl1_beta=1 / 9.0,
        debug=False),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=15)
    # soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
)
# NOTE:
# dataset settings
# if you use zip format to store all images of coco, please use CocoZipDataset
#dataset_type = 'CocoDataset'
#data_root = '/mnt/AI_RAID/coco_data/'
dataset_type = 'VIRAT_dataset'
data_root = '/mnt/AI_RAID/VIRAT/actev-data-repo/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=False)
# else
# dataset_type = 'CocoDataset'
# data_root = 'data/coco/'
# img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=0,
    num_segments = 8,
    train=dict(
        type=dataset_type,
        #ann_file=data_root + 'annotations/instances_train2017.json',
        #img_prefix=data_root + 'train2017',
        ann_file=data_root + 'mod_frame_list_hrnet/train_list.txt',
        img_prefix=data_root + 'dataset/train',
        img_scale=(1333, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=False,
        with_crowd=True,
        with_label=True,
        dense_sample=False,
        uniform_sample=True,
        random_sample=False,
        strided_sample=False,
        num_segments = 8),
    val=dict(
        type=dataset_type,
        #ann_file=data_root + 'annotations/instances_val2017.json',
        #img_prefix=data_root + 'val2017',
        ann_file=data_root + 'frame_list/val_list.txt',
        img_prefix=data_root + 'dataset/val',
        img_scale=(1333, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=True,
        with_label=True,
        dense_sample=False,
        uniform_sample=True,
        random_sample=False,
        strided_sample=False,
        num_segments = 8),
    test=dict(
        type=dataset_type,
        #ann_file=data_root + 'annotations/instances_val2017.json',
        #img_prefix=data_root + 'val2017',
        ann_file=data_root + 'frame_list/test_list.txt',
        img_prefix=data_root + 'dataset/test',
        img_scale=(1333, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True,
        dense_sample=False,
        uniform_sample=True,
        random_sample=False,
        strided_sample=False,
        num_segments = 8))
# optimizer
# if you use 8 GPUs for training, please change lr to 0.02
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2),acc_step=5)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/test_run_8'
#work_dir = './work_dirs/coco'
load_from = None
resume_from = None
workflow = [('train', 1)]
