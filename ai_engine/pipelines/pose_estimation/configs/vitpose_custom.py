_base_ = ['_base_/default_runtime.py']

# Runner (train / val / test loop)
train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=210,
    val_interval=10
)
val_cfg = dict(
    _delete_=True,
    type='ValLoop'
)
test_cfg = dict(
    _delete_=True,
    type='TestLoop'
)

# Optimizer
custom_imports = dict(
    imports=['mmpose.engine.optim_wrappers.layer_decay_optim_wrapper'],
    allow_failed_imports=False
)
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=5e-4,
        betas=(0.9, 0.999),
        weight_decay=0.1
    ),
    paramwise_cfg=dict(
        num_layers=12,
        layer_decay_rate=0.75,
        custom_keys=dict(
            bias=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
        )
    ),
    constructor='LayerDecayOptimWrapperConstructor',
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

# LR schedule
param_scheduler = [
    dict(type='LinearLR', begin=0, end=500,
         start_factor=0.001, by_epoch=False),
    dict(type='MultiStepLR', begin=0, end=210, milestones=[170, 200],
         gamma=0.1, by_epoch=True),
]

fp16 = dict(loss_scale='dynamic')

# Codec
codec = dict(
    type='UDPHeatmap',
    input_size=(192, 256),
    heatmap_size=(96, 128),
    sigma=2
)

# Model
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True
    ),
    backbone=dict(
        type='mmpretrain.VisionTransformer',
        arch='base',
        img_size=(256, 192),
        patch_size=16,
        qkv_bias=True,
        drop_path_rate=0.3,
        with_cls_token=False,
        out_type='featmap',
        patch_cfg=dict(padding=0),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/v1/pretrained_models/mae_pretrain_vit_base_20230913.pth'
        ),
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=768,
        out_channels=17,
        deconv_out_channels=(256, 256, 256),
        deconv_kernel_sizes=(4, 4, 4),
        final_layer=dict(kernel_size=1),
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec,
    ),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap'
    )
)

# Dataset meta
dataset_type = 'CocoDataset'
data_root = 'data/dataset/'
data_mode = 'topdown'

metainfo = dict(
    dataset_name='custom',
    keypoint_info={
        i: dict(name=name, swap=swap)
        for i, (name, swap) in enumerate([
            ('nose', 'nose'), ('left_eye', 'right_eye'), ('right_eye', 'left_eye'),
            ('left_ear', 'right_ear'), ('right_ear', 'left_ear'),
            ('left_shoulder', 'right_shoulder'), ('right_shoulder', 'left_shoulder'),
            ('left_elbow', 'right_elbow'), ('right_elbow', 'left_elbow'),
            ('left_wrist', 'right_wrist'), ('right_wrist', 'left_wrist'),
            ('left_hip', 'right_hip'), ('right_hip', 'left_hip'),
            ('left_knee', 'right_knee'), ('right_knee', 'left_knee'),
            ('left_ankle', 'right_ankle'), ('right_ankle', 'left_ankle')
        ])
    },
    skeleton_info={},
    joint_weights=[1.0] * 17,
    sigmas=[1.0] * 17
)

# Pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='PackPoseInputs'),
]

# Dataloaders
train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/'),
        pipeline=train_pipeline,
        metainfo=metainfo,
    )
)
val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=val_pipeline,
        metainfo=metainfo,
    )
)
test_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=2,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/test.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=val_pipeline,
        metainfo=metainfo,
    )
)

# Evaluators
val_evaluator = dict(type='CocoMetric')
test_evaluator = dict(
    _delete_=True,
    type='CocoMetric'
)

# Checkpoint hook
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_best='coco/AP',
        rule='greater',
        max_keep_ckpts=1,
        filename_tmpl='best_AP.pth'
    )
)
