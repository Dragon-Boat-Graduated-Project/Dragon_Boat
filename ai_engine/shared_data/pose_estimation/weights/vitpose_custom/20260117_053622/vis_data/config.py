backend_args = dict(backend='local')
codec = dict(
    heatmap_size=(
        96,
        128,
    ),
    input_size=(
        192,
        256,
    ),
    sigma=2,
    type='UDPHeatmap')
custom_hooks = [
    dict(type='SyncBuffersHook'),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'mmpose.engine.optim_wrappers.layer_decay_optim_wrapper',
    ])
data_mode = 'topdown'
data_root = 'data/dataset/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    badcase=dict(
        badcase_thr=5,
        enable=False,
        metric_type='loss',
        out_dir='badcase',
        type='BadCaseAnalysisHook'),
    checkpoint=dict(
        filename_tmpl='best_AP.pth',
        interval=10,
        max_keep_ckpts=1,
        rule='greater',
        save_best='coco/AP',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='PoseVisualizationHook'))
default_scope = 'mmpose'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
fp16 = dict(loss_scale='dynamic')
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(
    by_epoch=True, num_digits=6, type='LogProcessor', window_size=50)
metainfo = dict(
    dataset_name='custom',
    joint_weights=[
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ],
    keypoint_info=dict({
        0: dict(name='nose', swap='nose'),
        1: dict(name='left_eye', swap='right_eye'),
        10: dict(name='right_wrist', swap='left_wrist'),
        11: dict(name='left_hip', swap='right_hip'),
        12: dict(name='right_hip', swap='left_hip'),
        13: dict(name='left_knee', swap='right_knee'),
        14: dict(name='right_knee', swap='left_knee'),
        15: dict(name='left_ankle', swap='right_ankle'),
        16: dict(name='right_ankle', swap='left_ankle'),
        2: dict(name='right_eye', swap='left_eye'),
        3: dict(name='left_ear', swap='right_ear'),
        4: dict(name='right_ear', swap='left_ear'),
        5: dict(name='left_shoulder', swap='right_shoulder'),
        6: dict(name='right_shoulder', swap='left_shoulder'),
        7: dict(name='left_elbow', swap='right_elbow'),
        8: dict(name='right_elbow', swap='left_elbow'),
        9: dict(name='left_wrist', swap='right_wrist')
    }),
    sigmas=[
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ],
    skeleton_info=dict())
model = dict(
    backbone=dict(
        arch='base',
        drop_path_rate=0.3,
        img_size=(
            256,
            192,
        ),
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmpose/v1/pretrained_models/mae_pretrain_vit_base_20230913.pth',
            type='Pretrained'),
        out_type='featmap',
        patch_cfg=dict(padding=0),
        patch_size=16,
        qkv_bias=True,
        type='mmpretrain.VisionTransformer',
        with_cls_token=False),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='PoseDataPreprocessor'),
    head=dict(
        decoder=dict(
            heatmap_size=(
                96,
                128,
            ),
            input_size=(
                192,
                256,
            ),
            sigma=2,
            type='UDPHeatmap'),
        deconv_kernel_sizes=(
            4,
            4,
            4,
        ),
        deconv_out_channels=(
            256,
            256,
            256,
        ),
        final_layer=dict(kernel_size=1),
        in_channels=768,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        out_channels=17,
        type='HeatmapHead'),
    test_cfg=dict(flip_mode='heatmap', flip_test=True),
    type='TopdownPoseEstimator')
optim_wrapper = dict(
    clip_grad=dict(max_norm=1.0, norm_type=2),
    constructor='LayerDecayOptimWrapperConstructor',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0005, type='AdamW', weight_decay=0.1),
    paramwise_cfg=dict(
        custom_keys=dict(
            bias=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0)),
        layer_decay_rate=0.75,
        num_layers=12))
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=210,
        gamma=0.1,
        milestones=[
            170,
            200,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/test.json',
        data_mode='topdown',
        data_prefix=dict(img='images/'),
        data_root='data/dataset/',
        metainfo=dict(
            dataset_name='custom',
            joint_weights=[
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            keypoint_info=dict({
                0:
                dict(name='nose', swap='nose'),
                1:
                dict(name='left_eye', swap='right_eye'),
                10:
                dict(name='right_wrist', swap='left_wrist'),
                11:
                dict(name='left_hip', swap='right_hip'),
                12:
                dict(name='right_hip', swap='left_hip'),
                13:
                dict(name='left_knee', swap='right_knee'),
                14:
                dict(name='right_knee', swap='left_knee'),
                15:
                dict(name='left_ankle', swap='right_ankle'),
                16:
                dict(name='right_ankle', swap='left_ankle'),
                2:
                dict(name='right_eye', swap='left_eye'),
                3:
                dict(name='left_ear', swap='right_ear'),
                4:
                dict(name='right_ear', swap='left_ear'),
                5:
                dict(name='left_shoulder', swap='right_shoulder'),
                6:
                dict(name='right_shoulder', swap='left_shoulder'),
                7:
                dict(name='left_elbow', swap='right_elbow'),
                8:
                dict(name='right_elbow', swap='left_elbow'),
                9:
                dict(name='left_wrist', swap='right_wrist')
            }),
            sigmas=[
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            skeleton_info=dict()),
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(input_size=(
                192,
                256,
            ), type='TopdownAffine', use_udp=True),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    num_workers=2,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(type='CocoMetric')
train_cfg = dict(max_epochs=210, type='EpochBasedTrainLoop', val_interval=10)
train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file='annotations/train.json',
        data_mode='topdown',
        data_prefix=dict(img='images/'),
        data_root='data/dataset/',
        metainfo=dict(
            dataset_name='custom',
            joint_weights=[
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            keypoint_info=dict({
                0:
                dict(name='nose', swap='nose'),
                1:
                dict(name='left_eye', swap='right_eye'),
                10:
                dict(name='right_wrist', swap='left_wrist'),
                11:
                dict(name='left_hip', swap='right_hip'),
                12:
                dict(name='right_hip', swap='left_hip'),
                13:
                dict(name='left_knee', swap='right_knee'),
                14:
                dict(name='right_knee', swap='left_knee'),
                15:
                dict(name='left_ankle', swap='right_ankle'),
                16:
                dict(name='right_ankle', swap='left_ankle'),
                2:
                dict(name='right_eye', swap='left_eye'),
                3:
                dict(name='left_ear', swap='right_ear'),
                4:
                dict(name='right_ear', swap='left_ear'),
                5:
                dict(name='left_shoulder', swap='right_shoulder'),
                6:
                dict(name='right_shoulder', swap='left_shoulder'),
                7:
                dict(name='left_elbow', swap='right_elbow'),
                8:
                dict(name='right_elbow', swap='left_elbow'),
                9:
                dict(name='left_wrist', swap='right_wrist')
            }),
            sigmas=[
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            skeleton_info=dict()),
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(type='RandomFlip'),
            dict(type='RandomHalfBody'),
            dict(type='RandomBBoxTransform'),
            dict(input_size=(
                192,
                256,
            ), type='TopdownAffine', use_udp=True),
            dict(
                encoder=dict(
                    heatmap_size=(
                        96,
                        128,
                    ),
                    input_size=(
                        192,
                        256,
                    ),
                    sigma=2,
                    type='UDPHeatmap'),
                type='GenerateTarget'),
            dict(type='PackPoseInputs'),
        ],
        type='CocoDataset'),
    num_workers=4,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(input_size=(
        192,
        256,
    ), type='TopdownAffine', use_udp=True),
    dict(
        encoder=dict(
            heatmap_size=(
                96,
                128,
            ),
            input_size=(
                192,
                256,
            ),
            sigma=2,
            type='UDPHeatmap'),
        type='GenerateTarget'),
    dict(type='PackPoseInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='annotations/val.json',
        data_mode='topdown',
        data_prefix=dict(img='images/'),
        data_root='data/dataset/',
        metainfo=dict(
            dataset_name='custom',
            joint_weights=[
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            keypoint_info=dict({
                0:
                dict(name='nose', swap='nose'),
                1:
                dict(name='left_eye', swap='right_eye'),
                10:
                dict(name='right_wrist', swap='left_wrist'),
                11:
                dict(name='left_hip', swap='right_hip'),
                12:
                dict(name='right_hip', swap='left_hip'),
                13:
                dict(name='left_knee', swap='right_knee'),
                14:
                dict(name='right_knee', swap='left_knee'),
                15:
                dict(name='left_ankle', swap='right_ankle'),
                16:
                dict(name='right_ankle', swap='left_ankle'),
                2:
                dict(name='right_eye', swap='left_eye'),
                3:
                dict(name='left_ear', swap='right_ear'),
                4:
                dict(name='right_ear', swap='left_ear'),
                5:
                dict(name='left_shoulder', swap='right_shoulder'),
                6:
                dict(name='right_shoulder', swap='left_shoulder'),
                7:
                dict(name='left_elbow', swap='right_elbow'),
                8:
                dict(name='right_elbow', swap='left_elbow'),
                9:
                dict(name='left_wrist', swap='right_wrist')
            }),
            sigmas=[
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            skeleton_info=dict()),
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(input_size=(
                192,
                256,
            ), type='TopdownAffine', use_udp=True),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    num_workers=4,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(type='CocoMetric')
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(input_size=(
        192,
        256,
    ), type='TopdownAffine', use_udp=True),
    dict(type='PackPoseInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='PoseLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'work_dirs/vitpose_custom'
