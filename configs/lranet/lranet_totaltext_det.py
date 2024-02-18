find_unused_parameters=True
num_coefficients = 14
path_lra = './eigenanchors/lra_totaltext_14.npz'

model = dict(
    type='LRANet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        dcn=dict(type='DCNv2', deform_groups=2, fallback_on_stride=False),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        stage_with_dcn=(False, True, True, True)),
    neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        add_extra_convs='on_output',
        num_outs=3,
        relu_before_extra_convs=True,
        act_cfg=None),
    bbox_head=dict(
        type='LRAHead',
        in_channels=256,
        num_coefficients=num_coefficients,
        scales=(8, 16, 32),
        loss=dict(type='LRALoss', num_coefficients=num_coefficients,
                  path_lra = path_lra,),
        nms_thr=0.1,
        path_lra=path_lra,
        num_convs=4),
)

train_cfg = None
test_cfg = None

dataset_type = 'IcdarE2EDataset'
data_root = 'data/my_synthtext/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile',
         ),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='ColorJitter',
        brightness=32.0 / 255,
        saturation=0.5,
        contrast=0.5
    ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomScaling', size=960, scale=(3. / 8, 5. / 2)),
    dict(
        type='RandomCropFlip', crop_ratio=0.2, iter_num=1, min_area_ratio=0.01),
    dict(
        type='RandomCropPolyInstances',
        instance_key='gt_masks',
        crop_ratio=0.8,
        min_side_ratio=0.1),
    dict(
        type='RandomRotatePolyInstances',
        rotate_ratio=0.5,
        max_angle=90,
        pad_with_fixed_color=False),
    dict(type='SquareResizePad', target_size=960, pad_ratio=0.6),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='Pad', size_divisor=32),
    dict(
        type='LRATargets',
        level_proportion_range=((0, 0.25), (0.2, 0.65), (0.55, 1.0)),
        with_area=True,
        path_lra=path_lra,
        num_coefficients=num_coefficients,
        num_samples=3,
    ),
    dict(
        type='CustomFormatBundle',
        keys=['polygons_area', 'gt_texts','lra_polys'],
        visualize=dict(flag=False, boundary_key=None)),
    dict(type='Collect', keys=['img', 'p3_maps', 'p4_maps', 'p5_maps', 'polygons_area',  'gt_texts', 'lra_polys'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1800, 1000),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(1280, 800), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            ann_file=[
                'data/totaltext/totaltext_train.json'],
            img_prefix=[
                'data/totaltext/imgs/training'],
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file='data/totaltext/totaltext_test.json',
        img_prefix='data/totaltext/imgs',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='data/totaltext/totaltext_test.json',
        img_prefix='data/totaltext/imgs',
        pipeline=test_pipeline,))
evaluation = dict(interval=1, metric='hmean-e2e')

# optimizer
optimizer = dict(type='SGD', lr=1e-3, momentum=0.90, weight_decay=5e-4)

optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-7, by_epoch=True,
                 warmup='linear',warmup_iters=500,warmup_ratio=0.001,
                 )
total_epochs = 100

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook')

    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
