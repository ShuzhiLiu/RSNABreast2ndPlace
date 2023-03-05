default_scope = 'mmcls'
custom_imports = dict(
    imports=['mmdet.models'], allow_failed_imports=False)
num_classes = 2
split = 0
model = dict(
    type='RSNAAuxCls',
    backbone=dict(
        type='ConvNeXt',
        arch='small',
        out_indices=(3,),
        drop_path_rate=0.4,
        gap_before_final_norm=True,
        with_cp=True,
        init_cfg=[
            dict(
                type='TruncNormal',
                layer=['Conv2d', 'Linear'],
                std=0.02,
                bias=0.0),
            dict(type='Constant', layer=['LayerNorm'], val=1.0, bias=0.0)
        ]),
    head=dict(
        type='LinearClsHead',
        num_classes=num_classes,
        in_channels=768,
        init_cfg=None,
        loss=dict(type='SoftmaxEQLLoss', num_classes=num_classes),
        cal_acc=False),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.0),
        dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
    ],
    _scope_='mmcls')
data_preprocessor = dict(
    num_classes=num_classes,
    mean=[77.52425988, 77.52425988, 77.52425988],
    std=[51.8555656, 51.8555656, 51.8555656],
    to_rgb=True)
bgr_mean = [77.52425988, 77.52425988, 77.52425988]
bgr_std = [51.8555656, 51.8555656, 51.8555656]
size = (1536, 1536)
albu_trans = [
    dict(
        type='PadIfNeeded',
        min_height=size[0],
        min_width=size[1],
        border_mode=0,
        value=0,
        p=1.0),
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0,
        scale_limit=[-0.15, 0.15],
        rotate_limit=45,
        interpolation=2,
        border_mode=0,
        value=0,
        mask_value=0,
        p=0.5)
]
train_pipeline = [
    dict(
        type='LoadImageRSNABreastAux',
        img_prefix='/kaggle/input/rsna-breast-cancer-detection/train_images',
        cropped=False,
        _scope_='mmcls'),
    dict(
        type='Resize',
        scale=size,
        keep_ratio=True,
        interpolation='bicubic',
        _scope_='mmcls'),
    dict(
        type='RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal'],
        _scope_='mmcls'),
    dict(
        type='Albu',
        transforms=albu_trans,
        keymap=dict(img='image'),
        skip_img_without_anno=False,
        _scope_='mmdet'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.1,
        hparams=dict(pad_val=[47, 50, 79], interpolation='bicubic'),
        _scope_='mmcls'),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=0.3333333333333333,
        fill_color=[77.52425988, 77.52425988, 77.52425988],
        fill_std=[51.8555656, 51.8555656, 51.8555656],
        _scope_='mmcls'),
    dict(type='Pad', size=size, pad_to_square=False, _scope_='mmcls'),
    dict(type='PackMxInputs', _scope_='mmcls')
]
test_pipeline = [
    dict(
        type='LoadImageRSNABreastAux',
        img_prefix='/kaggle/input/rsna-breast-cancer-detection/train_images',
        cropped=False),
    dict(
        type='Resize',
        scale=size,
        keep_ratio=True,
        interpolation='bicubic'),
    dict(type='Pad', size=size, pad_to_square=False, _scope_='mmcls'),
    dict(type='PackClsInputs')
]
train_dataloader = dict(
    pin_memory=False,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    batch_size=16,
    num_workers=16,
    dataset=dict(
        type='CsvGeneralDataset',
        ann_file=
        '/kaggle/input/rsna-breast-cancer-detection/train_withbox_split_clean_encode.csv',
        metainfo=dict(classes=(0, 1)),
        split=-1,
        train=True,
        label_key='cancer',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True))
val_dataloader = dict(
    pin_memory=False,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    batch_size=16,
    num_workers=16,
    dataset=dict(
        type='CsvGeneralDataset',
        ann_file=
        '/kaggle/input/rsna-breast-cancer-detection/train_withbox_split_clean_encode.csv',
        metainfo=dict(classes=(0, 1)),
        split=-1,
        train=False,
        label_key='cancer',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False))
val_evaluator = dict(type='RSNAPFBeta')
test_dataloader = dict(
    pin_memory=False,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    batch_size=16,
    num_workers=16,
    dataset=dict(
        type='CsvClsDataset',
        ann_file=
        '/kaggle/input/rsna-breast-cancer-detection/train_withbox_split_clean_encode.csv',
        metainfo=dict(classes=(0, 1)),
        split=split,
        train=False,
        label_key='cancer',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False))
test_evaluator = dict(type='Accuracy', topk=(1,))
optim_wrapper = dict(
    type='AmpOptimWrapper',
    accumulative_counts=12,
    optimizer=dict(
        type='AdamW',
        lr=0.00015,
        weight_decay=0.05,
        eps=1e-08,
        betas=(0.9, 0.999),
        _scope_='mmcls'),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys=dict({
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        })))
epochs = 8
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=True,
        end=1,
        convert_to_iter_based=True,
        _scope_='mmcls'),
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-06,
        by_epoch=True,
        begin=1,
        T_max=epochs,
        convert_to_iter_based=True,
        _scope_='mmcls')
]
train_cfg = dict(by_epoch=True, max_epochs=epochs, val_interval=1)
val_cfg = dict()
test_cfg = dict()
auto_scale_lr = dict(base_batch_size=64)

default_hooks = dict(
    timer=dict(type='IterTimerHook', _scope_='mmcls'),
    logger=dict(type='LoggerHook', interval=100, _scope_='mmcls'),
    param_scheduler=dict(type='ParamSchedulerHook', _scope_='mmcls'),
    checkpoint=dict(type='CheckpointHook', interval=4, save_best='pf1',
                    max_keep_ckpts=3,
                    rule='greater', _scope_='mmcls'),
    sampler_seed=dict(type='DistSamplerSeedHook', _scope_='mmcls'),
    visualization=dict(
        type='VisualizationHook', enable=False, _scope_='mmcls'))
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExponentialMovingAverage',
        momentum=0.02,
        update_buffers=True,
        priority=49, ),
]
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend', _scope_='mmcls')]
visualizer = dict(
    type='ClsVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    _scope_='mmcls')
log_level = 'INFO'
load_from = '/kaggle/input/rsna-breast-cancer-detection/work_folder/base_cls_small_pretrain/epoch_24.pth'
resume = False
work_dir = '/kaggle/input/rsna-breast-cancer-detection/work_folder/base_cls_small_finetune'
fp16 = dict(loss_scale=256.0, velocity_accum_type='half', accum_type='half')
launcher = 'none'
