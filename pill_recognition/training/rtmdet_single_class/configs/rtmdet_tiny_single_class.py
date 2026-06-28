_base_ = "mmdet::rtmdet/rtmdet_tiny_8xb32-300e_coco.py"

data_root = "datasets/processed/rtmdet-single-class/"
work_dir = "training/runs/rtmdet-single-class"

metainfo = {
    "classes": ("pill",),
    "palette": [(34, 197, 94)],
}
image_size = (1024, 1024)
max_epochs = 12
stage2_epoch = 10

model = dict(
    bbox_head=dict(num_classes=1),
    test_cfg=dict(
        score_thr=0.05,
        nms_pre=3000,
        min_bbox_size=0,
        nms=dict(type="nms", iou_threshold=0.6),
        max_per_img=20,
    ),
)

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="CachedMosaic",
        img_scale=image_size,
        pad_val=114.0,
        max_cached_images=20,
        random_pop=False,
    ),
    dict(
        type="RandomResize",
        scale=(2048, 2048),
        ratio_range=(0.5, 2.0),
        keep_ratio=True,
    ),
    dict(type="RandomCrop", crop_size=image_size),
    dict(type="YOLOXHSVRandomAug"),
    dict(type="RandomFlip", prob=0.5),
    dict(type="Pad", size=image_size, pad_val=dict(img=(114, 114, 114))),
    dict(
        type="CachedMixUp",
        img_scale=image_size,
        ratio_range=(1.0, 1.0),
        max_cached_images=10,
        random_pop=False,
        pad_val=(114, 114, 114),
        prob=0.5,
    ),
    dict(type="PackDetInputs"),
]

train_pipeline_stage2 = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="RandomResize",
        scale=image_size,
        ratio_range=(0.5, 2.0),
        keep_ratio=True,
    ),
    dict(type="RandomCrop", crop_size=image_size),
    dict(type="YOLOXHSVRandomAug"),
    dict(type="RandomFlip", prob=0.5),
    dict(type="Pad", size=image_size, pad_val=dict(img=(114, 114, 114))),
    dict(type="PackDetInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="Resize", scale=image_size, keep_ratio=True),
    dict(type="Pad", size=image_size, pad_val=dict(img=(114, 114, 114))),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

train_dataloader = dict(
    _delete_=True,
    batch_size=8,
    num_workers=0,
    persistent_workers=False,
    pin_memory=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file="train_coco.json",
        data_prefix=dict(img="images/train/"),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    _delete_=True,
    batch_size=4,
    num_workers=0,
    persistent_workers=False,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file="val_coco.json",
        data_prefix=dict(img="images/val/"),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "val_coco.json",
    metric="bbox",
    format_only=False,
    proposal_nums=(20, 100, 300),
)
test_evaluator = val_evaluator

optim_wrapper = dict(
    _delete_=True,
    type="AmpOptimWrapper",
    accumulative_counts=1,
    loss_scale="dynamic",
    optimizer=dict(type="AdamW", lr=0.00025, weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
)
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-5, by_epoch=False, begin=0, end=500),
    dict(
        type="CosineAnnealingLR",
        eta_min=1e-6,
        begin=0,
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]
train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=max_epochs,
    val_interval=1,
)
custom_hooks = [
    dict(
        type="EMAHook",
        ema_type="ExpMomentumEMA",
        momentum=0.0001,
        update_buffers=True,
        priority=49,
    ),
    dict(
        type="PipelineSwitchHook",
        switch_epoch=stage2_epoch,
        switch_pipeline=train_pipeline_stage2,
    ),
]
default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        max_keep_ckpts=3,
        save_best="coco/bbox_mAP",
        rule="greater",
    ),
    logger=dict(type="LoggerHook", interval=50),
)

auto_scale_lr = dict(enable=False, base_batch_size=8)
randomness = dict(seed=42, deterministic=False)
env_cfg = dict(cudnn_benchmark=True)
