_base_ = [
    '../mmclassification/configs/_base_/models/resnet50.py',
    # '../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../mmclassification/configs/_base_/schedules/imagenet_bs2048_AdamW.py',
    '../mmclassification/configs/_base_/default_runtime.py'
]

# ---- Model configs ----
# Here we use init_cfg to load pre-trained model.
# In this way, only the weights of backbone will be loaded.
# And modify the num_classes to match our dataset.

model = dict(
    backbone=dict(
        init_cfg = dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb256-rsb-a1-600e_in1k_20211228-20e21305.pth',
            prefix='backbone')
    ),
    head=dict(
        num_classes=2,
        topk = (1, )
    )
)

# ---- Dataset configs ----
# We re-organized the dataset as ImageNet format.
dataset_type = 'CustomDataset'


data_preprocessor = dict(
    num_classes=2,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     # dict(type='RandomResizedCrop', size=224, backend='pillow'),
#     dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='ImageToTensor', keys=['img']),
#     dict(type='ToTensor', keys=['gt_label']),
#     dict(type='Collect', keys=['img', 'gt_label'])
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='Resize', size=(256, -1), backend='pillow'),
#     dict(type='CenterCrop', crop_size=224),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='ImageToTensor', keys=['img']),
#     dict(type='Collect', keys=['img'])
# ]



train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short', backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

    
# Specify the training dataset type and path
train_dataloader=dict(
    batch_size=4,
    num_workers=2,
    dataset =dict(
        type=dataset_type,
        data_root='../split_data/',
        data_prefix='train',
        # classes='../data/cats_dogs_dataset/classes.txt'),
        pipeline=train_pipeline
        ),
    )
    # Specify the validation dataset type and path
test_dataloader=dict(
    dataset = dict(
        type=dataset_type,
        data_root='../split_data/',
        data_prefix='train',
# ann_file='test.txt',
# classes='../data/cats_dogs_dataset/classes.txt'),
        pipeline=test_pipeline
        ),
    )

# val_dataloader = None
val_cfg = None
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval = 1)

# Specify evaluation metric
evaluation = dict(metric='SingleLabelMetric', items=['f1-score'])
# ---- Schedule configs ----
# Usually in fine-tuning, we need a smaller learning rate and less training epochs.
# Specify the learning rate
    
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=0.015, weight_decay=0.3),
    paramwise_cfg = dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
    ),
    clip_grad=dict(max_norm=1.0),
)
# Set the learning rate scheduler
lr_config = dict(policy='step', step=1, gamma=0.1)
runner = dict(type='EpochBasedRunner', max_epochs=2)

# ---- Runtime configs ----
# Output training log every 10 iterations.
log_config = dict(interval=10)

train_dataloader = train_dataloader
# test_dataloader = test_dataloader
# val_dataloader = test_dataloader

test_evaluator = evaluation
